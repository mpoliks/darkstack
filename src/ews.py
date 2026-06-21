"""
Early-warning signals (EWS) for catastrophic transitions.

An evaluator can sense an
incoming catastrophe (Thom fold/cusp) through "critical slowing down," measured
via rising variance, rising autocorrelation, and rising ensemble disagreement.

Mechanism (saddle-node / fold): as a control parameter approaches the fold, the
dominant eigenvalue of the linearised return map -> 1, so the system's recovery
rate from perturbations -> 0. A slower return rate inflates both the lag-1
autocorrelation (toward 1) and the stationary variance (toward infinity). These
are model-free leading indicators -- but with well-documented false positives.

References
----------
Scheffer et al. 2009, "Early-warning signals for critical transitions" (Nature);
Dakos et al. 2012, "Methods for detecting early warnings" (PLoS ONE);
Wissel 1984 (critical slowing down).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for t in range(window, len(x) + 1):
        out[t - 1] = np.var(x[t - window:t])
    return out


def rolling_ar1(x: np.ndarray, window: int) -> np.ndarray:
    """Lag-1 autocorrelation in a sliding window (the AR(1) coefficient)."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for t in range(window, len(x) + 1):
        w = x[t - window:t]
        w = w - w.mean()
        denom = np.dot(w, w)
        out[t - 1] = np.dot(w[:-1], w[1:]) / denom if denom > 0 else np.nan
    return out


def ensemble_disagreement(ensemble: np.ndarray) -> np.ndarray:
    """Cross-trajectory standard deviation at each time. `ensemble` is
    (n_members, T). The third EWS: heterogeneous judges
    disagreeing more as the system nears a cusp."""
    return np.nanstd(ensemble, axis=0)


def detrend(x: np.ndarray, window: int) -> np.ndarray:
    """Smooth detrend so EWS measure fluctuations, not the slow drift."""
    from scipy.ndimage import uniform_filter1d
    x = np.asarray(x, float)
    trend = uniform_filter1d(x, size=window, mode="nearest")
    return x - trend


def kendall_trend(series: np.ndarray) -> float:
    """Kendall's tau of an EWS series vs time: the standard EWS summary statistic
    (a value near +1 = strong rising trend = strong warning)."""
    s = np.asarray(series, float)
    m = ~np.isnan(s)
    if m.sum() < 5:
        return np.nan
    tau, _ = kendalltau(np.arange(m.sum()), s[m])
    return float(tau)


if __name__ == "__main__":
    # A slowly-forced 1-D system pushed through a fold (saddle-node) bifurcation:
    #   dx/dt = -dV/dx with V = x^4/4 - a(t) x ,  a(t) ramping.
    # We verify variance and AR(1) rise (critical slowing down) BEFORE the jump,
    # and that the Kendall trend flags it.
    # Bistable double well V = x^4/4 - b x^2/2 - a x ; force = -x^3 + b x + a.
    # b=1 => two wells while |a| < 2/(3 sqrt 3) ~ 0.385. Start in the LEFT well
    # and ramp a upward: the left well shallows and annihilates at the fold
    # a = +0.385, where the state jumps to the right well. Critical slowing down
    # builds as a -> 0.385 from below.
    rng = np.random.default_rng(0)
    T = 8000
    b = 1.0
    a = np.linspace(-0.6, 0.6, T)            # ramps through the fold near 82% of T
    x = np.empty(T)
    x[0] = -1.3                               # left well
    dt, sigma = 0.05, 0.08
    for t in range(1, T):
        drift = -(x[t - 1] ** 3) + b * x[t - 1] + a[t]
        x[t] = x[t - 1] + dt * drift + sigma * np.sqrt(dt) * rng.standard_normal()

    jump = int(np.argmax(np.diff(x)))         # locate the upward catastrophic jump
    pre = slice(0, jump)
    w = 300
    res = detrend(x[pre], w)
    var = rolling_variance(res, w)
    ar1 = rolling_ar1(res, w)
    print(f"fold jump at t={jump} (of {T})")
    print(f"variance Kendall tau (pre-jump): {kendall_trend(var):+.3f}")
    print(f"AR(1)   Kendall tau  (pre-jump): {kendall_trend(ar1):+.3f}")
    print("Both should be strongly positive (critical slowing down).")
