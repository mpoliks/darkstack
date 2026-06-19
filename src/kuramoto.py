"""
Kuramoto entrainment for the "anything factory" / phase-locked condensate.

The paper's closing sections argue that factory loops sharing a common medium
(e.g. a common foundation-model checkpoint acting as a global forcing function)
drift into synchrony like Huygens' coupled pendulum clocks; that this is the
mechanism of the 2010 flash crash ("a spiral of phase-locked, mutually-entrained
machine-time loops"); and that diversifying dependencies / jitter raises the
synchronisation threshold and prevents correlated collapse.

We model each factory's slowest control loop as a phase oscillator and couple
them. The Kuramoto order parameter r measures systemic synchronisation;
heterogeneity of natural frequencies (dependency diversity) sets the critical
coupling Kc below which the population stays desynchronised.

Theory: for natural frequencies drawn from a unimodal symmetric g(omega), the
incoherent state loses stability at Kc = 2 / (pi g(0)). For a Lorentzian of
half-width gamma, Kc = 2 gamma; for a Gaussian of s.d. sigma,
Kc = sigma * sqrt(8/pi) ~ 1.596 sigma. Above Kc, r ~ sqrt((K-Kc)/Kc).

References
----------
Kuramoto 1975/1984; Strogatz 2000, "From Kuramoto to Crawford";
Acebron et al. 2005 (Rev. Mod. Phys.).
"""
from __future__ import annotations

import numpy as np


def simulate(N: int, K: float, omega: np.ndarray, T: float = 80.0, dt: float = 0.01,
             noise: float = 0.0, rng: np.random.Generator | None = None,
             burn_frac: float = 0.5):
    """Integrate the Kuramoto model; return the time-averaged order parameter r
    over the post-burn-in window, plus the final phases and the r time series."""
    rng = rng or np.random.default_rng(0)
    steps = int(T / dt)
    theta = rng.uniform(0, 2 * np.pi, N)
    r_series = np.empty(steps)
    burn = int(burn_frac * steps)
    for s in range(steps):
        z = np.mean(np.exp(1j * theta))
        r, psi = np.abs(z), np.angle(z)
        r_series[s] = r
        # mean-field form: dtheta_i = omega_i + K r sin(psi - theta_i)
        dtheta = omega + K * r * np.sin(psi - theta)
        theta = theta + dt * dtheta
        if noise > 0:
            theta += noise * np.sqrt(dt) * rng.standard_normal(N)
    return float(r_series[burn:].mean()), theta, r_series


def critical_coupling_gaussian(sigma: float) -> float:
    return sigma * np.sqrt(8.0 / np.pi)


def critical_coupling_lorentzian(gamma: float) -> float:
    return 2.0 * gamma


def sweep_coupling(N: int, sigma: float, K_values, dist: str = "lorentzian",
                   rng: np.random.Generator | None = None, **kw):
    """Sweep coupling K and return steady-state r(K). `sigma` is the spread of
    natural frequencies (dependency diversity / jitter)."""
    rng = rng or np.random.default_rng(0)
    if dist == "lorentzian":
        omega = sigma * np.tan(np.pi * (rng.random(N) - 0.5))   # Cauchy, half-width sigma
        omega = np.clip(omega, -50, 50)
        omega = omega - np.median(omega)            # Cauchy location = median (mean undefined)
    else:
        omega = sigma * rng.standard_normal(N)
        omega = omega - omega.mean()
    rs = []
    for K in K_values:
        r, _, _ = simulate(N, K, omega, rng=np.random.default_rng(rng.integers(1 << 30)), **kw)
        rs.append(r)
    return np.array(rs), omega


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 600
    gamma = 0.5
    Kc = critical_coupling_lorentzian(gamma)
    Ks = np.linspace(0.0, 3.0, 16)
    rs, omega = sweep_coupling(N, gamma, Ks, dist="lorentzian", rng=rng, T=60, dt=0.02)
    print(f"Lorentzian half-width gamma={gamma}  =>  predicted Kc = 2*gamma = {Kc:.2f}")
    for K, r in zip(Ks, rs):
        flag = "  <-- onset" if (K <= Kc + 0.2 and K >= Kc - 0.2) else ""
        print(f"  K={K:4.2f}   r={r:5.3f}{flag}")
    # check: r should be ~0 below Kc and rise after
    below = rs[Ks < Kc - 0.3].mean()
    above = rs[Ks > Kc + 0.5].mean()
    print(f"mean r below Kc: {below:.3f}   mean r above Kc: {above:.3f}  (expect near 0 vs >0.5)")
