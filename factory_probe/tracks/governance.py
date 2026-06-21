"""Governance track: cascade-ratio stability of a governance loop.

Identifies the inner-loop settling time (apply a price step, time the response of
the price-responsive observable), then sweeps the governance repricing period and
measures price instability. Instability should fall as the period crosses several
times the inner settling time (the cascade band): govern slower than the loops you
steer.
"""
from __future__ import annotations

import numpy as np

from control import PID, instability_index
from ..report import Finding

_OBS = "norm_sat"   # the price-responsive observable in the governance regime


def _t_inner(substrate, seeds=(0, 1, 2, 3), settle=200, probe=500, step_lam=12.0) -> float:
    ests = []
    for s in seeds:
        fac = substrate.steppable(regime="governance", seed=s, n_sample=1)
        for _ in range(settle):
            fac.set_price(0.0); fac.step()
        tr = np.empty(probe)
        for t in range(probe):
            fac.set_price(step_lam); tr[t] = fac.step().behaviour[_OBS]
        y0, yf = tr[:5].mean(), tr[-100:].mean()
        if abs(yf - y0) < 0.02:
            continue
        st = int(np.argmax(np.abs(tr - yf) < 0.10 * abs(yf - y0)))
        if st > 0:
            ests.append(st)
    return float(np.median(ests)) if ests else 15.0


def _instability(substrate, T_gov, seed, n=2500, target=0.80) -> float:
    fac = substrate.steppable(regime="governance", seed=seed, n_sample=1)
    pid = PID(Kp=6.0, Ki=0.4, Kd=2.0, hi=30)
    lam = 0.0
    nh = np.empty(n); lamh = np.empty(n)
    for t in range(n):
        fac.set_price(lam)
        nh[t] = fac.step().behaviour[_OBS]
        lamh[t] = lam
        if (t + 1) % T_gov == 0:
            lam = pid.update(target - nh[max(0, t - T_gov + 1):t + 1].mean(), dt=T_gov)
    return instability_index(lamh)


def measure(substrate, seeds=(0, 1, 2, 3), T_gov=(3, 6, 12, 24, 48, 120, 300),
            band=3.0, n=2500, **kw) -> Finding:
    Ti = _t_inner(substrate, seeds=seeds)
    means = {t: float(np.mean([_instability(substrate, t, s, n=n) for s in seeds])) for t in T_gov}
    ratios = {t: t / Ti for t in T_gov}
    below = [means[t] for t in T_gov if ratios[t] < band]
    above = [means[t] for t in T_gov if ratios[t] >= band]
    lo = float(np.mean(below)) if below else float("nan")
    hi = float(np.mean(above)) if above else float("nan")
    confirms = bool(above and below and hi < lo)
    return Finding(
        track="governance", property="cascade-ratio stability",
        capability="steppable",
        measured=dict(T_inner=round(Ti, 1),
                      instability_by_ratio={round(ratios[t], 2): round(means[t], 2) for t in T_gov},
                      mean_below_band=round(lo, 2), mean_above_band=round(hi, 2)),
        confirms=confirms,
        summary=f"price instability {lo:.2f} (fast governance) -> {hi:.2f} (slow); "
                f"{'declines through cascade band' if confirms else 'no decline'}",
        confirm_criterion=f"instability above the {band}:1 cascade ratio < instability below it",
        falsify_criterion="instability flat or rising with the cascade ratio",
    )
