"""Catastrophe track: early-warning signals before a fold, with a null control.

The core measurement is over two behavioural epochs supplied by the caller: a
pre-fold ramp (ending at `fold_index`) and a matched healthy null. Variance and
lag-1 autocorrelation should rise toward the fold and stay flat in the null --
present at the fold and absent in the null, or it is not a specific warning.

For a live factory, pass `ramp` (a real pre-incident epoch), `null` (a prior
healthy epoch), and `fold_index`. With no epochs supplied, the reference
substrate's ramp-through-a-fold and no-fold regimes are used.
"""
from __future__ import annotations

import numpy as np

from ews import detrend, rolling_variance, rolling_ar1, kendall_trend
from ..report import Finding


def _trends(x, end, w):
    seg = np.asarray(x, float)[:end]
    if len(seg) < 2 * w:
        return float("nan"), float("nan")
    res = detrend(seg, w)
    return kendall_trend(rolling_variance(res, w)), kendall_trend(rolling_ar1(res, w))


def measure(substrate, ramp=None, null=None, fold_index=None, w: int = 400,
            pos_threshold: float = 0.3, null_threshold: float = 0.2, seed: int = 3,
            n: int = 12000, **kw) -> Finding:
    if ramp is None:
        fac = substrate.steppable(regime="fold_ramp", seed=seed, n_sample=1)
        fac.run(n)
        ramp = fac.trace.behaviour_series("mean_pos")
    ramp = np.asarray(ramp, float)
    if null is None:
        nul = substrate.steppable(regime="no_fold", seed=seed, n_sample=1)
        nul.run(n)
        null = nul.trace.behaviour_series("mean_pos")
    null = np.asarray(null, float)
    if fold_index is None:
        fold_index = int(np.argmin(np.diff(ramp)))

    if fold_index < 2 * w:
        return Finding(
            track="catastrophe", property="early-warning signals before a fold",
            capability="steppable",
            measured=dict(fold_at=int(fold_index), window=w,
                          note="fold window too short (< 2*w) to estimate trends"),
            confirms=False,
            summary=f"fold window too short ({fold_index} < {2 * w}); widen the epoch or shrink w",
            confirm_criterion=f"pre-fold variance trend > {pos_threshold} and null trend < {null_threshold}",
            falsify_criterion="no pre-fold rise, or the null run shows the same rise (not specific)",
        )

    vt, at = _trends(ramp, fold_index, w)
    vn, an = _trends(null, len(null), w)
    confirms = bool(np.isfinite(vt) and np.isfinite(vn) and vt > pos_threshold and vn < null_threshold)
    return Finding(
        track="catastrophe", property="early-warning signals before a fold",
        capability="steppable",
        measured=dict(fold_at=int(fold_index), variance_trend=round(vt, 3), ar1_trend=round(at, 3),
                      null_variance_trend=round(vn, 3), null_ar1_trend=round(an, 3)),
        confirms=confirms,
        summary=f"variance Kendall tau={vt:+.2f} pre-fold vs {vn:+.2f} null "
                f"({'specific warning' if confirms else 'not specific'})",
        confirm_criterion=f"pre-fold variance trend > {pos_threshold} and null trend < {null_threshold}",
        falsify_criterion="no pre-fold rise, or the null run shows the same rise (not specific)",
    )
