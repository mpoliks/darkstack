"""Versioning track: behavioural versions and their robustness.

Estimates the transfer operator from a single behavioural coordinate, reads its
spectrum, and reports the timescale separation tau2/tau3 -- the threshold-free
measure of whether one mode is an order of magnitude slower than the rest (a
durable version) or the spectrum is a smooth relaxation continuum (no versions).
"""
from __future__ import annotations

import numpy as np

from transfer_operator import (ulam_operator, reversibilize, spectrum,
                               coherence_timescale, n_metastable)
from ..report import Finding


def measure(substrate, regime: str = "versions", n: int = 60000, seed: int = 2,
            n_boxes: int = 30, sep_threshold: float = 5.0, coord: str = "mean_pos",
            **kw) -> Finding:
    fac = substrate.steppable(regime=regime, seed=seed, n_sample=1)
    fac.run(n)
    x = fac.trace.behaviour_series(coord)
    P, edges, occ = ulam_operator(x, n_boxes=n_boxes)
    P_rev, pi = reversibilize(P)
    vals, _ = spectrum(P_rev)
    taus = np.array([coherence_timescale(v) for v in vals])
    tsep = float(taus[1] / taus[2])
    # estimator stability: recompute at a coarser/finer box count
    seps = []
    for nb in (max(12, n_boxes - 10), n_boxes, n_boxes + 10):
        Pq, _, _ = ulam_operator(x, n_boxes=nb)
        Pqr, _ = reversibilize(Pq)
        vq, _ = spectrum(Pqr)
        tq = np.array([coherence_timescale(v) for v in vq])
        seps.append(float(tq[1] / tq[2]))
    stable = (np.std(seps) / (np.mean(seps) + 1e-9)) < 0.5
    confirms = bool(tsep >= sep_threshold and stable)
    return Finding(
        track="versioning", property="behavioural versions / robustness",
        capability="steppable",
        measured=dict(tau2=float(taus[1]), tau3=float(taus[2]), timescale_sep=tsep,
                      n_metastable_0p95=int(n_metastable(P_rev, 0.95)),
                      sep_across_boxes=[round(s, 1) for s in seps]),
        confirms=confirms,
        summary=f"tau2/tau3={tsep:.1f} ({'one slow mode -> versions' if confirms else 'no clear version gap'})",
        confirm_criterion=f"timescale separation tau2/tau3 >= {sep_threshold} and stable across box counts",
        falsify_criterion="tau2/tau3 ~ 1 (smooth relaxation, no versions) or separation swings with box count",
    )
