"""Pathology track: name the convergence pathology from behaviour alone.

Reads a behavioural fingerprint -- volatility, variety, norm-unmet, metric/norm
gap -- and classifies it as healthy, stable failure, overfitting, learning death,
or thrash. On the reference substrate it classifies every planted regime; on a
live factory it names whichever pathology the behaviour matches.
"""
from __future__ import annotations

import numpy as np

from ..report import Finding


def fingerprint(trace, tail: int = 4000) -> dict:
    mp = trace.behaviour_series("mean_pos")
    var = trace.behaviour_series("variety")
    nsat = trace.behaviour_series("norm_sat")
    msat = trace.behaviour_series("metric_sat")
    return dict(
        volatility=float(np.std(mp[-tail:])),
        variety=float(np.mean(var[-tail:])),
        norm_unmet=float(1.0 - np.mean(nsat[-tail:])),
        proxy_gap=float(max(0.0, np.mean(msat[-tail:]) - np.mean(nsat[-tail:]))),
    )


def classify(fp: dict) -> str:
    # the metric/norm gap is checked first: an overfitting factory also has low
    # variety (it has locked onto the metric peak), so the proxy gap, not the
    # variety collapse, is its defining signature.
    if fp["proxy_gap"] > 0.3:
        return "overfitting"
    if fp["variety"] < 1.5:
        return "learning_death"
    if fp["volatility"] > 0.06:
        return "thrash"
    if fp["norm_unmet"] > 0.5:
        return "stable_failure"
    return "healthy"


def measure(substrate, regimes=("healthy", "stable_failure", "overfitting",
                                "learning_death", "thrash"),
            n: int = 20000, seed: int = 1, **kw) -> Finding:
    results = {}
    correct = 0
    for regime in regimes:
        fac = substrate.steppable(regime=regime, seed=seed, n_sample=1)
        fac.run(n)
        label = classify(fingerprint(fac.trace))
        results[regime] = label
        correct += int(label == regime)
    confirms = bool(correct == len(regimes))
    return Finding(
        track="pathology", property="convergence-pathology fingerprints",
        capability="steppable",
        measured=dict(classified=results, correct=correct, total=len(regimes)),
        confirms=confirms,
        summary=f"{correct}/{len(regimes)} planted regimes named correctly",
        confirm_criterion="every planted regime's fingerprint classifies to its own label",
        falsify_criterion="regimes collapse to indistinguishable fingerprints (misclassification)",
    )
