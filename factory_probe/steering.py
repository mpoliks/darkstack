"""From measurement to action: recommend a boundary intervention for a diagnosed
condition, and verify on the reference substrate that applying it resolves the
condition (measure -> recommend -> re-measure).

The measurement tracks diagnose; this maps a diagnosis to the governance lever
that addresses it and closes the loop by re-measuring before and after the lever
is applied. On a live factory the operator applies the analogous knob; here each
lever is exercised on the mock, whose dynamics are known, so the resolution is
checkable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from .mock import MockSubstrate


@dataclass
class Intervention:
    condition: str
    knob: str
    action: str          # how to move the knob
    lever: str           # which boundary lever it is
    rationale: str


RECOMMENDATIONS = {
    "learning_death": Intervention(
        "learning_death", "exploration_floor", "raise", "exploration",
        "guarantee a protected mean-based explorer share / raise the exploration floor"),
    "thrash": Intervention(
        "thrash", "governance_cadence", "raise", "timing",
        "govern several times slower than the inner loop it steers (cascade ratio >= 3:1)"),
    "overfitting": Intervention(
        "overfitting", "metric_sampling_rate", "raise", "evaluation",
        "sample the metric above the norm's Nyquist rate (raise its rate, resolution, and variety)"),
    "stable_failure": Intervention(
        "stable_failure", "price_gain", "raise", "pricing",
        "ratchet the price on failure duration so the factory leaves the failing attractor"),
    "condensate": Intervention(
        "condensate", "dependency_diversity", "raise", "dependency",
        "diversify the dependency class so coupling falls below Kc (a one-shot halt re-locks)"),
}


def recommend(condition: str) -> Intervention:
    if condition not in RECOMMENDATIONS:
        raise KeyError(f"no recommendation for {condition!r}; known: {sorted(RECOMMENDATIONS)}")
    return RECOMMENDATIONS[condition]


# --- closed-loop verification on the reference substrate ---------------------
def _verify_learning_death(sub, seed=1, n=6000) -> dict:
    base = sub.steppable(regime="learning_death", seed=seed, n_sample=1); base.run(n)
    before = float(base.trace.behaviour_series("variety")[-2000:].mean())
    fixed = sub.steppable(regime="learning_death", seed=seed, n_sample=1, mu=0.06); fixed.run(n)
    after = float(fixed.trace.behaviour_series("variety")[-2000:].mean())
    return dict(metric="effective variety", before=round(before, 2), after=round(after, 2),
                resolved=after > before + 0.5)


def _verify_thrash(sub, seed=0, fast=3, slow=60) -> dict:
    from .tracks.governance import _instability
    before = _instability(sub, T_gov=fast, seed=seed, n=2000)   # govern too fast
    after = _instability(sub, T_gov=slow, seed=seed, n=2000)    # cascade-respecting
    # require a substantial drop: a within-band nudge (e.g. 3->4) drifts only a few
    # percent and must NOT count as resolving the cascade story
    return dict(metric="price instability", before=round(before, 2), after=round(after, 2),
                resolved=after < 0.7 * before)


def _verify_stable_failure(sub, seed=3, n=4000, target=0.8) -> dict:
    from control import PID
    params = dict(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=400, mu=0.03, c=0.9,
                  norm_target=0.7, cost_center=0.30, cost_width=0.12, seed=seed, n_sample=1)
    base = sub.steppable(**params); base.run(n)                 # ungoverned: stuck failing
    before = float(base.trace.behaviour_series("norm_sat")[-1500:].mean())
    gov = sub.steppable(**params)                               # priced governance leaves the attractor
    pid = PID(Kp=8.0, Ki=0.5, Kd=2.0, hi=40)
    lam = 0.0; T_gov = 40; ns = []
    for t in range(n):
        gov.set_price(lam)
        ns.append(gov.step().behaviour["norm_sat"])
        if (t + 1) % T_gov == 0:
            lam = pid.update(target - float(np.mean(ns[-T_gov:])), dt=T_gov)
    after = float(np.mean(ns[-1500:]))
    return dict(metric="norm satisfaction", before=round(before, 2), after=round(after, 2),
                resolved=after > before + 0.1)


def _verify_condensate(sub, seed=7) -> dict:
    loops = sub.coupled_loops(N=400, T=50)
    before = loops.order_parameter(coupling=2.2, diversity=0.3, seed=seed)   # Kc=0.6 < 2.2 -> locked
    after = loops.order_parameter(coupling=2.2, diversity=1.5, seed=seed)    # Kc=3.0 > 2.2 -> unlocked
    return dict(metric="synchronisation r", before=round(before, 2), after=round(after, 2),
                resolved=after < before - 0.3)


def _verify_overfitting(sub, seed=1, n=10000) -> dict:
    common = dict(K=80, heightA=0.0, c=0.0, align_weight=4.0, align_width=0.05, eta=6, M=400,
                  mu=0.02, norm_target=0.5, norm_amp=0.16, norm_freq=2 * np.pi / 120, seed=seed,
                  n_sample=1)
    slow = sub.steppable(metric_sample_period=60, **common); slow.run(n)   # under-sampled metric
    before = float(slow.trace.behaviour_series("norm_sat")[-5000:].mean())
    fast = sub.steppable(metric_sample_period=6, **common); fast.run(n)    # Nyquist-respecting
    after = float(fast.trace.behaviour_series("norm_sat")[-5000:].mean())
    return dict(metric="norm satisfaction", before=round(before, 2), after=round(after, 2),
                resolved=after > before + 0.05)


_VERIFIERS = {
    "learning_death": _verify_learning_death,
    "thrash": _verify_thrash,
    "overfitting": _verify_overfitting,
    "stable_failure": _verify_stable_failure,
    "condensate": _verify_condensate,
}


def verify_on_mock(condition: str, substrate=None) -> dict:
    """Apply the recommended lever for `condition` on the reference substrate and
    return before/after of the relevant observable plus whether it resolved."""
    sub = substrate or MockSubstrate()
    res = _VERIFIERS[condition](sub)
    res.update(condition=condition, intervention=asdict(recommend(condition)))
    return res


def run_all(substrate=None) -> list:
    return [verify_on_mock(c, substrate) for c in _VERIFIERS]
