"""Boundary levers: for each diagnosed condition, the one knob to move.

You do not reach into a running factory and fix it; you move a boundary lever -- a
design-time commitment -- and let the factory reorganise. Each lever here names the
knob, the direction, and the harness thing it stands in for. `recommend()` renders
a lever as a concrete diff against the design you actually wrote, so the advice is
"you set X = a; set it to b", not a generic sentence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .design import FactoryDesign


@dataclass
class Lever:
    condition: str
    knob: str
    rationale: str
    maps_to: str
    _propose: Callable[[FactoryDesign], object]   # design -> suggested new value for `knob`

    def proposed(self, design: FactoryDesign):
        return self._propose(design)


# Each lever moves exactly one knob, in the direction its mechanism prescribes.
LEVERS: dict[str, Lever] = {
    "learning_death": Lever(
        "learning_death", "explorer_floor",
        "guarantee a protected exploration share the kernel will not let scoring starve",
        "reserve compute for exploratory agents (raise the protected floor)",
        lambda d: max(0.06, round(d.explorer_floor + 0.06, 3)),
    ),
    "thrash": Lever(
        "thrash", "explore_rate",
        "the frontier scatters the population every round and the retentive core cannot hold a "
        "result; lower exploration so the core can settle on good outcomes",
        "dial down exploration / non-greedy sampling so proven behaviour survives between rounds",
        lambda d: 0.06,
    ),
    "overfitting": Lever(
        "overfitting", "eval_period",
        "sample the goal faster than it drifts (beat the aliasing that lets the factory game the metric)",
        "raise eval frequency -- re-score against ground truth more often",
        lambda d: max(1, d.eval_period // 4 if d.eval_period > 4 else 1),
    ),
    "overfitting_judge": Lever(
        "overfitting", "judge_fidelity",
        "make the metric track the true goal (a gameable judge is the overfitting)",
        "use a stronger / less gameable judge or rubric (raise judge fidelity)",
        lambda d: round(min(1.0, max(d.judge_fidelity + 0.5, 0.8)), 2),
    ),
    "stable_failure": Lever(
        "stable_failure", "controller",
        "price the failing attractor so the factory is pushed to leave it",
        "turn on / harden the governance controller that penalises constraint violation",
        lambda d: "aggressive",
    ),
    "iatrogenic_thrash": Lever(
        "iatrogenic_thrash", "repricing_period",
        "govern several times slower than the inner loop settles (cascade ratio >= 3:1)",
        "reprice less often -- let the factory settle between governance moves",
        lambda d: int(max(d.repricing_period * 4, 60)),
    ),
    "correlated_crash": Lever(
        "correlated_crash", "dependency_diversity",
        "diversify dependencies so coupling falls below the synchronisation threshold",
        "spread vendors / models / schedules across peers (raise dependency diversity)",
        lambda d: round(d.dependency_diversity + max(0.6, d.shared_dependency), 2),
    ),
}


def lever_for(condition: str, design: FactoryDesign) -> Lever:
    """Pick the lever for a condition. Overfitting has two routes -- a slow eval
    (aliasing) or a gameable judge -- chosen by which one the design actually has."""
    if condition == "overfitting":
        if design.goal_drift > 0 and design.eval_period > 4:
            return LEVERS["overfitting"]
        if design.judge_fidelity < 1.0:
            return LEVERS["overfitting_judge"]
        return LEVERS["overfitting"]
    if condition not in LEVERS:
        raise KeyError(f"no lever for {condition!r}; known: {sorted(set(l.condition for l in LEVERS.values()))}")
    return LEVERS[condition]


@dataclass
class Recommendation:
    condition: str
    knob: str
    before: object
    after: object
    rationale: str
    maps_to: str

    def __str__(self) -> str:
        return (f"[{self.condition}] set {self.knob}: {self.before!r} -> {self.after!r}\n"
                f"    why: {self.rationale}\n"
                f"    in your harness: {self.maps_to}")


def recommend(design: FactoryDesign, condition: str) -> Recommendation:
    """The lever for `condition`, as a concrete diff against `design`."""
    lev = lever_for(condition, design)
    return Recommendation(condition=condition, knob=lev.knob,
                          before=getattr(design, lev.knob), after=lev.proposed(design),
                          rationale=lev.rationale, maps_to=lev.maps_to)


def apply_lever(design: FactoryDesign, condition: str) -> FactoryDesign:
    """Return the design with the recommended lever applied."""
    rec = recommend(design, condition)
    return design.with_(**{rec.knob: rec.after})
