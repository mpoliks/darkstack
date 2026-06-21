"""Close the loop: apply the recommended lever and re-simulate to show it lands.

A recommendation is only worth printing if it actually resolves the condition on
the reference dynamics. `verify` simulates the design, applies the lever,
simulates again, and reports the before/after of the observable that matters for
that condition -- plus whether it crossed the bar. This is the design-time analogue
of `factory_probe`'s on-substrate verification: there the lever is exercised on a
live factory, here on the model whose dynamics are known.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .design import FactoryDesign
from .levers import recommend, apply_lever
from .simulate import simulate


@dataclass
class Verification:
    condition: str
    knob: str
    before_value: object
    after_value: object
    metric: str
    before: float
    after: float
    resolved: bool

    def __str__(self) -> str:
        mark = "OK" if self.resolved else "NOT RESOLVED"
        return (f"[{self.condition}] {self.knob}: {self.before_value!r} -> {self.after_value!r}\n"
                f"    {self.metric}: {self.before:.2f} -> {self.after:.2f}   [{mark}]")


# condition -> (metric label, how to read it from a report, direction, resolved test)
def _norm_sat_ungoverned(rep) -> float:
    return 1.0 - rep.lenses["pathology"].metrics["fingerprint"]["distance_from_goal"]


def _variety(rep) -> float:
    return rep.lenses["pathology"].metrics["fingerprint"]["variety"]


def _settledness(rep) -> float:
    return rep.lenses["pathology"].metrics["fingerprint"]["settledness"]


def _price_instability(rep) -> float:
    g = rep.lenses.get("governance")
    return g.metrics["price_instability"][0] if g and "price_instability" in g.metrics else float("nan")


def _governed_sat(rep) -> float:
    g = rep.lenses.get("governance")
    return g.metrics["final_satisfaction"][0] if g and "final_satisfaction" in g.metrics else float("nan")


def _sync(rep) -> float:
    e = rep.lenses.get("ecology")
    return e.metrics["synchronisation"] if e and "synchronisation" in e.metrics else float("nan")


def verify(design: FactoryDesign, condition: str, seeds: int = 8,
           quick: bool = False) -> Verification:
    """Apply the lever for `condition` and re-simulate; report before/after."""
    rec = recommend(design, condition)
    fixed = apply_lever(design, condition)

    def sim(d):
        return simulate(d, seeds=seeds, quick=quick, early_warning=False)

    if condition == "learning_death":
        b, a = sim(design), sim(fixed)
        before, after = _variety(b), _variety(a)
        metric, resolved = "effective variety", after > before + 0.2

    elif condition == "thrash":
        b, a = sim(design), sim(fixed)
        before, after = _settledness(b), _settledness(a)
        metric, resolved = "settledness", after > before + 0.15

    elif condition == "overfitting":
        b, a = sim(design), sim(fixed)
        before, after = _norm_sat_ungoverned(b), _norm_sat_ungoverned(a)
        metric, resolved = "true-goal satisfaction", after > before + 0.05

    elif condition == "stable_failure":
        b, a = sim(design), sim(fixed)
        before, after = _norm_sat_ungoverned(b), _governed_sat(a)
        metric, resolved = "goal satisfaction", after > before + 0.1

    elif condition == "iatrogenic_thrash":
        b, a = sim(design), sim(fixed)
        before, after = _price_instability(b), _price_instability(a)
        metric, resolved = "price instability", after < 0.7 * before

    elif condition == "correlated_crash":
        b, a = sim(design), sim(fixed)
        before, after = _sync(b), _sync(a)
        metric, resolved = "peer synchronisation", after < before - 0.3

    else:
        raise KeyError(f"no verifier for condition {condition!r}")

    return Verification(condition=condition, knob=rec.knob,
                        before_value=rec.before, after_value=rec.after,
                        metric=metric, before=float(before), after=float(after),
                        resolved=bool(resolved))
