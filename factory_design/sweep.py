"""Explore the design space: sweep one knob, or compare two designs.

`sweep` runs a design across values of one knob and reports where the verdict
stays healthy -- the operating region you can move in without tipping into a
failure mode. `compare` puts two designs side by side: their verdicts and the
knobs that differ. Both are thin loops over `simulate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .design import FactoryDesign, KNOBS
from .simulate import simulate


@dataclass
class SweepPoint:
    value: object
    verdict: str
    detail: str


@dataclass
class SweepResult:
    knob: str
    points: list[SweepPoint] = field(default_factory=list)

    @property
    def healthy_region(self) -> list:
        return [p.value for p in self.points if p.verdict == "healthy"]

    def __str__(self) -> str:
        out = [f"sweep of {self.knob}:"]
        for p in self.points:
            mark = "  ok" if p.verdict == "healthy" else f"  -> {p.verdict}"
            out.append(f"    {self.knob}={p.value!r:>10}  {p.verdict}{mark if p.verdict!='healthy' else ''}")
        region = self.healthy_region
        if region:
            out.append(f"  healthy at {self.knob} in {{{', '.join(map(str, region))}}}")
        else:
            out.append(f"  no healthy {self.knob} in the swept range")
        return "\n".join(out)


def sweep(design: FactoryDesign, knob: str, values, seeds: int = 8,
          quick: bool = True) -> SweepResult:
    """Vary one knob across `values` and report the verdict at each."""
    if knob not in KNOBS:
        raise KeyError(f"unknown knob {knob!r}; known: {sorted(KNOBS)}")
    res = SweepResult(knob=knob)
    for v in values:
        rep = simulate(design.with_(**{knob: v}), seeds=seeds, quick=quick, early_warning=False)
        res.points.append(SweepPoint(value=v, verdict=rep.verdict, detail=rep.verdict_detail))
    return res


@dataclass
class Comparison:
    verdict_a: str
    verdict_b: str
    diff: dict                       # {knob: (a_value, b_value)}

    def __str__(self) -> str:
        out = [f"A: {self.verdict_a}", f"B: {self.verdict_b}", "differences:"]
        if not self.diff:
            out.append("    (identical designs)")
        for knob, (a, b) in self.diff.items():
            out.append(f"    {knob}: {a!r} (A) vs {b!r} (B)")
        return "\n".join(out)


def compare(design_a: FactoryDesign, design_b: FactoryDesign, seeds: int = 8,
            quick: bool = True) -> Comparison:
    """Simulate two designs and report both verdicts and the knobs that differ."""
    ra = simulate(design_a, seeds=seeds, quick=quick, early_warning=False)
    rb = simulate(design_b, seeds=seeds, quick=quick, early_warning=False)
    return Comparison(verdict_a=ra.verdict, verdict_b=rb.verdict, diff=design_a.diff(design_b))
