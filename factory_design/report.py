"""The report a simulation returns: a headline verdict and a set of lenses.

A `Lens` is one view of the simulated factory -- its failure-mode verdict, its
distinct stable behaviours, its proximity to a cliff, the stability of its
governance loop, its exposure to a correlated crash. Every lens carries the real
harness thing it stands in for (`maps_to`) and the limit of what it can claim
(`limits`). A `FactoryReport` bundles the lenses with the provenance of the run
(how many seeds, how long, what was held fixed) so no number is read bare.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Lens:
    """One view of the simulated factory."""

    name: str
    summary: str                       # one line a non-specialist can read
    metrics: dict[str, Any] = field(default_factory=dict)
    maps_to: str = ""                  # the real harness thing this stands in for
    limits: str = ""                   # the limit of what the claim covers
    detail: dict[str, Any] = field(default_factory=dict)  # full data, for plots/sweeps

    def line(self) -> str:
        nums = "  ".join(f"{k}={_fmt(v)}" for k, v in self.metrics.items())
        return f"{self.summary}\n      {nums}" if nums else self.summary


@dataclass
class FactoryReport:
    """The result of `simulate(design)`."""

    design: Any                         # the FactoryDesign that produced this
    verdict: str                        # headline: "healthy" or a failure-mode name
    verdict_detail: str = ""            # one plain-English line explaining the verdict
    lenses: dict[str, Lens] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # ergonomics ------------------------------------------------------------
    def lens(self, name: str) -> Lens:
        return self.lenses[name]

    def __getitem__(self, name: str) -> Lens:
        return self.lenses[name]

    # rendering -------------------------------------------------------------
    def __str__(self) -> str:
        width = 78
        out = ["=" * width]
        out.append(f"  VERDICT: {self.verdict.upper()}")
        if self.verdict_detail:
            out.append(f"  {self.verdict_detail}")
        out.append("=" * width)
        for ln in self.lenses.values():
            out.append("")
            out.append(f"  [{ln.name}]  {ln.line()}")
            if ln.maps_to:
                out.append(f"      maps to: {ln.maps_to}")
            if ln.limits:
                out.append(f"      limits:  {ln.limits}")
        if self.provenance:
            out.append("")
            out.append("  " + "-" * (width - 2))
            prov = "  ".join(f"{k}={v}" for k, v in self.provenance.items())
            out.append(f"  provenance: {prov}")
        out.append("=" * width)
        return "\n".join(out)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"{v[0]:.3g}±{v[1]:.2g}"   # (median, spread)
    return str(v)
