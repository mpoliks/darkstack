"""Read a report and list the conditions worth fixing.

A design can have more than one problem at once: the population can be healthy
while governance reprices too fast, or a sound factory can be exposed to a
correlated crash through shared infrastructure. `diagnose` collects every
condition the report shows, ordered with the headline failure mode first.
"""
from __future__ import annotations

from dataclasses import dataclass

from .report import FactoryReport
from .simulate import _PRICE_INSTAB_THRESH


@dataclass
class Condition:
    name: str
    severity: str            # "primary" (the headline verdict) or "secondary"
    evidence: str

    def __str__(self) -> str:
        return f"{self.name} ({self.severity}): {self.evidence}"


def diagnose(report: FactoryReport) -> list[Condition]:
    conds: list[Condition] = []

    # the headline population failure mode
    if report.verdict != "healthy":
        path = report.lenses.get("pathology")
        ev = report.verdict_detail or ""
        if path and path.metrics.get("fingerprint"):
            ev += f"  fingerprint={path.metrics['fingerprint']}"
        conds.append(Condition(report.verdict, "primary", ev.strip()))

    # governance loop oscillating (iatrogenic thrash). The robust signal is the
    # measured price oscillation; the cascade ratio, when available, explains why.
    gov = report.lenses.get("governance")
    if gov and "price_instability" in gov.metrics:
        instab = gov.metrics["price_instability"][0]
        ratio = gov.metrics.get("cascade_ratio")
        below_cascade = ratio is not None and ratio[0] < 3.0
        if instab > _PRICE_INSTAB_THRESH or below_cascade:
            why = f"price oscillation {instab:.2f}"
            if ratio is not None:
                why += f", cascade ratio {ratio[0]}"
            conds.append(Condition("iatrogenic_thrash", "secondary",
                                   f"the governance loop is oscillating ({why})"))

    # exposure to a correlated crash through shared infrastructure
    eco = report.lenses.get("ecology")
    if eco and eco.metrics.get("synchronisation", 0) > 0.5 \
            and eco.metrics.get("coupling", 0) > eco.metrics.get("onset_threshold", 1e9):
        conds.append(Condition(
            "correlated_crash", "secondary",
            f"peers synchronise (r={eco.metrics['synchronisation']}) above the onset threshold"))

    return conds
