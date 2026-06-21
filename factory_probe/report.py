"""The result of a measurement: a Finding, and a Report that tabulates findings.

A Finding states one property, the numbers measured, whether the expected signal
was present, and the criteria that would confirm or falsify it on this run -- so
the same object reads as a result on the mock (against planted truth) and as a
falsification test on a live factory (where the truth is unknown).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Finding:
    track: str
    property: str
    capability: str
    measured: dict
    confirms: bool
    summary: str
    confirm_criterion: str
    falsify_criterion: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    findings: list = field(default_factory=list)

    def add(self, f: Finding) -> None:
        self.findings.append(f)

    def to_dict(self) -> dict:
        return {f.track: f.to_dict() for f in self.findings}

    def table(self) -> str:
        rows = [("track", "property", "signal", "summary")]
        for f in self.findings:
            rows.append((f.track, f.property, "present" if f.confirms else "absent",
                         f.summary))
        w = [max(len(r[i]) for r in rows) for i in range(4)]
        w[3] = min(w[3], 66)
        out = []
        for i, r in enumerate(rows):
            line = "  ".join(s.ljust(w[j])[:w[j]] for j, s in enumerate(r))
            out.append(line)
            if i == 0:
                out.append("  ".join("-" * w[j] for j in range(4)))
        n_ok = sum(f.confirms for f in self.findings)
        out.append("")
        out.append(f"{n_ok}/{len(self.findings)} expected signals present")
        return "\n".join(out)

    def falsification_table(self) -> str:
        out = []
        for f in self.findings:
            out.append(f"[{f.track}] {f.property}")
            out.append(f"    confirms if : {f.confirm_criterion}")
            out.append(f"    falsifies if: {f.falsify_criterion}")
            out.append(f"    measured    : {f.measured}")
            out.append("")
        return "\n".join(out)
