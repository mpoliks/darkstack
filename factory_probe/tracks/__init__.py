"""Measurement tracks. Each track's `measure(substrate, **config) -> Finding`
reads a substrate through one capability and returns a falsifiable result.

TRACKS maps a track name to (measure_fn, required_capability). A harness runs the
subset whose capability the substrate provides.
"""
from __future__ import annotations

from . import versioning, pathology, catastrophe, governance, entrainment, stackelberg, opacity

TRACKS = {
    "versioning":  (versioning.measure,  "steppable"),
    "pathology":   (pathology.measure,   "steppable"),
    "catastrophe": (catastrophe.measure, "steppable"),
    "governance":  (governance.measure,  "steppable"),
    "entrainment": (entrainment.measure, "coupled_loops"),
    "stackelberg": (stackelberg.measure, "learner_game"),
    "opacity":     (opacity.measure,     "dividend_task"),
}

__all__ = ["TRACKS"]
