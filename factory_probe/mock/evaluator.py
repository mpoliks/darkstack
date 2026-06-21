"""A planted evaluator that exercises the out-of-loop scorer.

Generates flagged/realised verdict pairs with a known sensitivity and
specificity, grades them through ConsequenceScorer, and returns the scorer -- so
a test can confirm the scorer recovers the planted precision and recall before it
is trusted on a live factory's verdict stream.
"""
from __future__ import annotations

import numpy as np

from ..scorer import ConsequenceScorer


def planted_evaluator(sensitivity: float = 0.8, specificity: float = 0.9,
                      base_rate: float = 0.3, n: int = 4000, seed: int = 0) -> ConsequenceScorer:
    rng = np.random.default_rng(seed)
    sc = ConsequenceScorer()
    for _ in range(n):
        realized = rng.random() < base_rate
        if realized:
            flagged = rng.random() < sensitivity
        else:
            flagged = rng.random() < (1.0 - specificity)
        sc.grade(flagged, realized)
    return sc
