"""Out-of-loop realized-consequence scorer.

A producer's in-round verifier score and an evaluator's realized-consequence
score must come from different loops, or the two co-adapt onto a shared target.
This scorer grades a verdict (a flag an evaluator raised) against the outcome
that later materialised, and tracks the evaluator's precision and recall so the
realized-consequence reward can be paid only for verdicts that predicted real
outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsequenceScorer:
    reward_true_positive: float = 1.0
    reward_false_positive: float = -1.0
    reward_true_negative: float = 0.0
    reward_false_negative: float = -0.5
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    _scores: list = field(default_factory=list)

    def grade(self, flagged: bool, realized: bool) -> float:
        """Score one verdict against its realised outcome and return the reward."""
        if flagged and realized:
            self.tp += 1; r = self.reward_true_positive
        elif flagged and not realized:
            self.fp += 1; r = self.reward_false_positive
        elif not flagged and realized:
            self.fn += 1; r = self.reward_false_negative
        else:
            self.tn += 1; r = self.reward_true_negative
        self._scores.append(r)
        return r

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    def summary(self) -> dict:
        return dict(precision=self.precision(), recall=self.recall(),
                    tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)
