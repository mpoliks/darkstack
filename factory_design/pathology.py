"""The pathology classifier: a fingerprint of a population's behaviour, and the
failure mode it most resembles.

Four features summarise what a population does over its late window:

  settledness   how strongly it sits in one stable behaviour (1) vs swings across
                the axis and never commits (0)            -- separates THRASH
  variety       how many distinct assemblies stay in play; collapses toward 0 when
                exploration dies                          -- separates LEARNING_DEATH
  norm_unmet    how far the realised behaviour sits from the goal the spec wants
                                                          -- separates STABLE_FAILURE
  proxy_gap     how much better it scores on the metric than on the true goal -- the
                signature of gaming the eval             -- separates OVERFITTING

Classification is nearest-prototype in this 4-D space, with the separating
features up-weighted. It returns soft scores (a softmax over negative distance),
the nearest label, and an `ambiguous` flag when the top two are within a margin --
so a near-boundary run is reported as near-boundary, not forced to a crisp label.

The prototypes are claims that must survive falsification: every named preset must
classify as itself across many seeds (see `tests/test_pathology.py`, which builds
the confusion matrix). If a pair confuses, that is a finding, reported -- not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FEATURES = ("settledness", "variety", "norm_unmet", "proxy_gap")
CLASSES = ("healthy", "stable_failure", "overfitting", "learning_death", "thrash")

# Prototype fingerprint per class, in [settledness, variety, norm_unmet, proxy_gap].
# These are hand-set anchors, informed by the measured preset fingerprints but not
# equal to their centroids; the preset round-trip in tests/test_fd_pathology.py is
# the falsification gate -- it confirms the anchors SEPARATE the presets (each
# classifies as itself), not that any anchor equals its preset's centroid.
#   - settledness separates THRASH (it never commits)
#   - variety separates LEARNING_DEATH (exploration collapses to a single assembly)
#   - proxy_gap separates OVERFITTING (it scores the metric, misses the goal)
#   - norm_unmet separates STABLE_FAILURE from HEALTHY among the settled regimes
_PROTOTYPES = {
    "healthy":        np.array([0.88, 1.05, 0.07, 0.00]),
    "stable_failure": np.array([0.88, 1.15, 1.00, 0.00]),
    "overfitting":    np.array([0.88, 0.82, 1.00, 0.94]),
    "learning_death": np.array([1.00, 0.00, 0.20, 0.00]),
    "thrash":         np.array([0.20, 1.50, 0.72, 0.00]),
}
# Up-weight the feature that crisply separates each mechanism.
_WEIGHTS = np.array([3.0, 2.0, 1.3, 2.5])
_TEMP = 0.30          # softmax temperature on negative weighted distance
_MARGIN = 0.15        # score gap below which the top two classes are "ambiguous"

# Feature scaling: raw observables -> features.
_VOLATILITY_FULL = 0.04   # std(mean_pos) at/above which settledness reads 0 (never settles)
_VARIETY_SPAN = 0.60      # raw effective-assembly count above 1.0 mapped per unit of feature
_VARIETY_CAP = 1.50       # cap on the variety feature (a wandering population saturates it)


def fingerprint(mean_pos: np.ndarray, variety: np.ndarray,
                norm_sat: np.ndarray, metric_sat: np.ndarray,
                tail: int = 4000) -> dict:
    """Compute the four features from a single behavioural trace's late window."""
    n = len(mean_pos)
    w = slice(max(0, n - tail), n)
    volatility = float(np.std(mean_pos[w]))
    settledness = 1.0 - min(1.0, volatility / _VOLATILITY_FULL)
    variety_feat = float(np.clip((np.mean(variety[w]) - 1.0) / _VARIETY_SPAN, 0.0, _VARIETY_CAP))
    norm_unmet = 1.0 - float(np.mean(norm_sat[w]))
    proxy_gap = max(0.0, float(np.mean(metric_sat[w]) - np.mean(norm_sat[w])))
    return dict(settledness=settledness, variety=variety_feat,
                norm_unmet=norm_unmet, proxy_gap=proxy_gap,
                _volatility=volatility)


def _vec(fp: dict) -> np.ndarray:
    return np.array([fp[f] for f in FEATURES])


@dataclass
class Diagnosis:
    label: str
    scores: dict           # class -> softmax score in [0, 1]
    ambiguous: bool
    runner_up: str
    margin: float
    fingerprint: dict      # the feature vector that produced it

    def __str__(self) -> str:
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        body = ", ".join(f"{c}={s:.2f}" for c, s in ranked)
        flag = f"  (ambiguous vs {self.runner_up}, margin {self.margin:.2f})" if self.ambiguous else ""
        return f"{self.label}{flag}  [{body}]"


def classify(fp: dict) -> Diagnosis:
    """Nearest-prototype classification with soft scores and an ambiguity flag."""
    f = _vec(fp)
    dists = {c: float(np.sqrt(np.sum(_WEIGHTS * (f - p) ** 2))) for c, p in _PROTOTYPES.items()}
    # softmax over negative distance -> scores
    d = np.array([dists[c] for c in CLASSES])
    z = np.exp(-(d - d.min()) / _TEMP)
    s = z / z.sum()
    scores = {c: float(s[i]) for i, c in enumerate(CLASSES)}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (top, top_s), (run, run_s) = ranked[0], ranked[1]
    margin = top_s - run_s
    return Diagnosis(label=top, scores=scores, ambiguous=margin < _MARGIN,
                     runner_up=run, margin=margin, fingerprint=fp)


def aggregate(fps: list[dict]) -> dict:
    """Median feature vector over a seed ensemble (the seed-averaged fingerprint)."""
    return {f: float(np.median([fp[f] for fp in fps])) for f in FEATURES} | {
        "_volatility": float(np.median([fp["_volatility"] for fp in fps]))}


def confusion(preset_fps: dict[str, list[dict]]) -> dict:
    """Confusion matrix for a {preset_name: [per-seed fingerprints]} mapping:
    {true_name: {predicted_label: count}}. Used by the falsification test."""
    out = {}
    for name, fps in preset_fps.items():
        counts = {}
        for fp in fps:
            lab = classify(fp).label
            counts[lab] = counts.get(lab, 0) + 1
        out[name] = counts
    return out
