"""The pathology classifier and its falsification gate.

The load-bearing test: every pathology preset classifies as its own failure mode
across a seed ensemble (the confusion matrix is diagonal). If this fails, the
classifier's prototypes no longer match the dynamics and the verdict is unsound.
"""
import numpy as np
import pytest

from factory_design import pathology
from factory_design.simulate import _run
from factory_design.presets import PRESETS, EXPECTED, PATHOLOGY_PRESETS


def _fp(design, seed, n=12000, tail=4000):
    r = _run(design, seed, n)
    return pathology.fingerprint(r["mean_pos"], r["variety"], r["norm_sat"], r["metric_sat"], tail)


@pytest.mark.parametrize("name", PATHOLOGY_PRESETS)
def test_preset_round_trips_to_its_failure_mode(name):
    fps = [_fp(PRESETS[name], s) for s in range(8)]
    labels = [pathology.classify(fp).label for fp in fps]
    want = EXPECTED[name]
    hits = sum(1 for lab in labels if lab == want)
    assert hits == len(labels), f"{name}: {hits}/{len(labels)} -> {want}; got {set(labels)}"


def test_scores_are_a_distribution():
    fp = _fp(PRESETS["healthy"], 0)
    diag = pathology.classify(fp)
    assert abs(sum(diag.scores.values()) - 1.0) < 1e-6
    assert set(diag.scores) == set(pathology.CLASSES)


def test_aggregate_is_a_median():
    fps = [_fp(PRESETS["healthy"], s) for s in range(4)]
    agg = pathology.aggregate(fps)
    for f in pathology.FEATURES:
        assert agg[f] == pytest.approx(np.median([fp[f] for fp in fps]))


def test_ambiguity_flag_present():
    fp = _fp(PRESETS["healthy"], 0)
    diag = pathology.classify(fp)
    assert isinstance(diag.ambiguous, bool)
