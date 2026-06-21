"""Diagnose, recommend, and verify: every lever must resolve its condition."""
import pytest

from factory_design import simulate, diagnose, recommend, verify
from factory_design.presets import get
from factory_design.levers import apply_lever


LEVER_CASES = [
    ("no_exploration_floor", "learning_death"),
    ("gameable_judge", "overfitting"),
    ("stuck_failing", "stable_failure"),
    ("never_settles", "thrash"),
    ("repricing_too_fast", "iatrogenic_thrash"),
    ("monoculture", "correlated_crash"),
]


@pytest.mark.parametrize("preset,condition", LEVER_CASES)
def test_lever_resolves_condition(preset, condition):
    v = verify(get(preset), condition, seeds=6, quick=True)
    assert v.resolved, f"{condition}: {v.metric} {v.before:.2f}->{v.after:.2f} did not resolve"


def test_recommend_is_a_diff_against_the_design():
    d = get("no_exploration_floor")
    rec = recommend(d, "learning_death")
    assert rec.knob == "explorer_floor"
    assert rec.before == d.explorer_floor
    assert rec.after != rec.before


def test_apply_lever_changes_only_its_knob():
    d = get("stuck_failing")
    fixed = apply_lever(d, "stable_failure")
    assert d.diff(fixed) == {"controller": ("off", "aggressive")}


def test_diagnose_lists_primary_and_secondary():
    rep = simulate(get("repricing_too_fast"), seeds=4, quick=True, early_warning=False)
    names = {c.name for c in diagnose(rep)}
    assert "iatrogenic_thrash" in names


def test_healthy_has_nothing_to_fix():
    rep = simulate(get("healthy"), seeds=4, quick=True, early_warning=False)
    assert diagnose(rep) == []
