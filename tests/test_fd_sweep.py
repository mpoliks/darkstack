"""Sweep and compare."""
from factory_design import FactoryDesign
from factory_design.sweep import sweep, compare
from factory_design.presets import get


def test_sweep_finds_a_healthy_region():
    res = sweep(FactoryDesign(), "explore_rate", [0.0005, 0.06, 0.3], seeds=5, quick=True)
    assert len(res.points) == 3
    assert 0.06 in res.healthy_region          # the default is healthy
    assert 0.0005 not in res.healthy_region    # too little exploration -> learning death


def test_compare_reports_both_verdicts_and_the_diff():
    c = compare(get("healthy"), get("no_exploration_floor"), seeds=5, quick=True)
    assert c.verdict_a == "healthy"
    assert c.verdict_b == "learning_death"
    assert "explore_rate" in c.diff
