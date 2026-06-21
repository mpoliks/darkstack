"""The design language: validation, translation to dynamics, ergonomics."""
import math
import pytest

from factory_design import FactoryDesign, KNOBS


def test_defaults_are_a_valid_design():
    d = FactoryDesign()
    p = d.to_params(seed=0)
    assert p.K == d.n_options and p.M == d.population_size
    assert p.mu == d.effective_explore


def test_validation_rejects_bad_knobs():
    with pytest.raises(ValueError):
        FactoryDesign(judge_fidelity=1.5)
    with pytest.raises(ValueError):
        FactoryDesign(controller="sometimes")
    with pytest.raises(ValueError):
        FactoryDesign(eval_period=0)
    with pytest.raises(ValueError):
        FactoryDesign(n_options=2)


def test_explorer_floor_protects_a_minimum():
    d = FactoryDesign(explore_rate=0.0001, explorer_floor=0.05)
    assert d.effective_explore == 0.05
    assert d.to_params().mu == 0.05


def test_judge_fidelity_routes_to_overfitting_machinery():
    perfect = FactoryDesign(judge_fidelity=1.0).to_params()
    assert perfect.metric_static == pytest.approx(perfect.norm_target)
    gameable = FactoryDesign(judge_fidelity=0.2, norm_target=0.7).to_params()
    assert gameable.align_weight > 0
    assert gameable.metric_static != pytest.approx(gameable.norm_target)


def test_drift_routes_to_sampling():
    d = FactoryDesign(goal_drift=0.15, drift_period=120, eval_period=30).to_params()
    assert d.norm_amp == pytest.approx(0.15)
    assert d.metric_sample_period == 30
    assert d.norm_freq == pytest.approx(2 * math.pi / 120)


def test_controller_presets():
    assert FactoryDesign(controller="off").pid_gains() is None
    assert FactoryDesign(controller="gentle").pid_gains()["Kp"] < \
           FactoryDesign(controller="aggressive").pid_gains()["Kp"]


def test_with_and_diff():
    a = FactoryDesign()
    b = a.with_(explore_rate=0.2)
    assert b.explore_rate == 0.2 and a.explore_rate == 0.06
    assert a.diff(b) == {"explore_rate": (0.06, 0.2)}
    with pytest.raises(KeyError):
        a.with_(nonsense=1)


def test_every_knob_is_documented():
    from dataclasses import fields
    for f in fields(FactoryDesign):
        assert f.name in KNOBS, f"{f.name} missing from KNOBS"
        gloss, maps_to = KNOBS[f.name]
        assert gloss and maps_to
