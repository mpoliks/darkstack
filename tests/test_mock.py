"""The reference substrate reproduces its planted ground truth."""
import numpy as np
import pytest

from factory_probe.mock import MockSubstrate, planted_evaluator


def test_capabilities():
    assert MockSubstrate().capabilities() == {
        "steppable", "learner_game", "dividend_task", "coupled_loops"}


def test_inject_contract_applies_and_rejects():
    fac = MockSubstrate().steppable(regime="governance", seed=0)
    assert fac.injectable_channels() == {"spec_target", "spec_height"}
    before = fac._f.p.norm_target
    fac.inject(spec_target=0.05)
    assert abs(fac._f.p.norm_target - (before + 0.05)) < 1e-9
    with pytest.raises(KeyError):
        fac.inject(nonsense=1.0)


def test_ustar_planted_per_action_count():
    g = MockSubstrate().learner_game()
    assert abs(g.steerable_value(3) - 0.5) < 1e-9
    assert abs(g.steerable_value(2) - g.stackelberg_value(2)) < 1e-9
    with pytest.raises(NotImplementedError):
        g.steerable_value(4)


def test_delayed_realized_channel_recovers_lag():
    fac = MockSubstrate().steppable(regime="versions", seed=2, n_sample=2, realized_delay=5)
    fac.run(200)
    lags = fac.trace.lags("realized")
    assert len(lags) > 0 and np.median(lags) == 5
    assert "realized" in fac.trace.summary()["reward_kinds"]


def test_planted_evaluator_recovers_recall():
    sc = planted_evaluator(sensitivity=0.8, specificity=0.9, base_rate=0.3, n=6000, seed=0)
    assert abs(sc.recall() - 0.8) < 0.05


def test_dividend_task_order_recovery():
    sub = MockSubstrate()
    sep = sub.dividend_task(d=10, order=1, kind="nk", seed=0)   # NK K=0 -> separable
    assert sep.walsh_order() == 1 and sep.dividend(budget=1024, order=1) < 0.1
    inter = sub.dividend_task(d=10, order=3, kind="nk", seed=0)
    assert inter.walsh_order() == 3 and inter.dividend(budget=1024, order=1) > 0.2
