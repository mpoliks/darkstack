"""The reusable bounded legible reader, and V/U* derived from run()."""
import numpy as np

from factory_probe.legible import FeatureDividendTask
from factory_probe.adapters.skeleton import LiveLearnerGame


def _task(order, n=1500, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    y = np.sin(2 * X[:, 0]) + X[:, 1] + 0.5 * X[:, 2]      # additive part
    if order >= 2:
        y = y + 2.0 * X[:, 0] * X[:, 1]                    # an interaction
    y = y + 0.05 * rng.standard_normal(n)
    cands = np.arange(n)
    return FeatureDividendTask(
        sample=lambda budget, s: cands if not budget else cands[:budget],
        featurize=lambda c: X[np.asarray(c)],
        verifier_score=lambda c: y[np.asarray(c)],
        kind="reg")


def test_additive_task_pays_near_zero():
    div = _task(order=1).dividend(None)
    assert div < 0.05                       # an additive reader loses almost nothing


def test_interacting_task_pays_a_real_floor():
    div = _task(order=2).dividend(None)
    assert div > 0.1                        # the interaction is value the additive reader forgoes


def test_interaction_order_detects_interactions():
    assert _task(order=2).interaction_order() >= 2


def test_V_and_Ustar_are_derived_from_run():
    # a live game only implements run(); V and U* must read off it
    class FakeGame(LiveLearnerGame):
        def run(self, T, disclosure=0.0, follower="mean_based", n_actions=3, seed=0):
            val = 0.0 if follower == "no_swap" else 0.5     # no-swap caps at V, mean-based reaches U*
            return np.full(T, val)

    g = FakeGame(client=None)
    assert g.stackelberg_value(3) == 0.0
    assert g.steerable_value(3) == 0.5
