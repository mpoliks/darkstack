"""The design invariant: every track runs against a substrate that implements
ONLY the capability interfaces' abstract methods -- no mock-only extras. A track
either produces a Finding or cleanly reports a sub-signal unavailable; none may
reach past the interface (which would raise AttributeError)."""
import numpy as np

from factory_probe.interfaces import (Substrate, SteppableFactory, LearnerGame,
                                      DividendTask, CoupledLoops)
from factory_probe.instrumentation import TraceStore
from factory_probe.records import Decision, Reward, RoundObs
from factory_probe.report import Finding
from factory_probe.tracks import TRACKS


class MinSteppable(SteppableFactory):
    def __init__(self):
        self.trace = TraceStore()
        self._t = 0
        self._rng = np.random.default_rng(0)
        self._lam = 0.0

    def reset(self, **config):
        self.trace = TraceStore()
        self._t = 0
        self._rng = np.random.default_rng(config.get("seed", 0))
        return self

    def behaviour(self):
        return dict(mean_pos=float(self._rng.random()), norm_sat=0.5,
                    metric_sat=0.5, variety=2.0)

    def set_price(self, lam):
        self._lam = float(lam)

    def step(self):
        o = self.behaviour()
        self.trace.record_behaviour(self._t, o)
        self.trace.record_control(self._t, dict(price=self._lam))
        d = Decision(f"{self._t}", self._t, "p", action=0, n_actions=2, propensity=0.5,
                     behaviour=o["mean_pos"])
        self.trace.record_decision(d)
        self.trace.record_reward(Reward(d.handle, self._t, 0.5))
        self._t += 1
        return RoundObs(round=self._t - 1, behaviour=o, control=dict(price=self._lam),
                        decisions=[d])


class MinGame(LearnerGame):
    def run(self, T, disclosure=0.0, follower="mean_based", n_actions=3, seed=0):
        return np.cumsum(np.full(T, 0.1)) / (np.arange(T) + 1)

    def stackelberg_value(self, n_actions=3):
        return 0.0

    def steerable_value(self, n_actions=3):
        return 0.5 if n_actions == 3 else 0.0


class MinDiv(DividendTask):
    def __init__(self, **spec):
        self._order = int(spec.get("order", 1))

    def free_floor(self, budget, seed=0):
        return 1.0

    def legible_floor(self, budget, order=1, seed=0):
        return max(0.0, 1.0 - 0.3 * (self._order - 1))

    def interaction_order(self, seed=0):
        return self._order


class MinLoops(CoupledLoops):
    # implements only the two abstract methods; condensate is intentionally absent
    def order_parameter(self, coupling, diversity, seed=0):
        return float(min(1.0, max(0.0, (coupling - 2 * diversity) + 0.4)))

    def critical_coupling(self, diversity):
        return 2.0 * diversity


class MinSubstrate(Substrate):
    name = "min"

    def capabilities(self):
        return {"steppable", "learner_game", "dividend_task", "coupled_loops"}

    def steppable(self, **config):
        return MinSteppable().reset(**config)

    def learner_game(self, **config):
        return MinGame()

    def dividend_task(self, **config):
        return MinDiv(**config)

    def coupled_loops(self, **config):
        return MinLoops()


_CONFIG = {
    "versioning":  dict(n=1500),
    "pathology":   dict(n=1200),
    "catastrophe": dict(ramp=np.r_[np.ones(1500), np.linspace(1, 0, 200)],
                        null=np.ones(1700) + 1e-3 * np.random.default_rng(0).standard_normal(1700),
                        fold_index=1500, w=300),
    "governance":  dict(seeds=(0,), T_gov=(3, 12), n=300),
    "entrainment": dict(N=40),
    "stackelberg": dict(T=400),
    "opacity":     dict(task_specs=[{"order": 1}, {"order": 2}, {"order": 3}], n_seeds=2, budget=64),
}


def test_every_track_runs_against_abc_only_substrate():
    sub = MinSubstrate()
    for name, (fn, cap) in TRACKS.items():
        f = fn(sub, **_CONFIG.get(name, {}))      # must not raise AttributeError
        assert isinstance(f, Finding)
        assert isinstance(f.confirms, bool)


def test_entrainment_handles_missing_condensate():
    # MinLoops has no condensate(); the track must catch NotImplementedError and
    # still return a Finding rather than crashing.
    from factory_probe.tracks import entrainment
    f = entrainment.measure(MinSubstrate(), N=40)
    assert isinstance(f, Finding)
    assert f.measured.get("condensate", "").startswith("unavailable")
