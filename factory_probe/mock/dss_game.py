"""A committed-leader game with a follower learner of a chosen class.

The leader commits a bait schedule; the follower is a mean-based no-regret learner
or a no-swap-regret learner. The propensity-disclosure level blends the mean-based
follower toward its swap-corrected counterpart. Each round logs the follower's
decision, its decision-time propensity, and the reward it received.
"""
from __future__ import annotations

import numpy as np

from darkfactory import _U_O, _U_L, leader_schedule, stackelberg_value
from learners import Hedge, BlumMansourSwap
from ..records import Decision, Reward
from ..instrumentation import TraceStore
from ..interfaces import LearnerGame


class MockDSSGame(LearnerGame):
    def __init__(self):
        self.trace = TraceStore()

    def run(self, T: int, disclosure: float = 0.0, follower: str = "mean_based",
            n_actions: int = 3, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        UL = _U_L[:, :n_actions]
        UO = _U_O[:, :n_actions]
        hedge = Hedge(n_actions, eta=0.7, rng=rng)
        swap = BlumMansourSwap(n_actions, eta=0.7, rng=rng)
        sched = leader_schedule(T)
        gain = np.empty(T)
        self.trace = TraceStore()
        for t in range(T):
            row = int(sched[t])
            ph = hedge.distribution()
            ps = swap.distribution()
            p = ps if follower == "no_swap" else (1 - disclosure) * ph + disclosure * ps
            p = p / p.sum()
            col = int(rng.choice(n_actions, p=p))
            gain[t] = UO[row, col]
            d = Decision(handle=f"g{t}", round=t, agent="follower", action=col,
                         n_actions=n_actions, propensity=float(p[col]),
                         distribution=tuple(float(x) for x in p), behaviour=float(UO[row, col]))
            self.trace.record_decision(d)
            self.trace.record_reward(Reward(handle=d.handle, round=t,
                                            score=float(UL[row, col]), kind="in_round"))
            self.trace.record_behaviour(t, {"extracted": float(UO[row, col]), "action": col})
            reward = UL[row, :].copy()
            hedge.update(reward)
            swap.update(reward)
        return np.cumsum(gain) / (np.arange(T) + 1)

    def stackelberg_value(self, n_actions: int = 3) -> float:
        return float(stackelberg_value(n_actions))

    def steerable_value(self, n_actions: int = 3) -> float:
        if n_actions == 3:
            return 0.5                                  # U* of the validated 3-action matrix
        if n_actions == 2:
            return float(stackelberg_value(2))          # U* == V when the learner has 2 actions
        raise NotImplementedError(
            "U* is planted only for the 2- and 3-action Deng-Schneider-Sivan game")
