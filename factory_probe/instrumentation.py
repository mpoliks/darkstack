"""Instrumentation a factory needs before any property can be measured.

Two objects:

  TraceStore   collects decisions, rewards, per-round behaviour, and the per-round
               control input; joins a reward back to the decision that earned it by
               handle; exposes the behavioural and control series the dynamical
               measurements consume.
  ReturnQueue  the addressable, stateful reward channel: a sampled action gets a
               persistent handle, and a later score finds it after a delay. An
               optional jitter decorrelates release times across producers.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .records import Decision, Reward


class TraceStore:
    def __init__(self):
        self.decisions: list[Decision] = []
        self.rewards: list[Reward] = []
        self._decision_by_handle: dict[str, Decision] = {}
        self._rewards_by_handle: dict[str, list[Reward]] = {}
        self._behaviour: dict[int, dict] = {}
        self._control: dict[int, dict] = {}

    # ---- ingest -------------------------------------------------------------
    def record_decision(self, d: Decision) -> None:
        self.decisions.append(d)
        self._decision_by_handle[d.handle] = d

    def record_reward(self, r: Reward) -> None:
        self.rewards.append(r)
        self._rewards_by_handle.setdefault(r.handle, []).append(r)

    def record_behaviour(self, rnd: int, obs: dict) -> None:
        self._behaviour[rnd] = dict(obs)

    def record_control(self, rnd: int, control: dict) -> None:
        """The control input applied this round (e.g. governance price, injected
        intent shift). Symmetric with behaviour: the input side of identification."""
        self._control[rnd] = dict(control)

    # ---- read ---------------------------------------------------------------
    def behaviour_series(self, key: str) -> np.ndarray:
        """The scalar behavioural observable `key` over rounds, in round order."""
        rounds = sorted(self._behaviour)
        return np.array([self._behaviour[r].get(key, np.nan) for r in rounds], dtype=float)

    def control_series(self, key: str) -> np.ndarray:
        """The scalar control input `key` over rounds, in round order."""
        rounds = sorted(self._control)
        return np.array([self._control[r].get(key, np.nan) for r in rounds], dtype=float)

    def rounds(self) -> np.ndarray:
        return np.array(sorted(self._behaviour), dtype=int)

    def joined(self, kind: Optional[str] = None):
        """(decision, score) pairs for decisions that received a reward. If `kind`
        is given, only rewards of that kind are joined. A decision with several
        rewards of the kind contributes the mean score."""
        out = []
        for h, d in self._decision_by_handle.items():
            rs = self._rewards_by_handle.get(h, [])
            if kind is not None:
                rs = [r for r in rs if r.kind == kind]
            if rs:
                out.append((d, float(np.mean([r.score for r in rs]))))
        return out

    def lags(self, kind: str = "realized") -> np.ndarray:
        """Round gap between each reward of `kind` and the decision it scored --
        recovers the reward channel's delay distribution."""
        out = []
        for r in self.rewards:
            if r.kind != kind:
                continue
            d = self._decision_by_handle.get(r.handle)
            if d is not None:
                out.append(r.round - d.round)
        return np.array(out, dtype=float)

    def propensity_reward(self, kind: Optional[str] = None):
        """Arrays (propensity, score) over joined decisions -- the raw material for
        importance-weighted counterfactual estimates."""
        pairs = self.joined(kind)
        if not pairs:
            return np.empty(0), np.empty(0)
        prop = np.array([d.propensity for d, _ in pairs])
        sc = np.array([s for _, s in pairs])
        return prop, sc

    def summary(self) -> dict:
        return dict(n_decisions=len(self.decisions), n_rewards=len(self.rewards),
                    n_rounds=len(self._behaviour),
                    reward_kinds=sorted({r.kind for r in self.rewards}),
                    behaviour_keys=sorted({k for o in self._behaviour.values() for k in o}),
                    control_keys=sorted({k for o in self._control.values() for k in o}))


class ReturnQueue:
    """Holds outstanding decisions until their score is due.

    delay   base number of rounds between a decision and its score
    jitter  uniform integer jitter added to the delay per item (decorrelates the
            release schedule across producers; 0 disables it)
    """

    def __init__(self, delay: int = 0, jitter: int = 0, rng: Optional[np.random.Generator] = None):
        self.delay = int(delay)
        self.jitter = int(jitter)
        self.rng = rng or np.random.default_rng(0)
        self._due: dict[int, list[str]] = {}

    def enqueue(self, handle: str, rnd: int) -> int:
        j = int(self.rng.integers(0, self.jitter + 1)) if self.jitter > 0 else 0
        due_round = rnd + self.delay + j
        self._due.setdefault(due_round, []).append(handle)
        return due_round

    def due(self, rnd: int) -> list[str]:
        """Handles whose score is due at `rnd` (and removes them from the queue)."""
        return self._due.pop(rnd, [])

    def outstanding(self) -> int:
        return sum(len(v) for v in self._due.values())
