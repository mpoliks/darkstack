"""Trace records an instrumented factory emits.

These are the only objects the measurement tracks read. A live substrate adapter
produces the same records from real agent traffic; the measurements never see a
factory's internals, only this stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Decision:
    """One producer decision.

    handle        persistent id a later score attaches to (the return-queue key)
    round         loop index when the decision was made
    agent         producer identity; fungible, used only for grouping
    action        index of the chosen action within the action set
    n_actions     size of the action set the decision was drawn from
    propensity    probability the producer assigned to the chosen action at
                  decision time (the road-taken weight)
    distribution  full decision-time distribution over the action set, when the
                  producer discloses it; empty otherwise
    behaviour     scalar behavioural coordinate at decision time (e.g. the
                  population's mean output position)
    ts            wall-clock timestamp if the substrate is asynchronous; NaN when
                  the factory is globally clocked and `round` is the time axis
    """

    handle: str
    round: int
    agent: str
    action: int
    n_actions: int
    propensity: float
    distribution: Sequence[float] = ()
    behaviour: float = float("nan")
    ts: float = float("nan")


@dataclass
class Reward:
    """A score attached to an earlier decision by its handle.

    kind   'in_round'   immediate verifier score
           'regression' delayed regression-prediction score
           'realized'   out-of-loop realized-consequence score (graded outside the
                        loop the scored producer runs in)
    ts     wall-clock timestamp if asynchronous; NaN under a globally clocked loop
    """

    handle: str
    round: int
    score: float
    kind: str = "in_round"
    ts: float = float("nan")


@dataclass
class RoundObs:
    """What one factory round returns: the behavioural observables, the control
    input applied this round (price and any injected intent shift), the decisions
    made, and any rewards that came due."""

    round: int
    behaviour: dict
    control: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
