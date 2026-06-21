"""Template for a live agent-to-agent substrate (e.g. OpenClaw / Hermes).

Fill in the methods marked TODO from your factory's real traffic. Nothing here
imports a specific framework: the contract is the four capability interfaces. A
factory provides whichever capabilities its instrumentation can serve; declare
them in `capabilities()` and `run` executes only those tracks.

Mapping a real factory onto the capabilities
--------------------------------------------
SteppableFactory  one "round" is one scoring cycle of a producer cohort. Each
                  round populate the behaviour dict with the keys the tracks read
                  (see SteppableFactory's docstring for the full schema): at least
                  mean_pos (a scalar summary of cohort output), and for the
                  governance/pathology tracks norm_sat / metric_sat / variety.
                  Emit one Decision per sampled producer carrying the decision-time
                  propensity it disclosed (token logprob of the chosen action,
                  softmax weight, or epsilon-greedy probability), attach scores as
                  Reward, and record the control input (price, injected shift).
                  set_price forwards the governance price; inject applies a small
                  spec perturbation for system identification; injectable_channels
                  declares which perturbations the factory accepts.

LearnerGame       a committed leader move against a follower agent configured as a
                  mean-based learner or a no-swap-regret learner; extracted value
                  is the leader's per-round payoff. Estimating V and U* off a
                  non-enumerable game is the open part -- see the package README.

DividendTask      one task with a hidden verifier (e.g. a SWE-bench Verified
                  problem, an MBPP+ item). free_floor runs unconstrained search
                  metered in verifier calls; legible_floor runs a bounded readable
                  class (sparse rule list, depth-capped program) under the same
                  meter; interaction_order estimates effective order (a
                  Shapley-interaction index or a degree-bounded surrogate sweep).
                  The opacity track sweeps a LIST of such tasks (pass task_specs)
                  and grades the dividend against each task's measured order.

CoupledLoops      two or more factory loops sharing a dependency (a model
                  checkpoint, an upstream service). order_parameter is the phase
                  concentration of their slowest-loop cycles. condensate is
                  optional: implement it only if you can stage a one-shot halt and
                  a structural decouple; otherwise the entrainment track uses the
                  onset and diversity signals alone.

catastrophe       supply two behavioural epochs to the track directly rather than a
                  regime: ramp = a real pre-incident series ending at fold_index,
                  null = a prior healthy series. See tracks/catastrophe.measure.
"""
from __future__ import annotations

from ..interfaces import (Substrate, SteppableFactory, LearnerGame, DividendTask,
                          CoupledLoops)
from ..instrumentation import TraceStore
from ..records import RoundObs


class LiveSteppableFactory(SteppableFactory):
    def __init__(self, client):
        self.client = client            # your factory handle
        self.trace = TraceStore()
        self._round = 0

    def reset(self, **config) -> "LiveSteppableFactory":
        self.trace = TraceStore()
        self._round = 0
        # TODO: configure the producer cohort / task family on self.client
        return self

    def behaviour(self) -> dict:
        # TODO: return the current behaviour dict. Populate the keys the tracks you
        # run consume (see SteppableFactory docstring): mean_pos and, for
        # governance/pathology, norm_sat (price-responsive), metric_sat, variety.
        raise NotImplementedError

    def set_price(self, lam: float) -> None:
        raise NotImplementedError("forward the governance price to the factory")

    def injectable_channels(self) -> set:
        return set()                    # e.g. {"spec_target"} once inject() is wired

    def step(self) -> RoundObs:
        # TODO: advance one scoring cycle; per sampled producer build a Decision
        # (handle, round=self._round, action, n_actions, propensity, ...) and call
        # self.trace.record_decision(...); attach scores via record_reward(...);
        # call record_behaviour(self._round, {...}) and record_control(self._round,
        # {...}). Return a RoundObs.
        raise NotImplementedError


class LiveLearnerGame(LearnerGame):
    def __init__(self, client):
        self.client = client

    def run(self, T, disclosure=0.0, follower="mean_based", n_actions=3, seed=0):
        raise NotImplementedError("play the committed leader move; return running extracted value")

    def stackelberg_value(self, n_actions=3):
        raise NotImplementedError("estimate V (see README on the non-enumerable case)")

    def steerable_value(self, n_actions=3):
        raise NotImplementedError("estimate U*")


class LiveDividendTask(DividendTask):
    def __init__(self, client, **spec):
        self.client = client
        self.spec = spec

    def free_floor(self, budget, seed=0):
        raise NotImplementedError("unconstrained search metered in verifier calls")

    def legible_floor(self, budget, order=1, seed=0):
        raise NotImplementedError("bounded legible class under the same verifier-call meter")

    def interaction_order(self, seed=0):
        raise NotImplementedError("estimate effective interaction order")


class LiveCoupledLoops(CoupledLoops):
    def __init__(self, client, **config):
        self.client = client

    def order_parameter(self, coupling, diversity, seed=0):
        raise NotImplementedError("phase concentration of the coupled loops' cycles")

    def critical_coupling(self, diversity):
        raise NotImplementedError("estimate or supply Kc")

    # condensate() is optional: implement only if interventions can be staged.


class LiveSubstrate(Substrate):
    name = "live"

    def __init__(self, client, capabilities=("steppable",)):
        self.client = client
        self._caps = set(capabilities)

    def capabilities(self) -> set:
        return set(self._caps)

    def steppable(self, **config) -> SteppableFactory:
        return LiveSteppableFactory(self.client).reset(**config)

    def learner_game(self, **config) -> LearnerGame:
        return LiveLearnerGame(self.client)

    def dividend_task(self, **config) -> DividendTask:
        return LiveDividendTask(self.client, **config)

    def coupled_loops(self, **config) -> CoupledLoops:
        return LiveCoupledLoops(self.client, **config)
