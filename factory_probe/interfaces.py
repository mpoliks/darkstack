"""Capability interfaces a substrate implements.

A substrate is anything that can stand a factory up and let the measurements
drive it. Different measurements need different capabilities, so the substrate
exposes them as separate objects rather than one wide interface:

  SteppableFactory  a population that advances one round at a time, exposes a
                    behavioural observable, accepts a governance price, and accepts
                    typed intent perturbations for system identification
                    (versioning, pathology, catastrophe, governance-cascade)
  LearnerGame       a committed-leader game played against a follower learner of a
                    named class, with a tunable propensity-disclosure level
                    (Stackelberg gap)
  DividendTask      a task whose searchers are metered in verifier calls; a free
                    searcher and a bounded legible reader compete under a shared
                    call budget (opacity floor)
  CoupledLoops      a set of factory loops coupled through a shared dependency,
                    reporting a synchronisation order parameter (entrainment)

`Substrate` bundles the four as factory methods and declares which it provides via
`capabilities()`. A mock implements all of them; a live adapter implements the
subset it can serve.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .records import RoundObs
from .instrumentation import TraceStore


class SteppableFactory(ABC):
    """A population that advances one round at a time and is read by behaviour.

    Behaviour-key contract. `behaviour()` and each round's `RoundObs.behaviour`
    are dicts; the tracks read these scalar keys, so a live adapter must populate
    the ones the tracks it wants to run consume:

      mean_pos    scalar behavioural coordinate, ~[0,1] (a low-dimensional summary
                  of the cohort's output)            -- versioning, catastrophe, pathology
      norm_sat    norm satisfaction in [0,1]; must respond to the governance price
                  (swing when set_price changes)      -- governance, pathology
      metric_sat  scored-metric satisfaction in [0,1] -- pathology (metric/norm gap)
      variety     effective number of distinct behaviours, >=1 -- pathology

    Regime/condition contract. `reset(regime=...)` asks the factory to enter a
    dynamical condition a track needs. A track is only meaningful if the adapter
    can actually drive that condition:

      versioning  a regime that telegraphs between metastable states vs one basin
      governance  a regime whose norm_sat is price-responsive (for T_inner)
      pathology   the five named regimes (healthy/stable_failure/overfitting/
                  learning_death/thrash)
      catastrophe a ramp toward a fold plus a prior healthy epoch -- or supply the
                  series directly to the catastrophe track (preferred for live use)
    """

    trace: TraceStore

    @abstractmethod
    def reset(self, **config) -> "SteppableFactory":
        ...

    @abstractmethod
    def step(self) -> RoundObs:
        """Advance one round; record behaviour, control input, and decisions to
        `self.trace`."""

    @abstractmethod
    def behaviour(self) -> dict:
        """Current scalar behavioural observables."""

    def set_price(self, lam: float) -> None:
        """Set the governance price lambda for subsequent rounds (no-op by default)."""

    def injectable_channels(self) -> set:
        """Names of the intent channels `inject()` accepts. Empty if the factory
        supports no perturbation (system identification then unavailable)."""
        return set()

    def inject(self, **channels) -> None:
        """Apply named scalar intent shifts (deltas) for system identification:
        nudge a channel, then measure how long the behaviour takes to settle. Each
        keyword must be in `injectable_channels()`; an unknown channel raises
        KeyError. Channels are deltas added to the current intent, not absolutes.
        """
        bad = set(channels) - set(self.injectable_channels())
        if bad:
            raise KeyError(f"unsupported inject channels {bad}; "
                           f"available: {sorted(self.injectable_channels())}")
        raise NotImplementedError("inject() declared channels but did not implement them")

    def run(self, n: int) -> TraceStore:
        for _ in range(n):
            self.step()
        return self.trace


class LearnerGame(ABC):
    """A committed-leader game played against a follower learner."""

    @abstractmethod
    def run(self, T: int, disclosure: float = 0.0, follower: str = "mean_based",
            n_actions: int = 3, seed: int = 0) -> np.ndarray:
        """Run T rounds; return the running per-round extracted value (length T).
        follower is 'mean_based' (steerable) or 'no_swap' (retentive); disclosure
        in [0,1] blends a mean-based follower toward its swap-corrected form."""

    @abstractmethod
    def stackelberg_value(self, n_actions: int = 3) -> float:
        """The one-shot committed value V (best against a best-responder)."""

    @abstractmethod
    def steerable_value(self, n_actions: int = 3) -> float:
        """The mean-based-extractable optimum U*."""


class DividendTask(ABC):
    """A task whose free and legible searchers are metered in verifier calls.

    The task owns its searchers (a legible class is domain-specific -- parity
    surrogates on a cube, sparse rule lists or depth-capped programs on code), so
    the budget loop lives here, not in the track. One unit of budget is one
    verifier call. Floors are reported as achieved verified value (higher is
    better) in comparable units, so the dividend (free - legible) is meaningful
    across tasks.
    """

    @abstractmethod
    def free_floor(self, budget: int, seed: int = 0) -> float:
        """Best verified value a free (full-interaction) searcher reaches at the
        given verifier-call budget."""

    @abstractmethod
    def legible_floor(self, budget: int, order: int = 1, seed: int = 0) -> float:
        """Best verified value a bounded legible reader of the given interaction
        order reaches at the same budget."""

    @abstractmethod
    def interaction_order(self, seed: int = 0) -> int:
        """Estimated forced-opacity order K*: the smallest reader order whose
        argmax reaches the task optimum."""

    def dividend(self, budget: int, order: int = 1, seed: int = 0) -> float:
        """Verified value the legible reader forgoes vs free search: free - legible."""
        return self.free_floor(budget, seed) - self.legible_floor(budget, order, seed)


class CoupledLoops(ABC):
    """Factory loops coupled through a shared dependency."""

    @abstractmethod
    def order_parameter(self, coupling: float, diversity: float, seed: int = 0) -> float:
        """Synchronisation order parameter r in [0, 1] at the given coupling and
        dependency diversity."""

    @abstractmethod
    def critical_coupling(self, diversity: float) -> float:
        """The coupling threshold Kc above which the loops synchronise."""

    def condensate(self, coupling: float, diversity: float, intervention: str = "none",
                   seed: int = 0):
        """Order-parameter time series r(t) of a locked population under an optional
        midpoint intervention: 'none', 'halt' (one-shot desynchronisation), or
        'decouple' (drop the coupling below Kc). A live adapter returns the measured
        r(t) of its coupled loops under the corresponding action. Optional: a
        substrate that cannot stage interventions leaves this unimplemented and the
        entrainment track reports those sub-signals as unavailable."""
        raise NotImplementedError(f"{type(self).__name__} does not provide condensate()")


class Substrate(ABC):
    """A factory back-end the measurements run against."""

    name: str = "substrate"

    def capabilities(self) -> set:
        """Names of the capabilities this substrate provides, a subset of
        {'steppable','learner_game','dividend_task','coupled_loops'}. A harness
        selects runnable tracks from this set."""
        return set()

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities()

    def default_dividend_specs(self):
        """Task-spec list the opacity track sweeps by default on this substrate
        (each a kwargs dict for dividend_task), or None to use the track's own
        default. Lets a substrate carry the task family appropriate to it."""
        return None

    def steppable(self, **config) -> SteppableFactory:
        raise NotImplementedError(f"{self.name} does not provide a SteppableFactory")

    def learner_game(self, **config) -> LearnerGame:
        raise NotImplementedError(f"{self.name} does not provide a LearnerGame")

    def dividend_task(self, **config) -> DividendTask:
        raise NotImplementedError(f"{self.name} does not provide a DividendTask")

    def coupled_loops(self, **config) -> CoupledLoops:
        raise NotImplementedError(f"{self.name} does not provide CoupledLoops")
