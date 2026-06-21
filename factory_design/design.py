"""The design language: a `FactoryDesign` and its translation to dynamics.

A `FactoryDesign` is the set of decisions an architect commits to before the
factory is switched on -- the choices that are expensive to change once a
population of agents is learning against them. The fields are named for the thing
a harness engineer actually sets (how much the population explores, how often the
judge re-checks the goal, how good the judge is, how often governance reprices),
not for the underlying dynamical parameter. `to_params()` performs that
translation onto `FactoryParams`, the parameter object the population organ reads.

Every knob carries a one-line gloss and a `maps_to` -- the real harness thing it
stands in for -- so a reader who has never seen a replicator equation can still
set it and read the result. The glosses live in `KNOBS` and are surfaced by the
report and the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import pi

from factory import FactoryParams  # from src/, placed on path by the package __init__


# --- knob documentation -----------------------------------------------------
# name -> (one-line gloss, what it stands in for in a real harness)
KNOBS = {
    "explore_rate": (
        "fraction of the population trying something other than the current best each round",
        "sampling temperature / share of runs on exploratory (non-greedy) policies",
    ),
    "explorer_floor": (
        "a protected minimum exploration share the kernel guarantees no matter what",
        "reserved compute for exploratory agents that scoring is not allowed to starve",
    ),
    "population_size": (
        "how many agents are in the population (smaller = noisier, can jump between behaviours)",
        "number of parallel agents / rollouts you run per round",
    ),
    "selection_strength": (
        "how hard the population piles onto the current best (higher = greedier)",
        "how aggressively you rank and keep winners (selection temperature)",
    ),
    "n_options": (
        "size of the space of distinct assemblies the population chooses among",
        "size of the action / configuration space an agent picks from",
    ),
    "norm_target": (
        "where on the design axis the goal actually sits (the behaviour the spec wants)",
        "the true objective your spec encodes, as a coordinate the factory can hit or miss",
    ),
    "goal_payoff": (
        "how rewarding the goal-satisfying region is on its own (low = the factory is not "
        "naturally pulled toward the goal and can settle into a stable failure)",
        "how much the goal pays off without governance pushing for it",
    ),
    "eval_period": (
        "rounds between re-evaluations of the goal (lower = the judge checks more often)",
        "how often your judge re-scores against ground truth (tasks between eval refreshes)",
    ),
    "judge_fidelity": (
        "how faithfully the scored metric tracks the true goal (1 = perfect, 0 = easily gamed)",
        "judge-model quality / how gameable your eval rubric is",
    ),
    "goal_drift": (
        "how far the true goal moves over time (0 = static; large = a fast-moving target)",
        "spec churn / market drift -- how fast the thing you are scoring against changes",
    ),
    "drift_period": (
        "rounds it takes the drifting goal to complete one cycle",
        "the timescale over which your objective changes",
    ),
    "controller": (
        "the governance loop that prices constraint violation: off / gentle / aggressive",
        "your automated repricing or penalty controller (none, conservative, or hard)",
    ),
    "repricing_period": (
        "rounds between governance repricings (too small relative to settling time = thrash)",
        "how often governance reprices / your eval-to-policy feedback cadence",
    ),
    "target_satisfaction": (
        "the goal-satisfaction level governance steers toward",
        "the SLA / acceptance threshold your controller drives to",
    ),
    "cost_pressure": (
        "how hard a priced constraint bites (a stand-in for budget pressure; 0 = unpriced)",
        "token/$ budget pressure on the priced constraint (a $0 budget is the kill switch)",
    ),
    "peer_factories": (
        "how many other factories share infrastructure with this one (0 = standalone)",
        "other teams/factories on the same base model, control plane, or vendor",
    ),
    "shared_dependency": (
        "how tightly peers are coupled through that shared infrastructure",
        "coupling through a common foundation model / control plane / upstream service",
    ),
    "dependency_diversity": (
        "how varied the peers' dependencies are (higher = less likely to move in lockstep)",
        "vendor / model / schedule diversity across the peer group",
    ),
}


@dataclass(frozen=True)
class FactoryDesign:
    """A factory you are about to build, described in plain terms.

    The defaults describe a healthy factory. Change one or two knobs to express a
    design you are worried about (a tiny `explore_rate`, a slow `eval_period`, a
    low `judge_fidelity`, a fast `repricing_period`) and simulate it.
    """

    # -- the population (who is learning, and how) --------------------------
    explore_rate: float = 0.06
    explorer_floor: float = 0.0
    population_size: int = 300
    selection_strength: float = 4.0
    n_options: int = 64

    # -- the goal and how it is scored --------------------------------------
    norm_target: float = 0.70
    goal_payoff: float = 1.40
    eval_period: int = 1
    judge_fidelity: float = 1.0
    goal_drift: float = 0.0
    drift_period: int = 120

    # -- governance (how constraint violation is priced) --------------------
    controller: str = "off"          # "off" | "gentle" | "aggressive"
    repricing_period: int = 200
    target_satisfaction: float = 0.80
    cost_pressure: float = 0.0

    # -- ecology (other factories on shared infrastructure) -----------------
    peer_factories: int = 0
    shared_dependency: float = 0.0
    dependency_diversity: float = 0.30

    # ---- validation -------------------------------------------------------
    def __post_init__(self):
        if not 0.0 <= self.judge_fidelity <= 1.0:
            raise ValueError("judge_fidelity must be in [0, 1]")
        if self.controller not in ("off", "gentle", "aggressive"):
            raise ValueError("controller must be 'off', 'gentle', or 'aggressive'")
        if not 0.0 <= self.explore_rate <= 1.0 or not 0.0 <= self.explorer_floor <= 1.0:
            raise ValueError("explore_rate and explorer_floor are fractions in [0, 1]")
        if not 0.0 <= self.target_satisfaction <= 1.0:
            raise ValueError("target_satisfaction must be in [0, 1]")
        if self.selection_strength < 0 or self.cost_pressure < 0:
            raise ValueError("selection_strength and cost_pressure must be non-negative")
        if self.eval_period < 1 or self.drift_period < 1 or self.repricing_period < 1:
            raise ValueError("periods must be >= 1 round")
        if self.n_options < 4:
            raise ValueError("n_options must be >= 4")
        if self.population_size < 8:
            raise ValueError("population_size must be >= 8")

    # ---- derived quantities ----------------------------------------------
    @property
    def effective_explore(self) -> float:
        """The exploration the population actually runs at: the kernel floor
        protects a minimum no matter how small `explore_rate` is set."""
        return max(self.explore_rate, self.explorer_floor)

    def pid_gains(self) -> dict:
        """PID gains for the chosen controller preset. 'off' returns None."""
        if self.controller == "off":
            return None
        if self.controller == "gentle":
            return dict(Kp=4.0, Ki=0.2, Kd=1.5, hi=20.0 + 20.0 * self.cost_pressure)
        return dict(Kp=8.0, Ki=0.5, Kd=2.0, hi=30.0 + 30.0 * self.cost_pressure)

    # ---- translation to dynamics -----------------------------------------
    def to_params(self, seed: int = 0) -> FactoryParams:
        """Translate the design onto `FactoryParams` for the population organ.

        The landscape is fixed: two ways to satisfy intent sit at 0.30 and 0.70 on
        the design axis. `norm_target` says which one the spec wants;
        `goal_payoff` says how attractive the goal region is on its own.
        Everything else maps a plain knob onto the mechanism that drives the
        corresponding behaviour.
        """
        p = dict(
            K=int(self.n_options),
            M=int(self.population_size),
            eta=float(self.selection_strength),
            mu=float(self.effective_explore),
            peakA=0.30, peakB=0.70, width=0.07, heightA=1.0,
            c=float(self.goal_payoff),       # height of the goal (peak B) region
            norm_target=float(self.norm_target),
            cost_scale=1.0,                        # the constraint surface is intrinsic;
            cost_center=0.30, cost_width=0.10,     # it bites near A, governance prices it via lambda
            lam=0.0,                               # (cost_pressure sets how hard governance may push)
            seed=int(seed),
        )
        # How the goal is scored, and where overfitting can come from.
        if self.goal_drift > 0.0:
            # a moving target: the metric samples it every `eval_period` rounds;
            # too slow a sample aliases away from the norm (overfitting by aliasing).
            p["norm_amp"] = float(self.goal_drift)
            p["norm_freq"] = 2.0 * pi / float(self.drift_period)
            p["metric_sample_period"] = int(self.eval_period)
            if self.judge_fidelity < 1.0:
                p["align_weight"] = (1.0 - self.judge_fidelity) * 4.0
                p["align_width"] = 0.05
        elif self.judge_fidelity < 1.0:
            # a static but gameable judge: the metric blesses a spot away from the
            # true norm and the population is rewarded for chasing it.
            p["metric_static"] = _proxy_target(self.norm_target)
            p["align_weight"] = (1.0 - self.judge_fidelity) * 4.0
            p["align_width"] = 0.05
        else:
            # a faithful static judge: the metric is the norm.
            p["metric_static"] = float(self.norm_target)
        return FactoryParams(**p)

    # ---- ergonomics -------------------------------------------------------
    def with_(self, **changes) -> "FactoryDesign":
        """Return a copy with some knobs changed (designs are immutable)."""
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"unknown knob(s): {sorted(unknown)}; known: {sorted(KNOBS)}")
        return replace(self, **changes)

    def diff(self, other: "FactoryDesign") -> dict:
        """Knobs where `other` differs from `self`: {knob: (self, other)}."""
        return {f.name: (getattr(self, f.name), getattr(other, f.name))
                for f in fields(self)
                if getattr(self, f.name) != getattr(other, f.name)}


def _proxy_target(norm_target: float) -> float:
    """Where a gameable judge places its blessing: offset from the true norm,
    toward the other well, so chasing the metric pulls away from the goal."""
    return 0.30 if norm_target >= 0.5 else 0.70
