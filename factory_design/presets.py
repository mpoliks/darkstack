"""Named designs -- the war stories a harness engineer recognises.

Each preset is a factory you might actually build, set up to land in one failure
mode (or none). They are the entry point: simulate one, read the verdict, apply
the fix, and you have learned the knob by perturbing a case you recognise. They
are also the falsification set -- each must classify as its own failure mode (see
tests/test_pathology.py).
"""
from __future__ import annotations

from .design import FactoryDesign

PRESETS: dict[str, FactoryDesign] = {
    # A factory that converges on the goal and keeps exploring.
    "healthy": FactoryDesign(),

    # The judge is gameable: it blesses a spot that is not the true goal, so the
    # factory scores the metric while missing what the spec actually wants.
    "gameable_judge": FactoryDesign(
        norm_target=0.70, goal_payoff=2.5, judge_fidelity=0.2, explore_rate=0.10),

    # No protected exploration: scoring starves the explorers, the population locks
    # onto one assembly and stops searching.
    "no_exploration_floor": FactoryDesign(
        explore_rate=0.0005, explorer_floor=0.0),

    # The goal sits in a region that pays off poorly on its own, and with no
    # controller pricing the failure the factory settles there and stays.
    "stuck_failing": FactoryDesign(
        goal_payoff=0.45, norm_target=0.70, controller="off"),

    # Governance reprices far faster than the factory settles, so the control loop
    # forces the oscillation it is trying to damp.
    "repricing_too_fast": FactoryDesign(
        goal_payoff=0.85, norm_target=0.70, controller="aggressive",
        repricing_period=3, cost_pressure=0.5),

    # A frontier-heavy population in a small pool with weak retention never commits.
    "never_settles": FactoryDesign(
        explore_rate=0.30, population_size=40, selection_strength=2.0),

    # Healthy on its own, but many peers share one base model, so they can crash
    # together when that shared dependency moves.
    "monoculture": FactoryDesign(
        peer_factories=400, shared_dependency=2.2, dependency_diversity=0.30),
}

# The population-level verdict each preset is built to show.
EXPECTED: dict[str, str] = {
    "healthy": "healthy",
    "gameable_judge": "overfitting",
    "no_exploration_floor": "learning_death",
    "stuck_failing": "stable_failure",
    # ungoverned, this factory sits in stable failure -- that is why it needs a
    # controller; the point of the preset is that the controller is timed wrong
    # (the governance lens reports iatrogenic_thrash on top of the verdict).
    "repricing_too_fast": "stable_failure",
    "never_settles": "thrash",
    # the population is fine; the fault is in the ecology lens (correlated_crash).
    "monoculture": "healthy",
}

# The presets that exercise the pathology classifier directly: each maps to a
# distinct failure mode (or healthy). The round-trip over these is the
# classifier's falsification gate (tests/test_pathology.py).
PATHOLOGY_PRESETS = ("healthy", "gameable_judge", "no_exploration_floor",
                     "stuck_failing", "never_settles")


def get(name: str) -> FactoryDesign:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]
