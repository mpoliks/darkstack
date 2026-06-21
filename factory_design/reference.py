"""Reference result: why a factory needs room to explore.

This is the one piece of the engine that is a property of a *game*, not of a
*design*, so it is kept out of the per-design verdict and shown on its own.

The committed value V is the most an architect can secure by committing to a
strategy a fully retentive (no-swap-regret) population will best-respond to. The
result this module demonstrates is that a population that keeps some mean-based
exploration can be steered *above* V, toward a larger value U*, while a purely
retentive one cannot, on the canonical Deng-Schneider-Sivan (2019) game:

  - V is computed exactly (the Stackelberg value).
  - A simple two-phase steering schedule is run against a mean-based population and
    against a no-swap-regret population; the mean-based one is driven well above V,
    the no-swap one is held at or below it.
  - With only two options the gap collapses (the theorem: U* = V when there are
    fewer than three actions).

Nothing here is a measurement of a real factory. The published optimum for this
game is U* = 1/2 (Deng-Schneider-Sivan 2019); the schedule below reaches a value
near it, which is a lower bound on what an optimal optimizer extracts.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from darkfactory import stackelberg_value, play   # from src/, exact game organs

PUBLISHED_U_STAR = 0.5   # Deng-Schneider-Sivan 2019, canonical game, n>=3


@dataclass
class SteeringResult:
    n_actions: int
    committed_value_V: float
    meanbased_reached: float       # value a steering schedule reaches vs a mean-based population
    noswap_reached: float          # value the same schedule reaches vs a no-swap population
    gap: float                     # meanbased_reached - committed_value_V

    def __str__(self) -> str:
        return (f"n={self.n_actions}: V={self.committed_value_V:+.2f}  "
                f"mean-based reaches {self.meanbased_reached:+.2f}  "
                f"no-swap reaches {self.noswap_reached:+.2f}  "
                f"(surplus above V: {self.gap:+.2f})")


def steering(n_actions: int = 3, seeds: int = 6, T: int = 40000) -> SteeringResult:
    """Run the steering demonstration for an `n_actions` game."""
    V = float(stackelberg_value(n_actions))
    mean_based = float(np.mean([play(T, 0.0, n_actions, seed=s)[-1] for s in range(seeds)]))
    no_swap = float(np.mean([play(T, 1.0, n_actions, seed=s)[-1] for s in range(seeds)]))
    return SteeringResult(n_actions=n_actions, committed_value_V=V,
                          meanbased_reached=mean_based, noswap_reached=no_swap,
                          gap=mean_based - V)


def collapse_check(seeds: int = 6) -> dict:
    """Show the surplus at n=3 and its collapse at n=2 (U* = V below three actions)."""
    return {3: steering(3, seeds), 2: steering(2, seeds)}


def summary(seeds: int = 6) -> str:
    r3 = steering(3, seeds)
    r2 = steering(2, seeds)
    return (
        "Steering reference (canonical Deng-Schneider-Sivan game; not your factory)\n"
        f"  {r3}\n"
        f"  {r2}\n"
        f"  published optimum at n>=3: U* = {PUBLISHED_U_STAR:+.2f} (the schedule above is a lower bound)\n"
        "  reading: a population that keeps mean-based exploration can be steered above the\n"
        "  committed value V; a purely retentive one cannot (V is the ceiling on what any\n"
        "  schedule extracts from it), and with two options the gap is gone.\n"
        "  in your factory: protect some exploratory (non-greedy) capacity, or you can only ever\n"
        "  reach what you committed at design time."
    )


if __name__ == "__main__":
    print(summary())
