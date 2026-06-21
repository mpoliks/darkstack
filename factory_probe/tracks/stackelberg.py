"""Stackelberg track: the committed-value gap and what closes it.

A committed leader move is run against a mean-based follower and a no-swap-regret
follower. The mean-based frontier should be steerable above the committed value V
toward U*; the no-swap core should stay at or below V. Disclosing follower
propensity should collapse the surplus, and the surplus should require more than
two follower actions.
"""
from __future__ import annotations

import numpy as np

from ..report import Finding


def measure(substrate, T: int = 12000, disclosure=(0.0, 0.25, 0.5, 0.75, 1.0),
            seed: int = 0, **kw) -> Finding:
    g = substrate.learner_game()
    V = g.stackelberg_value(3)
    U = g.steerable_value(3)
    frontier = float(g.run(T, follower="mean_based", disclosure=0.0, n_actions=3, seed=seed)[-1])
    core = float(g.run(T, follower="no_swap", n_actions=3, seed=seed)[-1])
    sweep = [float(g.run(T, follower="mean_based", disclosure=d, n_actions=3, seed=seed)[-1])
             for d in disclosure]
    n3 = float(g.run(T, follower="mean_based", disclosure=0.0, n_actions=3, seed=seed)[-1])
    n2 = float(g.run(T, follower="mean_based", disclosure=0.0, n_actions=2, seed=seed)[-1])

    steered = frontier > V + 0.1
    core_capped = core <= V + 1e-6
    disclosure_collapses = sweep[-1] < sweep[0] - 0.1
    needs_three = (n3 - n2) > 0.1
    confirms = bool(steered and core_capped and disclosure_collapses and needs_three)
    return Finding(
        track="stackelberg", property="committed-value gap V vs U*",
        capability="learner_game",
        measured=dict(V=round(V, 3), U_star=round(U, 3),
                      frontier_extracted=round(frontier, 3), core_extracted=round(core, 3),
                      disclosure_sweep=[round(s, 3) for s in sweep],
                      extracted_n3=round(n3, 3), extracted_n2=round(n2, 3)),
        confirms=confirms,
        summary=f"frontier {frontier:+.2f} > V={V:.2f}, core {core:+.2f} <= V; "
                f"disclosure {sweep[0]:+.2f}->{sweep[-1]:+.2f}; N=3 {n3:+.2f} vs N=2 {n2:+.2f}",
        confirm_criterion="mean-based frontier extracted > V, no-swap core <= V, disclosure "
                          "collapses the surplus, and the surplus needs >= 3 actions",
        falsify_criterion="frontier not steerable above V, or core extractable above V, or "
                          "disclosure leaves the surplus intact",
    )
