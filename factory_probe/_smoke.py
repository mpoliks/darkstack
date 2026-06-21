"""End-to-end smoke check: every capability of the reference substrate runs,
emits the records the measurements consume, and reproduces its planted ground
truth. Assertions fail loudly. Run: python -m factory_probe._smoke
"""
from __future__ import annotations

import numpy as np

from factory_probe.mock import MockSubstrate, planted_evaluator


def main():
    sub = MockSubstrate()
    print(f"substrate: {sub.name}   capabilities: {sorted(sub.capabilities())}\n")
    assert sub.capabilities() == {"steppable", "learner_game", "dividend_task", "coupled_loops"}

    # 1. Steppable population: behaviour series + decisions w/ propensity + control input
    fac = sub.steppable(regime="versions", seed=1)
    fac.run(2000)
    s = fac.trace.summary()
    mp = fac.trace.behaviour_series("mean_pos")
    price = fac.trace.control_series("price")
    spans = bool(mp.min() < 0.45 and mp.max() > 0.55)
    print("steppable population (regime=versions):")
    print(f"  {s['n_decisions']} decisions, {s['n_rewards']} rewards, {s['n_rounds']} rounds")
    print(f"  behaviour keys {s['behaviour_keys']}  control keys {s['control_keys']}")
    print(f"  mean_pos in [{mp.min():.2f}, {mp.max():.2f}] spans both peaks: {spans}")
    assert spans, "versions regime should visit both peaks"
    assert "price" in s["control_keys"] and len(price) == 2000, "control input not recorded"
    prop, _ = fac.trace.propensity_reward()
    assert len(prop) == 16000
    print(f"  propensity-reward pairs: {len(prop)} (mean propensity {prop.mean():.3f})\n")

    # 1b. inject() contract: declared channels apply, unknown channels raise
    fac2 = sub.steppable(regime="governance", seed=0)
    assert fac2.injectable_channels() == {"spec_target", "spec_height"}
    before = fac2._f.p.norm_target
    fac2.inject(spec_target=0.05)
    assert abs(fac2._f.p.norm_target - (before + 0.05)) < 1e-9, "inject(spec_target) did not apply"
    try:
        fac2.inject(nonsense=1.0); raised = False
    except KeyError:
        raised = True
    assert raised, "inject() must reject unknown channels"
    print("inject contract: spec_target applied, unknown channel rejected\n")

    # 1c. delayed realised-consequence channel: recover the planted lag
    facd = sub.steppable(regime="versions", seed=2, n_sample=2, realized_delay=5)
    facd.run(300)
    lags = facd.trace.lags("realized")
    print("delayed realised channel (planted delay=5):")
    print(f"  realised rewards: {len(lags)}  median lag: {np.median(lags):.0f}  "
          f"reward kinds: {facd.trace.summary()['reward_kinds']}")
    assert len(lags) > 0 and np.median(lags) == 5, "realised lag not recovered"
    assert "realized" in facd.trace.summary()["reward_kinds"], "no realised rewards emitted\n"
    print()

    # 2. Committed-leader game: extracted value + U* planted per action count
    game = sub.learner_game()
    V = game.stackelberg_value(3); U = game.steerable_value(3)
    frontier = game.run(8000, follower="mean_based", disclosure=0.0)[-1]
    core = game.run(8000, follower="no_swap")[-1]
    print("committed-leader game:")
    print(f"  V={V:.3f}  U*={U:.3f}  mean-based extracted={frontier:+.3f}  "
          f"no-swap extracted={core:+.3f}")
    assert abs(U - 0.5) < 1e-9 and abs(game.steerable_value(2) - game.stackelberg_value(2)) < 1e-9
    try:
        game.steerable_value(4); raised = False
    except NotImplementedError:
        raised = True
    assert raised, "U* must not return a stale constant for unsupported action counts"
    assert frontier > V + 0.1 and core <= V + 1e-6, "frontier should beat V, core should not"
    print()

    # 3. Coupled loops: order parameter below/above threshold
    loops = sub.coupled_loops(N=300, T=40)
    Kc = loops.critical_coupling(0.5)
    r_lo = loops.order_parameter(coupling=0.3, diversity=0.5)
    r_hi = loops.order_parameter(coupling=2.5, diversity=0.5)
    print("coupled loops:")
    print(f"  Kc(0.5)={Kc:.2f}  r(K=0.3)={r_lo:.3f}  r(K=2.5)={r_hi:.3f}")
    assert r_lo < 0.2 < r_hi, "order parameter should switch across Kc"
    print()

    # 4. Dividend task: free vs legible floor, recovered order
    task = sub.dividend_task(d=10, order=2, kind="nk", seed=0)
    free = task.free_floor(budget=512); leg = task.legible_floor(budget=512, order=1)
    print("dividend task (planted NK, Walsh order 2):")
    print(f"  free floor={free:.2f}  legible(order1) floor={leg:.2f}  dividend={free - leg:.2f}")
    print(f"  acting order K*={task.interaction_order()}  walsh order={task.walsh_order()}")
    assert free - leg > 0.2, "interacting task should carry a positive dividend"

    # 5. Out-of-loop scorer recovers planted precision/recall
    sc = planted_evaluator(sensitivity=0.8, specificity=0.9, base_rate=0.3, n=6000, seed=0)
    print("\nout-of-loop scorer (planted sensitivity 0.80, specificity 0.90):")
    print(f"  recall={sc.recall():.3f}  precision={sc.precision():.3f}")
    assert abs(sc.recall() - 0.8) < 0.05, "scorer did not recover planted recall"

    print("\nsmoke OK (all planted ground truth recovered)")


if __name__ == "__main__":
    main()
