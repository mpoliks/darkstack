"""
FIGURE 2 -- The Stackelberg Gap.   Tests Claim [b] (the Stackelberg move).

The paper: an architect's single committed move secures the Stackelberg value V
against best-responders; a population with a mean-based no-regret FRONTIER can be
steered to U* > V ("everything an architect can obtain beyond the option space
they encoded"); a no-swap-regret CORE is capped at V; and "a factory built
entirely out of no-swap-regret loops cannot find anything outside V ... a dark
stack needs at least some naive, mean-based no-regret learning somewhere."

We use the EXACT running game of Deng-Schneider-Sivan (2019), independently
re-verified three ways (V=0, U*=1/2 on the [-1,1] scale):

    Optimizer = ROW {Top, Bottom};  Learner = COLUMN {Left, Mid, Right}
    U_O = [[ 0,-2,-2],[ 0,-2, 2]] / 2      (optimizer payoff)
    U_L = [[ s,-1, 0],[-1, 1, 0]] / 2,  s = sqrt(gamma) ~ 1e-3 tie-break

Committed leader schedule: Top for the first half, then Bottom. The s-bonus on
(Top,Left) keeps Left the running argmax while the learner's cumulative Mid
reward is silently driven down; switching to Bottom makes Right the argmax and
the optimizer harvests +2 each round.

Panel A: extracted value vs round -- mean-based Hedge -> U*, Blum-Mansour
         no-swap-regret -> V.
Panel B: DISCLOSURE knob. The paper's reward channel can disclose a "propensity
         value" letting a learner reconstruct the road not taken and run
         no-swap-regret correction. We blend the played distribution from
         (1-delta)*Hedge + delta*BlumMansour; as disclosure delta: 0 -> 1 the
         exploitable surplus U*-V collapses to V. (delta, not lambda -- lambda is
         the Lagrangian price elsewhere in the paper.) (This is the paper's own logic
         chain -- disclosure enables swap-correction caps at V; the link to
         Kamenica-Gentzkow persuasion "transparency" is an interpretive bridge,
         flagged as such, not a theorem.)
Panel C: the falsifiable boundary. U* > V requires the learner to have >= 3
         actions (DSS Thm: with 2 actions no-regret == no-swap-regret, U*=V). We
         show the surplus vanishes under the 2-action restriction. Not hidden.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from learners import Hedge, BlumMansourSwap
style.apply()
P = style.PALETTE

S = 1e-3
U_O = np.array([[0., -2., -2.], [0., -2., 2.]]) / 2.0     # optimizer (row) payoff
U_L = np.array([[S, -1., 0.], [-1., 1., 0.]]) / 2.0       # learner (col) payoff


def leader_schedule(T):
    sched = np.zeros(T, dtype=int)
    sched[T // 2:] = 1                                     # Top half / Bottom half
    return sched


def stackelberg_value(n_actions=3):
    """V = max over leader commitments p=Prob(Bottom) of optimizer payoff at the
    learner's PURE best response (dense grid)."""
    ps = np.linspace(0, 1, 2_000_001)
    alpha = np.stack([1 - ps, ps], axis=1)
    learner_payoff = alpha @ U_L[:, :n_actions]
    br = np.argmax(learner_payoff, axis=1)
    opt = alpha[:, 0] * U_O[0, :n_actions][br] + alpha[:, 1] * U_O[1, :n_actions][br]
    return float(opt.max())


def play(T, disclosure=0.0, n_actions=3, seed=0):
    """Committed leader vs a learner that blends a mean-based core (Hedge) with a
    no-swap-regret correction (Blum-Mansour) in proportion `disclosure`.
    disclosure=0 -> pure mean-based; disclosure=1 -> pure no-swap-regret.
    Returns running-average optimizer payoff."""
    rng = np.random.default_rng(seed)
    UL = U_L[:, :n_actions]
    UO = U_O[:, :n_actions]
    hedge = Hedge(n_actions, eta=0.7, rng=rng)
    swap = BlumMansourSwap(n_actions, eta=0.7, rng=rng)
    sched = leader_schedule(T)
    gain = np.empty(T)
    for t in range(T):
        row = sched[t]
        ph = hedge.distribution()
        ps = swap.distribution()
        p = (1 - disclosure) * ph + disclosure * ps
        p = p / p.sum()
        col = int(rng.choice(n_actions, p=p))
        gain[t] = UO[row, col]
        reward = UL[row, :].copy()                        # full reward vector
        hedge.update(reward)
        swap.update(reward)
    return np.cumsum(gain) / (np.arange(T) + 1)


def main():
    T = 60_000
    reps = 6
    V = stackelberg_value(3)
    Ustar = 0.5

    # Panel A -----------------------------------------------------------------
    hedge_runs = np.array([play(T, 0.0, 3, seed=s) for s in range(reps)])
    swap_runs = np.array([play(T, 1.0, 3, seed=s) for s in range(reps)])
    hedge_avg, swap_avg = hedge_runs.mean(0), swap_runs.mean(0)

    # Panel B -----------------------------------------------------------------
    lambdas = np.linspace(0, 1, 11)
    transp = np.array([[np.mean([play(20_000, lam, 3, seed=s)[-1] for s in range(reps)]),
                        np.std([play(20_000, lam, 3, seed=s)[-1] for s in range(reps)]) / np.sqrt(reps)]
                       for lam in lambdas])

    # Panel C -----------------------------------------------------------------
    n3 = np.array([play(T, 0.0, 3, seed=s)[-1] for s in range(reps)])
    n2 = np.array([play(T, 0.0, 2, seed=s)[-1] for s in range(reps)])
    V3, V2 = stackelberg_value(3), stackelberg_value(2)

    # ----------------------------- plot -------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))

    axA = axes[0]
    axA.axhline(Ustar, color=P["frontier"], ls=":", lw=1.0, label="$U^\\ast=0.5$ (frontier ceiling)")
    axA.axhline(V, color=P["core"], ls=":", lw=1.0, label="$V=0$ (committed value)")
    axA.plot(hedge_avg, color=P["frontier"], label="mean-based frontier (Hedge)")
    axA.plot(swap_avg, color=P["core"], label="no-swap-regret core (Blum-Mansour)")
    axA.set_xlabel("round  $t$")
    axA.set_ylabel("optimizer's extracted value (running avg)")
    axA.set_title("Mean-based learners can be steered\nbeyond the committed value")
    axA.set_ylim(-0.12, 0.6)
    h, l = axA.get_legend_handles_labels()
    order = [2, 3, 0, 1]
    style.legend_below(axA, ncol=2, handles=[h[i] for i in order], labels=[l[i] for i in order])
    style.panel_tag(axA, "A")

    axB = axes[1]
    axB.axhline(Ustar, color=P["frontier"], ls=":", lw=1.0, label="$U^\\ast$")
    axB.axhline(V, color=P["core"], ls=":", lw=1.0, label="$V$")
    axB.errorbar(lambdas, transp[:, 0], yerr=transp[:, 1], fmt="o-",
                 color=P["neutral"], ms=4, capsize=2, label="extracted value")
    axB.set_xlabel("disclosure  (propensity revealed $\\to$ swap-correction)")
    axB.set_ylabel("extracted value at horizon")
    axB.set_title("Disclosure of alternatives reduces\nthe surplus beyond committed value")
    axB.set_ylim(-0.12, 0.6)
    style.legend_below(axB, ncol=3)
    style.panel_tag(axB, "B")

    axC = axes[2]
    x = [0, 1]
    means = [n3.mean(), n2.mean()]
    errs = [n3.std() / np.sqrt(reps), n2.std() / np.sqrt(reps)]
    axC.bar(x, means, width=0.5, yerr=errs, capsize=3,
            color=[P["frontier"], P["core"]], alpha=0.85, label="extracted value at horizon")
    axC.axhline(0, color=P["neutral"], lw=0.9, ls="--", label="Stackelberg $V=0$")
    axC.set_xticks(x)
    axC.set_xticklabels(["learner has\n3 actions", "learner has\n2 actions"])
    axC.set_ylabel("extracted value at horizon")
    axC.set_title("With binary decisions, mean-based\nlearners collapse into swap-based")
    axC.set_ylim(-0.40, 0.6)
    style.legend_below(axC, ncol=1)
    style.panel_tag(axC, "C")

    fig.suptitle("Figure 2.  The Stackelberg Gap:  the committed value $V$ is a floor for steerable frontiers and a ceiling for retentive cores",
                 fontsize=11, y=1.05)
    fig.tight_layout()
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "fig2_stackelberg.png")
    fig.savefig(out)
    print("wrote", out)
    print(f"V (grid) = {V:.4f}   U* (analytic) = {Ustar}")
    print(f"Hedge final = {hedge_avg[-1]:.3f}   Swap final = {swap_avg[-1]:.3f}")
    print(f"disclosure sweep endpoints: lam0={transp[0,0]:.3f}  lam1={transp[-1,0]:.3f}")
    print(f"N=3 final = {n3.mean():.3f} (V3={V3:.2f})   N=2 final = {n2.mean():.3f} (V2={V2:.2f})")
    json.dump(dict(V=V, Ustar=Ustar, hedge_final=float(hedge_avg[-1]),
                   swap_final=float(swap_avg[-1]), lambdas=list(lambdas),
                   transp_mean=list(transp[:, 0]), n3_final=float(n3.mean()),
                   n2_final=float(n2.mean()), V3=V3, V2=V2),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig2.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
