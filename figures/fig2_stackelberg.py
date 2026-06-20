"""
FIGURE 2 -- The Stackelberg Gap.  A plot-only VIEW of the unified DarkFactory:
data from DarkFactory.data_fig2() (the exact Deng-Schneider-Sivan game).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from darkfactory import DarkFactory
style.apply()
P = style.PALETTE


def main():
    D = DarkFactory().data_fig2()
    V, Ustar = D["V"], D["Ustar"]
    hedge_avg, swap_avg = D["hedge_avg"], D["swap_avg"]
    lambdas, transp = D["lambdas"], D["transp"]
    n3, n2, V3, V2, reps = D["n3"], D["n2"], D["V3"], D["V2"], D["reps"]

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
    json.dump(dict(V=V, Ustar=Ustar, hedge_final=float(hedge_avg[-1]),
                   swap_final=float(swap_avg[-1]), lambdas=list(lambdas),
                   transp_mean=list(transp[:, 0]), n3_final=float(n3.mean()),
                   n2_final=float(n2.mean()), V3=V3, V2=V2),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig2.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
