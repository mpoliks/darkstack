"""
FIGURE 1 -- The Opacity-Cost Frontier.  A plot-only VIEW of the unified DarkFactory:
the data comes from DarkFactory.data_fig1(); see src/darkfactory.py for the model.
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
    D = DarkFactory().data_fig1()
    ds, fracs, premB = D["ds"], D["fracs"], D["premB"]
    dC, ks, taxC, fracC = D["dC"], D["ks"], D["taxC"], D["fracC"]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    axA = axes[0]
    for (name, frac), col in zip(fracs.items(), [P["frontier"], P["core"], P["accent"]]):
        axA.semilogy(ds, frac, "o-", color=col, ms=4, label=name)
    axA.set_xlabel("configuration dimension  $d$")
    axA.set_ylabel(r"legible fraction  $\binom{d}{\leq k}/2^{d}$")
    axA.set_title("Legible assemblies are\nexponentially rare")
    style.legend_below(axA, ncol=3)
    axA.set_ylim(1e-7, 2)
    style.panel_tag(axA, "A")

    axB = axes[1]
    for (lab, (m, e)), col in zip(premB.items(), [P["neutral"], P["frontier"], P["green"]]):
        axB.errorbar(ds, m, yerr=e, fmt="o-", color=col, ms=4, capsize=2, label=lab)
    axB.axhline(0, color=style.FAINT, lw=0.8)
    axB.set_xlabel("configuration dimension  $d$")
    axB.set_ylabel("cost premium  $(f^* - f^*_{\\mathrm{legible}})/\\sigma_f$")
    axB.set_title("Legibility incurs a\npriceable premium")
    style.legend_below(axB, ncol=1)
    style.panel_tag(axB, "B")

    axC = axes[2]
    axC.plot(ks, taxC, "o-", color=P["core"], ms=4, label="cost premium")
    axC.set_xlabel(f"legibility budget  $k$   (at $d={dC}$)")
    axC.set_ylabel("cost premium  $/\\sigma_f$", color=P["core"])
    axC.tick_params(axis="y", labelcolor=P["core"])
    axC.set_title("Readability is priceable at\nan increasing marginal rate")
    axC2 = axC.twinx()
    axC2.semilogy(ks, fracC, "s--", color=P["frontier"], ms=3.2, label="legible fraction")
    axC2.set_ylabel(r"legible fraction", color=P["frontier"])
    axC2.tick_params(axis="y", labelcolor=P["frontier"])
    axC2.grid(False)
    axC2.spines["top"].set_visible(False)
    style.panel_tag(axC, "C")

    fig.suptitle("Figure 1.  The Opacity-Cost Frontier:  legible assemblies are exponentially rare and pay a measurable premium",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "fig1_opacity.png")
    fig.savefig(out)
    print("wrote", out)
    rec = dict(ds=ds, legible_fraction=fracs,
               premium_complex=list(premB["complex objective"][0]),
               premium_linear=list(premB["linear control"][0]),
               premium_planted=list(premB["planted-sparse control"][0]),
               taxC_k=ks, taxC_premium=taxC, taxC_fraction=fracC, dC=dC)
    json.dump(rec, open(os.path.join(os.path.dirname(__file__), "..", "out", "fig1.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
