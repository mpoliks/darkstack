"""
FIGURE 3 -- Versioning by the Transfer Operator.  A plot-only VIEW of the unified
DarkFactory: data from DarkFactory.data_fig3().
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
    D = DarkFactory().data_fig3()
    mp, pt_label, taus, tsep = D["mp"], D["pt_label"], D["taus"], D["tsep"]
    mus, seps, vals = D["mus"], D["seps"], D["vals"]

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))

    # Panel A
    axA = axes[0]
    w0, w1 = 0, 9000
    t = np.arange(w0, w1); seg = mp[w0:w1]; lab = pt_label[w0:w1]
    axA.scatter(t[lab == 0], seg[lab == 0], s=2, color=P["frontier"], label="version A")
    axA.scatter(t[lab == 1], seg[lab == 1], s=2, color=P["core"], label="version B")
    axA.axhline(0.4, color=style.FAINT, lw=0.7, ls=":", label="basin cores (0.4 / 0.6)")
    axA.axhline(0.6, color=style.FAINT, lw=0.7, ls=":")
    axA.set_xlabel("round  $t$")
    axA.set_ylabel("behavioural coordinate  (mean assembly position)")
    axA.set_title("Versions are emergent metastable\nregions of behaviour")
    style.legend_below(axA, ncol=3)
    for h in axA.get_legend().legend_handles[:2]:
        h.set_sizes([18])
    axA.set_ylim(0.2, 0.8)
    style.panel_tag(axA, "A")

    # Panel B
    axB = axes[1]
    k = 10
    idx = np.arange(1, k + 1)
    axB.vlines(idx, 1, taus[:k], color=style.MUTED, lw=1.0)
    axB.semilogy(idx[1], taus[1], "o", color=P["core"], ms=8, zorder=4,
                 label="inter-version mode  ($\\tau_2$)")
    axB.semilogy(idx[2:], taus[2:k], "o", color=P["frontier"], ms=4, zorder=4,
                 label="intra-version ladder")
    axB.semilogy(idx[0], taus[0], "o", color=style.FAINT, ms=5, zorder=3)
    axB.set_xlabel("mode index  $i$")
    axB.set_ylabel("relaxation timescale  $\\tau_i=-1/\\ln\\lambda_i$  (rounds)")
    axB.set_title(f"One slow inter-version mode over a\nfast relaxation ladder  ($\\tau_2/\\tau_3\\approx{tsep:.0f}$)")
    style.legend_below(axB, ncol=2)
    style.panel_tag(axB, "B")

    # Panel C
    axC = axes[2]
    axC.plot(mus, seps, "o-", color=P["core"], ms=4, label="timescale separation $\\tau_2/\\tau_3$")
    axC.axhline(1.0, color=style.FAINT, lw=0.8, ls="--", label="$\\tau_2/\\tau_3=1$ (versions merge)")
    axC.set_xlabel("exploration rate  $\\mu$  (frontier strength)")
    axC.set_ylabel("timescale separation  $\\tau_2/\\tau_3$")
    axC.set_title("Exploration erodes version\nrobustness")
    style.legend_below(axC, ncol=1)
    style.panel_tag(axC, "C")

    fig.suptitle("Figure 3.  Versioning by what the factory does:  a version is the slow near-invariant mode of the behavioural transfer operator",
                 fontsize=11, y=1.05)
    fig.tight_layout()
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "fig3_versioning.png")
    fig.savefig(out)
    print("wrote", out)
    json.dump(dict(top_eigs=list(np.round(vals[:10], 4)), taus=list(np.round(taus[:10], 1)),
                   tau2=float(taus[1]), tau3=float(taus[2]), timescale_sep=float(tsep),
                   n_meta_090=D["n_meta_090"], n_meta_095=D["n_meta_095"],
                   rhoA=D["rhoA"], rhoB=D["rhoB"], debounced_switches=int(D["nsw"]),
                   dwell_cv=D["dwell_cv"], mus=mus, sep_vs_mu=seps),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig3.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
