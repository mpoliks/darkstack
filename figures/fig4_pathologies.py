"""
FIGURE 4 -- Pathologies & Catastrophe.  A plot-only VIEW of the unified DarkFactory:
data from DarkFactory.data_fig4().
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from darkfactory import DarkFactory
style.apply()
P = style.PALETTE


def main():
    D = DarkFactory().data_fig4()
    rows, names, norm_cols = D["rows"], D["names"], D["norm_cols"]
    mp, jump, var, ar1 = D["mp"], D["jump"], D["var"], D["ar1"]
    vt, at, vn, an = D["vt"], D["at"], D["vn"], D["an"]
    rates, norm_s, met_s, nyq = D["rates"], D["norm_s"], D["met_s"], D["nyq"]

    fig = plt.figure(figsize=(12.2, 3.9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.12, wspace=0.34)
    axA = fig.add_subplot(gs[:, 0])
    axB1 = fig.add_subplot(gs[0, 1])
    axB2 = fig.add_subplot(gs[1, 1])
    axC = fig.add_subplot(gs[:, 2])

    # Panel A -- fingerprint heatmap
    cols = ["volatility", "variety", "norm\nunmet", "metric–\nnorm gap"]
    axA.imshow(norm_cols, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
    axA.set_xticks(range(4)); axA.set_xticklabels(cols, fontsize=7.5)
    axA.set_yticks(range(len(names))); axA.set_yticklabels(names, fontsize=8)
    raw = [["%.3f" % rows[i, 0], "%.1f" % rows[i, 1], "%.2f" % rows[i, 2], "%.2f" % rows[i, 3]]
           for i in range(len(names))]
    for i in range(len(names)):
        for j in range(4):
            axA.text(j, i, raw[i][j], ha="center", va="center", fontsize=6.6,
                     color="white" if norm_cols[i, j] > 0.55 else style.INK)
    axA.set_title("Each pathology has a distinct\nbehavioural fingerprint")
    axA.grid(False)
    style.panel_tag(axA, "A")

    # Panel B -- trace + critical slowing down
    axB1.plot(mp, color=P["neutral"], lw=0.8)
    axB1.axvline(jump, color=P["warn"], lw=1.0, ls="--")
    axB1.set_ylabel("behaviour", fontsize=8)
    axB1.set_xticklabels([])
    axB1.set_title("Critical slowing down before a fold", fontsize=10)
    style.panel_tag(axB1, "B")
    t = np.arange(jump)
    axB2.plot(t, var / np.nanmax(var), color=P["core"],
              label=f"variance  ($\\tau$ {vt:+.2f}, null {vn:+.2f})")
    axB2.plot(t, ar1, color=P["frontier"],
              label=f"lag-1 autocorr.  ($\\tau$ {at:+.2f}, null {an:+.2f})")
    axB2.axvline(jump, color=P["warn"], lw=1.0, ls="--", label="fold (version collapse)")
    axB2.set_xlim(axB1.get_xlim())
    axB2.set_xlabel("round  $t$  (spec height ramped $1.6\\!\\to\\!0.4$)")
    axB2.set_ylabel("EWS", fontsize=8)
    style.legend_below(axB2, ncol=1, y=-0.55, fontsize=6.4)

    # Panel C -- aliasing
    axC.fill_betweenx([0, 1.02], min(rates) * 0.8, nyq, color=style.FAINT, alpha=0.20,
                      label="aliased region (below Nyquist)")
    axC.plot(rates, met_s, "o-", color=P["accent"], ms=4, label="metric satisfied")
    axC.plot(rates, norm_s, "o-", color=P["core"], ms=4, label="norm satisfied")
    axC.axvline(nyq, color=P["neutral"], lw=1.0, ls=":", label="Nyquist rate (reference)")
    axC.set_xscale("log")
    axC.set_xlabel("metric sampling rate  $f_s$  (1/period)")
    axC.set_ylabel("satisfaction")
    axC.set_title("Overfitting as aliasing: the metric/norm\ngap opens as sampling slows")
    style.legend_below(axC, ncol=1)
    axC.set_ylim(0, 1.02)
    style.panel_tag(axC, "C")

    fig.suptitle("Figure 4.  Pathologies and catastrophe:  distinct behavioural fingerprints, early-warning signals, and overfitting as aliasing",
                 fontsize=11, y=1.04)
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "fig4_pathologies.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    json.dump(dict(fingerprints={nm: list(map(float, r)) for nm, r in zip(names, rows)},
                   ews_var_tau=vt, ews_ar1_tau=at, fold_t=jump,
                   ews_null_var_tau=vn, ews_null_ar1_tau=an,
                   alias_rates=list(rates), alias_norm=norm_s, alias_metric=met_s, nyquist=nyq),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig4.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
