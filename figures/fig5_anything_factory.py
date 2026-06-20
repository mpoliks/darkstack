"""
FIGURE 5 -- Governance timing & the Anything Factory.  A plot-only VIEW of the
unified DarkFactory: data from DarkFactory.data_fig5().
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
    D = DarkFactory().data_fig5()
    ratios, cas_mean, cas_sem, T_inner = D["ratios"], D["cas_mean"], D["cas_sem"], D["T_inner"]
    Ks, kseries = D["Ks"], D["kseries"]
    cond_t, r_lock, r_halt, r_dec = D["cond_t"], D["r_lock"], D["r_halt"], D["r_dec"]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    # Panel A -- cascade ratio
    axA = axes[0]
    axA.axvspan(3, 10, color=P["green"], alpha=0.12, label="cascade band (3:1–10:1)")
    axA.errorbar(ratios, cas_mean, yerr=cas_sem, fmt="o-", color=P["neutral"], ms=4, capsize=2,
                 label="governance instability (8 seeds)")
    axA.set_xscale("log")
    axA.set_xlabel("cascade ratio  $T_\\mathrm{gov}/T_\\mathrm{inner}$   ($T_\\mathrm{inner}$" +
                   f"$\\approx${T_inner:.0f})")
    axA.set_ylabel("governance instability  (oscillation of $\\lambda$)")
    axA.set_title("Mistimed governance thrashes;\nslower governance settles")
    style.legend_below(axA, ncol=1)
    style.panel_tag(axA, "A")

    # Panel B -- Kuramoto
    axB = axes[1]
    for (lab, Kc, rs), col in zip(kseries, [P["core"], P["frontier"]]):
        axB.plot(Ks, rs, "o-", color=col, ms=3.5, label=f"{lab} ($K_c$={Kc:.1f})")
        axB.axvline(Kc, color=col, ls=":", lw=1.0)
    axB.set_xlabel("shared-dependency coupling  $K$")
    axB.set_ylabel("synchronisation  $r$  (order parameter)")
    axB.set_title("Diverse dependencies make\nentrainment harder")
    style.legend_below(axB, ncol=1)
    axB.set_ylim(0, 1)
    style.panel_tag(axB, "B")

    # Panel C -- condensate
    axC = axes[2]
    axC.plot(cond_t, r_lock, color=P["core"], label="entrained condensate")
    axC.plot(cond_t, r_halt, color=P["accent"], lw=1.3, label="one-shot halt (re-locks)")
    axC.plot(cond_t, r_dec, color=P["frontier"], label="diversify dependencies (breaks)")
    axC.axvline(cond_t[len(cond_t) // 2], color=style.MUTED, lw=0.9, ls="--", label="intervention")
    axC.set_xlabel("time")
    axC.set_ylabel("systemic synchronisation  $r$")
    axC.set_title("Governance is required to break\na phase-locked condensate")
    style.legend_below(axC, ncol=1)
    axC.set_ylim(0, 1)
    style.panel_tag(axC, "C")

    fig.suptitle("Figure 5.  Governance timing and the anything factory:  cascade ratios, entrainment thresholds, and the phase-locked condensate",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "fig5_anything_factory.png")
    fig.savefig(out)
    print("wrote", out)
    json.dump(dict(cascade_ratios=ratios, cascade_instab_mean=cas_mean, cascade_instab_sem=cas_sem,
                   cascade_T_inner=T_inner, kuramoto_K=Ks,
                   condensate_locked=float(r_lock[-500:].mean()),
                   condensate_halt=float(r_halt[-500:].mean()),
                   condensate_decouple=float(r_dec[-500:].mean())),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig5.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
