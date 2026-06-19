"""
FIGURE 3 -- Versioning by the Transfer Operator.

The paper: don't version a dark factory by what it IS (unknowable) but by what it
DOES -- a version is a near-invariant region of its input-output distribution.
The transfer operator (Perron-Frobenius) measures versions; the separation of
timescales grades robustness.

We read ONLY the factory's behavioural observable (the population's mean assembly
position; we never inspect agent weights). The stochastic-replicator population
telegraphs between two metastable assembly peaks = two versions. We estimate the
Perron-Frobenius operator by Ulam's method, reversibilise it (the clean
eigenvalue<->metastable-set correspondence is a theorem for reversible operators;
the estimated operator is already near-reversible here, max |Im lambda| = 0.014),
and read its spectrum.

Panel A: the behavioural trace, coloured by the near-invariant set each point
         belongs to. Version switches are counted with hysteresis (enter a basin
         core at 0.4/0.6) to avoid boundary chatter; the debounced dwell times are
         approximately exponential (CV ~ 0.9), the signature of noise-induced
         (Kramers-like) barrier crossing -- the versions and their transitions are
         emergent, not imposed.
Panel B: the operator's relaxation timescales tau_i = -1/ln(lambda_i). ONE slow
         inter-version mode (tau_2 ~ 270 rounds) stands an order of magnitude above
         a ladder of fast intra-version relaxation modes (tau_3 ~ 16). This
         timescale separation -- not a threshold-fragile eigenvalue count -- is
         what makes two versions well-defined. The sub-dominant eigenvector (inset)
         splits the design axis into the two basins.
Panel C: robustness vs exploration mu. More exploration shortens the slow mode and
         compresses the timescale separation toward the relaxation ladder --
         versions bleed together (the paper's "small gap => transitional version").
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from factory import DarkFactory, FactoryParams
from transfer_operator import (ulam_operator, spectrum, almost_invariant_sets,
                               coherence_timescale, n_metastable, reversibilize,
                               coherence_ratio)
style.apply()
P = style.PALETTE

BASE = dict(peakA=0.35, peakB=0.65, width=0.09, eta=2.0, M=60, c=1.0)


def debounced_switches(mp, lo=0.4, hi=0.6):
    """Count version switches only when the trace enters a basin CORE (below lo or
    above hi), debouncing the chatter at the 0.5 boundary. Returns (count, dwells)."""
    state, count, dwells, last = None, 0, [], 0
    for i, x in enumerate(mp):
        ns = state
        if x < lo:
            ns = 0
        elif x > hi:
            ns = 1
        if ns is not None and ns != state:
            if state is not None:
                count += 1; dwells.append(i - last); last = i
            state = ns
    return count, np.array(dwells)


def timescales(P_rev):
    vals, vecs = spectrum(P_rev)
    taus = np.array([coherence_timescale(v) for v in vals])
    return vals, vecs, taus


def main():
    f = DarkFactory(FactoryParams(mu=0.03, seed=2, **BASE))
    o = f.run(60_000)
    mp = o["mean_pos"]
    n_boxes = 30
    Pop, edges, occ = ulam_operator(mp, n_boxes=n_boxes)
    P_rev, pi = reversibilize(Pop)                 # read spectrum from reversibilised op
    vals, vecs, taus = timescales(P_rev)
    labels, _ = almost_invariant_sets(P_rev, m=2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rhoA = coherence_ratio(Pop, labels == 0, pi)
    rhoB = coherence_ratio(Pop, labels == 1, pi)
    tsep = taus[1] / taus[2]
    nsw, dwells = debounced_switches(mp)
    dwell_cv = float(dwells.std() / dwells.mean()) if len(dwells) > 2 else float("nan")

    box_of = np.clip(np.digitize(mp, edges[1:-1]), 0, n_boxes - 1)
    pt_label = labels[box_of]

    # Panel C data: robustness vs exploration mu
    mus = [0.015, 0.03, 0.05, 0.08, 0.12, 0.18]
    seps = []
    for mu in mus:
        ff = DarkFactory(FactoryParams(mu=mu, seed=4, **BASE))
        Pq, _, _ = ulam_operator(ff.run(60_000)["mean_pos"], n_boxes=n_boxes)
        Pqr, _ = reversibilize(Pq)
        vq, _, tq = timescales(Pqr)
        seps.append(tq[1] / tq[2])

    # ----------------------------- plot -------------------------------------
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

    # Panel B -- timescales (log), one slow mode over a relaxation ladder
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

    # Panel C -- robustness vs mu
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
    print(f"reversibilised top eigs: {np.round(vals[:5],4)}")
    print(f"tau2={taus[1]:.0f} tau3={taus[2]:.0f} separation={tsep:.1f}")
    print(f"n_metastable: >0.90={n_metastable(P_rev,0.90)} >0.95={n_metastable(P_rev,0.95)} (threshold-sensitive)")
    print(f"coherence ratios A={rhoA:.3f} B={rhoB:.3f}")
    print(f"debounced version switches={nsw}  dwell CV={dwell_cv:.2f} (≈1 ⇒ Kramers-like)")
    json.dump(dict(top_eigs=list(np.round(vals[:10], 4)), taus=list(np.round(taus[:10], 1)),
                   tau2=float(taus[1]), tau3=float(taus[2]), timescale_sep=float(tsep),
                   n_meta_090=int(n_metastable(P_rev, 0.90)), n_meta_095=int(n_metastable(P_rev, 0.95)),
                   rhoA=rhoA, rhoB=rhoB, debounced_switches=int(nsw), dwell_cv=dwell_cv,
                   mus=mus, sep_vs_mu=seps),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig3.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
