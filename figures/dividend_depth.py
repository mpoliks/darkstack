"""
THE DIVIDEND, DEEPER -- the opacity dividend as a law, grounded in the NK model.

  A  The forced floor rises with epistasis. On Kauffman NK landscapes (interaction order
     K+1), the order-1 reader's dividend climbs from 0 (separable, K=0) and keeps rising:
     the more tangled the task, the more efficiency a fixed-order auditor must forgo.
  B  Why: the value migrates up the interaction orders. The Walsh power spectrum sits
     entirely at order 1 when K=0 and spreads to high order as K grows -- the legible
     band empties out.
  C  Fitting is not acting. Adding interaction order to the reader's model raises the
     variance it captures (R^2) monotonically, yet the regret of its argmax (D_K) is
     non-monotone: more model makes the decision worse in 12% of (landscape, order)
     pairs. A lower-order reader sometimes outperforms a higher-order one.
  D  Measurable from queries. The floor estimated from B verifier calls (fit a legible
     model from samples, act, measure realized regret) converges to the oracle floor --
     the precondition for carrying the law onto a real, non-enumerable task.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from dividend import dividend, nk_landscape, planted_order, walsh_coeffs, popcounts, budget_curves
style.apply()
P = style.PALETTE


# ---- A: the forced floor rises with epistasis (NK) --------------------------
def panel_floor_vs_epistasis(ax):
    d, seeds = 12, 14
    Knks = [0, 1, 2, 3, 4]
    means, errs, kstars = [], [], []
    for Knk in Knks:
        d1, ks = [], []
        for s in range(seeds):
            f, pc = nk_landscape(d, Knk, np.random.default_rng(100 * Knk + s))
            D, _, Kstar = dividend(f, pc, Kmax=min(Knk + 2, 6))
            d1.append(D[1]); ks.append(Kstar if Kstar else np.nan)
        means.append(np.mean(d1)); errs.append(np.std(d1) / np.sqrt(seeds))
        kstars.append(np.nanmedian(ks))
    ax.errorbar(Knks, means, yerr=errs, fmt="o-", color=P["core"], ms=5, capsize=2)
    ax.set_xlabel("NK interaction  $K$  (order $K\\!+\\!1$)")
    ax.set_ylabel("order-1 forced floor  $D_1$  ($\\sigma_f$)")
    ax.set_title("The forced floor rises with\nepistasis ($K^\\ast \\leq K\\!+\\!1$)")
    ax.set_xticks(Knks)
    style.panel_tag(ax, "A")
    return dict(K=Knks, D1=means, Kstar=kstars)


# ---- B: the value migrates up the interaction orders ------------------------
def panel_spectrum(ax):
    d = 12
    orders = list(range(1, 7))
    cols = [P["green"], P["frontier"], P["accent"], P["core"]]
    rec = {}
    for Knk, col in zip([0, 1, 2, 4], cols):
        spec = np.zeros(len(orders))
        for s in range(8):
            f, pc = nk_landscape(d, Knk, np.random.default_rng(700 * Knk + s))
            c2 = walsh_coeffs(f) ** 2
            var = c2[pc >= 1].sum()
            for j, o in enumerate(orders):
                spec[j] += c2[pc == o].sum() / var
        spec /= 8
        rec[f"K={Knk}"] = spec.tolist()
        ax.plot(orders, spec, "o-", color=col, ms=4, label=f"$K={Knk}$")
    ax.set_xlabel("interaction order of the value")
    ax.set_ylabel("share of variance")
    ax.set_title("The value migrates up the orders\n(the legible band empties)")
    ax.set_xticks(orders)
    style.legend_below(ax, ncol=4)
    style.panel_tag(ax, "B")
    return rec


# ---- C: fitting only weakly predicts acting ---------------------------------
def panel_fit_vs_act(ax):
    d = 12
    pts, worse, tot = [], 0, 0
    for r in [2, 3, 4]:
        for s in range(40):
            f, pc = planted_order(d, r, np.random.default_rng(20000 + 100 * r + s))
            D, R2, _ = dividend(f, pc, Kmax=r + 1)
            for K in range(1, r):
                pts.append((R2[K], D[K])); tot += 1; worse += int(D[K + 1] > D[K] + 1e-9)
    for Knk in [1, 2, 3]:
        for s in range(20):
            f, pc = nk_landscape(d, Knk, np.random.default_rng(30000 + 100 * Knk + s))
            D, R2, _ = dividend(f, pc, Kmax=Knk + 1)
            for K in range(1, Knk + 1):
                pts.append((R2[K], D[K]))
    pts = np.array(pts); corr = float(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1])
    hi = (pts[:, 0] > 0.55) & (pts[:, 1] > 0.15)        # fits well, still acts short
    ax.scatter(pts[~hi, 0], pts[~hi, 1], s=9, color=P["frontier"], alpha=0.4, edgecolor="none")
    ax.scatter(pts[hi, 0], pts[hi, 1], s=16, color=P["core"], label="fits well ($R^2\\!>\\!0.55$), acts short")
    ax.set_xlabel("variance captured  $R^2_K$  (fitting)")
    ax.set_ylabel("regret  $D_K$  (acting, $\\sigma_f$)")
    ax.set_title(f"Acting is not fitting: $R^2$ leaves\n{100*(1-corr**2):.0f}% of the regret unexplained")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "C")
    return dict(corr=corr, n=len(pts), frac_worse=worse / tot,
                frac_fitwell_actshort=float(hi.mean()))


# ---- D: the floor is measurable from queries (the bridge) -------------------
def panel_query_bridge(ax):
    d = 12
    f, pc = nk_landscape(d, 2, np.random.default_rng(11))     # interacting task
    oracle = float(dividend(f, pc, Kmax=1)[0][1])
    budgets = [32, 64, 128, 256, 512, 1024, 2048]
    cur = budget_curves(f, d, [1], budgets, np.random.default_rng(5), reps=48)
    ax.plot(budgets, cur["free"], "o-", color=P["accent"], ms=4, label="free search")
    ax.plot(budgets, cur[1], "s-", color=P["core"], ms=4, label="legible reader (from queries)")
    ax.axhline(oracle, color=P["core"], lw=0.8, ls=":", label="oracle floor  $\\Phi^\\ast$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("verifier queries  $B$")
    ax.set_ylabel("gap to optimum  ($\\sigma_f$)")
    ax.set_title("The floor is measurable from\nqueries (the bridge to real tasks)")
    ax.set_ylim(-0.05, None)
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "D")
    return dict(oracle=oracle, budgets=budgets, free=cur["free"].tolist(), legible=cur[1].tolist())


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.4))
    rec = {}
    rec["floor_vs_epistasis"] = panel_floor_vs_epistasis(axes[0, 0])
    rec["spectrum"] = panel_spectrum(axes[0, 1])
    rec["fit_vs_act"] = panel_fit_vs_act(axes[1, 0])
    rec["query_bridge"] = panel_query_bridge(axes[1, 1])
    fig.suptitle("Figure P.  The dividend as a law:  the forced floor rises with epistasis, the value migrates up the interaction orders, and fitting is not acting",
                 fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "figP_dividend.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    json.dump(rec, open(os.path.join(os.path.dirname(__file__), "..", "out", "figP.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
