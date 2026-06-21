"""
THE OPACITY DIVIDEND -- opacity priced in interaction-order space.

The old premium (Fig 1) asked whether the best design is SIMPLE (few parts on); it
reports a positive cost even for a separable objective whose optimum a degree-1 reader
recovers exactly. The dividend asks instead whether a reader limited to K-way
interactions can ACT near-optimally, and it reads zero precisely when opacity is not
forced. Four results:

  A  The dividend staircase D_K drops to zero exactly at the landscape's true
     interaction order r:  K* = r.
  B  That law is stable across dimension d (the staircase does not move as the cube
     grows).
  C  Opacity separated from performance: a free searcher's gap to the optimum vanishes
     with budget (search for performance); a legible reader plateaus at the
     budget-invariant floor Phi* (efficiency available only illegibly). Phi* = 0 for a
     separable task, > 0 for an interacting one.
  D  The dividend kills the part-count false positive: the linear control's old premium
     of ~0.9 becomes a true zero.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from dividend import dividend, planted_order, popcounts, budget_curves
from darkfactory import bit_matrix, make_objective, premium
style.apply()
P = style.PALETTE


# ---- A: the dividend staircase, K* = r --------------------------------------
def panel_staircase(ax):
    d, seeds = 12, 16
    rec = {}
    for r, col in [(1, P["green"]), (2, P["frontier"]), (3, P["core"])]:
        curves = []
        for s in range(seeds):
            f, pc = planted_order(d, r, np.random.default_rng(1000 * r + s))
            D, _, _ = dividend(f, pc, Kmax=4)
            curves.append([D[K] for K in range(1, 5)])      # K=0 (constant model) is degenerate
        m = np.mean(curves, axis=0)
        rec[f"r={r}"] = m.tolist()
        ax.plot(range(1, 5), m, "o-", color=col, ms=4, label=f"planted order $r={r}$")
        ax.axvline(r, color=col, lw=0.7, ls=":")
    ax.axhline(0, color=style.FAINT, lw=0.8)
    ax.set_xlabel("reader interaction order  $K$")
    ax.set_ylabel("opacity dividend  $D_K$  ($\\sigma_f$)")
    ax.set_title("The dividend staircase:\nfirst zero at $K^\\ast = r$")
    ax.set_xticks(range(1, 5))
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "A")
    return rec


# ---- B: K* = r is stable across dimension -----------------------------------
def panel_dimension(ax):
    ds = [8, 10, 12, 14]; rs = [1, 2, 3]; seeds = 20
    rec = {}
    for di, (d, mk) in enumerate(zip(ds, ["o", "s", "^", "D"])):
        xs, ys = [], []
        for r in rs:
            ks = []
            for s in range(seeds):
                f, pc = planted_order(d, r, np.random.default_rng(50 * d + 7 * r + s))
                _, _, Kstar = dividend(f, pc, Kmax=r + 1)
                if Kstar is not None:
                    ks.append(Kstar)
            frac = float(np.mean([k == r for k in ks]))
            rec[f"d{d}_r{r}"] = dict(median_Kstar=float(np.median(ks)), frac_eq_r=frac)
            xs.append(r + (di - 1.5) * 0.06); ys.append(np.median(ks))
        ax.plot(xs, ys, mk, color=P["frontier"], ms=6, mfc="none", label=f"$d={d}$")
    lim = [0.6, 3.4]
    ax.plot(lim, lim, "-", color=style.FAINT, lw=1.0, zorder=0, label="$K^\\ast = r$")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks(rs); ax.set_yticks(rs)
    ax.set_xlabel("true interaction order  $r$")
    ax.set_ylabel("measured forced order  $K^\\ast$")
    ax.set_title("The law is dimension-stable\n($K^\\ast=r$ across $d=8\\!-\\!14$)")
    style.legend_below(ax, ncol=3)
    style.panel_tag(ax, "B")
    return rec


# ---- C: opacity separated from performance (the budget floor Phi*) ----------
def panel_budget(ax):
    d = 12
    budgets = [16, 32, 64, 128, 256, 512, 1024, 2048]
    f2, _ = planted_order(d, 2, np.random.default_rng(11))     # interacting task
    f1, _ = planted_order(d, 1, np.random.default_rng(13))     # separable task
    c2 = budget_curves(f2, d, [1], budgets, np.random.default_rng(5), reps=48)
    c1 = budget_curves(f1, d, [1], budgets, np.random.default_rng(6), reps=48)
    phi = float(dividend(f2, popcounts(d), Kmax=1)[0][1])      # exact oracle floor Phi* = D_1
    ax.plot(budgets, c2["free"], "o-", color=P["accent"], ms=4, label="free search (order-2 task)")
    ax.plot(budgets, c2[1], "s-", color=P["core"], ms=4, label="legible reader (order-2 task)")
    ax.plot(budgets, c1[1], "^-", color=P["green"], ms=4, label="legible reader (separable task)")
    ax.axhline(phi, color=P["core"], lw=0.8, ls=":", label="oracle floor  $\\Phi^\\ast = D_1$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("metered evaluation budget  $B$")
    ax.set_ylabel("gap to optimum  ($\\sigma_f$)")
    ax.set_title("Opacity $\\neq$ performance: free search\ncloses, the legible floor $\\Phi^\\ast$ remains")
    ax.set_ylim(-0.05, None)
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "C")
    return dict(budgets=budgets, free_order2=c2["free"].tolist(),
                legible_order2=c2[1].tolist(), legible_separable=c1[1].tolist(), phi_star=phi)


# ---- D: the dividend kills the part-count false positive ---------------------
def panel_falsepositive(ax):
    d, seeds = 12, 16
    B, _ = bit_matrix(d); pc = popcounts(d)
    fams = [("linear", "separable"), ("planted", "sparse"), ("complex", "epistatic")]
    old, new = [], []
    kf = lambda dd: max(2, dd // 4)
    for kind, _lab in fams:
        om, _ = premium(d, kind, seeds, np.random.default_rng(200 + d), kf)
        d1 = np.mean([dividend(make_objective(d, kind, np.random.default_rng(300 + s))(B),
                               pc, Kmax=2)[0][1] for s in range(seeds)])
        old.append(om); new.append(d1)
    x = np.arange(len(fams)); w = 0.38
    ax.bar(x - w / 2, old, w, color=style.FAINT, label="old premium (part-count)")
    ax.bar(x + w / 2, new, w, color=P["frontier"], label="dividend $D_1$ (interaction)")
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n({l})" for k, l in fams], fontsize=7.5)
    ax.set_ylabel("opacity cost  ($\\sigma_f$)")
    ax.set_title("The dividend voids the part-count\nfalse positive (linear $\\to 0$)")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "D")
    return dict(families=[k for k, _ in fams], old_premium=old, dividend_D1=new)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.2))
    rec = {}
    rec["staircase"] = panel_staircase(axes[0, 0])
    rec["dimension"] = panel_dimension(axes[0, 1])
    rec["budget"] = panel_budget(axes[1, 0])
    rec["false_positive"] = panel_falsepositive(axes[1, 1])
    fig.suptitle("Figure O.  The opacity dividend:  forced-opacity order $K^\\ast$ equals interaction order $r$, and the legible floor $\\Phi^\\ast$ is zero iff the task is separable",
                 fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "figO_dividend.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    json.dump(rec, open(os.path.join(os.path.dirname(__file__), "..", "out", "figO.json"), "w"), indent=2)
    print(json.dumps({k: (v if not isinstance(v, dict) else "...") for k, v in rec.items()}, indent=0)[:200])


if __name__ == "__main__":
    main()
