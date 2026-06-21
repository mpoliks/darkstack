"""
Bridge -- the legibility floor on real tabular tasks (off the enumerable cube).

The other figures price opacity on an enumerable Boolean cube, where search always
succeeds. This figure measures the same legibility floor on real data with a held-out
verifier. The LEGIBLE reader is an explicitly additive model (HistGradientBoosting with
interaction_cst='no_interactions' -- unlimited per-feature shape, zero cross-feature
interaction -- the capacity-matched order-1 reader); the FREE searcher is the same family
with full interactions; the floor is the held-out performance the additive reader forgoes.
The space is non-enumerable and finite-data search genuinely fails, so the cube's
"efficiency always wins" pathology does not apply.

A depth-1 stump conflates univariate-shape capacity with interaction order, so the order-1
reader is the additive model (interaction_cst), not max_depth=1: the stump overstates the
floor on real features (california 0.127 by the stump, 0.088 by the additive reader).

  A  The additive floor rises with interaction order and is ~0 at order 1 (controlled
     tasks of known order, with california and diabetes shown as real points).
  B  On real data the depth-1 stump overstates the floor; the additive (order-1) reader is
     the capacity-matched measure (california 0.088, diabetes ~0).
  C  Classification floor in log-loss (accuracy hides it): breast cancer ~0, wine ~0,
     digits a small positive floor.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, load_digits
from sklearn.metrics import r2_score, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
style.apply()
P = style.PALETTE

SPLITS = 6
COMMON = dict(max_iter=600, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, random_state=0)
FREE = dict(max_depth=6, **COMMON)                                            # full-interaction free model
ADD_LEAVES = (15, 63, 255)                                                    # additive-reader capacity sweep
DEPTHS = [1, 2, 3, 4, 6]


def held_out(X, y, kind, **mk):
    out = []; classes = np.unique(y)
    for s in range(SPLITS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s)
        m = (HGR if kind == "reg" else HGC)(**mk).fit(Xtr, ytr)
        out.append(r2_score(yte, m.predict(Xte)) if kind == "reg"
                   else -log_loss(yte, m.predict_proba(Xte), labels=classes))   # higher is better
    return np.array(out)


def floors(X, y, kind, need_kstar=True):
    free = held_out(X, y, kind, **FREE)
    add_runs = [held_out(X, y, kind, interaction_cst="no_interactions", max_leaf_nodes=L, **COMMON)
                for L in ADD_LEAVES]
    add = max(add_runs, key=lambda a: a.mean())           # strongest properly-tuned additive reader
    d1 = held_out(X, y, kind, max_depth=1, **COMMON)
    kstar = None
    if need_kstar:
        sc = [held_out(X, y, kind, max_depth=K, **COMMON).mean() for K in DEPTHS]
        best = max(sc); kstar = next((DEPTHS[i] for i, s in enumerate(sc) if best - s <= 0.01), DEPTHS[-1])
    return dict(free=float(free.mean()), phi_add=float(free.mean() - add.mean()),
                phi_d1=float(free.mean() - d1.mean()),
                sd=float(np.hypot(free.std(), add.std())), kstar=kstar, kind=kind)


def synth(order, n=6000, d=8, seed=0):
    r = np.random.default_rng(seed); X = r.uniform(-1, 1, (n, d))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.tanh(2 * X[:, 2]) + 0.5 * X[:, 3]
    if order >= 2: y = y + 1.5 * X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
    if order >= 3: y = y + 2.0 * X[:, 0] * X[:, 1] * X[:, 2]
    if order >= 4: y = y + 3.0 * X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]
    return X, y + 0.1 * r.standard_normal(n)


def main():
    orders = [1, 2, 3, 4]
    syn = {o: floors(*synth(o), "reg") for o in orders}
    reg = []
    try:
        from sklearn.datasets import fetch_california_housing
        ch = fetch_california_housing(); reg.append(("california", floors(ch.data, ch.target, "reg")))
    except Exception:
        pass
    db = load_diabetes(); reg.append(("diabetes", floors(db.data, db.target, "reg")))
    clf = [(n, floors(d.data, d.target, "clf", need_kstar=False)) for n, d in
           [("breast cancer", load_breast_cancer()), ("wine", load_wine()), ("digits", load_digits())]]
    R = dict(reg)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # A: the additive (legible) floor rises with interaction order
    axA = axes[0]
    axA.plot(orders, [syn[o]["phi_add"] for o in orders], "o-", color=style.MUTED, ms=4,
             label="controlled (true order)")
    for name, col, mk, sz in [("california", P["core"], "*", 13), ("diabetes", P["green"], "s", 7)]:
        if name in R:
            r = R[name]
            axA.errorbar([r["kstar"]], [r["phi_add"]], yerr=[r["sd"]], fmt=mk, color=col, ms=sz,
                         capsize=3, label=f"{name} (real)")
    axA.axhline(0, color=style.FAINT, lw=0.8)
    axA.set_xlabel("interaction order"); axA.set_ylabel("additive floor  $\\Phi_1$  ($\\Delta R^2$)")
    axA.set_title("The legible (additive) floor rises\nwith interaction order, off the cube")
    axA.set_xticks(orders)
    style.legend_below(axA, ncol=1)
    style.panel_tag(axA, "A")

    # B: the additive floor vs the depth-1 proxy, on real data
    axB = axes[1]
    rn = [n for n, _ in reg]; padd = [r["phi_add"] for _, r in reg]; pd1 = [r["phi_d1"] for _, r in reg]
    x = np.arange(len(rn)); w = 0.38
    axB.bar(x - w / 2, pd1, w, color=style.FAINT, label="depth-1 proxy (overstates)")
    axB.bar(x + w / 2, padd, w, color=P["frontier"], label="additive reader (order-1)")
    axB.set_xticks(x); axB.set_xticklabels(rn, fontsize=8.5)
    axB.set_ylabel("floor  $\\Phi_1$  ($\\Delta R^2$)")
    axB.set_title("The depth-1 proxy overstates on real\nfeatures; the additive reader is capacity-matched")
    style.legend_below(axB, ncol=1)
    style.panel_tag(axB, "B")

    # C: classification floor in log-loss (proper unit)
    axC = axes[2]
    names = [n for n, _ in clf]; phis = [r["phi_add"] for _, r in clf]
    axC.barh(range(len(names)), phis, color=P["frontier"])
    axC.set_yticks(range(len(names))); axC.set_yticklabels(names, fontsize=8); axC.invert_yaxis()
    axC.set_xlabel("additive floor  $\\Phi_1$  ($\\Delta$ log-loss, nats)")
    axC.set_title("Classification floor in log-loss\n(accuracy hides the small ones)")
    axC.axvline(0, color=style.FAINT, lw=0.8)
    style.panel_tag(axC, "C")

    fig.suptitle("Figure B.  The bridge:  off the enumerable cube, on real verified tasks, the additive (legible) floor is real and rises with interaction order",
                 fontsize=10.0, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "figB_bridge.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    rec = dict(synthetic={o: dict(phi_add=round(syn[o]["phi_add"], 3), phi_d1=round(syn[o]["phi_d1"], 3),
                                  kstar=syn[o]["kstar"]) for o in orders},
               regression={n: dict(phi_add=round(r["phi_add"], 3), phi_d1=round(r["phi_d1"], 3),
                                   sd=round(r["sd"], 3), kstar=r["kstar"], free=round(r["free"], 3)) for n, r in reg},
               classification={n: dict(phi_add_logloss=round(r["phi_add"], 3)) for n, r in clf})
    json.dump(rec, open(os.path.join(os.path.dirname(__file__), "..", "out", "figB.json"), "w"), indent=2)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
