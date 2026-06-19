"""
FIGURE 1 -- The Opacity-Cost Frontier.   Tests Claim [a].

Claim [a]: "the regions of configuration space legible to a human operator are
vanishingly small relative to the regions in which constraint-satisfying
assemblies can be found ... readability is priceable."

We operationalise legibility as PARSEABLE LOW-DIMENSIONAL STRUCTURE -- a solution
is legible if it activates at most k of the d design knobs (a k-sparse / low-order
assembly a human can hold in mind). (We deliberately do NOT equate legibility
with short description length; the paper's "short => opaque" is backwards -- the
minimal program is the smallest object, hence easiest to read. The defensible
axis is structural sparsity, not code length.)

Configuration space: x in {0,1}^d. Objective to maximise:
    f(x) = w . x + sum_{(i,j) in P} W_ij x_i x_j     (random Ising / quadratic)
The global optimum generically uses high-order, spread-out structure.

Panel A: the legible fraction  C(d, <=k)/2^d  collapses exponentially in d
         (the incompressibility / pigeonhole counting fact, exact).
Panel B: the cost premium of the best LEGIBLE assembly over the global optimum,
         for the complex objective vs TWO honesty controls (a linear objective
         and an objective with a deliberately k-sparse planted optimum). The
         premium grows only when the problem's solutions are genuinely complex --
         exactly the paper's hedged claim, not a universal law.
Panel C: the legibility tax at fixed d -- premium vs legibility budget k, with
         the legible fraction on the twin axis. This is "readability is priceable"
         drawn as a curve: each bit of readability bought back costs output value.
"""
import os, sys, json
import numpy as np
from math import comb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
style.apply()
P = style.PALETTE


def bit_matrix(d):
    """All 2^d binary vectors as rows (uint8), plus popcounts."""
    n = 1 << d
    idx = np.arange(n, dtype=np.uint32)
    B = ((idx[:, None] >> np.arange(d)[None, :]) & 1).astype(np.uint8)
    return B, B.sum(axis=1)


def make_objective(d, kind, rng, planted_k=2):
    """Return a function values(B) -> f over all rows of B."""
    w = rng.standard_normal(d)
    if kind == "linear":
        pairs, Wp = np.zeros((0, 2), int), np.zeros(0)
    elif kind == "planted":
        # optimum is a specific k-sparse vector: reward matching it, no interactions
        support = rng.choice(d, size=planted_k, replace=False)
        w = -np.abs(rng.standard_normal(d))          # prefer zeros everywhere...
        w[support] = np.abs(rng.standard_normal(planted_k)) + 1.0  # ...except support
        pairs, Wp = np.zeros((0, 2), int), np.zeros(0)
    else:  # complex Ising: dense-ish random quadratic
        m = 2 * d
        pairs = np.array([rng.choice(d, size=2, replace=False) for _ in range(m)])
        Wp = rng.standard_normal(m)

    def values(B):
        f = B.astype(np.float64) @ w
        for (i, j), wij in zip(pairs, Wp):
            f += wij * (B[:, i] * B[:, j])
        return f
    return values


def premium(d, kind, n_draws, rng, k_func):
    """Mean cost premium (f_global - f_legible)/std(f), averaged over draws."""
    B, pc = bit_matrix(d)
    k = k_func(d)
    legible = pc <= k
    out = []
    for _ in range(n_draws):
        f = make_objective(d, kind, rng)(B)
        s = f.std()
        gap = (f.max() - f[legible].max()) / (s + 1e-12)
        out.append(gap)
    return float(np.mean(out)), float(np.std(out) / np.sqrt(n_draws))


def main():
    rng = np.random.default_rng(7)
    ds = [8, 10, 12, 14, 16, 18, 20]
    n_draws = 30

    # ---- Panel A data: legible fraction vs d, for several FIXED k -----------
    # (fixed k gives the exact, monotone counting statement; a d-proportional k
    # would introduce an integer-step sawtooth that is an artifact, not signal.)
    kA = {"k = 2": lambda d: 2, "k = 3": lambda d: 3, "k = 4": lambda d: 4}
    fracs = {name: [sum(comb(d, j) for j in range(0, kf(d) + 1)) / (1 << d) for d in ds]
             for name, kf in kA.items()}

    # ---- Panel B data: premium vs d (legibility budget k = d/4) -------------
    kbudget = lambda d: max(2, d // 4)
    premB = {}
    for kind, lab in [("complex", "complex objective"),
                      ("linear", "linear control"),
                      ("planted", "planted-sparse control")]:
        means, errs = [], []
        for d in ds:
            m, e = premium(d, kind, n_draws, np.random.default_rng(100 + d), kbudget)
            means.append(m); errs.append(e)
        premB[lab] = (np.array(means), np.array(errs))

    # ---- Panel C data: legibility tax at fixed d ----------------------------
    dC = 18
    B, pc = bit_matrix(dC)
    ks = list(range(1, dC + 1))
    taxC, fracC = [], []
    for k in ks:
        legible = pc <= k
        gaps = []
        for _ in range(n_draws):
            f = make_objective(dC, "complex", np.random.default_rng(2000 + k * 7 + _))(B)
            gaps.append((f.max() - f[legible].max()) / (f.std() + 1e-12))
        taxC.append(np.mean(gaps))
        fracC.append(sum(comb(dC, j) for j in range(0, k + 1)) / (1 << dC))

    # ----------------------------- plot -------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    axA = axes[0]
    for (name, _), col in zip(kA.items(), [P["frontier"], P["core"], P["accent"]]):
        axA.semilogy(ds, fracs[name], "o-", color=col, ms=4, label=name)
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

    # persist numbers for the writeup / audit
    rec = dict(ds=ds, legible_fraction=fracs,
               premium_complex=list(premB["complex objective"][0]),
               premium_linear=list(premB["linear control"][0]),
               premium_planted=list(premB["planted-sparse control"][0]),
               taxC_k=ks, taxC_premium=taxC, taxC_fraction=fracC, dC=dC)
    json.dump(rec, open(os.path.join(os.path.dirname(__file__), "..", "out", "fig1.json"), "w"), indent=2)
    print("global-opt premium @ d=20:  complex=%.2f  linear=%.2f  planted=%.2f" % (
        premB["complex objective"][0][-1], premB["linear control"][0][-1],
        premB["planted-sparse control"][0][-1]))


if __name__ == "__main__":
    main()
