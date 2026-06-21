"""
ROBUSTNESS -- are the five results regions of parameter space, or tuned points?

Each panel sweeps the parameters most open to a charge of tuning and shows the
phenomenon survives over a region (and, for Kuramoto, tracks the analytic law):
the conclusions do not depend on the particular operating points, which are marked.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from factory import DarkFactory, FactoryParams
from learners import Hedge, BlumMansourSwap
from transfer_operator import ulam_operator, reversibilize, spectrum, coherence_timescale
from ews import detrend, rolling_variance, kendall_trend
from kuramoto import simulate, critical_coupling_lorentzian
from darkfactory import bit_matrix, make_objective, _U_O, _U_L, leader_schedule, stackelberg_value
style.apply()
P = style.PALETTE


# ---- R-A: opacity premium over (d, k) --------------------------------------
def panel_opacity(ax):
    ds = [10, 12, 14, 16, 18]
    ks = [2, 3, 4, 5, 6]
    G = np.zeros((len(ks), len(ds)))
    for j, d in enumerate(ds):
        B, pc = bit_matrix(d)
        for i, k in enumerate(ks):
            legible = pc <= k
            g = []
            for s in range(8):
                f = make_objective(d, "complex", np.random.default_rng(10 * d + k + s))(B)
                g.append((f.max() - f[legible].max()) / (f.std() + 1e-12))
            G[i, j] = np.mean(g)
    im = ax.imshow(G, origin="lower", aspect="auto", cmap="YlOrBr",
                   extent=[ds[0] - 1, ds[-1] + 1, ks[0] - 0.5, ks[-1] + 0.5])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("premium $/\\sigma_f$", fontsize=7)
    ax.set_xticks(ds); ax.set_yticks(ks)
    ax.set_xlabel("dimension  $d$"); ax.set_ylabel("legibility budget  $k$")
    ax.set_title("Opacity: premium is positive\nacross $(d,k)$, not a point")
    ax.grid(False)
    style.panel_tag(ax, "A")
    return G.tolist()


# ---- R-B: Stackelberg gap across learning rate eta -------------------------
def _play_eta(T, disclosure, eta, seed):
    rng = np.random.default_rng(seed)
    UL, UO = _U_L[:, :3], _U_O[:, :3]
    hedge = Hedge(3, eta=eta, rng=rng); swap = BlumMansourSwap(3, eta=eta, rng=rng)
    sched = leader_schedule(T); g = np.empty(T)
    for t in range(T):
        ph, ps = hedge.distribution(), swap.distribution()
        p = (1 - disclosure) * ph + disclosure * ps; p /= p.sum()
        col = int(rng.choice(3, p=p)); g[t] = UO[sched[t], col]
        r = UL[sched[t], :].copy(); hedge.update(r); swap.update(r)
    return float((np.cumsum(g) / (np.arange(T) + 1))[-1])


def panel_stackelberg(ax):
    etas = [0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
    fr = [np.mean([_play_eta(20000, 0.0, e, s) for s in range(4)]) for e in etas]
    co = [np.mean([_play_eta(20000, 1.0, e, s) for s in range(4)]) for e in etas]
    ax.axhline(0.5, color=P["frontier"], ls=":", lw=1.0)
    ax.axhline(0.0, color=P["core"], ls=":", lw=1.0)
    ax.plot(etas, fr, "o-", color=P["frontier"], ms=4, label="mean-based frontier")
    ax.plot(etas, co, "o-", color=P["core"], ms=4, label="no-swap-regret core")
    ax.axvline(0.7, color=style.FAINT, lw=0.8, ls="--")
    ax.set_xlabel("learning rate  $\\eta$")
    ax.set_ylabel("extracted value at horizon")
    ax.set_title("Stackelberg gap holds across\nlearning rate  ($U^\\ast$ top, $V$ bottom)")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "B")
    return etas, fr, co


# ---- R-C: two-version structure over (M, mu) -------------------------------
def panel_versions(ax):
    Ms = [40, 60, 80, 120, 200]
    mus = [0.01, 0.03, 0.05, 0.08, 0.12]
    S = np.zeros((len(mus), len(Ms)))
    for j, M in enumerate(Ms):
        for i, mu in enumerate(mus):
            mp = DarkFactory(FactoryParams(peakA=0.35, peakB=0.65, width=0.09, eta=2.0,
                                           M=M, mu=mu, c=1.0, seed=3)).run(40000)["mean_pos"]
            Prev, _ = reversibilize(ulam_operator(mp, n_boxes=30)[0])
            v, _ = spectrum(Prev)
            S[i, j] = coherence_timescale(v[1]) / max(coherence_timescale(v[2]), 1e-9)
    im = ax.imshow(np.log10(S), origin="lower", aspect="auto", cmap="viridis",
                   extent=[Ms[0] - 10, Ms[-1] + 20, mus[0] - 0.005, mus[-1] + 0.01])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("$\\log_{10}(\\tau_2/\\tau_3)$", fontsize=7)
    ax.scatter([60], [0.03], marker="*", s=90, color="white", edgecolor="k", zorder=5)
    ax.set_xlabel("population  $M$"); ax.set_ylabel("exploration  $\\mu$")
    ax.set_title("Versions: robust separation over\na region of $(M,\\mu)$  ($\\star$=Fig 3)")
    ax.grid(False)
    style.panel_tag(ax, "C")
    return S.tolist()


# ---- R-D: pathology phase diagram over (mu, M) -----------------------------
def panel_pathologies(ax):
    mus = np.linspace(0.004, 0.22, 9)
    cs = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    # classes: 0 healthy, 1 stable failure, 2 learning death, 3 thrash  (fixed M=120)
    grid = np.zeros((len(mus), len(cs)), int)
    for j, c in enumerate(cs):
        for i, mu in enumerate(mus):
            o = DarkFactory(FactoryParams(peakA=0.3, peakB=0.7, width=0.07, eta=4, M=120,
                                          mu=float(mu), c=float(c), norm_target=0.7, seed=1)).run(16000)
            vol = o["mean_pos"][-4000:].std(); variety = o["variety"][-2000:].mean()
            norm = o["norm_sat"][-2000:].mean()
            if variety < 1.2:
                grid[i, j] = 2                       # learning death
            elif variety > 2.5 or vol > 0.02:
                grid[i, j] = 3                       # thrash
            elif norm < 0.6:
                grid[i, j] = 1                       # stable failure
            else:
                grid[i, j] = 0                       # healthy
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([P["green"], P["core"], P["accent"], P["frontier"]])
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=3,
                   extent=[cs[0] - 0.1, cs[-1] + 0.1, mus[0] - 0.012, mus[-1] + 0.012])
    ax.set_xlabel("spec height  $c$  (at $M=120$)"); ax.set_ylabel("exploration  $\\mu$")
    ax.set_title("Pathologies tile $(\\mu,c)$ space\n(each is a region)")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=c, label=l)
               for c, l in zip([P["green"], P["core"], P["accent"], P["frontier"]],
                               ["healthy", "stable failure", "learning death", "thrash"])]
    style.legend_below(ax, ncol=2, handles=handles, labels=[h.get_label() for h in handles])
    ax.grid(False)
    style.panel_tag(ax, "D")
    return grid.tolist()


# ---- R-E: EWS trend across ramp duration and noise -------------------------
def panel_ews(ax):
    ns = [8000, 12000, 16000, 20000]
    for M, col in [(200, P["accent"]), (400, P["core"]), (800, P["frontier"])]:
        taus = []
        for n in ns:
            f = DarkFactory(FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=M, mu=0.02, seed=3))
            f.run(1500, c_schedule=np.full(1500, 1.6))
            mp = f.run(n, c_schedule=np.linspace(1.6, 0.4, n))["mean_pos"]
            jump = int(np.argmin(np.diff(mp))); w = 400
            taus.append(kendall_trend(rolling_variance(detrend(mp[:jump], w), w)) if jump > w + 50 else np.nan)
        ax.plot(ns, taus, "o-", color=col, ms=4, label=f"M = {M}")
    ax.axhline(0, color=style.FAINT, lw=0.8)
    ax.set_ylim(-0.2, 1)
    ax.set_xlabel("ramp duration  (rounds)")
    ax.set_ylabel("pre-fold variance trend  (Kendall $\\tau$)")
    ax.set_title("Critical slowing down is positive\nacross ramp rate and noise")
    style.legend_below(ax, ncol=3)
    style.panel_tag(ax, "E")


# ---- R-F: Kuramoto onset vs analytic 2*gamma -------------------------------
def panel_kuramoto(ax):
    gammas = [0.3, 0.4, 0.5, 0.7, 0.9]
    Ks = np.linspace(0, 3.0, 25)
    onsets = []
    rng = np.random.default_rng(0)
    for g in gammas:
        omega = g * np.tan(np.pi * (rng.random(400) - 0.5))
        omega = np.clip(omega - np.median(omega), -50, 50)
        rs = np.array([simulate(400, K, omega, T=50, dt=0.02,
                                rng=np.random.default_rng(rng.integers(1 << 30)))[0] for K in Ks])
        thr = 0.3
        idx = np.argmax(rs > thr)
        if idx > 0:
            k0, k1 = Ks[idx - 1], Ks[idx]; r0, r1 = rs[idx - 1], rs[idx]
            onsets.append(k0 + (thr - r0) * (k1 - k0) / (r1 - r0 + 1e-9))
        else:
            onsets.append(np.nan)
    gl = np.linspace(0.25, 0.95, 50)
    ax.plot(gl, 2 * gl, "-", color=P["neutral"], lw=1.2, label="analytic  $K_c = 2\\gamma$")
    ax.plot(gammas, onsets, "o", color=P["core"], ms=6, label="measured onset")
    ax.set_xlabel("dependency spread  $\\gamma$")
    ax.set_ylabel("entrainment onset  $K_c$")
    ax.set_title("Entrainment threshold tracks the\nanalytic law across diversity")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "F")
    return gammas, onsets


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))
    rec = {}
    rec["opacity"] = panel_opacity(axes[0, 0])
    rec["stackelberg"] = panel_stackelberg(axes[0, 1])
    rec["versions"] = panel_versions(axes[0, 2])
    rec["pathologies"] = panel_pathologies(axes[1, 0])
    panel_ews(axes[1, 1])
    rec["kuramoto_onsets"] = panel_kuramoto(axes[1, 2])
    fig.suptitle("Figure R.  Robustness:  each result is a region of parameter space (and Kuramoto tracks theory), not a tuned point",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "figR_robustness.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    json.dump(rec, open(os.path.join(os.path.dirname(__file__), "..", "out", "figR.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
