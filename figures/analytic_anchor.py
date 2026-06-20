"""
ANALYTIC ANCHORING -- overlay closed-form laws on the simulation.

A toy model earns trust when its emergent numbers fall on lines drawn from
theory, not just on lines fit to themselves. Figures R and D show the results
are regions and are generic; this figure shows the two EMERGENT results -- the
metastable versions (Fig 3) and the catastrophe (Fig 4) -- obey the closed-form
laws their mechanisms predict. (The Kuramoto onset is anchored to Kc = 2*gamma
in Fig R-F; the Stackelberg V / U* are exact by construction.)

  A  Metastable escape is MEMORYLESS. The dwell-time distribution in a version is
     exponential -- the signature of Poisson barrier-crossing (Kramers / large
     deviations). Versions are a genuine two-state process, not a slow drift.

  B  The transfer-operator spectral gap (Fig 3's robustness grade) IS the escape
     clock: the measured mean dwell tracks the spectral relaxation time tau_2 at
     a constant ratio ~2 -- the two-state relaxation law  dwell = 2*tau_2.

  C  Approaching the fold (spec height c -> c* = 1, where the rival peak's height
     is matched), BOTH early-warning signals rise: lag-1 autocorrelation -> 1 and
     variance diverges. Critical slowing down.

  D  Those two signals are not independent -- they obey the AR(1) / Ornstein-
     Uhlenbeck law  sigma^2 = D / (1 - alpha^2)  (Scheffer et al., Box 3): plotted
     against 1/(1-alpha^2), the measured variance is linear.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from factory import DarkFactory, FactoryParams
from transfer_operator import ulam_operator, reversibilize, spectrum, coherence_timescale
style.apply()
P = style.PALETTE

CSTAR = 1.0   # the fold: B-peak height == A-peak height (1.0); below it, B loses


# ---------- helpers ----------------------------------------------------------
def dwells(mp, lo=0.42, hi=0.58, floor=20):
    """Hysteresis state labelling; a dwell is a stay in one basin past `floor`."""
    state = None; last = 0; dw = []
    for i, x in enumerate(mp):
        ns = state
        if x < lo: ns = 0
        elif x > hi: ns = 1
        if ns is not None and ns != state:
            if state is not None and (i - last) >= floor:
                dw.append(i - last)
            last = i; state = ns
    return np.array(dw)


def ar1(y):
    y = y - y.mean()
    return float(np.dot(y[:-1], y[1:]) / max(np.dot(y, y), 1e-30))


def _run_meta(M, mu, n, seed):
    return DarkFactory(FactoryParams(peakA=0.35, peakB=0.65, width=0.085, eta=2.5,
                                     M=M, mu=mu, c=1.0, seed=seed)).run(n)["mean_pos"]


# ---------- A: escape is memoryless (exponential dwell) ----------------------
def panel_memoryless(ax):
    dw = np.concatenate([dwells(_run_meta(55, 0.012, 250000, s)) for s in range(4)])
    md = dw.mean(); cv = dw.std() / md
    ts = np.sort(dw); S = 1.0 - np.arange(len(ts)) / len(ts)
    m = (S > 0.03) & (S < 0.9)                       # fit the bulk, not the floor/tail
    b, a = np.polyfit(ts[m], np.log(S[m]), 1)        # log S = a + b t
    R = np.corrcoef(ts[m], np.log(S[m]))[0, 1]
    tau = -1.0 / b
    ax.semilogy(ts, S, ".", ms=2.6, color=P["core"], alpha=0.55, label="measured dwells")
    tt = np.linspace(0, ts[int(0.985 * len(ts))], 200)
    ax.semilogy(tt, np.exp(a + b * tt), "-", color=P["frontier"], lw=1.6,
                label=f"exponential  ($R={R:.3f}$, CV$\\,={cv:.2f}$)")
    ax.set_ylim(0.02, 1.05)
    ax.set_xlabel("dwell time in a version  (rounds)")
    ax.set_ylabel("survival  $P(T>t)$")
    ax.set_title("Metastable escape is memoryless\n(dwell times are exponential)")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "A")
    return dict(mean_dwell=float(md), cv=float(cv), tau_fit=float(tau), R=float(R))


# ---------- B: spectral gap IS the escape clock (dwell = 2*tau2) --------------
def panel_spectral_clock(ax):
    Ms = [40, 60, 85, 120]
    md_pts, t2_pts = [], []
    for M in Ms:
        mds, t2s = [], []
        for s in range(3):
            mp = _run_meta(M, 0.012, 200000, s)
            mds.append(dwells(mp).mean())
            Prev, _ = reversibilize(ulam_operator(mp, n_boxes=30)[0])
            v, _ = spectrum(Prev)
            t2s.append(coherence_timescale(v[1]))
        md_pts.append(np.mean(mds)); t2_pts.append(np.mean(t2s))
    md_pts = np.array(md_pts); t2_pts = np.array(t2_pts)
    slope = float(np.sum(md_pts * t2_pts) / np.sum(t2_pts ** 2))   # through-origin fit
    tt = np.linspace(0, t2_pts.max() * 1.12, 50)
    ax.plot(tt, 2 * tt, "-", color=P["frontier"], lw=1.5, label="two-state law  dwell $=2\\tau_2$")
    ax.plot(t2_pts, md_pts, "o", color=P["core"], ms=6, label=f"measured  (slope $={slope:.2f}$)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel("spectral relaxation time  $\\tau_2$")
    ax.set_ylabel("measured mean dwell")
    ax.set_title("The spectral gap is the escape clock\n($\\tau_2$ sets the dwell, $\\times 2$)")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "B")
    return dict(M=Ms, mean_dwell=md_pts.tolist(), tau2=t2_pts.tolist(), slope=slope)


# ---------- fold sweep (shared by C and D) -----------------------------------
def _fold_sweep():
    cs = [1.30, 1.20, 1.12, 1.06, 1.03, 1.015, 1.008]
    out_c, var, al = [], [], []
    for c in cs:
        f = DarkFactory(FactoryParams(peakA=0.3, peakB=0.7, width=0.05, eta=6,
                                      M=4000, mu=0.006, beta=0.0, c=c, seed=4))
        f.x = np.exp(-0.5 * ((f.grid - 0.7) / 0.03) ** 2); f.x /= f.x.sum()
        f.run(3000, c_schedule=np.full(3000, c))
        mp = np.array([f.step(c=c)["mean_pos"] for _ in range(8000)])
        if mp.min() < 0.55:        # guard: discard if it slipped off the B branch
            continue
        out_c.append(c); var.append(float(mp.var())); al.append(ar1(mp))
    return np.array(out_c), np.array(var), np.array(al)


# ---------- C: critical slowing down before the fold -------------------------
def panel_slowing_down(ax, cs, var, al):
    dist = cs - CSTAR
    vn = var / var.max()
    ax.semilogx(dist, al, "o-", color=P["core"], ms=5, label="lag-1 autocorr.  $\\alpha$")
    ax.semilogx(dist, vn, "s-", color=P["accent"], ms=5, label="variance  (normalised)")
    ax.invert_xaxis()                          # approach the fold left -> right
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("distance to fold  $c-c^\\ast$")
    ax.set_ylabel("early-warning indicator")
    ax.set_title("Critical slowing down before the fold\n(both signals rise as $c\\to c^\\ast$)")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "C")


# ---------- D: the AR(1) / OU variance law -----------------------------------
def panel_ar1_law(ax, cs, var, al):
    x = 1.0 / (1.0 - al ** 2)
    A = np.vstack([x, np.ones_like(x)]).T
    (D, c0), *_ = np.linalg.lstsq(A, var, rcond=None)
    R = np.corrcoef(x, var)[0, 1]
    xx = np.linspace(1.0, x.max() * 1.05, 50)
    ax.plot(xx, D * xx + c0, "-", color=P["frontier"], lw=1.5,
            label=f"OU law  $\\sigma^2\\propto 1/(1-\\alpha^2)$\n($R={R:.3f}$)")
    ax.plot(x, var, "o", color=P["core"], ms=6, label="measured")
    ax.set_xlabel("$1/(1-\\alpha^2)$")
    ax.set_ylabel("variance  $\\sigma^2$")
    ax.set_title("Variance obeys the AR(1) law\n(rising autocorrelation $\\Rightarrow$ rising variance)")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "D")
    return dict(c=cs.tolist(), variance=var.tolist(), alpha=al.tolist(),
                D=float(D), intercept=float(c0), R=float(R))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
    rec = {}
    rec["memoryless"] = panel_memoryless(axes[0, 0])
    rec["spectral_clock"] = panel_spectral_clock(axes[0, 1])
    cs, var, al = _fold_sweep()
    panel_slowing_down(axes[1, 0], cs, var, al)
    rec["ar1_law"] = panel_ar1_law(axes[1, 1], cs, var, al)
    fig.suptitle("Figure A.  Analytic anchoring:  the emergent results fall on the closed-form laws their mechanisms predict",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "figA_anchor.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    print(json.dumps(rec, indent=2))
    json.dump(rec, open(os.path.join(os.path.dirname(__file__), "..", "out", "figA.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
