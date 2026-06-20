"""
DYNAMICS INVARIANCE -- are the results artifacts of the specific update rule?

The population runs on a stochastic replicator (exponential/imitation selection)
and the learners are multiplicative-weights. Two checks that the load-bearing
results are not artifacts of those particular choices:

  A  The Stackelberg gap is a property of the LEARNER CLASS, not of Hedge:
     several mean-based learners (Hedge, FTPL, EXP3) are all steered above V
     toward U*, while a no-swap-regret core is held at V. (EXP3 is bandit, so it
     climbs more slowly -- a feedback effect, not a learner effect.)

  B/C Metastable versions survive a change of selection functional: exponential
     (replicator) AND linear-fitness selection both telegraph between two
     versions. They do NOT survive a change to a different CLASS of dynamics --
     best-response (logit) gives one version -- which is the honest finding:
     metastability needs imitation / positive frequency dependence, and within
     that class it is robust.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from learners import Hedge, EXP3, BlumMansourSwap
from transfer_operator import ulam_operator, reversibilize, spectrum, coherence_timescale, n_metastable
from darkfactory import _U_O, _U_L, leader_schedule
style.apply()
P = style.PALETTE


# ----- a third mean-based learner: Follow-the-Perturbed-Leader --------------
class FTPL:
    def __init__(self, K, eta=0.7, rng=None):
        self.K = K; self.eta = eta; self.G = np.zeros(K)
        self.rng = rng or np.random.default_rng()

    def sample(self):
        return int(np.argmax(self.G + self.rng.gumbel(size=self.K) / self.eta))

    def update(self, rvec):
        self.G += rvec


def play_traj(make, T=60000, bandit=False, seed=0):
    rng = np.random.default_rng(seed); learner = make(rng)
    sched = leader_schedule(T); g = np.empty(T)
    for t in range(T):
        col = learner.sample(); g[t] = _U_O[sched[t], :3][col]; rv = _U_L[sched[t], :3]
        learner.update(col, rv[col]) if bandit else learner.update(rv)
    return np.cumsum(g) / (np.arange(T) + 1)


# ----- population dynamics on the same two-well landscape -------------------
_K = 64
_grid = np.linspace(0, 1, _K)


def _peaks(c=1.0):
    g = lambda ctr, h: h * np.exp(-0.5 * ((_grid - ctr) / 0.09) ** 2)
    return g(0.35, 1.0) + g(0.65, c)


def run_population(kind, n=60000, M=60, mu=0.03, seed=2):
    rng = np.random.default_rng(seed)
    x = np.exp(-0.5 * ((_grid - 0.35) / 0.04) ** 2); x /= x.sum()
    mp = np.empty(n)
    for t in range(n):
        reward = _peaks()
        if kind == "replicator":
            f = np.exp(2.0 * (reward - reward.max())); ps = x * f
        elif kind == "linear":
            f = np.clip(1 + 1.5 * (reward - reward.mean()), 1e-6, None); ps = x * f
        elif kind == "logit":
            tgt = np.exp(2.0 * (reward - reward.max())); tgt /= tgt.sum()
            ps = 0.5 * x + 0.5 * tgt
        ps = ps / ps.sum()
        x = rng.multinomial(M, (1 - mu) * ps + mu / _K) / M
        mp[t] = _grid @ x
    return mp


def _separation(mp):
    Prev, _ = reversibilize(ulam_operator(mp, n_boxes=30)[0])
    v, _ = spectrum(Prev)
    return n_metastable(Prev, 0.95), coherence_timescale(v[1]) / coherence_timescale(v[2])


def main():
    fig = plt.figure(figsize=(12.2, 3.9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.18, wspace=0.34)
    axA = fig.add_subplot(gs[:, 0])
    axB1 = fig.add_subplot(gs[0, 1]); axB2 = fig.add_subplot(gs[1, 1])
    axC = fig.add_subplot(gs[:, 2])

    # --- Panel A: Stackelberg gap across learner classes ---
    learners = [("Hedge (MWU)", lambda r: Hedge(3, eta=0.7, rng=r), False, P["frontier"]),
                ("FTPL", lambda r: FTPL(3, eta=0.7, rng=r), False, P["green"]),
                ("EXP3 (bandit)", lambda r: EXP3(3, gamma=0.05, rng=r), True, P["accent"]),
                ("Blum–Mansour core", lambda r: BlumMansourSwap(3, eta=0.7, rng=r), False, P["core"])]
    trajA = {}
    for name, mk, bd, col in learners:
        traj = np.mean([play_traj(mk, 60000, bd, s) for s in range(4)], axis=0)
        trajA[name] = float(traj[-1])
        axA.plot(traj, color=col, lw=1.4, label=name)
    axA.axhline(0.5, color=style.FAINT, lw=0.8, ls=":", label="$U^\\ast=0.5$")
    axA.axhline(0.0, color=style.FAINT, lw=0.8, ls=":", label="$V=0$")
    axA.set_xlabel("round  $t$")
    axA.set_ylabel("extracted value (running avg)")
    axA.set_title("The gap is a property of the\nlearner CLASS, not of Hedge")
    axA.set_ylim(-0.5, 0.62)
    style.legend_below(axA, ncol=2, fontsize=6.6)
    style.panel_tag(axA, "A")

    # --- Panel B (two rows): metastability under two selection functionals ---
    for ax, kind, col, lab in [(axB1, "replicator", P["frontier"], "exponential (replicator)"),
                               (axB2, "linear", P["green"], "linear-fitness")]:
        mp = run_population(kind)
        seg = mp[:9000]
        ax.scatter(np.arange(9000)[seg <= 0.5], seg[seg <= 0.5], s=1.5, color=P["frontier"])
        ax.scatter(np.arange(9000)[seg > 0.5], seg[seg > 0.5], s=1.5, color=P["core"])
        ax.set_ylim(0.2, 0.8); ax.set_yticks([0.35, 0.65])
        ax.set_ylabel(lab, fontsize=7.0)
        if ax is axB1:
            ax.set_xticklabels([]); ax.set_title("Two versions survive a change of\nselection functional", fontsize=10)
            style.panel_tag(ax, "B")
        else:
            ax.set_xlabel("round  $t$")

    # --- Panel C: separation under three dynamics (imitation vs best-response) ---
    kinds = [("exponential\n(replicator)", "replicator", P["frontier"]),
             ("linear\nfitness", "linear", P["green"]),
             ("best-response\n(logit)", "logit", P["core"])]
    seps, ns = [], []
    for _, kind, _c in kinds:
        nm = np.mean([_separation(run_population(kind, seed=s))[1] for s in range(3)])
        nv = _separation(run_population(kind, seed=2))[0]
        seps.append(nm); ns.append(nv)
    axC.bar(range(3), seps, color=[c for _, _, c in kinds], alpha=0.85)
    axC.axhline(1.0, color=style.FAINT, lw=0.9, ls="--")
    ticklabels = [f"{k}\n({nv} version{'s' if nv != 1 else ''})" for (k, _, _), nv in zip(kinds, ns)]
    axC.set_xticks(range(3)); axC.set_xticklabels(ticklabels, fontsize=7.0)
    axC.set_ylabel("timescale separation  $\\tau_2/\\tau_3$")
    axC.set_title("Metastability needs imitation;\nbest-response has none")
    axC.set_ylim(0, max(seps) * 1.25)
    style.panel_tag(axC, "C")

    fig.suptitle("Figure D.  Dynamics invariance:  the gap is generic across mean-based learners; metastability is robust within imitation dynamics",
                 fontsize=11, y=1.02)
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "figD_dynamics.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    print("Stackelberg finals:", {k: round(v, 3) for k, v in trajA.items()})
    print("separations:", {k: round(s, 1) for (k, _, _), s in zip(kinds, seps)}, "n_versions:", ns)
    json.dump(dict(stackelberg_finals=trajA,
                   separation={kind: float(s) for (_, kind, _), s in zip(kinds, seps)},
                   n_versions={kind: int(nv) for (_, kind, _), nv in zip(kinds, ns)}),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "figD.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
