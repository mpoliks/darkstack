"""
The Dark Factory — one object, five lenses.

This is the unifying spine: a single factory whose ORGANS are the exact models the
paper cites, so every exact result is preserved while the artifact is one runnable
thing. The five figure scripts are thin plot-only views that import their data from
here; `run_factory.py` drives all of them from one object.

  f = DarkFactory()
  f.lens_*()        # fast headline numbers for each view
  f.data_figN()     # the full data each figure plots (matches the standalone runs)

The compute below is moved verbatim from the original figure scripts, so the numbers
(and the rendered PNGs) are bit-for-bit identical; the unification is structural.
"""
from __future__ import annotations
import os, sys
from math import comb
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from factory import DarkFactory as _Population, FactoryParams
from learners import Hedge, BlumMansourSwap
from transfer_operator import (ulam_operator, spectrum, reversibilize,
                               almost_invariant_sets, coherence_ratio,
                               coherence_timescale, n_metastable)
from ews import detrend, rolling_variance, rolling_ar1, kendall_trend
from kuramoto import simulate, critical_coupling_lorentzian
from control import PID, instability_index

# --- the committed game organ: exact Deng-Schneider-Sivan (V=0, U*=1/2) ------
_S = 1e-3
_U_O = np.array([[0., -2., -2.], [0., -2., 2.]]) / 2.0
_U_L = np.array([[_S, -1., 0.], [-1., 1., 0.]]) / 2.0


# ============================ organ-level helpers ============================
# (moved verbatim from the figure scripts so behaviour is identical)

# -- opacity (Fig 1) ---------------------------------------------------------
def bit_matrix(d):
    n = 1 << d
    idx = np.arange(n, dtype=np.uint32)
    B = ((idx[:, None] >> np.arange(d)[None, :]) & 1).astype(np.uint8)
    return B, B.sum(axis=1)


def make_objective(d, kind, rng, planted_k=2):
    w = rng.standard_normal(d)
    if kind == "linear":
        pairs, Wp = np.zeros((0, 2), int), np.zeros(0)
    elif kind == "planted":
        support = rng.choice(d, size=planted_k, replace=False)
        w = -np.abs(rng.standard_normal(d))
        w[support] = np.abs(rng.standard_normal(planted_k)) + 1.0
        pairs, Wp = np.zeros((0, 2), int), np.zeros(0)
    else:
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


# -- stackelberg (Fig 2) -----------------------------------------------------
def leader_schedule(T):
    sched = np.zeros(T, dtype=int)
    sched[T // 2:] = 1
    return sched


def stackelberg_value(n_actions=3):
    ps = np.linspace(0, 1, 2_000_001)
    alpha = np.stack([1 - ps, ps], axis=1)
    learner_payoff = alpha @ _U_L[:, :n_actions]
    br = np.argmax(learner_payoff, axis=1)
    opt = alpha[:, 0] * _U_O[0, :n_actions][br] + alpha[:, 1] * _U_O[1, :n_actions][br]
    return float(opt.max())


def play(T, disclosure=0.0, n_actions=3, seed=0):
    rng = np.random.default_rng(seed)
    UL = _U_L[:, :n_actions]
    UO = _U_O[:, :n_actions]
    hedge = Hedge(n_actions, eta=0.7, rng=rng)
    swap = BlumMansourSwap(n_actions, eta=0.7, rng=rng)
    sched = leader_schedule(T)
    gain = np.empty(T)
    for t in range(T):
        row = sched[t]
        ph = hedge.distribution()
        ps = swap.distribution()
        p = (1 - disclosure) * ph + disclosure * ps
        p = p / p.sum()
        col = int(rng.choice(n_actions, p=p))
        gain[t] = UO[row, col]
        reward = UL[row, :].copy()
        hedge.update(reward)
        swap.update(reward)
    return np.cumsum(gain) / (np.arange(T) + 1)


# -- versioning (Fig 3) ------------------------------------------------------
_V3_BASE = dict(peakA=0.35, peakB=0.65, width=0.09, eta=2.0, M=60, c=1.0)


def debounced_switches(mp, lo=0.4, hi=0.6):
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


# -- pathologies (Fig 4) -----------------------------------------------------
def fingerprint(o):
    mp = o["mean_pos"]
    return dict(volatility=float(mp[-5000:].std()),
                variety=float(o["variety"][-3000:].mean()),
                norm_unmet=1 - float(o["norm_sat"][-3000:].mean()),
                proxy_gap=max(0.0, float(o["metric_sat"][-3000:].mean()) - float(o["norm_sat"][-3000:].mean())))


def _ews_run(seed, n=12000):
    c = np.linspace(1.6, 0.4, n)
    f = _Population(FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=400, mu=0.02, seed=seed))
    f.run(1500, c_schedule=np.full(1500, 1.6))
    o = f.run(n, c_schedule=c)
    mp = o["mean_pos"]
    jump = int(np.argmin(np.diff(mp)))
    w = 400
    res = detrend(mp[:jump], w)
    return mp, jump, c, kendall_trend(rolling_variance(res, w)), kendall_trend(rolling_ar1(res, w))


def _ews_null(seed, n=12000):
    f = _Population(FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=400, mu=0.02, seed=seed))
    f.run(1500, c_schedule=np.full(1500, 1.6))
    mp = f.run(n, c_schedule=np.full(n, 1.2))["mean_pos"]
    w = 400
    res = detrend(mp, w)
    return kendall_trend(rolling_variance(res, w)), kendall_trend(rolling_ar1(res, w))


# -- governance / entrainment (Fig 5) ----------------------------------------
def _make(seed):
    return _Population(FactoryParams(mu=0.05, c=0.85, norm_target=0.70, cost_center=0.30,
                                     cost_width=0.10, eta=4, M=600, seed=seed))


def _measure_T_inner():
    ests = []
    for seed in range(6):
        f = _make(seed)
        for _ in range(200):
            f.step(lam=0.0)
        tr = np.array([f.step(lam=12.0)["mean_pos"] for _ in range(500)])
        y0, yf = tr[:5].mean(), tr[-100:].mean()
        if abs(yf - y0) < 0.02:
            continue
        st = np.argmax(np.abs(tr - yf) < 0.10 * abs(yf - y0))
        if st > 0:
            ests.append(st)
    return float(np.median(ests)) if ests else 15.0


def _governed_instab(T_gov, seed, n=3000):
    f = _make(seed); pid = PID(Kp=6.0, Ki=0.4, Kd=2.0, hi=30); lam = 0.0
    nh = np.empty(n); lamh = np.empty(n)
    for t in range(n):
        o = f.step(lam=lam); nh[t] = o["norm_sat"]; lamh[t] = lam
        if (t + 1) % T_gov == 0:
            lam = pid.update(0.80 - nh[max(0, t - T_gov + 1):t + 1].mean(), dt=T_gov)
    return instability_index(lamh)


# ================================ the object ================================
class DarkFactory:
    def __init__(self, d=18, seed=0, **pop_kw):
        self.d = d
        self.seed = seed
        self._pop_kw = dict(_V3_BASE, mu=0.03, seed=seed); self._pop_kw.update(pop_kw)
        self._trace = None

    def run(self, n=60_000):
        self._trace = _Population(FactoryParams(**self._pop_kw)).run(n)["mean_pos"]
        return self

    # ---- LENS 1: opacity ---------------------------------------------------
    def lens_opacity(self, k=None, n_draws=30):
        d = self.d
        k = k if k is not None else max(2, d // 4)
        m, _ = premium(d, "complex", n_draws, np.random.default_rng(self.seed), lambda _d: k)
        return dict(d=d, k=k, cost_premium=float(m),
                    legible_fraction=sum(comb(d, j) for j in range(k + 1)) / (1 << d))

    # ---- LENS 2: stackelberg -----------------------------------------------
    def lens_stackelberg(self, T=20_000):
        return dict(V=stackelberg_value(3), U_star=0.5,
                    frontier=float(play(T, 0.0, 3, self.seed)[-1]),
                    core=float(play(T, 1.0, 3, self.seed)[-1]))

    # ---- LENS 3: versions --------------------------------------------------
    def lens_versions(self):
        if self._trace is None:
            self.run()
        Prev, _ = reversibilize(ulam_operator(self._trace, n_boxes=30)[0])
        _, _, taus = timescales(Prev)
        return dict(n_versions=int(n_metastable(Prev, 0.95)),
                    tau2=float(taus[1]), timescale_separation=float(taus[1] / taus[2]))

    # ---- LENS 4: pathologies -----------------------------------------------
    def lens_pathologies(self):
        mp, jump, c, vt, at = _ews_run(self.seed)
        return dict(fold_at=jump, variance_trend=float(vt), autocorr_trend=float(at))

    # ---- LENS 5: entrainment -----------------------------------------------
    @staticmethod
    def lens_entrainment(N=500, gamma=0.5, seed=0):
        rng = np.random.default_rng(seed)
        omega = gamma * np.tan(np.pi * (rng.random(N) - 0.5))
        omega = np.clip(omega - np.median(omega), -50, 50)
        Ks = np.linspace(0, 3, 16)
        r = np.array([simulate(N, K, omega, T=60, dt=0.02,
                               rng=np.random.default_rng(rng.integers(1 << 30)))[0] for K in Ks])
        Kc = critical_coupling_lorentzian(gamma)
        return dict(Kc=Kc, r_below=float(r[Ks < Kc - 0.3].mean()),
                    r_above=float(r[Ks > Kc + 0.5].mean()))

    # ======================= full per-figure data =======================
    def data_fig1(self):
        ds = [8, 10, 12, 14, 16, 18, 20]
        n_draws = 30
        kA = {"k = 2": lambda d: 2, "k = 3": lambda d: 3, "k = 4": lambda d: 4}
        fracs = {name: [sum(comb(d, j) for j in range(0, kf(d) + 1)) / (1 << d) for d in ds]
                 for name, kf in kA.items()}
        kbudget = lambda d: max(2, d // 4)
        premB = {}
        for kind, lab in [("complex", "complex objective"), ("linear", "linear control"),
                          ("planted", "planted-sparse control")]:
            means, errs = [], []
            for d in ds:
                m, e = premium(d, kind, n_draws, np.random.default_rng(100 + d), kbudget)
                means.append(m); errs.append(e)
            premB[lab] = (np.array(means), np.array(errs))
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
        return dict(ds=ds, fracs=fracs, premB=premB, dC=dC, ks=ks, taxC=taxC, fracC=fracC)

    def data_fig2(self):
        T, reps = 60_000, 6
        hedge_avg = np.array([play(T, 0.0, 3, seed=s) for s in range(reps)]).mean(0)
        swap_avg = np.array([play(T, 1.0, 3, seed=s) for s in range(reps)]).mean(0)
        lambdas = np.linspace(0, 1, 11)
        transp = np.array([[np.mean([play(20_000, lam, 3, seed=s)[-1] for s in range(reps)]),
                            np.std([play(20_000, lam, 3, seed=s)[-1] for s in range(reps)]) / np.sqrt(reps)]
                           for lam in lambdas])
        n3 = np.array([play(T, 0.0, 3, seed=s)[-1] for s in range(reps)])
        n2 = np.array([play(T, 0.0, 2, seed=s)[-1] for s in range(reps)])
        return dict(V=stackelberg_value(3), Ustar=0.5, hedge_avg=hedge_avg, swap_avg=swap_avg,
                    lambdas=lambdas, transp=transp, n3=n3, n2=n2,
                    V3=stackelberg_value(3), V2=stackelberg_value(2), reps=reps, T=T)

    def data_fig3(self):
        n_boxes = 30
        mp = _Population(FactoryParams(mu=0.03, seed=2, **_V3_BASE)).run(60_000)["mean_pos"]
        Pop, edges, occ = ulam_operator(mp, n_boxes=n_boxes)
        P_rev, pi = reversibilize(Pop)
        vals, vecs, taus = timescales(P_rev)
        labels, _ = almost_invariant_sets(P_rev, m=2)
        rhoA = coherence_ratio(Pop, labels == 0, pi)
        rhoB = coherence_ratio(Pop, labels == 1, pi)
        tsep = taus[1] / taus[2]
        nsw, dwells = debounced_switches(mp)
        dwell_cv = float(dwells.std() / dwells.mean()) if len(dwells) > 2 else float("nan")
        box_of = np.clip(np.digitize(mp, edges[1:-1]), 0, n_boxes - 1)
        pt_label = labels[box_of]
        mus = [0.015, 0.03, 0.05, 0.08, 0.12, 0.18]
        seps = []
        for mu in mus:
            Pq, _, _ = ulam_operator(_Population(FactoryParams(mu=mu, seed=4, **_V3_BASE)).run(60_000)["mean_pos"], n_boxes=n_boxes)
            Pqr, _ = reversibilize(Pq)
            _, _, tq = timescales(Pqr)
            seps.append(tq[1] / tq[2])
        return dict(mp=mp, pt_label=pt_label, taus=taus, tsep=tsep, mus=mus, seps=seps,
                    vals=vals, rhoA=rhoA, rhoB=rhoB, nsw=nsw, dwell_cv=dwell_cv,
                    n_meta_090=int(n_metastable(P_rev, 0.90)), n_meta_095=int(n_metastable(P_rev, 0.95)))

    def data_fig4(self):
        # Panel A
        regimes = {
            "healthy": FactoryParams(peakA=0.3, peakB=0.7, width=0.07, eta=4, M=300, mu=0.06,
                                     c=1.4, norm_target=0.7, metric_static=0.7, seed=1),
            "stable failure": FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=300, mu=0.05,
                                            c=0.45, norm_target=0.7, seed=1),
            "overfitting": FactoryParams(peakA=0.3, peakB=0.7, width=0.07, eta=4, M=300, mu=0.04,
                                         c=1.4, norm_target=0.5, metric_static=0.7, seed=1),
            "learning death": FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=300, mu=0.0008,
                                            c=0.45, norm_target=0.7, seed=1),
            "thrash": None,
        }
        rows, names = [], []
        for name, fp in regimes.items():
            if name == "thrash":
                n = 20000
                cosc = 1.0 + 0.6 * np.sin(2 * np.pi * np.arange(n) / 40)
                o = _Population(FactoryParams(peakA=0.3, peakB=0.7, width=0.08, eta=2, M=40,
                                              mu=0.2, norm_target=0.5, seed=1)).run(n, c_schedule=cosc)
            else:
                o = _Population(fp).run(20000)
            fp_ = fingerprint(o)
            rows.append([fp_["volatility"], fp_["variety"], fp_["norm_unmet"], fp_["proxy_gap"]])
            names.append(name)
        rows = np.array(rows)
        norm_cols = (rows - rows.min(0)) / (np.ptp(rows, axis=0) + 1e-12)
        # Panel B
        runs = [_ews_run(s) for s in range(5)]
        vtaus = [r[3] for r in runs]; ataus = [r[4] for r in runs]
        null_stats = [_ews_null(s) for s in range(5)]
        vnull = [v for v, a in null_stats]; anull = [a for v, a in null_stats]
        mp, jump, c = runs[3][0], runs[3][1], runs[3][2]
        w = 400
        res = detrend(mp[:jump], w)
        var = rolling_variance(res, w); ar1 = rolling_ar1(res, w)
        vt, at = float(np.mean(vtaus)), float(np.mean(ataus))
        vn, an = float(np.mean(vnull)), float(np.mean(anull))
        # Panel C
        Tnorm = 120
        Ps = [6, 12, 20, 30, 45, 60, 80, 110, 150]
        norm_s, met_s = [], []
        for Pp in Ps:
            o = _Population(FactoryParams(K=80, heightA=0.0, c=0.0, align_weight=4.0, align_width=0.05,
                                          eta=6, M=400, mu=0.02, norm_target=0.5, norm_amp=0.16,
                                          norm_freq=2 * np.pi / Tnorm, metric_sample_period=Pp, seed=1)).run(20000)
            norm_s.append(o["norm_sat"][-12000:].mean())
            met_s.append(o["metric_sat"][-12000:].mean())
        rates = 1.0 / np.array(Ps)
        nyq = 2.0 / Tnorm
        return dict(rows=rows, names=names, norm_cols=norm_cols,
                    mp=mp, jump=jump, var=var, ar1=ar1, vt=vt, at=at, vn=vn, an=an,
                    rates=rates, norm_s=norm_s, met_s=met_s, nyq=nyq)

    def data_fig5(self):
        # Panel A: cascade
        T_inner = _measure_T_inner()
        Tg = [3, 6, 12, 24, 48, 100, 200, 400]
        means, sems = [], []
        for t in Tg:
            vals = [_governed_instab(t, s) for s in range(8)]
            means.append(np.mean(vals)); sems.append(np.std(vals) / np.sqrt(8))
        ratios = list(np.array(Tg) / T_inner)
        # Panel B: kuramoto
        N = 500
        Ks = np.linspace(0, 3.0, 16)
        kseries = []
        rng = np.random.default_rng(0)
        for gamma, lab in [(0.35, "low diversity"), (0.85, "high diversity (jittered)")]:
            omega = gamma * np.tan(np.pi * (rng.random(N) - 0.5))
            omega = np.clip(omega - np.median(omega), -40, 40)
            rs = [simulate(N, K, omega, T=50, dt=0.02,
                           rng=np.random.default_rng(rng.integers(1 << 30)))[0] for K in Ks]
            kseries.append((lab, critical_coupling_lorentzian(gamma), rs))
        # Panel C: condensate
        Nc, dt, Tc = 400, 0.02, 80.0
        steps = int(Tc / dt)
        rng2 = np.random.default_rng(3)
        gamma = 0.3; K0 = 2.2
        omega0 = gamma * np.tan(np.pi * (rng2.random(Nc) - 0.5))
        omega0 = np.clip(omega0 - np.median(omega0), -30, 30)
        at = int(steps * 0.5)

        def integrate(mode):
            theta = rng2.uniform(0, 2 * np.pi, Nc).copy()
            omega = omega0.copy()
            r_t = np.empty(steps); K = K0
            for sidx in range(steps):
                z = np.mean(np.exp(1j * theta)); r = np.abs(z); psi = np.angle(z)
                r_t[sidx] = r
                theta = theta + dt * (omega + K * r * np.sin(psi - theta))
                if sidx == at:
                    if mode == "halt":
                        theta = theta + 8.0 * rng2.standard_normal(Nc)
                    elif mode == "decouple":
                        K = 0.3
            return r_t
        r_lock = integrate("none"); r_halt = integrate("halt"); r_dec = integrate("decouple")
        return dict(ratios=ratios, cas_mean=means, cas_sem=sems, T_inner=float(T_inner),
                    Ks=Ks.tolist(), kseries=kseries, cond_t=np.arange(steps) * dt,
                    r_lock=r_lock, r_halt=r_halt, r_dec=r_dec)


if __name__ == "__main__":
    f = DarkFactory(seed=2).run()
    print("ONE dark factory, five lenses (headline numbers):\n")
    print("  opacity     ", f.lens_opacity())
    print("  stackelberg ", f.lens_stackelberg())
    print("  versions    ", f.lens_versions())
    print("  pathologies ", f.lens_pathologies())
    print("  entrainment ", DarkFactory.lens_entrainment())
