"""`simulate(design)` -- run a factory design on the dynamical organs and return a
`FactoryReport`.

The work is done by the validated models in `src/`: a finite population of
learning agents (`factory.DarkFactory`), a transfer operator for versioning
(`transfer_operator`), early-warning signals (`ews`), a Kuramoto coupling for the
ecology (`kuramoto`), and a PID controller for governance (`control`). This module
runs them across a seed ensemble and assembles the lenses; it does not re-derive
any of the mathematics.

Every scalar that comes off a stochastic run is reported as (median, spread) over
the ensemble, never as a single sample.
"""
from __future__ import annotations

import numpy as np

from factory import DarkFactory, FactoryParams
from transfer_operator import ulam_operator, reversibilize, spectrum, n_metastable, coherence_timescale
from ews import detrend, rolling_variance, rolling_ar1, kendall_trend, ensemble_disagreement
from kuramoto import simulate as _kuramoto, critical_coupling_lorentzian
from control import PID, instability_index

from .design import FactoryDesign
from .report import FactoryReport, Lens
from . import pathology

# price-oscillation (instability_index of lambda) above which governance is thrashing;
# well-timed control sits near ~0.9, fast-repricing iatrogenic thrash near ~1.8.
_PRICE_INSTAB_THRESH = 1.3


# --- ensemble helpers -------------------------------------------------------
def _med_spread(xs) -> tuple[float, float]:
    """(median, half-IQR) over an ensemble -- a robust centre and spread."""
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    med = float(np.median(a))
    spread = float((np.percentile(a, 75) - np.percentile(a, 25)) / 2.0) if a.size > 1 else 0.0
    return (med, spread)


def _run(design: FactoryDesign, seed: int, n: int) -> dict:
    """One ungoverned population run -> its behavioural observables."""
    return DarkFactory(design.to_params(seed)).run(n)


# --- the public entry point -------------------------------------------------
def simulate(design: FactoryDesign, seeds: int = 12, quick: bool = False,
             early_warning: bool = True) -> FactoryReport:
    """Simulate `design` and return its report.

    Parameters
    ----------
    seeds : ensemble size; every reported scalar is a median over this many runs.
    quick : shorter runs / fewer seeds, for tests and fast iteration.
    early_warning : also probe proximity to a fold (a second, ramped run).
    """
    if quick:
        seeds = min(seeds, 4)
    n_pop = 6000 if quick else 15000
    tail = 2000 if quick else 4000

    runs = [_run(design, s, n_pop) for s in range(seeds)]

    lenses: dict[str, Lens] = {}
    verdict, verdict_detail = _pathology_lens(runs, tail, lenses)
    _versions_lens(runs, quick, lenses)
    if early_warning and not quick:
        _early_warning_lens(design, lenses)
    _governance_lens(design, quick, lenses)
    _ecology_lens(design, lenses)
    _steering_note(design, lenses)

    provenance = dict(seeds=seeds, run_len=n_pop, tail=tail,
                      organs="src/ (exact)", measures="toy model, not your factory")
    return FactoryReport(design=design, verdict=verdict, verdict_detail=verdict_detail,
                         lenses=lenses, provenance=provenance)


# --- lenses -----------------------------------------------------------------
_VERDICT_LINES = {
    "healthy": "the population converges on the goal and keeps exploring -- no failure mode dominates.",
    "stable_failure": "the population settles into a stable behaviour that misses the goal and resists leaving it.",
    "overfitting": "the population scores well on the metric while missing the true goal -- it is gaming the eval.",
    "learning_death": "exploration has collapsed; the population is stuck on one assembly and no longer searches.",
    "thrash": "the population never commits -- it discovers behaviours and abandons them, shipping nothing reliably.",
}


def _pathology_lens(runs, tail, lenses) -> tuple[str, str]:
    fps = [pathology.fingerprint(r["mean_pos"], r["variety"], r["norm_sat"], r["metric_sat"], tail)
           for r in runs]
    agg = pathology.aggregate(fps)
    diag = pathology.classify(agg)
    metrics = {c: round(s, 2) for c, s in sorted(diag.scores.items(), key=lambda kv: -kv[1])}
    # plain-English names for the four features in the displayed fingerprint
    _names = dict(settledness="settledness", variety="variety",
                  norm_unmet="distance_from_goal", proxy_gap="metric_minus_truth")
    metrics["fingerprint"] = {_names[f]: round(agg[f], 2) for f in pathology.FEATURES}
    # the scores already appear once in `metrics`; keep the headline to the label
    summary = diag.label
    if diag.ambiguous:
        summary += f"  (ambiguous vs {diag.runner_up}; read both)"
    lenses["pathology"] = Lens(
        name="pathology", summary=summary, metrics=metrics,
        maps_to="the post-mortem you would otherwise write by hand after the build",
        limits="scores are nearest-prototype over a seed-median fingerprint; an 'ambiguous' "
               "flag means the run sits between two failure modes.")
    return diag.label, _VERDICT_LINES.get(diag.label, "")


def _versions_lens(runs, quick, lenses):
    boxes = (20, 30) if quick else (20, 30, 40)
    per_box_nv = {nb: [] for nb in boxes}    # box count -> n_versions across seeds
    per_box_l2 = {nb: [] for nb in boxes}    # box count -> second eigenvalue across seeds
    for r in runs:
        trace = r["mean_pos"]
        for nb in boxes:
            P, _, occ = ulam_operator(trace, n_boxes=nb)
            if (occ > 0).sum() < 3:          # behaviour concentrated in <3 boxes: gap meaningless
                continue
            Prev, _ = reversibilize(P)
            vals, _ = spectrum(Prev)
            per_box_nv[nb].append(n_metastable(Prev, 0.9))
            per_box_l2[nb].append(float(vals[1]))
    nv_by_box = {nb: int(round(_med_spread(v)[0])) for nb, v in per_box_nv.items() if v}
    all_nv = [x for v in per_box_nv.values() for x in v]
    all_l2 = [x for v in per_box_l2.values() for x in v]
    nv = _med_spread(all_nv)
    l2_med = _med_spread(all_l2)[0]
    n = int(round(nv[0]))
    # the count is robust only if it agrees across bin granularities; surface the
    # per-bin counts in the headline ONLY when they disagree (otherwise it is noise).
    bins_agree = len(set(nv_by_box.values())) <= 1
    if n >= 2:
        life = coherence_timescale(l2_med)   # lambda2 near 1: dwell = -1/ln(lambda2)
        summary = (f"{n} distinct stable behaviours; the factory dwells in one for "
                   f"~{life:.0f} rounds before it can spontaneously flip to another")
        metrics = dict(n_versions=nv, dwell_rounds=round(life, 1))
    else:
        stickiness = round(1.0 - l2_med, 2)
        summary = (f"one stable behaviour, stickiness {stickiness} "
                   f"(1 = locked in place, 0 = drifts freely)")
        metrics = dict(n_versions=nv, stickiness=stickiness)
    if not bins_agree:
        summary += "  (count varies with bin granularity -- treat as approximate)"
        metrics["n_versions_by_box"] = nv_by_box
    lenses["versions"] = Lens(
        name="versions", summary=summary, metrics=metrics,
        detail=dict(n_versions_by_box=nv_by_box),
        maps_to="how many stable behaviours your factory will settle into, and how durable each is",
        limits="estimated from how the average behaviour moves between bins; it tracks the mean, "
               "not the full distribution, and the count can shift with bin granularity "
               "(full per-bin counts are in the lens detail).")


def _phase_randomize(x: np.ndarray, rng) -> np.ndarray:
    """A surrogate with the same power spectrum but randomised phases: same
    autocorrelation, no trend -- the proper null for a critical-slowing-down test."""
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, X.shape)
    ph[0] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def _early_warning_lens(design: FactoryDesign, lenses, n=12000, seeds=4, w=400):
    """Ramp the goal payoff down toward a tipping point and test whether rising
    variance and autocorrelation flag it -- against a stable baseline AND a
    phase-randomised surrogate test on the strongest run."""
    def ramp(seed, fold: bool):
        f = DarkFactory(design.to_params(seed))
        # warm up at the current payoff, then ramp down (toward the tip) or hold flat.
        f.run(1500, c_schedule=np.full(1500, design.goal_payoff))
        c = np.linspace(design.goal_payoff, 0.3, n) if fold else np.full(n, design.goal_payoff)
        mp = f.run(n, c_schedule=c)["mean_pos"]
        jump = max(int(np.argmin(np.diff(mp))), 800) if fold else n
        res = detrend(mp[:jump], w)
        return mp[:jump], res, kendall_trend(rolling_variance(res, w)), kendall_trend(rolling_ar1(res, w))

    fold = [ramp(s, True) for s in range(seeds)]
    null = [ramp(s, False) for s in range(seeds)]
    var_fold = _med_spread([f[2] for f in fold]); ar_fold = _med_spread([f[3] for f in fold])
    var_null = _med_spread([nr[2] for nr in null])

    # surrogate test on the run nearest the median variance trend
    med_idx = int(np.argsort([f[2] for f in fold])[len(fold) // 2])
    res = fold[med_idx][1]
    real_tau = kendall_trend(rolling_variance(res, w))
    rng = np.random.default_rng(0)
    surr = [kendall_trend(rolling_variance(_phase_randomize(res, rng), w)) for _ in range(40)]
    surr_p = float(np.mean([s >= real_tau for s in surr]))   # fraction of surrogates as strong

    # ensemble disagreement: spread across the fold traces, late vs early
    L = min(len(f[0]) for f in fold)
    dis = ensemble_disagreement(np.vstack([f[0][:L] for f in fold]))
    rising = float(np.nanmean(dis[-L // 4:]) - np.nanmean(dis[:L // 4]))

    fires = (var_fold[0] > 0.4 and ar_fold[0] > 0.4
             and var_fold[0] > var_null[0] + 0.2 and surr_p < 0.05)
    lenses["early_warning"] = Lens(
        name="early_warning",
        summary=("rising variance + autocorrelation DO flag the approaching tipping point "
                 f"(beats {1 - surr_p:.0%} of surrogates)" if fires
                 else "no early-warning trend above the stable baseline and surrogate test"),
        metrics=dict(variance_trend=var_fold, autocorr_trend=ar_fold,
                     variance_trend_baseline=var_null, surrogate_p=round(surr_p, 3),
                     disagreement_rise=round(rising, 3)),
        maps_to="the live signals to monitor in production: variance and lag-1 autocorrelation "
                "of a behavioural metric, plus disagreement across heterogeneous judges",
        limits="leading indicators with documented false positives; a trend that beats the "
               "baseline and the surrogate test suggests proximity to a tipping point, but "
               "absence of a trend does NOT imply safety.")


def _governance_lens(design: FactoryDesign, quick, lenses):
    gains = design.pid_gains()
    if gains is None:
        lenses["governance"] = Lens(
            name="governance",
            summary="no governance controller -- a stable failure will not self-correct",
            metrics={},
            maps_to="your automated repricing / penalty loop (currently off)",
            limits="with the controller off, the only lenses that move are population-level; "
                   "turn it on to price constraint violation.")
        return
    n = 1500 if quick else 3000
    seeds = 3 if quick else 6
    t_med, t_spread, estimated = _settling_time(design, seeds=seeds)
    instab, finals = [], []
    for s in range(seeds):
        lam_trace, final = _governed(design, gains, n, s)
        instab.append(instability_index(lam_trace))
        finals.append(final)
    price_instab = _med_spread(instab)
    rp = design.repricing_period
    # the robust governance-health signal is the price oscillation actually measured;
    # the cascade ratio is a secondary heuristic carrying the settling-time uncertainty.
    oscillating = price_instab[0] > _PRICE_INSTAB_THRESH
    if estimated:
        s = max(t_spread, 0.0) if np.isfinite(t_spread) else 0.0
        ratio_band = (rp / (t_med + s), rp / max(t_med - s, 1e-9))   # low, high
        ratio = rp / t_med
        timing_ok = ratio_band[0] >= 3.0                            # stable only if the whole band clears 3:1
        timing = (f"repricing every {rp} rounds vs ~{t_med:.0f}-round settling -> cascade ratio "
                  f"{ratio:.1f} (band {ratio_band[0]:.1f}-{ratio_band[1]:.1f}; "
                  f"{'clears 3:1' if timing_ok else 'below 3:1 -> iatrogenic thrash risk'})")
        ratio_metric = (round(ratio, 1), round((ratio_band[1] - ratio_band[0]) / 2, 1))
    else:
        timing_ok = None
        timing = (f"repricing every {rp} rounds; settling time could not be measured (the price "
                  f"step moved the factory too little) -- cascade ratio is indicative only")
        ratio_metric = None
    healthy_gov = (not oscillating) and (timing_ok is not False)
    metrics = dict(price_instability=price_instab, final_satisfaction=_med_spread(finals),
                   settling_estimated=estimated)
    if ratio_metric is not None:
        metrics["cascade_ratio"] = ratio_metric
    lenses["governance"] = Lens(
        name="governance",
        summary=(f"{'stable control loop' if healthy_gov else 'CONTROL LOOP OSCILLATING (iatrogenic thrash)'}; "
                 f"{timing}; price instability {price_instab[0]:.2f}"),
        metrics=metrics,
        maps_to="how often your eval-to-policy loop reprices, relative to how fast the factory settles",
        limits="settling time is a noisy estimate; the headline signal is the measured price "
               "oscillation. Govern 3-10x slower than the inner loop or the controller forces the "
               "oscillation it is trying to damp.")


def _ecology_lens(design: FactoryDesign, lenses):
    if design.peer_factories <= 0:
        return
    N = max(50, int(design.peer_factories))
    gamma = max(0.05, float(design.dependency_diversity))   # frequency spread = diversity
    K = float(design.shared_dependency)                     # coupling through shared infra
    rng = np.random.default_rng(0)
    omega = gamma * np.tan(np.pi * (rng.random(N) - 0.5))
    omega = np.clip(omega - np.median(omega), -40, 40)
    r, _, _ = _kuramoto(N, K, omega, T=50, dt=0.02, rng=np.random.default_rng(1))
    Kc = critical_coupling_lorentzian(gamma)
    locked = K > Kc and r > 0.5
    lenses["ecology"] = Lens(
        name="ecology",
        summary=(f"peers move in lockstep (sync r={r:.2f}) -> exposed to a correlated crash"
                 if locked else f"peers stay desynchronised (sync r={r:.2f}) -> no correlated-crash exposure"),
        metrics=dict(synchronisation=round(r, 2), coupling=round(K, 2),
                     onset_threshold=round(Kc, 2)),
        maps_to="when many factories share a base model / control plane, they can crash together; "
                "diversifying dependencies (raising diversity) raises the threshold",
        limits="the crash threshold is exact only for the modelled dependency spread and a large "
               "peer group; it is a regime indicator, not a guaranteed safe line.")


def _steering_note(design: FactoryDesign, lenses):
    """Qualitative only: a population with no protected exploration can only reach
    what was committed at design time. No number -- see `reference.py`."""
    if design.effective_explore < 0.01:
        lenses["steering"] = Lens(
            name="steering",
            summary="with almost no exploration the factory can only reach what you designed in; "
                    "leave room to explore (see `factory-design reference`)",
            metrics=dict(effective_explore=round(design.effective_explore, 4)),
            maps_to="reserve some non-greedy exploratory capacity",
            limits="qualitative; the size of the gain is a property of the game, not this design.")


# --- governance helpers -----------------------------------------------------
def _governed(design: FactoryDesign, gains: dict, n: int, seed: int):
    """Run the factory with an outer PID governance loop; return (lambda trace, final sat)."""
    f = DarkFactory(design.to_params(seed))
    pid = PID(**gains)
    lam = 0.0
    ns = np.empty(n); lamh = np.empty(n)
    T = design.repricing_period
    for t in range(n):
        o = f.step(lam=lam)
        ns[t] = o["norm_sat"]; lamh[t] = lam
        if (t + 1) % T == 0:
            recent = ns[max(0, t - T + 1):t + 1].mean()
            lam = pid.update(design.target_satisfaction - recent, dt=T)
    return lamh, float(ns[-500:].mean())


_NOMINAL_SETTLING = 15.0   # fallback inner-loop period when a price step produces no move


def _settling_time(design: FactoryDesign, seeds: int = 6) -> tuple[float, float, bool]:
    """Estimate the inner loop's settling time: apply a price step and measure how
    long the behaviour takes to reach 90% of its eventual move.

    Returns (median, spread, estimated). `estimated` is False when no seed's price
    step moved the behaviour enough to time -- the caller must then treat the
    cascade ratio as indicative, not measured.
    """
    ests = []
    for s in range(seeds):
        f = DarkFactory(design.to_params(s))
        for _ in range(200):
            f.step(lam=0.0)
        tr = np.array([f.step(lam=12.0)["mean_pos"] for _ in range(500)])
        y0, yf = tr[:5].mean(), tr[-100:].mean()
        if abs(yf - y0) < 0.02:
            continue
        st = int(np.argmax(np.abs(tr - yf) < 0.10 * abs(yf - y0)))
        if st > 0:
            ests.append(st)
    if not ests:
        return (_NOMINAL_SETTLING, float("nan"), False)
    med, spread = _med_spread(ests)
    return (med, spread, True)
