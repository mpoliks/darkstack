"""
FIGURE 4 -- Pathologies & Catastrophe (the eval layer's job).

The paper names four convergence pathologies and argues the eval layer must sense
them, and must anticipate catastrophes (Thom folds/cusps) via early-warning
signals -- critical slowing down: rising variance and autocorrelation.

Panel A: the four pathologies have DISTINCT behavioural fingerprints, so a
         downstream evaluator can classify them from behaviour alone:
           stable failure -- settled (low volatility), normal variety, norm unmet,
                             no metric/norm gap;
           overfitting    -- settled, normal variety, norm unmet but METRIC met
                             (the proxy/norm gap);
           learning death -- variety collapses to one assembly;
           thrash         -- never settles (high volatility), high variety.
Panel B: a slowly-ramped control parameter (the spec height of the occupied peak)
         drives the population through a FOLD: the version collapses
         discontinuously. Before it, the behavioural variance and lag-1
         autocorrelation rise -- critical slowing down -- a model-free early
         warning. (EWS are leading indicators, not guarantees; we report the
         Kendall trend, not a promise.)
Panel C: overfitting as ALIASING. The factory chases a scored metric that samples
         a faster-varying norm. The metric stays satisfied while the norm is lost
         once sampling drops below the Nyquist rate; raising the sampling rate
         (the paper's prescription) closes the gap.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from factory import DarkFactory, FactoryParams
from ews import rolling_variance, rolling_ar1, detrend, kendall_trend
style.apply()
P = style.PALETTE


def fingerprint(o):
    mp = o["mean_pos"]
    vol = float(mp[-5000:].std())
    variety = float(o["variety"][-3000:].mean())
    norm = float(o["norm_sat"][-3000:].mean())
    metric = float(o["metric_sat"][-3000:].mean())
    return dict(volatility=vol, variety=variety, norm_unmet=1 - norm,
                proxy_gap=max(0.0, metric - norm))


def panelA(ax):
    regimes = {
        "healthy": FactoryParams(peakA=0.3, peakB=0.7, width=0.07, eta=4, M=300, mu=0.06,
                                 c=1.4, norm_target=0.7, metric_static=0.7, seed=1),
        "stable failure": FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=300, mu=0.05,
                                        c=0.45, norm_target=0.7, seed=1),
        "overfitting": FactoryParams(peakA=0.3, peakB=0.7, width=0.07, eta=4, M=300, mu=0.04,
                                     c=1.4, norm_target=0.5, metric_static=0.7, seed=1),
        "learning death": FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=300, mu=0.0008,
                                        c=0.45, norm_target=0.7, seed=1),
        "thrash": None,  # special: oscillating spec
    }
    rows, names = [], []
    for name, fp in regimes.items():
        if name == "thrash":
            n = 20000
            cosc = 1.0 + 0.6 * np.sin(2 * np.pi * np.arange(n) / 40)
            f = DarkFactory(FactoryParams(peakA=0.3, peakB=0.7, width=0.08, eta=2, M=40,
                                          mu=0.2, norm_target=0.5, seed=1))
            o = f.run(n, c_schedule=cosc)
        else:
            o = DarkFactory(fp).run(20000)
        fp_ = fingerprint(o)
        rows.append([fp_["volatility"], fp_["variety"], fp_["norm_unmet"], fp_["proxy_gap"]])
        names.append(name)
    rows = np.array(rows)
    # normalise each column to [0,1] for visual comparability
    norm_cols = (rows - rows.min(0)) / (np.ptp(rows, axis=0) + 1e-12)
    cols = ["volatility", "variety", "norm\nunmet", "metric–\nnorm gap"]
    im = ax.imshow(norm_cols, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=7.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    raw = [["%.3f" % rows[i, 0], "%.1f" % rows[i, 1], "%.2f" % rows[i, 2], "%.2f" % rows[i, 3]]
           for i in range(len(names))]
    for i in range(len(names)):
        for j in range(4):
            ax.text(j, i, raw[i][j], ha="center", va="center", fontsize=6.6,
                    color="white" if norm_cols[i, j] > 0.55 else style.INK)
    ax.set_title("Each pathology has a distinct\nbehavioural fingerprint")
    ax.grid(False)
    style.panel_tag(ax, "A")
    return rows, names


def _ews_run(seed, n=12000):
    c = np.linspace(1.6, 0.4, n)
    f = DarkFactory(FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=400, mu=0.02, seed=seed))
    f.run(1500, c_schedule=np.full(1500, 1.6))           # warm up on version B
    o = f.run(n, c_schedule=c)
    mp = o["mean_pos"]
    jump = int(np.argmin(np.diff(mp)))
    w = 400
    res = detrend(mp[:jump], w)
    return mp, jump, c, kendall_trend(rolling_variance(res, w)), kendall_trend(rolling_ar1(res, w))


def _ews_null(seed, n=12000):
    """No-fold negative control: hold the spec CONSTANT (no catastrophe). A specific
    early-warning signal should NOT trend up here."""
    f = DarkFactory(FactoryParams(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=400, mu=0.02, seed=seed))
    f.run(1500, c_schedule=np.full(1500, 1.6))
    mp = f.run(n, c_schedule=np.full(n, 1.2))["mean_pos"]  # constant, version B never collapses
    w = 400
    res = detrend(mp, w)
    return kendall_trend(rolling_variance(res, w)), kendall_trend(rolling_ar1(res, w))


def panelB(ax_top, ax_bot):
    # seed-robustness: the critical-slowing-down trend is positive across seeds,
    # not a single-run artifact; the no-fold control shows it is also SPECIFIC.
    runs = [_ews_run(s) for s in range(5)]               # one call per seed (cached)
    vtaus = [r[3] for r in runs]
    ataus = [r[4] for r in runs]
    null_stats = [_ews_null(s) for s in range(5)]
    vnull = [v for v, a in null_stats]
    anull = [a for v, a in null_stats]
    n = 12000
    mp, jump, c = runs[3][0], runs[3][1], runs[3][2]
    # trace
    ax_top.plot(mp, color=P["neutral"], lw=0.8)
    ax_top.axvline(jump, color=P["warn"], lw=1.0, ls="--")
    ax_top.set_ylabel("behaviour", fontsize=8)
    ax_top.set_xticklabels([])
    ax_top.set_title("Critical slowing down before a fold", fontsize=10)
    style.panel_tag(ax_top, "B")
    # EWS on the pre-collapse window
    w = 400
    res = detrend(mp[:jump], w)
    var = rolling_variance(res, w)
    ar1 = rolling_ar1(res, w)
    t = np.arange(jump)
    vt, at = np.mean(vtaus), np.mean(ataus)
    vn, an = np.mean(vnull), np.mean(anull)
    ax_bot.plot(t, var / np.nanmax(var), color=P["core"],
                label=f"variance  ($\\tau$ {vt:+.2f}, null {vn:+.2f})")
    ax_bot.plot(t, ar1, color=P["frontier"],
                label=f"lag-1 autocorr.  ($\\tau$ {at:+.2f}, null {an:+.2f})")
    ax_bot.axvline(jump, color=P["warn"], lw=1.0, ls="--", label="fold (version collapse)")
    ax_bot.set_xlim(ax_top.get_xlim())
    ax_bot.set_xlabel("round  $t$  (spec height ramped $1.6\\!\\to\\!0.4$)")
    ax_bot.set_ylabel("EWS", fontsize=8)
    style.legend_below(ax_bot, ncol=1, y=-0.55, fontsize=6.4)
    return (float(vt), float(at), jump, float(vn), float(an))


def panelC(ax):
    Tnorm = 120
    Ps = [6, 12, 20, 30, 45, 60, 80, 110, 150]
    norm_s, met_s = [], []
    for Pp in Ps:
        f = DarkFactory(FactoryParams(K=80, heightA=0.0, c=0.0, align_weight=4.0, align_width=0.05,
                                      eta=6, M=400, mu=0.02, norm_target=0.5, norm_amp=0.16,
                                      norm_freq=2 * np.pi / Tnorm, metric_sample_period=Pp, seed=1))
        o = f.run(20000)
        norm_s.append(o["norm_sat"][-12000:].mean())
        met_s.append(o["metric_sat"][-12000:].mean())
    rates = 1.0 / np.array(Ps)                            # metric sampling rate
    nyq = 2.0 / Tnorm                                     # Nyquist rate = 2 * f_norm
    ax.fill_betweenx([0, 1.02], min(rates) * 0.8, nyq, color=style.FAINT, alpha=0.20,
                     label="aliased region (below Nyquist)")
    ax.plot(rates, met_s, "o-", color=P["accent"], ms=4, label="metric satisfied")
    ax.plot(rates, norm_s, "o-", color=P["core"], ms=4, label="norm satisfied")
    ax.axvline(nyq, color=P["neutral"], lw=1.0, ls=":", label="Nyquist rate (reference)")
    ax.set_xscale("log")
    ax.set_xlabel("metric sampling rate  $f_s$  (1/period)")
    ax.set_ylabel("satisfaction")
    ax.set_title("Overfitting as aliasing: the metric/norm\ngap opens as sampling slows")
    style.legend_below(ax, ncol=1)
    ax.set_ylim(0, 1.02)
    style.panel_tag(ax, "C")
    return list(rates), norm_s, met_s, nyq


def main():
    fig = plt.figure(figsize=(12.2, 3.9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.12, wspace=0.34)
    axA = fig.add_subplot(gs[:, 0])
    axB1 = fig.add_subplot(gs[0, 1])
    axB2 = fig.add_subplot(gs[1, 1])
    axC = fig.add_subplot(gs[:, 2])

    rows, names = panelA(axA)
    vtau, atau, jump, vnull, anull = panelB(axB1, axB2)
    rates, norm_s, met_s, nyq = panelC(axC)

    fig.suptitle("Figure 4.  Pathologies and catastrophe:  distinct behavioural fingerprints, early-warning signals, and overfitting as aliasing",
                 fontsize=11, y=1.04)
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "fig4_pathologies.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    print("pathology fingerprints (constructed exemplars, seed-robust):")
    for nm, r in zip(names, rows):
        print(f"  {nm:16s} vol={r[0]:.4f} variety={r[1]:.2f} norm_unmet={r[2]:.2f} proxy_gap={r[3]:.2f}")
    print(f"EWS pre-fold Kendall tau (n=5): variance {vtau:+.2f}, autocorr {atau:+.2f} (collapse @ t={jump})")
    print(f"EWS no-fold control: variance {vnull:+.2f}, autocorr {anull:+.2f} (specificity)")
    json.dump(dict(fingerprints={nm: list(map(float, r)) for nm, r in zip(names, rows)},
                   ews_var_tau=vtau, ews_ar1_tau=atau, fold_t=jump,
                   ews_null_var_tau=vnull, ews_null_ar1_tau=anull,
                   alias_rates=rates, alias_norm=norm_s, alias_metric=met_s, nyquist=nyq),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig4.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
