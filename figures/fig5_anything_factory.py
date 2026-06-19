"""
FIGURE 5 -- Governance timing and the Anything Factory.

Three of the paper's load-bearing temporal claims:

Panel A: iatrogenic thrash / cascade control. Governance (the outer loop) commands
         a factory that runs its own fast inner control loop. If governance revises
         near the inner loop's settling rate, it corrects against unfinished
         transients and the nested loops oscillate. Enforcing the cascade ratio
         (inner settles >= ~3x faster than the outer loop revises) restores
         stability -- the paper's 3:1..10:1 rule. We sweep the ratio and measure
         the slow (governance-scale) oscillation of the factory's behaviour.

Panel B: entrainment. Each factory's slowest loop is a phase oscillator; a shared
         dependency couples them (Kuramoto). Below a critical coupling Kc the
         population stays desynchronised; above it a phase-locked condensate
         forms. Dependency DIVERSITY (wider natural-frequency spread = jitter)
         raises Kc -- diversification is a real defence. Kc = 2*gamma (Lorentzian)
         is overplotted.

Panel C: the phase-locked condensate and its circuit-breaker. A strongly coupled,
         low-diversity population locks (order parameter r -> 1): a 2010-flash-
         crash-style correlated cascade. A one-shot jitter injection (the "Stop
         Logic" desynchroniser) collapses r and breaks the lock. (That synchrony
         implies systemic *collapse* is an interpretation of the dynamics, not a
         Kuramoto theorem -- flagged as such.)
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style  # noqa
from factory import DarkFactory, FactoryParams
from control import PID, instability_index
from kuramoto import simulate, critical_coupling_lorentzian
style.apply()
P = style.PALETTE


# ----------------------------- Panel A ----------------------------------- #
# Governance is a single PID loop that reprices lambda every T_gov rounds from a
# windowed estimate of the factory's behaviour. The factory's behavioural response
# to a lambda change has an inner response time T_inner. When governance samples
# faster than the factory settles (T_gov < ~few * T_inner), it reprices against
# unsettled transients and lambda oscillates -- iatrogenic thrash. This is a
# sampled-data / phase-lag instability (the inner loop is an overdamped lag, not a
# 2nd-order resonator, so there is no resonance peak); the cure is the cascade
# heuristic, governing several times slower than the inner loop.
def _make(seed):
    return DarkFactory(FactoryParams(mu=0.05, c=0.85, norm_target=0.70, cost_center=0.30,
                                     cost_width=0.10, eta=4, M=600, seed=seed))


def _measure_T_inner():
    """Characteristic inner-loop response time: the 10%-settling time of the mean
    behavioural coordinate to a fixed lambda step, measured from several seeds and
    several warm-up lambdas and reported as the MEDIAN (single-seed estimates are
    noisy -- the controlled variable can start near saturation -- so T_inner is an
    order-of-magnitude scale, not a tightly-determined constant)."""
    ests = []
    for seed in range(6):
        f = _make(seed)
        for _ in range(200):
            f.step(lam=0.0)
        tr = np.array([f.step(lam=12.0)["mean_pos"] for _ in range(500)])
        y0, yf = tr[:5].mean(), tr[-100:].mean()
        if abs(yf - y0) < 0.02:                  # step barely moved it: uninformative
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


def cascade_panel(ax):
    T_inner = _measure_T_inner()
    Tg = [3, 6, 12, 24, 48, 100, 200, 400]
    seeds = range(8)
    means, sems = [], []
    for t in Tg:
        vals = [_governed_instab(t, s) for s in seeds]
        means.append(np.mean(vals)); sems.append(np.std(vals) / np.sqrt(len(vals)))
    ratios = np.array(Tg) / T_inner
    ax.axvspan(3, 10, color=P["green"], alpha=0.12, label="cascade band (3:1–10:1)")
    ax.errorbar(ratios, means, yerr=sems, fmt="o-", color=P["neutral"], ms=4, capsize=2,
                label="governance instability (8 seeds)")
    ax.set_xscale("log")
    ax.set_xlabel("cascade ratio  $T_\\mathrm{gov}/T_\\mathrm{inner}$   ($T_\\mathrm{inner}$" +
                  f"$\\approx${T_inner:.0f})")
    ax.set_ylabel("governance instability  (oscillation of $\\lambda$)")
    ax.set_title("Mistimed governance thrashes;\nslower governance settles")
    style.legend_below(ax, ncol=1)
    style.panel_tag(ax, "A")
    return list(ratios), means, sems, float(T_inner)


# ----------------------------- Panel B ----------------------------------- #
def kuramoto_panel(ax):
    N = 500
    Ks = np.linspace(0, 3.0, 16)
    out = {}
    rng = np.random.default_rng(0)
    for gamma, lab, col in [(0.35, "low diversity", P["core"]),
                            (0.85, "high diversity (jittered)", P["frontier"])]:
        omega = gamma * np.tan(np.pi * (rng.random(N) - 0.5))
        omega = np.clip(omega - np.median(omega), -40, 40)
        rs = [simulate(N, K, omega, T=50, dt=0.02,
                       rng=np.random.default_rng(rng.integers(1 << 30)))[0] for K in Ks]
        out[lab] = rs
        Kc = critical_coupling_lorentzian(gamma)
        ax.plot(Ks, rs, "o-", color=col, ms=3.5, label=f"{lab} ($K_c$={Kc:.1f})")
        ax.axvline(Kc, color=col, ls=":", lw=1.0)
    ax.set_xlabel("shared-dependency coupling  $K$")
    ax.set_ylabel("synchronisation  $r$  (order parameter)")
    ax.set_title("Diverse dependencies make\nentrainment harder")
    style.legend_below(ax, ncol=1)
    ax.set_ylim(0, 1)
    style.panel_tag(ax, "B")
    return Ks.tolist(), out


# ----------------------------- Panel C ----------------------------------- #
def condensate_panel(ax):
    N, dt, T = 400, 0.02, 80.0
    steps = int(T / dt)
    rng = np.random.default_rng(3)
    gamma = 0.3                                   # low diversity -> prone to lock
    K0 = 2.2                                        # well above Kc = 0.6
    omega0 = gamma * np.tan(np.pi * (rng.random(N) - 0.5))
    omega0 = np.clip(omega0 - np.median(omega0), -30, 30)
    at = int(steps * 0.5)

    def integrate(mode):
        theta = rng.uniform(0, 2 * np.pi, N).copy()
        omega = omega0.copy()
        r_t = np.empty(steps); K = K0
        for sidx in range(steps):
            z = np.mean(np.exp(1j * theta)); r = np.abs(z); psi = np.angle(z)
            r_t[sidx] = r
            theta = theta + dt * (omega + K * r * np.sin(psi - theta))
            if sidx == at:
                if mode == "halt":            # one-shot phase scramble (a trading halt)
                    theta = theta + 8.0 * rng.standard_normal(N)
                elif mode == "decouple":      # diversify dependencies: drop K below Kc
                    K = 0.3                    # < Kc = 2*gamma = 0.6
        return r_t

    t = np.arange(steps) * dt
    r_lock = integrate("none")
    r_halt = integrate("halt")
    r_dec = integrate("decouple")
    ax.plot(t, r_lock, color=P["core"], label="entrained condensate")
    ax.plot(t, r_halt, color=P["accent"], lw=1.3, label="one-shot halt (re-locks)")
    ax.plot(t, r_dec, color=P["frontier"], label="diversify dependencies (breaks)")
    ax.axvline(T * 0.5, color=style.MUTED, lw=0.9, ls="--", label="intervention")
    ax.set_xlabel("time")
    ax.set_ylabel("systemic synchronisation  $r$")
    ax.set_title("Governance is required to break\na phase-locked condensate")
    style.legend_below(ax, ncol=1)
    ax.set_ylim(0, 1)
    style.panel_tag(ax, "C")
    return float(r_lock[-500:].mean()), float(r_halt[-500:].mean()), float(r_dec[-500:].mean())


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    ratios, cas_mean, cas_sem, T_inner = cascade_panel(axes[0])
    Ks, kout = kuramoto_panel(axes[1])
    r_lock, r_halt, r_dec = condensate_panel(axes[2])

    fig.suptitle("Figure 5.  Governance timing and the anything factory:  cascade ratios, entrainment thresholds, and the phase-locked condensate",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    style.drop_legends_below_labels(fig)
    out = os.path.join(os.path.dirname(__file__), "fig5_anything_factory.png")
    fig.savefig(out)
    print("wrote", out)
    print(f"cascade T_inner={T_inner:.0f}: instab(ratio<1)~{np.mean([m for r,m in zip(ratios,cas_mean) if r<1]):.3f}"
          f"  instab(ratio>10)~{np.mean([m for r,m in zip(ratios,cas_mean) if r>10]):.3f} (8 seeds)")
    print(f"condensate: entrained r={r_lock:.2f}  halt r={r_halt:.2f}  decouple r={r_dec:.2f}")
    json.dump(dict(cascade_ratios=ratios, cascade_instab_mean=cas_mean, cascade_instab_sem=cas_sem,
                   cascade_T_inner=T_inner, kuramoto_K=Ks,
                   condensate_locked=r_lock, condensate_halt=r_halt, condensate_decouple=r_dec),
              open(os.path.join(os.path.dirname(__file__), "..", "out", "fig5.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
