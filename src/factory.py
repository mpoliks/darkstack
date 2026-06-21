"""
The unified Dark Factory.

A finite population of producer agents searches a space of K "assemblies"
(configurations on a 1-D design axis). Each agent is a mean-based no-regret
learner (multiplicative weights / replicator) with exploration rate mu -- the
"surplus-generating frontier." The collective behaviour we read out is an
*input-output distribution*, the object a "version" lives
in: we never inspect the agents' internal weights, only what the population does.

Landscape. Assemblies sit on a line. Two quality peaks A and B encode two ways
to satisfy intent. Peak A has fixed height; peak B's height is the control
parameter `c` (a stand-in for a drifting spec / market). In the figures
(`beta = 0`) the bistability is supplied by the two fixed quality wells plus
finite-`M` demographic noise: the population occupies one well metastably and
crosses to the other only by a noise-driven (Kramers-like) escape. A separate
positive frequency-dependent term `beta` (incumbency / lock-in) is available and
widens the thin bistable sliver into a broad bistable band -- a full Thom cusp
with hysteresis (the exact edges depend on `mu` and the sweep protocol) -- but
the figures keep `beta = 0` for the cleanest metastability, so nothing in them
depends on incumbency. A price lambda * cost penalises constraint
violation (the Lagrangian governance channel).

Norm vs metric. The factory is rewarded on a METRIC (a proxy that samples the
true NORM at a finite rate). When the norm varies faster than the metric samples
it, optimisation against the metric aliases away from the norm -- a
Nyquist account of overfitting.

This single object instantiates, under different parameter regimes:
  * versions            (metastable occupancy of a peak; Fig. 3)
  * stable failure      (robust version that fails the spec)
  * overfitting         (metric/norm aliasing)
  * learning death      (mu -> 0, variety collapses)
  * thrash              (never settles)
  * fold/cusp catastrophe + critical slowing down (Fig. 4)
and it is governed by a PID-controlled lambda (control.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _gauss_peak(grid, center, width, height):
    return height * np.exp(-0.5 * ((grid - center) / width) ** 2)


@dataclass
class FactoryParams:
    K: int = 64                 # assemblies on the design axis [0,1]
    M: int = 400                # population size (finite -> sampling noise)
    eta: float = 6.0            # learning rate (replicator temperature)
    mu: float = 0.06            # exploration / mutation (mean-based frontier)
    beta: float = 0.0           # incumbency / positive frequency dependence (lock-in)
    align_weight: float = 0.0   # reward for tracking the scored metric (proxy-chasing)
    align_width: float = 0.06   # tolerance of the metric-alignment reward
    peakA: float = 0.30         # position of quality peak A
    peakB: float = 0.70         # position of quality peak B
    width: float = 0.05         # peak width
    heightA: float = 1.0        # fixed height of A
    c: float = 0.8              # control parameter = height of B (the drifting spec)
    lam: float = 0.0            # price lambda on cost
    cost_kind: str = "gauss"    # "gauss" (localised) or "linear_right" (ramp)
    cost_scale: float = 1.0     # overall magnitude of the priced cost
    cost_center: float = 0.70   # where the priced constraint bites (near B)
    cost_width: float = 0.12
    # Norm/metric (overfitting) machinery -------------------------------------
    norm_target: float = 0.50   # if static, the design coordinate the NORM wants
    norm_freq: float = 0.0      # angular freq of a time-varying norm target
    norm_amp: float = 0.0       # amplitude of norm oscillation around norm_target
    metric_sample_period: int = 1   # metric refreshes every this-many rounds (sampling)
    metric_static: float | None = None  # if set, the metric is a fixed proxy (overfitting)
    seed: int = 0


class DarkFactory:
    def __init__(self, p: FactoryParams):
        self.p = p
        self.grid = np.linspace(0, 1, p.K)
        self.rng = np.random.default_rng(p.seed)
        # population distribution over assemblies (start concentrated on A)
        self.x = _gauss_peak(self.grid, p.peakA, p.width, 1.0)
        self.x /= self.x.sum()
        self.t = 0
        self._metric_target = self._norm_target_at(0)

    # ---- intent ---------------------------------------------------------- #
    def _norm_target_at(self, t: int) -> float:
        p = self.p
        return p.norm_target + p.norm_amp * np.sin(p.norm_freq * t)

    def _quality(self, c: float) -> np.ndarray:
        p = self.p
        qA = _gauss_peak(self.grid, p.peakA, p.width, p.heightA)
        qB = _gauss_peak(self.grid, p.peakB, p.width, c)
        return qA + qB

    # ---- one factory round ----------------------------------------------- #
    def step(self, c: float | None = None, lam: float | None = None):
        p = self.p
        c = p.c if c is None else c
        lam = p.lam if lam is None else lam

        # metric samples the (possibly fast-moving) norm at a finite rate, OR is a
        # fixed proxy (static overfitting: the reward chases a peak the metric
        # blesses while the true norm sits elsewhere)
        if p.metric_static is not None:
            self._metric_target = p.metric_static
        elif self.t % max(1, p.metric_sample_period) == 0:
            self._metric_target = self._norm_target_at(self.t)

        quality = self._quality(c)
        incumbency = p.beta * self.x                      # positive freq-dependence
        cost = self._cost()
        # the factory is rewarded for OUTPUT that scores well on the (scored)
        # metric proxy -- this is what couples production to the eval layer and is
        # what overfitting games: chase the metric, not the norm.
        align = p.align_weight * _gauss_peak(self.grid, self._metric_target, p.align_width, 1.0)
        reward = quality + align + incumbency - lam * cost    # Lagrangian reward

        # Stochastic replicator = Wright-Fisher with selection + mutation. The
        # population state IS the empirical frequency of M agents, so demographic
        # noise is genuine: at small M the population can cross fitness barriers
        # between assembly peaks (Kramers escape), producing spontaneous version
        # transitions; at large M it is effectively deterministic.
        fitness = np.exp(p.eta * (reward - reward.max()))   # multiplicative-weights selection
        p_sel = self.x * fitness
        p_sel /= p_sel.sum()
        p_mut = (1 - p.mu) * p_sel + p.mu / p.K             # mutation = exploration frontier
        self.x = self.rng.multinomial(p.M, p_mut) / p.M     # demographic resampling
        self.t += 1
        return self.observe(c)

    def _cost(self) -> np.ndarray:
        p = self.p
        if p.cost_kind == "linear_right":
            return p.cost_scale * self.grid                  # ramp cost: higher pos costs more
        return p.cost_scale * _gauss_peak(self.grid, p.cost_center, p.cost_width, 1.0)

    # ---- behavioural read-outs (the only thing versioning may use) ------- #
    def observe(self, c: float | None = None) -> dict:
        p = self.p
        c = p.c if c is None else c
        mean_pos = float(self.grid @ self.x)              # behavioural coordinate
        variety = float(np.exp(-np.sum(self.x * np.log(self.x + 1e-12))))  # eff. # assemblies
        # spec satisfaction: closeness of realised output to the TRUE norm target now
        true_target = self._norm_target_at(self.t)
        norm_sat = float(np.exp(-((mean_pos - true_target) ** 2) / (2 * 0.04 ** 2)))
        # metric satisfaction: closeness to the (sampled, possibly aliased) target
        metric_sat = float(np.exp(-((mean_pos - self._metric_target) ** 2) / (2 * 0.04 ** 2)))
        occupancy_B = float(self.x[self.grid > 0.5].sum())  # which version (peak)
        return dict(mean_pos=mean_pos, variety=variety, norm_sat=norm_sat,
                    metric_sat=metric_sat, occupancy_B=occupancy_B)

    def run(self, n: int, c_schedule=None, lam_schedule=None) -> dict:
        keys = ["mean_pos", "variety", "norm_sat", "metric_sat", "occupancy_B"]
        out = {k: np.empty(n) for k in keys}
        for i in range(n):
            c = None if c_schedule is None else float(c_schedule[i])
            lam = None if lam_schedule is None else float(lam_schedule[i])
            o = self.step(c=c, lam=lam)
            for k in keys:
                out[k][i] = o[k]
        return out


if __name__ == "__main__":
    # The cusp: exploration rate mu controls version stickiness. Low mu => the
    # population is sticky (bistable versions, wide hysteresis loop in c). High mu
    # => fluid (tracks c immediately, little hysteresis). The SAME knob that sets
    # the "mean-based frontier" strength sets the version structure -- the
    # frontier/core tension made mechanical.
    n = 4000
    up = np.linspace(0.6, 1.6, n)
    down = up[::-1]
    iu, idn = np.argmin(np.abs(up - 1.1)), np.argmin(np.abs(down - 1.1))
    for mu, tag in [(0.01, "low explore"), (0.06, "mid explore"), (0.25, "high explore")]:
        f = DarkFactory(FactoryParams(beta=0.0, mu=mu, seed=1))
        ou = f.run(n, c_schedule=up)
        od = f.run(n, c_schedule=down)      # continues the same factory: a loop
        gap = abs(ou["occupancy_B"][iu] - od["occupancy_B"][idn])
        print(f"mu={mu:4.2f} ({tag:11s})  occ_B@c=1.1  up={ou['occupancy_B'][iu]:.2f}"
              f"  down={od['occupancy_B'][idn]:.2f}   hysteresis_gap={gap:.2f}"
              f"   mean_variety={ou['variety'].mean():4.1f}")
