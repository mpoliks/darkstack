"""A controllable population that advances one round at a time.

Wraps a finite-population stochastic-replicator search over a two-peak landscape
with priced feedback. Named regimes set the dynamics to a known ground truth:
two metastable versions, a single basin, each of the four convergence
pathologies, a slow ramp through a saddle-node fold, and a no-fold control. Every
round it records the behavioural observables, the control input applied (price and
injected intent), and a sample of producer decisions (with decision-time
propensity) to the trace store, so the measurements read only what a live factory
would expose. An optional delayed channel emits realised-consequence rewards
through the return queue so the out-of-loop reward path has planted ground truth.

The price-responsive observable in the 'governance' regime is norm_sat (it swings
0 -> ~0.95 under a price step); mean_pos barely moves there, so system
identification of that regime should read norm_sat.
"""
from __future__ import annotations

import numpy as np

from factory import DarkFactory as _Population, FactoryParams
from ..records import Decision, Reward, RoundObs
from ..instrumentation import TraceStore, ReturnQueue
from ..interfaces import SteppableFactory

# Ground-truth regimes. Parameters match the validated dynamics in src/.
REGIMES = {
    "versions":       dict(peakA=0.35, peakB=0.65, width=0.09, eta=2.0, M=60, mu=0.03, c=1.0),
    "single_basin":   dict(peakA=0.35, peakB=0.65, width=0.09, eta=2.0, M=60, mu=0.03, c=1.7),
    "healthy":        dict(peakA=0.3, peakB=0.7, width=0.07, eta=4, M=300, mu=0.06, c=1.4,
                           norm_target=0.7, metric_static=0.7),
    "stable_failure": dict(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=300, mu=0.05, c=0.45,
                           norm_target=0.7),
    "overfitting":    dict(peakA=0.3, peakB=0.7, width=0.07, eta=4, M=300, mu=0.04, c=1.4,
                           norm_target=0.5, metric_static=0.7),
    "learning_death": dict(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=300, mu=0.0008, c=0.45,
                           norm_target=0.7),
    "governance":     dict(mu=0.05, c=0.85, norm_target=0.70, cost_center=0.30, cost_width=0.10,
                           eta=4, M=600),
}

_CHANNELS = {"spec_target", "spec_height"}   # intent channels inject() accepts


class MockPopulation(SteppableFactory):
    def __init__(self):
        self.trace = TraceStore()
        self._f: _Population | None = None
        self._t = 0
        self._lam = 0.0
        self._c_schedule = None
        self._c_override = None
        self._osc = False
        self._n_sample = 8
        self._rng = np.random.default_rng(0)
        self._realized_delay = 0
        self._rq = ReturnQueue()
        self._pending: dict[str, float] = {}

    def reset(self, regime: str | None = None, n_sample: int = 8, seed: int = 0,
              c_schedule=None, realized_delay: int = 0, realized_jitter: int = 0,
              **params) -> "MockPopulation":
        base = dict(REGIMES.get(regime, {})) if regime else {}
        base.update(params)
        base.setdefault("seed", seed)
        self.trace = TraceStore()
        self._t = 0
        self._lam = 0.0
        self._c_override = None
        self._osc = False
        self._n_sample = int(n_sample)
        self._rng = np.random.default_rng(seed + 9973)
        self._realized_delay = int(realized_delay)
        self._rq = ReturnQueue(delay=int(realized_delay), jitter=int(realized_jitter),
                               rng=np.random.default_rng(seed + 4242))
        self._pending = {}

        if regime == "thrash":
            base = dict(peakA=0.3, peakB=0.7, width=0.08, eta=2, M=40, mu=0.2,
                        norm_target=0.5, seed=base.get("seed", seed))
            self._osc = True
        if regime == "fold_ramp":
            base = dict(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=400, mu=0.02,
                        seed=base.get("seed", seed))
            burn, ramp = 1500, 10500
            c_schedule = np.concatenate([np.full(burn, 1.6), np.linspace(1.6, 0.4, ramp)])
        if regime == "no_fold":
            base = dict(peakA=0.3, peakB=0.7, width=0.06, eta=4, M=400, mu=0.02,
                        c=1.2, seed=base.get("seed", seed))
            c_schedule = None

        self._c_schedule = None if c_schedule is None else np.asarray(c_schedule, float)
        self._f = _Population(FactoryParams(**base))
        return self

    # ---- intent control -----------------------------------------------------
    def injectable_channels(self) -> set:
        return set(_CHANNELS)

    def inject(self, **channels) -> None:
        bad = set(channels) - _CHANNELS
        if bad:
            raise KeyError(f"unsupported inject channels {bad}; available: {sorted(_CHANNELS)}")
        if "spec_target" in channels:
            self._f.p.norm_target = float(self._f.p.norm_target) + float(channels["spec_target"])
        if "spec_height" in channels:
            base_c = self._c_override if self._c_override is not None else float(self._f.p.c)
            self._c_override = base_c + float(channels["spec_height"])

    def set_price(self, lam: float) -> None:
        self._lam = float(lam)

    def _c_now(self):
        if self._c_override is not None:
            return float(self._c_override)
        if self._osc:
            return 1.0 + 0.6 * np.sin(2 * np.pi * self._t / 40)
        if self._c_schedule is not None:
            i = min(self._t, len(self._c_schedule) - 1)
            return float(self._c_schedule[i])
        return None

    def behaviour(self) -> dict:
        return self._f.observe(self._c_now())

    def step(self) -> RoundObs:
        # release any realised-consequence rewards that have come due
        due_rewards = []
        for h in self._rq.due(self._t):
            r = Reward(handle=h, round=self._t, score=float(self._pending.pop(h, 0.0)),
                       kind="realized")
            self.trace.record_reward(r)
            due_rewards.append(r)

        c = self._c_now()
        c_eff = self._f.p.c if c is None else c
        o = self._f.step(c=c, lam=self._lam)
        self.trace.record_behaviour(self._t, o)
        self.trace.record_control(self._t, dict(price=self._lam, spec_height=c_eff,
                                                 spec_target=float(self._f.p.norm_target)))
        x = self._f.x
        quality = self._f._quality(c_eff)
        qn = (quality - quality.min()) / (np.ptp(quality) + 1e-12)
        decs = []
        for i in range(self._n_sample):
            a = int(self._rng.choice(len(x), p=x))
            d = Decision(handle=f"{self._t}-{i}", round=self._t, agent=f"p{i}",
                         action=a, n_actions=len(x), propensity=float(x[a]),
                         distribution=(), behaviour=o["mean_pos"])
            self.trace.record_decision(d)
            self.trace.record_reward(Reward(handle=d.handle, round=self._t,
                                            score=float(qn[a]), kind="in_round"))
            if self._realized_delay > 0:
                self._rq.enqueue(d.handle, self._t)
                self._pending[d.handle] = float(qn[a])
            decs.append(d)
        self._t += 1
        return RoundObs(round=self._t - 1, behaviour=o,
                        control=dict(price=self._lam, spec_height=c_eff,
                                     spec_target=float(self._f.p.norm_target)),
                        decisions=decs, rewards=due_rewards)
