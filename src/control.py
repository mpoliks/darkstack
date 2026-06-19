"""
Governance as control: PID-priced lambda and cascade-control timing.

The paper's "Versioning" and "Spec and the Loop" sections specify governance as
a Lagrangian price lambda subtracted from reward, with lambda set by a PID
controller: P raises the penalty in proportion to current violation, I stores
sustained violation so persistent failure ratchets up in price, D reacts to the
rate of change to damp overshoot. It then warns of "iatrogenic thrash": if the
governing loop revises lambda near the natural frequency of the loop beneath it,
the two resonate and gain runs away (instability at unity loop gain / 180 deg
phase). Cascade-control practice demands the inner loop settle 3:1 to 10:1 faster
than the outer loop that commands it.

References
----------
classical PID / cascade control (Astrom & Murray, Feedback Systems);
the paper's own citations to cascade-ratio heuristics (3:1 .. 10:1).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PID:
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    lo: float = 0.0
    hi: float = 50.0
    _i: float = 0.0
    _prev: float | None = None

    def update(self, error: float, dt: float = 1.0) -> float:
        self._i += error * dt
        d = 0.0 if self._prev is None else (error - self._prev) / dt
        self._prev = error
        out = self.Kp * error + self.Ki * self._i + self.Kd * d
        return float(np.clip(out, self.lo, self.hi))

    def reset(self):
        self._i = 0.0
        self._prev = None


def governed_run(make_factory, n_steps: int, gov_period: int, pid: PID,
                 target_sat: float = 0.85, control_signal: str = "norm_sat",
                 c_schedule=None):
    """Run a factory with an OUTER governance loop that reprices lambda every
    `gov_period` inner steps from the PID controller, holding lambda constant
    between revisions (zero-order hold). Returns behavioural traces + lambda(t).

    The cascade ratio is gov_period / (inner settling time). Small gov_period
    (governance reacting to unsettled transients) produces iatrogenic thrash.
    """
    f = make_factory()
    keys = ["mean_pos", "variety", "norm_sat", "metric_sat", "occupancy_B"]
    traces = {k: np.empty(n_steps) for k in keys}
    lam_trace = np.empty(n_steps)
    lam = 0.0
    pid.reset()
    for t in range(n_steps):
        c = None if c_schedule is None else float(c_schedule[t])
        o = f.step(c=c, lam=lam)
        for k in keys:
            traces[k][t] = o[k]
        lam_trace[t] = lam
        if (t + 1) % gov_period == 0:
            # governance observes the recent window and reprices.
            recent = traces[control_signal][max(0, t - gov_period + 1):t + 1].mean()
            error = target_sat - recent           # how far below target we are
            lam = pid.update(error, dt=gov_period)
    traces["lam"] = lam_trace
    return traces


def instability_index(signal: np.ndarray, settle_frac: float = 0.5) -> float:
    """Quantify oscillation/instability of a control signal: the coefficient of
    variation of its later portion (high => thrash/oscillation)."""
    s = signal[int(settle_frac * len(signal)):]
    m = np.mean(s)
    return float(np.std(s) / (abs(m) + 1e-9))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from factory import DarkFactory, FactoryParams

    # A factory parked in a STABLE FAILURE: the spec wants the design coordinate
    # near 0.7 (peak B region) but B's height c is low, so the population sits on
    # A (~0.3) and fails. Governance should price the failing region and, with a
    # well-timed (slow) outer loop, ratchet lambda to push the population to B.
    def make():
        return DarkFactory(FactoryParams(
            mu=0.05, c=0.85, norm_target=0.70, cost_center=0.30, cost_width=0.10,
            seed=3))

    # Inner settling time of this factory is ~tens of steps; sweep gov_period
    # across the cascade ratio to expose iatrogenic thrash.
    print("cascade-ratio sweep (governance period vs inner loop):")
    for gov_period in [5, 15, 40, 120]:
        pid = PID(Kp=6.0, Ki=0.4, Kd=2.0, hi=30)
        tr = governed_run(make, 3000, gov_period, pid, target_sat=0.8,
                          control_signal="norm_sat")
        ii = instability_index(tr["lam"])
        sat = tr["norm_sat"][-500:].mean()
        print(f"  gov_period={gov_period:3d}  lambda_instability={ii:5.2f}"
              f"  final_norm_sat={sat:.2f}")
    print("Small gov_period -> high lambda_instability (iatrogenic thrash);")
    print("large gov_period (cascade ratio satisfied) -> stable, high satisfaction.")
