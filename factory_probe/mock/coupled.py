"""Factory loops coupled through a shared dependency.

Each loop's slowest control cycle is a phase oscillator; a shared dependency
couples them. The synchronisation order parameter r measures how locked the
population is. Frequency spread (dependency diversity / jitter) sets the coupling
threshold Kc below which the loops stay desynchronised.
"""
from __future__ import annotations

import numpy as np

from kuramoto import simulate, critical_coupling_lorentzian
from ..interfaces import CoupledLoops


class MockCoupledLoops(CoupledLoops):
    def __init__(self, N: int = 400, T: float = 50.0, dt: float = 0.02):
        self.N = int(N)
        self.T = float(T)
        self.dt = float(dt)

    def _omega(self, diversity: float, rng) -> np.ndarray:
        omega = diversity * np.tan(np.pi * (rng.random(self.N) - 0.5))   # Cauchy, half-width=diversity
        return np.clip(omega - np.median(omega), -40, 40)

    def order_parameter(self, coupling: float, diversity: float, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        omega = self._omega(diversity, rng)
        r, _, _ = simulate(self.N, coupling, omega, T=self.T, dt=self.dt,
                           rng=np.random.default_rng(seed + 1))
        return float(r)

    def critical_coupling(self, diversity: float) -> float:
        return float(critical_coupling_lorentzian(diversity))

    def condensate(self, coupling: float = 2.2, diversity: float = 0.3,
                   intervention: str = "none", seed: int = 3, at_frac: float = 0.5) -> np.ndarray:
        """Order-parameter time series of a locked population, with an optional
        midpoint intervention: 'halt' (one-shot desynchronisation) or 'decouple'
        (drop the coupling below Kc). Returns r(t)."""
        rng = np.random.default_rng(seed)
        N = self.N
        steps = int(self.T / self.dt)
        omega = diversity * np.tan(np.pi * (rng.random(N) - 0.5))
        omega = np.clip(omega - np.median(omega), -30, 30)
        theta = rng.uniform(0, 2 * np.pi, N)
        K = coupling
        at = int(steps * at_frac)
        r_t = np.empty(steps)
        for s in range(steps):
            z = np.mean(np.exp(1j * theta))
            r_t[s] = np.abs(z)
            psi = np.angle(z)
            theta = theta + self.dt * (omega + K * r_t[s] * np.sin(psi - theta))
            if s == at:
                if intervention == "halt":
                    theta = theta + 8.0 * rng.standard_normal(N)
                elif intervention == "decouple":
                    K = 0.3
        return r_t
