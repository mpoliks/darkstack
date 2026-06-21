"""A task of known interaction order, with a metered verifier.

A value landscape on the Boolean cube with a planted highest interaction order
(NK epistasis or a dense planted order). A free searcher draws best-of-budget
candidates; a legible reader of order K fits a degree-<=K model from the same
budget and acts on its argmax. Achieved verified value is reported in sigma
units, so the dividend (free - legible) is comparable across tasks. The effective
interaction order is read exactly from the Walsh spectrum (ground truth).
"""
from __future__ import annotations

import numpy as np

from dividend import nk_landscape, planted_order, budget_curves, dividend as _exact_dividend
from ..interfaces import DividendTask


class MockDividendTask(DividendTask):
    def __init__(self, d: int = 10, order: int = 2, kind: str = "nk", seed: int = 0):
        rng = np.random.default_rng(seed)
        if kind == "nk":
            self.f, self.pc = nk_landscape(d, max(0, order - 1), rng)   # NK Walsh order = K+1
        else:
            self.f, self.pc = planted_order(d, order, rng)
        self.d = int(d)
        self._planted = int(order)
        self._sd = float(self.f.std()) + 1e-12
        self._fmax_z = float(self.f.max()) / self._sd

    def _achieved(self, budget: int, order: int, seed: int):
        rng = np.random.default_rng(seed)
        out = budget_curves(self.f, self.d, [order], [int(budget)], rng, reps=12)
        free_z = self._fmax_z - float(out["free"][0])         # gap -> achieved value (sigma units)
        leg_z = self._fmax_z - float(out[order][0])
        return free_z, leg_z

    def free_floor(self, budget: int, seed: int = 0) -> float:
        return self._achieved(budget, 1, seed)[0]

    def legible_floor(self, budget: int, order: int = 1, seed: int = 0) -> float:
        return self._achieved(budget, order, seed)[1]

    def interaction_order(self, seed: int = 0) -> int:
        """Forced-opacity (acting) order K*: the smallest reader order whose argmax
        reaches the optimum. May be below walsh_order() when the optimum happens to
        sit on a lower-order face."""
        _, _, kstar = _exact_dividend(self.f, self.pc)
        return int(kstar) if kstar is not None else self._planted

    def walsh_order(self) -> int:
        """Highest non-vanishing Walsh interaction order present in the landscape
        (the fitting order, distinct from the acting order K* above)."""
        from dividend import walsh_coeffs
        fhat = walsh_coeffs(self.f)
        nz = np.abs(fhat) > 1e-9
        return int(self.pc[nz].max()) if nz.any() else 0
