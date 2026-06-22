"""A real-data substrate for the opacity track, off the enumerable cube.

A DividendTask backed by gradient-boosted models on real or synthetic tabular
data: the free searcher is a full-interaction model, the legible reader an
additive (order-1) model, and the floor is the held-out verified performance the
additive reader forgoes. Effective interaction order is estimated by a
degree-bounded surrogate sweep. The floor is a model-class gap, so it is
budget-invariant in the large-data limit; `budget` optionally caps the training
rows (a data budget), but the opacity track's default budget exceeds the tabular
training size, so the cap is inactive by default.

Scores are R^2 for regression and negative log-loss for classification (both
higher-is-better), so the dividend (free - legible) is positive when interactions
carry verified value an additive reader cannot reach.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (HistGradientBoostingRegressor as HGR,
                              HistGradientBoostingClassifier as HGC)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, log_loss
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, load_digits

from ..interfaces import Substrate, DividendTask

_COMMON = dict(max_iter=120, learning_rate=0.08, early_stopping=True,
               validation_fraction=0.15, random_state=0)
_DEPTHS = (1, 2, 3, 4)


def _synth(order, n=2500, d=6, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(-1, 1, (n, d))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.tanh(2 * X[:, 2]) + 0.5 * X[:, 3]
    if order >= 2:
        y = y + 1.5 * X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
    if order >= 3:
        y = y + 2.0 * X[:, 0] * X[:, 1] * X[:, 2]
    if order >= 4:
        y = y + 3.0 * X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]
    return X, y + 0.1 * r.standard_normal(n), "reg"


def _load(dataset):
    if dataset == "diabetes":
        d = load_diabetes(); return d.data, d.target, "reg"
    if dataset == "breast_cancer":
        d = load_breast_cancer(); return d.data, d.target, "clf"
    if dataset == "wine":
        d = load_wine(); return d.data, d.target, "clf"
    if dataset == "digits":
        d = load_digits(); return d.data, d.target, "clf"
    if dataset == "california":
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(); return d.data, d.target, "reg"
    raise ValueError(f"unknown dataset {dataset!r}")


class TabularDividendTask(DividendTask):
    def __init__(self, dataset=None, synthetic_order=None, splits=2, seed=0, **_):
        if synthetic_order is not None:
            self.X, self.y, self.kind = _synth(int(synthetic_order), seed=seed)
        elif dataset is not None:
            self.X, self.y, self.kind = _load(dataset)
        else:
            raise ValueError("TabularDividendTask needs dataset= or synthetic_order=")
        self.splits = int(splits)
        self._classes = np.unique(self.y) if self.kind == "clf" else None

    def _score(self, mk, budget, seed) -> float:
        out = []
        for s in range(self.splits):
            Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=0.3,
                                                  random_state=1000 * seed + s)
            if budget and budget < len(ytr):
                Xtr, ytr = Xtr[:budget], ytr[:budget]
            if self.kind == "reg":
                m = HGR(**mk).fit(Xtr, ytr)
                out.append(r2_score(yte, m.predict(Xte)))
            else:
                m = HGC(**mk).fit(Xtr, ytr)
                out.append(-log_loss(yte, m.predict_proba(Xte), labels=self._classes))
        return float(np.mean(out))

    def free_floor(self, budget, seed=0):
        return self._score(dict(max_depth=6, **_COMMON), budget, seed)

    def legible_floor(self, budget, order=1, seed=0):
        if order <= 1:
            mk = dict(interaction_cst="no_interactions", max_leaf_nodes=63, **_COMMON)
        else:
            mk = dict(max_depth=int(order), **_COMMON)
        return self._score(mk, budget, seed)

    def interaction_order(self, seed=0):
        """Coarse effective-order estimate: the smallest tree depth whose held-out
        score saturates. It is a surrogate, not exact -- it biases low orders upward
        by ~1 (a depth-1 stump under-fits univariate shape) and saturates at
        max(_DEPTHS)=4, so distinct true orders above ~2 collapse together. The
        decisive signal of the opacity track is the dividend itself, which is
        computed independently of this estimate; the estimate only orders the x-axis.
        """
        sc = [self._score(dict(max_depth=k, **_COMMON), None, seed) for k in _DEPTHS]
        best = max(sc)
        return next((_DEPTHS[i] for i, s in enumerate(sc) if best - s <= 0.01), _DEPTHS[-1])


class TabularSubstrate(Substrate):
    name = "tabular"

    def capabilities(self) -> set:
        return {"dividend_task"}

    def default_dividend_specs(self):
        return [dict(synthetic_order=o) for o in (1, 2, 3, 4)]

    def dividend_task(self, **spec) -> TabularDividendTask:
        return TabularDividendTask(**spec)
