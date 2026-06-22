"""A reusable bounded legible reader for the opacity track.

The opacity floor asks: how much verified value does a free, full-interaction
searcher reach that a bounded *legible* reader cannot? The legible reader here is
domain-general for any task whose candidates can be turned into a feature vector
and scored by a held-out verifier: an additive model for interaction order 1, a
depth-capped model for higher orders, against a full-interaction model as the free
searcher. The floor is the held-out score the legible reader forgoes.

`FeatureDividendTask` wires this to three callables -- sample candidates, featurize
them, score them with the verifier -- so a live factory whose outputs are
featurizable gets the dividend without writing the reader. The reference data
implementation is `substrates/tabular.py`; for a non-vector legible form (a
depth-capped program or a sparse rule list over code) you implement the floors
directly under the same one-call-per-candidate meter.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (HistGradientBoostingRegressor as HGR,
                              HistGradientBoostingClassifier as HGC)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, log_loss

from .interfaces import DividendTask

# Same reader recipe the tabular reference uses, so the two are comparable.
_COMMON = dict(max_iter=120, learning_rate=0.08, early_stopping=True,
               validation_fraction=0.15, random_state=0)
_DEPTHS = (1, 2, 3, 4)


def free_reader() -> dict:
    """The free searcher: a full-interaction model."""
    return dict(max_depth=6, **_COMMON)


def legible_reader(order: int) -> dict:
    """The bounded legible reader: additive at order 1, depth-capped above it."""
    if order <= 1:
        return dict(interaction_cst="no_interactions", max_leaf_nodes=63, **_COMMON)
    return dict(max_depth=int(order), **_COMMON)


def heldout_score(X, y, model_kwargs: dict, kind: str = "reg", splits: int = 2,
                  seed: int = 0, classes=None) -> float:
    """Mean held-out score of a model over `splits` train/test splits. R^2 for
    regression, negative log-loss for classification (both higher-is-better)."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    out = []
    for s in range(splits):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1000 * seed + s)
        if kind == "reg":
            m = HGR(**model_kwargs).fit(Xtr, ytr)
            out.append(r2_score(yte, m.predict(Xte)))
        else:
            m = HGC(**model_kwargs).fit(Xtr, ytr)
            out.append(-log_loss(yte, m.predict_proba(Xte), labels=classes))
    return float(np.mean(out))


class FeatureDividendTask(DividendTask):
    """A DividendTask for any featurizable, verifier-scored candidate space.

    Parameters
    ----------
    sample : (budget, seed) -> a sequence of `budget` candidates (one verifier call each)
    featurize : candidates -> array (n_candidates, n_features)
    verifier_score : candidates -> array (n_candidates,), higher is better
    kind : "reg" or "clf"
    """

    def __init__(self, sample, featurize, verifier_score, kind: str = "reg",
                 splits: int = 2, classes=None):
        self.sample = sample
        self.featurize = featurize
        self.verifier_score = verifier_score
        self.kind = kind
        self.splits = int(splits)
        self.classes = classes

    def _xy(self, budget, seed):
        cands = self.sample(budget, seed)
        return self.featurize(cands), self.verifier_score(cands)

    def _score(self, model_kwargs, budget, seed) -> float:
        X, y = self._xy(budget, seed)
        return heldout_score(X, y, model_kwargs, self.kind, self.splits, seed, self.classes)

    def free_floor(self, budget, seed: int = 0) -> float:
        return self._score(free_reader(), budget, seed)

    def legible_floor(self, budget, order: int = 1, seed: int = 0) -> float:
        return self._score(legible_reader(order), budget, seed)

    def interaction_order(self, seed: int = 0, budget=None) -> int:
        """Smallest tree depth whose held-out score saturates. A surrogate that
        biases low orders up by ~1 and saturates at 4; the dividend itself does not
        depend on it."""
        sc = [self._score(dict(max_depth=k, **_COMMON), budget, seed) for k in _DEPTHS]
        best = max(sc)
        return next((_DEPTHS[i] for i, s in enumerate(sc) if best - s <= 0.01), _DEPTHS[-1])
