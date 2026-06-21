"""Opacity track: the verified value a legible reader forgoes vs free search.

Over a set of tasks, an order-`reader_order` legible reader competes with a free
searcher under the same verifier-call budget; each task's dividend (free -
legible) is plotted against the task's own measured interaction order. The
dividend should be ~0 on a separable (low-order) task and rise with measured
interaction order.

`task_specs` is a list of kwargs dicts passed to `substrate.dividend_task(**spec)`
-- the synthetic family for the reference substrate, or a list of real problem
selectors for a live substrate. The track grades by each task's *measured* order
(`DividendTask.interaction_order`), so it is substrate-agnostic.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..report import Finding

_DEFAULT_SPECS = [dict(d=12, order=o, kind="nk") for o in (1, 2, 3, 4)]


def _rank_corr(x, y) -> float:
    """Tie-aware Spearman correlation; NaN when either axis has no spread (a flat
    measured-order axis must not be read as a monotone trend)."""
    if len(set(map(float, x))) < 2 or len(set(map(float, y))) < 2:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def measure(substrate, task_specs=None, reader_order: int = 1, n_seeds: int = 6,
            budget: int = None, sep_threshold: float = 0.15, trend_threshold: float = 0.5,
            **kw) -> Finding:
    specs = task_specs or substrate.default_dividend_specs() or _DEFAULT_SPECS
    rows = []
    for spec in specs:
        b = budget or ((1 << spec["d"]) if "d" in spec else 4096)
        ds = [float(substrate.dividend_task(seed=s, **spec).dividend(budget=b, order=reader_order, seed=s))
              for s in range(n_seeds)]
        # interaction order is a task property, not a per-seed measurement: estimate once
        measured_order = int(substrate.dividend_task(seed=0, **spec).interaction_order(seed=0))
        rows.append(dict(spec=spec, dividend=round(float(np.mean(ds)), 3),
                         measured_order=measured_order))
    rows.sort(key=lambda r: r["measured_order"])
    orders = [r["measured_order"] for r in rows]
    divs = [r["dividend"] for r in rows]
    spread_ok = len(set(orders)) >= 2 and orders[-1] > orders[0]   # need a real order axis
    rho = _rank_corr(orders, divs)
    separable = divs[0]
    # a real rise of at least sep_threshold above the separable floor (unit-agnostic:
    # sigma on the cube, R^2/nats on tabular tasks)
    rising = spread_ok and np.isfinite(rho) and rho >= trend_threshold and (divs[-1] - divs[0]) > sep_threshold
    sep_ok = np.isfinite(separable) and separable < sep_threshold
    confirms = bool(rising and sep_ok)
    return Finding(
        track="opacity", property="legible-reader floor vs interaction order",
        capability="dividend_task",
        measured=dict(by_task=rows, measured_orders=orders, dividends=divs,
                      trend_rho=(round(rho, 2) if np.isfinite(rho) else None),
                      order_spread=len(set(orders)), reader_order=reader_order),
        confirms=confirms,
        summary=f"dividend {divs[0]:.2f} (order {orders[0]}) -> {divs[-1]:.2f} (order {orders[-1]}), "
                f"trend rho={rho:+.2f} ({'rises with order' if confirms else 'no clear rise'})",
        confirm_criterion=f"dividend < {sep_threshold} on the lowest-order task and rises "
                          f"with measured interaction order (rank trend >= {trend_threshold})",
        falsify_criterion="dividend flat across orders, or positive on a separable task "
                          "(then a weak legible class, not a real floor)",
    )
