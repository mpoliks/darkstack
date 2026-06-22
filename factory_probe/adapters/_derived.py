"""Derive the behavioural keys the tracks read from a stream of run records.

A general agent framework records tokens, durations, statuses, and tool names per
run -- not a dark-stack behavioural coordinate. None of mean_pos / norm_sat /
metric_sat / variety is native to OpenClaw or Hermes (confirmed by reading their
state schemas: no such column exists). This module synthesises them from the
scalars a run does expose, by an explicit recipe, so an adapter over such a
framework can still feed the observational tracks (versioning, catastrophe,
pathology). They are derived proxies, documented as such, not values the framework
reports.

Recipe, over the run series in time order:
  mean_pos    a per-run coordinate in [0,1] the adapter supplies directly (e.g.
              Hermes tool-density), else total tokens min-max normalised over the
              series -- the coordinate versioning and catastrophe read
  norm_sat    rolling success rate over `window` runs (status == completed)
  metric_sat  rolling 1 - normalised duration (faster completion scores higher)
  variety     distinct tools used over the last `window` runs (>= 1)
A key is omitted when the run records hold nothing to derive it from; pathology
sub-signals that need a missing key degrade accordingly.
"""
from __future__ import annotations

import numpy as np


def _minmax(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def derive_series(features: list[dict], window: int = 200) -> list[dict]:
    """Turn per-run feature dicts into per-round behavioural dicts.

    Each feature dict may hold any of: `mean_pos` (a ready [0,1] coordinate),
    `tokens`, `duration_ms`, `success` (bool), `tools` (list of names).
    """
    toks = [f["tokens"] for f in features if "tokens" in f]
    durs = [f["duration_ms"] for f in features if "duration_ms" in f]
    tlo, thi = (min(toks), max(toks)) if toks else (0.0, 1.0)
    dlo, dhi = (min(durs), max(durs)) if durs else (0.0, 1.0)
    succ: list[float] = []
    tools: list[list] = []
    out: list[dict] = []
    for f in features:
        beh: dict = {}
        if "mean_pos" in f:
            beh["mean_pos"] = float(min(1.0, max(0.0, f["mean_pos"])))
        elif "tokens" in f:
            beh["mean_pos"] = _minmax(float(f["tokens"]), tlo, thi)
        if "success" in f:
            succ.append(1.0 if f["success"] else 0.0)
            beh["norm_sat"] = float(np.mean(succ[-window:]))
        if "duration_ms" in f:
            beh["metric_sat"] = 1.0 - _minmax(float(f["duration_ms"]), dlo, dhi)
        if "tools" in f:
            tools.append(list(f["tools"]))
            recent = [t for ts in tools[-window:] for t in ts]
            beh["variety"] = float(len(set(recent))) if recent else 1.0
        if "mean_pos" not in beh and "norm_sat" in beh:
            beh["mean_pos"] = beh["norm_sat"]      # fall back to the success-rate coordinate
        out.append(beh)
    return out
