"""The OpenClaw adapter: mapping the control-plane run table to the capability
interfaces, tested against a recorded-row source and an in-memory SQLite fixture."""
import os
import sqlite3
import tempfile

import numpy as np
import pytest

from factory_probe.adapters.openclaw import (OpenClawSubstrate, OpenClawFactory,
                                             SqliteOpenClawSource)


def _telegraph_tokens(n=5000, lo=1000, hi=5000, hop=0.004, seed=0):
    rng = np.random.default_rng(seed)
    out, level = [], lo
    for _ in range(n):
        if rng.random() < hop:
            level = hi if level == lo else lo
        out.append(level + int(20 * rng.standard_normal()))
    return out


class RecordedSource:
    """A stand-in for SqliteOpenClawSource: returns recorded cron_run_logs rows."""

    table = "cron_run_logs"

    def __init__(self, tokens):
        self._rows = [{"job_id": "job1", "seq": i, "ts": i, "status": "completed",
                       "duration_ms": 200.0, "model": "m", "provider": "p",
                       "total_tokens": t, "run_id": f"r{i}"} for i, t in enumerate(tokens)]

    def runs(self):
        return list(self._rows)


def test_capabilities_are_observational_only():
    sub = OpenClawSubstrate(RecordedSource([1000] * 3))
    assert sub.capabilities() == {"steppable"}
    assert sub.supports("steppable") and not sub.supports("coupled_loops")


def test_tokens_become_the_behavioural_coordinate_and_status_the_reward():
    tokens = _telegraph_tokens(n=300)
    fac = OpenClawSubstrate(RecordedSource(tokens)).steppable()
    fac.run(len(tokens))
    mp = fac.trace.behaviour_series("mean_pos")
    assert mp.min() >= 0.0 and mp.max() <= 1.0 and mp.max() - mp.min() > 0.5   # bimodal coordinate
    assert all(r.score == 1.0 for r in fac.trace.rewards)                      # completed -> reward 1
    assert all(d.n_actions == 1 for d in fac.trace.decisions)                  # coarse run-level decision


def test_versioning_detects_two_modes_through_the_adapter():
    tokens = _telegraph_tokens(n=5000)
    sub = OpenClawSubstrate(RecordedSource(tokens))
    from factory_probe.tracks import versioning
    f = versioning.measure(sub, n=len(tokens), n_boxes=24, seed=0)
    assert f.measured["n_metastable_0p95"] >= 2


def test_sqlite_source_orders_by_ts_not_insertion():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE cron_run_logs (job_id TEXT, seq INTEGER, ts INTEGER, "
                    "status TEXT, duration_ms REAL, model TEXT, provider TEXT, total_tokens INTEGER)")
        # insert with ts DESCENDING while seq ASCENDING, so ORDER BY ts,seq is load-bearing:
        # if the source did not order by ts, rows would come back in seq/insertion order.
        for i in range(5):
            con.execute("INSERT INTO cron_run_logs VALUES (?,?,?,?,?,?,?,?)",
                        ("job1", i, (4 - i) * 100, "completed", 200.0, "gpt", "openai", 1000 + i))
        con.execute("INSERT INTO cron_run_logs VALUES (?,?,?,?,?,?,?,?)",
                    ("job2", 0, 999, "failed", 50.0, "gpt", "openai", 7))
        con.commit()
        con.close()
        rows = SqliteOpenClawSource(path, table="cron_run_logs", job_id="job1").runs()
        assert len(rows) == 5                                  # job filter applied
        assert [r["ts"] for r in rows] == [0, 100, 200, 300, 400]   # time-ordered, not insertion-ordered
        assert [r["seq"] for r in rows] == [4, 3, 2, 1, 0]
    finally:
        os.remove(path)


def test_subagent_runs_uses_table_aware_fields():
    """Pointing at subagent_runs must map ended_reason -> reward (not all-zero)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE subagent_runs (run_id TEXT, ended_reason TEXT, "
                    "accumulated_runtime_ms REAL, created_at INTEGER, model TEXT)")
        for rid, reason, rt, ca in [("a", "completed", 100.0, 1), ("b", "failed", 50.0, 2),
                                    ("c", "completed", 120.0, 3)]:
            con.execute("INSERT INTO subagent_runs VALUES (?,?,?,?,?)", (rid, reason, rt, ca, "m"))
        con.commit()
        con.close()
        src = SqliteOpenClawSource(path, table="subagent_runs")
        fac = OpenClawSubstrate(src).steppable()
        fac.run(3)
        scores = [r.score for r in sorted(fac.trace.rewards, key=lambda r: r.round)]
        assert scores == [1.0, 0.0, 1.0]                       # ended_reason mapped, not silently zero
    finally:
        os.remove(path)


def test_missing_status_column_raises_rather_than_scoring_zero():
    class _Mislabeled:
        table = "subagent_runs"          # resolves status_field='ended_reason'...
        def runs(self):
            return [{"run_id": "x", "status": "completed"}]   # ...but rows only have 'status'

    with pytest.raises(ValueError):
        OpenClawSubstrate(_Mislabeled()).steppable()
