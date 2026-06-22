"""A real adapter for OpenClaw, the local-first agent gateway
(github.com/openclaw/openclaw).

OpenClaw keeps its runtime state in a SQLite control-plane database
(~/.openclaw/state/openclaw.sqlite, since the state-refactor in PR #78595). The
cleanest past-runs stream is the `cron_run_logs` table: one row per scheduled run
of a job, and a repeating `job_id` gives a time-ordered behavioural series.

Columns read, by table (documented in the OpenClaw state schema, PR #78595):
  cron_run_logs   job_id, seq, ts, status, duration_ms, total_tokens, model, run_id
  subagent_runs   run_id, ended_reason, accumulated_runtime_ms, created_at, model
The two tables expose different columns, so the field mapping is table-aware
(`_TABLE_DEFAULTS`); pointing the source at a table whose status column is absent
raises rather than silently scoring every run a failure.

What it serves. Reading the run history gives a behavioural series, so the
observational tracks run (versioning, catastrophe with supplied epochs, the
pathology classifier on the observed trace). The behavioural keys are NOT native
-- OpenClaw records tokens, durations, and a status enum, so they are derived
(adapters/_derived.py) and labelled as proxies. Neither table carries a tool-name
column, so the variety dimension of the pathology fingerprint (and its
learning-death branch) is unavailable here; only volatility, norm-unmet, and the
metric/norm gap are observed. Decision-time propensity is not stored anywhere, so
it is a flagged uniform stub; reward is the binary completion status; pricing is
config-only with no runtime price channel, so governance is unsupported.

The SQLite read is the only thing a live run exercises beyond the tested mapping;
the row-to-record mapping is covered against an in-memory database fixture and a
recorded-row source (tests/test_openclaw_adapter.py).
"""
from __future__ import annotations

import sqlite3

from ..interfaces import Substrate, SteppableFactory
from ..instrumentation import TraceStore
from ..records import Decision, Reward, RoundObs
from ._derived import derive_series

# Per-table column mapping. Each table reports different fields; token/duration are
# None when the table has no such column (the corresponding derived key is omitted).
_TABLE_DEFAULTS = {
    "cron_run_logs": dict(status_field="status", token_field="total_tokens",
                          duration_field="duration_ms", order="ts, seq"),
    "subagent_runs": dict(status_field="ended_reason", token_field=None,
                          duration_field="accumulated_runtime_ms", order="created_at"),
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class SqliteOpenClawSource:
    """Reads a run table from an OpenClaw control-plane SQLite database.

    Any object with `table` and `runs() -> list[dict]` works in its place; the
    adapter is tested against a recorded-row source and an in-memory database.
    """

    def __init__(self, db_path: str, table: str = "cron_run_logs",
                 job_id: str | None = None, limit: int | None = None):
        if table not in _TABLE_DEFAULTS:
            raise ValueError(f"unknown table {table!r}; known: {sorted(_TABLE_DEFAULTS)}")
        self.db_path = db_path
        self.table = table
        self.job_id = job_id
        self.limit = limit

    def runs(self) -> list[dict]:
        con = sqlite3.connect(self.db_path)
        try:
            con.row_factory = sqlite3.Row
            order = _TABLE_DEFAULTS[self.table]["order"]
            sql = f"SELECT * FROM {self.table}"
            params: tuple = ()
            if self.job_id is not None and self.table == "cron_run_logs":
                sql += " WHERE job_id = ?"
                params = (self.job_id,)
            sql += f" ORDER BY {order} ASC"
            if self.limit:
                sql += f" LIMIT {int(self.limit)}"
            return [dict(r) for r in con.execute(sql, params).fetchall()]
        finally:
            con.close()


class OpenClawFactory(SteppableFactory):
    """Replays an OpenClaw run table as a behavioural stream."""

    def __init__(self, source, success_statuses=("completed",),
                 status_field: str | None = None, token_field: str | None = None,
                 duration_field: str | None = None, window: int = 200):
        self.source = source
        self.success = set(success_statuses)
        self._explicit = dict(status_field=status_field, token_field=token_field,
                              duration_field=duration_field)
        self.window = int(window)
        self.trace = TraceStore()
        self._round = 0
        self._price = 0.0
        self._runs: list = []
        self._behaviour: list = []
        self._cursor = 0
        # resolved at reset() from the source's table
        self.status_field = "status"
        self.token_field: str | None = "total_tokens"
        self.duration_field: str | None = "duration_ms"

    def _resolve_fields(self):
        table = getattr(self.source, "table", "cron_run_logs")
        d = _TABLE_DEFAULTS.get(table, _TABLE_DEFAULTS["cron_run_logs"])
        e = self._explicit
        self.status_field = e["status_field"] or d["status_field"]
        self.token_field = e["token_field"] if e["token_field"] is not None else d["token_field"]
        self.duration_field = e["duration_field"] if e["duration_field"] is not None else d["duration_field"]

    def _features(self) -> list:
        feats = []
        for r in self._runs:
            f = {"success": (r.get(self.status_field) in self.success)}
            if self.token_field:
                tok = _num(r.get(self.token_field))
                if tok is not None:
                    f["tokens"] = tok
            if self.duration_field:
                dur = _num(r.get(self.duration_field))
                if dur is not None:
                    f["duration_ms"] = dur
            feats.append(f)
        return feats

    def reset(self, **config) -> "OpenClawFactory":
        self.trace = TraceStore()
        self._round = 0
        self._price = 0.0
        self._runs = list(self.source.runs())
        self._resolve_fields()
        if self._runs and self.status_field not in self._runs[0]:
            raise ValueError(
                f"status_field {self.status_field!r} is not a column in these rows "
                f"(columns: {sorted(self._runs[0])}); pass status_field= for this table")
        self._behaviour = derive_series(self._features(), window=self.window)
        self._cursor = 0
        return self

    def behaviour(self) -> dict:
        if not self._behaviour:
            self.reset()
        idx = min(self._cursor, len(self._behaviour) - 1) if self._behaviour else 0
        return dict(self._behaviour[idx]) if self._behaviour else {}

    def set_price(self, lam: float) -> None:
        # OpenClaw pricing is static config; there is no runtime price channel.
        self._price = float(lam)

    def run(self, n: int):
        # observe up to n rounds, capped at the available table window.
        if not self._runs:
            self.reset()
        return super().run(min(n, len(self._runs)))

    def step(self) -> RoundObs:
        if not self._runs:
            self.reset()
        if self._cursor >= len(self._runs):
            raise RuntimeError("no more OpenClaw runs in the table window; raise limit or "
                               "collect a longer history before measuring")
        i = self._cursor
        self._cursor += 1
        run = self._runs[i]
        beh = dict(self._behaviour[i])
        rid = str(run.get("run_id", run.get("seq", i)))
        # the run tables are run-level logs (no per-tool steps): one coarse decision
        # per run, with a flagged uniform-stub propensity (no logprob is stored).
        handle = f"{rid}:0"
        self.trace.record_decision(Decision(
            handle=handle, round=self._round, agent=str(run.get("model", "openclaw")),
            action=0, n_actions=1, propensity=1.0,
            behaviour=beh.get("mean_pos", float("nan"))))
        self.trace.record_reward(Reward(
            handle=handle, round=self._round,
            score=1.0 if run.get(self.status_field) in self.success else 0.0))
        self.trace.record_behaviour(self._round, beh)
        self.trace.record_control(self._round, {"price": self._price})
        obs = RoundObs(round=self._round, behaviour=beh, control={"price": self._price})
        self._round += 1
        return obs


class OpenClawSubstrate(Substrate):
    """An OpenClaw instance (its control-plane DB) as a measurement substrate."""

    name = "openclaw"

    def __init__(self, source, **factory_kwargs):
        self.source = source
        self.factory_kwargs = factory_kwargs

    def capabilities(self) -> set:
        return {"steppable"}

    def steppable(self, **config) -> SteppableFactory:
        return OpenClawFactory(self.source, **self.factory_kwargs).reset(**config)
