"""A real adapter for OpenFang, the open-source Agent Operating System
(github.com/RightNow-AI/openfang).

OpenFang runs autonomous Hands on schedules, drives them through workflows, and
records every run to a Merkle audit chain with per-Hand dashboard metrics. This
adapter reads a running instance over its REST API and replays its workflow-run
history as the behavioural stream the measurement tracks consume:

    GET /api/workflows/{id}/runs   the run history (one run -> one round)

Each run carries dashboard metrics (mapped to the behavioural keys the tracks read)
and, when present, per-step action and log-probability (mapped to Decisions with
their decision-time propensity).

What it serves. Observation of a live instance gives a behavioural series, so the
versioning track runs directly, the catastrophe track runs when you hand it a real
pre-incident epoch and a healthy epoch, and the pathology classifier
(`tracks.pathology.fingerprint`/`classify`) names the condition of the observed
factory. The governance track needs a price channel and reset-and-replay, which a
passively observed production instance does not give; designate `price_setting` (an
OpenFang trigger whose value penalises the constrained behaviour) and drive a
controllable instance to run it.

What is tested. The mapping from OpenFang's documented run/metric JSON to the
capability interfaces is covered against recorded API responses (see
tests/test_openfang_adapter.py). Point `OpenFangSubstrate` at a real base URL with
a live `HttpOpenFangClient` to run against an instance; the SDK calls are the only
part a live run exercises beyond the tested mapping.
"""
from __future__ import annotations

import json
import math
import urllib.request

from ..interfaces import Substrate, SteppableFactory
from ..instrumentation import TraceStore
from ..records import Decision, Reward, RoundObs

# Default mapping from a run's dashboard-metric keys to the behavioural keys the
# tracks read. Override per instance to match your Hands' declared metrics.
_DEFAULT_METRIC_MAP = {"mean_pos": "mean_pos", "norm_sat": "norm_sat",
                       "metric_sat": "metric_sat", "variety": "variety"}


class HttpOpenFangClient:
    """Minimal HTTP client for a live OpenFang instance (stdlib only).

    Any object with `get(path, **params) -> dict` and `put(path, body) -> dict`
    works in its place; the adapter is tested with a recorded-response stand-in.
    """

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get(self, path: str, **params) -> dict:
        url = self.base_url + path
        if params:
            from urllib.parse import urlencode
            url += "?" + urlencode(params)
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def put(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method="PUT",
                                     headers={**self._headers(), "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())


def _as_runs(resp) -> list:
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        return resp.get("runs", resp.get("data", []))
    return []


class OpenFangFactory(SteppableFactory):
    """Replays a workflow's run history as a behavioural stream."""

    def __init__(self, client, workflow_id: str, metric_map: dict | None = None,
                 price_setting: str | None = None, propensity_field: str = "logprob",
                 score_field: str = "score"):
        self.client = client
        self.workflow_id = workflow_id
        self.metric_map = dict(metric_map or _DEFAULT_METRIC_MAP)
        self.price_setting = price_setting
        self.propensity_field = propensity_field
        self.score_field = score_field
        self.trace = TraceStore()
        self._round = 0
        self._price = 0.0
        self._runs: list = []
        self._cursor = 0

    # ---- live read ---------------------------------------------------------
    def _fetch_runs(self) -> list:
        return _as_runs(self.client.get(f"/api/workflows/{self.workflow_id}/runs"))

    def _metrics_of(self, run: dict) -> dict:
        m = run.get("metrics", {})
        return {key: float(m[src]) for key, src in self.metric_map.items() if src in m}

    def reset(self, **config) -> "OpenFangFactory":
        # config (regime/seed/n_sample) is a mock concept; a live instance is just
        # observed. Snapshot the run history so the replay is deterministic.
        self.trace = TraceStore()
        self._round = 0
        self._price = 0.0
        self._runs = self._fetch_runs()
        self._cursor = 0
        return self

    def behaviour(self) -> dict:
        if not self._runs:
            self._runs = self._fetch_runs()
        if not self._runs:
            return {}
        idx = min(self._cursor, len(self._runs) - 1)
        return self._metrics_of(self._runs[idx])

    def set_price(self, lam: float) -> None:
        self._price = float(lam)
        if self.price_setting is not None:
            # designate an OpenFang trigger whose value is the penalty weight
            self.client.put(f"/api/triggers/{self.price_setting}", {"value": float(lam)})
        # with no price_setting this records the intended price but does not steer:
        # a passively observed instance has no governance channel (see module docs).

    def injectable_channels(self) -> set:
        return {"spec_target"} if self.price_setting is not None else set()

    def step(self) -> RoundObs:
        if not self._runs:
            self._runs = self._fetch_runs()
        if self._cursor >= len(self._runs):
            raise RuntimeError(
                "no more OpenFang runs in the collected window; size n to the run "
                "history you have, or collect a longer window before measuring")
        run = self._runs[self._cursor]
        self._cursor += 1
        beh = self._metrics_of(run)
        rid = str(run.get("id", self._round))
        for j, st in enumerate(run.get("steps", [])):
            handle = f"{rid}:{j}"
            lp = st.get(self.propensity_field)
            prop = math.exp(lp) if isinstance(lp, (int, float)) else float("nan")
            self.trace.record_decision(Decision(
                handle=handle, round=self._round, agent=str(st.get("agent", st.get("hand", rid))),
                action=int(st.get("action", 0)), n_actions=int(st.get("n_actions", 1)),
                propensity=prop, behaviour=beh.get("mean_pos", float("nan"))))
            sc = st.get(self.score_field)
            if isinstance(sc, (int, float)):
                self.trace.record_reward(Reward(handle=handle, round=self._round, score=float(sc)))
        self.trace.record_behaviour(self._round, beh)
        self.trace.record_control(self._round, {"price": self._price})
        obs = RoundObs(round=self._round, behaviour=beh, control={"price": self._price})
        self._round += 1
        return obs


class OpenFangSubstrate(Substrate):
    """An OpenFang instance as a measurement substrate (observational tracks)."""

    name = "openfang"

    def __init__(self, client, workflow_id: str, **factory_kwargs):
        self.client = client
        self.workflow_id = workflow_id
        self.factory_kwargs = factory_kwargs

    def capabilities(self) -> set:
        return {"steppable"}

    def steppable(self, **config) -> SteppableFactory:
        return OpenFangFactory(self.client, self.workflow_id, **self.factory_kwargs).reset(**config)
