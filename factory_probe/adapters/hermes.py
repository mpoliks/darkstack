"""A real adapter for Hermes, the NousResearch self-improving agent
(github.com/NousResearch/hermes-agent, docs hermes-agent.nousresearch.com).

Hermes records every session to a SQLite store and serves it over an HTTP gateway.
The past-runs stream is the sessions surface (there is no runs-list endpoint):

    GET /api/sessions                 the session roster (one session -> one round)
    GET /api/sessions/{id}/messages   the per-step actions (assistant tool calls)
    GET /v1/toolsets                  the enumerable toolset (the action set size)

Per-session scalars confirmed in the gateway's response (gateway/platforms/
api_server.py `_session_response` safe_keys) and the store (hermes_state.py
`sessions` table): message_count, tool_call_count, input_tokens, output_tokens,
started_at, ended_at, end_reason.

What it serves. Observation gives a behavioural series, so the observational tracks
run (versioning, catastrophe with supplied epochs, the pathology classifier on the
observed trace). The behavioural keys are NOT native to Hermes -- it reports tokens,
tool counts, and statuses, not a dark-stack coordinate -- so they are derived
(adapters/_derived.py) and labelled as proxies. Decision-time propensity is not
recoverable (the gateway's `logprobs` field is an always-empty placeholder), so it
is stubbed uniform and flagged; reward is the binary completion status; there is no
price channel, so governance is unsupported.

Tested against recorded gateway JSON (tests/test_hermes_adapter.py); point
`HttpHermesClient` at a real base URL to run against an instance.
"""
from __future__ import annotations

import json
import urllib.request

from ..interfaces import Substrate, SteppableFactory
from ..instrumentation import TraceStore
from ..records import Decision, Reward, RoundObs
from ._derived import derive_series


class HttpHermesClient:
    """Minimal HTTP client for a live Hermes gateway (stdlib only). Any object with
    `get(path, **params) -> dict|list` works in its place."""

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def get(self, path: str, **params):
        url = self.base_url + path
        if params:
            from urllib.parse import urlencode
            url += "?" + urlencode(params)
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())


def _unwrap(resp, *keys) -> list:
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for k in keys:
            if isinstance(resp.get(k), list):
                return resp[k]
    return []


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class HermesFactory(SteppableFactory):
    """Replays a Hermes session roster as a behavioural stream."""

    def __init__(self, client, limit: int = 500, fetch_messages: bool = True,
                 toolset: list | None = None, success_end_reasons=("completed",),
                 window: int = 200):
        self.client = client
        self.limit = int(limit)
        self.fetch_messages = bool(fetch_messages)
        self.toolset = list(toolset) if toolset else None
        self.success = set(success_end_reasons)
        self.window = int(window)
        self.trace = TraceStore()
        self._round = 0
        self._price = 0.0
        self._sessions: list = []
        self._behaviour: list = []
        self._tools_per: list = []     # tool-name list per session (for decisions)
        self._cursor = 0

    # ---- live read ---------------------------------------------------------
    def _fetch_sessions(self) -> list:
        rows = _unwrap(self.client.get("/api/sessions", limit=self.limit), "sessions", "data")
        # the store indexes started_at DESC; replay oldest-first
        return sorted(rows, key=lambda s: _num(s.get("started_at")) or 0.0)

    def _session_tools(self, sid) -> list:
        msgs = _unwrap(self.client.get(f"/api/sessions/{sid}/messages"), "messages", "data")
        tools = []
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            if m.get("tool_name"):
                tools.append(m["tool_name"])
            for call in m.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name") or call.get("name")
                if name:
                    tools.append(name)
        return tools

    def _features(self) -> list:
        feats = []
        self._tools_per = []
        for s in self._sessions:
            mc = _num(s.get("message_count")) or 0.0
            tc = _num(s.get("tool_call_count")) or 0.0
            f = {"mean_pos": (tc / mc) if mc > 0 else 0.0,
                 "success": (s.get("end_reason") in self.success)}
            started, ended = _num(s.get("started_at")), _num(s.get("ended_at"))
            if started is not None and ended is not None and ended >= started:
                f["duration_ms"] = (ended - started) * 1000.0
            tools = self._session_tools(s["id"]) if self.fetch_messages and "id" in s else []
            self._tools_per.append(tools)
            if tools:
                f["tools"] = tools
            feats.append(f)
        return feats

    def reset(self, **config) -> "HermesFactory":
        self.trace = TraceStore()
        self._round = 0
        self._price = 0.0
        self._sessions = self._fetch_sessions()
        if self.toolset is None:
            self.toolset = _unwrap(self._safe_get("/v1/toolsets"), "toolsets", "data") or None
        self._behaviour = derive_series(self._features(), window=self.window)
        self._cursor = 0
        return self

    def _safe_get(self, path):
        try:
            return self.client.get(path)
        except Exception:
            return []

    def behaviour(self) -> dict:
        if not self._behaviour:
            self.reset()
        idx = min(self._cursor, len(self._behaviour) - 1) if self._behaviour else 0
        return dict(self._behaviour[idx]) if self._behaviour else {}

    def set_price(self, lam: float) -> None:
        # Hermes has no runtime price channel (pricing is static config); record the
        # intended price for the control series, but it does not steer the factory.
        self._price = float(lam)

    def _n_actions(self, observed_tools: list) -> int:
        if self.toolset:
            return max(1, len(self.toolset))
        return max(1, len(set(observed_tools)))

    def run(self, n: int):
        # observe up to n rounds, capped at the fetched session window.
        if not self._sessions:
            self.reset()
        return super().run(min(n, len(self._sessions)))

    def step(self) -> RoundObs:
        if not self._sessions:
            self.reset()
        if self._cursor >= len(self._sessions):
            raise RuntimeError("no more Hermes sessions in the fetched window; raise limit "
                               "or collect a longer history before measuring")
        i = self._cursor
        self._cursor += 1
        beh = dict(self._behaviour[i])
        sess = self._sessions[i]
        sid = str(sess.get("id", i))
        tools = self._tools_per[i] if i < len(self._tools_per) else []
        n_act = self._n_actions(tools)
        last_handle = None
        for j, name in enumerate(tools):
            # a tool outside the known toolset maps to a sentinel index (n_actions,
            # out of the valid [0, n_actions) range) so it does not collide with the
            # first toolset entry; with no toolset, the index is positional.
            if self.toolset:
                action = self.toolset.index(name) if name in self.toolset else len(self.toolset)
            else:
                action = j
            last_handle = f"{sid}:{j}"
            self.trace.record_decision(Decision(
                handle=last_handle, round=self._round, agent=str(sess.get("model", sid)),
                action=int(action), n_actions=n_act,
                propensity=1.0 / n_act,              # uniform stub: no logprob is exposed
                behaviour=beh.get("mean_pos", float("nan"))))
        if last_handle is not None:
            self.trace.record_reward(Reward(
                handle=last_handle, round=self._round,
                score=1.0 if sess.get("end_reason") in self.success else 0.0))
        self.trace.record_behaviour(self._round, beh)
        self.trace.record_control(self._round, {"price": self._price})
        obs = RoundObs(round=self._round, behaviour=beh, control={"price": self._price})
        self._round += 1
        return obs


class HermesSubstrate(Substrate):
    """A Hermes gateway as a measurement substrate (observational tracks)."""

    name = "hermes"

    def __init__(self, client, **factory_kwargs):
        self.client = client
        self.factory_kwargs = factory_kwargs

    def capabilities(self) -> set:
        return {"steppable"}

    def steppable(self, **config) -> SteppableFactory:
        return HermesFactory(self.client, **self.factory_kwargs).reset(**config)
