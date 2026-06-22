"""The Hermes adapter: mapping the sessions roster to the capability interfaces,
tested against recorded gateway JSON (no live gateway needed)."""
import numpy as np

from factory_probe.adapters.hermes import HermesSubstrate, HermesFactory
from factory_probe.tracks import versioning


def _telegraph_ratio(n=5000, lo=0.2, hi=0.8, hop=0.004, seed=0):
    rng = np.random.default_rng(seed)
    out, level = [], lo
    for _ in range(n):
        if rng.random() < hop:
            level = hi if level == lo else lo
        out.append(level)
    return out


def _sessions(ratios, mc=10):
    return [{"id": f"s{i}", "model": "hermes-x", "message_count": mc,
             "tool_call_count": int(round(r * mc)), "end_reason": "completed",
             "started_at": float(i), "ended_at": float(i) + 0.5}
            for i, r in enumerate(ratios)]


class RecordedHermesClient:
    def __init__(self, sessions, messages=None, toolset=None):
        self.sessions = sessions
        self.messages = messages or {}
        self.toolset = toolset
        self.calls = []

    def get(self, path, **params):
        self.calls.append((path, params))
        if path == "/api/sessions":
            return {"sessions": self.sessions}
        if path.endswith("/messages"):
            sid = path.split("/")[-2]
            return {"messages": self.messages.get(sid, [])}
        if path == "/v1/toolsets":
            return {"toolsets": self.toolset or []}
        return {}


def test_capabilities_are_observational_only():
    sub = HermesSubstrate(RecordedHermesClient([]))
    assert sub.capabilities() == {"steppable"}
    assert sub.supports("steppable") and not sub.supports("learner_game")


def test_tool_density_becomes_the_behavioural_coordinate():
    ratios = _telegraph_ratio(n=300)
    client = RecordedHermesClient(_sessions(ratios))
    fac = HermesSubstrate(client, fetch_messages=False).steppable()
    fac.run(len(ratios))
    got = fac.trace.behaviour_series("mean_pos")
    assert np.allclose(got, ratios, atol=1e-6)                       # tool-density mapped faithfully
    assert any(c[0] == "/api/sessions" for c in client.calls)        # hit the real endpoint


def test_versioning_detects_two_modes_through_the_adapter():
    ratios = _telegraph_ratio(n=5000)
    sub = HermesSubstrate(RecordedHermesClient(_sessions(ratios)), fetch_messages=False)
    f = versioning.measure(sub, n=len(ratios), n_boxes=24, seed=0)
    assert f.measured["n_metastable_0p95"] >= 2


def test_messages_become_decisions_with_a_stubbed_propensity():
    sessions = _sessions([0.5] * 3)
    messages = {"s0": [{"role": "assistant", "tool_name": "write_file"},
                       {"role": "assistant", "tool_calls": [{"function": {"name": "run_tests"}}]}]}
    client = RecordedHermesClient(sessions, messages=messages,
                                  toolset=["read_file", "write_file", "run_tests"])
    fac = HermesSubstrate(client, fetch_messages=True).steppable()
    fac.run(3)
    decs = [d for d in fac.trace.decisions if d.handle.startswith("s0:")]
    assert {d.action for d in decs} == {1, 2}                        # write_file -> 1, run_tests -> 2
    assert all(d.n_actions == 3 and abs(d.propensity - 1 / 3) < 1e-9 for d in decs)
    # a completion reward was attached to the session's last decision
    assert any(r.score == 1.0 for r in fac.trace.rewards)


def test_set_price_is_recorded_but_does_not_call_the_gateway():
    client = RecordedHermesClient(_sessions([0.5] * 5))
    fac = HermesSubstrate(client, fetch_messages=False).steppable()
    fac.set_price(7.0)
    assert fac._price == 7.0                                         # no price channel exists; recorded only
