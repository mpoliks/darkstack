"""The OpenFang adapter: its mapping from run/metric JSON to the capability
interfaces, tested against recorded API responses (no live instance needed).
"""
import numpy as np
import pytest

from factory_probe.adapters.openfang import OpenFangSubstrate, OpenFangFactory
from factory_probe.tracks import versioning, catastrophe


def _telegraph(n=5000, lo=0.30, hi=0.70, hop=0.004, noise=0.02, seed=0):
    """A two-level series with rare switches -- two metastable behavioural modes."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    level = lo
    for t in range(n):
        if rng.random() < hop:
            level = hi if level == lo else lo
        x[t] = level + noise * rng.standard_normal()
    return x


def _runs_from_series(series, variety=3.0):
    runs = []
    for i, mp in enumerate(series):
        runs.append({
            "id": f"run-{i}",
            "metrics": {"mean_pos": float(mp), "norm_sat": 0.9, "metric_sat": 0.9,
                        "variety": variety},
            "steps": [{"hand": "Researcher", "action": 1, "n_actions": 3,
                       "logprob": -0.7, "score": 0.8}],
        })
    return runs


class RecordedClient:
    """Returns canned OpenFang responses and records every call made."""

    def __init__(self, runs):
        self._runs = runs
        self.calls = []

    def get(self, path, **params):
        self.calls.append(("GET", path, params))
        if path.endswith("/runs"):
            return {"runs": self._runs}
        return {}

    def put(self, path, body):
        self.calls.append(("PUT", path, body))
        return {"ok": True}


def test_capabilities_are_observational_only():
    sub = OpenFangSubstrate(RecordedClient([]), workflow_id="wf1")
    assert sub.capabilities() == {"steppable"}
    assert sub.supports("steppable")
    assert not sub.supports("learner_game")
    assert not sub.supports("dividend_task")


def test_replays_run_history_into_the_behavioural_series():
    series = _telegraph(n=400)
    client = RecordedClient(_runs_from_series(series))
    fac = OpenFangSubstrate(client, workflow_id="wf1").steppable()
    fac.run(len(series))
    got = fac.trace.behaviour_series("mean_pos")
    assert np.allclose(got, series, atol=1e-6)                 # the mapping is faithful
    assert any(c[1] == "/api/workflows/wf1/runs" for c in client.calls)   # hit the real endpoint


def test_versioning_detects_two_modes_through_the_adapter():
    series = _telegraph(n=5000)
    sub = OpenFangSubstrate(RecordedClient(_runs_from_series(series)), workflow_id="wf1")
    f = versioning.measure(sub, n=len(series), n_boxes=24, seed=0)
    assert f.track == "versioning"
    assert f.measured["n_metastable_0p95"] >= 2                # two behavioural versions, off live data


def test_set_price_puts_to_the_designated_trigger_only_when_wired():
    # no price channel: set_price records the intent but issues no API call
    plain = RecordedClient(_runs_from_series(_telegraph(n=50)))
    OpenFangFactory(plain, "wf1").reset().set_price(5.0)
    assert not any(c[0] == "PUT" for c in plain.calls)

    # wired: set_price PUTs the trigger value (a real control)
    wired_client = RecordedClient(_runs_from_series(_telegraph(n=50)))
    OpenFangFactory(wired_client, "wf1", price_setting="trg-9").reset().set_price(5.0)
    puts = [c for c in wired_client.calls if c[0] == "PUT"]
    assert puts and puts[0][1] == "/api/triggers/trg-9" and puts[0][2] == {"value": 5.0}


def test_catastrophe_runs_on_supplied_live_epochs():
    rng = np.random.default_rng(1)
    ramp = np.concatenate([0.3 + 0.02 * rng.standard_normal(5000),
                           np.linspace(0.3, 0.7, 200)])           # healthy then a fold
    null = 0.3 + 0.02 * rng.standard_normal(5200)                 # matched healthy epoch
    sub = OpenFangSubstrate(RecordedClient([]), workflow_id="wf1")
    f = catastrophe.measure(sub, ramp=ramp, null=null, fold_index=5000, w=400)
    assert f.track == "catastrophe" and "variance_trend" in f.measured
