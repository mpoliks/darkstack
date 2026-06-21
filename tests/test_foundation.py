"""Foundation: records, trace store, return queue, scorer, substrate contract."""
import numpy as np
import pytest

from factory_probe import (Decision, Reward, TraceStore, ReturnQueue,
                           ConsequenceScorer, Substrate)


def test_tracestore_join_lags_propensity():
    ts = TraceStore()
    ts.record_decision(Decision("h1", 0, "a", action=1, n_actions=3, propensity=0.6))
    ts.record_decision(Decision("h2", 1, "a", action=0, n_actions=3, propensity=0.2))
    ts.record_reward(Reward("h1", 0, 1.0, kind="in_round"))
    ts.record_reward(Reward("h1", 5, 0.5, kind="realized"))
    ts.record_reward(Reward("h2", 4, 0.0, kind="realized"))

    joined = dict((d.handle, s) for d, s in ts.joined("realized"))
    assert joined == {"h1": 0.5, "h2": 0.0}
    assert dict((d.handle, s) for d, s in ts.joined("in_round")) == {"h1": 1.0}

    lags = sorted(ts.lags("realized").tolist())
    assert lags == [3.0, 5.0]                       # h2: 4-1, h1: 5-0

    prop, sc = ts.propensity_reward("realized")
    assert len(prop) == 2 and set(np.round(prop, 1)) == {0.6, 0.2}


def test_tracestore_behaviour_and_control_series():
    ts = TraceStore()
    for t in range(4):
        ts.record_behaviour(t, dict(mean_pos=0.1 * t))
        ts.record_control(t, dict(price=float(t)))
    assert np.allclose(ts.behaviour_series("mean_pos"), [0.0, 0.1, 0.2, 0.3])
    assert np.allclose(ts.control_series("price"), [0.0, 1.0, 2.0, 3.0])
    assert np.isnan(ts.behaviour_series("absent_key")).all()


def test_returnqueue_delay_and_jitter():
    q = ReturnQueue(delay=3)
    q.enqueue("a", 0); q.enqueue("b", 2)
    assert q.due(1) == [] and q.due(3) == ["a"] and q.due(5) == ["b"]
    assert q.outstanding() == 0

    qj = ReturnQueue(delay=2, jitter=4, rng=np.random.default_rng(0))
    dues = [qj.enqueue(f"h{i}", 0) for i in range(200)]
    assert min(dues) >= 2 and max(dues) <= 6           # delay + [0, jitter]


def test_scorer_recovers_and_handles_empty():
    sc = ConsequenceScorer()
    assert np.isnan(sc.precision()) and np.isnan(sc.recall())   # zero-denominator
    for _ in range(70):
        sc.grade(True, True)
    for _ in range(30):
        sc.grade(True, False)
    assert abs(sc.precision() - 0.7) < 1e-9
    assert sc.recall() == 1.0


def test_bare_substrate_reports_no_capabilities():
    class Bare(Substrate):
        name = "bare"
    b = Bare()
    assert b.capabilities() == set() and not b.supports("steppable")
    with pytest.raises(NotImplementedError):
        b.steppable()
    with pytest.raises(NotImplementedError):
        b.learner_game()
