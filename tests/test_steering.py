"""Each diagnosed condition has a boundary lever, and applying it on the reference
substrate resolves the condition (measure -> recommend -> re-measure)."""
from factory_probe.steering import recommend, run_all


def test_every_condition_has_a_lever():
    for c in ("learning_death", "thrash", "overfitting", "stable_failure", "condensate"):
        iv = recommend(c)
        assert iv.knob and iv.lever and iv.rationale


def test_closed_loop_resolves_each_condition():
    results = run_all()
    assert {r["condition"] for r in results} == {
        "learning_death", "thrash", "overfitting", "stable_failure", "condensate"}
    for r in results:
        assert r["resolved"], (r["condition"], r["metric"], r["before"], r["after"])
        assert r["intervention"]["condition"] == r["condition"]


def test_thrash_subband_nudge_does_not_resolve():
    """A within-band cadence nudge (3->4) never crosses the 3:1 cascade band, so it
    must NOT count as resolving thrash, since it never crosses the 3:1 cascade band."""
    from factory_probe.steering import _verify_thrash
    from factory_probe.mock import MockSubstrate
    sub = MockSubstrate()
    assert not _verify_thrash(sub, fast=3, slow=4)["resolved"]
    assert _verify_thrash(sub, fast=3, slow=60)["resolved"]
