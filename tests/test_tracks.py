"""Each track recovers the planted signal, and reports it absent on a control."""
from factory_probe.mock import MockSubstrate
from factory_probe.tracks import (versioning, pathology, catastrophe, governance,
                                  entrainment, stackelberg, opacity)
from factory_probe.tracks.pathology import classify


def sub():
    return MockSubstrate()


def test_versioning_present_on_versions_absent_on_single_basin():
    pos = versioning.measure(sub(), regime="versions", n=20000, seed=2)
    neg = versioning.measure(sub(), regime="single_basin", n=20000, seed=2)
    assert pos.confirms and pos.measured["timescale_sep"] >= 5
    assert not neg.confirms and neg.measured["timescale_sep"] < 5


def test_pathology_names_every_regime():
    f = pathology.measure(sub(), n=15000, seed=1)
    assert f.confirms and f.measured["correct"] == 5


def test_classify_rules():
    assert classify(dict(volatility=0.0, variety=3, norm_unmet=0.0, proxy_gap=0.9)) == "overfitting"
    assert classify(dict(volatility=0.0, variety=1.0, norm_unmet=0.6, proxy_gap=0.0)) == "learning_death"
    assert classify(dict(volatility=0.2, variety=4, norm_unmet=0.4, proxy_gap=0.0)) == "thrash"
    assert classify(dict(volatility=0.0, variety=3, norm_unmet=0.8, proxy_gap=0.0)) == "stable_failure"
    assert classify(dict(volatility=0.0, variety=3, norm_unmet=0.1, proxy_gap=0.0)) == "healthy"


def test_catastrophe_specific_warning():
    f = catastrophe.measure(sub(), seed=3)
    assert f.confirms
    assert f.measured["variance_trend"] > f.measured["null_variance_trend"]


def test_governance_declines_through_cascade_band():
    f = governance.measure(sub(), seeds=(0, 1), T_gov=(3, 12, 48, 200), n=1500)
    assert f.confirms and f.measured["mean_above_band"] < f.measured["mean_below_band"]


def test_entrainment_threshold_and_condensate():
    f = entrainment.measure(sub(), N=250)
    assert f.confirms
    assert f.measured["onset_high_div"] > f.measured["onset_low_div"]
    assert f.measured["r_after_halt"] > f.measured["r_after_decouple"]


def test_stackelberg_gap_and_boundary():
    f = stackelberg.measure(sub(), T=6000)
    assert f.confirms
    assert f.measured["frontier_extracted"] > f.measured["V"]
    assert f.measured["extracted_n3"] > f.measured["extracted_n2"]   # surplus needs >= 3 actions


def test_opacity_floor_rises_with_order():
    f = opacity.measure(sub(), task_specs=[dict(d=10, order=o, kind="nk") for o in (1, 2, 4)],
                        n_seeds=5)
    assert f.confirms
    assert f.measured["dividends"][0] < 0.15 < f.measured["dividends"][-1]


def test_opacity_rejects_tied_measured_orders():
    """Rising dividends with a flat measured-order axis must NOT confirm: a flat
    order axis carries no trend to read."""
    from factory_probe.interfaces import Substrate, DividendTask

    class TiedTask(DividendTask):
        def __init__(self, level):
            self.level = level

        def free_floor(self, budget, seed=0):
            return 1.0

        def legible_floor(self, budget, order=1, seed=0):
            return max(0.0, 1.0 - 0.4 * self.level)   # dividend rises with level

        def interaction_order(self, seed=0):
            return 1                                  # but measured order is always tied

    class TiedSub(Substrate):
        name = "tied"

        def capabilities(self):
            return {"dividend_task"}

        def dividend_task(self, **spec):
            return TiedTask(spec.get("level", 0))

    f = opacity.measure(TiedSub(), task_specs=[{"level": 0}, {"level": 1}, {"level": 2}],
                        n_seeds=1, budget=8)
    assert not f.confirms
    assert f.measured["order_spread"] == 1
