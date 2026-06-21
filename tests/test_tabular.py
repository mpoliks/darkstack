"""The opacity track runs unchanged on a real-data substrate (off the cube)."""
from factory_probe.substrates.tabular import TabularSubstrate, TabularDividendTask
from factory_probe.tracks import opacity


def test_tabular_substrate_capabilities():
    sub = TabularSubstrate()
    assert sub.capabilities() == {"dividend_task"}
    assert not sub.supports("steppable")
    assert sub.default_dividend_specs()


def test_opacity_runs_on_tabular_real_data():
    f = opacity.measure(TabularSubstrate(),
                        task_specs=[{"synthetic_order": o} for o in (1, 2, 4)], n_seeds=2)
    assert f.confirms
    assert f.measured["dividends"][0] < 0.15 < f.measured["dividends"][-1]


def test_separable_real_dataset_pays_near_zero():
    db = TabularDividendTask(dataset="diabetes")          # diabetes is ~additive
    div = db.free_floor(None) - db.legible_floor(None, 1)
    assert div < 0.05
    assert db.interaction_order() <= 2
