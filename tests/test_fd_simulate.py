"""simulate(): report shape, determinism, and headline verdicts."""
import pytest

from factory_design import FactoryDesign, simulate
from factory_design.presets import get


def test_report_shape_and_provenance():
    rep = simulate(FactoryDesign(), seeds=4, quick=True, early_warning=False)
    assert rep.verdict in ("healthy", "stable_failure", "overfitting", "learning_death", "thrash")
    assert "pathology" in rep.lenses and "versions" in rep.lenses
    assert rep.provenance["seeds"] == 4
    # every lens carries its provenance fields
    for lens in rep.lenses.values():
        assert lens.summary


def test_seed_determinism():
    a = simulate(FactoryDesign(), seeds=4, quick=True, early_warning=False)
    b = simulate(FactoryDesign(), seeds=4, quick=True, early_warning=False)
    assert a.verdict == b.verdict
    assert a.lenses["pathology"].metrics["fingerprint"] == b.lenses["pathology"].metrics["fingerprint"]


@pytest.mark.parametrize("preset,expected", [
    ("healthy", "healthy"),
    ("gameable_judge", "overfitting"),
    ("no_exploration_floor", "learning_death"),
    ("stuck_failing", "stable_failure"),
    ("never_settles", "thrash"),
])
def test_preset_verdicts(preset, expected):
    rep = simulate(get(preset), seeds=6, quick=True, early_warning=False)
    assert rep.verdict == expected


def test_governance_lens_flags_fast_repricing():
    from factory_design import diagnose
    rep = simulate(get("repricing_too_fast"), seeds=4, quick=True, early_warning=False)
    gov = rep.lenses["governance"]
    # the robust signal is the measured price oscillation, and diagnose must flag it
    assert gov.metrics["price_instability"][0] > 1.0
    assert "iatrogenic_thrash" in {c.name for c in diagnose(rep)}


def test_ecology_lens_only_with_peers():
    assert "ecology" not in simulate(FactoryDesign(), seeds=2, quick=True, early_warning=False).lenses
    rep = simulate(get("monoculture"), seeds=2, quick=True, early_warning=False)
    assert rep.lenses["ecology"].metrics["synchronisation"] > 0.5
