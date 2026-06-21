"""The steering reference: the committed value and the surplus above it."""
from factory_design import reference


def test_committed_value_is_zero_on_the_canonical_game():
    assert abs(reference.steering(3, seeds=4).committed_value_V) < 1e-9
    assert abs(reference.steering(2, seeds=4).committed_value_V) < 1e-9


def test_mean_based_is_steered_above_no_swap_at_three_actions():
    r = reference.steering(3, seeds=6)
    assert r.meanbased_reached > r.committed_value_V + 0.2
    assert r.noswap_reached <= r.committed_value_V + 1e-6
    assert r.gap > 0.2


def test_gap_collapses_at_two_actions():
    r2 = reference.steering(2, seeds=6)
    # with two actions the surplus over V is gone (U* = V)
    assert r2.meanbased_reached <= r2.committed_value_V + 1e-6
