"""CLI robustness: bad input fails cleanly, not with a traceback."""
import pytest

from factory_design.cli import main


def test_non_numeric_set_value_errors_cleanly():
    with pytest.raises(SystemExit) as e:
        main(["sim", "healthy", "--set", "explore_rate=banana", "--quick"])
    assert "must be a number" in str(e.value)


def test_empty_set_value_errors_cleanly():
    with pytest.raises(SystemExit) as e:
        main(["sim", "healthy", "--set", "explore_rate=", "--quick"])
    assert "must be a number" in str(e.value)


def test_seeds_must_be_positive():
    with pytest.raises(SystemExit) as e:
        main(["--seeds", "0", "sim", "healthy", "--quick"])
    assert "seeds" in str(e.value)


def test_unknown_preset_message_has_no_repr_quotes():
    with pytest.raises(SystemExit) as e:
        main(["sim", "boguspreset", "--quick"])
    msg = str(e.value)
    assert "unknown preset" in msg and not msg.lstrip().startswith('"')


def test_unknown_knob_errors_cleanly():
    with pytest.raises(SystemExit) as e:
        main(["sim", "healthy", "--set", "frobnicate=3", "--quick"])
    assert "unknown knob" in str(e.value)


def test_set_key_value_whitespace_is_stripped():
    # a copy-paste with spaces around '=' should still work, not be an unknown knob
    main(["sim", "healthy", "--set", "explore_rate = 0.1", "--quick"])


def test_sweep_trailing_comma_is_tolerated():
    # the everyday trailing-comma typo must not crash
    main(["sweep", "healthy", "explore_rate", "0.06,0.3,"])
