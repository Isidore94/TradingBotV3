from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import load_fixture_contract


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


FIXTURE_NAME = "laguerre_rsi_v1"


def _bars(contract):
    closes = [float(value) for value in contract["closes"]]
    padding = float(contract["bar_construction"]["high_low_padding"])
    volume_value = float(contract["bar_construction"]["volume"])
    opens = [closes[0], *closes[:-1]]
    highs = [max(open_value, close) + padding for open_value, close in zip(opens, closes)]
    lows = [min(open_value, close) - padding for open_value, close in zip(opens, closes)]
    volumes = [volume_value] * len(closes)
    return opens, highs, lows, closes, volumes


def _transitions_after(states, start):
    transitions = []
    prior = None
    for index, state in enumerate(states):
        if state != prior:
            if index >= start:
                transitions.append([index, state.value if state else None])
            prior = state
    return transitions


def _mean_absolute_change(values, start, end):
    changes = [abs(values[index] - values[index - 1]) for index in range(start + 1, end)]
    return sum(changes) / len(changes)


def test_laguerre_rsi_golden_trend_decay_reversal_path():
    from indicators.laguerre_rsi import LaguerreRsiConfig, compute_laguerre_rsi

    contract = load_fixture_contract(FIXTURE_NAME)
    opens, highs, lows, closes, volumes = _bars(contract)
    config = LaguerreRsiConfig(**contract["configuration"])
    result = compute_laguerre_rsi(opens, highs, lows, closes, volumes, config=config)

    assert result.feature_version == contract.feature_version
    assert len(result.oscillator) == len(closes)
    assert all(0.0 <= value <= 1.0 for value in result.oscillator)
    assert all(
        value is None or 0.0 <= value <= 1.0
        for value in result.fractal_energy
    )
    contract.assert_matches(
        _transitions_after(result.states, 20),
        contract["expected"]["transitions_after_warmup"],
        "state transitions",
    )
    selected = {
        index: result.oscillator[int(index)]
        for index in contract["expected"]["selected_oscillator"]
    }
    contract.assert_matches(
        selected,
        contract["expected"]["selected_oscillator"],
        "selected oscillator values",
    )


def test_fractal_energy_modulation_is_smoother_in_choppy_segment():
    from indicators.laguerre_rsi import LaguerreRsiConfig, compute_laguerre_rsi

    contract = load_fixture_contract(FIXTURE_NAME)
    bars = _bars(contract)
    adaptive = compute_laguerre_rsi(*bars, config=LaguerreRsiConfig())
    fixed = compute_laguerre_rsi(*bars, config=LaguerreRsiConfig(fixed_gamma=0.5))

    adaptive_change = _mean_absolute_change(adaptive.oscillator, 13, 18)
    fixed_change = _mean_absolute_change(fixed.oscillator, 13, 18)
    contract.assert_matches(
        adaptive_change,
        contract["expected"]["adaptive_chop_mad"],
        "adaptive choppy-segment movement",
    )
    contract.assert_matches(
        fixed_change,
        contract["expected"]["fixed_chop_mad"],
        "fixed-gamma choppy-segment movement",
    )
    assert adaptive_change < fixed_change
    assert all(value == pytest.approx(1.0) for value in adaptive.fractal_energy[13:18])


def test_classifier_requires_a_threshold_before_decay_and_uses_four_states():
    from indicators.laguerre_rsi import LaguerreState, classify_laguerre_states

    values = [0.5, 0.79, 0.81, 0.9, 0.7, 0.4, 0.19, 0.1, 0.3, 0.7, 0.81]
    states = classify_laguerre_states(values)

    assert states[:2] == (None, None)
    assert states[2:4] == (LaguerreState.TREND_UP, LaguerreState.TREND_UP)
    assert states[4:6] == (LaguerreState.DECAY_UP, LaguerreState.DECAY_UP)
    assert states[6:8] == (LaguerreState.TREND_DOWN, LaguerreState.TREND_DOWN)
    assert states[8:10] == (LaguerreState.DECAY_DOWN, LaguerreState.DECAY_DOWN)
    assert states[10] == LaguerreState.TREND_UP


def test_multitimeframe_wrapper_never_uses_partial_higher_bar():
    from indicators.laguerre_rsi import compute_multitimeframe_laguerre_rsi

    contract = load_fixture_contract(FIXTURE_NAME)
    bars = [values[:85] for values in _bars(contract)]
    baseline = compute_multitimeframe_laguerre_rsi(*bars, higher_factor=3)

    assert len(baseline.higher.oscillator) == 28
    assert baseline.higher_completed_base_indices[-1] == 83
    assert baseline.higher_oscillator_on_base[0:2] == (None, None)
    assert baseline.higher_oscillator_on_base[83] == baseline.higher.oscillator[-1]
    assert baseline.higher_oscillator_on_base[84] == baseline.higher.oscillator[-1]

    changed = [list(values) for values in bars]
    changed[0][-1] += 10.0
    changed[1][-1] += 10.0
    changed[2][-1] += 10.0
    changed[3][-1] += 10.0
    changed_result = compute_multitimeframe_laguerre_rsi(*changed, higher_factor=3)
    assert changed_result.higher == baseline.higher


@pytest.mark.parametrize(
    "kwargs",
    [
        {"price_source": "last"},
        {"fractal_energy_lookback": 1},
        {"lower_threshold": 0.8, "upper_threshold": 0.2},
        {"fixed_gamma": 1.0},
    ],
)
def test_invalid_configuration_fails_loudly(kwargs):
    from indicators.laguerre_rsi import LaguerreRsiConfig

    with pytest.raises(ValueError):
        LaguerreRsiConfig(**kwargs)
