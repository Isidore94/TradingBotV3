"""`calc_anchored_vwap_bands` parity to 1e-9 (plan Phase 5, sec 19.3).

The frozen running-deviation sigma (decision 0008) is calibrated into every
existing consumer - events, zones, tracker families, scoring history. The
warehouse therefore **calls** the champion function and never re-derives it.
Two things are pinned here:

* the wrapper reproduces the champion's values exactly (1e-9) on golden
  fixtures and on generated frames, including the awkward inputs;
* the wrapper contains no sigma math of its own, so it cannot drift.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.research_warehouse import features  # noqa: E402

TOLERANCE = 1e-9


def _champion():
    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    return calc_anchored_vwap_bands


def _bars(values):
    return [
        {"open": o, "high": h, "low": low, "close": c, "volume": v}
        for o, h, low, c, v in values
    ]


def _frame(bars):
    import pandas as pd

    return pd.DataFrame(bars)


CASES = {
    "trending_up": _bars(
        [
            (100.0, 102.0, 99.5, 101.5, 1_000_000),
            (101.5, 104.0, 101.0, 103.5, 1_200_000),
            (103.5, 106.0, 103.0, 105.5, 900_000),
            (105.5, 108.0, 104.5, 107.0, 1_500_000),
            (107.0, 109.5, 106.0, 108.5, 1_100_000),
        ]
    ),
    "choppy": _bars(
        [
            (50.0, 51.0, 49.0, 50.5, 400_000),
            (50.5, 50.8, 48.5, 49.0, 650_000),
            (49.0, 52.0, 48.9, 51.8, 720_000),
            (51.8, 52.2, 50.1, 50.4, 310_000),
        ]
    ),
    "zero_volume_bars_skipped": _bars(
        [
            (10.0, 10.5, 9.5, 10.2, 0),
            (10.2, 11.0, 10.0, 10.8, 250_000),
            (10.8, 11.4, 10.6, 11.2, 0),
            (11.2, 11.9, 11.0, 11.6, 480_000),
        ]
    ),
    "single_bar": _bars([(7.0, 7.5, 6.8, 7.3, 100_000)]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_wrapper_matches_the_champion_exactly(case):
    bars = CASES[case]
    expected_vwap, expected_stdev, expected_bands = _champion()(_frame(bars), 0)
    vwap, stdev, bands = features.anchored_vwap_bands(bars, 0)

    assert vwap == pytest.approx(expected_vwap, abs=TOLERANCE)
    assert stdev == pytest.approx(expected_stdev, abs=TOLERANCE)
    assert set(bands) == set(expected_bands)
    for name, value in expected_bands.items():
        assert bands[name] == pytest.approx(value, abs=TOLERANCE)


@pytest.mark.parametrize("anchor_index", [0, 1, 2])
def test_parity_holds_from_every_anchor_index(anchor_index):
    bars = CASES["trending_up"]
    expected = _champion()(_frame(bars), anchor_index)
    actual = features.anchored_vwap_bands(bars, anchor_index)
    assert actual[0] == pytest.approx(expected[0], abs=TOLERANCE)
    assert actual[1] == pytest.approx(expected[1], abs=TOLERANCE)
    for name, value in expected[2].items():
        assert actual[2][name] == pytest.approx(value, abs=TOLERANCE)


def test_all_zero_volume_returns_the_champion_nan_contract():
    bars = _bars([(10.0, 10.5, 9.5, 10.2, 0), (10.2, 10.6, 10.0, 10.4, 0)])
    vwap, stdev, bands = features.anchored_vwap_bands(bars, 0)
    assert vwap != vwap and stdev != stdev and bands == {}  # NaN, NaN, {}
    assert features.anchored_vwap_bands([], 0)[2] == {}


def test_golden_fixture_pins_the_frozen_formula():
    """A committed expectation, so a silent champion formula change fails here.

    The fixture carries the Milestone-3 contract, so its declared tolerance
    (1e-9) is the one applied and its raw inputs are re-hashed on load: editing
    the bars without re-freezing the expectations fails at load time.
    """
    from conftest import load_fixture_contract

    contract = load_fixture_contract("warehouse_avwap_bands_v1")
    assert contract["feature_version"] == features.AVWAP_FORMULA_VERSION
    assert contract.tolerance == TOLERANCE

    expected_by_name = {case["name"]: case for case in contract["expected"]}
    for case in contract["cases_input"]:
        bars = _bars([tuple(row) for row in case["bars"]])
        vwap, stdev, bands = features.anchored_vwap_bands(bars, case["anchor_index"])
        expected = expected_by_name[case["name"]]
        contract.assert_matches(vwap, expected["vwap"], f"{case['name']}/vwap")
        contract.assert_matches(stdev, expected["stdev"], f"{case['name']}/stdev")
        contract.assert_matches(bands, expected["bands"], f"{case['name']}/bands")
        # And the champion agrees with the fixture too, so the two can never
        # drift apart unnoticed.
        champion = _champion()(_frame(bars), case["anchor_index"])
        contract.assert_matches(champion[0], expected["vwap"], f"{case['name']}/champion")


def test_the_wrapper_holds_no_sigma_math_of_its_own():
    """Never reimplemented: the wrapper shapes inputs and calls the champion."""
    tree = ast.parse(Path(features.__file__).read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "anchored_vwap_bands"
    )
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
    }
    assert "calc_anchored_vwap_bands" in calls
    # No accumulation, no square roots, no deviation loop anywhere in it.
    assert not any(isinstance(node, ast.For) for node in ast.walk(wrapper))
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) for node in ast.walk(wrapper)
    )
    source = Path(features.__file__).read_text(encoding="utf-8")
    assert "sqrt" not in source and "cumSD" not in source
