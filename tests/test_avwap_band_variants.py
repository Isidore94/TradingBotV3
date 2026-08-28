"""Phase 0.10 B-0 - the OneOption anchored-VWAP band, held to the trader's hovers.

The expectations here are readings the trader took off OneOption / Option
Stalker Pro on 2026-08-26, not this repository's output. That matters: a golden
value produced by the code under test proves only that the code agrees with
itself, and this module exists to replicate somebody else's formula.

Three things are asserted, and the second is the load-bearing one:

1. the replicated formula reproduces both hover readings;
2. the champion (`calc_anchored_vwap_bands`) and the killed sample-OHLC form
   both give DIFFERENT answers on the same bars - so a future edit that drifts
   toward either is caught here rather than in a study six weeks later;
3. the module cannot reach the champion at all (AST check on its imports).

`docs/AVWAP_BAND_VARIANT_STUDY.md` section 2b is the governing record.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from indicators.avwap_band_variants import (  # noqa: E402
    FEATURE_VERSION,
    oneoption_avwap_band_series,
    oneoption_avwap_bands,
)
from master_avwap_lib.legacy import calc_anchored_vwap_bands  # noqa: E402

FIXTURE = "avwap_band_variant_oneoption_v1"
MODULE_PATH = SCRIPTS_DIR / "indicators" / "avwap_band_variants.py"


@pytest.fixture(scope="module")
def contract():
    return load_fixture_contract(FIXTURE)


@pytest.fixture(scope="module")
def bars(contract):
    return list(contract["bars"])


@pytest.fixture(scope="module")
def rules(contract):
    return contract["rules_under_test"]


def _index_of(bars, date: str) -> int:
    for index, bar in enumerate(bars):
        if bar["date"] == date:
            return index
    raise AssertionError(f"{date} is not a session in the fixture")


def _anchor(bars, rules) -> int:
    return _index_of(bars, rules["anchor_date"])


# ---------------------------------------------------------------------------
# 1. The golden values.
# ---------------------------------------------------------------------------


def test_the_module_declares_the_replicated_formula_version():
    assert FEATURE_VERSION == "avwap_bands_oneoption_bb20_v1"


def test_the_hover_readings_reproduce(contract, bars, rules):
    """Both of the trader's OneOption readings, centre and +/-1 sigma."""
    anchor = _anchor(bars, rules)
    series = oneoption_avwap_band_series(
        bars, anchor, lookback=rules["lookback"], ddof=rules["ddof"]
    )
    centre_tolerance = rules["centre_relative_tolerance"]
    sigma_tolerance = rules["sigma_absolute_tolerance"]

    for reading in contract["expected"]["readings"]:
        index = _index_of(bars, reading["date"])
        centre = series["centre"][index]
        sigma = series["sigma"][index]
        assert centre is not None and sigma is not None, reading["date"]

        # The centre carries a RELATIVE tolerance because the only difference
        # left is a volume feed (consolidated vs IB), and a volume difference
        # scales with price rather than sitting at a fixed number of cents.
        assert abs(centre - reading["centre"]) / reading["centre"] <= centre_tolerance, (
            f"{reading['date']} centre {centre} vs vendor {reading['centre']}"
        )

        # The sigma uses no volume at all, so it must reproduce absolutely.
        vendor_sigma = (reading["upper_1"] - reading["lower_1"]) / 2.0
        assert abs(sigma - vendor_sigma) <= sigma_tolerance, (
            f"{reading['date']} sigma {sigma} vs vendor {vendor_sigma}"
        )

        # And the bands themselves, on the vendor's own centre, so the band
        # assertion is not silently absorbing the centre's 0.2% allowance.
        assert (
            abs((reading["centre"] + sigma) - reading["upper_1"]) <= sigma_tolerance
        ), reading["date"]
        assert (
            abs((reading["centre"] - sigma) - reading["lower_1"]) <= sigma_tolerance
        ), reading["date"]


def test_the_anchor_bar_centre_is_that_bars_hlc3_to_the_cent(bars, rules):
    """The single fact that killed OHLC/4 and every anchor-offset candidate."""
    anchor = _anchor(bars, rules)
    series = oneoption_avwap_band_series(bars, anchor, lookback=rules["lookback"])
    bar = bars[anchor]
    hlc3 = (bar["high"] + bar["low"] + bar["close"]) / 3.0
    assert series["centre"][anchor] == pytest.approx(hlc3, abs=1e-9)
    ohlc4 = (bar["open"] + bar["high"] + bar["low"] + bar["close"]) / 4.0
    assert abs(hlc3 - ohlc4) > 2.0, "the fixture bar no longer discriminates hlc3 from ohlc4"


def test_the_final_bar_helper_returns_the_champions_shape(bars, rules):
    anchor = _anchor(bars, rules)
    upto = bars[: _index_of(bars, "2026-06-02") + 1]
    centre, sigma, band_map = oneoption_avwap_bands(upto, anchor, lookback=rules["lookback"])
    series = oneoption_avwap_band_series(upto, anchor, lookback=rules["lookback"])
    assert centre == series["centre"][-1]
    assert sigma == series["sigma"][-1]
    assert set(band_map) == {"UPPER_1", "LOWER_1", "UPPER_2", "LOWER_2", "UPPER_3", "LOWER_3"}
    assert band_map["UPPER_2"] == pytest.approx(centre + 2 * sigma)
    assert band_map["LOWER_3"] == pytest.approx(centre - 3 * sigma)


# ---------------------------------------------------------------------------
# 2. The discriminators. Both of these formulas are WRONG and must stay wrong.
# ---------------------------------------------------------------------------


def _champion_upper_1(bars, anchor: int) -> float:
    frame = pd.DataFrame(bars)
    vwap, _stdev, band_map = calc_anchored_vwap_bands(frame, anchor)
    return band_map["UPPER_1"]


def _sample_ohlc_upper_1(bars, anchor: int) -> float:
    """The killed S1 form: sample stdev of every OHLC print around the AVWAP.

    Reproduced here rather than in the module so no live code carries a
    formula the study already killed. It survived the anchor bar (128.51
    against a reading of 128.47) and died on the second hover.
    """
    cumulative_volume = 0.0
    cumulative_price_volume = 0.0
    prints: list[float] = []
    for bar in bars[anchor:]:
        volume = float(bar["volume"] or 0.0)
        if volume > 0:
            typical = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            cumulative_volume += volume
            cumulative_price_volume += typical * volume
        prints.extend([bar["open"], bar["high"], bar["low"], bar["close"]])
    vwap = cumulative_price_volume / cumulative_volume
    total = sum((value - vwap) ** 2 for value in prints)
    return vwap + math.sqrt(total / (len(prints) - 1))


def test_the_champion_gives_a_different_answer_on_the_same_bars(bars, rules):
    """The champion's sigma is ZERO on a one-bar anchor. OneOption read 10.28."""
    anchor = _anchor(bars, rules)
    anchor_only = bars[: anchor + 1]
    champion_upper = _champion_upper_1(anchor_only, anchor)
    variant = oneoption_avwap_band_series(anchor_only, anchor, lookback=rules["lookback"])
    variant_upper = variant["upper_1"][anchor]
    # The champion's band collapses onto its own centre; the variant's does not.
    _vwap, champion_sigma, _bands = calc_anchored_vwap_bands(pd.DataFrame(anchor_only), anchor)
    assert champion_sigma == pytest.approx(0.0, abs=1e-9)
    assert variant["sigma"][anchor] > 10.0
    assert abs(variant_upper - champion_upper) > 5.0


def test_the_killed_sample_ohlc_form_predicts_138_where_the_trader_read_144(bars, rules):
    """The reading that decided it. Keep both numbers visible in the test."""
    anchor = _anchor(bars, rules)
    upto = bars[: _index_of(bars, "2026-06-02") + 1]
    killed = _sample_ohlc_upper_1(upto, anchor)
    assert killed == pytest.approx(138.09, abs=0.05)

    series = oneoption_avwap_band_series(upto, anchor, lookback=rules["lookback"])
    replicated = series["upper_1"][-1]
    assert replicated == pytest.approx(144.82, abs=0.05)
    assert abs(replicated - killed) > 6.0


# ---------------------------------------------------------------------------
# 3. Unmeasurable is None, never padded and never zero.
# ---------------------------------------------------------------------------


def test_sigma_is_none_until_the_lookback_is_full(bars, rules):
    """Nineteen closes is not a twenty-close standard deviation."""
    lookback = rules["lookback"]
    series = oneoption_avwap_band_series(bars, 0, lookback=lookback)
    assert all(value is None for value in series["sigma"][: lookback - 1])
    assert series["sigma"][lookback - 1] is not None
    assert all(series[key][lookback - 2] is None for key in ("upper_1", "lower_1", "upper_3"))
    # The centre exists from the anchor regardless - only the width is missing.
    assert series["centre"][0] is not None


def test_a_short_series_yields_a_centre_but_no_band(bars, rules):
    short = bars[:5]
    centre, sigma, band_map = oneoption_avwap_bands(short, 0, lookback=rules["lookback"])
    assert centre is not None
    assert sigma is None
    assert band_map == {}


def test_the_sigma_window_reaches_back_before_the_anchor(bars, rules):
    """It is a Bollinger width, not an anchored deviation. It has no anchor memory."""
    anchor = _anchor(bars, rules)
    lookback = rules["lookback"]
    series = oneoption_avwap_band_series(bars, anchor, lookback=lookback)
    window = [bar["close"] for bar in bars[anchor + 1 - lookback : anchor + 1]]
    mean = sum(window) / len(window)
    expected = math.sqrt(sum((value - mean) ** 2 for value in window) / len(window))
    assert series["sigma"][anchor] == pytest.approx(expected, abs=1e-9)
    # Two different anchors on the same tape carry the SAME width. That is a
    # real weakness of the formula and it is pinned so nobody "fixes" it here.
    other = oneoption_avwap_band_series(bars, anchor - 3, lookback=lookback)
    assert other["sigma"][anchor] == pytest.approx(series["sigma"][anchor], abs=1e-12)


def test_an_empty_series_is_empty_not_zero():
    series = oneoption_avwap_band_series([], 0)
    assert all(series[key] == () for key in series)
    assert oneoption_avwap_bands([], 0) == (None, None, {})


def test_bars_before_the_anchor_are_none_in_every_series(bars, rules):
    anchor = _anchor(bars, rules)
    series = oneoption_avwap_band_series(bars, anchor, lookback=rules["lookback"])
    for key, values in series.items():
        assert len(values) == len(bars), key
        assert all(value is None for value in values[:anchor]), key


# ---------------------------------------------------------------------------
# Zero-volume handling: skipped in the centre, still counted in the sigma.
# ---------------------------------------------------------------------------


def _synthetic(volumes) -> list[dict]:
    return [
        {
            "date": f"2026-01-{index + 1:02d}",
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.0 + index,
            "volume": volume,
        }
        for index, volume in enumerate(volumes)
    ]


@pytest.mark.parametrize("blank", [0.0, -5.0, None, float("nan")])
def test_a_blank_volume_bar_is_skipped_in_the_centre_exactly_as_the_champion_skips_it(blank):
    volumes = [1000.0, blank, 1000.0]
    bars = _synthetic(volumes)
    series = oneoption_avwap_band_series(bars, 0, lookback=3)
    weighted = [(bar["high"] + bar["low"] + bar["close"]) / 3.0 for bar in bars]
    expected = (weighted[0] + weighted[2]) / 2.0
    assert series["centre"][2] == pytest.approx(expected, abs=1e-9)

    # ...and the champion agrees about WHICH bars carry weight, which is the
    # thing that has to stay identical between the two formulas.
    frame = pd.DataFrame(bars)
    champion_vwap, _sigma, _bands = calc_anchored_vwap_bands(frame, 0)
    ohlc4 = [(b["open"] + b["high"] + b["low"] + b["close"]) / 4.0 for b in bars]
    assert champion_vwap == pytest.approx((ohlc4[0] + ohlc4[2]) / 2.0, abs=1e-9)


def test_a_blank_volume_bars_close_still_counts_in_the_sigma():
    """The sigma is not volume-weighted, so a zero-volume session is still a close."""
    bars = _synthetic([1000.0, 0.0, 1000.0])
    series = oneoption_avwap_band_series(bars, 0, lookback=3)
    closes = [bar["close"] for bar in bars]
    mean = sum(closes) / 3.0
    expected = math.sqrt(sum((value - mean) ** 2 for value in closes) / 3.0)
    assert series["sigma"][2] == pytest.approx(expected, abs=1e-12)


def test_a_missing_close_makes_the_whole_window_unmeasurable():
    bars = _synthetic([1000.0] * 5)
    bars[1]["close"] = None
    series = oneoption_avwap_band_series(bars, 0, lookback=3)
    # Every window still touching the hole is unmeasurable...
    assert series["sigma"][2] is None
    assert series["upper_1"][2] is None
    assert series["sigma"][3] is None
    # ...and the first window clear of it measures normally again.
    assert series["sigma"][4] is not None


def test_a_dataframe_and_a_list_of_dicts_agree(bars, rules):
    anchor = _anchor(bars, rules)
    from_list = oneoption_avwap_band_series(bars, anchor, lookback=rules["lookback"])
    from_frame = oneoption_avwap_band_series(
        pd.DataFrame(bars), anchor, lookback=rules["lookback"]
    )
    assert from_frame == from_list


def test_an_out_of_range_anchor_raises(bars):
    with pytest.raises(IndexError):
        oneoption_avwap_band_series(bars, len(bars))
    with pytest.raises(IndexError):
        oneoption_avwap_band_series(bars, -1)


def test_a_nonsense_lookback_raises(bars):
    with pytest.raises(ValueError):
        oneoption_avwap_band_series(bars, 0, lookback=0)


# ---------------------------------------------------------------------------
# 3. The module must not be able to reach the champion.
# ---------------------------------------------------------------------------


def test_the_module_never_imports_master_avwap_lib():
    """Structural, not conventional: the fence is checked in the AST.

    The champion is frozen (decision 0008). A challenger that imported it could
    start sharing its state or, worse, be edited into calling it - and the
    review that would catch that is exactly the review this test replaces.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [
        name
        for name in imported
        if name.split(".")[0] in {"master_avwap_lib", "master_avwap", "pandas", "numpy"}
    ]
    assert offenders == [], f"pure indicator imported {offenders}"
