"""R10.V step 1 - the mixed-unit AVWAP golden fixture.

The durable daily-bar store mixes two volume units: IB returns regular-session
volume in round lots (`whatToShow="TRADES"`, `useRTH=1`) and Yahoo returns the
consolidated session in shares. AVWAP is volume-weighted, so a splice between
them re-weights every level computed across it - measured on the live store as
30,003 of 60,519 mark-days carrying different levels after the 2026-08-21 scan.

This fixture **pins the wrong answer on purpose**. Its `mixed` expectations are
what the store produces today; its `shares` expectations are what a clean store
would produce. R10.V's backfill does not move either, because the fixture feeds
fixed frames - it changes which frame the store hands the detector. When an
AVWAP-derived fixture is re-frozen in step 5, this one is the control that says
why it moved.

The `lots` case is the load-bearing control: a **uniform** rescale cannot move
an AVWAP, because a volume-weighted ratio cancels a constant factor. Only a
splice moves it. That is the argument for R10.V refusing IB volume rather than
converting it, and it is asserted here as arithmetic rather than asserted in a
document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib.legacy import calc_anchored_vwap_bands  # noqa: E402

FIXTURE = "mixed_unit_avwap_v1"
BAND_KEYS = ("UPPER_1", "LOWER_1", "UPPER_2", "LOWER_2", "UPPER_3", "LOWER_3")


@pytest.fixture(scope="module")
def contract():
    return load_fixture_contract(FIXTURE)


def _measure(contract, name):
    frame = pd.DataFrame(contract["series"][name])
    anchor = contract["rules_under_test"]["anchor_index"]
    vwap, stdev, bands = calc_anchored_vwap_bands(frame, anchor)
    return vwap, stdev, bands


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


# ---------------------------------------------------------------------------
# the fixture reproduces
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["shares", "mixed", "lots"])
def test_each_series_reproduces_its_frozen_levels(contract, name):
    vwap, stdev, bands = _measure(contract, name)
    expected = contract["expected"][name]
    tolerance = contract.tolerance
    assert _close(vwap, expected["vwap"], tolerance), f"{name} vwap moved"
    assert _close(stdev, expected["stdev"], tolerance), f"{name} stdev moved"
    for key in BAND_KEYS:
        assert _close(bands[key], expected["bands"][key], tolerance), f"{name} {key} moved"


def test_the_three_series_differ_only_in_volume(contract):
    """If prices differed, the comparison below would prove nothing."""
    series = contract["series"]
    price_only = [
        [{k: v for k, v in row.items() if k != "volume"} for row in series[name]]
        for name in ("shares", "mixed", "lots")
    ]
    assert price_only[0] == price_only[1] == price_only[2]


# ---------------------------------------------------------------------------
# the control: a uniform rescale is not the mechanism
# ---------------------------------------------------------------------------
def test_a_uniform_rescale_does_not_move_the_levels(contract):
    """Every bar divided by 100 - AVWAP is a ratio, so the factor cancels.

    This is why R10.V refuses IB volume rather than converting it: if the store
    were uniformly mis-scaled, nothing would need repairing at all.
    """
    shares_vwap, shares_stdev, shares_bands = _measure(contract, "shares")
    lots_vwap, lots_stdev, lots_bands = _measure(contract, "lots")
    tolerance = contract.tolerance
    assert _close(lots_vwap, shares_vwap, tolerance)
    assert _close(lots_stdev, shares_stdev, tolerance)
    for key in BAND_KEYS:
        assert _close(lots_bands[key], shares_bands[key], tolerance)


def test_the_splice_does_move_them_and_by_how_much(contract):
    """The damage is recorded as a number, not as an adjective."""
    shares_vwap, shares_stdev, shares_bands = _measure(contract, "shares")
    mixed_vwap, mixed_stdev, mixed_bands = _measure(contract, "mixed")
    damage = contract["expected"]["damage"]
    tolerance = contract.tolerance

    assert _close(mixed_vwap - shares_vwap, damage["vwap_delta"], tolerance)
    assert _close(
        (mixed_vwap - shares_vwap) / shares_vwap * 100.0, damage["vwap_delta_pct"], tolerance
    )
    assert _close(mixed_stdev / shares_stdev, damage["stdev_ratio"], tolerance)
    assert _close(
        mixed_bands["UPPER_2"] - shares_bands["UPPER_2"], damage["upper_2_delta"], tolerance
    )

    # Direction and scale, stated so a reader does not have to open the JSON:
    # the spliced series is dragged toward its pre-splice half and its sigma
    # collapses, which is what freezes a pre-splice anchor near its last good
    # value on the live store.
    assert damage["vwap_delta"] < 0
    assert abs(damage["vwap_delta_pct"]) > 1.0
    assert damage["stdev_ratio"] < 0.75


# ---------------------------------------------------------------------------
# the sigma formula is not to be swapped (plan.md sec 5)
# ---------------------------------------------------------------------------
def test_sigma_is_the_running_deviation_variant(contract):
    """An independent reimplementation of the documented formula must agree.

    Every band consumer - events, zones, tracker families, scoring history - is
    calibrated to deviation from the RUNNING AVWAP at each bar, accumulated
    volume-weighted. This test exists so a swap to the distribution-stdev
    variant fails here rather than being discovered in a re-frozen fixture.
    """
    rows = contract["series"]["shares"]
    cum_vol = cum_vp = cum_sd = 0.0
    for row in rows:
        volume = float(row["volume"])
        if volume <= 0:
            continue
        typical = (row["open"] + row["high"] + row["low"] + row["close"]) / 4.0
        cum_vol += volume
        cum_vp += typical * volume
        running = cum_vp / cum_vol
        deviation = typical - running
        cum_sd += deviation * deviation * volume
    expected_vwap = cum_vp / cum_vol
    expected_stdev = (cum_sd / cum_vol) ** 0.5

    vwap, stdev, _ = _measure(contract, "shares")
    assert _close(vwap, expected_vwap, contract.tolerance)
    assert _close(stdev, expected_stdev, contract.tolerance)


def test_the_distribution_variant_would_give_a_different_answer(contract):
    """Proof the test above discriminates, rather than passing on any formula."""
    rows = contract["series"]["shares"]
    cum_vol = sum(float(row["volume"]) for row in rows)
    cum_vp = sum(
        (row["open"] + row["high"] + row["low"] + row["close"]) / 4.0 * float(row["volume"])
        for row in rows
    )
    final_vwap = cum_vp / cum_vol
    distribution = (
        sum(
            ((row["open"] + row["high"] + row["low"] + row["close"]) / 4.0 - final_vwap) ** 2
            * float(row["volume"])
            for row in rows
        )
        / cum_vol
    ) ** 0.5

    _, stdev, _ = _measure(contract, "shares")
    assert abs(distribution - stdev) > 1e-3, (
        "the two sigma variants happen to agree on this fixture, so it cannot "
        "detect a swap - rebuild it with more trend"
    )


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_the_fixture_reads_no_live_store(contract):
    """Hand-constructed bars for a synthetic symbol; no vendor can revise it."""
    assert contract["symbol"] == "MIXQ"
    assert "No provider call" in contract["provider_assumptions"]
    assert contract["intentional_difference"], "this fixture pins a known-wrong answer"


def test_zero_volume_bars_are_skipped_not_treated_as_price_information():
    """The band function's own rule, pinned: a zero-volume bar contributes nothing."""
    rows = [
        {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1_000.0},
        {"open": 11.0, "high": 11.5, "low": 10.5, "close": 11.0, "volume": 0.0},
        {"open": 12.0, "high": 12.5, "low": 11.5, "close": 12.0, "volume": 1_000.0},
    ]
    with_zero = calc_anchored_vwap_bands(pd.DataFrame(rows), 0)
    without = calc_anchored_vwap_bands(pd.DataFrame([rows[0], rows[2]]), 0)
    assert with_zero[0] == without[0]
    assert with_zero[1] == without[1]
