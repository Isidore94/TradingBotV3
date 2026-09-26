"""S15 items 5 and 6 - sector strength, liquidity and size facets, and the earnings cycle split.

Shadow only. The scan appends six `perm_` columns (dollar volume, market cap, the sector's rank
over 5 and 20 sessions and how many sectors were ranked, and the latest earnings gap signed in
ATR); five facets read them. Missing input is unknown. The scan golden proves every other column
and the detector/scoring output are unchanged with the hooks on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_backfill as bf  # noqa: E402
import setup_permutation_context as spc  # noqa: E402
import setup_permutations as sp  # noqa: E402

DV, CAP, RANK5, RANK20, COUNT, GAP = sp.S15_COLUMNS


def _facet(name, **row):
    return sp.facets_for_row({"side": "LONG", "setup_family": "f", **row}).get(name)


# ---------------------------------------------------------------------------
# dollar volume and market cap
# ---------------------------------------------------------------------------
def _bars(count, close=50.0, volume=2e6):
    return [{"close": close, "volume": volume} for _ in range(count)]


def test_dollar_volume_is_the_20_session_mean_in_millions():
    bars = _bars(5, close=10.0, volume=1e6) + _bars(20, close=50.0, volume=2e6)
    assert sp.liquidity_columns(bars)[DV] == 100.0


def test_dollar_volume_needs_20_whole_bars():
    assert sp.liquidity_columns(_bars(19))[DV] is None
    holed = _bars(25)
    holed[-3] = {"close": 50.0, "volume": None}
    assert sp.liquidity_columns(holed)[DV] is None
    assert sp.liquidity_columns(None) == {DV: None, CAP: None}


def test_market_cap_is_carried_only_when_positive():
    assert sp.liquidity_columns([], market_cap_m=12_345.67)[CAP] == 12_345.7
    assert sp.liquidity_columns([], market_cap_m=0.0)[CAP] is None
    assert sp.liquidity_columns([], market_cap_m="nan")[CAP] is None


@pytest.mark.parametrize("value, expected", [
    (10.0, "dollar_vol_below_50m"), (50.0, "dollar_vol_50_125m"), (124.9, "dollar_vol_50_125m"),
    (125.0, "dollar_vol_125_300m"), (300.0, "dollar_vol_300m_plus"), (None, sp.UNKNOWN), (0.0, sp.UNKNOWN),
])
def test_the_dollar_volume_facet(value, expected):
    assert _facet("dollar_volume", **{DV: value}) == expected


@pytest.mark.parametrize("value, expected", [
    (1_500.0, "cap_below_2b"), (2_000.0, "cap_2_10b"), (10_000.0, "cap_10_50b"), (75_000.0, "cap_50b_plus"),
    (None, sp.UNKNOWN), (-1.0, sp.UNKNOWN),
])
def test_the_market_cap_facet(value, expected):
    assert _facet("market_cap", **{CAP: value}) == expected


def _cache(tmp_path, fetched_at, caps):
    path = tmp_path / "market_caps.json"
    path.write_text(json.dumps({"caps": caps, "fetched_at": fetched_at}), encoding="utf-8")
    return path


def test_the_cap_cache_is_read_when_fresh(tmp_path):
    path = _cache(tmp_path, "2026-09-24T13:02:08", {"aapl": 3_000_000.0, "ZERO": 0.0, "BAD": "x"})
    assert spc.load_market_caps(path=path, now=datetime(2026, 9, 26, 9, 0)) == {"AAPL": 3_000_000.0}


def test_a_stale_missing_or_future_cap_cache_is_unknown(tmp_path):
    path = _cache(tmp_path, "2026-08-01T13:00:00", {"AAPL": 3_000_000.0})
    assert spc.load_market_caps(path=path, now=datetime(2026, 9, 26)) == {}
    assert spc.load_market_caps(path=path, now=datetime(2026, 7, 31)) == {}
    assert spc.load_market_caps(path=tmp_path / "missing.json") == {}


def test_the_cap_cache_path_is_the_universe_builders():
    import universe_builder

    assert spc.market_cap_cache_path() == Path(universe_builder.MARKET_CAP_CACHE)


# ---------------------------------------------------------------------------
# sector rank
# ---------------------------------------------------------------------------
AS_OF = "2026-09-25"


def _closes(r5, r20):
    """21 closes ending AS_OF with the given 5- and 20-session returns."""
    last = 100.0
    closes = [last / (1 + r20)] + [100.0] * 14 + [last / (1 + r5)] + [100.0] * 4 + [last]
    return [(f"2026-08-{index + 1:02d}", close) for index, close in enumerate(closes[:-1])] + [(AS_OF, last)]


def _universe(sectors=6, members=5):
    closes, sector_of = {}, {}
    for k in range(sectors):
        for m in range(members):
            symbol = f"S{k}M{m}"
            # sector0 is strongest over 20 sessions and weakest over 5.
            closes[symbol] = _closes(r5=0.01 * k, r20=0.05 * (sectors - k))
            sector_of[symbol] = f"sector{k}"
    return closes, sector_of


def test_sectors_rank_by_median_member_return():
    closes, sector_of = _universe()
    rows = [{"symbol": "s0m1"}, {"symbol": "S5M0"}, {"symbol": "S2M2"}]
    assert sp.sector_rank_columns(rows, closes, sector_of, as_of=AS_OF) == 3
    assert (rows[0][RANK5], rows[0][RANK20], rows[0][COUNT]) == (6, 1, 6)
    assert (rows[1][RANK5], rows[1][RANK20], rows[1][COUNT]) == (1, 6, 6)
    assert (rows[2][RANK5], rows[2][RANK20]) == (4, 3)


def test_a_row_without_a_ranked_sector_is_blank():
    closes, sector_of = _universe()
    sector_of["LONE"] = "tiny"
    closes["LONE"] = _closes(0.5, 0.5)
    rows = [{"symbol": "LONE"}, {"symbol": "NOSECTOR"}]
    assert sp.sector_rank_columns(rows, closes, sector_of, as_of=AS_OF) == 0
    assert all(row[column] is None for row in rows for column in (RANK5, RANK20, COUNT))


def test_too_few_sectors_rank_nothing():
    closes, sector_of = _universe(sectors=5)
    rows = [{"symbol": "S0M0"}]
    assert sp.sector_rank_columns(rows, closes, sector_of, as_of=AS_OF) == 0
    assert rows[0][RANK20] is None


def test_stale_or_short_members_do_not_count():
    closes, sector_of = _universe(sectors=7)
    # Two of sector0's five members are stale (last bar not AS_OF) or too short: sector0 drops out.
    closes["S0M0"] = closes["S0M0"][:-1]
    closes["S0M1"] = closes["S0M1"][-10:]
    rows = [{"symbol": "S0M2"}, {"symbol": "S1M0"}]
    assert sp.sector_rank_columns(rows, closes, sector_of, as_of=AS_OF) == 1
    assert rows[0][RANK20] is None
    assert (rows[1][RANK20], rows[1][COUNT]) == (1, 6)


@pytest.mark.parametrize("rank, count, expected", [
    (1, 11, "top"), (3, 11, "top"), (4, 11, "mid"), (8, 11, "mid"), (9, 11, "bottom"), (11, 11, "bottom"),
    (2, 6, "top"), (3, 6, "mid"), (5, 6, "bottom"),
    (1, 5, None), (12, 11, None), (0, 11, None), (1.5, 11, None), (None, 11, None), (1, None, None),
])
def test_the_sector_facets_read_thirds(rank, count, expected):
    for name, column, prefix in (("sector_rs_20d", RANK20, "sector_rs20_"), ("sector_rs_5d", RANK5, "sector_rs5_")):
        value = _facet(name, **{column: rank, COUNT: count})
        assert value == (prefix + expected if expected else sp.UNKNOWN)


# ---------------------------------------------------------------------------
# the earnings cycle
# ---------------------------------------------------------------------------
def test_the_gap_column_is_signed_by_direction():
    base = {"gap_date": "2026-09-01", "gap_atr_multiple": 2.5, "pre_gap_close": 100.0}
    assert sp.earnings_gap_columns({**base, "gap_open": 104.0}) == {GAP: 2.5}
    assert sp.earnings_gap_columns({**base, "gap_open": 96.0}) == {GAP: -2.5}
    assert sp.earnings_gap_columns({**base, "gap_open": 100.0}) == {GAP: 0.0}


def test_the_gap_column_is_blank_without_a_measured_gap():
    assert sp.earnings_gap_columns({}) == {GAP: None}
    assert sp.earnings_gap_columns(None) == {GAP: None}
    assert sp.earnings_gap_columns({"gap_date": "", "gap_atr_multiple": 2.0, "gap_open": 1, "pre_gap_close": 1}) \
        == {GAP: None}
    assert sp.earnings_gap_columns({"gap_date": "2026-09-01", "gap_open": 104.0, "pre_gap_close": 100.0}) \
        == {GAP: None}


@pytest.mark.parametrize("sessions, gap, expected", [
    (0, 2.0, "drift_gap_up_0_13s"), (13, 1.0, "drift_gap_up_0_13s"),
    (5, -1.0, "drift_gap_down_0_13s"), (5, 0.4, "drift_small_gap_0_13s"), (5, -0.99, "drift_small_gap_0_13s"),
    (5, None, sp.UNKNOWN),
    (14, None, "mid_cycle_14_27s"), (27, 3.0, "mid_cycle_14_27s"), (28, None, "mid_cycle_28_60s"),
    (60, -2.0, "mid_cycle_28_60s"), (61, None, "late_cycle_61s_plus"),
    (None, 2.0, sp.UNKNOWN), (-1, 2.0, sp.UNKNOWN), (3.5, 2.0, sp.UNKNOWN),
])
def test_the_earnings_cycle_splits_drift_by_gap_direction(sessions, gap, expected):
    assert _facet("earnings_cycle", latest_release_sessions_since_gap=sessions, **{GAP: gap}) == expected


# ---------------------------------------------------------------------------
# registration, key and backfill
# ---------------------------------------------------------------------------
def test_the_facets_are_registered_off_the_label():
    for name, group in (("sector_rs_5d", "strength"), ("sector_rs_20d", "strength"), ("dollar_volume", "size"),
                        ("market_cap", "size"), ("earnings_cycle", "earnings")):
        assert sp.FACETS[name].group == group
        assert sp.FACETS[name].in_label is False
        assert f"f_{name}" in bf.output_columns()


def test_a_row_without_the_columns_keys_as_before():
    key = sp.facets_for_row({"side": "LONG", "setup_family": "f", "latest_release_sessions_since_gap": 5})
    for name in ("sector_rs_5d", "sector_rs_20d", "dollar_volume", "market_cap", "earnings_cycle"):
        assert key.get(name) == sp.UNKNOWN
        assert f"{name}=" not in key.compact_key
    assert key.permutation_rule_version == "setup_permutations.v1"


def test_the_columns_are_appended_last_and_perm_prefixed():
    assert sp.SCAN_ROW_COLUMNS[-len(sp.S15_COLUMNS):] == sp.S15_COLUMNS
    assert all(column.startswith("perm_") for column in sp.S15_COLUMNS)


# ---------------------------------------------------------------------------
# the runner's completed-bar helpers
# ---------------------------------------------------------------------------
def _runner():
    from master_avwap_lib import runner

    return runner


def _frame(days, volume=True):
    frame = pd.DataFrame({"datetime": pd.to_datetime(days), "close": [10.0 + i for i in range(len(days))]})
    if volume:
        frame["volume"] = 1e6
    return frame


def test_a_forming_last_bar_is_left_out():
    runner = _runner()
    frames = {"A": _frame(["2026-09-24", "2026-09-25"]), "B": _frame(["2026-09-23", "2026-09-24"])}
    # 2026-09-25 10:00 ET: the 09-25 bar is forming.
    assert runner._permutation_completed_through(frames, datetime(2026, 9, 25, 10, 0)) == "2026-09-24"
    # After the close it is completed.
    assert runner._permutation_completed_through(frames, datetime(2026, 9, 25, 16, 30)) == "2026-09-25"
    assert runner._permutation_completed_through({}, datetime(2026, 9, 25)) is None


def test_completed_bars_stop_at_the_cutoff_and_keep_holes():
    runner = _runner()
    frame = _frame(["2026-09-23", "2026-09-24", "2026-09-25"])
    frame.loc[0, "close"] = float("nan")
    bars = runner._permutation_completed_bars(frame, "2026-09-24")
    assert [bar["date"] for bar in bars] == ["2026-09-23", "2026-09-24"]
    assert bars[0]["close"] is None and bars[1] == {"date": "2026-09-24", "close": 11.0, "volume": 1e6}
    assert runner._permutation_completed_bars(_frame(["2026-09-24"], volume=False), "2026-09-24")[0]["volume"] is None
    assert runner._permutation_completed_bars(frame, None) == []


# ---------------------------------------------------------------------------
# the scan golden: output unchanged, the S15 columns on the row
# ---------------------------------------------------------------------------
def _parity_module():
    path = Path(__file__).with_name("test_setup_permutation_scan_parity.py")
    spec = importlib.util.spec_from_file_location("_s15_scan_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan_runs(tmp_path_factory):
    parity = _parity_module()
    base = tmp_path_factory.mktemp("s15-scan")
    for mode in ("on", "off"):
        # A fresh cap cache in each child's scratch LOCALAPPDATA (the universe builder's file).
        cache = base / f"{mode}-260" / "localappdata" / "TradingBotV3" / "machine_cache" / "universe"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "market_caps.json").write_text(json.dumps({
            "caps": {"PKEY": 7_500.0}, "fetched_at": (datetime.now() - timedelta(days=1)).isoformat()}),
            encoding="utf-8")
    return parity, parity._run(base, "on"), parity._run(base, "off")


def test_the_scan_writes_the_s15_columns(scan_runs):
    _parity, stamped, _plain = scan_runs
    row = stamped["history"][-1]
    assert list(row)[-len(sp.S15_COLUMNS):] == list(sp.S15_COLUMNS)
    # The child's bars: close ~60-112, volume 1M, last bar yesterday (completed).
    assert 50.0 < float(row[DV]) < 125.0
    assert float(row[CAP]) == 7_500.0
    assert row[RANK20] == "" and row[COUNT] == ""  # one scanned name: no sector ranks
    key = dict(part.split("=", 1) for part in row["permutation_key"].split("|")[3].split(";"))
    assert key["dollar_volume"] == "dollar_vol_50_125m"
    assert key["market_cap"] == "cap_2_10b"
    assert "sector_rs_20d" not in key


def test_the_scan_output_is_identical_with_and_without_the_s15_columns(scan_runs):
    parity, stamped, plain = scan_runs
    assert stamped["priority_row"] and plain["priority_row"]
    assert parity._strip(stamped["priority_row"]) == parity._strip(plain["priority_row"])
    assert parity._strip(stamped["ai_state_entry"]) == parity._strip(plain["ai_state_entry"])
    assert len(stamped["history"]) == len(plain["history"])
    for on_row, off_row in zip(stamped["history"], plain["history"], strict=True):
        assert parity._strip(on_row) == parity._strip(off_row)
        assert list(on_row) == list(off_row)  # same header, same order
    assert all(plain["history"][-1][column] == "" for column in sp.S15_COLUMNS)


def test_the_scan_has_a_completed_last_bar_to_measure():
    # Guards the golden above: the child's last bar is a past weekday, so it is completed.
    end = date.today() - timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    assert _runner().daily_bar_status(end, reference=datetime.now()) == "completed"
