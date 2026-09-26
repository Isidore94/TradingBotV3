"""S15 item 2 - a point-in-time market regime label on every scan row.

Shadow only. The scan appends `perm_regime_*` columns: the trader's structural regime (S16) for the
scan date, then the machine's checks from completed daily bars (SPY vs its 20-day, the 20-day's
slope, breadth = share of the scanned universe above its own 20-day). The sector's 5/20-day RS rank
is already on the row (S15 item 5) and is not recomputed. `long_regime_working` is the one place
"the market is on a long's side" is defined. The scan golden proves every other column and the
detector/scoring output are unchanged with the hook on.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_context as spc  # noqa: E402
import setup_permutations as sp  # noqa: E402

(TRADER, TRADER_SESSIONS, AS_OF, SPY_VS, SPY_SLOPE, BREADTH, BREADTH_N, WORKING, RULE) = sp.REGIME_COLUMNS
DAY = "2026-09-25"


def _days(count, end=DAY):
    last = date.fromisoformat(end)
    return [(last - timedelta(days=count - 1 - index)).isoformat() for index in range(count)]


def _pairs(closes, end=DAY):
    return list(zip(_days(len(closes), end), closes, strict=True))


# ---------------------------------------------------------------------------
# SPY trend
# ---------------------------------------------------------------------------
def test_spy_trend_is_close_vs_sma20_and_the_sma20_change_over_5_sessions():
    closes = [100.0] * 5 + [110.0] * 20  # the last 20 closes are 110; the SMA20 five sessions back mixes in 100s
    vs, slope = sp.spy_trend(closes)
    assert vs == 0.0
    # SMA20 then = (5*100 + 15*110)/20 = 107.5; now = 110.
    assert slope == pytest.approx((110.0 - 107.5) / 107.5 * 100.0, abs=1e-4)


def test_spy_trend_needs_25_whole_closes():
    assert sp.spy_trend([100.0] * 24) == (None, None)
    holed = [100.0] * 30
    holed[-3] = None
    assert sp.spy_trend(holed) == (None, None)
    assert sp.spy_trend(None) == (None, None)


# ---------------------------------------------------------------------------
# "working": the one definition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trader, vs, slope, expected", [
    ("bull_run", -5.0, -1.0, ("yes", sp.WORKING_RULE_TRADER)),  # the trader's word beats SPY
    ("recovery", None, None, ("yes", sp.WORKING_RULE_TRADER)),
    ("bear_channel_lower_highs", 5.0, 1.0, ("no", sp.WORKING_RULE_TRADER)),
    ("range", 5.0, 1.0, ("no", sp.WORKING_RULE_TRADER)),
    (None, 1.0, 0.5, ("yes", sp.WORKING_RULE_SPY)),
    (None, 1.0, -0.5, ("no", sp.WORKING_RULE_SPY)),  # above a falling 20-day is not working
    (None, -1.0, 0.5, ("no", sp.WORKING_RULE_SPY)),
    ("not_a_regime", 1.0, 0.5, ("yes", sp.WORKING_RULE_SPY)),  # an unknown label is no label
    (None, None, 0.5, (sp.UNKNOWN, sp.UNKNOWN)),
    ("", 1.0, None, (sp.UNKNOWN, sp.UNKNOWN)),
])
def test_long_regime_working(trader, vs, slope, expected):
    assert sp.long_regime_working(trader, vs, slope) == expected


# ---------------------------------------------------------------------------
# breadth and the row columns
# ---------------------------------------------------------------------------
def _universe(above, below, end=DAY):
    closes = {}
    for index in range(above):
        closes[f"UP{index}"] = _pairs([100.0] * 20 + [110.0], end)
    for index in range(below):
        closes[f"DN{index}"] = _pairs([100.0] * 20 + [90.0], end)
    return closes


def test_breadth_is_the_share_above_its_own_sma20():
    assert sp.breadth_above_sma20(_universe(15, 5), as_of=DAY) == (75.0, 20)


def test_breadth_needs_20_current_names():
    closes = _universe(15, 4)
    closes["STALE"] = _pairs([100.0] * 21, end="2026-09-24")  # its last bar is not as_of
    closes["SHORT"] = _pairs([100.0] * 19)
    assert sp.breadth_above_sma20(closes, as_of=DAY) == (None, None)
    assert sp.breadth_above_sma20(_universe(15, 5), as_of=None) == (None, None)


def test_regime_columns_fill_every_row_the_same():
    rows = [{"symbol": "A"}, {"symbol": "B"}, "not a row"]
    spy = _pairs([100.0 + index for index in range(30)])
    segment = {"regime": "bear_channel_lower_highs", "session_count": 23}
    assert sp.regime_columns(rows, trader_segment=segment, spy_closes=spy,
                             closes_by_symbol=_universe(15, 5), as_of=DAY) == 2
    for row in rows[:2]:
        assert row[TRADER] == "bear_channel_lower_highs" and row[TRADER_SESSIONS] == 23
        assert row[AS_OF] == DAY
        assert row[SPY_VS] > 0 and row[SPY_SLOPE] > 0
        assert (row[BREADTH], row[BREADTH_N]) == (75.0, 20)
        assert (row[WORKING], row[RULE]) == ("no", sp.WORKING_RULE_TRADER)


def test_regime_columns_blank_what_is_unknown():
    rows = [{"symbol": "A"}]
    # SPY's last completed bar is not the scan's: stale SPY is unknown, never "confirmed".
    stale_spy = _pairs([100.0 + index for index in range(30)], end="2026-09-24")
    sp.regime_columns(rows, trader_segment=None, spy_closes=stale_spy, closes_by_symbol={}, as_of=DAY)
    row = rows[0]
    assert row[TRADER] is None and row[TRADER_SESSIONS] is None
    assert row[SPY_VS] is None and row[SPY_SLOPE] is None
    assert row[BREADTH] is None and row[BREADTH_N] is None
    assert row[WORKING] is None and row[RULE] is None


def test_the_regime_columns_are_appended_last_and_off_the_key():
    assert sp.SCAN_ROW_COLUMNS[-len(sp.REGIME_COLUMNS):] == sp.REGIME_COLUMNS
    assert all(column.startswith("perm_") for column in sp.REGIME_COLUMNS)
    key = sp.facets_for_row({"side": "LONG", "setup_family": "f", WORKING: "yes", TRADER: "bull_run"})
    assert "regime" not in key.compact_key.split("|", 3)[-1]
    assert key.permutation_rule_version == "setup_permutations.v1"


# ---------------------------------------------------------------------------
# the trader's regime, read-only from the journal
# ---------------------------------------------------------------------------
def test_the_trader_regime_is_read_without_touching_the_journal(tmp_path):
    from journal_store import JournalStore

    path = tmp_path / "trade_journal.sqlite3"
    store = JournalStore(path)
    stamp = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    store.append_structural_regime(start_date="2026-08-03", regime="bear_channel_lower_highs", entered_at=stamp)
    before = path.stat().st_mtime_ns
    segment = spc.load_trader_regime("2026-09-25", path=path)
    assert segment["regime"] == "bear_channel_lower_highs"
    assert segment["session_count"] and segment["session_count"] > 30
    assert spc.load_trader_regime("2026-07-01", path=path) is None  # before the first segment
    assert path.stat().st_mtime_ns == before


def test_a_missing_or_old_journal_is_unknown(tmp_path):
    import sqlite3

    assert spc.load_trader_regime(DAY, path=tmp_path / "missing.sqlite3") is None
    old = tmp_path / "old.sqlite3"
    sqlite3.connect(old).close()
    assert spc.load_trader_regime(DAY, path=old) is None


# ---------------------------------------------------------------------------
# the scan golden: output unchanged, the regime columns on the row
# ---------------------------------------------------------------------------
def _parity_module():
    path = Path(__file__).with_name("test_setup_permutation_scan_parity.py")
    spec = importlib.util.spec_from_file_location("_regime_scan_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan_runs(tmp_path_factory):
    parity = _parity_module()
    base = tmp_path_factory.mktemp("regime-scan")
    return parity, parity._run(base, "on"), parity._run(base, "off")


def test_the_scan_writes_the_regime_columns(scan_runs):
    _parity, stamped, _plain = scan_runs
    row = stamped["history"][-1]
    assert list(row)[-len(sp.REGIME_COLUMNS):] == list(sp.REGIME_COLUMNS)
    # The child's SPY is its one rising series (the fetch stub serves every symbol); no trader regime.
    assert row[TRADER] == ""
    assert float(row[SPY_VS]) > 0 and float(row[SPY_SLOPE]) > 0
    assert (row[WORKING], row[RULE]) == ("yes", sp.WORKING_RULE_SPY)
    assert row[BREADTH] == ""  # one scanned name: no breadth
    assert row[AS_OF] == row["last_trade_date"][:10]


def test_the_scan_output_is_identical_with_and_without_the_regime_columns(scan_runs):
    parity, stamped, plain = scan_runs
    assert stamped["priority_row"] and plain["priority_row"]
    assert parity._strip(stamped["priority_row"]) == parity._strip(plain["priority_row"])
    assert parity._strip(stamped["ai_state_entry"]) == parity._strip(plain["ai_state_entry"])
    assert len(stamped["history"]) == len(plain["history"])
    for on_row, off_row in zip(stamped["history"], plain["history"], strict=True):
        assert parity._strip(on_row) == parity._strip(off_row)
        assert list(on_row) == list(off_row)
    assert all(plain["history"][-1][column] == "" for column in sp.REGIME_COLUMNS)
