"""What the charts looked like when the note was written (2026-08-27).

The trader wrote five entries through the Desk tab on 2026-08-27, opened the
Market Journal page, and found it empty and - in their words - "very useless":
words with no tape beside them. Two separate defects and one missing feature
sat behind that, and these tests pin the third: every entry now stores the M5
and D1 of its symbol and of SPY exactly as they stood when it was written.

The rules that matter here are the same ones every evidence store in this repo
keeps. A capture never costs the note it belongs to. A series that was not
cached is SAID to be absent rather than drawn as a flat chart. A digest states
only what it measured. And the bars live in a sidecar while a short text digest
goes to the ledger, because the AI grant that reads this must not be starved by
a bar window that only the page can use.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_journal as mj  # noqa: E402
import market_journal_capture as mjc  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point both halves of the store - sidecars and ledger - at tmp_path."""
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)
    return tmp_path


def _m5(count: int = 20, *, start: datetime | None = None, base: float = 100.0):
    start = start or datetime(2026, 8, 27, 9, 30)
    return [
        {
            "dt": start + timedelta(minutes=5 * index),
            "open": base + index * 0.10,
            "high": base + index * 0.10 + 0.30,
            "low": base + index * 0.10 - 0.30,
            "close": base + index * 0.10 + 0.20,
            "volume": 1_000 + index,
        }
        for index in range(count)
    ]


def _d1(count: int = 60):
    start = datetime(2026, 5, 1)
    return [
        {
            "dt": start + timedelta(days=index),
            "open": 50.0 + index * 0.05,
            "high": 50.6 + index * 0.05,
            "low": 49.4 + index * 0.05,
            "close": 50.2 + index * 0.05,
            "volume": 2_000_000 + index * 1_000,
        }
        for index in range(count)
    ]


# ==========================================================================
# bars in, bars out
# ==========================================================================
def test_only_the_tail_is_kept_and_the_stamps_survive_a_round_trip():
    rows = mjc.trim_bars(_m5(300), 160)

    assert len(rows) == 160
    assert isinstance(rows[0]["dt"], str)
    revived = mjc.revive_bars(rows)
    assert len(revived) == 160
    assert isinstance(revived[0]["dt"], datetime)


def test_a_bar_missing_a_price_is_not_in_the_picture():
    """A capture is a picture of a chart, and a row the chart could not draw
    was never in it."""
    bars = _m5(3)
    bars[1] = {"dt": bars[1]["dt"], "open": 1.0, "high": 2.0, "low": None, "close": 1.5}

    assert len(mjc.trim_bars(bars, 10)) == 2


def test_an_unreadable_stamp_is_dropped_and_counted_never_drawn():
    """The axis formats every stamp with strftime, so one string would take the
    chart down rather than degrade it. It is a gap, and the page says so."""
    stored = mjc.trim_bars(_m5(4), 10)
    stored[2]["dt"] = "not a timestamp"

    assert len(mjc.revive_bars(stored)) == 3
    assert mjc.unreadable_bar_count(stored) == 1


# ==========================================================================
# the digest - what the AI reads
# ==========================================================================
def test_the_intraday_line_reads_the_way_a_trader_reads_a_chart():
    line = mjc.describe_m5("DT", mjc.trim_bars(_m5(30), 160))

    assert line.startswith("DT M5: 30 bars")
    assert "session open" in line
    assert "session H" in line and "L " in line
    assert "up the session range" in line
    assert "session VWAP" in line


def test_the_prior_session_extremes_are_named_when_there_are_two_sessions():
    yesterday = _m5(20, start=datetime(2026, 8, 26, 9, 30), base=90.0)
    today = _m5(20, start=datetime(2026, 8, 27, 9, 30), base=100.0)
    line = mjc.describe_m5("DT", mjc.trim_bars(yesterday + today, 160))

    assert "prior session H" in line
    assert "above" in line


def test_a_series_that_was_not_cached_says_so_rather_than_drawing_flat():
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
    assert "no bars were cached" in mjc.describe_m5("DT", ())
    assert "no daily bars were cached" in mjc.describe_d1("DT", ())


def test_an_average_over_too_few_bars_is_absent_rather_than_wrong():
    """A 200-day average over 60 bars is a different number wearing the same
    label."""
    line = mjc.describe_d1("DT", mjc.trim_bars(_d1(60), 120))

    assert "20d SMA" in line
    assert "50d SMA" in line
    assert "200d SMA" not in line


def test_relative_volume_is_measured_against_the_prior_twenty_days():
    line = mjc.describe_d1("DT", mjc.trim_bars(_d1(60), 120))
    assert "RVOL" in line


def test_the_benchmark_is_not_described_twice_when_it_is_the_symbol():
    """An auto-mode flip captures SPY alone. Printing "SPY: no bars were
    cached" under a captured SPY chart would read as a failure."""
    capture = mjc.build_capture(
        entry_id="mj-2026-08-27-aaa",
        symbol="SPY",
        m5_bars=_m5(20),
        d1_bars=_d1(40),
        reason=mjc.REASON_MODE_FLIP,
    )

    assert capture["digest"].count("SPY M5:") == 1
    assert "no bars were cached" not in capture["digest"]


def test_a_symbolless_capture_still_describes_the_market():
    capture = mjc.build_capture(entry_id="mj-1", benchmark_m5=_m5(20), benchmark_d1=_d1(40))

    assert "SPY M5:" in capture["digest"]
    assert capture["symbol"] == ""


# ==========================================================================
# storage
# ==========================================================================
def test_the_bars_go_to_a_sidecar_and_only_the_digest_goes_to_the_ledger(store):
    capture = mjc.build_capture(
        entry_id="mj-2026-08-27-bbb",
        symbol="DT",
        m5_bars=_m5(40),
        d1_bars=_d1(40),
        benchmark_m5=_m5(40),
        benchmark_d1=_d1(40),
        now=datetime(2026, 8, 27, 18, 0),
    )
    result = mjc.record_capture(capture)

    assert result["ok"] is True
    row = result["row"]
    assert row["schema"] == mjc.SCHEMA_MARKET_JOURNAL_CHART
    assert row["digest"]
    assert "series" not in row and "bars" not in row
    assert row["bar_counts"]["symbol_m5"] == 40
    # And the bars are on disk where the page can redraw them.
    assert mjc.load_capture("mj-2026-08-27-bbb")["series"]["symbol_d1"]


def test_a_capture_with_nothing_in_it_is_refused_rather_than_stored(store):
    """An empty capture row would promise a chart that never existed."""
    capture = mjc.build_capture(entry_id="mj-2026-08-27-ccc", symbol="DT")
    result = mjc.record_capture(capture)

    assert result["ok"] is False
    assert "no bars" in result["reason"]
    assert mjc.load_capture("mj-2026-08-27-ccc") is None
    assert not list(mjc.chart_ledger().read().rows)


def test_the_sidecar_exists_before_the_row_that_points_at_it(store):
    capture = mjc.build_capture(entry_id="mj-2026-08-27-ddd", symbol="DT", m5_bars=_m5(10))
    result = mjc.record_capture(capture)

    assert Path(result["row"]["bars_file"]).exists()


def test_a_missing_capture_reads_as_missing_never_as_an_error(store):
    assert mjc.load_capture("mj-nope") is None
    assert mjc.load_capture("") is None


def test_a_torn_sidecar_is_reported_as_absent_rather_than_raising(store):
    path = mjc.capture_path("mj-2026-08-27-eee", captured_at="2026-08-27T18:00:00+00:00")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert mjc.load_capture("mj-2026-08-27-eee") is None


def test_the_capture_and_its_entry_land_in_the_same_month_folder(store):
    """The month comes from the entry id when the stamp is unusable, so a
    capture never files itself away from the entry it belongs to."""
    path = mjc.capture_path("mj-2026-08-27-fff")
    assert path.parent.name == "202608"


def test_digests_are_keyed_by_entry_and_the_latest_one_wins(store):
    for index in range(2):
        mjc.record_capture(
            mjc.build_capture(
                entry_id="mj-2026-08-27-ggg",
                symbol="DT",
                m5_bars=_m5(10 + index),
                now=datetime(2026, 8, 27, 18 + index, 0),
            )
        )
    digests = mjc.digests_by_entry()

    assert set(digests) == {"mj-2026-08-27-ggg"}
    assert digests["mj-2026-08-27-ggg"]["bar_counts"]["symbol_m5"] == 11


# ==========================================================================
# the entry side
# ==========================================================================
def test_a_mode_flip_row_is_marked_as_written_by_the_desk():
    """The journal reads as one timeline - what the trader thought AND what the
    desk did - so a reader counting "what did you think?" must be able to tell
    them apart without reading the sentence."""
    flip = mj.build_entry(
        text="Auto mode DESK -> AWAY.",
        session_date="2026-08-27",
        origin=mj.ORIGIN_AUTO_MODE_FLIP,
    )
    typed = mj.build_entry(text="Chop.", session_date="2026-08-27")

    assert mj.is_machine_entry(flip) is True
    assert mj.is_machine_entry(typed) is False


def test_the_entry_schema_did_not_change_to_carry_a_capture():
    """A capture joins by entry_id from the outside. That is what lets it be
    written AFTER the entry, on a worker, without the note ever waiting."""
    entry = mj.build_entry(text="x", session_date="2026-08-27")

    assert mj.SCHEMA_MARKET_JOURNAL_ENTRY == "market_journal_entry_v1"
    assert "chart_capture" not in entry
    assert "digest" not in entry
