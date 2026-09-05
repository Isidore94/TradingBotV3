"""Packet Q1 - `held_run_score` says what it MEASURED.

Process review 2026-09-04, findings 1 and 2. Live count that day: 979 of 8,161
recent episodes read `held=True` because nothing had ever answered the
thirty-minute question for them; 8 of 2,646 "D1 present" episodes were the
OPPOSITE side of the swing setup. Every test here failed on `6b74165`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

AS_OF = "2026-09-01"


def _row(event_id, *, kind="update", date="2026-09-01", symbol="AAA", direction="long",
         stop_hit=False, minutes="60", mfe="2.0"):
    return {
        "event_id": event_id,
        "event_type": kind,
        "trade_date": date,
        "symbol": symbol,
        "direction": direction,
        "entry_time": f"{date}T10:00:00",
        "context_json": json.dumps({"market_environment": "trend_up"}),
        "stop_hit": "True" if stop_hit else "False",
        "mfe_r": mfe,
        "minutes_elapsed": minutes,
    }


def _event(symbol, *, date="20260901", direction="long"):
    return f"{symbol}_{direction}_{date}_10_00_00_ema_15"


# ---------------------------------------------------------------------------
# Q1.1 / Q1.2 - measurement state
# ---------------------------------------------------------------------------


def test_a_registered_only_event_is_pending_on_its_own_session_and_never_held():
    import held_run_score as hrs

    rows = [_row(_event("A"), kind="registered", minutes="", mfe="")]
    episode = hrs.build_episodes(rows, as_of=AS_OF)[0]

    assert episode.held is False, "nothing answered the question; it is not held"
    assert episode.measurement == hrs.PENDING
    assert episode.measurement_reason == hrs.REASON_NO_FOLLOW_UP


def test_a_registered_only_event_on_a_past_session_is_unmeasured():
    import held_run_score as hrs

    rows = [_row(_event("A"), kind="registered", minutes="", mfe="")]
    episode = hrs.build_episodes(rows, as_of="2026-09-03")[0]

    assert episode.held is False
    assert episode.measurement == hrs.UNMEASURED
    assert episode.measurement_reason == hrs.REASON_NO_FOLLOW_UP


def test_updates_that_never_reach_the_window_do_not_count_as_held():
    import held_run_score as hrs

    event = _event("A")
    rows = [
        _row(event, kind="registered", minutes="", mfe=""),
        _row(event, minutes="5"),
        _row(event, minutes="20"),
    ]
    episode = hrs.build_episodes(rows, as_of="2026-09-03")[0]

    assert episode.held is False
    assert episode.measurement == hrs.UNMEASURED
    assert episode.measurement_reason == hrs.REASON_WINDOW_NOT_REACHED


def test_a_late_stop_after_a_bracketing_row_is_a_measured_hold():
    import held_run_score as hrs

    event = _event("A")
    rows = [_row(event, minutes="40"), _row(event, minutes="95", stop_hit=True)]
    episode = hrs.build_episodes(rows, as_of=AS_OF)[0]

    assert episode.measurement == hrs.MEASURED_HELD
    assert episode.held is True


def test_a_late_stop_with_no_bracketing_row_is_break_time_unknown():
    """`stop_hit` is a boolean over ALL bars since entry and the log carries no
    first-break time, so a stop first reported at minute 95 may have gone at
    minute 13. The old code called it held."""
    import held_run_score as hrs

    event = _event("A")
    rows = [_row(event, minutes="12"), _row(event, minutes="95", stop_hit=True)]
    episode = hrs.build_episodes(rows, as_of=AS_OF)[0]

    assert episode.held is False
    assert episode.measurement == hrs.UNMEASURED
    assert episode.measurement_reason == hrs.REASON_BREAK_TIME_UNKNOWN
    assert episode.broke_early is False, "not broken either - unknown is unknown"


def test_the_cell_counts_every_state_and_rates_the_measured_ones_only():
    import held_run_score as hrs

    rows = [
        _row(_event("HELD"), symbol="HELD", minutes="60"),
        _row(_event("BROKE"), symbol="BROKE", minutes="10", stop_hit=True),
        _row(_event("SHORT"), symbol="SHORT", minutes="12"),
        _row(_event("SHORT"), symbol="SHORT", minutes="95", stop_hit=True),
        _row(_event("PEND"), symbol="PEND", kind="registered", minutes="", mfe=""),
    ]
    cells = hrs.build_segments(hrs.build_episodes(rows, as_of=AS_OF), as_of=AS_OF, min_n=1)

    assert len(cells) == 1
    cell = cells[0]
    assert cell["n"] == 4
    assert cell["n_held"] == 1 and cell["n_broken"] == 1
    assert cell["n_measured"] == 2
    assert cell["n_unmeasured"] == 1 and cell["n_pending"] == 1
    assert cell["hold_rate"] == 0.5, "held / MEASURED, never held / n"
    assert cell["coverage"] == 0.5


def test_dimension_summaries_carry_the_same_counts():
    import held_run_score as hrs

    rows = [
        _row(_event("HELD"), symbol="HELD", minutes="60"),
        _row(_event("PEND"), symbol="PEND", kind="registered", minutes="", mfe=""),
    ]
    summaries = hrs.dimension_summaries(hrs.build_episodes(rows, as_of=AS_OF), as_of=AS_OF, min_n=1)
    cell = summaries[("bounce_type", "long", "ema_15")]

    assert cell["n"] == 2 and cell["n_measured"] == 1 and cell["n_pending"] == 1
    assert cell["hold_rate"] == 1.0
    assert cell["coverage"] == 0.5


# ---------------------------------------------------------------------------
# Q1.3 - the window is the exchange-session window
# ---------------------------------------------------------------------------


def test_the_window_is_the_shared_lately_window_and_not_the_last_n_dates():
    import evidence_stats
    import held_run_score as hrs

    # Rows on every 7th calendar day over six months: the old code kept the
    # last twenty DATES present, ~140 calendar days of them.
    from datetime import date, timedelta

    rows = []
    cursor = date(2026, 3, 2)  # a Monday; every 7th day stays a weekday
    while cursor <= date(2026, 9, 1):
        stamp = cursor.isoformat()
        rows.append(_row(_event("A", date=stamp.replace("-", "")), date=stamp))
        cursor += timedelta(days=7)
    episodes = hrs.build_episodes(rows, as_of=AS_OF)

    kept = hrs.recent_sessions(episodes, as_of=AS_OF)
    start, end = evidence_stats.lately_window(end=AS_OF)
    assert kept, "the window is not empty"
    assert all(start <= day <= end for day in kept), (start, end, sorted(kept))
    assert min(kept) >= start, "the old code reached back to March"

    report = hrs.window_report(episodes, as_of=AS_OF)
    assert report["start"] == start and report["end"] == end
    assert report["sessions"] == evidence_stats.LATELY_SESSIONS
    from market_calendar import is_session

    assert report["sessions_with_data"] == len(
        [day for day in kept if is_session(date.fromisoformat(day))]
    )
    assert len(report["missing_sessions"]) == report["sessions"] - report["sessions_with_data"]
    assert report["missing_sessions"] == sorted(report["missing_sessions"])


def test_read_outcome_rows_keeps_only_the_window(tmp_path):
    import csv

    import evidence_stats
    import held_run_score as hrs

    start, end = evidence_stats.lately_window(end=AS_OF)
    path = tmp_path / "outcomes.csv"
    rows = [
        _row(_event("OLD", date="20260301"), date="2026-03-01"),
        _row(_event("IN", date="20260901"), date="2026-09-01"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    kept = hrs.read_outcome_rows(path, as_of=AS_OF)
    assert [row["trade_date"] for row in kept] == ["2026-09-01"]
    assert start <= "2026-09-01" <= end


# ---------------------------------------------------------------------------
# Q1.4 - the D1 overlap keeps the side
# ---------------------------------------------------------------------------


def _snapshot(tmp_path, entries):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"setups": entries}), encoding="utf-8")
    return path


def test_the_snapshot_reader_keeps_the_side(tmp_path):
    import held_run_score as hrs

    path = _snapshot(
        tmp_path,
        {
            "k1": {"scan_date": "2026-09-01", "symbol": "NVDA", "side": "SHORT",
                   "priority_bucket": "favorite_setup"},
            "k2": {"scan_date": "2026-09-01", "symbol": "AMD", "side": "LONG",
                   "priority_bucket": "near_favorite_zone"},
            "k3": {"scan_date": "2026-09-01", "symbol": "TSLA", "side": "LONG",
                   "priority_bucket": "watch"},
        },
    )
    setups = hrs.d1_setups_by_session(hrs.d1_setup_rows(path))

    assert setups == {"2026-09-01": {"NVDA": {"SHORT"}, "AMD": {"LONG"}}}


def test_an_opposite_side_setup_is_opposed_and_carries_no_d1_privilege():
    import held_run_score as hrs

    setups = {"2026-09-01": {"NVDA": {"SHORT"}, "AMD": {"LONG"}}}
    rows = [
        _row(_event("NVDA"), symbol="NVDA"),
        _row(_event("AMD"), symbol="AMD"),
        _row(_event("XYZ"), symbol="XYZ"),
    ]
    by_symbol = {
        episode.symbol: episode
        for episode in hrs.build_episodes(rows, d1_setups_by_session=setups, as_of=AS_OF)
    }

    assert by_symbol["NVDA"].d1_alignment == hrs.D1_OPPOSED
    assert by_symbol["NVDA"].d1_setup_present is False
    assert by_symbol["AMD"].d1_alignment == hrs.D1_ALIGNED
    assert by_symbol["AMD"].d1_setup_present is True
    assert by_symbol["XYZ"].d1_alignment == hrs.D1_NONE
    assert by_symbol["NVDA"].segment() != by_symbol["XYZ"].segment()


def test_a_missing_snapshot_reads_unknown_not_false(tmp_path):
    import held_run_score as hrs

    assert hrs.d1_setup_rows(tmp_path / "nope.json") is None
    setups = hrs.d1_setups_by_session(hrs.d1_setup_rows(tmp_path / "nope.json"))
    episode = hrs.build_episodes([_row(_event("A"))], d1_setups_by_session=setups, as_of=AS_OF)[0]

    assert episode.d1_alignment == hrs.D1_UNKNOWN
    assert episode.d1_setup_present is False


def test_unknown_and_none_are_different_cells():
    import held_run_score as hrs

    known = hrs.build_episodes(
        [_row(_event("A"))], d1_setups_by_session={"2026-09-01": {}}, as_of=AS_OF
    )[0]
    unknown = hrs.build_episodes([_row(_event("A"))], d1_setups_by_session={}, as_of=AS_OF)[0]

    assert known.d1_alignment == hrs.D1_NONE
    assert unknown.d1_alignment == hrs.D1_UNKNOWN
    assert known.segment() != unknown.segment()


def test_the_summary_names_its_basis_and_alert_cell_accepts_both_spellings():
    import held_run_score as hrs

    setups = {"2026-09-01": {"AAA": {"LONG"}}}
    cells = hrs.build_segments(
        hrs.build_episodes([_row(_event("AAA"))], d1_setups_by_session=setups, as_of=AS_OF),
        as_of=AS_OF,
        min_n=1,
    )
    cell = cells[0]
    assert cell["d1_alignment"] == hrs.D1_ALIGNED
    assert cell["d1_setup_present"] is True
    assert cell["d1_basis"] == hrs.D1_BASIS == "same_session_retrospective"

    index = hrs.segment_index(cells)
    by_string = hrs.alert_cell(
        index, bounce_type="ema_15", entry_time="2026-09-01T10:00:00",
        market_environment="trend_up", d1_alignment="aligned",
    )
    by_bool = hrs.alert_cell(
        index, bounce_type="ema_15", entry_time="2026-09-01T10:00:00",
        market_environment="trend_up", d1_setup_present=True,
    )
    assert by_string is cell and by_bool is cell


def test_d1_alignment_helper():
    import held_run_score as hrs

    setups = {"2026-09-01": {"AAA": {"LONG"}, "BBB": {"LONG", "SHORT"}}}
    assert hrs.d1_alignment(setups, "2026-09-01", "AAA", "long") == hrs.D1_ALIGNED
    assert hrs.d1_alignment(setups, "2026-09-01", "AAA", "SHORT") == hrs.D1_OPPOSED
    assert hrs.d1_alignment(setups, "2026-09-01", "BBB", "short") == hrs.D1_ALIGNED
    assert hrs.d1_alignment(setups, "2026-09-01", "CCC", "long") == hrs.D1_NONE
    assert hrs.d1_alignment(setups, "2026-09-02", "AAA", "long") == hrs.D1_UNKNOWN
    assert hrs.d1_alignment(None, "2026-09-01", "AAA", "long") == hrs.D1_UNKNOWN
    assert hrs.d1_alignment(setups, "2026-09-01", "AAA", "") == hrs.D1_UNKNOWN


# ---------------------------------------------------------------------------
# The Daytrade Tracker shows coverage beside the headline
# ---------------------------------------------------------------------------


def test_the_tracker_carries_a_measured_column_fed_from_the_cell():
    from ui.panels.daytrade_tracker_panel import PERFORMANCE_COLUMNS, apply_held_and_ran

    keys = [key for key, _label in PERFORMANCE_COLUMNS]
    assert "measured" in keys
    assert keys.index("measured") == keys.index("held_run_score") + 1

    rows = apply_held_and_ran(
        [{"dimension": "bounce_type", "direction": "long", "segment": "vwap"},
         {"dimension": "bounce_type", "direction": "long", "segment": "nope"}],
        {("bounce_type", "long", "vwap"): {"hold_rate": 0.9, "held_run_score": 2.0,
                                           "n": 41, "n_measured": 35}},
    )
    assert rows[0]["measured"] == "35 / 41"
    assert rows[1]["measured"] == ""


def test_the_tracker_status_line_names_the_window(monkeypatch):
    import held_run_score as hrs
    from ui.panels import daytrade_tracker_panel as panel_module

    report = {"start": "2026-08-04", "end": "2026-09-01", "sessions": 20,
              "sessions_with_data": 19, "missing_sessions": ["2026-08-14"]}
    text = panel_module.held_run_window_text(report)
    assert text == "Held/ran window: 20 sessions (2026-08-04 to 2026-09-01), 19 with data, 1 missing."
    assert hrs.window_report is not None
