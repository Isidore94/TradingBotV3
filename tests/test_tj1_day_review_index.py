"""TJ-1 item 4 - the per-session index, so opening the page is not a 476 MB read.

Measured on the live desk 2026-09-17: `daily_recap_reader._read_intraday_outcomes`
streams the WHOLE `intraday_bounce_outcomes.csv` (476 MB) on every open, and
`master_avwap_session_horizon_outcomes.csv` (31 MB) behind it, and then every view
filters by session AFTERWARDS.

The index stores, per session, exactly the two `_Store`s `read_session` would have
built - `rows`, the FULL-FILE `coverage`, and `raw_rows_by_session` - so an
indexed read is the SAME ANSWER off a small file. That equality is the whole
contract, and it is asserted on the whole `RecapSession` dataclass, coverage
included: an index that dropped the coverage counts would still "work" and would
quietly relabel a 476 MB file as a 40-row one.

The contract these tests pin (`scripts/day_review_index.py`):

* ``build_index(session, *, lookback_sessions=3, sources=None, now=None) -> dict``
* ``write_index(index, *, root=None) -> Path | None``   (temp-and-rename)
* ``read_index(session, *, root=None) -> dict | None``  (corrupt -> None, logged)
* ``index_path(session, *, root=None) -> Path``
* ``is_stale(index, *, now=None) -> bool``              (the ONE staleness rule)
* ``daily_recap_reader.read_session(..., index=Mapping | None = None)``
* ``project_paths.DAY_REVIEW_DIR``
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# The fixture session, on the real exchange calendar
# ---------------------------------------------------------------------------
SESSION = "2026-09-10"
PRIOR_1 = "2026-09-09"
PRIOR_2 = "2026-09-08"
#: Friday morning, mid-session: 2026-09-10 is the last COMPLETED session.
NOW = datetime(2026, 9, 11, 7, 30)
#: The pending horizon row below targets 2026-09-11. This is after its close.
AFTER_THE_PENDING_TARGET_CLOSED = datetime(2026, 9, 12, 8, 0)
#: ...and this is while it was still forming.
BEFORE_THE_PENDING_TARGET_CLOSED = datetime(2026, 9, 11, 11, 0)

LOOKBACK = 3


def test_the_calendar_still_says_what_this_fixture_assumes():
    """Guard: the fixture's session arithmetic is real, not assumed."""
    import market_calendar

    day = date.fromisoformat(SESSION)
    assert market_calendar.is_session(day)
    assert market_calendar.previous_session(day).isoformat() == PRIOR_1
    assert market_calendar.previous_session(date.fromisoformat(PRIOR_1)).isoformat() == PRIOR_2
    assert market_calendar.last_completed_session(NOW).isoformat() == SESSION


# ---------------------------------------------------------------------------
# The two big stores, in their live column order
# ---------------------------------------------------------------------------
INTRADAY_HEADER = (
    "schema_version,event_id,event_type,logged_at,trade_date,symbol,direction,"
    "entry_time,entry_price,stop_price,risk_per_share,bars_elapsed,minutes_elapsed,"
    "close_r,mfe_r,mae_r,best_price,worst_price,target_1r_hit,target_2r_hit,stop_hit,"
    "status,milestone_bar,context_json,outcome_mode,eod_close,eod_move_pct,mfe_pct,mae_pct"
)

#: (event_id, trade_date, symbol, direction, entry_time, status, eod_close,
#:  eod_move_pct, mfe_pct, mae_pct)
#:
#: Modelled on the real file: the log is APPEND-ONLY, so `NVDA` appears twice -
#: once registered with empty measures and once finalized. `read_session` keeps
#: the LAST append per `event_id` while coverage counts every line on disk, and
#: an index that stores one number for both is wrong.
INTRADAY_ROWS = (
    ("NVDA_long_20260910_09_40_00_ema_15", SESSION, "NVDA", "long", "09:40:00",
     "open", "", "", "", ""),
    ("NVDA_long_20260910_09_40_00_ema_15", SESSION, "NVDA", "long", "09:40:00",
     "closed", "104.00", "4.00", "5.50", "-0.80"),
    ("TSLA_short_20260910_09_45_00_ema_15", SESSION, "TSLA", "short", "09:45:00",
     "closed", "95.00", "5.00", "6.00", "-0.80"),
    # Present and EMPTY, which is what an unfinalized row looks like on disk.
    ("AMD_long_20260910_10_05_00_ema_15", SESSION, "AMD", "long", "10:05:00",
     "open", "", "", "", ""),
    # A prior session's row, which must never leak into today's views.
    ("NFLX_long_20260909_09_40_00_ema_15", PRIOR_1, "NFLX", "long", "09:40:00",
     "closed", "707.00", "1.00", "2.00", "-1.00"),
    # No event_id at all: an incomplete identity, kept as its own row.
    ("", SESSION, "KBR", "short", "09:35:00", "closed", "77.60", "3.00", "4.00", "-1.50"),
)

SESSION_HORIZON_HEADER = (
    "observation_id,scan_row_id,symbol,side,scan_date,target_session,horizon_sessions,"
    "sessions_spanned,entry_close,entry_close_source,target_close,side_return_pct,"
    "favorable,measured,maturity,unmeasured_reason,outcome_kind,knowledge_basis,tier,"
    "tier_source,priority_bucket,setup_family,favorite_zone,collapsed_same_session"
)

#: (symbol, side, scan_date, target_session, horizon, side_return_pct, favorable,
#:  measured, maturity)
SESSION_HORIZON_ROWS = (
    ("AAPL", "LONG", PRIOR_1, SESSION, "1", "2.00", "True", "True", "mature"),
    ("ORCL", "SHORT", PRIOR_2, PRIOR_1, "1", "1.25", "True", "True", "mature"),
    # PENDING: the 3-session end has not arrived. Present and EMPTY, never zero.
    ("MSFT", "SHORT", PRIOR_2, "2026-09-11", "3", "", "", "", "immature"),
    # Outside the three-session lookback entirely.
    ("GOOG", "LONG", "2026-09-02", "2026-09-03", "1", "4.00", "True", "True", "mature"),
)


def _csv_text(header: str, rows) -> str:
    columns = header.split(",")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        assert len(row) == len(columns), (len(row), len(columns))
        writer.writerow(row)
    return buffer.getvalue()


def _intraday_csv() -> str:
    context = json.dumps({"market_environment": "bullish_strong"}, sort_keys=True)
    rows = []
    for (event_id, trade_date, symbol, direction, entry_time, status,
         eod_close, eod_move, mfe_pct, mae_pct) in INTRADAY_ROWS:
        rows.append([
            "1", event_id, "registered", f"{trade_date}T13:10:00-07:00", trade_date,
            symbol, direction, f"{trade_date}T{entry_time}", "100.00", "", "", "", "",
            "", "", "", "", "", "False", "False", "False", status, "", context, "",
            eod_close, eod_move, mfe_pct, mae_pct,
        ])
    return _csv_text(INTRADAY_HEADER, rows)


def _session_horizon_csv() -> str:
    rows = []
    for (symbol, side, scan_date, target, horizon, ret, favorable,
         measured, maturity) in SESSION_HORIZON_ROWS:
        scan_row_id = f"{symbol}:{scan_date}:{scan_date}-130129"
        rows.append([
            f"{scan_row_id}:{horizon}", scan_row_id, symbol, side, scan_date, target,
            horizon, horizon, "100.00", "session_bar", "", ret, favorable, measured,
            maturity, "" if measured else "horizon_not_reached",
            "favorable_direction_session_v2",
            "entry_session_close_to_target_session_close", "S", "derived_from_bucket",
            "favorite_setup", "avwap_band_bounce", "", "1",
        ])
    return _csv_text(SESSION_HORIZON_HEADER, rows)


@pytest.fixture()
def sources(tmp_path):
    """A `RecapSources` whose twelve paths all live under `tmp_path`.

    Every path is explicit, including the two with a `default_factory`: a read
    that silently reached a real store would not be a fixture.
    """
    import daily_recap_reader

    root = tmp_path / "home"
    root.mkdir()
    intraday = root / "intraday_bounce_outcomes.csv"
    horizon = root / "master_avwap_session_horizon_outcomes.csv"
    intraday.write_text(_intraday_csv(), encoding="utf-8")
    horizon.write_text(_session_horizon_csv(), encoding="utf-8")
    return daily_recap_reader.RecapSources(
        intraday_outcomes=intraday,
        tier_outcomes=root / "master_avwap_tier_outcomes.csv",
        session_horizon_outcomes=horizon,
        annotations=root / "trader_annotations.jsonl",
        pick_feedback=root / "pick_feedback.jsonl",
        swing_favorites=root / "swing_favorites.jsonl",
        human_focus_outcomes=root / "human_focus_outcomes.csv",
        review_events=root / "alert_review_events.jsonl",
        preference_report=root / "preference_trade_outcomes.csv",
        staged_picks=root / "auto_populate_pending.json",
        environment_labels=root / "d1_environment.jsonl",
        working_lately=root / "snapshot_latest.json",
    )


def _streamed(sources):
    import daily_recap_reader

    return daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources
    )


def _index(sources):
    import day_review_index

    return day_review_index.build_index(
        SESSION, lookback_sessions=LOOKBACK, sources=sources, now=NOW
    )


# ---------------------------------------------------------------------------
# 0. the fixture, measured against the CURRENT streaming reader
# ---------------------------------------------------------------------------
def test_the_fixture_stores_read_the_way_the_numbers_below_claim(sources):
    """Guard: the counts asserted further down are pinned from today's reader,
    so a wrong index cannot agree with a wrong expectation."""
    import daily_recap_reader

    store = daily_recap_reader._read_intraday_outcomes(Path(sources.intraday_outcomes))
    assert store.coverage.rows == 6, "six data lines on disk"
    assert len(store.rows) == 5, "five kept rows - NVDA's two appends are one event"
    assert store.raw_rows_by_session == {SESSION: 5, PRIOR_1: 1}
    nvda = [row for row in store.rows if row["symbol"] == "NVDA"]
    assert len(nvda) == 1 and nvda[0]["status"] == "closed"

    session = _streamed(sources)
    assert session.session_date == SESSION
    assert session.provisional is False
    assert set(session.coverage) == set(daily_recap_reader.SOURCE_NAMES)


# ---------------------------------------------------------------------------
# 1. the named path
# ---------------------------------------------------------------------------
def test_the_day_review_directory_is_a_named_project_paths_constant():
    """plan.md §12.3: every new path is a named `project_paths` constant."""
    import project_paths

    assert project_paths.DAY_REVIEW_DIR == project_paths.PERSISTENT_DATA_DIR / "day_review"


def test_the_index_lives_one_file_per_session_under_that_directory(tmp_path):
    import day_review_index

    path = day_review_index.index_path(SESSION, root=tmp_path)
    assert path == tmp_path / "sessions" / SESSION / "outcomes.json"


# ---------------------------------------------------------------------------
# 2. the equality that is the whole point
# ---------------------------------------------------------------------------
def test_an_indexed_read_is_the_same_session_as_a_streamed_one(sources):
    """Dataclass equality over the WHOLE `RecapSession`, coverage included."""
    import daily_recap_reader

    streamed = _streamed(sources)
    indexed = daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=_index(sources)
    )
    assert indexed == streamed


def test_the_index_carries_the_full_file_coverage_and_not_the_kept_row_count(sources):
    """Six data lines on disk, five kept rows (NVDA appears twice). Coverage
    answers "how big was the file", so it says 6 - carried forward unchanged."""
    index = _index(sources)
    intraday = index["intraday_outcomes"]
    assert intraday["coverage"]["rows"] == 6
    assert len(intraday["rows"]) == 5
    assert intraday["raw_rows_by_session"] == {SESSION: 5, PRIOR_1: 1}


def test_the_stored_intraday_rows_are_the_last_append_for_each_event(sources):
    """The append-only log's identity rule survives the round trip: the NVDA
    row in the index is the CLOSED one, not the registered one."""
    index = _index(sources)
    nvda = [
        row for row in index["intraday_outcomes"]["rows"]
        if str(row.get("symbol")) == "NVDA"
    ]
    assert len(nvda) == 1, nvda
    assert nvda[0]["status"] == "closed"
    assert nvda[0]["mfe_pct"] == "5.50"


def test_the_index_declares_its_schema_its_session_and_whether_it_is_pending(sources):
    index = _index(sources)
    assert index["schema"] == "day_review_index_v1"
    assert index["session_date"] == SESSION
    assert index["built_at"]
    # The MSFT 3-session row has not matured, so this index is pending.
    assert index["pending"] is True


def test_an_index_over_only_matured_rows_is_not_pending(sources, tmp_path):
    """The same builder over a horizon file with nothing immature."""
    import day_review_index

    matured = [row for row in SESSION_HORIZON_ROWS if row[8] == "mature"]
    rows = []
    for (symbol, side, scan_date, target, horizon, ret, favorable,
         measured, maturity) in matured:
        scan_row_id = f"{symbol}:{scan_date}:{scan_date}-130129"
        rows.append([
            f"{scan_row_id}:{horizon}", scan_row_id, symbol, side, scan_date, target,
            horizon, horizon, "100.00", "session_bar", "", ret, favorable, measured,
            maturity, "", "favorable_direction_session_v2",
            "entry_session_close_to_target_session_close", "S", "derived_from_bucket",
            "favorite_setup", "avwap_band_bounce", "", "1",
        ])
    Path(sources.session_horizon_outcomes).write_text(
        _csv_text(SESSION_HORIZON_HEADER, rows), encoding="utf-8"
    )
    index = day_review_index.build_index(
        SESSION, lookback_sessions=LOOKBACK, sources=sources, now=NOW
    )
    assert index["pending"] is False


# ---------------------------------------------------------------------------
# 3. it really replaces the streaming
# ---------------------------------------------------------------------------
def test_an_indexed_read_never_opens_the_476_mb_intraday_file(sources, monkeypatch):
    """The speed claim, proved rather than timed: with the index in hand, the
    streaming reader is not called at all."""
    import daily_recap_reader

    expected = _streamed(sources)
    index = _index(sources)

    def _refuse(_path):
        raise AssertionError("read_session streamed the intraday log despite the index")

    monkeypatch.setattr(daily_recap_reader, "_read_intraday_outcomes", _refuse)
    got = daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=index
    )
    assert got == expected


def test_an_indexed_read_never_opens_the_session_horizon_file(sources, monkeypatch):
    import daily_recap_reader

    expected = _streamed(sources)
    index = _index(sources)
    original = daily_recap_reader._read_csv

    def _guard(name, path, clock_field):
        if name == "session_horizon_outcomes":
            raise AssertionError("read_session read the horizon CSV despite the index")
        return original(name, path, clock_field)

    monkeypatch.setattr(daily_recap_reader, "_read_csv", _guard)
    got = daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=index
    )
    assert got == expected


def test_the_small_stores_are_still_read_as_they_are_today(sources, monkeypatch):
    """The index covers TWO files. The eight small ones are read live, so a note
    written since the index was built is still on the page."""
    import daily_recap_reader

    seen: list[str] = []
    original = daily_recap_reader._read_jsonl

    def _spy(name, path, clock_field):
        seen.append(name)
        return original(name, path, clock_field)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl", _spy)
    daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=_index(sources)
    )
    assert "annotations" in seen


# ---------------------------------------------------------------------------
# 4. staleness - ONE rule, one function
# ---------------------------------------------------------------------------
def test_a_pending_index_whose_horizon_could_have_matured_is_stale(sources):
    """The MSFT row targets 2026-09-11. Once that session has closed the stored
    answer may be wrong, so the page rebuilds rather than printing it."""
    import day_review_index

    index = _index(sources)
    assert day_review_index.is_stale(index, now=AFTER_THE_PENDING_TARGET_CLOSED) is True


def test_the_same_pending_index_is_not_stale_before_that_session_closes(sources):
    """Rebuilding on every open would give the index no purpose."""
    import day_review_index

    index = _index(sources)
    assert day_review_index.is_stale(index, now=BEFORE_THE_PENDING_TARGET_CLOSED) is False


def test_an_index_with_nothing_pending_is_never_stale(sources):
    import day_review_index

    index = dict(_index(sources))
    index["pending"] = False
    assert day_review_index.is_stale(index, now=AFTER_THE_PENDING_TARGET_CLOSED) is False
    assert day_review_index.is_stale(index, now=datetime(2027, 1, 4, 9, 0)) is False


# ---------------------------------------------------------------------------
# 5. writing and reading it, and every failure path
# ---------------------------------------------------------------------------
def test_a_written_index_reads_back_to_the_same_session(sources, tmp_path):
    import daily_recap_reader
    import day_review_index

    written = day_review_index.write_index(_index(sources), root=tmp_path)
    assert Path(written).is_file()
    assert not list(Path(written).parent.glob("*.tmp")), "temp-and-rename leaves nothing"

    stored = day_review_index.read_index(SESSION, root=tmp_path)
    assert stored is not None
    assert daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=stored
    ) == _streamed(sources)


def test_a_missing_index_is_simply_absent(tmp_path):
    import day_review_index

    assert day_review_index.read_index(SESSION, root=tmp_path) is None


def test_a_corrupt_index_file_falls_back_to_streaming_rather_than_raising(sources, tmp_path):
    """A half-written file after a power cut is uncertainty, never a broken page."""
    import daily_recap_reader
    import day_review_index

    path = day_review_index.index_path(SESSION, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema": "day_review_index_v1", "rows": [', encoding="utf-8")

    assert day_review_index.read_index(SESSION, root=tmp_path) is None
    assert daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=None
    ) == _streamed(sources)


def test_a_nonsense_index_is_ignored_and_the_stores_are_read_instead(sources):
    """Belt and braces: `read_session` given a mapping that is not an index of
    this session streams, and never returns a half-empty session."""
    import daily_recap_reader

    got = daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources,
        index={"schema": "something_else", "session_date": "1999-01-04"},
    )
    assert got == _streamed(sources)


def test_a_write_that_fails_is_logged_and_never_raised(sources, tmp_path, monkeypatch):
    """A derived, rebuildable artefact may never cost the page that wanted it."""
    import os

    import day_review_index

    def _refuse(*_args, **_kwargs):
        raise OSError("the day_review folder is read-only")

    monkeypatch.setattr(os, "replace", _refuse)
    assert day_review_index.write_index(_index(sources), root=tmp_path) is None


def test_the_index_is_json_serialisable_as_written(sources, tmp_path):
    """It is a file, not an object graph: no datetime, no Path, no dataclass."""
    import day_review_index

    path = day_review_index.write_index(_index(sources), root=tmp_path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["session_date"] == SESSION
    assert payload["schema"] == "day_review_index_v1"
    assert set(payload["intraday_outcomes"]) >= {"rows", "coverage", "raw_rows_by_session"}
    assert set(payload["session_horizon_outcomes"]) >= {"rows", "coverage"}
