"""TJ-1 round-2 blocker - an APPEND is not a REWRITE.

The stamp clause invalidated the index on any change to the four indexed stores,
and `intraday_bounce_outcomes.csv` is appended to all day by the M5 scanner.
Measured on the staged home (reviewer, round 2): a warm open was 609 ms, and after
ONE appended row for a DIFFERENT session it was 12,124 ms with the 22 MB index
rewritten - and nothing the page prints had changed.

So a mismatch is now read, not assumed:

* the file only GREW -> read ONLY the appended tail (seek to the stored size, and
  re-attach the header so the rows parse) and rebuild only if an appended row
  falls inside THIS index's scope: its session, its lookback window, or a target
  session of a windowed swing observation it carries;
* the file SHRANK, or changed at the SAME SIZE -> that is a rewrite (the warehouse
  recompute) and it rebuilds, as does a bare `os.utime` that moves nothing but the
  clock, which a stamp cannot tell from a same-size rewrite;
* anything the tail cannot ANSWER - a row with an unparseable session, a boundary
  that is not a line end, an unreadable or header-less file - rebuilds. Uncertainty
  rebuilds; it never assumes.

An out-of-scope append leaves the 22 MB body alone and records the new stamp in a
few hundred bytes beside it (`stamp.json`), so the next open compares sizes rather
than reading the same tail again.
"""

from __future__ import annotations

import io as _io
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SESSION = "2026-09-10"
PRIOR_1 = "2026-09-09"
#: Far outside the window: this is what the M5 scanner's live appends look like
#: to an index of a session last week.
OUT_OF_SCOPE = "2026-09-17"
NOW = datetime(2026, 9, 11, 7, 30)
LOOKBACK = 3

INTRADAY_HEADER = (
    "schema_version,event_id,event_type,logged_at,trade_date,symbol,direction,"
    "entry_time,entry_price,stop_price,risk_per_share,bars_elapsed,minutes_elapsed,"
    "close_r,mfe_r,mae_r,best_price,worst_price,target_1r_hit,target_2r_hit,stop_hit,"
    "status,milestone_bar,context_json,outcome_mode,eod_close,eod_move_pct,mfe_pct,mae_pct"
)
HORIZON_HEADER = (
    "observation_id,scan_row_id,symbol,side,scan_date,target_session,horizon_sessions,"
    "sessions_spanned,entry_close,entry_close_source,target_close,side_return_pct,"
    "favorable,measured,maturity,unmeasured_reason,outcome_kind,knowledge_basis,tier,"
    "tier_source,priority_bucket,setup_family,favorite_zone,collapsed_same_session"
)
TIER_HEADER = "observation_id,run_timestamp,scan_date,symbol,side,side_return_pct"
HUMAN_FOCUS_HEADER = "trade_date,symbol,side,updated_at,h1_return"


def _row(header: str, **values) -> str:
    columns = header.split(",")
    buffer = _io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", restval="")
    writer.writerow({key: values.get(key, "") for key in columns})
    return buffer.getvalue()


def _intraday_row(session: str, symbol: str, *, status: str = "closed") -> str:
    return _row(
        INTRADAY_HEADER,
        schema_version="1",
        event_id=f"{symbol}_long_{session.replace('-', '')}_09_40_00_ema_15",
        event_type="registered",
        logged_at=f"{session}T13:10:00-07:00",
        trade_date=session,
        symbol=symbol,
        direction="long",
        entry_time=f"{session}T09:40:00",
        entry_price="100.00",
        status=status,
        eod_close="104.00",
        eod_move_pct="4.00",
        mfe_pct="5.50",
        mae_pct="-0.80",
    )


@pytest.fixture()
def sources(tmp_path):
    """The four indexed stores with one in-scope row each."""
    import daily_recap_reader

    home = tmp_path / "home"
    home.mkdir()
    intraday = home / "intraday_bounce_outcomes.csv"
    horizon = home / "master_avwap_session_horizon_outcomes.csv"
    tier = home / "master_avwap_tier_outcomes.csv"
    focus = home / "human_focus_outcomes.csv"

    intraday.write_text(
        INTRADAY_HEADER + "\n" + _intraday_row(SESSION, "NVDA"), encoding="utf-8"
    )
    horizon.write_text(
        HORIZON_HEADER
        + "\n"
        + _row(
            HORIZON_HEADER,
            observation_id="AAPL:1", scan_row_id="AAPL", symbol="AAPL", side="LONG",
            scan_date=PRIOR_1, target_session=SESSION, horizon_sessions="1",
            sessions_spanned="1", entry_close="100.00", side_return_pct="2.00",
            favorable="True", measured="True", maturity="mature",
        ),
        encoding="utf-8",
    )
    tier.write_text(
        TIER_HEADER
        + "\n"
        + _row(TIER_HEADER, observation_id="t1", run_timestamp=f"{SESSION}T13:01:29-07:00",
               scan_date=SESSION, symbol="NVDA", side="LONG", side_return_pct="1.00"),
        encoding="utf-8",
    )
    focus.write_text(
        HUMAN_FOCUS_HEADER
        + "\n"
        + _row(HUMAN_FOCUS_HEADER, trade_date=SESSION, symbol="NVDA", side="LONG",
               updated_at=f"{SESSION}T13:30:00-07:00", h1_return="0.50"),
        encoding="utf-8",
    )
    return daily_recap_reader.RecapSources(
        intraday_outcomes=intraday,
        tier_outcomes=tier,
        session_horizon_outcomes=horizon,
        human_focus_outcomes=focus,
        annotations=home / "trader_annotations.jsonl",
        pick_feedback=home / "pick_feedback.jsonl",
        swing_favorites=home / "swing_favorites.jsonl",
        review_events=home / "alert_review_events.jsonl",
        preference_report=home / "preference_trade_outcomes.csv",
        staged_picks=home / "auto_populate_pending.json",
        environment_labels=home / "d1_environment.jsonl",
        working_lately=home / "snapshot_latest.json",
    )


def _index(sources):
    import day_review_index

    return day_review_index.build_index(
        SESSION, lookback_sessions=LOOKBACK, sources=sources, now=NOW
    )


def _append(path, text: str) -> None:
    """Append, and make sure the mtime really moves on a fast filesystem."""
    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        handle.write(text)
    stat = Path(path).stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


# ---------------------------------------------------------------------------
# 1. an append outside this index's scope
# ---------------------------------------------------------------------------
def test_one_appended_out_of_scope_row_keeps_the_index(sources):
    """The M5 scanner appending today's rows must not expire last week's index."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(OUT_OF_SCOPE, "TSLA"))

    assert day_review_index.stamp_verdict(index, sources=sources)[0] == "moved"
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False


def test_the_warm_read_still_uses_that_index(sources):
    """End to end: the page's answer comes off the index, not the stores."""
    import daily_recap_reader

    index = _index(sources)
    expected = daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=index
    )
    _append(sources.intraday_outcomes, _intraday_row(OUT_OF_SCOPE, "TSLA"))

    def _refuse(_path):
        raise AssertionError("the intraday log was streamed after an out-of-scope append")

    import day_review_index

    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False
    original = daily_recap_reader._read_intraday_outcomes
    daily_recap_reader._read_intraday_outcomes = _refuse
    try:
        got = daily_recap_reader.read_session(
            SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=index
        )
    finally:
        daily_recap_reader._read_intraday_outcomes = original
    assert got == expected


def test_the_new_stamp_is_recorded_without_rewriting_the_body(sources, tmp_path):
    """22 MB stays put; a few hundred bytes land beside it."""
    import day_review_index

    root = tmp_path / "day_review"
    body = Path(day_review_index.write_index(_index(sources), root=root, now=NOW))
    before = (body.stat().st_mtime_ns, body.stat().st_size)

    _append(sources.intraday_outcomes, _intraday_row(OUT_OF_SCOPE, "TSLA"))
    stored = day_review_index.read_index(SESSION, root=root)
    assert day_review_index.is_stale(stored, now=NOW, sources=sources) is False

    written = day_review_index.refresh_stamp(stored, sources=sources, root=root)
    assert written is not None and Path(written).name == "stamp.json"
    assert (body.stat().st_mtime_ns, body.stat().st_size) == before, "the body was rewritten"
    assert Path(written).stat().st_size < 4096

    # ...and the next read compares against the NEW sizes, so no tail is needed.
    again = day_review_index.read_index(SESSION, root=root)
    assert day_review_index.stamp_verdict(again, sources=sources)[0] == "same"


def test_a_stamp_sidecar_for_another_session_is_ignored(sources, tmp_path):
    import day_review_index

    root = tmp_path / "day_review"
    day_review_index.write_index(_index(sources), root=root, now=NOW)
    path = day_review_index.stamp_path(SESSION, root=root)
    path.write_text(
        json.dumps({"schema": day_review_index.SCHEMA, "session_date": "1999-01-04",
                    "sources_stamp": {}}),
        encoding="utf-8",
    )
    stored = day_review_index.read_index(SESSION, root=root)
    assert stored["sources_stamp"], "a foreign sidecar replaced the real stamp"


def test_a_fresh_body_write_clears_the_sidecar(sources, tmp_path):
    import day_review_index

    root = tmp_path / "day_review"
    day_review_index.write_index(_index(sources), root=root, now=NOW)
    _append(sources.intraday_outcomes, _intraday_row(OUT_OF_SCOPE, "TSLA"))
    stored = day_review_index.read_index(SESSION, root=root)
    day_review_index.refresh_stamp(stored, sources=sources, root=root)
    assert day_review_index.stamp_path(SESSION, root=root).is_file()

    # A real rebuild (the content changed) replaces the body and the sidecar goes.
    _append(sources.intraday_outcomes, _intraday_row(SESSION, "AMD"))
    day_review_index.write_index(_index(sources), root=root, now=NOW)
    assert not day_review_index.stamp_path(SESSION, root=root).exists()


# ---------------------------------------------------------------------------
# 2. an append INSIDE this index's scope
# ---------------------------------------------------------------------------
def test_a_late_finalisation_for_the_indexed_session_rebuilds(sources):
    """The append-only log's own case: the same event, finalized later."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(SESSION, "NVDA", status="closed"))

    assert day_review_index.stamp_verdict(index, sources=sources)[0] == "rebuild"
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_appended_row_inside_the_lookback_window_rebuilds(sources):
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(PRIOR_1, "NFLX"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_appended_row_for_a_target_session_rebuilds(sources):
    """A windowed swing observation is measured INTO a session, and the swing
    view reads that name's M5 excursion on it."""
    import day_review_index

    _append(
        sources.session_horizon_outcomes,
        _row(HORIZON_HEADER, observation_id="MSFT:3", scan_row_id="MSFT", symbol="MSFT",
             side="SHORT", scan_date=PRIOR_1, target_session="2026-09-14",
             horizon_sessions="3", sessions_spanned="3", entry_close="100.00",
             measured="", maturity="immature"),
    )
    index = _index(sources)
    assert "2026-09-14" in {
        row["target_session"] for row in index["session_horizon_outcomes"]["rows"]
    }

    _append(sources.intraday_outcomes, _intraday_row("2026-09-14", "MSFT"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_appended_row_with_no_readable_session_rebuilds(sources):
    """Uncertainty rebuilds, never assumes."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row("", "GOOG"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_tail_that_does_not_start_on_a_line_boundary_rebuilds(sources):
    """A stored size taken mid-append points into the middle of a row."""
    import day_review_index

    index = _index(sources)
    stamp = dict(index["sources_stamp"]["intraday_outcomes"])
    stamp["size"] = max(stamp["size"] - 12, 1)
    index["sources_stamp"]["intraday_outcomes"] = stamp
    _append(sources.intraday_outcomes, _intraday_row(OUT_OF_SCOPE, "TSLA"))

    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


# ---------------------------------------------------------------------------
# 3. a rewrite, however it looks
# ---------------------------------------------------------------------------
def test_a_truncated_file_rebuilds(sources):
    import day_review_index

    index = _index(sources)
    Path(sources.tier_outcomes).write_text(TIER_HEADER + "\n", encoding="utf-8")
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_same_size_rewrite_rebuilds(sources):
    """The warehouse recompute rewrites in place; a row can change value without
    changing the file's length."""
    import day_review_index

    index = _index(sources)
    path = Path(sources.human_focus_outcomes)
    text = path.read_text(encoding="utf-8")
    rewritten = text.replace("0.50", "0.90")
    assert len(rewritten) == len(text), "the fixture must keep the size identical"
    path.write_text(rewritten, encoding="utf-8")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))

    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_bare_touch_with_no_new_bytes_rebuilds(sources):
    """Documented: a stamp cannot tell `os.utime` from a same-size rewrite, and a
    touched store with no new bytes is rare enough to pay for one rebuild."""
    import day_review_index

    index = _index(sources)
    path = Path(sources.intraday_outcomes)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000))

    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_store_that_vanished_rebuilds(sources):
    import day_review_index

    index = _index(sources)
    Path(sources.session_horizon_outcomes).unlink()
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_unchanged_store_set_reads_as_same(sources):
    import day_review_index

    index = _index(sources)
    assert day_review_index.stamp_verdict(index, sources=sources)[0] == "same"
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False


def test_the_tail_is_the_only_thing_read(sources, monkeypatch):
    """The point of the rule: an out-of-scope append costs a few hundred bytes,
    not 476 MB. `_read_intraday_outcomes` is the streaming reader, and it is not
    called at all."""
    import daily_recap_reader
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(OUT_OF_SCOPE, "TSLA"))
    monkeypatch.setattr(
        daily_recap_reader,
        "_read_intraday_outcomes",
        lambda _path: pytest.fail("the staleness check streamed the whole log"),
    )
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False


def test_the_check_is_fast_enough_to_run_on_every_open(sources):
    """Measured rather than asserted by shape: a bounded number of stats plus a
    tail read."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(OUT_OF_SCOPE, "TSLA"))
    start = time.perf_counter()
    for _ in range(20):
        day_review_index.is_stale(index, now=NOW, sources=sources)
    each_ms = (time.perf_counter() - start) * 1000 / 20
    assert each_ms < 25, f"{each_ms:.1f} ms per check"
