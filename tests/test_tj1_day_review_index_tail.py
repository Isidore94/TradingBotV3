"""TJ-1 rounds 2 and 3 - an APPEND is not a REWRITE, and the scope is a NAME.

Round 2: the stamp clause invalidated the index on any change to the four indexed
stores, and `intraday_bounce_outcomes.csv` is appended to all day by the M5
scanner. A warm open of 609 ms became **12,124 ms** after ONE appended row for a
different session, and the 22 MB index was rewritten with nothing on the page
different. So a mismatch is READ: a file that only GREW has its appended TAIL
parsed, and only a row this index would have KEPT makes it stale.

Round 3: the first cut of that scope asked about the SESSION alone, and it was
useless live. Every recent index carries target sessions running weeks forward -
six live indexes from 2026-08-28 to 2026-09-17 all hold 2026-09-18, with targets
out to 2026-10-01 - so `trade_date = today` appends always landed on `rebuild`.
The scope is now `(session, symbol)`: the day's own views read EVERY name of the
selected session and its lookback window, while a target session weeks ahead only
matters for the handful of NAMES the index's own observations reference.

So this fixture is built the way the live store is: windowed swing observations
whose targets run weeks forward, including today. The three cases the rule turns
on are:

* an appended intraday row for TODAY on a name no observation references -> the
  index stands (`moved`), and its 22 MB body is not touched;
* the same row on a name one of them references -> `rebuild`;
* an appended horizon row inside the window -> `rebuild`.

Everything a rewrite can look like - a shrink, a same-size rewrite, a bare touch,
a vanished store - rebuilds, and so does anything the tail cannot answer.
"""

from __future__ import annotations

import csv
import io as _io
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

#: The session the index is OF, and the two before it (its lookback window).
SESSION = "2026-09-10"
PRIOR_1 = "2026-09-09"
PRIOR_2 = "2026-09-08"
#: The clock: Friday morning, so 2026-09-10 is the last completed session and
#: 2026-09-11 is TODAY - the session the M5 scanner is appending rows for while
#: the trader opens yesterday's page.
NOW = datetime(2026, 9, 11, 7, 30)
TODAY = "2026-09-11"
#: Weeks forward, the way a real 3-session horizon row's target looks a month out.
FAR_TARGET = "2026-10-01"

#: A name one of the index's own windowed observations is measured INTO today: an
#: appended row for it changes what the page prints.
TODAY_REFERENCED_SYMBOL = "AAPL"
#: A name none of them reference. Live, this is the RARE case: a session's index
#: references about 1,198 names for the next session (review round 4), so most
#: of the scanner's appends land on a referenced name and rebuild.
TODAY_UNREFERENCED_SYMBOL = "ZZZZ"

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


def _horizon_row(
    symbol: str, scan_date: str, target: str, horizon: str, *, measured: bool = True
) -> str:
    scan_row_id = f"{symbol}:{scan_date}:{scan_date}-130129"
    return _row(
        HORIZON_HEADER,
        observation_id=f"{scan_row_id}:{horizon}",
        scan_row_id=scan_row_id,
        symbol=symbol,
        side="LONG",
        scan_date=scan_date,
        target_session=target,
        horizon_sessions=horizon,
        sessions_spanned=horizon,
        entry_close="100.00",
        entry_close_source="session_bar",
        side_return_pct="2.00" if measured else "",
        favorable="True" if measured else "",
        measured="True" if measured else "",
        maturity="mature" if measured else "immature",
        unmeasured_reason="" if measured else "horizon_not_reached",
        outcome_kind="favorable_direction_session_v2",
        knowledge_basis="entry_session_close_to_target_session_close",
        tier="S",
        tier_source="derived_from_bucket",
        priority_bucket="favorite_setup",
        setup_family="avwap_band_bounce",
        collapsed_same_session="1",
    )


@pytest.fixture()
def sources(tmp_path):
    """The four indexed stores, shaped like the live ones.

    The horizon store is the point: its windowed observations are measured into
    sessions WEEKS AHEAD - one into today, one into October - which is what made a
    session-only scope useless.
    """
    import daily_recap_reader

    home = tmp_path / "home"
    home.mkdir()
    intraday = home / "intraday_bounce_outcomes.csv"
    horizon = home / "master_avwap_session_horizon_outcomes.csv"
    tier = home / "master_avwap_tier_outcomes.csv"
    focus = home / "human_focus_outcomes.csv"

    intraday.write_text(
        INTRADAY_HEADER
        + "\n"
        + _intraday_row(SESSION, "NVDA")
        + _intraday_row(PRIOR_1, "NFLX"),
        encoding="utf-8",
    )
    horizon.write_text(
        HORIZON_HEADER
        + "\n"
        # Inside the window, measured into TODAY: this is the pair that matters.
        + _horizon_row(TODAY_REFERENCED_SYMBOL, PRIOR_1, TODAY, "1", measured=False)
        # Inside the window, measured into next month.
        + _horizon_row("MSFT", PRIOR_2, FAR_TARGET, "3", measured=False)
        # Inside the window and already matured into the selected session.
        + _horizon_row("ORCL", PRIOR_1, SESSION, "1"),
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
# 0. the fixture really has the shape the rule is about
# ---------------------------------------------------------------------------
def test_the_index_carries_observations_measured_weeks_forward(sources):
    """Otherwise every test below would prove nothing about the live store."""
    index = _index(sources)
    targets = {
        row["target_session"] for row in index["session_horizon_outcomes"]["rows"]
    }
    assert TODAY in targets, "no observation is measured into today"
    assert FAR_TARGET in targets, "no observation runs weeks forward"
    assert max(targets) > SESSION


def test_the_scope_is_a_name_and_a_session_not_a_session_alone(sources):
    import day_review_index

    scope = day_review_index._index_scope(_index(sources))
    assert (TODAY, TODAY_REFERENCED_SYMBOL) in scope.pairs
    assert (TODAY, TODAY_UNREFERENCED_SYMBOL) not in scope.pairs
    assert (FAR_TARGET, "MSFT") in scope.pairs
    # The selected session and its window count for ANY name.
    assert scope.session == SESSION
    assert scope.first <= PRIOR_1 <= scope.last


# ---------------------------------------------------------------------------
# 1. today's appends, which is what the scanner does all session
# ---------------------------------------------------------------------------
def test_todays_append_for_an_unreferenced_name_keeps_the_index(sources):
    """The ordinary case: hundreds of names the index never heard of."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_UNREFERENCED_SYMBOL))

    assert day_review_index.stamp_verdict(index, sources=sources)[0] == "moved"
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False


def test_todays_append_for_a_referenced_name_rebuilds(sources):
    """`AAPL`'s windowed observation is measured INTO today, and the swing view
    reads that name's own excursion on it."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_REFERENCED_SYMBOL))

    assert day_review_index.stamp_verdict(index, sources=sources)[0] == "rebuild"
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_appended_horizon_row_inside_the_window_rebuilds(sources):
    """A new observation scanned inside the window changes the swing view."""
    import day_review_index

    index = _index(sources)
    _append(
        sources.session_horizon_outcomes,
        _horizon_row(TODAY_REFERENCED_SYMBOL, PRIOR_1, TODAY, "3", measured=False),
    )
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_hundred_unreferenced_appends_still_keep_it(sources):
    """A whole session of scanner output, none of it this index's business."""
    import day_review_index

    index = _index(sources)
    for number in range(100):
        _append(sources.intraday_outcomes, _intraday_row(TODAY, f"SYM{number:03d}"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False


def test_one_referenced_row_among_many_unreferenced_ones_rebuilds(sources):
    import day_review_index

    index = _index(sources)
    for number in range(20):
        _append(sources.intraday_outcomes, _intraday_row(TODAY, f"SYM{number:03d}"))
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_REFERENCED_SYMBOL))
    for number in range(20, 40):
        _append(sources.intraday_outcomes, _intraday_row(TODAY, f"SYM{number:03d}"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_the_warm_read_still_uses_the_index_after_an_unreferenced_append(sources):
    """End to end: the page's answer comes off the index, not the stores."""
    import daily_recap_reader
    import day_review_index

    index = _index(sources)
    expected = daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=index
    )
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_UNREFERENCED_SYMBOL))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False

    original = daily_recap_reader._read_intraday_outcomes

    def _refuse(_path):
        raise AssertionError("the intraday log was streamed after an unreferenced append")

    daily_recap_reader._read_intraday_outcomes = _refuse
    try:
        got = daily_recap_reader.read_session(
            SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=index
        )
    finally:
        daily_recap_reader._read_intraday_outcomes = original
    assert got == expected


def test_the_tail_is_the_only_thing_read(sources, monkeypatch):
    """An unreferenced append costs a few hundred bytes, not 476 MB."""
    import daily_recap_reader
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_UNREFERENCED_SYMBOL))
    monkeypatch.setattr(
        daily_recap_reader,
        "_read_intraday_outcomes",
        lambda _path: pytest.fail("the staleness check streamed the whole log"),
    )
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False


# ---------------------------------------------------------------------------
# 2. the stamp sidecar
# ---------------------------------------------------------------------------
def test_the_new_stamp_is_recorded_without_rewriting_the_body(sources, tmp_path):
    """22 MB stays put; a few hundred bytes land beside it."""
    import day_review_index

    root = tmp_path / "day_review"
    body = Path(day_review_index.write_index(_index(sources), root=root, now=NOW))
    before = (body.stat().st_mtime_ns, body.stat().st_size)

    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_UNREFERENCED_SYMBOL))
    stored = day_review_index.read_index(SESSION, root=root)
    verdict, stamp = day_review_index.stamp_verdict(stored, sources=sources)
    assert verdict == "moved"

    written = day_review_index.refresh_stamp(stored, stamp=stamp, root=root)
    assert written is not None and Path(written).name == "stamp.json"
    assert (body.stat().st_mtime_ns, body.stat().st_size) == before, "the body was rewritten"
    assert Path(written).stat().st_size < 4096

    # ...and the next read compares against the NEW sizes, so no tail is needed.
    again = day_review_index.read_index(SESSION, root=root)
    assert day_review_index.stamp_verdict(again, sources=sources)[0] == "same"


def test_refresh_stamp_with_a_precomputed_stamp_does_not_stat_again(sources, tmp_path):
    """One verdict per open (reviewer, round 3): the caller has already stat-ed
    the stores and read the tail."""
    import day_review_index

    root = tmp_path / "day_review"
    day_review_index.write_index(_index(sources), root=root, now=NOW)
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_UNREFERENCED_SYMBOL))
    stored = day_review_index.read_index(SESSION, root=root)
    _verdict, stamp = day_review_index.stamp_verdict(stored, sources=sources)

    calls: list[int] = []
    original = day_review_index.sources_stamp
    day_review_index.sources_stamp = lambda *a, **k: (
        calls.append(1) or original(*a, **k)
    )
    try:
        assert day_review_index.refresh_stamp(stored, stamp=stamp, root=root) is not None
    finally:
        day_review_index.sources_stamp = original
    assert calls == [], "refresh_stamp re-stat-ed the stores"


def test_a_stamp_that_matches_the_body_writes_nothing(sources, tmp_path):
    import day_review_index

    root = tmp_path / "day_review"
    day_review_index.write_index(_index(sources), root=root, now=NOW)
    stored = day_review_index.read_index(SESSION, root=root)
    _verdict, stamp = day_review_index.stamp_verdict(stored, sources=sources)
    assert day_review_index.refresh_stamp(stored, stamp=stamp, root=root) is None
    assert not day_review_index.stamp_path(SESSION, root=root).exists()


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
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_UNREFERENCED_SYMBOL))
    stored = day_review_index.read_index(SESSION, root=root)
    day_review_index.refresh_stamp(stored, sources=sources, root=root)
    assert day_review_index.stamp_path(SESSION, root=root).is_file()

    # A real rebuild (the content changed) replaces the body and the sidecar goes.
    _append(sources.intraday_outcomes, _intraday_row(SESSION, "AMD"))
    day_review_index.write_index(_index(sources), root=root, now=NOW)
    assert not day_review_index.stamp_path(SESSION, root=root).exists()


# ---------------------------------------------------------------------------
# 3. in-scope appends of the older kinds
# ---------------------------------------------------------------------------
def test_a_late_finalisation_for_the_indexed_session_rebuilds(sources):
    """The append-only log's own case: the same event, finalized later."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(SESSION, "NVDA", status="closed"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_new_name_on_the_indexed_session_rebuilds(sources):
    """The selected session counts for ANY name: it is the day being read."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(SESSION, TODAY_UNREFERENCED_SYMBOL))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_new_name_inside_the_lookback_window_rebuilds(sources):
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(PRIOR_1, TODAY_UNREFERENCED_SYMBOL))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_append_for_a_far_target_of_a_referenced_name_rebuilds(sources):
    """October is out of every window, and the swing view still reads MSFT there."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(FAR_TARGET, "MSFT"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_append_for_a_far_session_on_an_unreferenced_name_does_not(sources):
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(FAR_TARGET, TODAY_UNREFERENCED_SYMBOL))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False


# ---------------------------------------------------------------------------
# 4. anything the tail cannot answer
# ---------------------------------------------------------------------------
def test_an_appended_row_with_no_readable_session_rebuilds(sources):
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row("", "GOOG"))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_an_appended_row_with_no_symbol_rebuilds(sources):
    """A `(session, symbol)` scope cannot answer a row with no name."""
    import day_review_index

    index = _index(sources)
    _append(sources.intraday_outcomes, _intraday_row(TODAY, ""))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_tail_that_does_not_start_on_a_line_boundary_rebuilds(sources):
    """A stored size taken mid-append points into the middle of a row."""
    import day_review_index

    index = _index(sources)
    stamp = dict(index["sources_stamp"]["intraday_outcomes"])
    stamp["size"] = max(stamp["size"] - 12, 1)
    index["sources_stamp"]["intraday_outcomes"] = stamp
    _append(sources.intraday_outcomes, _intraday_row(TODAY, TODAY_UNREFERENCED_SYMBOL))
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


# ---------------------------------------------------------------------------
# 5. a rewrite, however it looks
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


def test_the_check_is_fast_enough_to_run_on_every_open(sources):
    """Measured rather than asserted by shape: a bounded number of stats plus a
    tail read."""
    import day_review_index

    index = _index(sources)
    for number in range(50):
        _append(sources.intraday_outcomes, _intraday_row(TODAY, f"SYM{number:03d}"))
    start = time.perf_counter()
    for _ in range(20):
        day_review_index.is_stale(index, now=NOW, sources=sources)
    each_ms = (time.perf_counter() - start) * 1000 / 20
    assert each_ms < 25, f"{each_ms:.1f} ms per check"
