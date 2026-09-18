"""TJ-1 follow-up - the index covers every store over a megabyte, and only those.

Gate #145 asks for an open "in under one second". The first index covered the two
biggest stores and left an indexed read at 2,217 ms on a staged copy of the live
home folder. Measured there, store by store (2026-09-17):

    intraday_bounce_outcomes.csv              476.45 MB   4,995 ms
    master_avwap_session_horizon_outcomes.csv  30.87 MB   2,704 ms
    master_avwap_tier_outcomes.csv             14.31 MB     944 ms
    human_focus_outcomes.csv                    1.04 MB     164 ms
    pick_feedback.jsonl                         0.40 MB      36 ms
    alert_review_events.jsonl                   0.27 MB      15 ms
    preference_trade_outcomes.csv               0.44 MB       8 ms
    trader_annotations.jsonl                    0.35 MB       4 ms
    d1_environment / staged_picks / favorites / snapshot      ~2 ms

So the index now covers FOUR stores, and the six small ones stay live - which is
the freshness rule, not an optimisation: a veto, a note, a favorite or a staged
pick from a minute ago has to be on the page.

What this file pins:

* the whole-`RecapSession` equality, EXTENDED to the two stores that just joined -
  an index is a cache and a cache may never change the answer;
* the full-file coverage of the new stores, carried forward, beside a kept slice
  that is smaller than it;
* all-or-nothing revival, which is how an index written by the previous build
  (two stores, same schema name) is treated as ABSENT rather than half-used;
* that the six small stores are still opened on every read;
* the two lookups that were rewritten from a walk into a dict when the profile
  showed them to be the whole remaining cost - each checked against a
  brute-force reference walk, on rows with the blank sides and repeated names
  that make the two able to disagree.
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

SESSION = "2026-09-10"
PRIOR_1 = "2026-09-09"
PRIOR_2 = "2026-09-08"
#: Long before the window; it must never reach the slice.
LONG_AGO = "2026-09-02"
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
#: The live column order of `master_avwap_tier_outcomes.csv`.
TIER_HEADER = (
    "observation_id,scan_row_id,run_id,run_timestamp,run_date,watchlist_label,scan_date,"
    "future_scan_date,horizon_sessions,tier,tier_source,symbol,side,priority_bucket,"
    "priority_score,setup_family,favorite_zone,entry_close,future_close,raw_return_pct,"
    "side_return_pct,win,spy_forward_return_pct,spy_relative_side_return_pct,"
    "sessions_spanned,stale_horizon,positive_scan_factor_match_count,"
    "positive_scan_factor_matches,outcome_kind"
)
#: The live column order of `human_focus_outcomes.csv`.
HUMAN_FOCUS_HEADER = (
    "trade_date,symbol,side,source,entry_date,entry_close,h1_date,h1_return,h3_date,"
    "h3_return,h5_date,h5_return,h10_date,h10_return,matured_horizons,fully_matured,"
    "updated_at"
)


def _csv(header: str, rows) -> str:
    """Rows as dicts of the columns that matter; the rest stay empty."""
    columns = header.split(",")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", restval="")
    writer.writeheader()
    for row in rows:
        unknown = set(row) - set(columns)
        assert not unknown, unknown
        writer.writerow(row)
    return buffer.getvalue()


def _intraday_text() -> str:
    context = json.dumps({"market_environment": "bullish_strong"}, sort_keys=True)
    rows = [
        # The append-only log's identity rule: NVDA registered, then finalized.
        {"event_id": "NVDA_long_1", "trade_date": SESSION, "symbol": "NVDA",
         "direction": "long", "status": "open", "context_json": context,
         "logged_at": f"{SESSION}T13:10:00-07:00", "entry_time": f"{SESSION}T09:40:00"},
        {"event_id": "NVDA_long_1", "trade_date": SESSION, "symbol": "NVDA",
         "direction": "long", "status": "closed", "context_json": context,
         "logged_at": f"{SESSION}T13:20:00-07:00", "entry_time": f"{SESSION}T09:40:00",
         "mfe_pct": "5.50", "mae_pct": "-0.80", "eod_close": "104.00", "eod_move_pct": "4.00"},
        {"event_id": "TSLA_short_1", "trade_date": SESSION, "symbol": "TSLA",
         "direction": "short", "status": "closed", "context_json": context,
         "logged_at": f"{SESSION}T13:20:00-07:00", "entry_time": f"{SESSION}T09:45:00",
         "mfe_pct": "6.00", "mae_pct": "-0.80", "eod_close": "95.00", "eod_move_pct": "5.00"},
        # A prior session's row, which the swing view reads as a first-favorable.
        {"event_id": "NFLX_long_1", "trade_date": PRIOR_1, "symbol": "NFLX",
         "direction": "long", "status": "closed", "context_json": context,
         "logged_at": f"{PRIOR_1}T13:20:00-07:00", "entry_time": f"{PRIOR_1}T09:40:00",
         "mfe_pct": "2.00", "mae_pct": "-1.00", "eod_close": "707.00", "eod_move_pct": "1.00"},
        # Outside the window entirely.
        {"event_id": "GOOG_long_1", "trade_date": LONG_AGO, "symbol": "GOOG",
         "direction": "long", "status": "closed", "context_json": context,
         "logged_at": f"{LONG_AGO}T13:20:00-07:00", "entry_time": f"{LONG_AGO}T09:40:00",
         "mfe_pct": "9.00", "mae_pct": "-9.00", "eod_close": "99.00", "eod_move_pct": "9.00"},
    ]
    return _csv(INTRADAY_HEADER, rows)


def _horizon_text() -> str:
    rows = []
    for symbol, side, scan_date, target, horizon, ret, measured, maturity in (
        ("AAPL", "LONG", PRIOR_1, SESSION, "1", "2.00", "True", "mature"),
        ("AAPL", "LONG", PRIOR_1, SESSION, "3", "3.00", "True", "mature"),
        ("ORCL", "SHORT", PRIOR_2, PRIOR_1, "1", "1.25", "True", "mature"),
        ("MSFT", "SHORT", PRIOR_2, "2026-09-11", "3", "", "", "immature"),
        ("GOOG", "LONG", LONG_AGO, "2026-09-03", "1", "4.00", "True", "mature"),
    ):
        scan_row_id = f"{symbol}:{scan_date}:{scan_date}-130129"
        rows.append({
            "observation_id": f"{scan_row_id}:{horizon}", "scan_row_id": scan_row_id,
            "symbol": symbol, "side": side, "scan_date": scan_date,
            "target_session": target, "horizon_sessions": horizon,
            "sessions_spanned": horizon, "entry_close": "100.00",
            "entry_close_source": "session_bar", "side_return_pct": ret,
            "favorable": measured, "measured": measured, "maturity": maturity,
            "unmeasured_reason": "" if measured else "horizon_not_reached",
            "outcome_kind": "favorable_direction_session_v2",
            "knowledge_basis": "entry_session_close_to_target_session_close",
            "tier": "S", "tier_source": "derived_from_bucket",
            "priority_bucket": "favorite_setup", "setup_family": "avwap_band_bounce",
            "collapsed_same_session": "1",
        })
    return _csv(HORIZON_HEADER, rows)


#: Tier rows: two inside the window, one long before it. NOTHING in `read_session`
#: reads a tier ROW - the store is opened for its coverage line alone - which is
#: exactly why a slice of it cannot change the answer.
TIER_ROWS = (
    (SESSION, "NVDA"),
    (PRIOR_1, "AAPL"),
    (LONG_AGO, "GOOG"),
)
#: Human-focus rows, keyed by their own `trade_date`.
HUMAN_FOCUS_ROWS = (
    (SESSION, "NVDA"),
    (PRIOR_2, "ORCL"),
    (LONG_AGO, "GOOG"),
)


def _tier_text() -> str:
    return _csv(TIER_HEADER, [
        {
            "observation_id": f"{symbol}:{scan_date}:1", "scan_row_id": f"{symbol}:{scan_date}",
            "run_id": "run-1", "run_timestamp": f"{scan_date}T13:01:29-07:00",
            "run_date": scan_date, "scan_date": scan_date, "future_scan_date": scan_date,
            "horizon_sessions": "1", "tier": "S", "symbol": symbol, "side": "LONG",
            "side_return_pct": "1.00", "win": "True", "outcome_kind": "tier_v1",
        }
        for scan_date, symbol in TIER_ROWS
    ])


def _human_focus_text() -> str:
    return _csv(HUMAN_FOCUS_HEADER, [
        {
            "trade_date": trade_date, "symbol": symbol, "side": "LONG",
            "source": "focus", "entry_date": trade_date, "entry_close": "100.00",
            "h1_date": trade_date, "h1_return": "0.50", "matured_horizons": "1",
            "fully_matured": "no", "updated_at": f"{trade_date}T13:30:00-07:00",
        }
        for trade_date, symbol in HUMAN_FOCUS_ROWS
    ])


@pytest.fixture()
def sources(tmp_path):
    """A `RecapSources` whose twelve paths all live under `tmp_path`.

    The four indexed stores carry rows; the eight others are absent, which is a
    NAMED coverage reason and is what an index must reproduce exactly.
    """
    import daily_recap_reader

    home = tmp_path / "home"
    home.mkdir()
    files = {
        "intraday_outcomes": ("intraday_bounce_outcomes.csv", _intraday_text()),
        "session_horizon_outcomes": (
            "master_avwap_session_horizon_outcomes.csv", _horizon_text(),
        ),
        "tier_outcomes": ("master_avwap_tier_outcomes.csv", _tier_text()),
        "human_focus_outcomes": ("human_focus_outcomes.csv", _human_focus_text()),
    }
    written = {}
    for name, (filename, text) in files.items():
        path = home / filename
        path.write_text(text, encoding="utf-8")
        written[name] = path
    return daily_recap_reader.RecapSources(
        intraday_outcomes=written["intraday_outcomes"],
        tier_outcomes=written["tier_outcomes"],
        session_horizon_outcomes=written["session_horizon_outcomes"],
        annotations=home / "trader_annotations.jsonl",
        pick_feedback=home / "pick_feedback.jsonl",
        swing_favorites=home / "swing_favorites.jsonl",
        human_focus_outcomes=written["human_focus_outcomes"],
        review_events=home / "alert_review_events.jsonl",
        preference_report=home / "preference_trade_outcomes.csv",
        staged_picks=home / "auto_populate_pending.json",
        environment_labels=home / "d1_environment.jsonl",
        working_lately=home / "snapshot_latest.json",
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


def _indexed(sources, index=None):
    import daily_recap_reader

    return daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources,
        index=_index(sources) if index is None else index,
    )


# ---------------------------------------------------------------------------
# 1. the equality, extended
# ---------------------------------------------------------------------------
def test_the_calendar_still_says_what_this_fixture_assumes():
    import market_calendar

    day = date.fromisoformat(SESSION)
    assert market_calendar.previous_session(day).isoformat() == PRIOR_1
    assert market_calendar.previous_session(date.fromisoformat(PRIOR_1)).isoformat() == PRIOR_2
    assert market_calendar.last_completed_session(NOW).isoformat() == SESSION


def test_an_indexed_read_is_the_same_session_as_a_streamed_one(sources):
    """Dataclass equality over the WHOLE `RecapSession`, coverage included, with
    all four stores populated."""
    assert _indexed(sources) == _streamed(sources)


def test_the_index_declares_a_payload_for_every_store_it_covers(sources):
    import day_review_index

    index = _index(sources)
    assert set(day_review_index.INDEXED_SOURCES) == {
        "intraday_outcomes", "session_horizon_outcomes",
        "tier_outcomes", "human_focus_outcomes",
    }
    for name in day_review_index.INDEXED_SOURCES:
        assert set(index[name]) >= {"rows", "coverage", "raw_rows_by_session"}, name
        assert index[name]["coverage"]["name"] == name


def test_the_new_stores_carry_the_full_file_coverage_over_a_smaller_slice(sources):
    """The coverage line answers "was the file there and how big was it", so it
    counts every row on disk; the slice under it is the window's rows."""
    index = _index(sources)

    tier = index["tier_outcomes"]
    assert tier["coverage"]["rows"] == len(TIER_ROWS)
    assert [row["scan_date"] for row in tier["rows"]] == [SESSION, PRIOR_1]
    assert LONG_AGO not in [row["scan_date"] for row in tier["rows"]]

    focus = index["human_focus_outcomes"]
    assert focus["coverage"]["rows"] == len(HUMAN_FOCUS_ROWS)
    assert [row["trade_date"] for row in focus["rows"]] == [SESSION, PRIOR_2]
    assert len(focus["rows"]) < focus["coverage"]["rows"]


def test_the_coverage_stamps_survive_the_round_trip(sources):
    """`oldest` / `newest` are datetimes on the way in and on the way out: a
    coverage line that lost its clock would relabel a fresh file as undated."""
    streamed = _streamed(sources)
    indexed = _indexed(sources)
    for name in ("tier_outcomes", "human_focus_outcomes"):
        assert indexed.coverage[name] == streamed.coverage[name]
        assert indexed.coverage[name].newest is not None


def test_an_indexed_read_opens_neither_new_store(sources, monkeypatch):
    """The speed claim, proved rather than timed."""
    import daily_recap_reader

    expected = _streamed(sources)
    index = _index(sources)
    original = daily_recap_reader._read_csv
    opened: list[str] = []

    def _spy(name, path, clock_field):
        opened.append(name)
        return original(name, path, clock_field)

    monkeypatch.setattr(daily_recap_reader, "_read_csv", _spy)
    monkeypatch.setattr(
        daily_recap_reader,
        "_read_intraday_outcomes",
        lambda _path: pytest.fail("the intraday log was streamed despite the index"),
    )
    got = daily_recap_reader.read_session(
        SESSION, lookback_sessions=LOOKBACK, now=NOW, sources=sources, index=index
    )
    assert got == expected
    assert "tier_outcomes" not in opened
    assert "human_focus_outcomes" not in opened
    assert "session_horizon_outcomes" not in opened
    # The one CSV that is NOT indexed is still read: it is 0.44 MB and it carries
    # the journal joins, which change as trades arrive.
    assert "preference_report" in opened


def test_every_small_store_is_still_read_live(sources, monkeypatch):
    """The freshness rule. A veto, a note, a favorite, a staged pick, a session
    label and the snapshot are all read on every open."""
    import daily_recap_reader

    seen: list[str] = []
    original = daily_recap_reader._read_jsonl

    def _spy(name, path, clock_field):
        seen.append(name)
        return original(name, path, clock_field)

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl", _spy)
    _indexed(sources)
    assert set(seen) == {"annotations", "pick_feedback", "swing_favorites", "review_events"}


# ---------------------------------------------------------------------------
# 2. all or nothing
# ---------------------------------------------------------------------------
def test_an_index_that_carries_only_the_old_two_stores_is_treated_as_absent(sources):
    """An index written by the previous build carries the same schema NAME and
    two of the four payloads. Half a cache is the shape of bug where one store is
    a week old and the page looks right, so it reads as absent and the stores are
    streamed."""
    import day_review_index

    index = dict(_index(sources))
    index.pop("tier_outcomes")
    index.pop("human_focus_outcomes")

    assert day_review_index.stores_for(
        index, session_date=SESSION, lookback_sessions=LOOKBACK
    ) is None
    assert _indexed(sources, index) == _streamed(sources)


def test_a_store_payload_that_lost_its_coverage_block_is_treated_as_absent(sources):
    import day_review_index

    index = dict(_index(sources))
    index["tier_outcomes"] = {"rows": []}
    assert day_review_index.stores_for(
        index, session_date=SESSION, lookback_sessions=LOOKBACK
    ) is None
    assert _indexed(sources, index) == _streamed(sources)


def test_the_revived_stores_come_back_keyed_by_name(sources):
    import day_review_index

    stores = day_review_index.stores_for(
        _index(sources), session_date=SESSION, lookback_sessions=LOOKBACK
    )
    assert set(stores) == set(day_review_index.INDEXED_SOURCES)
    assert stores["tier_outcomes"].coverage.name == "tier_outcomes"


def test_the_whole_index_is_still_one_json_file(sources, tmp_path):
    import day_review_index

    path = day_review_index.write_index(_index(sources), root=tmp_path / "day_review")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in day_review_index.INDEXED_SOURCES:
        assert set(payload[name]) >= {"rows", "coverage"}


# ---------------------------------------------------------------------------
# 3. the two lookups that were rewritten, against a brute-force reference
# ---------------------------------------------------------------------------
def _reference_outcome_for(outcomes, session_date, symbol, side):
    """The walk `_outcome_for` used to be, kept here as the reference."""
    import daily_recap_reader

    candidates = [
        row
        for row in outcomes
        if row.trade_date == session_date
        and row.symbol == symbol
        and (not side or not row.side or row.side == side)
    ]
    return daily_recap_reader._best_outcome(candidates)


def _reference_d1_horizon_row(store, session_date, symbol, side, lookback_sessions):
    """The walk `_d1_horizon_row` used to be, kept here as the reference."""
    import daily_recap_reader as r

    for row in store.rows:
        if r._session_text(row.get("scan_date")) != session_date:
            continue
        if r._symbol(row.get("symbol")) != symbol:
            continue
        row_side = r._side(row.get("side"))
        if side and row_side and row_side != side:
            continue
        try:
            if int(str(row.get("horizon_sessions") or "")) != lookback_sessions:
                continue
        except ValueError:
            continue
        return row
    return None


def _outcome(**overrides):
    import daily_recap_reader as r

    row = {
        "event_id": "e1", "logged_at": None, "trade_date": SESSION, "symbol": "NVDA",
        "side": "LONG", "entry_time": None, "status": "closed", "mfe_pct": 1.0,
        "mae_pct": -1.0, "eod_move_pct": 0.5,
    }
    row.update(overrides)
    return r._Outcome(**row)


#: The rows that make a keyed lookup and a walk able to disagree: a BLANK side
#: (which matches either request), the same name twice on one day, the same name
#: on another day, and another name entirely.
AMBIGUOUS_OUTCOMES = (
    _outcome(event_id="a", side="LONG", mfe_pct=1.0),
    _outcome(event_id="b", side="", mfe_pct=4.0),
    _outcome(event_id="c", side="SHORT", mfe_pct=9.0),
    _outcome(event_id="d", side="LONG", mfe_pct=None),
    _outcome(event_id="e", trade_date=PRIOR_1, side="LONG", mfe_pct=7.0),
    _outcome(event_id="f", symbol="TSLA", side="LONG", mfe_pct=3.0),
)


@pytest.mark.parametrize("session", [SESSION, PRIOR_1, LONG_AGO])
@pytest.mark.parametrize("symbol", ["NVDA", "TSLA", "AMD"])
@pytest.mark.parametrize("side", ["LONG", "SHORT", ""])
def test_the_grouped_outcome_lookup_answers_exactly_what_the_walk_did(session, symbol, side):
    import daily_recap_reader

    grouped = daily_recap_reader._outcomes_by_name(AMBIGUOUS_OUTCOMES)
    assert daily_recap_reader._outcome_for(grouped, session, symbol, side) == (
        _reference_outcome_for(AMBIGUOUS_OUTCOMES, session, symbol, side)
    )


def test_the_flat_sequence_still_works_because_other_callers_hand_one_in():
    import daily_recap_reader

    for side in ("LONG", "SHORT", ""):
        assert daily_recap_reader._outcome_for(AMBIGUOUS_OUTCOMES, SESSION, "NVDA", side) == (
            _reference_outcome_for(AMBIGUOUS_OUTCOMES, SESSION, "NVDA", side)
        )


@pytest.mark.parametrize("symbol", ["AAPL", "ORCL", "MSFT", "AMD"])
@pytest.mark.parametrize("side", ["LONG", "SHORT", ""])
@pytest.mark.parametrize("horizon", [1, 3])
def test_the_grouped_horizon_lookup_answers_exactly_what_the_walk_did(
    sources, symbol, side, horizon
):
    """Including the FIRST-match rule: AAPL has a 1-session and a 3-session row
    scanned on the same date, in file order."""
    import daily_recap_reader

    store = daily_recap_reader._read_csv(
        "session_horizon_outcomes", sources.session_horizon_outcomes, "scan_date"
    )
    for session in (SESSION, PRIOR_1, PRIOR_2):
        grouped = daily_recap_reader._horizon_rows_by_name(store, session)
        assert daily_recap_reader._d1_horizon_row(
            grouped.get(symbol, ()), symbol, side, horizon
        ) == _reference_d1_horizon_row(store, session, symbol, side, horizon)


def test_a_horizon_row_with_an_unreadable_horizon_is_skipped_by_both(sources):
    import daily_recap_reader

    store = daily_recap_reader._Store(
        (
            {"scan_date": SESSION, "symbol": "AMD", "side": "LONG", "horizon_sessions": "one"},
            {"scan_date": SESSION, "symbol": "AMD", "side": "LONG", "horizon_sessions": "3"},
        ),
        daily_recap_reader.SourceCoverage("session_horizon_outcomes", "x", 2, None, None),
    )
    grouped = daily_recap_reader._horizon_rows_by_name(store, SESSION)
    got = daily_recap_reader._d1_horizon_row(grouped.get("AMD", ()), "AMD", "LONG", 3)
    assert got == _reference_d1_horizon_row(store, SESSION, "AMD", "LONG", 3)
    assert got is not None and got["horizon_sessions"] == "3"
