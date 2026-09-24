"""Day Recap coach: the one writer for the trader's recap inputs (recap_store)."""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

ET = timezone(timedelta(hours=-4))
NOW = datetime(2026, 9, 22, 17, 30, tzinfo=ET)


@pytest.fixture()
def store(tmp_path):
    import recap_store

    return recap_store, tmp_path / "recap.jsonl"


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_every_record_carries_id_session_tz_aware_time_schema_and_supersedes(store):
    rs, path = store
    row = rs.record_lesson(
        session_date="2026-09-22", keep="waited for the pullback", stop="chasing the open",
        try_="size down in chop", mood=3, now=NOW, path=path,
    )
    assert row["kind"] == rs.KIND_LESSON
    assert row["id"].startswith("rc-")
    assert row["session_date"] == "2026-09-22"
    assert datetime.fromisoformat(row["recorded_at"]).tzinfo is not None
    assert row["schema"] == rs.SCHEMA
    assert row["supersedes"] == ""
    assert row["recorded_after_close"] is True
    assert _lines(path) == [row]


def test_a_naive_clock_is_refused(store):
    rs, path = store
    with pytest.raises(rs.RecapError):
        rs.record_rule(session_date="2026-09-22", text="hold winners", now=datetime(2026, 9, 22, 17), path=path)
    assert not path.exists()


def test_vocabularies_are_enforced(store):
    rs, path = store
    with pytest.raises(rs.RecapError):
        rs.record_rule(session_date="2026-09-22", text="x", tag="be_rich", now=NOW, path=path)
    with pytest.raises(rs.RecapError):
        rs.record_rule_check(session_date="2026-09-22", answer="maybe", now=NOW, path=path)
    with pytest.raises(rs.RecapError):
        rs.record_clue(
            session_date="2026-09-22", symbol="AMD", timeframe="M5",
            bar_time=datetime(2026, 9, 22, 10, 5, tzinfo=ET), price=150.0,
            clue_tag="moon_phase", now=NOW, path=path,
        )
    with pytest.raises(rs.RecapError):
        rs.record_environment_verdict(
            session_date="2026-09-22", auto_label="bullish_weak", verdict="sideways", now=NOW, path=path,
        )
    with pytest.raises(rs.RecapError):
        rs.record_card_answer(
            session_date="2026-09-22", card_id="c1", card_kind="horoscope",
            subject={"trade_id": "t1"}, option="good", now=NOW, path=path,
        )
    with pytest.raises(rs.RecapError):
        rs.record_lesson(session_date="2026-09-22", mood=9, keep="x", now=NOW, path=path)
    with pytest.raises(rs.RecapError):
        rs.record_lesson(session_date="2026-09-22", now=NOW, path=path)
    with pytest.raises(rs.RecapError):
        rs.record_lesson(session_date="2026-09-20", keep="weekend", now=NOW, path=path)
    assert not path.exists()


def test_clue_needs_a_tz_aware_bar_time_and_keeps_its_links(store):
    rs, path = store
    with pytest.raises(rs.RecapError):
        rs.record_clue(
            session_date="2026-09-22", symbol="AMD", timeframe="M5",
            bar_time=datetime(2026, 9, 22, 10, 5), price=150.0,
            clue_tag="volume_dry_up", now=NOW, path=path,
        )
    row = rs.record_clue(
        session_date="2026-09-22", symbol="amd", timeframe="m5",
        bar_time="2026-09-22T10:05:00-04:00", price=150.25, clue_tag="volume_dry_up",
        text="volume dried up into the pullback", card_id="card-7", trade_id="t1",
        now=NOW, path=path,
    )
    assert row["symbol"] == "AMD"
    assert row["timeframe"] == "M5"
    assert row["bar_time"] == "2026-09-22T10:05:00-04:00"
    assert row["links"] == {"card_id": "card-7", "trade_id": "t1", "pick_id": ""}


def test_environment_verdict_uses_the_auto_environment_vocabulary(store):
    """The corrected label is one the desk's own auto environment can emit."""
    rs, path = store
    source = (SCRIPTS / "bounce_bot_lib" / "legacy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    keys = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MARKET_ENVIRONMENTS" for target in node.targets
        ):
            keys = tuple(ast.literal_eval(key) for key in node.value.keys)
    assert keys is not None
    assert set(rs.ENVIRONMENT_LABELS) == set(keys)
    clue = rs.record_clue(
        session_date="2026-09-22", symbol="SPY", timeframe="M5",
        bar_time=datetime(2026, 9, 22, 11, 0, tzinfo=ET), price=774.0,
        clue_tag="vwap_reclaim", now=NOW, path=path,
    )
    row = rs.record_environment_verdict(
        session_date="2026-09-22", auto_label="bearish_strong", verdict="neutral_chop",
        clue_ids=[clue["id"]], text="it was chop, not a trend", now=NOW, path=path,
    )
    assert row["verdict"] == "neutral_chop"
    assert row["clue_ids"] == [clue["id"]]
    agree = rs.record_environment_verdict(
        session_date="2026-09-22", auto_label="bullish_weak", verdict="agree", now=NOW, path=path,
    )
    assert agree["verdict"] == rs.VERDICT_AGREE


def test_a_correction_supersedes_and_readers_fold_it(store):
    rs, path = store
    first = rs.record_rule(session_date="2026-09-22", text="hold winners", tag="hold_winners", now=NOW, path=path)
    fixed = rs.record_rule(
        session_date="2026-09-22", text="hold winners to 2R", tag="hold_winners",
        supersedes=first["id"], now=NOW + timedelta(minutes=1), path=path,
    )
    assert len(_lines(path)) == 2  # append-only: the first row is still on disk
    assert [row["id"] for row in rs.records_for("2026-09-22", path=path)] == [fixed["id"]]
    assert len(rs.records_for("2026-09-22", path=path, include_superseded=True)) == 2
    with pytest.raises(rs.RecapError):  # a superseded row cannot be corrected twice
        rs.record_rule(session_date="2026-09-22", text="y", supersedes=first["id"], now=NOW, path=path)
    with pytest.raises(rs.RecapError):  # nor can a row of another kind or an unknown id
        rs.record_lesson(session_date="2026-09-22", keep="x", supersedes=fixed["id"], now=NOW, path=path)
    with pytest.raises(rs.RecapError):
        rs.record_lesson(session_date="2026-09-22", keep="x", supersedes="rc-nope", now=NOW, path=path)


def test_records_for_filters_by_session_and_kind(store):
    rs, path = store
    rs.record_rule(session_date="2026-09-21", text="a", now=NOW, path=path)
    rs.record_rule(session_date="2026-09-22", text="b", now=NOW, path=path)
    rs.record_lesson(session_date="2026-09-22", keep="c", now=NOW, path=path)
    assert [row["text"] for row in rs.records_for("2026-09-22", kinds=[rs.KIND_RULE], path=path)] == ["b"]
    assert {row["kind"] for row in rs.records_for("2026-09-22", path=path)} == {rs.KIND_RULE, rs.KIND_LESSON}


def test_a_torn_tail_costs_only_its_own_line(store):
    rs, path = store
    rs.record_rule(session_date="2026-09-22", text="a", now=NOW, path=path)
    with path.open("ab") as handle:
        handle.write(b'{"torn": ')
    rs.record_rule(session_date="2026-09-22", text="b", now=NOW, path=path)
    assert [row["text"] for row in rs.records_for("2026-09-22", path=path)] == ["a", "b"]


def test_latest_rule_before_is_the_newest_effective_rule_from_an_earlier_session(store):
    rs, path = store
    assert rs.latest_rule_before("2026-09-22", path=path) is None
    rs.record_rule(session_date="2026-09-18", text="old", now=NOW, path=path)
    friday = rs.record_rule(session_date="2026-09-21", text="wait for confirmation", now=NOW, path=path)
    rs.record_rule(session_date="2026-09-22", text="same day is not before", now=NOW, path=path)
    assert rs.latest_rule_before("2026-09-22", path=path)["id"] == friday["id"]


def test_rule_streak_counts_consecutive_kept_sessions_and_stops_at_a_miss_or_gap(store):
    rs, path = store
    assert rs.rule_streak("2026-09-22", path=path) == 0
    rs.record_rule_check(session_date="2026-09-15", answer="yes", now=NOW, path=path)
    rs.record_rule_check(session_date="2026-09-16", answer="no", now=NOW, path=path)
    rs.record_rule_check(session_date="2026-09-17", answer="yes", now=NOW, path=path)
    rs.record_rule_check(session_date="2026-09-18", answer="yes", now=NOW, path=path)
    # 2026-09-19/20 is a weekend, so Monday continues the run.
    monday = rs.record_rule_check(session_date="2026-09-21", answer="partly", now=NOW, path=path)
    assert rs.rule_streak("2026-09-18", path=path) == 2
    assert rs.rule_streak("2026-09-21", path=path) == 0
    rs.record_rule_check(
        session_date="2026-09-21", answer="yes", supersedes=monday["id"], now=NOW, path=path,
    )
    assert rs.rule_streak("2026-09-21", path=path) == 3
    # No check yet for Tuesday: the streak is read through the last checked session.
    assert rs.rule_streak("2026-09-22", path=path) == 3
    # A gap (an unchecked session) ends the run: nothing on 09-23, then 09-24 kept.
    rs.record_rule_check(session_date="2026-09-24", answer="yes", now=NOW, path=path)
    assert rs.rule_streak("2026-09-24", path=path) == 1


def test_a_failed_write_is_loud(store, monkeypatch):
    rs, path = store
    blocker = path.parent / "blocked"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    with pytest.raises(rs.RecapWriteError):
        rs.record_rule(session_date="2026-09-22", text="x", now=NOW, path=blocker / "recap.jsonl")


def test_a_card_answer_is_one_recap_row(store):
    rs, path = store
    row = rs.record_card_answer(
        session_date="2026-09-22", card_id="card-1", card_kind="miss",
        subject={"symbol": "amd", "side": "long"}, option="should_have_taken", text="clean pullback",
        now=NOW, path=path,
    )
    assert row["subject"] == {"symbol": "AMD", "side": "LONG"}
    assert row["option"] == "should_have_taken"
    assert rs.records_for("2026-09-22", kinds=[rs.KIND_CARD_ANSWER], path=path) == [row]


def test_a_mentor_card_answer_goes_through_the_mentors_own_writer_and_not_here(store):
    rs, path = store
    from mentor_questions import Subject

    calls: list[dict] = []

    class Journal:
        def record_opportunity_event(self, **kwargs):
            calls.append(kwargs)
            return {"event_id": "ev-1", **kwargs}

    subject = Subject(
        kind="trade_origin", subject_id="t1", options=("impulse",),
        detail={"trade_id": "t1", "symbol": "AMD"},
    )
    result = rs.record_card_answer(
        session_date="2026-09-22", card_id="card-2", card_kind="trade",
        subject={"trade_id": "t1"}, option="impulse", now=NOW, path=path,
        mentor_subject=subject, journal_store=Journal(),
    )
    assert result["routed_to"] == "mentor"
    assert calls and calls[0]["trade_id"] == "t1"
    assert calls[0]["payload"]["trade_origin"] == "impulse"
    assert not path.exists()


def test_a_failed_mentor_route_is_loud(store):
    rs, path = store
    from mentor_questions import Subject

    subject = Subject(kind="trade_origin", subject_id="t1", detail={"trade_id": "t1"})
    with pytest.raises(rs.RecapWriteError):
        rs.record_card_answer(
            session_date="2026-09-22", card_id="card-2", card_kind="trade",
            subject={"trade_id": "t1"}, option="impulse", now=NOW, path=path,
            mentor_subject=subject, journal_store=None,
        )


def test_the_default_path_is_the_project_paths_constant():
    import inspect

    import project_paths
    import recap_store

    assert project_paths.DAY_RECAP_EVENTS_FILE.name == "day_recap_events.jsonl"
    assert project_paths.DAY_RECAP_EVENTS_FILE.parent == project_paths.PERSISTENT_DATA_DIR
    assert "DAY_RECAP_EVENTS_FILE" in inspect.getsource(recap_store._default_path)


def test_a_clue_can_carry_a_context_snapshot_and_old_calls_still_work(store):
    rs, path = store
    bar_time = datetime(2026, 9, 22, 10, 5, tzinfo=ET)
    plain = rs.record_clue(
        session_date="2026-09-22", symbol="AMD", timeframe="M5", bar_time=bar_time,
        price=150.0, clue_tag="gap", now=NOW, path=path,
    )
    assert "context" not in plain
    snap = {
        "bar": {"open": 149.5, "high": 150.5, "low": 149.0, "close": 150.2, "volume": 1200.0},
        "spy_close": 512.3,
    }
    row = rs.record_clue(
        session_date="2026-09-22", symbol="AMD", timeframe="M5", bar_time=bar_time,
        price=150.0, clue_tag="volume_surge", context=snap, now=NOW, path=path,
    )
    assert row["context"] == snap
    assert _lines(path)[-1]["context"] == snap
    for bad in (["not", "a", "dict"], {"when": object()}):
        with pytest.raises(rs.RecapError):
            rs.record_clue(
                session_date="2026-09-22", symbol="AMD", timeframe="M5", bar_time=bar_time,
                price=150.0, clue_tag="gap", context=bad, now=NOW, path=path,
            )
    assert len(_lines(path)) == 2
