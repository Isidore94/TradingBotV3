"""Mentor app P9: /mirror (one cited narration per pack hash; brain down = the pack), the Monday Inbox
card (once a week, never reposted after a restart, never pops), the tilt watch (observations only, at
most one Inbox item per 30 min, never moves the transcript), day-end tilt grading and /tilt."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import challenge, commands, mirror, settings, tilt_watch  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import mirror_pack, tilt_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
MON_0650 = datetime(2026, 9, 28, 6, 50, tzinfo=PT)


def _reply(observations=None, questions=None):
    return {"summary": {
        "observations": observations if observations is not None else [
            {"text": "Liked longs won 75% at 5 sessions (n=36).", "evidence_refs": ["mirror:liked:LONG:5"]}],
        "questions": questions if questions is not None else [
            {"text": "What made the vetoed compressed longs look wrong?", "evidence_refs": ["mirror:veto:compressed:LONG"]}],
    }, "model": "m"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return mirror_pack.write_fixture_world(tmp_path_factory.mktemp("mirror"))


@pytest.fixture
def pack(world):
    return mirror_pack.build(now=mirror_pack.FIXTURE_NOW, paths=world)


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


# ---------------------------------------------------------------- commands
def test_mirror_and_tilt_commands_parse_and_are_in_help():
    assert (commands.handle("/mirror").action, commands.handle("/mirror").arg) == ("mirror", 6)
    assert commands.handle("/mirror 8").arg == 8
    assert commands.handle("/mirror 99").action == "error" and commands.handle("/mirror x").action == "error"
    assert commands.handle("/tilt").action == "tilt" and commands.handle("/tilt now").action == "error"
    assert "/mirror" in commands.HELP_TEXT and "/tilt" in commands.HELP_TEXT


# ---------------------------------------------------------------- narration
def test_narration_keeps_cited_lines_and_drops_uncited(pack):
    card = mirror.narrate(pack, pack_hash="h", model="m", endpoint="http://x", request=lambda **_: _reply(
        questions=[{"text": "Why?", "evidence_refs": []}]))
    assert card.narrated and len(card.observations) == 1 and card.questions == []
    assert [d["kind"] for d in card.dropped] == ["questions"]
    text = mirror.card_markdown(pack, card)
    assert "[mirror:liked:LONG:5]" in text and "1 uncited line dropped" in text
    assert text.rstrip().endswith(mirror.FOOTER) and "rules live in trading_plan.md" in mirror.FOOTER


def test_a_foreign_id_rejects_the_whole_reply(pack):
    card = mirror.narrate(pack, pack_hash="h", model="m", endpoint="http://x", request=lambda **_: _reply(
        observations=[{"text": "NVDA is great.", "evidence_refs": ["pick:NVDA:cell"]}]))
    assert not card.narrated and "rejected" in card.error and card.observations == []


def test_the_call_is_capped_at_600_tokens_with_the_mirror_schema(pack):
    seen, posted = {}, {}

    def request(**kwargs):
        seen.update(kwargs)
        kwargs["post"]("u", json={"max_tokens": 4000})
        return _reply()

    mirror.narrate(pack, pack_hash="h", model="gemma3:12b", endpoint="http://x/", request=request,
                   post=lambda url, **kw: posted.update(kw["json"]))
    assert posted["max_tokens"] == 600 and seen["schema"] is mirror.SCHEMA and seen["endpoint"] == "http://x/v1"
    assert set(mirror.SCHEMA["properties"]) == {"observations", "questions"}
    assert "never propose a rule" in seen["evidence"]["task"].lower()


def test_brain_down_is_the_pack_and_too_few_cuts_are_counted_not_read(pack):
    text = mirror.card_markdown(pack, None, brain_reason="the brain is off")
    assert "No narration: the brain is off" in text and "[mirror:liked:LONG:5]" in text
    assert "too few to read (n under 30)" in text and "[mirror:liked:SHORT:5] n=3" in text
    assert "*Caveats* [mirror:caveats]" in text and "3 min" in text


def test_an_unchanged_pack_is_narrated_once_and_cached_by_hash(store, pack):
    calls = []

    def narrate(p, digest):
        calls.append(digest)
        return mirror.narrate(p, pack_hash=digest, model="m", endpoint="http://x", request=lambda **_: _reply())

    first = mirror.run_mirror_job(store=store, build_pack=lambda: pack, narrate=narrate)
    again = mirror.run_mirror_job(store=store, build_pack=lambda: pack, narrate=narrate)
    assert len(calls) == 1 and first["narrated"] and not again["narrated"]
    assert again["card"].observations == first["card"].observations
    assert store.get_pack(mirror.CACHE_NAME, {"hash": first["hash"]}) is not None


def test_a_failed_narration_is_not_cached(store, pack):
    out = mirror.run_mirror_job(store=store, build_pack=lambda: pack,
                                narrate=lambda p, d: mirror.MirrorCard(pack_hash=d, error="down"))
    assert out["narrated"] and mirror.cached_card(store, out["hash"]) is None


# ---------------------------------------------------------------- the weekly schedule
@pytest.mark.parametrize(("local", "due"), [
    (datetime(2026, 9, 28, 6, 49, tzinfo=PT), False),
    (datetime(2026, 9, 28, 6, 50, tzinfo=PT), True),
    (datetime(2026, 9, 29, 9, 0, tzinfo=PT), True),   # first session day the app sees this week
    (datetime(2026, 10, 3, 9, 0, tzinfo=PT), False),  # Saturday
])
def test_weekly_card_is_due_from_0650_on_a_session_day(local, due):
    assert mirror.WeeklySchedule().due(local) is due


def test_weekly_card_is_due_once_a_week():
    schedule = mirror.WeeklySchedule()
    schedule.mark(MON_0650)
    assert not schedule.due(MON_0650 + timedelta(days=2))
    assert schedule.due(MON_0650 + timedelta(days=7))
    assert mirror.week_key(MON_0650) == "2026-W40"


# ---------------------------------------------------------------- tilt watch (Qt-free)
@pytest.fixture
def journal(tmp_path):
    return tilt_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")


def test_each_new_observation_is_one_challenge_row_with_its_leg_ids(store, journal):
    now = tilt_pack.FIXTURE_NOW
    out = tilt_watch.run_watch(store, now, build=lambda: tilt_pack.build(now=now, journal=journal))
    assert [row["kind"] for row in out["new"]] == ["reentry", "size", "burst", "streak"]
    rows = {row["id"]: row for row in store.challenges(kind="tilt")}
    burst = rows["tilt:2026-09-29:burst:094000"]
    assert json.loads(burst["evidence_ids_json"]) == ["tilt:burst:094000", "leg:2", "leg:3", "leg:4", "leg:5", "leg:7"]
    assert json.loads(burst["outcome_json"])["status"] == "open"
    again = tilt_watch.run_watch(store, now, build=lambda: tilt_pack.build(now=now, journal=journal))
    assert again["new"] == [] and len(store.challenges(kind="tilt")) == 4, "seen once, never repeated"


def test_an_unchanged_journal_is_not_rebuilt(store, journal):
    now = tilt_pack.FIXTURE_NOW
    from mentor_packs import journal_read

    builds = []

    def build():
        builds.append(1)
        return tilt_pack.build(now=now, journal=journal)

    sig = lambda: journal_read.leg_signature(journal, now.date().isoformat())  # noqa: E731
    tilt_watch.run_watch(store, now, build=build, signature=sig)
    assert tilt_watch.run_watch(store, now, build=build, signature=sig)["skipped"] and len(builds) == 1


def test_one_item_per_30_minutes():
    now = datetime(2026, 9, 29, 7, 30, tzinfo=PT)
    assert tilt_watch.may_post(None, now)
    assert not tilt_watch.may_post((now - timedelta(minutes=29)).isoformat(), now)
    assert tilt_watch.may_post((now - timedelta(minutes=30)).isoformat(), now)


@pytest.mark.parametrize(("local", "due"), [
    (datetime(2026, 9, 29, 6, 29, tzinfo=PT), False), (datetime(2026, 9, 29, 6, 30, tzinfo=PT), True),
    (datetime(2026, 9, 29, 12, 59, tzinfo=PT), True), (datetime(2026, 9, 29, 13, 0, tzinfo=PT), False),
    (datetime(2026, 10, 3, 9, 0, tzinfo=PT), False),
])
def test_the_watch_runs_in_the_session_on_weekdays(local, due):
    assert tilt_watch.TiltSchedule().due(local) is due


def test_day_end_grading_hit_is_a_red_rest_of_day_and_the_scorecard_shows_it(store, journal):
    now = tilt_pack.FIXTURE_NOW
    tilt_watch.run_watch(store, now, build=lambda: tilt_pack.build(now=now, journal=journal))
    assert tilt_watch.grade_open(store, datetime(2026, 9, 29, 16, 0, tzinfo=ET), journal=journal) == 0, \
        "the day is not over"
    assert tilt_watch.grade_open(store, datetime(2026, 9, 29, 16, 20, tzinfo=ET), journal=journal) == 4
    rows = {row["id"]: json.loads(row["outcome_json"]) for row in store.challenges(kind="tilt")}
    burst = rows["tilt:2026-09-29:burst:094000"]  # after 09:49: -30, -20, +50 = 0, not red
    assert (burst["before_pnl"], burst["rest_pnl"], burst["hit"]) == (-120.0, 0.0, False)
    reentry = rows["tilt:2026-09-29:reentry:AMD:094500"]  # after 09:45: -30, -20, +50 = 0
    assert reentry["status"] == "graded" and reentry["hit"] is False
    text = challenge.scorecard(store, floor=30)
    assert "**Scorecard: tilt challenges**" in text and "issued 4, fully graded 4" in text
    assert "red rest of day after the observation" in text and "too few (n=4, floor 30)" in text


def test_challenge_grade_open_grades_tilt_too(store, journal, tmp_path):
    now = tilt_pack.FIXTURE_NOW
    tilt_watch.run_watch(store, now, build=lambda: tilt_pack.build(now=now, journal=journal))
    updated = challenge.grade_open(store, datetime(2026, 9, 30, 8, 0, tzinfo=PT), veto_outcomes=tmp_path / "none.csv",
                                   journal=journal)
    assert updated == 4 and not store.challenges(kind="tilt", open_only=True)


def test_tilt_code_never_touches_a_model_or_an_order():
    for name in ("mentor_app/tilt_watch.py", "mentor_packs/tilt_pack.py"):
        source = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert "PySide6" not in source and "request_ai_summary" not in source and "place_order" not in source, name


# ---------------------------------------------------------------- the window
@pytest.fixture
def win(tmp_path, monkeypatch, world, journal):
    from PySide6.QtWidgets import QApplication

    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    clock = {"now": MON_0650}
    calls: list = []

    def request(**kwargs):
        calls.append(kwargs)
        return _reply()

    window = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(), news_queue=PrefetchQueue(),
        inbox=Inbox(now=lambda: clock["now"]),
        stream_post=lambda *a, **k: [], post=lambda *a, **k: {}, now=lambda: clock["now"], mentor_enabled=False,
        mirror_builder=lambda weeks: mirror_pack.build(weeks, now=mirror_pack.FIXTURE_NOW, paths=world),
        mirror_request=request,
        tilt_builder=lambda: tilt_pack.build(now=clock["now"], journal=journal), tilt_journal=journal,
    )
    window.clock, window.calls = clock, calls
    yield window
    window.shutdown()
    window.deleteLater()


def _drain(window):
    while window.queue.run_one() or window.news_queue.run_one():
        pass
    window._io.submit(lambda: None).result(5)


def _up(window):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "gemma3:12b"


def test_mirror_with_the_brain_off_is_the_pack_and_no_model_call(win):
    win._brain_reason = "the night AI owns the GPU"
    win.send("/mirror")
    _drain(win)
    text = win.transcript.toPlainText()
    assert "Mirror (weeks=6)" in text and "[mirror:liked:LONG:5]" in text and win.calls == []
    assert "night AI owns the GPU" in text and "rules live in trading_plan.md" in text


def test_mirror_narrates_once_per_pack_hash(win):
    _up(win)
    win.send("/mirror")
    _drain(win)
    win.send("/mirror")
    _drain(win)
    assert len(win.calls) == 1 and win.transcript.toPlainText().count("Liked longs won 75%") == 2
    stored = [row for row in win.store.turns() if row["role"] == "assistant"]
    assert stored and '"mirror_pack"' in stored[-1]["tool_calls_json"]


def test_the_monday_card_lands_once_in_the_inbox_after_quiet_hours_and_never_pops(win):
    before = win.transcript.toPlainText()
    win.maybe_mirror_week()
    _drain(win)
    assert win.inbox.items() == [], "06:50 is inside the quiet hours: held, not posted"
    win.clock["now"] = MON_0650 + timedelta(minutes=11)
    win.maybe_mirror_week()
    _drain(win)
    items = win.inbox.items()
    assert [item.text for item in items] == ["Your week in the mirror"]
    assert win.transcript.toPlainText() == before, "the transcript never moves on its own"
    assert "Your week in the mirror" in win._inbox_cards[items[0].id]
    assert win.store.get_state(mirror.POSTED_KEY) == "2026-W40"
    win.clock["now"] += timedelta(days=1)
    win.maybe_mirror_week()
    _drain(win)
    assert len(win.inbox.items()) == 1, "once a week"


def test_a_restart_never_reposts_the_weekly_card(win):
    win.store.set_state(mirror.POSTED_KEY, "2026-W40")
    win.clock["now"] = MON_0650 + timedelta(minutes=15)
    win.maybe_mirror_week()
    _drain(win)
    win.maybe_mirror_week()
    _drain(win)
    assert win.inbox.items() == []


def test_a_tilt_observation_is_one_inbox_item_with_leg_ids_and_no_pop(win):
    win.clock["now"] = tilt_pack.FIXTURE_NOW.astimezone(PT)
    before = win.transcript.toPlainText()
    win.maybe_watch_tilt()
    _drain(win)
    items = win.inbox.items()
    assert len(items) == 1 and items[0].kind == "tilt"
    assert "legs 2, 3" in items[0].text and "Observation, not a rule." in items[0].text and "(+3 more: /tilt)" in items[0].text
    assert win.transcript.toPlainText() == before
    assert len(win.store.challenges(kind="tilt")) == 4


def _at_et(hour, minute):
    return datetime(2026, 9, 29, hour, minute, tzinfo=ET).astimezone(PT)


def test_quiet_hours_hold_the_tilt_item_until_they_end(win):
    win.clock["now"] = _at_et(9, 48)  # 06:48 PT: the open's quiet hours
    win.maybe_watch_tilt()
    _drain(win)
    assert win.inbox.items() == [] and len(win.store.challenges(kind="tilt")) == 2
    win.clock["now"] = _at_et(10, 0)
    win.maybe_watch_tilt()
    _drain(win)
    items = win.inbox.items()
    assert len(items) == 1 and "Re-opened AMD" in items[0].text


def test_at_most_one_tilt_item_per_30_minutes(win, journal):
    import sqlite3

    win.clock["now"] = _at_et(10, 8)
    win.maybe_watch_tilt()
    _drain(win)
    assert len(win.inbox.items()) == 1 and len(win.store.challenges(kind="tilt")) == 4
    conn = sqlite3.connect(journal)
    conn.executescript(
        "INSERT INTO trades VALUES ('T6', 'M1', 'MSFT', 'STK', 'LONG', 'CLOSED', '2026-09-29T10:21:00-04:00', "
        "'2026-09-29T10:25:00-04:00', 10, -40.0, -40.0);"
        "INSERT INTO trade_legs VALUES (60, 'T6', 'BUY', 'OPEN', 10, 400.0, '2026-09-29T10:21:00-04:00');"
        "INSERT INTO trade_legs VALUES (61, 'T6', 'SELL', 'CLOSE', 10, 400.0, '2026-09-29T10:25:00-04:00');"
        "INSERT INTO trades VALUES ('T7', 'M1', 'MSFT', 'STK', 'LONG', 'OPEN', '2026-09-29T10:27:00-04:00', '', 10, "
        "NULL, NULL);"
        "INSERT INTO trade_legs VALUES (62, 'T7', 'BUY', 'OPEN', 10, 400.0, '2026-09-29T10:27:00-04:00');"
    )
    conn.commit()
    conn.close()
    win.clock["now"] = _at_et(10, 30)
    win.maybe_watch_tilt()
    _drain(win)
    assert len(win.inbox.items()) == 1, "22 min after the first item: held, not posted"
    assert len(win.store.challenges(kind="tilt")) == 5
    win.clock["now"] = _at_et(10, 38)
    win.maybe_watch_tilt()
    _drain(win)
    assert len(win.inbox.items()) == 2 and "Re-opened MSFT" in win.inbox.items()[-1].text


def test_the_tilt_watch_runs_while_ai_is_paused_and_uses_no_model(win):
    win._paused_until = datetime(2026, 9, 30, 6, 0, tzinfo=PT)
    win.clock["now"] = tilt_pack.FIXTURE_NOW.astimezone(PT)
    win.maybe_watch_tilt()
    _drain(win)
    assert len(win.inbox.items()) == 1 and win.calls == []


def test_tilt_command_shows_today_and_the_base_rates(win):
    win.clock["now"] = tilt_pack.FIXTURE_NOW.astimezone(PT)
    win.send("/tilt")
    _drain(win)
    text = win.transcript.toPlainText()
    assert "Tilt watch: today" in text and "[tilt:burst:094000]" in text
    assert "too few (n=2, floor 30)" in text and "Observations, not rules" in text


def test_no_trades_after_the_observation_is_not_a_miss(store, journal):
    import sqlite3

    now = tilt_pack.FIXTURE_NOW
    tilt_watch.run_watch(store, now, build=lambda: tilt_pack.build(now=now, journal=journal))
    conn = sqlite3.connect(journal)
    conn.execute("DELETE FROM trades WHERE trade_id = 'T5'")
    conn.commit()
    conn.close()
    tilt_watch.grade_open(store, datetime(2026, 9, 29, 16, 20, tzinfo=ET), journal=journal)
    rows = {row["id"]: json.loads(row["outcome_json"]) for row in store.challenges(kind="tilt")}
    streak = rows["tilt:2026-09-29:streak:094000"]  # the last close was the streak's own third loss
    assert streak["result"] == "no trades after" and "hit" not in streak and streak["status"] == "graded"
    assert "hit" in rows["tilt:2026-09-29:burst:094000"], "the burst had closes after it"
    assert "(n=3, floor 30)" in challenge.scorecard(store, floor=30).split("tilt challenges")[1]
