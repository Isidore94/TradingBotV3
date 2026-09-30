"""P3 veto challenges: the wording call and its checks, the challenges table, grading and the
scorecard, /vetoes, and the quiet 06:45 Inbox card."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import challenge  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs.citations import CitationRejected  # noqa: E402
from mentor_packs import veto_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = veto_pack.FIXTURE_NOW
AAA = "veto:2026-09-29:AAA:1"


def _good(**change):
    item = {"veto_id": AAA, "claim": "You passed on AAA, but compressed breakout vetoes like it won 34 of 40.",
            "evidence_refs": [AAA, f"{AAA}:slice"], "n": 40, "lb": 0.71}
    item.update(change)
    return item


@pytest.fixture()
def world(tmp_path):
    return veto_pack.write_fixture_world(tmp_path / "desk")


@pytest.fixture()
def pack(world):
    return veto_pack.build(now=NOW, paths=world)


# ---------------------------------------------------------------- the reply check
def test_a_good_challenge_is_kept_with_the_packs_numbers(pack):
    kept, drops = challenge.check_reply({"challenges": [_good()]}, pack)
    assert drops == [] and len(kept) == 1
    assert (kept[0]["symbol"], kept[0]["n"], kept[0]["lb"]) == ("AAA", 40, 0.71)


@pytest.mark.parametrize("change, reason", [
    ({"n": 41}, "n or lb differ from the pack"),
    ({"lb": 0.8}, "n or lb differ from the pack"),
    ({"veto_id": "veto:2026-09-29:BBB:1", "evidence_refs": ["veto:2026-09-29:BBB:1:slice"]}, "not a candidate veto"),
    ({"evidence_refs": [AAA]}, "does not cite its slice"),
    ({"evidence_refs": []}, "uncited"),
    ({"claim": ""}, "uncited"),
])
def test_an_unsupported_challenge_is_dropped(pack, change, reason):
    kept, drops = challenge.check_reply({"challenges": [_good(**change)]}, pack)
    assert kept == [] and drops[0]["reason"] == reason


def test_an_id_no_pack_carries_rejects_the_whole_reply(pack):
    with pytest.raises(CitationRejected):
        challenge.check_reply({"challenges": [_good(), _good(evidence_refs=[AAA, "veto:2026-09-29:ZZZ:1:slice"])]}, pack)
    with pytest.raises(CitationRejected):
        challenge.check_reply({"nope": []}, pack)


def test_one_veto_is_challenged_once(pack):
    kept, drops = challenge.check_reply({"challenges": [_good(), _good()]}, pack)
    assert len(kept) == 1 and drops[0]["reason"] == "a second challenge for one veto"


# ---------------------------------------------------------------- the call
def test_no_candidate_means_no_model_call(world):
    quiet = veto_pack.build("2026-09-28", now=NOW, paths=world)
    card = challenge.word(quiet, pack_hash="h", model="m", endpoint="http://h",
                          request=lambda **_: pytest.fail("no candidate: no model call"))
    assert card.done and card.challenges == []


def test_the_model_sees_only_the_candidates_and_the_numbers_to_copy(pack):
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return {"summary": {"challenges": [_good()]}}

    card = challenge.word(pack, pack_hash="h", model="gpt-oss:20b", endpoint="http://h", request=request)
    assert card.done and [item["veto_id"] for item in card.challenges] == [AAA]
    evidence = calls[0]["evidence"]
    assert [item["veto_id"] for item in evidence["candidates"]] == [AAA]
    assert evidence["candidates"][0]["n"] == 40 and evidence["candidates"][0]["lb"] == 0.71
    assert "BBB" not in json.dumps(evidence) and "CCC" not in json.dumps(evidence)
    assert calls[0]["schema"] == challenge.SCHEMA and calls[0]["endpoint"] == "http://h/v1"


def test_the_call_is_capped_at_600_tokens_through_request_ai_summary(pack):
    payloads = []

    class _Response:
        status_code = 200

        def __init__(self):
            body = {"id": "r", "choices": [{"message": {"content": json.dumps({"challenges": [_good()]})},
                                            "finish_reason": "stop"}]}
            self._body, self.text = body, json.dumps(body)

        def json(self):
            return self._body

    def post(url, **kwargs):
        payloads.append(kwargs["json"])
        return _Response()

    card = challenge.word(pack, pack_hash="h", model="gpt-oss:20b", endpoint="http://h", post=post)
    assert card.done, card.error
    assert payloads and all(int(p["max_tokens"]) <= 600 for p in payloads)


def test_a_failed_or_rejected_call_keeps_the_slices_and_says_so(pack):
    card = challenge.word(pack, pack_hash="h", model="m", endpoint="http://h",
                          request=lambda **_: (_ for _ in ()).throw(TimeoutError("slow")))
    assert not card.done and "TimeoutError" in card.error
    text = challenge.card_markdown(card)
    assert "not worded" in text and "n=40, wins 34 (85%), LB=0.71" in text
    rejected = challenge.word(pack, pack_hash="h", model="m", endpoint="http://h",
                              request=lambda **_: {"summary": {"challenges": [_good(evidence_refs=["made:up"])]}})
    assert "reply rejected" in rejected.error


# ---------------------------------------------------------------- the card
def test_the_card_lists_every_veto_with_its_slice_and_ends_with_the_note(pack):
    card = challenge.word(pack, pack_hash="h", model="m", endpoint="http://h",
                          request=lambda **_: {"summary": {"challenges": [_good()]}})
    text = challenge.card_markdown(card)
    for symbol in ("AAA", "BBB", "CCC", "DDD", "EEE"):
        assert f"[veto:2026-09-29:{symbol}:1]" in text and f"[veto:2026-09-29:{symbol}:1:slice]" in text
    assert "too few (n=10, floor 30)" in text
    assert "(n=40, LB=0.71)" in text and "**Challenges**" in text
    assert text.rstrip().endswith("rules go through the plan.*") and "a note, not a rule" in text
    assert challenge.inbox_line(card) == "Yesterday's vetoes: 1 challenge"


def test_a_card_with_no_challenge_says_why(world):
    lines = world.tier_outcomes.read_text(encoding="utf-8").splitlines()
    world.tier_outcomes.write_text("\n".join(line.replace(",True,False", ",False,False") if line.startswith("L") else line
                                             for line in lines) + "\n", encoding="utf-8")
    card = challenge.word(veto_pack.build(now=NOW, paths=world), pack_hash="h", model="m", endpoint="http://h",
                          request=lambda **_: pytest.fail("no candidate"))
    text = challenge.card_markdown(card)
    assert "**No challenge**: n too small (1 slice under 30) / no edge over baseline (2)." in text
    assert "a note, not a rule" in text


# ---------------------------------------------------------------- the table, grading, scorecard
def _worded(pack):
    return challenge.word(pack, pack_hash="h", model="m", endpoint="http://h",
                          request=lambda **_: {"summary": {"challenges": [_good()]}}, now=lambda: NOW)


def test_a_challenge_is_one_open_row_never_two(pack, tmp_path):
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    card = _worded(pack)
    assert challenge.record(store, card) == 1
    assert challenge.record(store, card) == 0, "a veto is challenged once"
    rows = store.challenges(kind="veto")
    assert len(rows) == 1 and rows[0]["graded_utc"] is None and rows[0]["symbol"] == "AAA"
    assert json.loads(rows[0]["evidence_ids_json"]) == [AAA, f"{AAA}:slice"]
    assert json.loads(rows[0]["outcome_json"])["session"] == "2026-09-29"


def _outcomes(path, horizons, trade_date="2026-09-29"):
    header = "trade_date,symbol,side,source,h1_date,h1_return,h3_date,h3_return,h5_date,h5_return,h10_date,h10_return"
    cells = []
    for horizon, date_text, value in (
        (1, "2026-09-30", "0.01"), (3, "2026-10-02", "0.02"), (5, "2026-10-06", "0.03"), (10, "2026-10-13", "-0.01"),
    ):
        cells += [date_text, value] if horizon in horizons else ["", ""]
    path.write_text(header + "\n" + ",".join([trade_date, "AAA", "LONG", "veto_v6_compressed", *cells]) + "\n",
                    encoding="utf-8")


def test_grading_fills_matured_horizons_and_closes_at_ten_sessions(pack, tmp_path):
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    challenge.record(store, _worded(pack))
    outcomes = tmp_path / "veto_cohort_outcomes.csv"
    assert challenge.grade_open(store, NOW, veto_outcomes=outcomes) == 1
    row = store.challenges(kind="veto")[0]
    assert row["graded_utc"] is None and json.loads(row["outcome_json"])["reason"] == "no veto cohort row yet"
    _outcomes(outcomes, {1, 3, 5})
    later = datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc)
    challenge.grade_open(store, later, veto_outcomes=outcomes)
    row = store.challenges(kind="veto")[0]
    outcome = json.loads(row["outcome_json"])
    assert row["graded_utc"] is None and outcome["status"] == "maturing"
    assert outcome["returns"] == {"1": 0.01, "3": 0.02, "5": 0.03} and outcome["hit"] is True
    _outcomes(outcomes, {1, 3, 5, 10})
    assert challenge.grade_open(store, later, veto_outcomes=outcomes) == 0, "the 10-session date is still ahead"
    done = datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc)
    assert challenge.grade_open(store, done, veto_outcomes=outcomes) == 1
    row = store.challenges(kind="veto")[0]
    assert row["graded_utc"] and json.loads(row["outcome_json"])["status"] == "graded"
    assert store.challenges(kind="veto", open_only=True) == []


def test_the_scorecard_says_too_few_under_thirty_and_a_rate_with_n_above(tmp_path):
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    assert "too few (n=0, floor 30)" in challenge.scorecard(store)
    for index in range(30):
        store.add_challenge(f"veto:2026-09-{index:02d}:X:1", kind="veto", symbol="X", claim="c",
                            outcome={"hit": index < 18})
    store.update_challenge("veto:2026-09-00:X:1", outcome={"hit": True}, graded_utc="2026-10-14T00:00:00+00:00")
    text = challenge.scorecard(store)
    assert "issued 30, fully graded 1" in text and "hit rate 60% (n=30, floor 30)" in text


def test_an_old_not_null_graded_column_is_rebuilt_when_empty(tmp_path):
    path = tmp_path / "chat.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE challenges (id TEXT PRIMARY KEY, kind TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '', "
                     "claim TEXT NOT NULL, evidence_ids_json TEXT NOT NULL DEFAULT '[]', issued_utc TEXT NOT NULL, "
                     "graded_utc TEXT NOT NULL DEFAULT '', outcome_json TEXT NOT NULL DEFAULT '{}')")
    store = MentorChatStore(path)
    assert store.add_challenge("veto:x:A:1", kind="veto", symbol="A", claim="c")
    assert store.challenges(open_only=True)[0]["graded_utc"] is None


# ---------------------------------------------------------------- the window
Qt = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture()
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(app, world, tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    gpu = {"reason": ""}
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: gpu["reason"])
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    calls: list[dict] = []

    def request(**kwargs):
        calls.append(kwargs)
        return {"summary": {"challenges": [_good()]}}

    clock = {"now": NOW}
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        stream_post=lambda url, payload, cancelled: [],
        post=lambda url, payload, timeout: {},
        now=lambda: clock["now"],
        mentor_enabled=False,
        liked_source=lambda: [],
        veto_builder=lambda day: veto_pack.build(day, now=clock["now"], paths=world),
        challenge_request=request,
        veto_outcomes=tmp_path / "veto_cohort_outcomes.csv",
    )
    win.calls, win.clock, win.gpu, win.world = calls, clock, gpu, world
    yield win
    win.shutdown()
    win.deleteLater()


def _drain(win, app):
    for _ in range(50):
        app.processEvents()
        if not win.queue.run_one():
            app.processEvents()
            if not win.queue.pending():
                break
    win._io.submit(lambda: None).result(5)


def _text(win):
    return win.transcript.toPlainText()


def test_vetoes_with_the_brain_off_show_the_slices_and_no_guess(window, app):
    window._brain_reason = "not connected"
    window.send("/vetoes")
    assert "building" in _text(window)
    _drain(window, app)
    text = _text(window)
    assert "Vetoes, session 2026-09-29" in text and "not worded (the brain is off: not connected)" in text
    assert "n=40, wins 34 (85%), LB=0.71" in text and window.calls == []
    assert window.store.challenges() == []


def test_vetoes_are_worded_recorded_and_then_answered_from_the_cache(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"
    window.send("/vetoes")
    _drain(window, app)
    assert len(window.calls) == 1 and "(n=40, LB=0.71)" in _text(window)
    assert [row["id"] for row in window.store.challenges(kind="veto")] == [AAA]
    window.send("/vetoes 2026-09-29")
    _drain(window, app)
    assert len(window.calls) == 1, "same session, same pack hash: the cached card answers"
    assert _text(window).count("Vetoes, session 2026-09-29") == 2


def test_a_session_without_candidates_never_calls_the_model(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "m"
    window.send("/vetoes 2026-09-28")
    _drain(window, app)
    assert window.calls == [] and "0 vetoes, 0 passes" in _text(window)


def test_the_0645_card_lands_once_in_the_inbox_after_quiet_hours(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "m"
    window.clock["now"] = datetime(2026, 9, 30, 6, 44, tzinfo=PT)
    window.maybe_veto_card()
    assert "veto_morning_pack" not in window.queue.pending(), "not before 06:45"
    blocks = list(window._blocks)
    window.clock["now"] = datetime(2026, 9, 30, 6, 45, tzinfo=PT)
    window.maybe_veto_card()
    jobs = {job.name: job for job in window.queue._jobs}
    assert jobs["veto_morning_pack"].priority == 1 and not jobs["veto_morning_pack"].needs_model
    _drain(window, app)
    word = [name for name in window.queue.ran if name == "veto_morning_word"]
    assert word == ["veto_morning_word"] and len(window.calls) == 1
    assert window.inbox.items() == [], "06:45 is inside the quiet hours around the open"
    window.clock["now"] = datetime(2026, 9, 30, 7, 0, tzinfo=PT)
    window.maybe_veto_card()
    window.maybe_veto_card()
    _drain(window, app)
    items = window.inbox.items()
    assert [item.text for item in items] == ["Yesterday's vetoes: 1 challenge"]
    assert window._blocks == blocks, "nothing pops: the transcript does not move"
    from PySide6.QtWidgets import QListWidgetItem

    row = QListWidgetItem("x")
    from PySide6.QtCore import Qt as QtCore

    row.setData(QtCore.ItemDataRole.UserRole, items[0].id)
    window._open_inbox_item(row)
    assert "(n=40, LB=0.71)" in _text(window) and "a note, not a rule" in _text(window)


def test_the_morning_card_respects_the_daily_cap(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "m"
    window.inbox.per_day_cap = 0
    window.clock["now"] = datetime(2026, 9, 30, 7, 5, tzinfo=PT)
    window.maybe_veto_card()
    _drain(window, app)
    window.maybe_veto_card()
    assert window.inbox.items() == [] and window._veto_inbox_waiting == []


def test_no_morning_card_on_a_weekend_or_after_a_session_without_vetoes(window, app):
    window.clock["now"] = datetime(2026, 10, 3, 7, 5, tzinfo=PT)  # Saturday
    window.maybe_veto_card()
    assert "veto_morning_pack" not in window.queue.pending()
    window.clock["now"] = datetime(2026, 9, 29, 7, 5, tzinfo=PT)  # Tuesday: Monday 09-28 had nothing
    window.maybe_veto_card()
    _drain(window, app)
    assert window.inbox.items() == [] and window.calls == []


def test_grading_is_queued_once_a_day_at_idle_priority_without_the_model(window, app):
    from mentor_app.prefetch import PRIORITY_IDLE

    window.clock["now"] = datetime(2026, 9, 30, 5, 0, tzinfo=PT)
    window.maybe_veto_card()
    job = next(job for job in window.queue._jobs if job.name == "grade_challenges")
    assert job.priority == PRIORITY_IDLE and not job.needs_model
    _drain(window, app)
    window.maybe_veto_card()
    assert "grade_challenges" not in window.queue.pending(), "once a day"
    window.clock["now"] += timedelta(days=1)
    window.maybe_veto_card()
    assert "grade_challenges" in window.queue.pending()


def test_scorecard_command_prints_counts_with_n(window, app):
    window.send("/scorecard")
    _drain(window, app)
    assert "issued 0, fully graded 0" in _text(window) and "too few (n=0, floor 30)" in _text(window)


# ---------------------------------------------------------------- review fixes (87f06e34)
def test_an_after_close_veto_grades_on_its_session_date_not_its_decision_session(world, tmp_path):
    # HLIT repro: vetoed Friday 21:04 PT, so session_date is Saturday while the judged session is Friday.
    # The veto cohort keys trade_date = session_date; grading must use it.
    lines = world.annotations.read_text(encoding="utf-8").splitlines()
    fixed = []
    for line in lines:
        if '"event_id": "aaa-click"' in line:
            row = json.loads(line)
            row["session_date"] = "2026-09-30"
            line = json.dumps(row)
        fixed.append(line)
    world.annotations.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    pack = veto_pack.build(now=NOW, paths=world)
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    challenge.record(store, _worded(pack))
    seed = json.loads(store.challenges(kind="veto")[0]["outcome_json"])
    assert seed["session"] == "2026-09-29" and seed["session_date"] == "2026-09-30"
    outcomes = tmp_path / "veto_cohort_outcomes.csv"
    _outcomes(outcomes, {1, 3, 5}, trade_date="2026-09-30")
    challenge.grade_open(store, datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc), veto_outcomes=outcomes)
    outcome = json.loads(store.challenges(kind="veto")[0]["outcome_json"])
    assert outcome.get("reason") != "no veto cohort row yet" and outcome["returns"]["5"] == 0.03


def test_an_old_challenge_without_session_date_falls_back_to_its_session(tmp_path):
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    store.add_challenge(AAA, kind="veto", symbol="AAA", claim="c", outcome={"session": "2026-09-29", "side": "LONG"})
    outcomes = tmp_path / "veto_cohort_outcomes.csv"
    _outcomes(outcomes, {1})
    challenge.grade_open(store, datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc), veto_outcomes=outcomes)
    assert json.loads(store.challenges(kind="veto")[0]["outcome_json"])["returns"] == {"1": 0.01}


def test_an_old_not_null_graded_column_with_rows_is_migrated_and_keeps_them(tmp_path):
    path = tmp_path / "chat.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE challenges (id TEXT PRIMARY KEY, kind TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '', "
                     "claim TEXT NOT NULL, evidence_ids_json TEXT NOT NULL DEFAULT '[]', issued_utc TEXT NOT NULL, "
                     "graded_utc TEXT NOT NULL DEFAULT '', outcome_json TEXT NOT NULL DEFAULT '{}')")
        conn.execute("INSERT INTO challenges (id, kind, symbol, claim, issued_utc) VALUES ('old:1', 'veto', 'A', 'c', 't')")
        conn.execute("INSERT INTO challenges (id, kind, symbol, claim, issued_utc, graded_utc) "
                     "VALUES ('old:2', 'veto', 'B', 'c', 't', '2026-10-01T00:00:00+00:00')")
    store = MentorChatStore(path)
    assert store.add_challenge("veto:x:C:1", kind="veto", symbol="C", claim="c"), "a new row writes after the migration"
    rows = {row["id"]: row for row in store.challenges()}
    assert set(rows) == {"old:1", "old:2", "veto:x:C:1"}
    assert rows["old:1"]["graded_utc"] is None and rows["old:2"]["graded_utc"] == "2026-10-01T00:00:00+00:00"
    assert {row["id"] for row in store.challenges(open_only=True)} == {"old:1", "veto:x:C:1"}
    assert not _has_table(path, "challenges_old")


def _old_table(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE challenges (id TEXT PRIMARY KEY, kind TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '', "
                     "claim TEXT NOT NULL, evidence_ids_json TEXT NOT NULL DEFAULT '[]', issued_utc TEXT NOT NULL, "
                     "graded_utc TEXT NOT NULL DEFAULT '', outcome_json TEXT NOT NULL DEFAULT '{}')")
        conn.execute("INSERT INTO challenges (id, kind, symbol, claim, issued_utc) VALUES ('old:1', 'veto', 'A', 'c', 't')")
    conn.close()


def _has_table(path, name):
    conn = sqlite3.connect(path)
    try:
        return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone())
    finally:
        conn.close()


def test_a_migration_that_fails_midway_rolls_back_whole_and_runs_again(tmp_path, monkeypatch):
    from mentor_app import store as store_module

    path = tmp_path / "chat.sqlite3"
    _old_table(path)
    real = store_module._copy_old_challenges
    monkeypatch.setattr(store_module, "_copy_old_challenges", lambda conn: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    assert MentorChatStore(path).challenges() == [], "the failed open reads nothing, loudly in the log"
    assert not _has_table(path, "challenges_old"), "the rename rolled back with the failed copy"
    monkeypatch.setattr(store_module, "_copy_old_challenges", real)
    reopened = MentorChatStore(path)
    assert [row["id"] for row in reopened.challenges()] == ["old:1"]
    assert reopened.challenges()[0]["graded_utc"] is None and not _has_table(path, "challenges_old")


def test_a_leftover_challenges_old_is_finished_on_open(tmp_path):
    path = tmp_path / "chat.sqlite3"
    _old_table(path)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE challenges RENAME TO challenges_old")  # the state an interrupted older migration left
    conn.commit()
    conn.close()
    store = MentorChatStore(path)
    assert [row["id"] for row in store.challenges()] == ["old:1"]
    assert not _has_table(path, "challenges_old")
    assert store.add_challenge("veto:x:C:1", kind="veto", symbol="C", claim="c")


def test_a_restart_after_the_morning_card_posted_never_posts_it_again(window, app, tmp_path):
    window._brain_ok, window._endpoint, window._model = True, "http://h", "m"
    window.clock["now"] = datetime(2026, 9, 30, 7, 5, tzinfo=PT)
    window.maybe_veto_card()
    _drain(window, app)
    assert [item.text for item in window.inbox.items()] == ["Yesterday's vetoes: 1 challenge"]
    assert window.store.get_state("veto_morning_posted") == "2026-09-29"
    # A restarted app: fresh schedule and inbox, same chat store.
    from mentor_app.challenge import VetoSchedule
    from mentor_app.inbox import Inbox

    window._veto_schedule = VetoSchedule()
    window.inbox = Inbox(now=window._now)
    window.clock["now"] = datetime(2026, 9, 30, 8, 0, tzinfo=PT)
    window.maybe_veto_card()
    _drain(window, app)
    assert window.inbox.items() == [] and len(window.calls) == 1, "posted for that session already"


def test_a_card_held_past_midnight_is_dropped_not_posted_the_next_day(window, app):
    from datetime import timedelta as _td

    window._brain_ok, window._endpoint, window._model = True, "http://h", "m"
    window.inbox.mute(_td(hours=12))
    window.clock["now"] = datetime(2026, 9, 30, 7, 5, tzinfo=PT)
    window.maybe_veto_card()
    _drain(window, app)
    assert window.inbox.items() == [] and len(window._veto_inbox_waiting) == 1, "muted: held"
    window.inbox.muted_until = None
    window.clock["now"] = datetime(2026, 10, 1, 0, 5, tzinfo=PT)
    window._deliver_veto_inbox()
    assert window.inbox.items() == [] and window._veto_inbox_waiting == [], "yesterday's card is dropped"
    assert window.store.get_state("veto_morning_posted") in (None, "")

