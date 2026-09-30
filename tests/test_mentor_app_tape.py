"""Mentor app P5: /tape narration (cited, cached by pack hash), the 30-min session refresh, and the
optional 06:30 phone line (off by default, pack facts only, at most once per PT day)."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import brief_push, settings, tape  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import regime_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
TUE_0631 = datetime(2026, 9, 29, 6, 31, tzinfo=PT)


def _reply(read=None, watch=None):
    return {"summary": {"read": read if read is not None else [{"text": "SPY is weak.", "evidence_refs": ["tape:d1env"]}],
                        "watch": watch if watch is not None else [{"text": "ISM at 10.", "evidence_refs": ["tape:econ:t1"]}]},
            "model": "m"}


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


# ---------------------------------------------------------------- narration
def test_narration_keeps_cited_lines_and_drops_uncited():
    pack = regime_pack.fixture()
    card = tape.narrate(pack, pack_hash="h", model="m", endpoint="http://x", request=lambda **_: _reply(
        read=[{"text": "SPY is weak.", "evidence_refs": ["tape:d1env"]}, {"text": "Feels heavy.", "evidence_refs": []}]))
    assert card.narrated and [row["text"] for row in card.read] == ["SPY is weak."]
    assert [drop["kind"] for drop in card.dropped] == ["read"]
    text = tape.card_markdown(pack, card)
    assert "[tape:d1env]" in text and "1 uncited line dropped" in text and "[tape:asof]" in text


def test_a_foreign_id_rejects_the_whole_reply():
    card = tape.narrate(regime_pack.fixture(), pack_hash="h", model="m", endpoint="http://x", request=lambda **_: _reply(
        watch=[{"text": "NVDA runs.", "evidence_refs": ["pick:NVDA:cell"]}]))
    assert not card.narrated and "rejected" in card.error and card.read == []


def test_the_call_is_capped_at_600_tokens_with_the_tape_schema():
    seen = {}

    def request(**kwargs):
        seen.update(kwargs)
        kwargs["post"]("u", json={"max_tokens": 4000})
        return _reply()

    posted = {}
    tape.narrate(regime_pack.fixture(), pack_hash="h", model="gemma3:12b", endpoint="http://x/", request=request,
                 post=lambda url, **kw: posted.update(kw["json"]))
    assert posted["max_tokens"] == 600 and seen["schema"] is tape.SCHEMA and seen["endpoint"] == "http://x/v1"
    assert set(tape.SCHEMA["properties"]) == {"read", "watch"}


def test_brain_down_shows_the_pack_and_no_guess():
    pack = regime_pack.fixture()
    text = tape.card_markdown(pack, None, brain_reason="the night AI owns the GPU")
    assert text.startswith("**Tape**") and "[tape:mode]" in text and "the night AI owns the GPU" in text
    assert "the read" not in text


# ---------------------------------------------------------------- cache by hash
def test_the_hash_ignores_the_clock_but_not_the_facts():
    early = regime_pack.fixture()
    later = regime_pack.build(now=datetime(2026, 9, 29, 14, 30, tzinfo=timezone.utc), sources=regime_pack.fixture_sources())
    assert tape.pack_hash(early) == tape.pack_hash(later)
    moved = regime_pack.build(now=datetime(2026, 9, 29, 14, 30, tzinfo=timezone.utc),
                              sources=replace(regime_pack.fixture_sources(), d1_env=lambda day: "bullish_trend"))
    assert tape.pack_hash(moved) != tape.pack_hash(early)


def test_an_unchanged_pack_is_narrated_once(store):
    calls = []

    def narrate(pack, digest):
        calls.append(digest)
        return tape.narrate(pack, pack_hash=digest, model="m", endpoint="http://x", request=lambda **_: _reply())

    first = tape.run_tape_job(store=store, build_pack=regime_pack.fixture, narrate=narrate)
    again = tape.run_tape_job(store=store, build_pack=regime_pack.fixture, narrate=narrate, last_hash=first["hash"])
    assert len(calls) == 1 and first["narrated"] and not again["narrated"] and not again["changed"]
    assert again["card"].read == first["card"].read


def test_a_failed_narration_is_not_cached(store):
    bad = tape.run_tape_job(store=store, build_pack=regime_pack.fixture, narrate=lambda pack, digest: tape.TapeCard(
        pack_hash=digest, error="down"))
    assert bad["narrated"] and tape.cached_card(store, bad["hash"]) is None


def test_the_job_yields_to_a_chat_turn_on_one_slot(store):
    out = tape.run_tape_job(store=store, build_pack=regime_pack.fixture,
                            narrate=lambda *_: pytest.fail("no model while a turn waits"), should_yield=lambda: True)
    assert out["yielded"] and out["card"] is None


@pytest.mark.parametrize(("local", "due"), [
    (datetime(2026, 9, 29, 5, 59, tzinfo=PT), False),
    (datetime(2026, 9, 29, 6, 0, tzinfo=PT), True),
    (datetime(2026, 9, 29, 12, 59, tzinfo=PT), True),
    (datetime(2026, 9, 29, 13, 0, tzinfo=PT), False),
    (datetime(2026, 10, 3, 9, 0, tzinfo=PT), False),  # Saturday
])
def test_the_refresh_runs_in_the_session_only(local, due):
    assert tape.TapeSchedule().due(local) is due


def test_the_refresh_is_every_30_minutes():
    schedule = tape.TapeSchedule()
    at = datetime(2026, 9, 29, 7, 0, tzinfo=PT)
    schedule.mark(at)
    assert not schedule.due(at + timedelta(minutes=29))
    assert schedule.due(at + timedelta(minutes=30))


# ---------------------------------------------------------------- the phone line
def test_with_the_setting_off_nothing_is_sent(store, monkeypatch):
    monkeypatch.setattr(settings, "_setting", lambda key, default=None: default)
    assert settings.push_brief_enabled() is False, "off by default"
    sent = []
    assert brief_push.maybe_send(store, regime_pack.fixture(), now=TUE_0631,
                                 send=lambda *a, **k: sent.append(a) or {"ok": True}) == ""
    assert sent == [] and store.get_state(brief_push.STATE_KEY) is None


def test_with_the_setting_on_exactly_one_line_a_day(store, monkeypatch):
    monkeypatch.setattr(settings, "_setting", lambda key, default=None: True if key == "mentor_push_brief" else default)
    sent = []

    def send(title, message, **kwargs):
        sent.append((title, message))
        return {"ok": True, "kind": "delivered"}

    pack = regime_pack.fixture()
    assert brief_push.maybe_send(store, pack, now=TUE_0631 - timedelta(minutes=2), send=send) == "", "not before 06:30"
    line = brief_push.maybe_send(store, pack, now=TUE_0631, send=send)
    for minutes in (1, 15, 30):
        brief_push.maybe_send(store, pack, now=TUE_0631 + timedelta(minutes=minutes), send=send)
    assert sent == [(brief_push.TITLE, line)] and line == regime_pack.push_line(pack) and len(line) <= 200
    assert store.get_state(brief_push.STATE_KEY) == "2026-09-29"
    brief_push.maybe_send(store, pack, now=TUE_0631 + timedelta(days=1), send=send)
    assert len(sent) == 2, "the next PT day gets its own line"


def test_the_line_never_carries_model_text(store):
    sent = []
    card_text = "SPY looks like it wants to break down hard"
    pack = regime_pack.fixture()
    brief_push.maybe_send(store, pack, now=TUE_0631, enabled=True, send=lambda t, m, **k: sent.append(m) or {"ok": True})
    assert card_text not in sent[0] and "night read" not in sent[0].lower()
    assert "stayed below" not in sent[0], "the night read is model text"


def test_an_unmarkable_day_sends_nothing(store, monkeypatch):
    monkeypatch.setattr(store, "set_state", lambda key, value: False)
    sent = []
    assert brief_push.maybe_send(store, regime_pack.fixture(), now=TUE_0631, enabled=True,
                                 send=lambda *a, **k: sent.append(a) or {"ok": True}) == ""
    assert sent == []


# ---------------------------------------------------------------- the window
@pytest.fixture
def win(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    clock = {"now": datetime(2026, 9, 29, 7, 0, tzinfo=PT)}
    calls: list = []
    sent: list = []

    def request(**kwargs):
        calls.append(kwargs)
        return _reply()

    window = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
        stream_post=lambda *a, **k: [], post=lambda *a, **k: {}, now=lambda: clock["now"], mentor_enabled=False,
        tape_builder=lambda: regime_pack.build(now=clock["now"], sources=regime_pack.fixture_sources()),
        tape_request=request, push_send=lambda title, line, **k: sent.append(line) or {"ok": True},
    )
    window.clock, window.calls, window.sent = clock, calls, sent
    yield window
    window.shutdown()
    window.deleteLater()


def _drain(window):
    while window.queue.run_one():
        pass


def _up(window):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "gemma3:12b"


def test_tape_with_the_brain_off_is_the_pack_and_no_model_call(win):
    win._brain_reason = "the night AI owns the GPU"
    win.send("/tape")
    _drain(win)
    text = win.transcript.toPlainText()
    assert "[tape:mode]" in text and "night AI owns the GPU" in text and win.calls == []


def test_tape_narrates_once_then_answers_from_the_cache_fast(win):
    _up(win)
    win.send("/tape")
    _drain(win)
    assert len(win.calls) == 1 and "the read" in win.transcript.toPlainText()
    assert "[tape:d1env]" in win.transcript.toPlainText()
    import time

    started = time.perf_counter()
    win.send("/tape")
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert win.queue.pending() == [], "a fresh tape answers without a job"
    assert elapsed_ms < 300 and win.transcript.toPlainText().count("Tape: the read") == 2
    assert len(win.calls) == 1, "the cached read is not narrated again"
    win._io.submit(lambda: None).result(5)
    stored = [row for row in win.store.turns() if row["role"] == "assistant"]
    assert stored and '"regime_pack"' in stored[-1]["tool_calls_json"]


def test_the_prefetch_rebuilds_every_30_min_and_never_moves_the_transcript(win):
    _up(win)
    before = win.transcript.toPlainText()
    win.maybe_prefetch_tape()
    _drain(win)
    assert len(win.calls) == 1 and win._tape_last["card"].narrated
    win.clock["now"] += timedelta(minutes=10)
    win.maybe_prefetch_tape()
    assert win.queue.pending() == [], "not due yet"
    win.clock["now"] += timedelta(minutes=21)
    win.maybe_prefetch_tape()
    _drain(win)
    assert len(win.calls) == 1, "same hash: no second narration"
    win.clock["now"] += timedelta(minutes=31)
    win._tape_builder = lambda: regime_pack.build(now=win.clock["now"], sources=replace(
        regime_pack.fixture_sources(), d1_env=lambda day: "bullish_trend"))
    win.maybe_prefetch_tape()
    _drain(win)
    assert len(win.calls) == 2, "a changed pack is narrated again"
    assert win.transcript.toPlainText() == before, "the prefetch never moves the transcript"


def test_a_failed_re_read_keeps_the_good_read_for_the_same_tape(win):
    _up(win)
    win.send("/tape")
    _drain(win)
    good = win._tape_last["card"]
    win._on_tape_ready({**win._tape_last, "card": tape.TapeCard(pack_hash=win._tape_last["hash"], error="down"),
                        "narrated": True})
    assert win._tape_last["card"] is good


def test_turning_the_push_on_after_0630_still_sends_once_that_day(win, monkeypatch):
    enabled = {"on": False}
    monkeypatch.setattr(settings, "push_brief_enabled", lambda: enabled["on"])
    for minutes in (0, 10):
        win.clock["now"] = TUE_0631 + timedelta(minutes=minutes)
        win.maybe_push_brief()
        _drain(win)
    assert win.sent == []
    enabled["on"] = True  # the trader flips it on at 06:50
    for minutes in (19, 20, 40):
        win.clock["now"] = TUE_0631 + timedelta(minutes=minutes)
        win.maybe_push_brief()
        _drain(win)
    assert len(win.sent) == 1


def test_the_window_pushes_once_a_day_only_with_the_setting_on(win, monkeypatch):
    win.clock["now"] = TUE_0631
    monkeypatch.setattr(settings, "push_brief_enabled", lambda: False)
    win.maybe_push_brief()
    _drain(win)
    assert win.sent == []
    win._push_checked_day = None
    monkeypatch.setattr(settings, "push_brief_enabled", lambda: True)
    for minutes in (0, 1, 20):
        win.clock["now"] = TUE_0631 + timedelta(minutes=minutes)
        win.maybe_push_brief()
        _drain(win)
    win._push_checked_day = None  # even a restarted app (fresh memory) sends nothing twice
    win.maybe_push_brief()
    _drain(win)
    assert len(win.sent) == 1 and win.sent[0].startswith("Tape: Auto DESK")
