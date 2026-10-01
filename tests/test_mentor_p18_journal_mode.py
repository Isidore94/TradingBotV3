"""P18 A: journal mode - self talk is kept as a tagged journal line with a one-line "Noted"; questions are answered."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import commands, journal_mode  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import journal_pack  # noqa: E402

NOW = journal_pack.FIXTURE_NOW  # Wed 2026-09-30 11:00 ET


@pytest.mark.parametrize("text", [
    "I'm annoyed, I chased TWLO again",
    "feeling tired today",
    "tempted to revenge trade that one",
    "bored, nothing is moving and I want to click something",
])
def test_self_talk_is_a_statement_that_does_not_ask(text):
    kind = journal_mode.classify(text)
    assert kind.statement and not kind.asks


@pytest.mark.parametrize("text", [
    "Should I short NVDA here?",
    "what is the tape doing",
    "how did my reads do this week",
    "show me my trades",
    "Is NVDA worth it?",
    "mode?",
    "I stop after two losses.",  # plan talk: the model and plan inference see it
    "From now on I only trade the first hour",
])
def test_questions_are_not_journal_lines(text):
    kind = journal_mode.classify(text)
    assert not kind.statement and kind.asks


def test_a_statement_with_a_question_is_stored_and_answered():
    kind = journal_mode.classify("I'm feeling tilted after that stop, should I walk away?")
    assert kind.statement and kind.asks


def test_journal_on_forces_every_message_into_the_journal():
    kind = journal_mode.classify("NVDA holding the vwap nicely", forced=True)
    assert kind.statement and not kind.asks
    assert journal_mode.classify("NVDA holding the vwap nicely").statement is False


def test_mood_tags_come_from_the_desk_vocabulary_deterministically():
    codes, version = journal_mode.vocabulary_codes()
    assert version == 1 and set(journal_mode.TAG_WORDS) <= set(codes)
    assert journal_mode.mood_tags("I'm annoyed, I chased TWLO", codes) == ("fomo", "tilted")
    assert journal_mode.mood_tags("so tired and bored", codes) == ("bored", "tired")
    assert journal_mode.mood_tags("madness", codes) == ()  # whole words only
    # A code the vocabulary does not hold is never emitted.
    assert journal_mode.mood_tags("I'm tired", ("calm",)) == ()


def test_entry_fields_carry_tape_trade_bucket_and_weekday(tmp_path):
    db = journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")
    rows = [{"id": "ctx:regime", "text": "Trader's regime: chop since 2026-09-20 (day 8)"},
            {"id": "ctx:today", "text": "Today so far: 2 closed trade(s)"}]
    # 10:42 ET: the ALL spread closed 10:40 (+$50 net), the NVDA long 10:20 (22 min ago).
    now = datetime(2026, 9, 30, 14, 42, tzinfo=timezone.utc)
    fields = journal_mode.entry_fields("I'm annoyed, I chased it", now, context_rows=rows, journal=db)
    assert fields["day_et"] == "2026-09-30" and fields["time_bucket_et"] == "10:30" and fields["weekday"] == "Wed"
    assert fields["regime"].startswith("Trader's regime: chop") and fields["tape"].startswith("Today so far")
    assert fields["last_trade_id"] in ("W3", "W4") and fields["last_trade_text"].startswith("the ALL ")
    assert fields["mood_tags"] == ["fomo", "tilted"] and fields["vocab_version"] == 1
    line = journal_mode.noted_line(fields, 7)
    assert line.startswith("Noted, 10:42, after the ALL ") and "(fomo, tilted)" in line
    assert line.endswith("[jrn:2026-09-30:entry:7]") and "\n" not in line
    assert journal_mode.noted_line(fields, None).startswith("That journal line was NOT saved")


def test_a_loss_inside_30_minutes_reads_as_a_stop(tmp_path):
    db = journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")
    now = datetime(2026, 9, 30, 14, 10, tzinfo=timezone.utc)  # 10:10 ET: AMD short lost at 10:05
    trade = journal_mode.last_closed_trade(db, now)
    assert trade["trade_id"] == "W2" and trade["text"] == "the AMD stop"
    assert journal_mode.last_closed_trade(db, datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)) is None


def test_store_keeps_entries_and_recall_falls_back_to_them(tmp_path):
    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    fields = journal_mode.entry_fields("I'm bored and tempted", NOW)
    entry_id = store.add_journal_entry(fields)
    assert entry_id
    got = store.journal_entries("2026-09-30", "2026-09-30")
    assert [row["text"] for row in got] == ["I'm bored and tempted"]
    assert got[0]["mood_tags"] == ["fomo", "bored"]
    assert store.journal_entries("2026-09-29", "2026-09-29") == []
    assert [(row["kind"], row["ref_id"]) for row in store.search_text("tempted")] == [("journal", entry_id)]
    assert [row["id"] for row in store.unembedded_journal("m")] == [entry_id]
    store.put_embedding("journal", entry_id, "m", [1.0, 0.0], text="x")
    assert store.unembedded_journal("m") == []


def test_journal_pack_shows_the_days_entries_with_tags_next_to_the_trades(tmp_path):
    db = journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")
    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    now = datetime(2026, 9, 30, 14, 10, tzinfo=timezone.utc)
    store.add_journal_entry(journal_mode.entry_fields("I'm annoyed, I chased AMD", now, journal=db))
    store.add_journal_entry(journal_mode.entry_fields("tired", datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)))
    pack = journal_pack.build("today", now=NOW, journal=db, chat_db=store.path)
    entries = [row for row in pack.rows if row.get("kind") == "entry"]
    assert [row["id"] for row in entries] == ["jrn:2026-09-30:entry:1"]
    assert entries[0]["tags"] == ["fomo", "tilted"] and entries[0]["trade"] == "W2"
    assert "[fomo, tilted]" in entries[0]["text"] and "after the AMD stop (W2)" in entries[0]["text"]
    assert "I'm annoyed, I chased AMD" in entries[0]["text"]
    yesterday = journal_pack.build("yesterday", now=NOW, journal=db, chat_db=store.path)
    assert [row["id"] for row in yesterday.rows if row.get("kind") == "entry"] == ["jrn:2026-09-29:entry:2"]


def test_journal_pack_reads_the_chat_store_read_only(tmp_path):
    db = journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")
    chat = tmp_path / "missing_chat.sqlite3"
    pack = journal_pack.build("today", now=NOW, journal=db, chat_db=chat)
    assert not chat.exists() and not [row for row in pack.rows if row.get("kind") == "entry"]
    # An older chat store without the table reads as no entries.
    old = tmp_path / "old_chat.sqlite3"
    sqlite3.connect(old).execute("CREATE TABLE turns (id INTEGER)").connection.close()
    assert journal_pack.read_entries(old, NOW.date(), NOW.date()) == []


def test_journal_command_parses():
    assert commands.handle("/journal") == commands.CommandResult("journal", "", "")
    assert commands.handle("/journal on").arg == "on" and commands.handle("/journal OFF").arg == "off"
    assert commands.handle("/journal maybe").action == "error"
    assert "/journal" in commands.HELP_TEXT


# ------------------------------------------------------------------ the window
@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    db = journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")
    win = MentorWindow(store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
                       stream_post=lambda url, payload, cancelled: [], post=lambda url, payload, timeout: {},
                       tilt_journal=db, now=lambda: datetime(2026, 9, 30, 14, 10, tzinfo=timezone.utc))
    yield win
    win.shutdown()
    win.deleteLater()


def _settle(win):
    from PySide6.QtWidgets import QApplication

    win._io.submit(lambda: None).result(5)
    QApplication.processEvents()


def test_window_keeps_a_statement_and_answers_in_one_line_without_the_model(window):
    window._brain_ok = False
    window.send("I'm annoyed, I chased AMD")
    _settle(window)
    text = window.transcript.toPlainText()
    assert "Noted, 10:10, after the AMD stop (fomo, tilted)." in text
    assert "brain is off" not in text.lower() and window._worker is None
    assert [row["text"] for row in window.store.journal_entries()] == ["I'm annoyed, I chased AMD"]
    assert window.store.turns() == []  # a journal line is not a chat turn


def test_window_answers_a_question_and_keeps_no_journal_line(window):
    window._brain_ok = False
    window.send("Is NVDA worth it?")
    _settle(window)
    assert window.store.journal_entries() == []
    assert [row["text"] for row in window.store.turns()] == ["Is NVDA worth it?"]


def test_window_statement_with_a_question_is_stored_and_still_answered(window):
    window._brain_ok = False
    window.send("I'm tilted, should I stop for the day?")
    _settle(window)
    assert len(window.store.journal_entries()) == 1
    assert [row["text"] for row in window.store.turns()] == ["I'm tilted, should I stop for the day?"]
    assert "brain is off" in window.transcript.toPlainText().lower()  # it went on to be answered


def test_window_journal_on_persists_and_journal_lists_today(window):
    window.send("/journal on")
    window.send("SPY holding the open range")
    _settle(window)
    assert window.store.get_state(journal_mode.MODE_KEY) == "on"
    assert [row["forced"] for row in window.store.journal_entries()] == [1]
    window.send("/journal")
    _settle(window)
    assert "**Journal 2026-09-30**" in window.transcript.toPlainText() or "Journal 2026-09-30" in (
        window.transcript.toPlainText())
    window.send("/journal off")
    window.send("SPY holding the open range")
    _settle(window)
    assert len(window.store.journal_entries()) == 1
