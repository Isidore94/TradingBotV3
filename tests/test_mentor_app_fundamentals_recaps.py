"""Mentor app P15b: the pasted brief in memory, /paste through the Market Journal writer, /tape with the brief,
recaps and their recurrence table, the feelings question, and the app owning the Mentor by default."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import commands, memory  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import fundamentals_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 30, 7, 0, tzinfo=PT)  # Wed, the fixture brief's day


@pytest.fixture()
def fund(tmp_path):
    return fundamentals_pack.write_fixture_world(tmp_path / "fund")


# ---------------------------------------------------------------- memory tier (pure)
def test_the_brief_tier_sits_between_the_coach_brief_and_the_digests(tmp_path, fund):
    from mentor_packs import night_pack

    root = tmp_path / "ai"
    root.mkdir()
    (root / "mentor_day_digest_2026-09-29.json").write_text(json.dumps({
        "session_date": "2026-09-29", "digest": [{"text": "Asked about NVDA twice."}]}), encoding="utf-8")
    (root / "mentor_coach_brief_2026-09-29.json").write_text(json.dumps({
        "session_date": "2026-09-29", "one_line": {"text": "Watch the 10-year.", "evidence_refs": ["x:1"]},
        "watch": [], "missing": [], "issues": []}), encoding="utf-8")
    loaded = memory.load(MentorChatStore(tmp_path / "c.sqlite3"), ai_root=root, night_paths=night_pack.NightPaths(),
                         now=NOW, fund_paths=fund)
    ids = [item.id for item in loaded.items]
    assert ids[0] == "night:coach:2026-09-29:0"
    fund_ids = [i for i in ids if i.startswith("fund:")]
    assert fund_ids == ["fund:2026-09-30:asof", "fund:2026-09-30:bottom:1", "fund:2026-09-30:bull:1",
                        "fund:2026-09-30:bear:1", "fund:2026-09-30:turb:1"]
    assert ids.index("fund:2026-09-30:turb:1") < ids.index("mem:digest:2026092900")
    assert len(fund_ids) <= memory.FUND_LINES
    assert "[fund:2026-09-30:bottom:1] (2026-09-30) Bottom line: Softer core PCE" in loaded.text
    assert memory.PRIORITY_COACH < memory.PRIORITY_FUND < memory.PRIORITY_DIGEST


def test_no_brief_for_today_puts_nothing_in_memory(tmp_path, fund):
    tomorrow = datetime(2026, 10, 1, 7, 0, tzinfo=PT)
    assert memory.fund_items(fund, tomorrow) == [], "yesterday's brief is not today's; a none row is never memory"
    assert memory.fund_items(fundamentals_pack.FundPaths(), NOW) == []


def test_the_brief_paragraphs_are_embedded_once_per_paste(tmp_path, fund):
    from mentor_packs import night_pack

    loaded = memory.Memory()
    first = [row for row in memory.embed_candidates(loaded, night_pack.NightPaths(), NOW, fund_paths=fund)
             if row[0] == "fund"]
    assert first and all(text.startswith("[fund:2026-09-30:p") for _k, _r, text in first)
    assert first == [row for row in memory.embed_candidates(loaded, night_pack.NightPaths(), NOW, fund_paths=fund)
                     if row[0] == "fund"]
    assert "fund" in memory.EMBED_KINDS


def test_paste_is_a_command_that_keeps_every_line():
    assert commands.handle("/paste") == commands.CommandResult("paste", "", ("", ""))
    got = commands.handle("/paste **Brief**\n\n- line one\n- line two")
    assert got.action == "paste" and got.arg == ("**Brief**\n\n- line one\n- line two", "")
    assert commands.handle("/paste for 2026-09-29 body\nmore").arg == ("body\nmore", "2026-09-29")
    assert commands.handle("/paste for 2026-09-29").arg == ("", "2026-09-29")
    assert commands.handle("/paste for tomorrow x").action == "error"
    assert "/paste" in commands.HELP_TEXT


def test_the_tape_reads_the_brief_too():
    from mentor_app import tape
    from mentor_packs.registry import make_pack

    regime = make_pack("regime_pack", [{"id": "tape:regime", "text": "bear channel"}])
    merged = tape.with_fundamentals(regime, make_pack("fundamentals_pack", [
        {"id": "fund:2026-09-30:bottom:1", "text": "Bottom line: rates lead."}]))
    assert merged.name == "regime_pack" and merged.ids == ("tape:regime", "fund:2026-09-30:bottom:1")
    assert tape.with_fundamentals(regime, None) is regime
    assert "'fund' row is the morning brief" in tape.TASK


# ---------------------------------------------------------------- the window
@pytest.fixture(scope="module")
def app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakeForecastService:
    """Stands in for ``MarketJournalService``: records the call and appends like the real writer."""

    def __init__(self, paths: fundamentals_pack.FundPaths, *, fail: str = "") -> None:
        self.paths, self.fail, self.calls = paths, fail, []

    def import_daily_forecast(self, *, text, target_session="", now=None, **kwargs):
        import threading

        import market_journal
        from evidence_ledger import EvidenceLedger

        self.calls.append({"text": text, "target_session": target_session, "now": now,
                           "thread": threading.current_thread().name, **kwargs})
        if self.fail:
            return {"ok": False, "reason": self.fail}
        ledger = EvidenceLedger(stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
                                directory=self.paths.ledger_dir)
        row = ledger.append({"entry_id": f"mj-paste-{len(self.calls)}", "event_type": "entry",
                             "origin": "external_forecast", "text": text,
                             "supersedes": "mj-2026-09-30-second" if target_session == "2026-09-30" else "",
                             "created_at": now.astimezone(timezone.utc).isoformat(timespec="seconds")},
                            now=now, subject_session_date=target_session)
        return {"ok": True, "entry": row}


@pytest.fixture()
def window(app, tmp_path, monkeypatch, fund):
    from mentor_app import settings
    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    clock = {"now": NOW}
    service = FakeForecastService(fund)
    prompts: list = []
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        inbox=Inbox(per_day_cap=6, now=lambda: clock["now"]),
        stream_post=lambda url, payload, cancelled: [],
        post=lambda url, payload, timeout: {},
        now=lambda: clock["now"],
        mentor_enabled=False,
        liked_source=lambda: [],
        memory_root=tmp_path / "ai",
        forecast_service=service,
        paste_prompt=lambda: prompts.pop(0) if prompts else None,
    )
    win.fund_paths = fund
    win.clock, win.service, win.prompts = clock, service, prompts
    yield win
    win.shutdown()
    win.deleteLater()


def _drain(win, app):
    for _ in range(50):
        win._io.submit(lambda: None).result(5)
        app.processEvents()
        if not win.queue.run_one():
            app.processEvents()
            if not win.queue.pending():
                break
    win._io.submit(lambda: None).result(5)
    app.processEvents()


def _text(win):
    return win.transcript.toPlainText()


NEW_BRIEF = "# Brief — September 30, 2026\n\n## Bottom line\n\nOil leads today. Yields are second.\n"


def test_paste_saves_through_the_journal_writer_off_the_qt_thread_and_refreshes_memory(window, app):
    window.send("/paste " + NEW_BRIEF)
    _drain(window, app)
    call = window.service.calls[0]
    assert call["text"] == NEW_BRIEF.strip() and call["target_session"] == "2026-09-30"
    assert call["thread"].startswith("mentor-store"), "the write runs on the IO thread, never the Qt thread"
    assert "Brief saved for 2026-09-30: Oil leads today." in _text(window)
    assert "[fund:2026-09-30:bottom:1] (2026-09-30) Bottom line: Oil leads today." in window._memory_block
    window.send("/memory")
    assert "fund:2026-09-30:bottom:1" in _text(window)


def test_the_paste_button_opens_the_box_and_cancel_saves_nothing(window, app):
    window.paste_button.click()  # the prompt returns None: cancelled
    _drain(window, app)
    assert window.service.calls == []
    window.prompts.append(NEW_BRIEF)
    window.paste_button.click()
    _drain(window, app)
    assert len(window.service.calls) == 1 and "Brief saved for 2026-09-30" in _text(window)
    window.send("/paste   ")
    window.prompts.append("   ")
    _drain(window, app)
    assert len(window.service.calls) == 1, "an empty paste writes nothing"


def test_a_refused_paste_says_not_saved(window, app):
    window.service.fail = "the ledger is locked"
    window.send("/paste " + NEW_BRIEF)
    _drain(window, app)
    assert "Brief NOT saved: the ledger is locked" in _text(window)


# ---------------------------------------------------------------- review fix: the desk's session rule
def test_paste_files_by_the_desks_rule_not_the_paste_moment(window, app):
    """Wed 21:30 PT (Thu 00:30 ET): a Wednesday-titled brief files under Wednesday, exactly as the desk files it,
    and /memory reads it back as today's brief."""
    from ui.services.market_journal_service import forecast_session

    window.clock["now"] = datetime(2026, 9, 30, 21, 30, tzinfo=PT)
    assert forecast_session(NEW_BRIEF, window.clock["now"]) == "2026-09-30"
    window.send("/paste " + NEW_BRIEF)
    _drain(window, app)
    assert window.service.calls[-1]["target_session"] == "2026-09-30"
    assert "Brief saved for 2026-09-30: Oil leads today." in _text(window) and "not today" not in _text(window)
    window.send("/memory")
    assert "[fund:2026-09-30:bottom:1]" in _text(window)


def test_an_untitled_paste_files_under_the_last_closed_session_and_says_so(window, app):
    window.clock["now"] = datetime(2026, 9, 30, 5, 45, tzinfo=PT)  # before Wednesday's open
    window.send("/paste Rates lead today.\n\nOil is the risk.")
    _drain(window, app)
    assert window.service.calls[-1]["target_session"] == "2026-09-29", "no title date: the last closed session"
    assert "filed for 2026-09-29, not today 2026-09-30; /paste for 2026-09-30" in _text(window)
    window.send("/paste for 2026-09-30 Rates lead today.")
    _drain(window, app)
    assert window.service.calls[-1]["target_session"] == "2026-09-30", "the trader can name the session"


def test_a_saturday_paste_titled_friday_reads_as_the_last_brief_until_mondays(window, app, fund):
    friday = "# Brief — October 2, 2026\n\n## Bottom line\n\nPayrolls hot. Yields up.\n"
    window.clock["now"] = datetime(2026, 10, 3, 9, 0, tzinfo=PT)  # Saturday
    window.send("/paste " + friday)
    _drain(window, app)
    assert window.service.calls[-1]["target_session"] == "2026-10-02"
    for moment in (datetime(2026, 10, 3, 9, 0, tzinfo=PT), datetime(2026, 10, 5, 6, 0, tzinfo=PT)):
        pack = fundamentals_pack.build("today", now=moment, paths=fund)
        assert pack.ids[0] == "fund:2026-10-05:none", moment
        assert "last brief: 2026-10-02" in pack.rows[1]["text"]
        assert memory.fund_items(fund, moment) == [], "not today's brief: not in memory"
    monday = "# Brief — October 5, 2026\n\n## Bottom line\n\nQuiet open.\n"
    window.clock["now"] = datetime(2026, 10, 5, 6, 0, tzinfo=PT)
    window.send("/paste " + monday)
    _drain(window, app)
    assert fundamentals_pack.build("today", now=window.clock["now"], paths=fund).ids[0] == "fund:2026-10-05:asof"


def test_the_tape_build_carries_the_compact_brief(window):
    from mentor_packs.registry import make_pack

    window._tape_builder = lambda: make_pack("regime_pack", [{"id": "tape:regime", "text": "bear channel"}])
    window._fund_builder = lambda section: fundamentals_pack.build("today", section, now=NOW, paths=window.fund_paths)
    ids = window._build_tape().ids
    assert ids[0] == "tape:regime" and "fund:2026-09-30:bottom:1" in ids and len(ids) <= 1 + 8


# ---------------------------------------------------------------- step 2: recaps and issues
@pytest.fixture()
def recaps(tmp_path):
    from mentor_packs import recaps_pack

    return recaps_pack.write_fixture_world(tmp_path / "recaps")


def test_recap_words_attach_the_recaps_pack_and_night_words_keep_the_night():
    from mentor_app import attach

    known = {"ALL": "SHORT", "NVDA": "LONG"}
    for question in ("what are my most pertinent issues", "what have I been doing wrong lately",
                     "what did yesterday's recap say", "what am I missing", "am I making the same mistake",
                     "what should I keep doing", "any pattern in what I miss lately", "review my last week"):
        names = [r.name for r in attach.plan_attachments(question, known, NOW)]
        assert "recaps_pack" in names, question
    assert "night_pack" in [r.name for r in attach.plan_attachments("what did the night say overnight", known, NOW)]
    assert [r.name for r in attach.plan_attachments("how has my record been lately", known, NOW)] == ["mirror_pack"]
    request = next(r for r in attach.plan_attachments("what are my issues", known, NOW) if r.name == "recaps_pack")
    assert request.args == {"days": 10, "section": "all"}


def test_the_coach_is_told_to_rank_the_table_and_never_invent_an_issue():
    from mentor_app.chat_model import SYSTEM_PROMPT

    assert ("When asked about issues, rank the recurrence table by sessions count, cite each row, and say in one "
            "sentence what to watch for today; never invent an issue not in a row.") in SYSTEM_PROMPT


def test_the_recap_rows_are_embedded_as_kind_recap(recaps):
    from mentor_packs import night_pack

    rows = [row for row in memory.embed_candidates(memory.Memory(), night_pack.NightPaths(), NOW,
                                                   fund_paths=fundamentals_pack.FundPaths(), recap_paths=recaps)
            if row[0] == "recap"]
    assert rows and all(text.startswith("[recap:2026-") for _k, _r, text in rows)
    assert "recap" in memory.EMBED_KINDS


def _chat_db(tmp_path):
    path = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(path)
    store.add_turn(store.start_session("m"), "user", "what are my issues")
    return path


def test_the_night_issue_candidates_carry_the_recap_recurrence_rows(tmp_path, recaps):
    from ai_jobs import mentor_review
    from mentor_packs import night_pack

    found = mentor_review.issue_candidates(_chat_db(tmp_path), "2026-09-29", night_paths=night_pack.NightPaths(),
                                           recap_paths=recaps)
    by_key = {item["key"]: item for item in found}
    item = by_key["recap:missed:compressed"]
    assert item["id"] == "issue:recap:missed:compressed" and item["count"] == 2
    assert item["refs"] == ["recap:issues:missed:compressed"] and item["first_seen"] == "2026-09-25"
    assert "recap:wrong_reads" not in by_key, "the night already has its own wrong-reads issue"
    night = mentor_review.night_inputs(_chat_db(tmp_path / "b"), "2026-09-29", datetime(2026, 9, 29, 23, 0, tzinfo=PT),
                                       night_paths=night_pack.NightPaths(), mirror_builder=lambda: _empty_pack(),
                                       recap_paths=recaps)
    assert "recap:issues:missed:compressed" in [row["id"] for row in night["recap_issues"]]


def _empty_pack():
    from mentor_packs.registry import make_pack

    return make_pack("mirror_pack", ())


def test_the_facts_brief_and_the_registry_name_the_recap_issues(tmp_path, recaps):
    from ai_jobs import mentor_review
    from mentor_packs import night_pack

    out = mentor_review.run_mentor_review(
        session_date="2026-09-29", now=datetime(2026, 9, 29, 23, 0, tzinfo=PT), chat_db=_chat_db(tmp_path),
        ai_root=tmp_path / "ai", veto_outcomes=tmp_path / "none.csv", ask=False, night_paths=night_pack.NightPaths(),
        mirror_builder=_empty_pack, recap_paths=recaps)
    assert out["status"] == "ok", out["reason"]
    brief = json.loads((tmp_path / "ai" / "mentor_coach_brief_2026-09-29.json").read_text(encoding="utf-8"))
    keys = [item["key"] for item in brief["issues"]]
    assert "recap:missed:compressed" in keys and "recap:clue:volume_dry_up" in keys
    registry = mentor_review.read_issue_registry(tmp_path / "ai")
    assert registry["recap:rule_broken:wait_for_confirmation"]["first_seen"] == "2026-09-28"


def test_recaps_and_issues_commands(window, app, recaps):
    window.recap_paths = recaps
    window.send("/recaps 2")
    window.send("/issues")
    _drain(window, app)
    text = _text(window)
    assert "[recap:2026-09-29:card:missed]" in text and "[recap:2026-09-25:" not in text.split("Recurring")[0]
    assert "No recurring issues in the last night's brief." in text
    assert "Recurring in your day recaps" in text and "[recap:issues:missed:compressed]" in text
    assert commands.handle("/recaps all").arg == "all" and commands.handle("/recaps").arg == 10
    assert commands.handle("/recaps 99").action == "error"


# ---------------------------------------------------------------- step 4: one feelings question per closed trade
@pytest.fixture()
def journal(tmp_path):
    from mentor_packs import journal_pack

    # Wed 2026-09-30: NVDA long, AMD short, a two-leg ALL put spread closed; MSFT still open.
    return journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")


def test_closed_trades_are_found_once_and_never_asked_twice(tmp_path, journal):
    from mentor_app import tilt_watch

    closed = tilt_watch.closed_today(journal, "2026-09-30")
    assert [(row["trade_id"], row["symbol"], row["side"]) for row in closed] == [
        ("W2", "AMD", "SHORT"), ("W1", "NVDA", "LONG"), ("W3", "ALL", "LONG"), ("W4", "ALL", "SHORT")]
    store = MentorChatStore(tmp_path / "c.sqlite3")
    owed = tilt_watch.pending_closes(store, "2026-09-30", closed)
    assert [row["trade_id"] for row in owed] == ["W2", "W1", "W3", "W4"]
    assert tilt_watch.pending_closes(store, "2026-09-30", closed) == owed, "held, not lost, until the Inbox takes it"
    tilt_watch.mark_asked(store, "2026-09-30", ["W2"])
    assert [row["trade_id"] for row in tilt_watch.pending_closes(store, "2026-09-30", closed)] == ["W1", "W3", "W4"]
    assert tilt_watch.waiting_closes(store, "2026-09-30")[0]["trade_id"] == "W1", "a restart reads the held ones"
    assert tilt_watch.feel_text(closed[1]) == "How did the NVDA long feel? One word or a sentence."


@pytest.fixture()
def feel_window(app, tmp_path, monkeypatch, journal):
    from mentor_app import settings
    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow
    from mentor_packs.registry import make_pack

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    clock = {"now": datetime(2026, 9, 30, 8, 0, tzinfo=PT)}
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(), news_queue=PrefetchQueue(),
        inbox=Inbox(per_day_cap=3, now=lambda: clock["now"]), stream_post=lambda *a, **k: [],
        post=lambda *a, **k: {}, now=lambda: clock["now"], mentor_enabled=False,
        tilt_builder=lambda: make_pack("tilt_pack", ()), tilt_journal=journal, journal_path=journal,
    )
    win.clock = clock
    yield win
    win.shutdown()
    win.deleteLater()


def test_a_restart_during_quiet_hours_still_asks_once(app, tmp_path, monkeypatch, journal):
    """Review advisory 1: a close is marked asked only when the Inbox takes its item; a held one survives a restart."""
    from mentor_app import settings, tilt_watch
    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow
    from mentor_packs.registry import make_pack

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    clock = {"now": datetime(2026, 9, 30, 6, 40, tzinfo=PT)}  # inside the open's quiet hours

    def open_window():
        return MentorWindow(
            store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(), news_queue=PrefetchQueue(),
            inbox=Inbox(per_day_cap=6, now=lambda: clock["now"]), stream_post=lambda *a, **k: [],
            post=lambda *a, **k: {}, now=lambda: clock["now"], mentor_enabled=False,
            # A pack with no observation: the watch stores its signature, so the next pass after a restart skips.
            tilt_builder=lambda: make_pack("tilt_pack", [{"id": "tilt:base:x", "kind": "base", "text": "b"}]),
            tilt_journal=journal)

    first = open_window()
    first.maybe_watch_tilt()
    _drain_all(first, app)
    assert first.inbox.items() == [], "quiet hours: held"
    store = first.store
    assert [row["trade_id"] for row in tilt_watch.waiting_closes(store, "2026-09-30")] == ["W2", "W1", "W3", "W4"]
    first.shutdown()
    first.deleteLater()
    clock["now"] = datetime(2026, 9, 30, 7, 5, tzinfo=PT)
    second = open_window()
    second.maybe_watch_tilt()
    _drain_all(second, app)
    assert [item.text for item in second.inbox.items()] == ["How did the AMD short feel? One word or a sentence."]
    assert [row["trade_id"] for row in tilt_watch.waiting_closes(second.store, "2026-09-30")] == ["W1", "W3", "W4"]
    second.shutdown()
    second.deleteLater()
    clock["now"] = datetime(2026, 9, 30, 7, 40, tzinfo=PT)
    third = open_window()
    third.maybe_watch_tilt()
    _drain_all(third, app)
    assert [item.text for item in third.inbox.items()] == ["How did the NVDA long feel? One word or a sentence."], \
        "AMD was asked once; the next held close is next"
    third.shutdown()
    third.deleteLater()


def _drain_all(win, app):
    while win.queue.run_one() or win.news_queue.run_one():
        pass
    win._io.submit(lambda: None).result(5)
    app.processEvents()


def test_one_feelings_item_per_closed_trade_spaced_and_capped(feel_window, app):
    from datetime import timedelta

    win = feel_window
    before = _text(win)
    win.maybe_watch_tilt()
    _drain_all(win, app)
    items = win.inbox.items()
    assert [item.kind for item in items] == ["feeling"], "one item, never a pop"
    assert items[0].text == "How did the AMD short feel? One word or a sentence."
    assert _text(win) == before, "the transcript never moves on its own"
    win.maybe_watch_tilt()
    _drain_all(win, app)
    assert len(win.inbox.items()) == 1, "the 30-min spacing is shared with tilt"
    for _step in (1, 2, 3):
        win.clock["now"] += timedelta(minutes=31)
        win.maybe_watch_tilt()
        _drain_all(win, app)
    assert [item.text.split(" feel")[0] for item in win.inbox.items()] == [
        "How did the AMD short", "How did the NVDA long", "How did the ALL long"]
    assert win._feel_waiting == [], "the daily cap (3) drops the rest; nothing pops later"


def test_the_answer_is_a_feeling_note_with_its_trade_id_and_rides_on_the_journal_row(feel_window, app):
    from mentor_packs import journal_pack

    win = feel_window
    win.maybe_watch_tilt()
    _drain_all(win, app)
    item = win.inbox.items()[0]
    row = win.inbox_list.item(0)
    win._open_inbox_item(row)
    assert win.input.toPlainText() == "/feel W2 " and item.id in win._inbox_feel
    win.input.setPlainText("/feel W2 rushed it, felt revenge-y")
    win.send_current()
    win.send("/feel NVDA calm and patient")
    _drain_all(win, app)
    feelings = win.store.feelings()
    assert [(row["trade_id"], row["kind"]) for row in feelings] == [("W2", "feeling"), ("W1", "feeling")]
    assert feelings[0]["text"] == "How the AMD short (W2) felt: rushed it, felt revenge-y"
    assert "Kept with NVDA (W1): calm and patient" in _text(win)
    assert win.store.profile_notes() == [], "a feeling is not a note about him; it rides on its trade"
    assert win.store.unembedded_notes("m"), "feelings are embedded with the notes"
    pack = journal_pack.build("today", now=datetime(2026, 9, 30, 8, 0, tzinfo=PT), journal=win._tilt_journal,
                              chat_db=win.store.path)
    rows = {row["id"]: row for row in pack.rows}
    assert rows["jrn:2026-09-30:W2:feel"]["text"] == "How AMD (W2) felt, in your words: rushed it, felt revenge-y"
    assert pack.ids.index("jrn:2026-09-30:W1:feel") == pack.ids.index("jrn:2026-09-30:W1") + 1
    win.send("/feel ZZZZ whatever")
    _drain_all(win, app)
    assert "I can't find a trade for ZZZZ" in _text(win)


def test_the_night_reads_a_feeling_by_its_trade_id(tmp_path):
    import sqlite3

    from ai_jobs import mentor_review

    path = tmp_path / "mentor_chat.sqlite3"
    store = MentorChatStore(path)
    store.add_profile_note("rule: two losses and I stop")
    store.add_feeling("W2", "How the AMD short (W2) felt: first try")
    store.add_feeling("W2", "How the AMD short (W2) felt: rushed")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE profile_notes SET ts_utc = ?", ("2026-09-29T17:00:00+00:00",))
    facts = mentor_review.day_facts(path, "2026-09-29")
    inputs = mentor_review.build_inputs(path, "2026-09-29", facts)
    notes = {row["id"]: row["text"] for row in inputs["profile_notes"]}
    assert notes["feel:W2"].endswith("rushed") and "note:1" in notes and len(notes) == 2
    assert "feel:W2" in inputs["allowed_evidence_ids"]
    assert facts["remember_notes"] == 1, "a feeling is not a /remember note"


def test_feel_command_parsing():
    assert commands.handle("/feel SHOP rushed it") == commands.CommandResult("feel", "", ("SHOP", "rushed it"))
    assert commands.handle("/feel SHOP").action == "error"
