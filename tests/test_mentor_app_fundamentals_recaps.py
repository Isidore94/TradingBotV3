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
    assert commands.handle("/paste") == commands.CommandResult("paste", "", "")
    got = commands.handle("/paste **Brief**\n\n- line one\n- line two")
    assert got.action == "paste" and got.arg == "**Brief**\n\n- line one\n- line two"
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
                             "origin": "external_forecast", "text": text, "supersedes": "mj-2026-09-30-second",
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


def test_the_tape_build_carries_the_compact_brief(window):
    from mentor_packs.registry import make_pack

    window._tape_builder = lambda: make_pack("regime_pack", [{"id": "tape:regime", "text": "bear channel"}])
    window._fund_builder = lambda section: fundamentals_pack.build("today", section, now=NOW, paths=window.fund_paths)
    ids = window._build_tape().ids
    assert ids[0] == "tape:regime" and "fund:2026-09-30:bottom:1" in ids and len(ids) <= 1 + 8
