"""TJ-1 items 1 and 2 - the desk stops writing its own rows, and the ONE filter.

Trader, 2026-09-17: *"i don't need to see the SPY auto modes pasted in there."*
Measured on the live desk that evening: **34 of 77** Market Journal rows were the
desk's own `Auto mode X -> Y` rows, and the 2026-09-16 nightly narration repeated
them back.

Decision 0021 answer 3 is binding and has two halves:

* **Forward.** `MainWindow._record_auto_mode_flip` keeps its name and its caller,
  and its whole body becomes ONE Auto Pilot log line. No journal write, no chart
  capture.
* **Backward.** The 34 rows already on disk are FILTERED, never deleted - the
  ledger is append-only. `market_journal.is_machine_entry` is the one filter and
  every reader that feeds a trader surface or an AI input inherits it.

These tests drive the real seams: the bound method with stubbed panels, and a
real service over a real temp ledger walked by each reader in turn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6", reason="the desk window and journal service are Qt objects")

SESSION = "2026-09-16"

TRADER_TEXT = "SPY faded the open and the semis never confirmed. I stood down."
MENTOR_TEXT = "10:00 read: the tape is heavy, breadth is negative, I am patient."
MACHINE_TEXT = "Auto mode DESK -> AWAY. Written by the desk, not the trader."


# ==========================================================================
# 1. a mode flip is one Auto Pilot log line and nothing else
# ==========================================================================
class _RecordingJournal:
    """Stands in for `MarketJournalService`. Records, never writes."""

    def __init__(self) -> None:
        self.writes: list[dict] = []
        self.captures: list[dict] = []

    def write_entry(self, **kwargs):
        self.writes.append(dict(kwargs))
        return {"ok": True, "entry": {"entry_id": "mj-stub-0001"}}

    def capture_charts(self, **kwargs):
        self.captures.append(dict(kwargs))
        return True


class _RecordingAlertCenter:
    def __init__(self) -> None:
        self.bar_calls: list[str] = []

    def journal_chart_bars(self, symbol):
        self.bar_calls.append(str(symbol))
        return [], []


class _RecordingAutopilot:
    def __init__(self, *, explode: bool = False) -> None:
        self.logged: list[str] = []
        self._explode = explode

    def log(self, message):
        if self._explode:
            raise RuntimeError("the log file is on a disconnected share")
        self.logged.append(str(message))


def _desk(*, explode_log: bool = False) -> SimpleNamespace:
    """The three panels `_record_auto_mode_flip` can reach, all stubbed.

    Stubbed rather than a real `MainWindow`: what is under test is the BODY of
    one method, and a desk window would bring a hundred other writers with it.
    """
    journal = _RecordingJournal()
    alert_center = _RecordingAlertCenter()
    autopilot = _RecordingAutopilot(explode=explode_log)
    return SimpleNamespace(
        market_journal_panel=SimpleNamespace(service=journal),
        day_review_panel=SimpleNamespace(service=journal),
        trading_panel=SimpleNamespace(alert_center=alert_center),
        autopilot_panel=SimpleNamespace(service=autopilot),
        _journal=journal,
        _alert_center=alert_center,
        _autopilot=autopilot,
    )


def _flip(desk, previous="DESK", current="AWAY") -> None:
    from ui.app import MainWindow

    MainWindow._record_auto_mode_flip(desk, previous, current)


def test_a_mode_flip_writes_one_auto_pilot_log_line():
    desk = _desk()
    _flip(desk)
    assert desk._autopilot.logged == ["Auto mode DESK -> AWAY."]


def test_a_mode_flip_writes_no_market_journal_row():
    """The 34-of-77 defect, at its source."""
    desk = _desk()
    _flip(desk)
    assert desk._journal.writes == [], desk._journal.writes


def test_a_mode_flip_captures_no_charts_and_reads_no_bars():
    """The capture existed only to illustrate the row that is gone."""
    desk = _desk()
    _flip(desk)
    assert desk._journal.captures == []
    assert desk._alert_center.bar_calls == []


def test_an_unknown_side_of_the_flip_is_named_unknown_not_blank():
    desk = _desk()
    _flip(desk, "", "AWAY")
    assert desk._autopilot.logged == ["Auto mode UNKNOWN -> AWAY."]
    desk = _desk()
    _flip(desk, "DESK", "")
    assert desk._autopilot.logged == ["Auto mode DESK -> UNKNOWN."]


def test_the_log_line_does_not_explain_itself_to_the_journal_any_more():
    """"Written by the desk, not the trader" was a sentence for a JOURNAL reader.
    In the Auto Pilot log every line is the desk's."""
    desk = _desk()
    _flip(desk)
    assert desk._autopilot.logged
    assert "not the trader" not in desk._autopilot.logged[0]


def test_a_log_that_cannot_be_written_never_raises_into_the_mode_change():
    """The mode has already changed by the time this runs; an evidence store
    must never cost the thing it records."""
    desk = _desk(explode_log=True)
    _flip(desk)  # must not raise
    assert desk._journal.writes == []


def test_the_autopilot_service_gained_one_public_log_forwarding_to_its_one_seam():
    """`_log` is the only writer (deque + file + logging + signal). The public
    wrapper forwards; it does not become a second writer."""
    from ui.services.autopilot_service import AutopilotService

    assert hasattr(AutopilotService, "log"), "the one-line public wrapper TJ-1 asks for"
    service = AutopilotService.__new__(AutopilotService)
    seen: list[str] = []
    service._log = seen.append  # type: ignore[method-assign]
    service.log("Auto mode DESK -> AWAY.")
    assert seen == ["Auto mode DESK -> AWAY."]


def test_the_old_capture_reason_stays_defined_because_old_sidecars_carry_it():
    import market_journal_capture

    assert market_journal_capture.REASON_MODE_FLIP


# ==========================================================================
# 2. the ONE filter, walked by every reader
# ==========================================================================
@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    """A real service over a real temp ledger holding three rows.

    One trader note, one Trade Mentor answer (the trader's own words, so NOT a
    machine row) and one `auto_mode_flip` row of the kind the desk used to
    write. Returns (service, ledger_dir, ids).
    """
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        project_paths, "MARKET_THESES_FILE", tmp_path / "market_theses.jsonl", raising=False
    )

    import market_journal
    from evidence_ledger import default_ledger_dir
    from ui.services.market_journal_service import MarketJournalService

    service = MarketJournalService()
    ids = {}
    ids["trader"] = service.write_entry(
        text=TRADER_TEXT, session_date=SESSION, timeframe=market_journal.TIMEFRAME_D1,
        origin=market_journal.ORIGIN_JOURNAL_PAGE,
    )["entry"]["entry_id"]
    ids["mentor"] = service.write_entry(
        text=MENTOR_TEXT, session_date=SESSION, timeframe=market_journal.TIMEFRAME_M5,
        origin=market_journal.ORIGIN_TRADE_MENTOR,
    )["entry"]["entry_id"]
    ids["machine"] = service.write_entry(
        text=MACHINE_TEXT, session_date=SESSION, timeframe=market_journal.TIMEFRAME_M5,
        symbols=["SPY"], origin=market_journal.ORIGIN_AUTO_MODE_FLIP,
    )["entry"]["entry_id"]
    return service, Path(default_ledger_dir()), ids


def test_the_flag_still_names_only_the_desks_own_hand():
    """A Mentor answer is the trader's words - the desk only chose the moment."""
    import market_journal

    assert market_journal.ORIGIN_AUTO_MODE_FLIP in market_journal.MACHINE_ORIGINS
    assert market_journal.ORIGIN_TRADE_MENTOR not in market_journal.MACHINE_ORIGINS
    assert market_journal.ORIGIN_EXTERNAL_FORECAST not in market_journal.MACHINE_ORIGINS
    assert market_journal.is_machine_entry({"origin": market_journal.ORIGIN_AUTO_MODE_FLIP})
    assert not market_journal.is_machine_entry({"origin": market_journal.ORIGIN_TRADE_MENTOR})


def test_entries_about_drops_the_machine_row_and_keeps_both_of_the_traders(ledger):
    service, _dir, ids = ledger
    got = {str(row.get("entry_id")) for row in service.entries_about(SESSION)}
    assert ids["trader"] in got
    assert ids["mentor"] in got, "a Trade Mentor answer is what the trader said"
    assert ids["machine"] not in got


def test_the_row_is_hidden_and_still_on_disk(ledger):
    """Append-only (plan.md §12.3): a hidden row is filtered by its origin."""
    _service, ledger_dir, _ids = ledger
    lines: list[str] = []
    for path in sorted(ledger_dir.glob("market_journal-*.jsonl")):
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    texts = [json.loads(line).get("text") for line in lines if line.strip()]
    assert MACHINE_TEXT in texts, "nothing is deleted from the ledger"


def test_the_daily_story_has_no_machine_row_in_its_words_or_its_sources(ledger):
    """The nightly narration read these rows back. A story is an AI INPUT, so
    the machine row must be out of `sources` as well as out of `trader_said`."""
    service, _dir, ids = ledger
    story = service.daily_story(SESSION)
    said = {str(row.get("entry_id")) for row in story.trader_said}
    assert ids["trader"] in said
    assert ids["mentor"] in said
    assert ids["machine"] not in said
    assert ids["machine"] not in set(story.sources["entry_ids"])
    assert "Auto mode" not in " ".join(str(row.get("text") or "") for row in story.trader_said)


def test_build_daily_story_drops_a_machine_row_it_is_handed_directly(ledger):
    """Defensively, in the pure function: TJ-4's packs call it with their own
    entry list and must not have to remember the filter."""
    import market_story

    service, _dir, ids = ledger
    handed = service.entries_for()  # every row, machine one included
    assert any(str(row.get("entry_id")) == ids["machine"] for row in handed)

    story = market_story.build_daily_story(SESSION, entries=handed)
    assert ids["machine"] not in {str(row.get("entry_id")) for row in story.trader_said}
    assert ids["machine"] not in set(story.sources["entry_ids"])
    assert len(story.trader_said) == 2


def test_theses_for_never_drafts_a_thesis_from_a_row_nobody_thought(ledger):
    service, _dir, ids = ledger
    rows = service.theses_for(SESSION)
    entry_ids = {str(row.get("entry_id")) for row in rows}
    assert ids["trader"] in entry_ids
    assert ids["machine"] not in entry_ids


def test_the_nightly_rollups_read_the_ledger_without_the_machine_rows(ledger):
    import market_story_rollups

    _service, ledger_dir, ids = ledger
    stories = market_story_rollups._stories_from_journal(ledger_dir)
    matching = [story for story in stories if story.session_date == SESSION]
    assert matching, f"the fixture session is missing from {[s.session_date for s in stories]}"
    story = matching[0]
    assert ids["machine"] not in {str(row.get("entry_id")) for row in story.trader_said}
    assert ids["machine"] not in set(story.sources["entry_ids"])
    assert len(story.trader_said) == 2
