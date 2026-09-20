"""TJ-14A lead decision 4 - a clicks-only row reads back everywhere.

ADDED BY THE BUILDER. The tester's files pin the WRITE (`text == ""`, the
prediction on `mentor.prediction`, `is_publishable` relaxed for that row and
nothing else). These pin the READ: every surface that prints journal `text` now
meets an entry whose text is empty ON PURPOSE, and not one of them may show a
blank row, invent a sentence, or raise.

The readers named by the lead's decision 4, each driven here:

* `market_journal.is_publishable` - relaxed for a clicked row ONLY;
* `MarketJournalService.entries_about` - the ONE filter every trader-facing
  read inherits;
* `market_story.build_daily_story` - the AI's day pack input;
* `market_thesis.extract_thesis` - the old extraction lane, which must read a
  wordless row as `unstated` rather than falling over;
* Day Review's entry list and reader pane;
* the bounded AI evidence package.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SESSION = "2026-09-14"
REST_OF_DAY = "rest_of_day"


def _clicked_row(**overrides):
    """The row a card answered with clicks and no words actually writes."""
    import market_journal

    entry = market_journal.build_entry(
        text="",
        session_date=SESSION,
        timeframe="M5",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        now=datetime(2026, 9, 14, 18, 12, 41, tzinfo=timezone.utc),
        mentor={
            "slot_id": "2026-09-14-1100-m5",
            "prompt_kind": "m5",
            "observation": "",
            "prediction": market_journal.build_prediction(
                direction="down", horizon=REST_OF_DAY, confidence="high",
                because="lower highs since the open",
            ),
        },
    )
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# the write side of the relaxation
# ---------------------------------------------------------------------------


def test_only_a_clicked_row_may_be_filed_with_no_words():
    """The door is exactly one row wide.

    A blank journal entry is still refused: a store full of blanks makes the
    record look denser than the thinking behind it, and that rule loses nothing
    here because a click IS the thinking.
    """
    import market_journal

    ok, reason = market_journal.is_publishable(_clicked_row())
    assert ok is True and reason == ""

    empty = market_journal.build_entry(
        text="", session_date=SESSION, origin=market_journal.ORIGIN_DESK_TAB
    )
    ok, reason = market_journal.is_publishable(empty)
    assert ok is False and "empty" in reason

    # A mentor row whose prediction never got a direction is not a click.
    half = _clicked_row()
    half["mentor"] = {"slot_id": "x", "prediction": {"horizon": REST_OF_DAY}}
    assert market_journal.is_publishable(half)[0] is False


def test_two_wordless_rows_from_one_card_do_not_share_an_identity():
    """A D1 card filed with two calls and no words writes two rows in the same
    second with the same empty text. `entry_id` is what every join over this
    store uses, so they may not collide - and a row WITH words keeps exactly
    the id it would have had before this packet."""
    import market_journal

    moment = datetime(2026, 9, 14, 15, 4, 0, tzinfo=timezone.utc)

    def row(timeframe: str, horizon: str):
        return market_journal.build_entry(
            text="", session_date=SESSION, timeframe=timeframe,
            origin=market_journal.ORIGIN_TRADE_MENTOR, now=moment,
            mentor={
                "prediction": market_journal.build_prediction(
                    direction="up", horizon=horizon, confidence="low"
                )
            },
        )

    m5 = row("M5", REST_OF_DAY)
    d1 = row("D1", "next_5_sessions")
    assert m5["entry_id"] != d1["entry_id"]

    worded = market_journal.build_entry(
        text="SPY is holding the open.", session_date=SESSION, timeframe="M5",
        origin=market_journal.ORIGIN_TRADE_MENTOR, now=moment,
    )
    created = worded["created_at"]
    assert worded["entry_id"] == market_journal.entry_id(
        SESSION, created, "SPY is holding the open."
    ), "an entry with words is never salted; its id is unchanged"


def test_the_call_is_worded_once_and_a_no_view_says_so():
    import market_journal

    assert market_journal.prediction_line(_clicked_row()) == (
        "Rest of day: Down (high confidence) - lower highs since the open"
    )
    quiet = _clicked_row()
    quiet["mentor"] = {
        "prediction": market_journal.build_prediction(
            direction="no_view", horizon=REST_OF_DAY, confidence="high"
        )
    }
    assert market_journal.prediction_line(quiet) == "Rest of day: No view"
    assert market_journal.prediction_line({"text": "words only"}) == ""


# ---------------------------------------------------------------------------
# the readers
# ---------------------------------------------------------------------------


def test_the_session_reader_and_the_day_story_keep_a_wordless_answer(tmp_path):
    """`entries_about` is the ONE filter every trader-facing read inherits, and
    the story is the AI's day pack input. Neither may drop a row because its
    words are empty, and the story carries the CALL so no reader downstream has
    to re-derive it from words that are not there."""
    import market_journal
    from evidence_ledger import EvidenceLedger
    from market_story import build_daily_story
    from ui.services.market_journal_service import MarketJournalService

    service = MarketJournalService()
    service._ledger = EvidenceLedger(
        stream=market_journal.STREAM,
        schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        directory=tmp_path / "ledger",
    )
    moment = datetime(2026, 9, 14, 18, 12, 41, tzinfo=timezone.utc)
    written = service.write_entry(
        text="",
        session_date=SESSION,
        timeframe="M5",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        now=moment,
        mentor={
            "observation": "",
            "prediction": market_journal.build_prediction(
                direction="chop", horizon=REST_OF_DAY, confidence="low"
            ),
        },
    )
    assert written["ok"] is True, written.get("reason")

    rows = service.entries_about(SESSION)
    assert len(rows) == 1
    assert rows[0]["text"] == ""
    assert market_journal.prediction_of(rows[0]).direction == "chop"

    story = build_daily_story(SESSION, entries=rows)
    assert len(story.trader_said) == 1
    said = story.trader_said[0]
    assert said["text"] == "", "nobody wrote a sentence, so none is invented"
    assert said["prediction"]["direction"] == "chop"
    assert said["prediction"]["horizon"] == REST_OF_DAY
    assert "No note was written" not in " ".join(story.notes)


def test_the_old_extraction_lane_reads_a_wordless_row_as_unstated():
    """`extract_thesis` is the `extracted` lane TJ-10 keeps for the history and
    for notes typed outside a card. Handed a row with no words it must say
    `unstated` - never guess, never raise."""
    from market_thesis import extract_thesis

    draft = extract_thesis(_clicked_row())
    assert draft.stance == "unstated"
    # Its own named absence, not a sentence and not a guess.
    assert draft.claim == "unstated"
    assert draft.spans == {}


def test_day_review_shows_the_call_where_the_words_would_have_been(tmp_path):
    """The list row and the reader pane, driven as widgets. A blank line on
    this page would read as "they said nothing", which is the opposite of what
    a clicked call means."""
    pytest.importorskip("PySide6", reason="Day Review is Qt")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from ui.panels.day_review_panel import DayReviewPanel

    panel = DayReviewPanel()
    try:
        panel._render_entries([_clicked_row()])
        app.processEvents()
        assert panel.entries.count() == 1
        label = panel.entries.item(0).text()
        assert "Rest of day: Down" in label
        assert not label.rstrip().endswith("·"), "never a blank row"

        panel.entries.setCurrentRow(0)
        app.processEvents()
        assert "Rest of day: Down" in panel.entry_reader.toPlainText()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_the_bounded_ai_package_still_carries_a_wordless_answer(tmp_path):
    """The AI packs read `text`. A clicks-only row reaches them as a real row
    with its prediction, not as a banner and not as a dropped line."""
    import market_journal
    from ai_summary import build_evidence_package
    from evidence_ledger import EvidenceLedger
    from ui.services.market_journal_service import MarketJournalService

    service = MarketJournalService()
    service._ledger = EvidenceLedger(
        stream=market_journal.STREAM,
        schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        directory=tmp_path / "ledger",
    )
    moment = datetime(2026, 9, 14, 18, 12, 41, tzinfo=timezone.utc)
    service.write_entry(
        text="",
        session_date=SESSION,
        timeframe="M5",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        now=moment,
        mentor={
            "observation": "",
            "prediction": market_journal.build_prediction(
                direction="up", horizon=REST_OF_DAY, confidence="medium"
            ),
        },
    )
    package = build_evidence_package(
        ["market_journal"],
        source_overrides={"journal.entries": next((tmp_path / "ledger").glob("*.jsonl"))},
        now=datetime(2026, 9, 14, 20, tzinfo=timezone.utc),
        session_date=SESSION,
        budget_chars=6_000,
    )
    source = next(row for row in package["sources"] if row["source_id"] == "journal.entries")
    row = next(item for item in source["content"] if isinstance(item, dict))
    assert row["text"] == ""
    assert row["mentor"]["prediction"]["direction"] == "up"
