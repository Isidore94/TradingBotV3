"""TJ-14A items 1-3 - What I see / What I expect, and the one accessor.

RED BEFORE THE FIX. On `claude/tj14a-mentor-card`'s base (`8077a758`) the card
has ONE text box, `submit()` refuses an empty read, nothing on the card is
clickable, the `mentor` payload carries `slot_id / prompt_kind / scheduled_at /
responded_at / context` and nothing else, and `market_journal` has no
`prediction_of`.

THE CONTRACT THESE PIN (decision 0021 answer 29; plan.md TJ-10 items 4-6)
------------------------------------------------------------------------
* A DESCRIPTION is never graded. `mentor.observation` is the words; the entry's
  `text` stays those same words so every existing reader keeps working.
* A PREDICTION is a CLICK: `mentor.prediction = {direction, horizon,
  confidence, because, schema}`. It is the only graded thing, and it is forced -
  the file verbs stay grey until each direction row on the card is clicked.
* Horizons are `rest_of_day` (every card) and `next_5_sessions` (the 08:00 and
  12:00 D1 cards only). Directions are `up / down / chop / no_view` for the day
  and `up / down / range / no_view` for the five sessions. `no_view` is a
  COMPLETE answer, and `How sure` disappears on it.
* `read_unchanged` restates the WORDS and never carries a prediction forward.
* Rows written before this packet have no clicked prediction and
  `prediction_of` returns `None` for every one of them. The live file holds FOUR
  vintages, counted on a copy of `market_journal-202609.jsonl` on 2026-09-19:
  28 rows with no `mentor` key at all, 13 with `mentor == {}`, 6 with a mentor
  payload and no context, 22 with a full v1 context. All four are modelled below.

Everything is driven through the REAL widgets offscreen and the REAL journal
service over an `EvidenceLedger` in `tmp_path`; the last test calls the Qt slot
`TradeMentorService.promptDue` is connected to.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

PACIFIC = ZoneInfo("America/Los_Angeles")
#: A plain Monday: 07-12 Pacific slots, D1 at 08 and 12, the trade check at 09.
SESSION = date(2026, 9, 14)

REST_OF_DAY = "rest_of_day"
NEXT_5 = "next_5_sessions"
PREDICTION_SCHEMA = "mentor_prediction_v1"


class _Clock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def __call__(self) -> datetime:
        return self._moment

    def set(self, moment: datetime) -> None:
        self._moment = moment


def _pacific(hour: int, minute: int = 0) -> datetime:
    return datetime(SESSION.year, SESSION.month, SESSION.day, hour, minute, tzinfo=PACIFIC)


def _journal(tmp_path: Path):
    import market_journal
    from evidence_ledger import EvidenceLedger
    from ui.services.market_journal_service import MarketJournalService

    service = MarketJournalService()
    service._ledger = EvidenceLedger(
        stream=market_journal.STREAM,
        schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        directory=tmp_path / "ledger",
    )
    return service


def _card(tmp_path: Path, journal, clock, *, context_service=None):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    return TradeMentorCard(
        journal=journal,
        clock=clock,
        drafts_path=tmp_path / "drafts.json",
        context_service=context_service,
    )


def _slot(hour: int):
    from trade_mentor_schedule import slots_for_session

    for slot in slots_for_session(SESSION):
        if slot.scheduled_at.hour == hour:
            return slot
    raise AssertionError(f"no {hour:02d}:00 slot on {SESSION}")


def _click(card, horizon: str, direction: str) -> None:
    button = card.prediction_button(horizon, direction)
    assert button is not None, f"no {direction} button on the {horizon} row"
    assert button.isVisibleTo(card), f"the {horizon} row is not on this card"
    button.click()
    _app.processEvents()


def _confidence(card, horizon: str, level: str) -> None:
    button = card.confidence_button(horizon, level)
    assert button is not None, f"no {level} confidence button for {horizon}"
    button.click()
    _app.processEvents()


def _mentor_rows(journal) -> list[dict[str, Any]]:
    return [
        row
        for row in journal.entries_for(SESSION.isoformat())
        if row.get("origin") == "trade_mentor"
    ]


# ---------------------------------------------------------------------------
# Item 1 - the forced click
# ---------------------------------------------------------------------------


def test_the_hourly_card_will_not_file_until_the_direction_row_is_clicked(tmp_path):
    """Forced means the button is grey, exactly as TJ-9's Save gate is grey.

    The trader may still say nothing in words - what they may not do is file a
    read with no call in it, because the call is the only thing TJ-10 grades.
    """
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(11, 3)))

    card.show_slot(_slot(11))
    assert card.submit_button.isEnabled() is False
    card.text_box.setPlainText("Range between the overnight levels, no edge yet.")
    _app.processEvents()
    assert card.submit_button.isEnabled() is False, "words are not a prediction"

    _click(card, REST_OF_DAY, "chop")
    assert card.submit_button.isEnabled() is True


def test_no_view_is_a_complete_answer_and_takes_how_sure_away(tmp_path):
    """"I have no call" is an answer, and asking how sure they are of nothing
    is the kind of question that teaches a trader to click past the card."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(11, 3)))

    card.show_slot(_slot(11))
    _click(card, REST_OF_DAY, "no_view")

    assert card.submit_button.isEnabled() is True
    for level in ("low", "medium", "high"):
        button = card.confidence_button(REST_OF_DAY, level)
        assert button is None or not button.isVisibleTo(card), (
            "How sure has nothing to measure on a No view"
        )

    card.submit_button.click()
    _app.processEvents()
    rows = _mentor_rows(journal)
    assert len(rows) == 1
    prediction = rows[0]["mentor"]["prediction"]
    assert prediction["direction"] == "no_view"
    assert prediction["horizon"] == REST_OF_DAY
    assert not str(prediction.get("confidence") or "")


def test_an_hourly_card_never_offers_the_five_session_call(tmp_path):
    """The 11:00 card is an M5 read. A swing call there would be a click the
    trader makes six times a day about the same five sessions."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(11, 3)))

    card.show_slot(_slot(11))
    for direction in ("up", "down", "range", "no_view"):
        button = card.prediction_button(NEXT_5, direction)
        assert button is None or not button.isVisibleTo(card)


def test_a_card_answered_with_clicks_and_no_words_is_a_complete_answer(tmp_path):
    """The whole point of the split: the trader who has nothing to say still
    leaves a graded call behind. `text` stays the OBSERVATION, so it is empty
    here - it is not filled in with a sentence nobody wrote."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(11, 3)))

    card.show_slot(_slot(11))
    _click(card, REST_OF_DAY, "down")
    _confidence(card, REST_OF_DAY, "high")
    card.submit_button.click()
    _app.processEvents()

    rows = _mentor_rows(journal)
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == ""
    assert row["mentor"]["observation"] == ""
    assert row["mentor"]["prediction"]["direction"] == "down"
    assert row["mentor"]["prediction"]["confidence"] == "high"


# ---------------------------------------------------------------------------
# Item 2 - storage
# ---------------------------------------------------------------------------


def test_what_i_see_and_what_i_expect_are_stored_as_separate_keys(tmp_path):
    """Two parts, never one field. The words describe NOW; the click is the call."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(11, 3)))

    card.show_slot(_slot(11))
    card.text_box.setPlainText("SPY is grinding under VWAP on falling volume.")
    _click(card, REST_OF_DAY, "down")
    _confidence(card, REST_OF_DAY, "medium")
    because = card.because_box(REST_OF_DAY)
    assert because is not None, "an optional one-line Because… belongs on the row"
    because.setText("lower highs since the open")
    _app.processEvents()
    card.submit_button.click()
    _app.processEvents()

    row = _mentor_rows(journal)[0]
    assert row["text"] == "SPY is grinding under VWAP on falling volume."
    assert row["mentor"]["observation"] == "SPY is grinding under VWAP on falling volume."
    assert row["mentor"]["prediction"] == {
        "direction": "down",
        "horizon": REST_OF_DAY,
        "confidence": "medium",
        "because": "lower highs since the open",
        "schema": PREDICTION_SCHEMA,
    }


def test_a_d1_card_stores_the_day_call_and_the_five_session_call_on_their_own_rows(tmp_path):
    """One card, two timeframes, two entries - the shape the live file already
    has (seven `m5_d1` slots wrote two rows each in September). Each entry keeps
    the call that belongs to ITS horizon; a five-session call filed against the
    M5 row would be graded at the wrong moment."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, _Clock(_pacific(8, 4)))

    card.show_slot(_slot(8))
    card.text_box.setPlainText("Opening drive faded into the first 30 minutes.")
    card.d1_box.setPlainText("Still under the 20-day; lower highs since Tuesday.")
    _click(card, REST_OF_DAY, "chop")
    _confidence(card, REST_OF_DAY, "low")
    assert card.submit_button.isEnabled() is False, "the D1 row is still unanswered"
    _click(card, NEXT_5, "range")
    _confidence(card, NEXT_5, "high")
    assert card.submit_button.isEnabled() is True
    card.submit_button.click()
    _app.processEvents()

    rows = _mentor_rows(journal)
    assert [row["timeframe"] for row in rows] == ["M5", "D1"]
    m5, d1 = rows
    assert m5["mentor"]["prediction"]["horizon"] == REST_OF_DAY
    assert m5["mentor"]["prediction"]["direction"] == "chop"
    assert m5["mentor"]["prediction"]["confidence"] == "low"
    assert d1["mentor"]["prediction"]["horizon"] == NEXT_5
    assert d1["mentor"]["prediction"]["direction"] == "range"
    assert d1["mentor"]["prediction"]["confidence"] == "high"
    assert m5["mentor"]["observation"] == "Opening drive faded into the first 30 minutes."
    assert d1["mentor"]["observation"] == "Still under the 20-day; lower highs since Tuesday."


def test_read_unchanged_restates_the_words_and_still_demands_a_fresh_click(tmp_path):
    """"My view has not changed" is a statement about the WORDS. Copying the
    09:00 call onto the 11:00 row would manufacture a second graded prediction
    the trader never made - and the two hours would always agree."""
    journal = _journal(tmp_path)
    clock = _Clock(_pacific(9, 12))
    card = _card(tmp_path, journal, clock)

    card.show_slot(_slot(9))
    card.text_box.setPlainText("Buyers defending the overnight low.")
    _click(card, REST_OF_DAY, "up")
    _confidence(card, REST_OF_DAY, "high")
    because = card.because_box(REST_OF_DAY)
    because.setText("third hold of the same level")
    card.submit_button.click()
    _app.processEvents()
    first = _mentor_rows(journal)[-1]

    clock.set(_pacific(11, 3))
    card.show_slot(_slot(11), previous=first)
    assert card.unchanged_button.isEnabled() is False, (
        "a reaffirmed read still needs this hour's call"
    )
    _click(card, REST_OF_DAY, "chop")
    assert card.unchanged_button.isEnabled() is True
    card.unchanged_button.click()
    _app.processEvents()

    rows = _mentor_rows(journal)
    assert len(rows) == 2
    second = rows[-1]
    assert second["text"] == first["text"]
    assert second["reaffirms"] == first["entry_id"]
    assert second["mentor"]["prediction"]["direction"] == "chop"
    assert second["mentor"]["prediction"]["because"] == ""


def test_an_unsaved_click_survives_the_card_being_put_away(tmp_path):
    """Drafts keep unsaved clicks exactly as they keep unsaved text - and across
    a restart, because that is what the drafts FILE is for."""
    journal = _journal(tmp_path)
    clock = _Clock(_pacific(11, 3))
    card = _card(tmp_path, journal, clock)
    eleven = _slot(11)

    card.show_slot(eleven)
    card.text_box.setPlainText("half a thought about the ")
    _click(card, REST_OF_DAY, "down")
    _confidence(card, REST_OF_DAY, "low")
    card.hide_card()

    restarted = _card(tmp_path, journal, clock)
    restarted.show_slot(eleven)
    _app.processEvents()

    assert restarted.text_box.toPlainText() == "half a thought about the "
    assert restarted.prediction_button(REST_OF_DAY, "down").isChecked() is True
    assert restarted.submit_button.isEnabled() is True
    assert _mentor_rows(journal) == [], "a draft is never a read"


# ---------------------------------------------------------------------------
# Item 3 - the one accessor
# ---------------------------------------------------------------------------


def _row(mentor: Any, *, present: bool = True) -> dict[str, Any]:
    row = {
        "event_type": "entry",
        "entry_id": "mj-2026-09-14-abc",
        "session_date": "2026-09-14",
        "created_at": "2026-09-14T16:12:41+00:00",
        "origin": "trade_mentor",
        "text": "SPY looks weak to me and I think it breaks down",
        "timeframe": "M5",
    }
    if present:
        row["mentor"] = mentor
    return row


def test_prediction_of_reads_a_clicked_row_and_refuses_every_older_vintage():
    """Four vintages live in the September file, and none of them is a click.

    An extracted stance from an old row is TJ-10's `extracted` lane and is never
    pooled with a clicked one (decision 0021 answer 29), so this accessor must
    say `None` rather than guess.
    """
    from market_journal import prediction_of

    # 28 live rows: no `mentor` key at all (pre-WISHLIST-10J).
    assert prediction_of(_row(None, present=False)) is None
    # 13 live rows: the key PRESENT and EMPTY.
    assert prediction_of(_row({})) is None
    # 6 live rows: a mentor payload with no context and no prediction.
    assert prediction_of(_row({"slot_id": "2026-09-14-1100-m5", "prompt_kind": "m5"})) is None
    # 22 live rows: a full v1 context, still no click.
    assert (
        prediction_of(
            _row(
                {
                    "slot_id": "2026-09-14-1100-m5",
                    "prompt_kind": "m5",
                    "context": {"schema": "trade_mentor_context_v1", "readings": []},
                }
            )
        )
        is None
    )
    # And the shape a half-written row could take: present and empty.
    assert prediction_of(_row({"slot_id": "x", "prediction": {}})) is None

    clicked = prediction_of(
        _row(
            {
                "slot_id": "2026-09-14-1100-m5",
                "observation": "SPY looks weak to me and I think it breaks down",
                "prediction": {
                    "direction": "down",
                    "horizon": REST_OF_DAY,
                    "confidence": "medium",
                    "because": "lower highs",
                    "schema": PREDICTION_SCHEMA,
                },
            }
        )
    )
    assert clicked is not None
    assert clicked.direction == "down"
    assert clicked.horizon == REST_OF_DAY
    assert clicked.confidence == "medium"
    assert clicked.because == "lower highs"


def test_a_reader_handed_a_clicked_row_never_looks_at_the_words():
    """The guard. `extract_thesis` read *"D1 SPY is still downtrending"* as
    `unstated`; a reader that falls back to the text for a row that HAS a click
    would quietly re-introduce that reading. This entry raises if the words are
    touched at all."""
    from market_journal import prediction_of

    class _WordsAreOffLimits(dict):
        def __getitem__(self, key):
            if key in ("text", "observation"):
                raise AssertionError(f"a row with a click must not read {key!r}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in ("text", "observation"):
                raise AssertionError(f"a row with a click must not read {key!r}")
            return super().get(key, default)

    entry = _WordsAreOffLimits(
        {
            "event_type": "entry",
            "entry_id": "mj-2026-09-14-guard",
            "session_date": "2026-09-14",
            "origin": "trade_mentor",
            "text": "D1 SPY is still downtrending",
            "mentor": {
                "prediction": {
                    "direction": "up",
                    "horizon": REST_OF_DAY,
                    "confidence": "low",
                    "because": "",
                    "schema": PREDICTION_SCHEMA,
                }
            },
        }
    )
    assert prediction_of(entry).direction == "up"


# ---------------------------------------------------------------------------
# The LIVE slot - the seam `promptDue` is connected to
# ---------------------------------------------------------------------------


def test_the_live_prompt_slot_puts_the_forced_click_on_the_card(tmp_path, monkeypatch):
    """`MainWindow._show_trade_mentor_prompt` is what the running desk calls.

    A card built only in a unit test is a card the trader never sees; TJ-9's
    review round found a dead live path exactly here. The 08:00 slot must come
    up with BOTH direction rows and a Submit that is still grey.
    """
    import journal_store as journal_store_module
    from journal_store import JournalStore
    from ui.app import MainWindow
    from ui.state import UiState

    store = JournalStore(Path(tmp_path) / "journal.sqlite3")
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)

    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        window._show_trade_mentor_prompt(_slot(8))
        _app.processEvents()
        card = window.trading_panel.alert_center.chart_review.mentor_card

        assert card.prediction_button(REST_OF_DAY, "chop").isVisibleTo(card)
        assert card.prediction_button(NEXT_5, "range").isVisibleTo(card)
        assert card.submit_button.isEnabled() is False

        window._show_trade_mentor_prompt(_slot(11))
        _app.processEvents()
        assert card.prediction_button(REST_OF_DAY, "chop").isVisibleTo(card)
        five = card.prediction_button(NEXT_5, "range")
        assert five is None or not five.isVisibleTo(card)
    finally:
        window.close()
