"""TJ-14A fix round - a row's timeframe and its prediction horizon always agree.

RED BEFORE THE FIX (proven at `efaec622`). The reviewer's reproduction: an
answered 08:00 `m5_d1` card writes an M5 row then a D1 row, so `rows[-1]` is the
D1 one; `MainWindow._previous_mentor_read` handed that row to the 09:00 card as
"your last read"; `read_unchanged` kept `previous.timeframe` (`D1`) while
`_previous_horizon` silently fell back to `rest_of_day` - and filed a D1
entry carrying a rest-of-day call. A row true of neither timeframe, permanent in
an append-only ledger, and poison for TJ-16's calibration by horizon.

THE RULE THESE PIN
------------------
* the WRITER refuses a mismatched pair (`build_entry` raises,
  `is_publishable` refuses), so no future caller can file one;
* the host supplies the latest read of EACH timeframe, not the latest row;
* `Read unchanged` reaffirms PER TIMEFRAME - an `m5` card files one M5 row with
  this hour's rest-of-day call, an `m5_d1` card files both, each with its own
  fresh click;
* a timeframe the card shows with no earlier read makes the verb unavailable and
  the button SAYS SO. There is no silent fallback left to fall back on.

The first two tests drive the REAL seam: `MainWindow._show_trade_mentor_prompt`
and the real `_previous_mentor_read`, over a scratch journal in `tmp_path`.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

SESSION = date(2026, 9, 14)
REST_OF_DAY = "rest_of_day"
NEXT_5 = "next_5_sessions"


def _slot(hour: int):
    from trade_mentor_schedule import slots_for_session

    for slot in slots_for_session(SESSION):
        if slot.scheduled_at.hour == hour:
            return slot
    raise AssertionError(f"no {hour:02d}:00 slot on {SESSION}")


def _scratch_journal(tmp_path: Path, monkeypatch):
    """The ONE store both the card and `_previous_mentor_read` reach."""
    import market_journal
    from evidence_ledger import EvidenceLedger
    from ui.services import market_journal_service as service_module

    service = service_module.MarketJournalService()
    service._ledger = EvidenceLedger(
        stream=market_journal.STREAM,
        schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        directory=tmp_path / "ledger",
    )
    monkeypatch.setattr(service_module, "shared_journal_service", lambda: service)
    return service


def _window(tmp_path: Path, monkeypatch):
    import journal_store as journal_store_module
    from journal_store import JournalStore
    from ui.app import MainWindow
    from ui.state import UiState

    store = JournalStore(Path(tmp_path) / "journal.sqlite3")
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)
    return MainWindow(UiState(workspace_mode="workspace"))


def _rows(journal):
    return [
        row
        for row in journal.entries_for(SESSION.isoformat())
        if row.get("origin") == "trade_mentor"
    ]


def _answer(card, horizon: str, direction: str, confidence: str = "medium"):
    card.prediction_button(horizon, direction).click()
    if direction != "no_view":
        card.confidence_button(horizon, confidence).click()
    _app.processEvents()


# ---------------------------------------------------------------------------
# the blocker, through the live seam
# ---------------------------------------------------------------------------


def test_read_unchanged_after_a_d1_card_files_an_m5_row_with_an_m5_call(
    tmp_path, monkeypatch
):
    """The reviewer's exact sequence, end to end on the desk's own slot.

    At `efaec622` the second row came back `timeframe=D1 horizon=rest_of_day
    text='D1 words'`. It must come back M5, with the M5 words and the M5 call.
    """
    journal = _scratch_journal(tmp_path, monkeypatch)
    window = _window(tmp_path, monkeypatch)
    try:
        window._show_trade_mentor_prompt(_slot(8))
        _app.processEvents()
        card = window.trading_panel.alert_center.chart_review.mentor_card
        card.text_box.setPlainText("M5 words")
        card.d1_box.setPlainText("D1 words")
        _answer(card, REST_OF_DAY, "up", "high")
        _answer(card, NEXT_5, "range", "low")
        assert card.submit()["ok"] is True

        first = _rows(journal)
        assert [(row["timeframe"], row["text"]) for row in first] == [
            ("M5", "M5 words"),
            ("D1", "D1 words"),
        ]

        window._show_trade_mentor_prompt(_slot(9))
        _app.processEvents()
        _answer(card, REST_OF_DAY, "down", "medium")
        assert card.unchanged_button.isEnabled() is True
        assert card.read_unchanged()["ok"] is True

        rows = _rows(journal)
        assert len(rows) == 3
        filed = rows[-1]
        assert filed["timeframe"] == "M5"
        assert filed["text"] == "M5 words", "the M5 row reaffirms the M5 words"
        assert filed["mentor"]["prediction"]["horizon"] == REST_OF_DAY
        assert filed["mentor"]["prediction"]["direction"] == "down"
        assert filed["reaffirms"] == first[0]["entry_id"]
    finally:
        window.close()


def test_a_d1_card_reaffirms_each_timeframe_from_its_own_last_read(
    tmp_path, monkeypatch
):
    """08:00 answers both, 11:00 answers M5, 12:00 reaffirms BOTH: the M5 row
    from 11:00 and the D1 row from 08:00, each with its own fresh click."""
    journal = _scratch_journal(tmp_path, monkeypatch)
    window = _window(tmp_path, monkeypatch)
    try:
        card = window.trading_panel.alert_center.chart_review.mentor_card

        window._show_trade_mentor_prompt(_slot(8))
        _app.processEvents()
        card.text_box.setPlainText("eight M5")
        card.d1_box.setPlainText("eight D1")
        _answer(card, REST_OF_DAY, "up", "high")
        _answer(card, NEXT_5, "up", "high")
        card.submit()

        window._show_trade_mentor_prompt(_slot(11))
        _app.processEvents()
        card.text_box.setPlainText("eleven M5")
        _answer(card, REST_OF_DAY, "chop", "low")
        card.submit()

        window._show_trade_mentor_prompt(_slot(12))
        _app.processEvents()
        _answer(card, REST_OF_DAY, "down", "medium")
        assert card.unchanged_button.isEnabled() is False, "the D1 row has no call yet"
        _answer(card, NEXT_5, "range", "high")
        assert card.unchanged_button.isEnabled() is True
        assert card.read_unchanged()["ok"] is True

        rows = _rows(journal)
        m5_row, d1_row = rows[-2], rows[-1]
        assert (m5_row["timeframe"], m5_row["text"]) == ("M5", "eleven M5")
        assert m5_row["mentor"]["prediction"]["horizon"] == REST_OF_DAY
        assert m5_row["mentor"]["prediction"]["direction"] == "down"
        assert (d1_row["timeframe"], d1_row["text"]) == ("D1", "eight D1")
        assert d1_row["mentor"]["prediction"]["horizon"] == NEXT_5
        assert d1_row["mentor"]["prediction"]["direction"] == "range"
        assert d1_row["reaffirms"] == rows[1]["entry_id"], "the 08:00 D1 row"
    finally:
        window.close()


def test_the_first_card_of_the_day_offers_no_read_unchanged(tmp_path, monkeypatch):
    """Nothing to reaffirm, and the button says which timeframe is missing -
    never a silent substitute from another timeframe or another session."""
    _scratch_journal(tmp_path, monkeypatch)
    window = _window(tmp_path, monkeypatch)
    try:
        window._show_trade_mentor_prompt(_slot(8))
        _app.processEvents()
        card = window.trading_panel.alert_center.chart_review.mentor_card
        _answer(card, REST_OF_DAY, "up", "high")
        _answer(card, NEXT_5, "up", "high")

        assert card.submit_button.isEnabled() is True
        assert card.unchanged_button.isEnabled() is False
        tip = card.unchanged_button.toolTip().lower()
        assert "no earlier" in tip and "m5" in tip and "d1" in tip
        refused = card.read_unchanged()
        assert refused["ok"] is False and "no earlier" in refused["reason"]
    finally:
        window.close()


def test_an_m5_card_will_not_reaffirm_a_d1_only_history(tmp_path, monkeypatch):
    """The exact hole the fallback used to fill. A session whose only read is a
    D1 one leaves an `m5` card with nothing to reaffirm, and it says so rather
    than restating D1 words under an M5 heading."""
    from ui.widgets.trade_mentor_card import TradeMentorCard

    _scratch_journal(tmp_path, monkeypatch)
    card = TradeMentorCard(journal=None, drafts_path=tmp_path / "drafts.json")
    d1_only = {
        "entry_id": "mj-2026-09-14-d1",
        "timeframe": "D1",
        "text": "Still under the 20-day.",
        "origin": "trade_mentor",
    }
    card.show_slot(_slot(11), previous=d1_only)
    _answer(card, REST_OF_DAY, "down", "low")

    assert card.unchanged_button.isEnabled() is False
    assert "M5" in card.unchanged_button.toolTip()
    assert card.read_unchanged()["ok"] is False


# ---------------------------------------------------------------------------
# the writer refuses the pair outright
# ---------------------------------------------------------------------------


def test_the_writer_refuses_a_d1_row_carrying_a_rest_of_day_call():
    """Belt and braces at the store: the card can no longer build the pair, and
    if any future caller does, it is an exception and a refusal - not a row."""
    import market_journal

    call = market_journal.build_prediction(
        direction="down", horizon=REST_OF_DAY, confidence="high"
    )
    with pytest.raises(market_journal.PredictionTimeframeError) as raised:
        market_journal.build_entry(
            text="D1 words",
            session_date=SESSION.isoformat(),
            timeframe="D1",
            origin=market_journal.ORIGIN_TRADE_MENTOR,
            mentor={"prediction": call},
        )
    assert "next_5_sessions" in str(raised.value)

    # And a row assembled as a dict literal, which never calls `build_entry`.
    ok, reason = market_journal.is_publishable(
        {
            "text": "D1 words",
            "session_date": SESSION.isoformat(),
            "timeframe": "D1",
            "mentor": {"prediction": call},
        }
    )
    assert ok is False and "next_5_sessions" in reason

    # The pairs that DO agree still build, both ways round.
    for timeframe, horizon in (("M5", REST_OF_DAY), ("D1", NEXT_5)):
        entry = market_journal.build_entry(
            text="words",
            session_date=SESSION.isoformat(),
            timeframe=timeframe,
            origin=market_journal.ORIGIN_TRADE_MENTOR,
            mentor={
                "prediction": market_journal.build_prediction(
                    direction="up", horizon=horizon, confidence="low"
                )
            },
        )
        assert market_journal.prediction_of(entry).horizon == horizon
        assert market_journal.is_publishable(entry)[0] is True


def test_the_host_hands_the_card_the_latest_read_of_each_timeframe(
    tmp_path, monkeypatch
):
    """`_previous_mentor_read` used to answer `rows[-1]`, which is the D1 row on
    every day the 08:00 card was answered."""
    journal = _scratch_journal(tmp_path, monkeypatch)
    window = _window(tmp_path, monkeypatch)
    try:
        import market_journal

        for timeframe, horizon, text in (
            ("M5", REST_OF_DAY, "first M5"),
            ("M5", REST_OF_DAY, "second M5"),
            ("D1", NEXT_5, "the D1 read"),
        ):
            journal.write_entry(
                text=text,
                session_date=SESSION.isoformat(),
                timeframe=timeframe,
                origin=market_journal.ORIGIN_TRADE_MENTOR,
                mentor={
                    "prediction": market_journal.build_prediction(
                        direction="up", horizon=horizon, confidence="low"
                    )
                },
            )
        answer = window._previous_mentor_read(SESSION.isoformat())
        assert set(answer) == {"M5", "D1"}
        assert answer["M5"]["text"] == "second M5"
        assert answer["D1"]["text"] == "the D1 read"
    finally:
        window.close()
