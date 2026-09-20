"""TJ-14A - the guards the BUILDER added, on the lead's decisions 1, 2, 3 and 5.

ADDED BY THE BUILDER, none of them a restatement of a tester assertion:

1. `How sure` is FORCED whenever the direction is not `No view` (lead decision
   1), asked here from BOTH ends - grey after a bare direction, open at once on
   `No view`.
2. The gate lives in `submit()` and `read_unchanged()` THEMSELVES, not only on
   the buttons (lead decision 2): a caller that never touches a widget still
   cannot file an hourly answer without its clicks, and TJ-9's forced
   TRADE-CHECK section keeps its own separate gate.
3. `internals_bars_at` is the thin loader beside the pure `internals_at`, and
   it reads the durable tape and the same daily cache the live service reads
   (lead decision 3). Driven against a STAGED store, never the live one.
5. An `m5_d1` card writes BOTH rows once both calls are clicked - words or no
   words (lead decision 5).
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
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

import tj14a_support as fixture  # noqa: E402

_app = QApplication.instance() or QApplication([])

PACIFIC = ZoneInfo("America/Los_Angeles")
SESSION = date(2026, 9, 14)
REST_OF_DAY = "rest_of_day"
NEXT_5 = "next_5_sessions"


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


def _card(tmp_path: Path, journal, hour: int = 11, minute: int = 3):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    return TradeMentorCard(
        journal=journal,
        clock=lambda: datetime(2026, 9, 14, hour, minute, tzinfo=PACIFIC),
        drafts_path=tmp_path / "drafts.json",
    )


def _slot(hour: int):
    from trade_mentor_schedule import slots_for_session

    for slot in slots_for_session(SESSION):
        if slot.scheduled_at.hour == hour:
            return slot
    raise AssertionError(f"no {hour:02d}:00 slot")


def _mentor_rows(journal):
    return [
        row
        for row in journal.entries_for(SESSION.isoformat())
        if row.get("origin") == "trade_mentor"
    ]


# ---------------------------------------------------------------------------
# decision 1 - How sure is forced, and No view takes it away
# ---------------------------------------------------------------------------


def test_how_sure_is_forced_on_a_real_call_and_absent_on_no_view(tmp_path):
    """A prediction IS direction, horizon and confidence (decision 0021 answer
    29). TJ-16 reads calibration BY confidence, so a call filed without one
    could never join it - and asking how sure somebody is of nothing is the
    question that teaches a trader to click past the card."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal)
    card.show_slot(_slot(11))

    assert card.submit_button.isEnabled() is False
    card.prediction_button(REST_OF_DAY, "up").click()
    _app.processEvents()
    assert card.submit_button.isEnabled() is False, "direction alone is half a call"
    assert card.confidence_button(REST_OF_DAY, "low").isVisibleTo(card)
    card.confidence_button(REST_OF_DAY, "low").click()
    _app.processEvents()
    assert card.submit_button.isEnabled() is True

    # Changing the call to No view withdraws the confidence with it: a stored
    # "No view, high confidence" would be graded by TJ-16 as a confident call.
    card.prediction_button(REST_OF_DAY, "no_view").click()
    _app.processEvents()
    assert card.submit_button.isEnabled() is True
    assert not card.confidence_button(REST_OF_DAY, "low").isVisibleTo(card)
    card.submit()
    assert _mentor_rows(journal)[0]["mentor"]["prediction"]["confidence"] == ""


# ---------------------------------------------------------------------------
# decision 2 - the gate is in the methods, not only on the buttons
# ---------------------------------------------------------------------------


def test_no_code_path_files_an_hourly_answer_without_its_clicks(tmp_path):
    """The buttons are grey, but `submit()` and `read_unchanged()` are reachable
    from a shortcut, a host and any future caller. Forced means the REFUSAL is
    in the verb, and it says which row is still open."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal)
    card.show_slot(_slot(11))
    card.text_box.setPlainText("Words, and no call in them at all.")

    refused = card.submit()
    assert refused["ok"] is False
    assert "call" in refused["reason"].lower()
    assert _mentor_rows(journal) == []

    card.prediction_button(REST_OF_DAY, "down").click()
    half = card.submit()
    assert half["ok"] is False and "how sure" in half["reason"].lower()
    assert _mentor_rows(journal) == []

    card.confidence_button(REST_OF_DAY, "medium").click()
    assert card.submit()["ok"] is True
    assert len(_mentor_rows(journal)) == 1

    # `read_unchanged` is the same rule: it files a graded call too.
    second = _card(tmp_path, journal, hour=12, minute=4)
    second.show_slot(_slot(12), previous=_mentor_rows(journal)[-1])
    assert second.read_unchanged()["ok"] is False
    assert len(_mentor_rows(journal)) == 1


def test_the_forced_trade_check_keeps_its_own_separate_gate(tmp_path):
    """TJ-9's section sits in the same card and OUTSIDE this gate. Its Save
    button answers to the trade questions alone, and a missing prediction must
    not grey it out - nor may a clicked prediction arm it."""
    import trade_mentor_trade_check as check

    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, hour=9, minute=12)
    card.show_slot(_slot(9))
    task = check.TradeCheckTask(
        reviewed_session="2026-09-11",
        journal_ready=True,
        trades=(
            check.TradeQuestion(
                trade_id="T1", symbol="NVDA", direction="LONG", missing=("stop",)
            ),
        ),
    )
    card.set_trade_check(task)
    _app.processEvents()

    assert card.save_answers_button.isEnabled() is False
    card.prediction_button(REST_OF_DAY, "up").click()
    card.confidence_button(REST_OF_DAY, "high").click()
    _app.processEvents()
    assert card.save_answers_button.isEnabled() is False, "the trade gate is its own"
    assert card.submit_button.isEnabled() is True

    combo = card._answer_inputs["T1"]["stop"][0]
    combo.setCurrentIndex(combo.findData(check.ANSWER_STATES[0]))
    _app.processEvents()
    assert card.save_answers_button.isEnabled() is True
    assert card.open_answers_session() == str(_slot(9).session)


# ---------------------------------------------------------------------------
# decision 5 - a D1 card always writes both rows
# ---------------------------------------------------------------------------


def test_a_d1_card_writes_both_rows_with_no_words_at_all(tmp_path):
    """Words or no words: the M5 row carries the rest-of-day call and the D1
    row the five-session one. Before TJ-14A an empty box skipped its entry, so
    a wordless swing call would have had nowhere to live."""
    journal = _journal(tmp_path)
    card = _card(tmp_path, journal, hour=8, minute=4)
    card.show_slot(_slot(8))

    card.prediction_button(REST_OF_DAY, "chop").click()
    card.confidence_button(REST_OF_DAY, "low").click()
    card.prediction_button(NEXT_5, "up").click()
    card.confidence_button(NEXT_5, "high").click()
    assert card.submit()["ok"] is True

    rows = _mentor_rows(journal)
    assert [row["timeframe"] for row in rows] == ["M5", "D1"]
    assert [row["text"] for row in rows] == ["", ""]
    assert rows[0]["entry_id"] != rows[1]["entry_id"]
    assert rows[0]["mentor"]["prediction"]["horizon"] == REST_OF_DAY
    assert rows[1]["mentor"]["prediction"]["horizon"] == NEXT_5
    assert rows[1]["mentor"]["prediction"]["direction"] == "up"


# ---------------------------------------------------------------------------
# decision 3 - the thin loader beside the pure rebuild
# ---------------------------------------------------------------------------


def test_the_thin_loader_reads_the_tape_and_the_daily_cache_and_nothing_else(
    tmp_path, monkeypatch
):
    """`internals_at` is pure; this is the only part that touches a store.

    Both stores are STAGED here (`day_review_bars.DAY_REVIEW_DIR` and
    `project_paths.DAILY_BARS_CACHE_DIR` are redirected into `tmp_path`), so
    the test proves the wiring without reading a live file. The rebuild of a
    closed hour then equals what the live card would have shown.
    """
    import day_review_bars
    import project_paths
    import trade_mentor_context as context_module

    monkeypatch.setattr(day_review_bars, "DAY_REVIEW_DIR", tmp_path / "day_review")
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", tmp_path / "daily")
    (tmp_path / "daily").mkdir(parents=True, exist_ok=True)

    payload = fixture.bars()
    day_review_bars.write_session_bars(SESSION.isoformat(), payload["m5"])
    for symbol, rows in payload["d1"].items():
        with open(tmp_path / "daily" / f"{symbol}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            for row in rows:
                writer.writerow(
                    [row["dt"], row["open"], row["high"], row["low"], row["close"], 1_000]
                )
            # A bar AFTER the rebuilt moment, which the point-in-time cut drops.
            writer.writerow(
                [(SESSION + timedelta(days=1)).isoformat(), 900.0, 999.0, 1.0, 950.0, 1_000]
            )

    loaded = context_module.internals_bars_at(SESSION.isoformat(), fixture.NOW)
    assert set(loaded["m5"]) == set(context_module.SYMBOLS)
    assert set(loaded["d1"]) == set(context_module.SYMBOLS)

    rebuilt = context_module.internals_at(SESSION.isoformat(), fixture.NOW, loaded)
    live = fixture.context()
    assert rebuilt["rebuilt_for_session"] == SESSION.isoformat()
    assert rebuilt["derived"] == live["derived"], "one builder, or the numbers drift"
    spy = context_module.internals_at(SESSION.isoformat(), fixture.NOW, loaded)
    assert (
        next(row for row in spy["readings"] if row["symbol"] == "SPY")["day_change_pct"]
        == pytest.approx(0.5, abs=1e-6)
    ), "tomorrow's daily bar may not reach a rebuild of today"


def test_a_symbol_missing_from_both_stores_is_unmeasured_not_guessed(tmp_path, monkeypatch):
    """Nothing staged at all: every reading is `unavailable` and every derived
    line `unmeasured`. Missing data is uncertainty, never zero."""
    import day_review_bars
    import project_paths
    import trade_mentor_context as context_module

    monkeypatch.setattr(day_review_bars, "DAY_REVIEW_DIR", tmp_path / "empty")
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", tmp_path / "empty-daily")

    loaded = context_module.internals_bars_at(SESSION.isoformat(), fixture.NOW)
    assert loaded["m5"] == {} and loaded["d1"] == {}
    rebuilt = context_module.internals_at(SESSION.isoformat(), fixture.NOW, loaded)
    assert all(row["m5_status"] == "unavailable" for row in rebuilt["readings"])
    assert all(row["day_change_pct"] is None for row in rebuilt["readings"])
    assert {line["status"] for line in rebuilt["derived"].values()} == {"unmeasured"}
    assert all(str(line["reason"]).strip() for line in rebuilt["derived"].values())
