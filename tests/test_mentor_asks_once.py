"""The Trade Mentor asks about a trade ONCE.

Trader, 2026-09-23: *"in trade mentor when I get asked about trades, dont keep
asking me again about them all day. just assume anything not filled should be
filled by AI based on my response and if the AI cant determine then just leave
it blank. I find I get asked about my DRAM trade every hour so far from
yesterday"*.

What the live journal showed: the trader answered every hourly market read, but
no trade row had been saved since 2026-09-14 - each trade's Save stayed grey
until every field and the exit box were answered, so the DRAM trades rode onto
every later card. Now any answer opens Save, the card files what is on it when
it is left, a `MENTOR_ASKED` marker retires the trade, and the local model
fills the blanks it can quote.

Offscreen, over a scratch journal under `tmp_path`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_support import (  # noqa: E402
    REVIEWED,
    SESSION,
    _execution,
    add_open_position,
    add_round_trip,
    mark_covered,
    new_store,
    pacific,
    slot_at,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402


class _Journal:
    """The Market Journal service seam `submit` writes the hourly read through."""

    def write_entry(self, **_kwargs):
        return {"ok": True, "entry": {}}


def _card(tmp_path):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    QApplication.instance() or QApplication([])
    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json", journal=_Journal())
    card._clock = lambda: pacific(SESSION, 9, 5)
    return card


@pytest.fixture()
def one_trade(tmp_path):
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "DRAM")
    card = _card(tmp_path)
    card.show_slot(slot_at(SESSION, 9))
    card.set_trade_check(check.build_task(store, SESSION), store=store)
    assert trade_id in card._answer_inputs
    return store, card, trade_id


def test_answering_the_market_read_files_the_untouched_trade_and_retires_it(one_trade, monkeypatch):
    import trade_mentor_trade_check as check

    store, card, trade_id = one_trade
    monkeypatch.setattr(card, "_service", lambda: _Journal())
    monkeypatch.setattr(card, "_missing_prediction_reason", lambda _h: "")

    assert card.submit()["ok"] is True

    assert check.asked_state(store, trade_id)[0] is True
    assert check.recalled_fields(store, trade_id) == [], "blank stays blank"
    assert check.build_task(store, SESSION).trades == (), "never asked again"


def test_skipping_the_card_also_retires_the_trade(one_trade):
    import trade_mentor_trade_check as check

    store, card, trade_id = one_trade
    card.skip()

    assert check.build_task(store, SESSION).trades == ()
    assert check.asked_state(store, trade_id)[0] is True


def test_the_raw_note_goes_to_local_ai_for_the_blank_fields_only(one_trade):
    import trade_mentor_trade_check as check

    store, card, trade_id = one_trade
    started = []
    card._start_ai_fill = lambda _store, tid, words, blank, _q, _m: started.append(
        (tid, words, blank)
    )
    stop_combo = card._answer_inputs[trade_id]["stop"][0]
    stop_combo.setCurrentIndex(stop_combo.findData(check.ANSWER_NOT_APPLICABLE))
    card._raw_trade_inputs[trade_id].setPlainText("Bought the bounce off VWAP.")

    card.show_slot(slot_at(SESSION, 10))

    assert check.answered_fields(store, trade_id) == {"stop"}
    assert len(started) == 1
    tid, words, blank = started[0]
    assert tid == trade_id and "Bought the bounce off VWAP." in words
    assert "stop" not in blank and "thesis" in blank


def test_local_ai_answers_are_stored_and_labelled_as_machine_filled(tmp_path):
    import trade_mentor_ai
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    trade_id = add_round_trip(store, "DRAM")
    words = "Bought the bounce off VWAP. no target, just trailing."

    def _model(**_kwargs):
        return {
            "model": "stub",
            "summary": {
                "answers": [
                    {
                        "field": "thesis",
                        "state": check.ANSWER_NOT_SUPPLIED,
                        "text": "bounce off VWAP",
                        "value": None,
                        "unit": "",
                        "source_span": "Bought the bounce off VWAP.",
                    }
                ],
                "follow_up": "What was your stop?",
            },
        }

    rows = trade_mentor_ai.fill_blank_fields(
        store, trade_id, words, ("thesis", "stop"), {"trade_id": trade_id}, request=_model
    )

    assert len(rows) == 1
    stored = store.list_opportunity_events(trade_id=trade_id, event_type=check.EVENT_RECALLED)
    assert stored[0]["payload"]["field"] == "thesis"
    assert stored[0]["payload"]["filled_by"] == "local_ai"
    assert stored[0]["payload"]["model"] == "stub"
    assert check.answered_fields(store, trade_id) == {"thesis"}, "stop stays blank"


def test_a_model_that_cannot_quote_the_words_stores_nothing(tmp_path):
    import trade_mentor_ai
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    trade_id = add_round_trip(store, "DRAM")

    def _model(**_kwargs):
        return {
            "summary": {
                "answers": [
                    {
                        "field": "stop",
                        "state": check.ANSWER_NOT_SUPPLIED,
                        "text": "61.50",
                        "value": None,
                        "unit": "",
                        "source_span": "words the trader never wrote",
                    }
                ],
                "follow_up": "",
            }
        }

    with pytest.raises(ValueError):
        trade_mentor_ai.fill_blank_fields(
            store, trade_id, "felt good", ("stop",), {"trade_id": trade_id}, request=_model
        )
    assert check.recalled_fields(store, trade_id) == []


def test_a_failed_journal_write_writes_no_marker_and_keeps_the_trade(one_trade, monkeypatch):
    import trade_mentor_trade_check as check

    store, card, trade_id = one_trade
    card._raw_trade_inputs[trade_id].setPlainText("my words")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(check, "save_raw_reply", _boom)
    card.show_slot(slot_at(SESSION, 10))

    assert check.asked_state(store, trade_id)[0] is False, "asked again, never lost"
    assert "NOT saved" in card.status_label.text()
    monkeypatch.undo()
    assert [q.trade_id for q in check.build_task(store, SESSION).trades] == [trade_id]


def test_an_exit_on_a_later_session_is_a_new_one_time_ask(tmp_path):
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    opened = "2026-09-10"
    mark_covered(store, opened)
    mark_covered(store, REVIEWED)
    trade_id = add_open_position(store, "DRAM", day=opened)
    check.record_asked(store, trade_id, session=opened, exit_session="")

    store.upsert_executions(
        [
            _execution(
                f"DRAM-{REVIEWED}-close",
                symbol="DRAM",
                side="SELL",
                qty=100,
                price=11.0,
                timestamp=f"{REVIEWED}T09:05:00",
            )
        ]
    )
    store.rebuild_trades(refresh_tags=False)

    task = check.build_task(store, SESSION)
    asked = [q for q in task.trades if q.trade_id == trade_id]
    assert len(asked) == 1, "the exit is new, so it is asked"
    assert asked[0].missing == (), "the entry fields were retired already"
    assert asked[0].exit_session == REVIEWED

    check.record_asked(store, trade_id, session=SESSION.isoformat(), exit_session=REVIEWED)
    assert check.build_task(store, SESSION).trades == (), "and asked once"
