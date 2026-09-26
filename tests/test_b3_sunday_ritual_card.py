"""B3 Sunday ritual card: tags waiting + plan review + exits per setup family.

Hand-built journal on 2026-09-11 (holds in hours, P&L sign):
AAA avwap_breakout win 2h (trader said took_profit_early), BBB avwap_breakout
win 4h (no exit reason), CCC avwap_breakout loss 6h, DDD avwap_breakout loss 1h,
EEE pullback_sma_reclaim loss 2h, FFF needs_review win 3h, GGG needs_review win 1h.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_support import REVIEWED, _execution, new_store  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TRADES = (
    # symbol, entry hour, hold hours, exit price (entry 10.0), setup, confirmed
    ("AAA", 6, 2, 11.0, "avwap_breakout", True),
    ("BBB", 7, 4, 11.0, "avwap_breakout", True),
    ("CCC", 8, 6, 9.0, "avwap_breakout", True),
    ("DDD", 9, 1, 9.0, "avwap_breakout", True),
    ("EEE", 10, 2, 9.0, "pullback_sma_reclaim", True),
    ("FFF", 11, 3, 11.0, "", False),
    ("GGG", 12, 1, 11.0, "", False),
)


def _round_trip(store, symbol, entry_hour, hold, exit_price):
    store.upsert_executions(
        [
            _execution(f"{symbol}-open", symbol=symbol, side="BUY", qty=100, price=10.0,
                       timestamp=f"{REVIEWED}T{entry_hour:02d}:31:00"),
            _execution(f"{symbol}-close", symbol=symbol, side="SELL", qty=100, price=exit_price,
                       timestamp=f"{REVIEWED}T{entry_hour + hold:02d}:31:00"),
        ]
    )


@pytest.fixture
def journal(tmp_path):
    store = new_store(tmp_path)
    for symbol, hour, hold, price, _setup, _confirmed in TRADES:
        _round_trip(store, symbol, hour, hold, price)
    store.rebuild_trades(refresh_tags=False)
    ids = {str(t["symbol"]): str(t["trade_id"]) for t in store.list_trades()}
    for symbol, _hour, _hold, _price, setup, confirmed in TRADES:
        if confirmed:
            store.save_trade_annotation(ids[symbol], setup_tags=setup, notes="")
        else:
            assert store.mark_tags_needing_review(ids[symbol])
    store.record_opportunity_event(
        opportunity_id=f"trade:{ids['AAA']}",
        event_type="EXIT_NOTE_FIELDS",
        trade_id=ids["AAA"],
        occurred_at=datetime(2026, 9, 12, 9, 0).astimezone(),
        reason="exit_fields:confirm",
        payload={"fields": {"why": {"code": "took_profit_early"}}, "exit_session": REVIEWED},
        source="trade_mentor",
    )
    return store, ids


def _no_plan():
    return {"exists": False, "error": "", "parsed": {}}


def _read(store):
    import sunday_ritual

    return sunday_ritual.read_card(
        lambda: store, plan_reader=_no_plan, challenges_reader=lambda: []
    )


def test_family_lines_match_the_fixture_journal(journal):
    import sunday_ritual

    store, _ids = journal
    payload = _read(store)

    assert payload["waiting"] == 2
    lines = sunday_ritual.format_card(payload)["families"].splitlines()
    assert lines == [
        "Exits by setup family (closed trades, from the journal):",
        "avwap_breakout: 4 closed, 2 won, 2 lost · exit early 1 of 2 winners "
        "(1 with no exit reason) · held losers 1 of 2 (held past the median winner) · "
        "median hold win 3.0h, loss 3.5h",
        "pullback_sma_reclaim: 1 closed, 0 won, 1 lost · exit early 0 of 0 winners · "
        "held losers unknown of 1 (no winner to compare) · median hold win unknown, loss 2.0h",
        "not confirmed yet: 2 closed, 2 won, 0 lost · exit early 0 of 2 winners "
        "(2 with no exit reason) · held losers 0 of 0 · median hold win 2.0h, loss unknown",
    ]


def test_unreadable_exit_reasons_say_unknown_never_zero(journal, monkeypatch):
    import sunday_ritual

    store, _ids = journal

    def boom(**_k):
        raise OSError("locked")

    monkeypatch.setattr(store, "list_opportunity_events", boom)
    text = sunday_ritual.format_card(_read(store))["families"]
    assert "Exit reasons could not be read (locked)" in text
    assert "exit early unknown of 2 winners" in text
    assert "exit early 0" not in text


def test_an_unreadable_journal_is_stated():
    import sunday_ritual

    def boom():
        raise OSError("disk gone")

    texts = sunday_ritual.format_card(
        sunday_ritual.read_card(boom, plan_reader=_no_plan, challenges_reader=lambda: [])
    )
    assert "could not be read (disk gone)" in texts["tags"]
    assert texts["families"] == "Exits by setup family: unknown (journal unreadable)."


def test_the_plan_review_reads_without_creating_or_snapshotting(monkeypatch):
    import sunday_ritual
    import trading_plan

    seen = {}

    def fake_read_plan(**kwargs):
        seen.update(kwargs)
        return {"exists": True, "error": "", "parsed": trading_plan.parse_plan(
            "## Setups I trade\n- D1 AVWAP reclaim\n## What I am testing\n- one trade a day\n"
        )}

    monkeypatch.setattr(trading_plan, "read_plan", fake_read_plan)
    challenge = {"plan_line_text": "one trade a day", "text": "you took 3 on Tuesday"}
    payload = sunday_ritual.read_card(lambda: None, challenges_reader=lambda: [challenge])
    plan = sunday_ritual.format_card(payload)["plan"].splitlines()

    assert seen == {"create": False, "snapshot": False}
    assert plan[0] == "Setups I trade: D1 AVWAP reclaim"
    assert plan[1] == "What I am testing: one trade a day"
    assert "Night AI plan challenges open: 1" in plan[3]
    assert plan[4] == '- "one trade a day": you took 3 on Tuesday'


@pytest.mark.qt
def test_bulk_confirm_from_the_card_writes_exactly_the_selected_tags(journal):
    pytest.importorskip("PySide6", reason="the card is Qt")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ui.panels.journal import bulk_confirm_dialog as dialog_module
    from ui.widgets.sunday_ritual_card import SundayRitualCard

    QApplication.instance() or QApplication([])
    store, ids = journal
    card = SundayRitualCard(
        reader=lambda: _read(store), store_factory=lambda: store, threaded=False
    )
    card.load()
    assert card.tags_label.text().startswith("Setup tags: 2 waiting")
    emitted = []
    card.tagsConfirmed.connect(emitted.append)

    card.confirm_button.click()
    dialog = card._dialog
    assert dialog is not None and dialog.isModal() is False
    # The preview comes first: listed, nothing written.
    assert dialog.table.rowCount() == 2
    assert {store.annotation_state(ids[s])["tag_status"] for s in ("FFF", "GGG")} == {"needs_review"}

    rows = {dialog.table.item(r, dialog_module.COL_SYMBOL).text(): r for r in range(2)}
    dialog.table.item(rows["FFF"], dialog_module.COL_CHECK).setCheckState(Qt.Checked)
    box = dialog.setup_box(rows["FFF"])
    box.setCurrentIndex(box.findData("avwap_breakout"))
    dialog.table.item(rows["GGG"], dialog_module.COL_CHECK).setCheckState(Qt.Unchecked)
    dialog.confirm_button.click()

    fff = store.annotation_state(ids["FFF"])
    assert (fff["tag_status"], fff["setup_tags"]) == ("confirmed", "avwap_breakout")
    assert store.annotation_state(ids["GGG"])["tag_status"] == "needs_review"
    assert emitted and emitted[0]["confirmed_ids"] == [ids["FFF"]]
    assert card.tags_label.text().startswith("Setup tags: 1 waiting"), "the card re-reads"
    dialog.close()


@pytest.mark.qt
def test_the_card_only_formats_and_reads_on_a_worker(journal, monkeypatch):
    pytest.importorskip("PySide6", reason="the card is Qt")
    from PySide6.QtWidgets import QApplication

    import sunday_ritual
    from ui.widgets.sunday_ritual_card import SundayRitualCard

    app = QApplication.instance() or QApplication([])
    store, _ids = journal
    payload = _read(store)

    # apply() sets exactly what format_card built, and reads nothing.
    def no_reads(*_a, **_k):
        raise AssertionError("the card read on the Qt thread")

    monkeypatch.setattr(sunday_ritual, "read_card", no_reads)
    card = SundayRitualCard(threaded=True)
    card.apply(payload)
    texts = sunday_ritual.format_card(payload)
    assert (card.tags_label.text(), card.plan_label.text(), card.families_label.text()) == (
        texts["tags"], texts["plan"], texts["families"]
    )

    threads = []

    def reader():
        threads.append(threading.get_ident())
        return payload

    card = SundayRitualCard(reader=reader, threaded=True)
    card.load()
    card._worker.wait(10000)
    app.processEvents()
    assert threads and threads[0] != threading.get_ident()
    assert card.families_label.text() == texts["families"]
    card.shutdown()


@pytest.mark.qt
def test_the_tag_week_page_carries_the_card_and_weekend_prep_still_fits_2160(monkeypatch):
    pytest.importorskip("PySide6", reason="the panel is Qt")
    from PySide6.QtWidgets import QApplication

    import sunday_ritual
    from ui.panels import weekend_prep_panel as panel_module
    from ui.widgets.sunday_ritual_card import SundayRitualCard

    app = QApplication.instance() or QApplication([])
    loads = []
    monkeypatch.setattr(SundayRitualCard, "load", lambda self: loads.append(self))
    panel = panel_module.WeekendPrepPanel()
    try:
        page = panel.tag_week
        assert isinstance(page.ritual_card, SundayRitualCard)
        monkeypatch.setattr(page, "_read_everything", lambda: {})
        page.reload()
        assert loads == [page.ritual_card]

        # The biggest card the format allows: 20 families, 5 challenges.
        families = [
            {"family": f"family_{i}", "closed": 30 - i, "winners": 10, "losers": 9,
             "exit_early": 3, "winners_no_reason": 4, "held_losers": 5, "losers_no_hold": 0,
             "median_winner_hours": 30.0, "median_loser_hours": 70.0}
            for i in range(20)
        ]
        challenges = [{"plan_line_text": "a rule " * 8, "text": "evidence " * 12}] * 5
        payload = {"waiting": 197, "families": families, "challenges": challenges,
                   "plan": {"exists": True, "parsed": {"sections": {
                       "Setups I trade": ["setup " * 10] * 6,
                       "What I am testing": ["test " * 10] * 4}}}, "errors": {}}
        assert "... and 8 smaller families." in sunday_ritual.format_card(payload)["families"]
        page.ritual_card.apply(payload)
        panel.resize(3456, 2160)
        panel.show()
        app.processEvents()
        assert panel.height() <= 2160, panel.minimumSizeHint().height()
    finally:
        panel.shutdown()
        panel.close()
