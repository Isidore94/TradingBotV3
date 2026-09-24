"""Easier tags on the Trades tab (overhaul 2026-09-23): chips for your tags,
a completer, one-click common tags, and dashed suggestion chips that accept.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")

from journal_store import JournalStore  # noqa: E402
from ui.panels.journal import tag_chips as tc  # noqa: E402
from ui.services import journal_feed  # noqa: E402


def test_add_and_remove_keep_one_spelling():
    assert tc.add_tag("", "bull flag") == "bull flag"
    assert tc.add_tag("bull flag", "gap fill") == "bull flag; gap fill"
    assert tc.add_tag("bull flag; gap fill", "Bull Flag") == "bull flag; gap fill"
    assert tc.remove_tag("bull flag; gap fill", "BULL FLAG") == "gap fill"
    assert tc.parse_tags("a; b; A") == ["a", "b"]


def test_common_tags_are_yours_only():
    rows = [
        {"setup_tags": "bull flag; gap fill", "tag_status": "confirmed"},
        {"setup_tags": "bull flag", "tag_status": "confirmed"},
        {"setup_tags": "machine guess", "tag_status": "provisional"},
        {"setup_tags": "link:focus; vetoed:too_extended", "tag_status": "confirmed"},
        {"setup_tags": "", "tag_status": "confirmed"},
    ]
    assert tc.common_tags(rows) == ["bull flag", "gap fill"]
    assert tc.common_tags(rows, limit=1) == ["bull flag"]


def test_short_tag():
    assert tc.short_tag("short") == "short"
    assert tc.short_tag("x" * 40, limit=10) == "xxxxxxx..."


def test_the_completer_completes_the_last_tag_only(qapp):
    from PySide6.QtWidgets import QLineEdit

    edit = QLineEdit()
    completer = tc.MultiTagCompleter(edit)
    edit.setCompleter(completer)
    completer.set_tags(["bull flag", "gap fill", "bull flag"])
    assert completer.tags() == ["bull flag", "gap fill"]
    edit.setText("gap fill; bu")
    assert completer.splitPath(edit.text()) == ["bu"]
    index = completer.model().index(0, 0)
    assert completer.pathFromIndex(index) == "gap fill; bull flag"


# ---------------------------------------------------------------------------
# On the real Trades tab, against a throwaway store
# ---------------------------------------------------------------------------

_DAY = (date.today() - timedelta(days=date.today().weekday() + 7)).isoformat()


def _execution(uid, side, quantity, price, **overrides):
    row = {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": "51830546",
        "account_label": "TFSA", "account_type": "TFSA", "symbol": "AAPL",
        "security_type": "STK", "currency": "USD", "side": side, "quantity": quantity,
        "price": price, "timestamp": f"{_DAY}T09:31:00-04:00", "trade_date": _DAY,
        "commission": 1.0, "fees": 0.0, "gross_amount": None, "net_amount": None,
        "order_id": "", "exchange_exec_id": "", "raw_json": "{}",
    }
    row.update(overrides)
    return row


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def trades_tab(qapp, tmp_path, monkeypatch):
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    monkeypatch.setattr(journal_feed, "_STORE", store)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    store.upsert_executions(
        [
            _execution("T:1", "BUY", 10, 100.0),
            _execution("T:2", "SELL", 10, 110.0),
            _execution("T:3", "BUY", 5, 50.0, symbol="AMD"),
            _execution("T:4", "SELL", 5, 55.0, symbol="AMD"),
        ]
    )
    store.rebuild_trades(refresh_tags=False)
    trades = journal_feed.load_trades()
    journal_feed.save_annotation(trades[0].trade_id, setup_tags="bull flag", notes="")
    monkeypatch.setattr(
        journal_feed,
        "unaccepted_auto_tag_candidates",
        lambda _trade_id, _tags: [
            {"tag": "link:focus", "confidence": 1.0},
            {"tag": "gap fill", "confidence": 0.8},
        ],
    )
    from ui.panels.journal_panel import JournalPanel

    panel = JournalPanel()
    panel.trades_tab.reload()
    yield panel.trades_tab
    panel.shutdown()
    panel.deleteLater()


def _select(tab, symbol):
    for row, trade in enumerate(tab._visible_trades()):
        if trade.symbol == symbol:
            tab.table.selectRow(row)
            return trade
    raise AssertionError(symbol)


def test_your_tags_show_as_chips_and_a_click_removes_one(trades_tab):
    trade = next(t for t in trades_tab._trades if t.raw.get("setup_tags"))
    _select(trades_tab, trade.symbol)
    assert trades_tab.tag_chips.tags() == ["bull flag"]
    trades_tab.tag_chips.chips[0].click()
    assert trades_tab.tags_input.text() == ""
    assert trades_tab.tag_chips.tags() == []


def test_common_tags_are_one_click_and_the_completer_knows_them(trades_tab):
    assert trades_tab.quick_tag_chips.tags() == ["bull flag"]
    assert "bull flag" in trades_tab.tag_completer.tags()
    other = next(t for t in trades_tab._trades if not t.raw.get("setup_tags"))
    _select(trades_tab, other.symbol)
    trades_tab.tags_input.setText("gap fill")
    trades_tab.quick_tag_chips.chips[0].click()
    assert trades_tab.tags_input.text() == "gap fill; bull flag"
    assert trades_tab.tag_chips.tags() == ["gap fill", "bull flag"]


def test_a_suggestion_chip_accepts_in_one_click_and_links_are_not_chips(trades_tab, monkeypatch):
    other = next(t for t in trades_tab._trades if not t.raw.get("setup_tags"))
    _select(trades_tab, other.symbol)
    assert trades_tab.suggested_tag_chips.tags() == ["gap fill"]
    assert trades_tab.suggested_tag_chips.chips[0].objectName() == "JournalSuggestedTagChip"
    accepted = []
    monkeypatch.setattr(
        journal_feed, "accept_auto_tags",
        lambda trade_id, tags: accepted.append((trade_id, list(tags))) or "gap fill",
    )
    trades_tab.suggested_tag_chips.chips[0].click()
    assert accepted == [(other.trade_id, ["gap fill"])]


def test_provisional_tags_are_dashed_chips(trades_tab):
    trade = next(t for t in trades_tab._trades if t.raw.get("setup_tags"))
    _select(trades_tab, trade.symbol)
    trades_tab._show_tag_status({"tag_status": "provisional"})
    assert trades_tab.tag_chips.chips[0].property("provisional") is True
    trades_tab._show_tag_status({"tag_status": "confirmed"})
    assert trades_tab.tag_chips.chips[0].property("provisional") in (None, False)
