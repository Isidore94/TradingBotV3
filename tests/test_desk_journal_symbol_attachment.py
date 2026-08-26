"""The quick journal entry knew which chart it was about and threw it away.

`MarketJournalService.write_entry` has taken `symbols` since R10.H, and the
Market Journal page passes it. The Desk "Journal" tab - the fast Ctrl+Enter
capture, the one actually used mid-session with a chart in front of you - never
did. So the entries most likely to be about a specific name were the ones
stored with no name attached, and nothing downstream could join them to it.

Scope note: `scripts/ui/panels/alert_center_panel.py` is a fenced file under
the file-scoped ask-first rule. The trader pre-authorized THIS change in it and
nothing else - the symbols attachment on the quick journal write.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


class _Alert:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


def _panel_and_calls(tmp_path):
    from ui.panels.alert_center_panel import AlertCenterPanel

    QApplication.instance() or QApplication([])
    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")

    calls: list[dict] = []

    def fake_write_entry(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    panel.market_journal_service.write_entry = fake_write_entry
    return panel, calls


@pytest.mark.qt
def test_the_quick_journal_entry_carries_the_chart_symbol(tmp_path):
    panel, calls = _panel_and_calls(tmp_path)
    panel._current_review_alert = _Alert("AAA")
    panel._journal_text.setPlainText("reclaimed VWAP and held it")

    panel._commit_journal_entry()

    assert calls, "nothing was written"
    assert list(calls[0].get("symbols") or []) == ["AAA"], calls[0]


@pytest.mark.qt
def test_no_chart_means_no_symbol_never_a_wrong_one(tmp_path):
    """A general note about the session is not a note about the last symbol.

    Attaching a stale symbol would be worse than attaching none: it would
    quietly assert a link the trader never made.
    """
    panel, calls = _panel_and_calls(tmp_path)
    panel._current_review_alert = None
    panel._journal_text.setPlainText("chop all morning, stood down")

    panel._commit_journal_entry()

    assert calls, "nothing was written"
    assert list(calls[0].get("symbols") or []) == [], calls[0]
