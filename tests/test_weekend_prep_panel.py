"""R8 §9 steps 6-10 - the Weekend Prep tab, constructed for real and offline.

Injected fake service and fake FocusService. No network, no broker, no writes to
the trader's state file or watchlists.
"""

from __future__ import annotations

import ast
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

import weekend_strength as ws  # noqa: E402
from ui.services.weekend_prep_service import STEP_IDS, WeekendPrepService  # noqa: E402

_app = QApplication.instance() or QApplication([])

PANEL_SOURCE = (SCRIPTS_DIR / "ui" / "panels" / "weekend_prep_panel.py").read_text(encoding="utf-8")


class _FakeFocus:
    """Records exactly what the panel asked for, and tolerates a duplicate."""

    def __init__(self, already: set | None = None) -> None:
        self.calls: list[tuple] = []
        self.kwargs: list[dict] = []
        self._already = already or set()

    def add(self, symbol, side, category="m5", *, origin="", context=""):
        self.calls.append((symbol, side, category))
        self.kwargs.append({"origin": origin, "context": context})
        return (symbol, side) not in self._already


@pytest.fixture
def service(tmp_path):
    svc = WeekendPrepService(state_path=tmp_path / "state.json", now=datetime(2026, 8, 15, 10, 0))
    yield svc
    svc.shutdown()


@pytest.fixture
def panel(service):
    from ui.panels.weekend_prep_panel import WeekendPrepPanel

    focus = _FakeFocus()
    widget = WeekendPrepPanel(service=service, focus_service=focus)
    widget._test_focus = focus
    yield widget
    widget.shutdown()
    widget.deleteLater()


# ---------------------------------------------------------------------------
# Shell and stepper
# ---------------------------------------------------------------------------


def test_the_routine_runs_in_the_trader_s_order(panel):
    """SIX steps since V2 appended "Tag this week".

    The count is not the point and pinning it made this test fail for the
    sanctioned way to add a step. What the routine has to hold is its ENDS: the
    week is read first and planned last.
    """
    assert panel.rail.count() == len(STEP_IDS)
    labels = [panel.rail.item(i).text() for i in range(panel.rail.count())]
    assert "Week in review" in labels[0]
    assert "Week ahead" in labels[-1]
    # V2 item 2e: the correcting half of the nightly tagger, after the week has
    # been read and before the week ahead is planned.
    assert "Tag this week" in " ".join(labels)
    assert STEP_IDS.index("tag_week") > STEP_IDS.index("week_review")
    assert STEP_IDS.index("tag_week") < STEP_IDS.index("week_ahead")


def test_the_rail_shows_each_step_s_status(panel, service):
    service.set_step_status("week_review", "done")
    service.set_step_status("focus_review", "skipped")
    assert panel.rail.item(0).text().startswith("●")
    assert panel.rail.item(1).text().startswith("–")
    assert panel.rail.item(2).text().startswith("○")


def test_selecting_a_step_shows_its_page(panel):
    # BY NAME rather than by row number: V2 appended a step and the number moved.
    panel.rail.setCurrentRow(STEP_IDS.index("discovery"))
    assert panel.pages.currentWidget() is panel.discovery
    panel.rail.setCurrentRow(STEP_IDS.index("tag_week"))
    assert panel.pages.currentWidget() is panel.tag_week


def test_the_header_names_the_weekend_and_the_reviewed_week(panel):
    assert "2026-08-14" in panel.header.text()
    assert "2026-08-10" in panel.header.text()


def test_marking_every_step_completes_the_routine(panel, service):
    for step in STEP_IDS:
        service.set_step_status(step, "done")
    assert "complete" in panel.header.text()


# ---------------------------------------------------------------------------
# Nothing runs by itself; nothing is ever removed
# ---------------------------------------------------------------------------


def test_building_the_panel_fetches_nothing(panel, service):
    assert service.board("h1") is None
    assert service.board("d1") is None
    assert service.week_ahead_markdown == ""


def test_the_panel_has_no_removal_path_at_all():
    """Adds only. Checked by parsing, because this is a structural promise.

    The trader's own watchlist names are untouchable, and the way to keep that
    true is for the code that could remove them not to exist here.
    """
    tree = ast.parse(PANEL_SOURCE)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)
    for forbidden in ("remove", "remove_many", "remove_if_auto_adopted", "discard", "clear_focus"):
        assert forbidden not in called, f"Weekend Prep must not call {forbidden}"


def test_the_panel_owns_no_timer():
    tree = ast.parse(PANEL_SOURCE)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "QTimer" not in names and "singleShot" not in names


def test_walkaway_runs_off_the_gui_thread_and_renders_the_structured_result(
    panel, monkeypatch
):
    from PySide6.QtCore import QThread

    called_on = []

    def run_summary(*_args):
        called_on.append(QThread.currentThread())
        return {"journal_rows": [], "focus_rows": [], "skipped_non_equity": 0}

    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.walkaway_summary", run_summary)
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.week_trades",
                        lambda *_args: {"closed": [], "still_open": []})
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.week_tag_candidates",
                        lambda *_args: [])

    panel.walkaway.reload()
    assert panel.walkaway._worker.wait(5000)
    _app.processEvents()

    assert called_on and called_on[0] is not _app.thread()
    assert "WALKAWAY ANALYSIS" in panel.walkaway.output.toPlainText()
    assert "journal_rows" not in panel.walkaway.output.toPlainText()


def test_confirming_a_tag_does_not_rerun_walkaway(panel, monkeypatch):
    row = {
        "trade_id": "trade-1", "trade_date": "2026-08-12", "symbol": "AAPL",
        "current_tags": "breakout", "candidates": [{"tag": "avwap-reclaim"}],
    }
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.accept_auto_tags",
                        lambda *_args: None)
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.week_trades",
                        lambda *_args: {"closed": [], "still_open": []})
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.week_tag_candidates",
                        lambda *_args: [row])
    monkeypatch.setattr(
        "ui.panels.weekend_prep_panel.journal_feed.walkaway_summary",
        lambda *_args: pytest.fail("tag review must not replay market history"),
    )
    panel.walkaway._tag_rows = [row]
    panel.walkaway.tag_table.setRowCount(1)
    panel.walkaway.tag_table.selectRow(0)

    panel.walkaway._confirm_tag()

    assert service_recorded(panel.service, "trade-1")


def service_recorded(service, trade_id):
    return trade_id in service.weekend_state()["tag_review"]["confirmed"]


# ---------------------------------------------------------------------------
# Discovery and Adopt
# ---------------------------------------------------------------------------


def _board(timeframe="d1", side="long", symbol="AAPL"):
    board = ws.WeekendBoard(timeframe=timeframe, side=side, as_of="2026-08-15T10:00:00")
    board.offered, board.measured, board.in_percentile = 100, 90, 22
    board.filtered_out = 21
    board.rows = [
        {"symbol": symbol, "side": side, "timeframe": timeframe, "score": 12.5,
         "last_close": 150.0, "ema": 148.0, "atr": 2.0, "bar_count": 60, "reason": ""}
    ]
    return board


def test_the_discovery_tab_has_one_sub_tab_per_timeframe(panel):
    labels = [panel.discovery.tabs.tabText(i) for i in range(panel.discovery.tabs.count())]
    assert labels == [tf.label for tf in ws.TIMEFRAMES]


def test_a_refreshed_board_renders_with_its_accounting_and_as_of(panel, service):
    service._boards["d1"] = _board()
    service.boardChanged.emit("d1")
    widgets = panel.discovery._boards["d1"]
    assert widgets["table"].rowCount() == 1
    assert widgets["table"].item(0, 0).text() == "AAPL"
    text = widgets["accounting"].text()
    assert "100 offered" in text and "90 measurable" in text and "2026-08-15" in text


def test_a_failure_banner_keeps_the_last_good_board(panel, service):
    service._boards["d1"] = _board()
    service.boardChanged.emit("d1")
    panel.discovery.show_failure("d1", "provider unavailable")
    widgets = panel.discovery._boards["d1"]
    banner = widgets["banner"]
    # Against its own parent: the D1 sub-tab is not the current tab, so asking
    # the panel would report every widget on it as hidden regardless.
    assert banner.isVisibleTo(banner.parentWidget()) is True
    assert "last good board" in widgets["banner"].text()
    assert widgets["table"].rowCount() == 1, "the rows are still there"


def test_service_failure_signal_wires_the_discovery_banner(panel, service):
    service._boards["d1"] = _board()
    service.boardChanged.emit("d1")

    service._on_failed("board:d1", "provider unavailable")

    banner = panel.discovery._boards["d1"]["banner"]
    assert banner.isVisibleTo(banner.parentWidget())
    assert "provider unavailable" in banner.text()


def test_adopt_calls_focus_with_exactly_the_spec_s_arguments(panel, service, monkeypatch):
    service._boards["d1"] = _board()
    service.boardChanged.emit("d1")
    panel.discovery._boards["d1"]["table"].selectRow(0)
    monkeypatch.setattr(
        "ui.panels.weekend_prep_panel.QMessageBox.question",
        lambda *a, **k: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.Yes,
    )
    panel.discovery._adopt("d1")

    focus = panel._test_focus
    assert focus.calls == [("AAPL", "long", "swing")]
    assert focus.kwargs[0] == {
        "origin": "weekend_prep",
        "context": f"weekend_prep:d1:{service.weekend}",
    }


def test_a_duplicate_adopt_is_tolerated_and_says_so(panel, service, monkeypatch):
    panel._test_focus._already.add(("AAPL", "long"))
    service._boards["d1"] = _board()
    service.boardChanged.emit("d1")
    panel.discovery._boards["d1"]["table"].selectRow(0)
    monkeypatch.setattr(
        "ui.panels.weekend_prep_panel.QMessageBox.question",
        lambda *a, **k: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.Yes,
    )
    messages: list[str] = []
    panel.discovery.statusChanged.connect(messages.append)
    panel.discovery._adopt("d1")
    assert any("already on the swing list" in m for m in messages)


def test_declining_the_confirmation_adopts_nothing(panel, service, monkeypatch):
    service._boards["d1"] = _board()
    service.boardChanged.emit("d1")
    panel.discovery._boards["d1"]["table"].selectRow(0)
    monkeypatch.setattr(
        "ui.panels.weekend_prep_panel.QMessageBox.question",
        lambda *a, **k: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.No,
    )
    panel.discovery._adopt("d1")
    assert panel._test_focus.calls == []


def test_an_adoption_is_recorded_in_the_weekend_state(panel, service, monkeypatch):
    service._boards["d1"] = _board()
    service.boardChanged.emit("d1")
    panel.discovery._boards["d1"]["table"].selectRow(0)
    monkeypatch.setattr(
        "ui.panels.weekend_prep_panel.QMessageBox.question",
        lambda *a, **k: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.Yes,
    )
    panel.discovery._adopt("d1")
    adopted = service.weekend_state()["adopted"]
    assert adopted and adopted[0]["symbol"] == "AAPL" and adopted[0]["tf"] == "d1"


def test_the_m5_adoption_gate_is_not_applied_to_weekend_swing_adds():
    """A recorded decision (§7), not an oversight.

    The R2 gate is session VWAP plus yesterday's extreme - an intraday-session
    test. Applying it to a swing add would refuse names on a Saturday for a
    reason that has nothing to do with a swing thesis.
    """
    assert "focus_adoption_gate" not in PANEL_SOURCE
    assert "session_vwap" not in PANEL_SOURCE


# ---------------------------------------------------------------------------
# Week ahead
# ---------------------------------------------------------------------------


def test_the_week_ahead_renders_what_the_service_emits(panel, service):
    service.refresh_week_ahead(runner=lambda: "# Week ahead\n\nCPI Tuesday.", blocking=True)
    assert "Week ahead" in panel.week_ahead.report.toMarkdown()


def test_the_week_ahead_keeps_its_last_report_when_a_rebuild_fails(panel, service):
    service.refresh_week_ahead(runner=lambda: "# Good report", blocking=True)

    def _boom():
        raise RuntimeError("market_prep unavailable")

    service.refresh_week_ahead(runner=_boom, blocking=True)
    assert "Good report" in panel.week_ahead.report.toMarkdown()


# ---------------------------------------------------------------------------
# Registration (step 10)
# ---------------------------------------------------------------------------


def test_the_tab_is_registered_once_in_page_specs():
    from ui.app import PAGE_SPECS

    entries = [spec for spec in PAGE_SPECS if spec.title == "Weekend Prep"]
    assert len(entries) == 1
    assert entries[0].attribute == "weekend_prep_panel"


def test_the_desk_still_resolves_every_page_after_the_new_tab():
    """The bug step 1 fixed is exactly the one adding a page used to cause."""
    from ui.app import PAGE_SPECS

    titles = [spec.title for spec in PAGE_SPECS]
    assert len(set(titles)) == len(titles)
    assert titles[-1] == "Settings", "Settings stays last and reachable"


# ---------------------------------------------------------------------------
# AI-P2 - the auto-tag backlog toggle (trader-approved amendment, 2026-08-24)
#
# R8 locked the journal hook to "the weekly auto-tag review only", and
# `week_tag_candidates` scopes to the week's closed trades. But the backlog was
# 220 proposals spanning history against ONE confirmed annotation, so the
# confirmation stream could only ever fill at the weekly trickle - and every
# analysis downstream of the trader's own tags waits on it.
# ---------------------------------------------------------------------------
def test_the_backlog_toggle_is_off_until_asked(panel, monkeypatch):
    """Default off: the weekend ritual stays the weekly ritual unless the
    trader deliberately opens the backlog."""
    week_calls, pending_calls = [], []
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.walkaway_summary",
                        lambda *_args: {"journal_rows": [], "focus_rows": [], "skipped_non_equity": 0})
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.week_trades",
                        lambda *_args: {"closed": [], "still_open": []})
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.week_tag_candidates",
                        lambda *_args: week_calls.append(1) or [])
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.pending_tag_candidates",
                        lambda *a, **k: pending_calls.append(1) or [])

    assert panel.walkaway.backlog_toggle.isChecked() is False
    panel.walkaway.reload()
    assert panel.walkaway._worker.wait(5000)
    _app.processEvents()

    assert week_calls and not pending_calls


def test_the_backlog_toggle_shows_every_pending_proposal(panel, monkeypatch):
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.walkaway_summary",
                        lambda *_args: {"journal_rows": [], "focus_rows": [], "skipped_non_equity": 0})
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.week_trades",
                        lambda *_args: {"closed": [], "still_open": []})
    monkeypatch.setattr(
        "ui.panels.weekend_prep_panel.journal_feed.week_tag_candidates",
        lambda *_args: pytest.fail("the backlog view must not fall back to the week"),
    )
    monkeypatch.setattr(
        "ui.panels.weekend_prep_panel.journal_feed.pending_tag_candidates",
        lambda *a, **k: [
            {"trade_id": "t1", "trade_date": "2026-03-02", "symbol": "DELL",
             "current_tags": "", "candidates": [{"tag": "avwap_signal"}],
             "already_tagged": False},
        ],
    )

    panel.walkaway.backlog_toggle.setChecked(True)
    panel.walkaway.reload()
    assert panel.walkaway._worker.wait(5000)
    _app.processEvents()

    assert [row["symbol"] for row in panel.walkaway._tag_rows] == ["DELL"]
    assert "backlog" in panel.walkaway.tag_note.text().lower()


def test_the_confirm_and_correct_paths_are_untouched_by_the_toggle(panel, monkeypatch):
    """The amendment widens WHAT IS LISTED and nothing else. If the toggle
    changed how a confirmation is written, it would have quietly forked the
    trader's own annotation stream in two."""
    accepted = []
    row = {
        "trade_id": "trade-9", "trade_date": "2026-03-02", "symbol": "DELL",
        "current_tags": "", "candidates": [{"tag": "avwap_signal"}],
        "already_tagged": False,
    }
    monkeypatch.setattr("ui.panels.weekend_prep_panel.journal_feed.accept_auto_tags",
                        lambda trade_id, tags: accepted.append((trade_id, list(tags))))
    panel.walkaway.backlog_toggle.setChecked(True)
    panel.walkaway._tag_rows = [row]
    panel.walkaway.tag_table.setRowCount(1)
    panel.walkaway.tag_table.selectRow(0)

    panel.walkaway._confirm_tag()

    assert accepted == [("trade-9", ["avwap_signal"])]
    assert service_recorded(panel.service, "trade-9")
