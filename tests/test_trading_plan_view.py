"""P1-7 7a: the plan is shown read-only on Day Review and Weekend Prep, read off the Qt thread."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _wait(view, app):
    for _ in range(200):
        worker = view._worker
        if worker is None or worker.wait(25):
            break
    for _ in range(20):
        app.processEvents()


def test_the_view_reads_on_a_worker_and_is_read_only(app, tmp_path, monkeypatch):
    import project_paths
    from ui.widgets.trading_plan_view import TradingPlanView

    monkeypatch.setattr(project_paths, "TRADING_PLAN_FILE", tmp_path / "trading_plan.md")
    monkeypatch.setattr(project_paths, "TRADING_PLAN_HISTORY_DIR", tmp_path / "trading_plan_history")
    (tmp_path / "trading_plan.md").write_text("## Goals\n- make 2R a week\n", encoding="utf-8")
    gui = threading.get_ident()
    threads = []
    import trading_plan

    def loader():
        threads.append(threading.get_ident())
        return trading_plan.read_plan()

    view = TradingPlanView(loader=loader)
    try:
        view.refresh()
        _wait(view, app)
        assert threads and threads[0] != gui
        assert view.text.isReadOnly()
        assert "make 2R a week" in view.text.toPlainText()
        assert "Missing headings" in view.note.text()
        assert len(trading_plan.snapshots()) == 1
    finally:
        view.shutdown()


def test_a_failed_read_keeps_the_last_text(app):
    from ui.widgets.trading_plan_view import TradingPlanView

    view = TradingPlanView(loader=lambda: {"text": "## Rules\n- one\n", "path": "x", "parsed": {}})
    try:
        view.refresh()
        _wait(view, app)
        view._loader = lambda: {"error": "the plan could not be read: locked"}
        view.refresh()
        _wait(view, app)
        assert "- one" in view.text.toPlainText()
        assert "locked" in view.note.text()
    finally:
        view.shutdown()


def test_day_review_and_week_review_carry_the_plan_view():
    """Static: both pages build the view and refresh it with their own read."""
    day = (SCRIPTS / "ui" / "panels" / "day_review_panel.py").read_text(encoding="utf-8")
    week = (SCRIPTS / "ui" / "panels" / "weekend_prep_panel.py").read_text(encoding="utf-8")
    for source in (day, week):
        assert "TradingPlanView(self)" in source
        assert "self.plan_view.refresh()" in source
        assert "self.plan_view.shutdown()" in source
