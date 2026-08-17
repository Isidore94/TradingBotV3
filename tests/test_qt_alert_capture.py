"""R4 sections 2.3 and 5: capture and the badge on the Alert Center pane.

Section 2.3's whole risk is that LIKE blurs into placement. The spec is
explicit: CaptureRail LIKE is analysis-only and *never* writes Focus
membership; "Add to Focus Picks" stays the one explicit placement verb. Several
of these tests exist only to keep that boundary from eroding -- an earlier draft
of the rail did route likes through FocusService.add, and it had to be removed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def pane(tmp_path):
    import pick_feedback
    from ui.widgets.alert_chart_review import AlertChartReview

    pick_feedback.clear_reviewed_today_cache()
    widget = AlertChartReview(annotations_path=tmp_path / "trader_annotations.jsonl")
    yield widget
    widget.deleteLater()


def _alert(symbol: str = "AAPL", side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="09:31:00",
        symbol=symbol,
        side=side,
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[B-TIER] {symbol}: Bounce confirmed",
    )


def _show(pane, monkeypatch, symbol: str = "AAPL", side: str = "LONG"):
    monkeypatch.setattr(pane.snapshot, "set_symbol", lambda *a, **k: None)
    pane.set_alert(_alert(symbol, side))


# --------------------------------------------------------------------------
# the rail exists and follows the alert
# --------------------------------------------------------------------------
def test_the_alert_pane_carries_a_capture_rail(pane):
    from ui.widgets.capture_rail import CaptureRail

    assert isinstance(pane.capture_rail, CaptureRail)


def test_setting_an_alert_points_the_rail_at_its_symbol(pane, monkeypatch):
    _show(pane, monkeypatch, "NVDA", "SHORT")
    assert pane.capture_rail._symbol == "NVDA"
    assert pane.capture_rail._side == "SHORT"


def test_a_new_alert_clears_the_previous_level_reference(pane, monkeypatch):
    _show(pane, monkeypatch, "NVDA")
    pane.snapshot.d1LevelSelected.emit("NVDA", "d1_horizontal:2026-06-01:100.00", "d1_horizontal", 100.0)
    assert pane.capture_rail._ref_level_id
    _show(pane, monkeypatch, "AMD")
    assert pane.capture_rail._ref_level_id == ""


def test_a_selected_level_becomes_the_capture_reference(pane, monkeypatch):
    _show(pane, monkeypatch, "NVDA")
    pane.snapshot.d1LevelSelected.emit("NVDA", "d1_horizontal:2026-06-01:100.00", "d1_horizontal", 100.0)
    assert pane.capture_rail._ref_level_family == "d1_horizontal"


# --------------------------------------------------------------------------
# LIKE is analysis-only - the boundary section 2.3 exists to protect
# --------------------------------------------------------------------------
def test_like_writes_one_annotation_row(pane, monkeypatch, tmp_path):
    _show(pane, monkeypatch, "AAPL")
    pane.capture_rail.setup_input.setCurrentIndex(0)
    row = pane.capture_rail.commit_like()
    assert row is not None
    lines = (tmp_path / "trader_annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["symbol"] == "AAPL"


def test_like_never_writes_focus_membership(pane, monkeypatch, tmp_path):
    """The explicit placement verb stays the only thing that places."""
    _show(pane, monkeypatch, "AAPL")
    placed: list = []
    pane.focusRequested.connect(placed.append)
    pane.capture_rail.setup_input.setCurrentIndex(0)
    pane.capture_rail.commit_like()
    assert placed == []
    # And nothing but the annotation file appeared.
    assert {path.name for path in tmp_path.iterdir()} == {"trader_annotations.jsonl"}


def test_the_focus_verb_still_places(pane, monkeypatch):
    """The other half of the same boundary: placement still works, and is
    still a different button from LIKE."""
    _show(pane, monkeypatch, "AAPL")
    placed: list = []
    pane.focusRequested.connect(placed.append)
    pane.focus_button.click()
    assert len(placed) == 1
    assert placed[0].symbol == "AAPL"


def test_capture_does_not_advance_the_review_queue(pane, monkeypatch):
    """Capture is a recorder. Only the three queue verbs move the queue."""
    _show(pane, monkeypatch, "AAPL")
    moved: list = []
    pane.skipRequested.connect(moved.append)
    pane.removeTodayRequested.connect(moved.append)
    pane.focusRequested.connect(moved.append)
    pane.capture_rail.note_input.setText("heavy into the 50")
    pane.capture_rail.commit_note()
    assert moved == []


# --------------------------------------------------------------------------
# section 5 badge
# --------------------------------------------------------------------------
def test_a_symbol_decided_today_shows_the_badge(pane, monkeypatch, tmp_path):
    import pick_feedback

    feedback = tmp_path / "pick_feedback.jsonl"
    today = datetime.now().date().isoformat()
    feedback.write_text(
        json.dumps(
            {
                "ts": f"{today}T09:31:00",
                "trade_date": today,
                "symbol": "AAPL",
                "side": "LONG",
                "verdict": "dislike",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pick_feedback.clear_reviewed_today_cache()
    monkeypatch.setattr(
        pane,
        "_reviewed_symbols",
        lambda: pick_feedback.reviewed_symbols_today(
            market_date=today,
            pick_feedback_path=feedback,
            review_events_path=tmp_path / "none.jsonl",
            annotations_path=tmp_path / "none2.jsonl",
        ),
    )
    _show(pane, monkeypatch, "AAPL")
    assert "Reviewed today" in pane.reviewed_badge.text()
    _show(pane, monkeypatch, "TSLA")
    assert pane.reviewed_badge.text() == ""


def test_capturing_makes_the_badge_appear_immediately(pane, monkeypatch):
    seen: set[str] = set()
    monkeypatch.setattr(pane, "_reviewed_symbols", lambda: set(seen))
    _show(pane, monkeypatch, "AAPL")
    assert pane.reviewed_badge.text() == ""
    seen.add("AAPL")
    pane.capture_rail.note_input.setText("checked")
    pane.capture_rail.commit_note()
    assert "Reviewed today" in pane.reviewed_badge.text()


def test_a_badge_lookup_failure_never_takes_down_the_pane(pane, monkeypatch):
    def boom():
        raise OSError("home folder went away")

    monkeypatch.setattr(pane, "_reviewed_symbols", boom)
    _show(pane, monkeypatch, "AAPL")
    assert pane.reviewed_badge.text() == ""
