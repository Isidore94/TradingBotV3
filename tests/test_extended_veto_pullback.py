"""AR-3: the exact extended-from-base veto arms a narrow, persistent pullback watch."""

from __future__ import annotations

import json
import os
import sys
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


VETO_CODE = "too_extended_from_base"
VETO_SOURCE = f"veto: {VETO_CODE}"


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _scan(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="10:05:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) zone1 bounce off AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{side.lower()}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) zone1 bounce",
        is_d1=True,
    )


def _panel(tmp_path, monkeypatch):
    import pick_feedback
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_a, **_k: None)
    pick_feedback.clear_reviewed_today_cache()
    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        chart_watches_path=tmp_path / "chart_watches.json",
        review_events_path=tmp_path / "review_events.jsonl",
    )
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(panel, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(panel, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(panel.chart_review, "_reviewed_symbols", lambda: set())
    rail = panel.chart_review.capture_rail
    monkeypatch.setattr(rail, "_annotations_path", tmp_path / "annotations.jsonl")
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True, "added": 0}
    return panel


def _choose_extended_reason(rail) -> None:
    from ui.widgets.capture_rail import _REASON_ROLE

    for row in range(rail.reason_list.count()):
        item = rail.reason_list.item(row)
        if item.data(_REASON_ROLE) == VETO_CODE:
            rail.reason_list.setCurrentItem(item)
            rail.reason_list.itemActivated.emit(item)
            return
    raise AssertionError(f"the loaded veto vocabulary has no {VETO_CODE!r} row")


def _annotation_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_extended_veto_arms_only_m30_and_h1_pullback_and_keeps_the_normal_veto_retirement(
    tmp_path, monkeypatch
):
    """The real list activation writes first, arms through the host, then retires.

    A pullback watch's trigger names alone cannot express the timeframe rule:
    the normal SMA triggers evaluate both M15 and M30.  The persisted scope is
    therefore part of this test's public contract, and has to survive reload.
    """
    from chart_watch import load_chart_watches
    from ui.panels.alert_center_panel import CHART_WATCH_TAG
    from ui.models.bounce import BounceAlert

    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.add_alert(_scan("AAPL"))
        panel.add_alert(_scan("NVDA", "SHORT"))
        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "AAPL"

        _choose_extended_reason(panel.chart_review.capture_rail)
        QApplication.processEvents()

        rows = _annotation_rows(tmp_path / "annotations.jsonl")
        assert len(rows) == 1
        assert rows[0]["event_type"] == "veto"
        assert rows[0]["reason_code"] == VETO_CODE
        assert rows[0]["symbol"] == "AAPL" and rows[0]["side"] == "LONG"

        assert len(panel._chart_watches) == 1
        watch = panel._chart_watches[0]
        assert watch.kind == "pullback"
        assert watch.symbol == "AAPL" and watch.side == "LONG"
        assert watch.source_text == VETO_SOURCE
        assert tuple(watch.timeframes) == ("M30", "H1")
        assert watch.watch_id and not watch.fired and not watch.declined
        assert load_chart_watches(tmp_path / "chart_watches.json") == [watch]
        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "NVDA", "the normal veto retirement still advances"

        # A later real armed hit remains visible despite today's veto.  It is
        # an alert the trader asked the desk to watch, never an ordinary scan.
        hit = BounceAlert(
            time_text="10:30:00",
            symbol="AAPL",
            side="LONG",
            trigger="M30 reclaim fired",
            timeframe="D1",
            tag=CHART_WATCH_TAG,
            raw_text="CHART WATCH AAPL (LONG): M30 reclaim fired",
            is_d1=True,
        )
        panel.add_alert(hit)
        assert hit in panel._d1_alerts
        assert any(alert is hit for alert in panel._review_queue)

        # Re-visiting and vetoing the same chart creates another evidence row,
        # but never resets or duplicates the already armed condition.
        panel.chart_alert(_scan("AAPL"))
        _choose_extended_reason(panel.chart_review.capture_rail)
        QApplication.processEvents()
        assert panel._chart_watches == [watch]
        assert len(_annotation_rows(tmp_path / "annotations.jsonl")) == 2
    finally:
        panel.close()
        panel.deleteLater()


def test_extended_veto_never_changes_a_manual_pullback_or_claims_a_failed_arm(tmp_path, monkeypatch):
    """The exception is additive, idempotent, and honest when its store fails."""
    from chart_watch import PULLBACK_KIND
    from ui.panels import alert_center_panel as panel_module

    panel = _panel(tmp_path, monkeypatch)
    try:
        panel.add_alert(_scan("MSFT"))
        panel.add_alert(_scan("NVDA", "SHORT"))
        assert panel.arm_chart_watch_for("MSFT", "LONG", PULLBACK_KIND, source_text="chart")
        manual = panel._chart_watches[0]
        _choose_extended_reason(panel.chart_review.capture_rail)
        QApplication.processEvents()
        assert panel._chart_watches == [manual], "a veto must not narrow or reset a manual watch"
    finally:
        panel.close()
        panel.deleteLater()

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed = _panel(failed_root, monkeypatch)
    try:
        failed.add_alert(_scan("AMD"))
        failed.add_alert(_scan("NVDA", "SHORT"))
        monkeypatch.setattr(
            panel_module,
            "save_chart_watches",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )
        _choose_extended_reason(failed.chart_review.capture_rail)
        QApplication.processEvents()

        assert _annotation_rows(failed_root / "annotations.jsonl")[0]["reason_code"] == VETO_CODE
        assert failed._current_review_alert is not None
        assert failed._current_review_alert.symbol == "NVDA"
        assert not failed._chart_watches
        assert "not armed" in failed.chart_review.capture_rail.status_label.text().casefold()
    finally:
        failed.close()
        failed.deleteLater()
