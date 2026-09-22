"""AR-3: the exact extended-from-base veto arms a narrow, persistent pullback watch."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
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


def _choose_another_reason(rail) -> None:
    """Click a real, loaded veto reason that is deliberately not the exception."""
    from ui.widgets.capture_rail import _REASON_ROLE

    for row in range(rail.reason_list.count()):
        item = rail.reason_list.item(row)
        if item.data(_REASON_ROLE) != VETO_CODE:
            rail.reason_list.setCurrentItem(item)
            rail.reason_list.itemActivated.emit(item)
            return
    raise AssertionError("the loaded veto vocabulary has no non-extended reason")


def _show_all_scans_if_the_view_has_the_new_control(panel) -> None:
    """AR-3 needs scan rows only after the AR-2B UI explicitly reveals them.

    The old panel has no view switch and already shows every scan.  This keeps
    the fixture useful on the old branch while forcing the finished UI path
    whenever the new control exists.
    """
    from PySide6.QtWidgets import QPushButton

    buttons = [
        button
        for button in panel.findChildren(QPushButton)
        if "show all" in button.text().casefold()
    ]
    if not buttons:
        return
    assert len(buttons) == 1
    buttons[0].click()
    QApplication.processEvents()


def _queue_scans_for_veto(panel, symbol: str, side: str = "LONG") -> None:
    panel.add_alert(_scan(symbol, side))
    panel.add_alert(_scan("NVDA", "SHORT"))
    _show_all_scans_if_the_view_has_the_new_control(panel)
    assert panel._current_review_alert is not None
    assert panel._current_review_alert.symbol == symbol


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
    from chart_watch import ChartWatchTrigger, load_chart_watches

    panel = _panel(tmp_path, monkeypatch)
    try:
        _queue_scans_for_veto(panel, "AAPL")

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
        assert watch.timeframes == ("M30", "H1")
        assert watch.watch_id and not watch.fired and not watch.declined
        assert load_chart_watches(tmp_path / "chart_watches.json") == [watch]
        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "NVDA", "the normal veto retirement still advances"

        # A later real armed hit remains visible despite today's veto.  It is
        # an alert the trader asked the desk to watch, never an ordinary scan.
        hit = panel._chart_watch_alert(
            ChartWatchTrigger(
                watch=watch,
                price=101.0,
                bar_dt=datetime(2026, 9, 22, 10, 30),
                message="M30 reclaim fired",
                resolved_side="long",
            ),
            datetime(2026, 9, 22, 10, 30),
        )
        assert hit.is_d1 is False and hit.timeframe == "D1"
        panel.add_alert(hit)
        assert hit in panel._alerts, "the real chart-watch event keeps its existing backing feed"
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
        _queue_scans_for_veto(panel, "MSFT")
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
        _queue_scans_for_veto(failed, "AMD")
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


def test_extended_day_trade_veto_arms_before_its_existing_focus_placement_and_retirement(
    tmp_path, monkeypatch
):
    """The day-trade button shares the saved veto exception, not its retirement timing."""
    from ui.widgets.capture_rail import _REASON_ROLE

    panel = _panel(tmp_path, monkeypatch)
    try:
        _queue_scans_for_veto(panel, "AAPL")
        rail = panel.chart_review.capture_rail
        for row in range(rail.reason_list.count()):
            item = rail.reason_list.item(row)
            if item.data(_REASON_ROLE) == VETO_CODE:
                rail.reason_list.setCurrentItem(item)
                break
        else:
            raise AssertionError("the loaded veto vocabulary has no extended row")

        assert rail.commit_veto_day_trade() is not None
        QApplication.processEvents()

        assert len(panel._chart_watches) == 1
        watch = panel._chart_watches[0]
        assert watch.symbol == "AAPL" and watch.side == "LONG"
        assert watch.source_text == VETO_SOURCE
        assert watch.timeframes == ("M30", "H1")
        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "NVDA"
    finally:
        panel.close()
        panel.deleteLater()


def test_extended_veto_limits_the_real_pullback_dispatch_and_keeps_the_existing_lifecycle(
    tmp_path, monkeypatch
):
    """M15 may be a companion fetch, but it is never a direct veto-watch leg."""
    import armed_alert_expiry
    from ui.panels import alert_center_panel as panel_module

    panel = _panel(tmp_path, monkeypatch)
    try:
        _queue_scans_for_veto(panel, "AAPL")
        _choose_extended_reason(panel.chart_review.capture_rail)
        watch = panel._chart_watches[0]

        class _Cache:
            def __init__(self, minutes):
                self.minutes = minutes
                self.asked = []

            def request(self, symbol, *, now):
                self.asked.append((symbol, now))

            def bars_for(self, _symbol):
                return []

        caches = {15: _Cache(15), 30: _Cache(30)}
        monkeypatch.setattr(panel, "_intraday_history_cache", lambda minutes: caches[minutes])
        jobs = []

        class _InlineThread:
            def __init__(self, *, target, args, **_kwargs):
                self._target = target
                self._args = args

            def start(self):
                self._target(*self._args)

        monkeypatch.setattr(panel_module.threading, "Thread", _InlineThread)
        monkeypatch.setattr(panel, "_run_pullback_sma_evaluation", lambda built, _now: jobs.extend(built))
        moment = datetime(2026, 9, 22, 11, 5)
        assert panel._dispatch_pullback_sma_evaluation([watch], moment)
        assert len(jobs) == 1
        assert [minutes for minutes, _sma, _end in jobs[0]["due"]] == [30]
        assert caches[30].asked, "the M30 leg must use the real cache request seam"

        # Persistent pullback watches reload across the day boundary, unlike a
        # session alert.  The existing public disarm path still removes one.
        from chart_watch import load_chart_watches

        assert load_chart_watches(tmp_path / "chart_watches.json", market_date="2026-09-23") == [watch]
        assert panel.disarm_chart_watch_for("AAPL", "pullback")
        assert load_chart_watches(tmp_path / "chart_watches.json") == []

        # Rebuild the exact stored watch as an old arm and run the panel's own
        # expiry poll.  The real expiry policy is ten TRADING days; the ledger
        # append alone is redirected to this test so no home-folder store moves.
        old_watch = replace(watch, armed_at=datetime(2026, 9, 1, 10, 0))
        panel._chart_watches = [old_watch]
        panel._save_chart_watches()
        expiry_rows = []
        monkeypatch.setattr(armed_alert_expiry, "record_expiries", lambda rows: expiry_rows.extend(rows) or len(rows))
        panel._poll_pullback_watches(now=datetime(2026, 9, 16, 12, 0))
        assert panel._chart_watches == []
        assert len(expiry_rows) == 1
        assert expiry_rows[0]["store"] == "chart_watches"
        assert expiry_rows[0]["symbol"] == "AAPL"
        assert expiry_rows[0]["kind"] == "pullback"
        assert expiry_rows[0]["trading_days"] == 10
    finally:
        panel.close()
        panel.deleteLater()


def test_only_the_saved_matching_extended_veto_can_request_the_watch(tmp_path, monkeypatch):
    """A failed capture, another reason, and a stale row each leave no new arm."""
    from ui.widgets import capture_rail as rail_module

    # A write failure is before the host signal: the chart is kept and no
    # watch can appear merely because a reason was selected.
    failed_root = tmp_path / "capture_failed"
    failed_root.mkdir()
    failed = _panel(failed_root, monkeypatch)
    try:
        _queue_scans_for_veto(failed, "AAPL")
        with monkeypatch.context() as scoped:
            scoped.setattr(rail_module, "record_annotation", lambda *_a, **_k: None)
            _choose_extended_reason(failed.chart_review.capture_rail)
        assert failed._chart_watches == []
        assert failed._current_review_alert.symbol == "AAPL"
    finally:
        failed.close()
        failed.deleteLater()

    other_root = tmp_path / "other_reason"
    other_root.mkdir()
    other = _panel(other_root, monkeypatch)
    try:
        _queue_scans_for_veto(other, "AAPL")
        _choose_another_reason(other.chart_review.capture_rail)
        assert other._chart_watches == []
        assert other._current_review_alert.symbol == "NVDA"

        # The host may receive a delayed rail signal after the chart changed.
        # It must never arm the old symbol or side from an otherwise matching
        # reason row.
        from ui.annotations.store import EVENT_VETO

        other.chart_symbol("MSFT", side="SHORT", origin="lookup")
        other.chart_review._on_captured(
            EVENT_VETO,
            {"symbol": "AAPL", "side": "LONG", "reason_code": VETO_CODE},
        )
        assert other._chart_watches == []
        assert other._current_review_alert is not None
        assert other._current_review_alert.symbol == "MSFT"
    finally:
        other.close()
        other.deleteLater()

    matching_root = tmp_path / "matching"
    matching_root.mkdir()
    matching = _panel(matching_root, monkeypatch)
    try:
        _queue_scans_for_veto(matching, "AAPL")
        _choose_extended_reason(matching.chart_review.capture_rail)
        assert len(matching._chart_watches) == 1
    finally:
        matching.close()
        matching.deleteLater()


def test_extended_veto_preserves_an_opposite_side_manual_pullback(tmp_path, monkeypatch):
    from chart_watch import PULLBACK_KIND

    panel = _panel(tmp_path, monkeypatch)
    try:
        _queue_scans_for_veto(panel, "AAPL", "SHORT")
        assert panel.arm_chart_watch_for("AAPL", "LONG", PULLBACK_KIND, source_text="chart")
        manual = panel._chart_watches[0]
        _choose_extended_reason(panel.chart_review.capture_rail)
        QApplication.processEvents()

        assert panel._chart_watches == [manual]
        assert "not armed" in panel.chart_review.capture_rail.status_label.text().casefold()
        assert "long pullback already armed" in panel.chart_review.capture_rail.status_label.text().casefold()
    finally:
        panel.close()
        panel.deleteLater()


def test_extended_veto_keeps_a_same_side_m15_only_watch_without_claiming_m30_h1(
    tmp_path, monkeypatch
):
    """A saved narrow manual watch is not the requested M30/H1 follow-up."""
    from chart_watch import PULLBACK_KIND, load_chart_watches

    panel = _panel(tmp_path, monkeypatch)
    try:
        _queue_scans_for_veto(panel, "AAPL")
        assert panel.arm_chart_watch_for(
            "AAPL", "LONG", PULLBACK_KIND,
            source_text="chart", timeframes=("M15",),
        )
        manual = panel._chart_watches[0]
        _choose_extended_reason(panel.chart_review.capture_rail)
        QApplication.processEvents()

        assert panel._chart_watches == [manual]
        assert load_chart_watches(tmp_path / "chart_watches.json") == [manual]
        assert "not armed" in panel.chart_review.capture_rail.status_label.text().casefold()
    finally:
        panel.close()
        panel.deleteLater()


def test_extended_veto_does_not_claim_legacy_h1_only_watch_covers_m30(
    tmp_path, monkeypatch
):
    from chart_watch import (
        PULLBACK_KIND,
        TRIGGER_H1_EMA15_BOUNCE,
        load_chart_watches,
        save_chart_watches,
    )

    panel = _panel(tmp_path, monkeypatch)
    try:
        _queue_scans_for_veto(panel, "AAPL")
        assert panel.arm_chart_watch_for("AAPL", "LONG", PULLBACK_KIND)
        legacy = replace(panel._chart_watches[0], triggers=(TRIGGER_H1_EMA15_BOUNCE,))
        panel._chart_watches = [legacy]
        save_chart_watches([legacy], tmp_path / "chart_watches.json")
        _choose_extended_reason(panel.chart_review.capture_rail)
        QApplication.processEvents()

        assert panel._chart_watches == [legacy]
        assert load_chart_watches(tmp_path / "chart_watches.json") == [legacy]
        assert "not armed" in panel.chart_review.capture_rail.status_label.text().casefold()
    finally:
        panel.close()
        panel.deleteLater()
