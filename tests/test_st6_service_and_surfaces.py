"""ST6 - BUILDER tests: the service's four triggers, and where the strip lives.

The tester's eleven (`tests/test_st6_working_lately.py`) pin the snapshot, the
events file and the switch. These are the ones the packet asks the BUILDER to
add, and each covers a seam the eleven do not reach:

* the four refresh triggers, and that all four fold into ONE coalesced reaction;
* the strip's position - the TOP of the M5 alerts column, above the list, with
  the column's own two panes untouched - and its click-through to the Setup
  Tracker;
* the AWAY digest's Working-lately line, in the EXISTING body, absent when there
  is no snapshot;
* the AWAY Recap's `alert_cell` + held x ran suffix travelling rather than being
  recomputed;
* the `observational leader among K cells` caveat, which is the sentence that
  keeps the strip from reading as a claim.

Every one of these was run against the pre-change file and seen to FAIL first.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module", autouse=True)
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _last_completed_session() -> date:
    import market_calendar

    return market_calendar.last_completed_session(datetime.now(market_calendar.MARKET_TZ))


def _recent_row(family: str, *, wins: int, losses: int, session: date, side: str = "LONG") -> dict:
    return {
        "namespace": "live",
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "closed_setups": str(wins + losses),
        "tracked_setups": str(wins + losses),
        "avg_closed_r": "0.5",
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "0",
        "n_unmeasured": "0",
        "n_pending": "0",
        "n_symbols": "12",
        "n_entry_sessions": "9",
        "outcome_kind": "trade_r_representative_exit",
        "outcome_version": "recent_types_v2",
        "knowledge_basis": "entry_scan_row_close_to_representative_exit",
        "horizon_basis": "30d lookback, representative exit",
        "latest_measured_session": session.isoformat(),
    }


def _snapshot_payload(tmp_path: Path) -> dict:
    """A published snapshot with a real leader, through the real service."""
    from ui.services.working_lately_service import WorkingLatelyService
    from working_lately import build_snapshot

    service = WorkingLatelyService(store_dir=tmp_path / "wl")
    sessions = [_last_completed_session()]
    import market_calendar

    sessions.insert(0, market_calendar.previous_session(sessions[0]))
    for session in sessions:
        snapshot = build_snapshot(
            recent_rows=[
                _recent_row("alpha", wins=45, losses=15, session=session),
                _recent_row("beta", wins=30, losses=30, session=session),
            ],
            last_completed_session=session,
            previous_verdicts=service.previous_verdicts(),
        )
        service.publish(snapshot)
    payload = service.load_snapshot()
    assert payload["verdicts"]["swing_trade_r"]["state"] == "leader"
    return payload


# ===========================================================================
# the four triggers
# ===========================================================================


def test_all_four_triggers_route_through_one_coalescer_and_a_burst_is_one_build(tmp_path):
    """Window-shown, day roll, post-export and the 30-minute timer.

    Four callers, ONE reaction. A day roll that lands beside a finished scan
    must not be two builds of the same evidence, which is the listener-side
    coalescing rule the desk applies everywhere else.
    """
    from ui.services import working_lately_service as svc

    service = svc.WorkingLatelyService(store_dir=tmp_path / "wl")
    builds: list[int] = []
    service._refresh_now = lambda: builds.append(1)  # type: ignore[method-assign]
    service._coalescer._target = service._refresh_now

    # (a) the window shows - once, and idempotent.
    service.start()
    assert service._timer.isActive(), "the 30-minute timer is the fourth trigger"
    service.start()
    # (b) the day roll, (c) the export, (d) the timer's own tick.
    service.on_day_roll()
    service.on_tracker_export()
    service._timer.timeout.emit()
    service.request_refresh()

    # Nothing has fired yet - the window is open and the reaction is owed.
    assert builds == []
    service._coalescer.flush()
    assert builds == [1], "five requests inside one window are one build"

    service._coalescer.flush()
    assert builds == [1], "a flush with nothing owed does nothing"
    service.shutdown()
    assert not service._timer.isActive()


def test_a_second_request_while_a_build_is_running_does_not_start_a_second(tmp_path):
    """Single-flight. A build in flight already answers the request behind it."""
    from ui.services import working_lately_service as svc

    service = svc.WorkingLatelyService(store_dir=tmp_path / "wl")
    started: list[int] = []

    class _Signal:
        def connect(self, *_args):
            pass

    class _Fake:
        def __init__(self, *args, **kwargs):
            started.append(1)
            self.built = _Signal()
            self.failed = _Signal()
            self.finished = _Signal()

        def start(self):
            pass

        def deleteLater(self):  # noqa: N802 - Qt's own spelling
            pass

    real = svc._BuildWorker
    svc._BuildWorker = _Fake  # type: ignore[misc]
    try:
        service._refresh_now()
        service._refresh_now()
        assert started == [1], "the second request rides the build already running"
        service._clear_inflight()
        service._refresh_now()
        assert started == [1, 1]
    finally:
        svc._BuildWorker = real
        service._worker = None


# ===========================================================================
# where the strip lives, and what happens when it is clicked
# ===========================================================================


def test_the_strip_is_the_top_of_the_m5_column_and_the_column_keeps_its_two_panes():
    """Above the M5 list, inside the alerts widget.

    Mounted INSIDE the bar rather than as a third splitter child on purpose:
    the M5 column is a saved, draggable two-pane split (the alert list over the
    swing favorites strip) and a third child would have the trader's saved sizes
    replayed onto a layout they were never dragged for.
    """
    from ui.panels.trading_desk import TradingDeskPanel
    from ui.widgets.working_lately_strip import WorkingLatelyStrip

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        assert isinstance(desk.working_lately_strip, WorkingLatelyStrip)
        # The column is untouched: two panes, the alert bar on top.
        assert desk.m5_column.count() == 2
        assert desk.m5_column.widget(0) is desk.m5_alert_bar
        # And inside the alert bar, the strip is the first thing.
        assert desk.m5_alert_bar.layout().itemAt(0).widget() is desk.working_lately_strip
        # The list is BELOW it - "above the M5 list" is the trader's words.
        strip_top = desk.working_lately_strip.geometry().top()
        list_top = desk.m5_alert_bar.list.geometry().top()
        assert strip_top <= list_top
    finally:
        desk.shutdown()
        desk.close()


def test_clicking_the_strip_asks_for_the_setup_tracker():
    """The strip is a door, not a report. One signal, and the desk relays it."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    asked: list[int] = []
    desk.workingLatelyOpenRequested.connect(lambda: asked.append(1))
    try:
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(2, 2),
            QPointF(2, 2),
            QPointF(2, 2),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        desk.working_lately_strip.mouseReleaseEvent(event)
        assert asked == [1]
    finally:
        desk.shutdown()
        desk.close()


def test_the_research_page_can_raise_the_setup_tracker_tab():
    """The click-through's other half: the tab the strip points at."""
    from ui.panels.research_panel import ResearchPanel

    panel = ResearchPanel(None)
    try:
        panel.tabs.setCurrentIndex(0)
        panel.show_setup_tracker()
        assert panel.tabs.currentWidget() is panel.setup_tracker_panel
    finally:
        panel.shutdown()
        panel.deleteLater()


# ===========================================================================
# the strip's own words
# ===========================================================================


def test_the_strip_prints_the_observational_caveat_and_never_the_word_proven(tmp_path):
    """The trader: *"do not call every winner proven."*

    The multiple-testing exposure is PRINTED rather than corrected: K cells were
    read and the best of K was named. The sentence is the honesty, and it is on
    the line the trader actually sees - not only in the tooltip.
    """
    from ui.widgets.working_lately_strip import WorkingLatelyStrip

    payload = _snapshot_payload(tmp_path)
    strip = WorkingLatelyStrip()
    try:
        strip.set_snapshot(payload)
        line = strip.line_text()
        tooltip = strip.tooltip_text()
    finally:
        strip.deleteLater()

    assert "observational leader among 2 cells" in line, line
    assert "proven" not in line.lower(), line
    assert "proven" not in tooltip.lower(), tooltip
    # Every cell is in the tooltip, each saying what it measured and on how much.
    assert "LONG alpha" in tooltip and "LONG beta" in tooltip
    assert "win rate (closed basis)" in tooltip
    assert "wilson_lower_bound_95" in tooltip


def test_an_empty_snapshot_says_so_rather_than_showing_a_blank_line():
    """No snapshot is a sentence, not an empty strip."""
    from ui.widgets.working_lately_strip import WorkingLatelyStrip

    strip = WorkingLatelyStrip()
    try:
        assert "no snapshot yet" in strip.line_text().lower()
        assert strip.prioritise_box.isChecked() is False, "the switch defaults OFF"
    finally:
        strip.deleteLater()


# ===========================================================================
# ST6.6 - the AWAY digest line and the recap suffix
# ===========================================================================


def test_the_away_digest_carries_the_working_lately_line_in_its_existing_body(tmp_path):
    """No new push: the line rides `autopilot_today.txt`, which AWAY already writes.

    And an ABSENT snapshot is an ABSENT SECTION - a phone report is the worst
    possible place to invent a sentence about evidence nobody read.
    """
    import autopilot_core

    payload = _snapshot_payload(tmp_path)
    import working_lately

    line = f"{working_lately.snapshot_line(payload)} [{working_lately.snapshot_stamp(payload)}]"

    with_line = autopilot_core.render_away_report(
        {"generated_at": "2026-09-06T13:00:00", "working_lately_line": line}
    )
    assert "== WORKING LATELY ==" in with_line
    assert "Working lately (20 sessions):" in with_line
    assert "alpha" in with_line
    # Still parseable: the swings block is found by its own header.
    assert "== BEST SWING TRADES ==" in with_line

    without = autopilot_core.render_away_report({"generated_at": "2026-09-06T13:00:00"})
    assert "== WORKING LATELY ==" not in without


def test_the_away_digest_line_reads_the_published_snapshot_and_never_rebuilds_one(
    tmp_path, monkeypatch
):
    """The digest and the strip are ONE reading, so the digest must not build."""
    from ui.services import autopilot_service as svc
    from ui.services import working_lately_service as wl

    payload = _snapshot_payload(tmp_path)
    monkeypatch.setattr(wl, "read_persisted_snapshot", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        wl,
        "read_recent_rows",
        lambda: (_ for _ in ()).throw(AssertionError("the digest rebuilt the reading")),
    )
    assert payload["snapshot_id"][:8] in svc._working_lately_report_line()

    monkeypatch.setattr(wl, "read_persisted_snapshot", lambda *_a, **_k: {})
    assert svc._working_lately_report_line() == ""


def test_the_recap_alert_rows_carry_the_cell_and_the_held_run_suffix():
    """ST6.6. Both TRAVEL from the M5 row; this page classifies nothing."""
    import away_recap

    recap = away_recap.build_recap(
        session_date="2026-09-04",
        alerts=[
            {
                "symbol": "AAA",
                "side": "LONG",
                "tier": "S",
                "trigger": "[S-TIER] ema_15",
                "time_text": "10:00:00",
                "is_d1": False,
                "cell": "ema_15 LONG",
                "held_run_suffix": "held 62% / ran 1.4R (n=31)",
            }
        ],
    )
    row = recap["classified_alerts"][0]
    assert row["cell"] == "ema_15 LONG"
    assert row["held_run_suffix"] == "held 62% / ran 1.4R (n=31)"


def test_the_recap_summary_names_the_sessions_leader_changes_and_their_cause(tmp_path):
    """A recap that said "the leader changed" without the cause would teach the
    trader that the desk found something, when a restatement found nothing."""
    import away_recap

    payload = _snapshot_payload(tmp_path)
    events = [
        {
            "kind": "swing_trade_r",
            "prior_leader": "",
            "new_leader": "LONG alpha",
            "cause": "window_rollover",
            "as_of": payload["as_of"],
            "new_snapshot_id": payload["snapshot_id"],
        },
        {
            "kind": "swing_trade_r",
            "prior_leader": "LONG alpha",
            "new_leader": "LONG beta",
            "cause": "corrected_data",
            "as_of": "1999-01-04",
            "new_snapshot_id": "other",
        },
    ]
    recap = away_recap.build_recap(
        session_date=payload["as_of"],
        digest_swings=[],
        working_lately=payload,
        leader_events=events,
    )
    assert len(recap["leader_changes"]) == 1, "another session's event is not this one's"
    assert "window_rollover" in recap["summary"]
    assert "corrected_data" not in recap["summary"]
    assert payload["snapshot_id"][:8] in recap["summary"]


# ===========================================================================
# the banner is the SERVICE's snapshot, and says so when it is not
# ===========================================================================


def test_the_tracker_banner_labels_its_own_read_and_drops_the_label_with_a_snapshot(tmp_path):
    """The panel read is a FALLBACK and is labelled. An unlabelled fallback is
    how the desk grew two answers to one question in the first place."""
    from ui.panels import setup_tracker_panel

    payload = _snapshot_payload(tmp_path)
    panel = setup_tracker_panel.SetupTrackerPanel()
    try:
        fallback = setup_tracker_panel._best_now_banner_html(panel)
        assert "panel read" in fallback
        panel.set_working_lately_snapshot(payload)
        shared = setup_tracker_panel._best_now_banner_html(panel)
    finally:
        panel.shutdown()
        panel.deleteLater()

    assert "panel read" not in shared
    assert payload["snapshot_id"][:8] in shared
    assert "observational leader among" in shared
    assert "alpha" in shared


def test_the_summary_card_and_the_banner_render_one_verdict_not_two(tmp_path):
    """ST2's fix round put the Summary card and the banner on ONE `select_leader`.

    ST6 must not undo that by a different door: with a snapshot present the
    banner renders the SHARED verdict, so the card three lines above it has to
    render the same one. The case that would have split them is the persistence
    rule - a family the panel's own read crowns while the snapshot is still
    holding it for a second distinct `as_of`.
    """
    import research_explanations
    from ui.panels import setup_tracker_panel
    import working_lately

    payload = _snapshot_payload(tmp_path)
    shared = working_lately.verdicts_from_payload(payload)["swing_trade_r"]
    assert shared.state == "leader" and shared.leader is not None

    panel = setup_tracker_panel.SetupTrackerPanel()
    try:
        panel.set_working_lately_snapshot(payload)
        # ONE computation per page, and the snapshot IS it.
        verdicts = setup_tracker_panel.panel_verdicts(panel)
        assert verdicts["swing"].reason == shared.reason
        assert verdicts["swing"].leader["setup_family"] == "alpha"
        assert verdicts["snapshot"]["snapshot_id"] == payload["snapshot_id"]
        html = setup_tracker_panel._best_now_banner_html(panel, verdicts)
    finally:
        panel.shutdown()
        panel.deleteLater()
    assert "alpha" in html

    card = research_explanations.build_plain_english_whats_working(
        recent_rows=[],  # deliberately EMPTY: the card must not re-decide
        verdicts=verdicts,
    )
    swing_bullets = [
        line for line in card["bullets"] if line.startswith("Among recently closed swings")
    ]
    assert len(swing_bullets) == 1, card["bullets"]
    assert "alpha" in swing_bullets[0], swing_bullets[0]

    # The panel read alone, with no rows, could never have named a family.
    empty = research_explanations.build_plain_english_whats_working(recent_rows=[])
    assert not any(
        "alpha" in line for line in empty["bullets"]
    ), "the card named a family it was not handed - it re-decided"


def test_the_persisted_snapshot_is_small_enough_to_read_by_eye(tmp_path):
    """Ground rule: a display cache is not an evidence store. Two families and
    two kinds of absent source must not produce a file nobody will ever open."""
    payload = _snapshot_payload(tmp_path)
    text = json.dumps(payload)
    assert len(text) < 64_000, len(text)
    assert set(payload["sources"]) >= {
        "swing_trade_r",
        "swing_favorable",
        "daytrade_held_run",
    }
    # An ABSENT source is None, never a zero: a question that was not asked.
    assert payload["sources"]["swing_favorable"]["rows"] is None
