"""The Focus fade where the trader meets it - Phase 0.12 A3/A4.

The store's rules are pinned in `test_focus_fade.py`. This file pins the desk
side of them: which surfaces reset the clock, that the faded walkthrough goes
through the ONE door with a tag the movers-only filter never touches, that the
two verbs restore and discard, and that the buttons say how many.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _qt():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])


def _service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    return FocusService(
        FocusPickStore(
            focus_longs_path=tmp_path / "focus_longs.txt",
            focus_shorts_path=tmp_path / "focus_shorts.txt",
            longs_path=tmp_path / "longs.txt",
            shorts_path=tmp_path / "shorts.txt",
            membership_path=tmp_path / "focus_pick_membership.json",
        )
    )


def _panel(tmp_path, service):
    from ui.panels.alert_center_panel import AlertCenterPanel

    return AlertCenterPanel(
        service,
        ignored_symbols_path=tmp_path / "ignored.txt",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "review_events.jsonl",
        focus_d1_flags_path=tmp_path / "focus_d1_flags.json",
        d1_event_watches_path=tmp_path / "d1_events.json",
    )


def test_an_armed_watch_hit_resets_the_fade_clock(tmp_path, monkeypatch):
    """Every armed poll builds its alert through `_chart_watch_alert`, so one
    reset there covers the level, event and any-bounce lanes."""
    try:
        _qt()
        from ui.panels.alert_center_panel import AlertCenterPanel  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    from chart_watch import D1EventWatch

    service = _service(tmp_path)
    service.store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    panel = _panel(tmp_path, service)

    class _Hit:
        message = "new 20d high"
        resolved_side = "LONG"
        watch = D1EventWatch("NVDA", "new_20d_high", datetime(2026, 8, 10, 9, 40))

    panel._chart_watch_alert(_Hit(), datetime(2026, 8, 10, 9, 40))
    assert service.store.pick_clock("NVDA", "long", "swing") == date.today()


def test_keeping_a_pick_on_the_review_chart_resets_the_fade_clock(tmp_path):
    try:
        _qt()
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    from ui.models.bounce import FOCUS_REVIEW_TAG, BounceAlert

    service = _service(tmp_path)
    service.store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    panel = _panel(tmp_path, service)

    panel._add_review_alert_to_focus(
        BounceAlert(
            time_text="09:40:00",
            symbol="NVDA",
            side="LONG",
            trigger="Focus review",
            timeframe="D1",
            tag=FOCUS_REVIEW_TAG,
            raw_text="FOCUS REVIEW NVDA",
        )
    )
    assert service.store.pick_clock("NVDA", "long", "swing") == date.today()


def test_the_faded_walkthrough_queues_through_the_one_door_and_beats_movers_only(
    tmp_path, monkeypatch
):
    """A faded pick is by definition one that has NOT been moving, so the
    movers-only presentation filter would hide every single row of the list
    the trader just asked to see."""
    try:
        _qt()
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    from ui.models.bounce import FOCUS_FADED_TAG

    service = _service(tmp_path)
    service.store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    service.store.fade_stale_picks(today=date(2026, 8, 17))
    panel = _panel(tmp_path, service)
    # Movers-only is on by default and every symbol reads "inside yesterday's
    # range" here - the strictest case the filter can present.
    assert panel._review_movers_only is True
    monkeypatch.setattr(panel, "_review_chart_state", lambda alert: "closed")

    panel.review_faded_picks()

    charted = [panel._current_review_alert] + list(panel._review_queue)
    charted = [alert for alert in charted if alert is not None]
    assert [alert.symbol for alert in charted] == ["NVDA"]
    assert charted[0].tag == FOCUS_FADED_TAG
    assert panel.hidden_inside_range_count() == 0


def test_the_faded_verbs_restore_with_a_fresh_clock_and_discard(tmp_path):
    try:
        _qt()
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    from ui.models.bounce import FOCUS_FADED_TAG, BounceAlert

    service = _service(tmp_path)
    service.store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    service.store.add("AMD", "long", "swing", today=date(2026, 8, 3))
    service.store.fade_stale_picks(today=date(2026, 8, 17))
    panel = _panel(tmp_path, service)

    def _alert(symbol):
        return BounceAlert(
            time_text="09:40:00",
            symbol=symbol,
            side="LONG",
            trigger="Faded swing long",
            timeframe="D1",
            tag=FOCUS_FADED_TAG,
            raw_text=f"FADED {symbol}",
            payload={"faded_side": "long", "faded_category": "swing"},
        )

    panel._add_review_alert_to_focus(_alert("NVDA"))
    assert service.store.focus_symbols("long", "swing") == ["NVDA"]
    assert service.store.pick_clock("NVDA", "long", "swing") == date.today()

    panel._remove_review_alert_for_today(_alert("AMD"))
    assert service.store.focus_symbols("long", "swing") == ["NVDA"]
    assert service.store.faded_picks() == []


def test_the_two_buttons_carry_their_counts(tmp_path):
    try:
        _qt()
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    from ui.widgets.focus_strength_board import FocusStrengthBoard

    service = _service(tmp_path)
    service.store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    service.store.add("AMD", "long", "m5", today=date(2026, 8, 17))

    board = FocusStrengthBoard()
    board.set_focus_service(service)
    assert board.review_button.text() == "Focus pick review (2)"
    assert board.faded_button.text() == "Faded review (0)"
    assert board.faded_button.isEnabled() is False

    service.store.fade_stale_picks(today=date(2026, 8, 17))
    board.flush_pending_refresh()
    assert board.review_button.text() == "Focus pick review (1)"
    assert board.faded_button.text() == "Faded review (1)"
    assert board.faded_button.isEnabled() is True


def test_fading_a_hand_vetted_swing_pick_appends_a_retraction_never_an_edit(tmp_path):
    try:
        _qt()
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise
    import swing_favorites
    from ui.services.swing_favorites_service import ORIGIN_FADE, SwingFavoritesService

    service = _service(tmp_path)
    favorites = SwingFavoritesService(service, path=tmp_path / "swing_favorites.jsonl")
    assert favorites.add("NVDA", "long") == ["NVDA"]
    # The strip writes with today's clock; back-date it so the fade is due.
    service.store.note_focus_activity("NVDA", today=date(2026, 8, 3))
    # A name that was never hand-vetted here must not gain a retraction row.
    service.store.add("AMD", "long", "swing", today=date(2026, 8, 3))

    faded = service.store.fade_stale_picks(today=date(2026, 8, 31))
    assert {row["symbol"] for row in faded} == {"NVDA", "AMD"}
    assert favorites.retract_faded_picks(faded) == 1

    rows = swing_favorites.load_rows(tmp_path / "swing_favorites.jsonl")
    assert [(row["symbol"], row["action"]) for row in rows] == [
        ("NVDA", swing_favorites.ACTION_ADD),
        ("NVDA", swing_favorites.ACTION_REMOVE),
    ]
    assert rows[1]["origin"] == ORIGIN_FADE
