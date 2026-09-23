"""Day Recap step A: the Day Review page cleaned up (trader, 2026-09-23).

One clock, plain words, a glance strip, day navigation, one miss table, trade
charts, no Qt-thread parquet reads, and a no-trade day that is not an error.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

SESSION = "2026-09-22"
NOW = datetime(2026, 9, 23, 7, 30)
PACIFIC = ZoneInfo("America/Los_Angeles")


class _Service:
    """Reads nothing: every test here renders a payload by hand."""

    def read_day(self, session_date, **_kwargs):
        from ui.services.day_review_service import empty_payload

        return empty_payload(session_date)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app, monkeypatch):
    from ui.panels import day_review_panel

    monkeypatch.setattr(day_review_panel, "_display_zone", lambda: PACIFIC, raising=False)
    widget = day_review_panel.DayReviewPanel(service=_Service(), clock=lambda: NOW)
    widget.show_session(SESSION)
    # Let the fixture's own (empty) read land first, so it cannot repaint over
    # a payload a test renders by hand.
    for _ in range(100):
        worker = widget._worker
        if worker is None:
            break
        worker.wait(50)
        app.processEvents()
    yield widget
    widget.shutdown()
    widget.deleteLater()
    app.processEvents()


def _payload(**extra):
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload.update(extra)
    return payload


# -- 1. one clock ------------------------------------------------------------
def test_a_utc_note_stamp_is_shown_in_the_desk_zone(panel):
    """The live bug: a 07:43 PT call stored as 14:43 UTC read "14:43"."""
    panel.render(_payload(entries=[{
        "entry_id": "e1", "created_at": "2026-09-22T14:43:24+00:00",
        "timeframe": "M5", "text": "choppy open",
    }]))
    assert panel.entries.item(0).text().startswith("07:43"), panel.entries.item(0).text()
    panel.entries.setCurrentRow(0)
    assert "written 07:43" in panel.entry_meta.text()


def test_the_calls_table_shows_a_clock_not_raw_iso(panel):
    panel.render(_payload(reads=[{
        "entry_id": "e1", "stamp": "2026-09-22T07:43:24.344534-07:00",
        "horizon": "rest_of_day", "direction": "chop", "confidence": "medium",
        "verdict": "right",
    }]))
    assert panel.calls_table.item(0, 0).text() == "07:43"


def test_trade_times_are_in_the_desk_zone_with_a_date_only_off_session(panel):
    panel.render(_payload(
        trades=[{"trade_id": "t1", "symbol": "DRAM", "opened_at": "2026-09-22T12:33:41.781000-04:00"}],
        trade_reviews=[{
            "trade_id": "t1", "symbol": "DRAM",
            "opened_at": "2026-09-22T12:33:41.781000-04:00",
            "closed_at": "2026-09-23T10:07:28.588000-04:00",
            "net_pnl": -17.78, "currency": "USD",
        }],
    ))
    assert panel.trades_table.item(0, 0).text() == "09:33"
    detail = panel.trade_detail.toPlainText()
    assert "Opened 09:33" in detail
    assert "Closed Sep 23 07:07" in detail
    assert "-04:00" not in detail


def test_the_zone_is_named_once_in_the_header(panel):
    panel.render(_payload())
    assert panel.zone_note.text() == "Times in PDT"


# -- 8. a no-trade day is not an error ---------------------------------------
def test_a_no_trade_day_is_not_a_failure_in_the_status_line(panel):
    from ui.services.day_review_service import NO_TRADES_EXIT_NOTE

    panel.render(_payload(error=NO_TRADES_EXIT_NOTE))
    assert panel.status.text() == f"Day Review: {SESSION}"


def test_a_real_failure_still_shows_beside_the_no_trade_fact(panel):
    from ui.services.day_review_service import NO_TRADES_EXIT_NOTE

    panel.render(_payload(error=f"the open theses could not be read: boom · {NO_TRADES_EXIT_NOTE}"))
    assert panel.status.text() == "the open theses could not be read: boom"


# -- 7. no Qt-thread parquet reads -------------------------------------------
def test_no_bars_file_is_read_on_the_qt_thread(app, monkeypatch):
    """`_backfill_bars_for` and the per-exit loop in `render` read on a worker."""
    import threading

    import day_review_bars
    from ui.panels.day_review_panel import DayReviewPanel

    reads: list[tuple[str, int]] = []

    def _read(session, **_kwargs):
        reads.append((str(session), threading.get_ident()))
        return {"SPY": []}

    class _Service(_BarsService):
        pass

    monkeypatch.setattr(day_review_bars, "read_session_bars", _read)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *_a, **_k: True)
    panel = DayReviewPanel(service=_Service(), clock=lambda: NOW)
    try:
        gui = threading.get_ident()
        panel._backfill_bars_for("2026-09-18")
        panel.render(_payload(walkaway_backfill_sessions=("2026-09-17",)))
        for _ in range(200):
            worker = panel._bars_worker
            if worker is not None:
                worker.wait(2000)
            app.processEvents()
            if len(reads) >= 2 and (panel._bars_worker is None or not panel._bars_worker.isRunning()):
                break
        assert {session for session, _ in reads} == {"2026-09-18", "2026-09-17"}
        assert all(thread != gui for _session, thread in reads), reads
        # The file was there, so nothing was fetched, and it is not asked twice.
        assert panel.service.fetched == []
        before = len(reads)
        panel._backfill_bars_for("2026-09-18")
        assert len(reads) == before
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


class _BarsService(_Service):
    def __init__(self):
        self.fetched: list[str] = []

    def backfill_session_bars_for(self, session_date, **_kwargs):
        self.fetched.append(str(session_date))


def test_the_forecast_dialog_is_non_modal_and_prefills_no_source(panel, app, monkeypatch):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox

    imported: list[dict] = []
    monkeypatch.setattr(panel, "_import_forecast", lambda values: imported.append(values) or {})
    monkeypatch.setattr(
        QDialog, "exec", lambda *_a: pytest.fail("the forecast dialog blocked the desk")
    )
    panel._paste_daily_forecast()
    dialog = panel._forecast_dialog
    assert dialog.isVisible() and not dialog.isModal()
    assert dialog.model_box.text() == ""
    dialog.text_box.setPlainText("# Market Morning Brief")
    dialog.buttons.button(QDialogButtonBox.Ok).click()
    assert imported and imported[0]["text"] == "# Market Morning Brief"
    assert imported[0]["source_model"] == ""


# -- shared fixtures for items 2-6 ---------------------------------------------
def _row(symbol, *, ran=5.0, real="run", time=None):
    from walkaway_day import WalkawayRow

    return WalkawayRow(
        decision_id=(SESSION, symbol, "LONG", "chart_review", "veto", "D1", ""),
        time=time or datetime(2026, 9, 22, 13, 0, tzinfo=ZoneInfo("America/New_York")),
        symbol=symbol, side="LONG", category="chart_review", what_you_did="veto",
        ran_after_pct=ran, real_miss=real, state="measured",
    )


def _day():
    from walkaway_day import WalkawayDay

    return WalkawayDay(
        rejected=(_row("SGRY", ran=6.64), _row("STX", ran=4.31, real="no_run")),
        liked_not_traded=(_row("AAA", ran=2.0),),
        claimed_d1=(_row("GMAB"), _row("HAWK"), _row("KKR")),
    )


def _bars():
    return [
        {"dt": datetime(2026, 9, 22, 6, 30 + 5 * i), "open": 10.0, "high": 11.0,
         "low": 9.0, "close": 10.5, "volume": 100}
        for i in range(6)
    ]


def _card(**lines):
    return {"lines": [{"key": key, **value} for key, value in lines.items()]}


# -- 2. plain words ------------------------------------------------------------
def test_plain_words_drops_ids_packets_and_zero_clauses():
    import day_report_card

    text = day_report_card.plain_words(
        "Congruence: 1 of 4 checks measured. · picks_side_mix: 22 of 42 were LONG "
        "· desk_d1_label unmeasured - missing: the label · fills_bias unmeasured"
    )
    for word in ("picks_side_mix", "desk_d1_label", "fills_bias", "unmeasured"):
        assert word not in text, text
    assert "not measured" in text and "your D1 picks' sides" in text
    text = day_report_card.plain_words(
        "Process: 3 trade(s). Focus adds are not read yet (TJ-12F), so be careful. "
        "0 labelled (claimed_before_entry 0, same_session 0), 3 unlabelled. "
        "Exits explained 0 of 3, 0 of those confirmed by you."
    )
    assert "TJ-" not in text and "0 labelled" not in text and "0 of those" not in text, text
    assert "3 unlabelled" in text and "Exits explained 0 of 3." in text
    assert day_report_card.plain_words(
        "Your reads: Rest of day: 5 finished, 3 right, 0 wrong; with the D1 environment scored 0 of 0"
    ) == "Your reads: Rest of day: 5 finished, 3 right"


def test_the_card_the_notes_and_the_chart_note_speak_plain_words(panel):
    from ui.panels.day_review_panel import NO_CHART_NOTE

    assert "TJ-" not in NO_CHART_NOTE
    panel.render(_payload(
        report_card=_card(congruence={"text": "Congruence: picks_side_mix: 3 of 3 LONG (TJ-12F)"}),
        congruence=({"kind": "desk_d1_label", "text": "fills_bias read", "verdict": "unmeasured"},),
        entries=[{"entry_id": "e1", "created_at": "2026-09-22T14:43:24+00:00",
                  "timeframe": "M5", "text": "chop", "origin": "trade_mentor"}],
        reads=[{"entry_id": "e1", "stamp": "2026-09-22T07:43:00-07:00",
                "horizon": "next_5_sessions", "direction": "up", "verdict": "right"}],
    ))
    card = panel._report_card_lines["congruence"].text()
    assert "picks_side_mix" not in card and "TJ-" not in card, card
    assert "fills_bias" not in panel.congruence_text()
    assert "unmeasured" not in panel.congruence_text()
    assert panel.calls_table.item(0, 1).text() == "next 5 sessions"
    panel.entries.setCurrentRow(0)
    assert "trade_mentor" not in panel.entry_meta.text()


def test_story_citation_ids_are_not_shown(panel):
    panel.render(_payload(d1_view={"narration": {
        "belief_now": "SPY holds its gains [said:mj-2026-09-22-b6559bf2669f:prediction].",
    }}))
    assert "[said:" not in panel.d1_view_note.text()
    assert "SPY holds its gains" in panel.d1_view_note.text()


# -- 3. the glance strip -------------------------------------------------------
def _glance_payload(**extra):
    return _payload(
        trades=[
            {"trade_id": "t1", "symbol": "DRAM", "net_pnl": -17.78},
            {"trade_id": "t2", "symbol": "DRAM", "net_pnl": -9.96},
            {"trade_id": "t3", "symbol": "ZETA", "net_pnl": 33.60},
        ],
        report_card=_card(
            process={"text": "Process: 3 trade(s).", "planned": 0, "unplanned": 3, "unmeasured": 0},
            your_reads={"text": "Your reads.", "n": 5, "right": 3, "wrong": 0, "flat": 2, "pending": 2},
        ),
        walkaway=_day(),
        **extra,
    )


def test_the_glance_strip_reads_the_day_at_a_glance(panel):
    panel.render(_glance_payload())
    strip = panel.glance_strip
    assert strip.tile_value("pnl") == "+$5.86"
    assert strip.tile_value("trades") == "1W / 2L"
    assert strip.tile_value("planned") == "0 / 3"
    assert strip.tile_value("calls") == "3 / 0 / 2"
    assert strip.tile_value("day_type") == "not labelled"
    assert strip.tile_value("biggest_win") == "ZETA +$33.60"
    assert strip.tile_value("biggest_miss") == "SGRY +6.64%"
    for tile in strip.tiles.values():
        assert tile.toolTip().strip(), tile.text()


def test_unknown_is_never_shown_as_zero_on_the_strip(panel):
    panel.render(_payload(trades=[{"trade_id": "t1", "symbol": "ABC", "net_pnl": None}]))
    assert panel.glance_strip.tile_value("pnl") == "not measured"
    panel.render(_payload())
    assert panel.glance_strip.tile_value("pnl") == "no trades"
    assert panel.glance_strip.tile_value("calls") == "no calls"


def test_a_stored_day_type_and_the_sparkline_are_shown(panel):
    panel.render(_glance_payload(
        day_type="trend_up",
        pnl_by_session=(("2026-09-18", 10.0), ("2026-09-21", None), (SESSION, 5.86)),
    ))
    assert panel.glance_strip.tile_value("day_type") == "trend up"
    assert panel.glance_strip.sparkline.values() == [10.0, None, 5.86]


def test_a_tile_click_opens_its_section(panel):
    panel.render(_glance_payload())
    panel.select_miss_population("claimed_d1")
    panel.glance_strip.tiles["biggest_miss"].click()
    assert panel.miss_population() == "rejected"
    assert panel.miss_table.currentRow() == 0
    revealed: list[str] = []
    panel.reveal_card_target = lambda target: revealed.append(target)
    panel.glance_strip.tiles["calls"].click()
    assert revealed == ["said"]


def test_the_full_card_sits_under_a_details_toggle(panel):
    panel.render(_glance_payload())
    assert not panel.report_card_section.isVisibleTo(panel)
    panel.details_toggle.setChecked(True)
    assert panel.report_card_section.isVisibleTo(panel)


def test_the_service_builds_the_glance_on_the_worker():
    from ui.services.day_review_service import PAYLOAD_KEYS, _pnl_by_session, empty_payload

    assert {"glance", "day_type", "pnl_by_session"} <= set(PAYLOAD_KEYS)
    assert empty_payload(SESSION)["glance"] == {}
    rows = [
        {"trade_date": SESSION, "net_pnl": 5.0},
        {"trade_date": SESSION, "net_pnl": -2.0},
        {"trade_date": "2026-09-21", "net_pnl": None},
    ]
    assert _pnl_by_session(("2026-09-21", SESSION), rows) == (("2026-09-21", None), (SESSION, 3.0))


# -- 4. day navigation ---------------------------------------------------------
def test_arrows_and_alt_keys_step_one_session(panel):
    start = panel.session_picker.currentIndex()
    older = panel.session_picker.itemData(start + 1)
    panel.prev_session_button.click()
    assert panel.session_date() == older
    panel.session_shortcuts["Alt+Right"].activated.emit()
    assert panel.session_picker.currentIndex() == start
    panel.session_shortcuts["Alt+Left"].activated.emit()
    assert panel.session_date() == older
    assert panel.prev_session_button.toolTip().endswith("(Alt+Left)")


def test_the_newest_session_disables_the_next_arrow(panel):
    panel.session_picker.setCurrentIndex(0)
    assert not panel.next_session_button.isEnabled()
    assert panel.prev_session_button.isEnabled()


def test_the_selected_session_is_remembered(app, monkeypatch):
    from project_paths import get_local_setting, save_local_setting
    from ui.panels import day_review_panel

    monkeypatch.setattr(day_review_panel, "_display_zone", lambda: PACIFIC)
    original = get_local_setting(day_review_panel.SESSION_SETTING_KEY, "")
    first = day_review_panel.DayReviewPanel(service=_Service(), clock=lambda: NOW, remember_session=True)
    try:
        first.show_session("2026-09-17")
        assert get_local_setting(day_review_panel.SESSION_SETTING_KEY) == "2026-09-17"
        second = day_review_panel.DayReviewPanel(
            service=_Service(), clock=lambda: NOW, remember_session=True
        )
        assert second.session_date() == "2026-09-17"
        second.shutdown()
        second.deleteLater()
    finally:
        first.shutdown()
        first.deleteLater()
        save_local_setting(day_review_panel.SESSION_SETTING_KEY, original)
        app.processEvents()


def test_j_and_k_move_the_row_in_the_focused_table(panel, app):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    panel.render(_payload(walkaway=_day()))
    table = panel.miss_table_for("claimed_d1")

    def _press(key):
        QApplication.sendEvent(table, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))

    _press(Qt.Key.Key_J)
    assert table.currentRow() == 0
    _press(Qt.Key.Key_J)
    _press(Qt.Key.Key_J)
    assert table.currentRow() == 2
    _press(Qt.Key.Key_K)
    assert table.currentRow() == 1


# -- 5. one miss table ---------------------------------------------------------
def test_five_populations_are_one_table_with_counted_chips(panel):
    from PySide6.QtWidgets import QTableWidget

    panel.render(_payload(walkaway=_day()))
    assert len(panel.findChildren(QTableWidget, "DayReviewMissTable")) == 1
    assert panel.miss_chips["rejected"].text() == "Passed && ran (2)"
    assert panel.miss_chips["claimed_d1"].text() == "Claimed D1 (3)"
    assert panel.miss_table.rowCount() == 2
    panel.miss_chips["claimed_d1"].click()
    assert panel.miss_table.rowCount() == 3
    assert panel.miss_table.item(0, 1).text() == "GMAB"
    shown = [i for i in range(panel.miss_table.columnCount()) if not panel.miss_table.isColumnHidden(i)]
    assert 6 <= len(shown) <= 9


def test_a_single_click_charts_the_row_beside_the_table(panel, app):
    panel.render(_payload(walkaway=_day(), name_charts={"STX": {"bars": _bars(), "markers": ()}}))
    asked: list[tuple[str, str]] = []
    panel.chartRequested.connect(lambda symbol, side: asked.append((symbol, side)))
    panel.miss_table.setCurrentCell(1, 1)
    app.processEvents()
    assert panel.name_chart_symbol() == "STX"
    assert asked == [], "a single click must not open the board chart"
    panel.miss_table.itemDoubleClicked.emit(panel.miss_table.item(1, 1))
    assert asked == [("STX", "LONG")]


def test_the_legacy_nine_column_walkaway_is_gone():
    from ui.panels import day_review_panel

    for name in ("WALKAWAY_COLUMNS", "WALKAWAY_PLACEHOLDERS", "WALKAWAY_PLACEHOLDER_CELLS"):
        assert not hasattr(day_review_panel, name), name
    assert not hasattr(day_review_panel.DayReviewPanel, "_render_walkaway")


# -- 6. trades get charts --------------------------------------------------------
def test_selecting_a_trade_draws_its_chart_with_the_trade_marks(panel, app):
    panel.render(_payload(
        trades=[{"trade_id": "t1", "symbol": "DRAM"}, {"trade_id": "t2", "symbol": "ZETA"}],
        trade_reviews=[{"trade_id": "t1", "symbol": "DRAM"}, {"trade_id": "t2", "symbol": "ZETA"}],
        name_charts={"DRAM": {"bars": _bars(), "markers": ()}},
    ))
    assert panel.trade_chart_symbol() == "DRAM"
    assert panel._trade_chart is not None and panel._trade_chart.bar_count() == 6
    assert panel._trade_chart is not panel._name_chart
    panel.trades_table.selectRow(1)
    assert panel.trade_chart_symbol() == ""
    assert "ZETA" in panel.trade_chart_note.text()
    detail = panel.trade_detail.toPlainText()
    assert "instrument unknown" not in detail and "none recorded" not in detail
    assert "you did not write one" in detail


def test_a_no_trade_day_says_so_plainly(panel):
    from ui.panels.day_review_panel import TRADE_CHART_NO_TRADES_NOTE

    panel.render(_payload())
    assert panel.trade_detail.toPlainText() == "No trades this session."
    assert panel.trade_chart_note.text() == TRADE_CHART_NO_TRADES_NOTE
