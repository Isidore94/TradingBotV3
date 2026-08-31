"""An AWAY day ends in a recap, not a queue - R1 amendment 2026-08-24.

The trader came back from a full AWAY day to **317 alerts waiting in the chart
review queue**, plus 128 more hidden inside yesterday's range. Verbatim: "Auto
away should NOT produce that much noise. In general it should just send nothing
until EOD, where it will show me what produced the best results intraday and
also what focus picks needed managing... Only auto desk should send 317 signals
over the course of the day."

What this must NOT change is the harder half, and most of these tests defend
it. The repetition-control precedent is the constraint: **display decisions
withhold nothing from evidence.** The backing alert list, History, the D1 badge
and every evidence stream keep filling exactly as before; only the chart-review
queue - a return surface for a trader who is not there - stops accumulating.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import away_recap  # noqa: E402

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.models.bounce import BounceAlert  # noqa: E402


def _alert(symbol, side="LONG", *, tag="", trigger="[S-TIER] VWAP reclaim"):
    return BounceAlert(
        time_text="11:30:00",
        symbol=symbol,
        side=side,
        trigger=trigger,
        timeframe="5m",
        tag=tag,
        raw_text=f"[S-TIER] {symbol}: {trigger}",
    )


def _panel(monkeypatch, mode="AWAY"):
    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: mode)
    monkeypatch.setattr(panel, "mover_state", lambda symbol, side="": "open")
    return panel


# ==========================================================================
# the routing
# ==========================================================================
def test_an_away_day_accumulates_no_review_queue(monkeypatch):
    """Fail-before-fix. This is the 317."""
    panel = _panel(monkeypatch, mode="AWAY")

    for index in range(40):
        panel.add_alert(_alert(f"SYM{index}"))

    assert panel._review_queue == []
    assert panel._current_review_alert is None


def test_the_desk_lists_intraday_alerts_in_the_m5_bar_and_queues_none(monkeypatch):
    """DESK was untouched by the AWAY amendment. Since 2026-08-27 (trader
    rule) an ordinary intraday alert lists in the M5 alert bar instead of
    queueing a chart - in DESK and EVENING alike - while AWAY still assembles
    the recap and never posts to the bar."""
    panel = _panel(monkeypatch, mode="DESK")
    posted = []
    panel.m5AlertPosted.connect(posted.append)

    for index in range(5):
        panel.add_alert(_alert(f"SYM{index}"))

    assert [alert.symbol for alert in posted] == [f"SYM{i}" for i in range(5)]
    assert panel._current_review_alert is None
    assert panel._review_queue == []


def test_evening_lists_intraday_alerts_in_the_m5_bar_too(monkeypatch):
    """The amendment changed AWAY only; EVENING is for sleeping through the
    morning, and what the trader wakes up to is now the M5 bar plus the D1
    queue rather than a queue of M5 charts."""
    panel = _panel(monkeypatch, mode="EVENING")
    posted = []
    panel.m5AlertPosted.connect(posted.append)

    panel.add_alert(_alert("AAA"))
    panel.add_alert(_alert("BBB"))

    assert [alert.symbol for alert in posted] == ["AAA", "BBB"]
    assert panel._current_review_alert is None
    assert panel._review_queue == []


# ==========================================================================
# what must NOT change - the repetition-control precedent
# ==========================================================================
def test_the_backing_alert_list_still_fills_exactly_as_before(monkeypatch):
    """Display decisions withhold nothing from evidence. History, the AWAY
    push and every evidence stream read from this list."""
    away = _panel(monkeypatch, mode="AWAY")
    desk = _panel(monkeypatch, mode="DESK")

    for index in range(12):
        away.add_alert(_alert(f"SYM{index}"))
        desk.add_alert(_alert(f"SYM{index}"))

    assert len(away._alerts) == len(desk._alerts) == 12
    assert [alert.symbol for alert in away._alerts] == [
        alert.symbol for alert in desk._alerts
    ]


def test_the_d1_feed_and_badge_still_fill(monkeypatch):
    """The D1 badge is how the trader sees what accrued. It is not a queue."""
    away = _panel(monkeypatch, mode="AWAY")
    desk = _panel(monkeypatch, mode="DESK")

    # The Focus D1-event tag is the branch that needs only a tag; the other D1
    # branch additionally requires `is_ready_d1_alert`, which is a bucket
    # upgrade this fixture has no business synthesising.
    from ui.models.bounce import FOCUS_D1_EVENT_TAG

    d1 = _alert("AAA", tag=FOCUS_D1_EVENT_TAG, trigger="MASTER_AVWAP_D1_ZONE: zone1")
    d1.is_d1 = True
    away.add_alert(d1)
    desk.add_alert(d1)

    assert len(away._d1_alerts) == len(desk._d1_alerts) == 1


def test_the_diverted_alerts_are_counted_so_the_return_is_honest(monkeypatch):
    """"Nothing accumulated" and "nothing happened" must not look the same."""
    panel = _panel(monkeypatch, mode="AWAY")

    for index in range(7):
        panel.add_alert(_alert(f"SYM{index}"))

    assert panel.away_recap_count() == 7


def test_the_count_resets_on_a_new_session(monkeypatch):
    panel = _panel(monkeypatch, mode="AWAY")
    panel.add_alert(_alert("AAA"))
    panel._away_recap_session = "2026-08-20"

    panel.add_alert(_alert("BBB"))

    assert panel.away_recap_count() == 1


# ==========================================================================
# the recap itself
# ==========================================================================
def test_the_recap_ranks_only_what_the_day_already_ranked():
    """Presentation only: no new detector, score, ranking or writer. The order
    is the order the day produced, and the recap says where each row came
    from."""
    recap = away_recap.build_recap(
        session_date="2026-08-21",
        alerts=[
            {"symbol": "AAA", "side": "LONG", "tier": "S", "trigger": "VWAP reclaim"},
            {"symbol": "BBB", "side": "SHORT", "tier": "C", "trigger": "LRSI cross"},
        ],
        staged_picks={"long": ["CCC"], "short": ["DDD"]},
        digest_swings=["FTAI (SHORT)", "PSNL (LONG)"],
        focus_picks={"long": ["EEE"], "short": []},
    )

    assert recap["session_date"] == "2026-08-21"
    assert [row["symbol"] for row in recap["best_swings"]] == ["FTAI", "PSNL"]
    assert recap["counts"]["alerts"] == 2
    assert recap["counts"]["staged"] == 2
    for section in ("best_swings", "classified_alerts", "staged_picks", "focus_to_manage"):
        assert recap["provenance"][section]


def test_the_recap_never_invents_a_rank():
    """A day that produced no numbered swings has none, and the recap says so
    rather than ranking something else into the slot."""
    recap = away_recap.build_recap(
        session_date="2026-08-21", alerts=[], staged_picks={}, digest_swings=[], focus_picks={}
    )

    assert recap["best_swings"] == []
    assert recap["counts"] == {"alerts": 0, "staged": 0, "swings": 0, "focus": 0}
    assert "nothing" in recap["summary"].lower()


def test_an_unreadable_source_is_named_absent_not_silently_empty():
    """A recap that shows nothing because a file would not open must not look
    like a quiet day."""
    recap = away_recap.build_recap(
        session_date="2026-08-21",
        alerts=[],
        staged_picks={},
        digest_swings=[],
        focus_picks={},
        unavailable={"autopilot_today.txt": "file not found"},
    )

    assert recap["unavailable"] == {"autopilot_today.txt": "file not found"}
    assert "could not be read" in recap["summary"]


def test_a_digest_line_that_cannot_be_parsed_is_kept_and_marked():
    """Dropping it would quietly narrow the day."""
    recap = away_recap.build_recap(
        session_date="2026-08-21",
        alerts=[],
        staged_picks={},
        digest_swings=["FTAI (SHORT)", "a line nobody can parse"],
        focus_picks={},
    )
    rows = recap["best_swings"]

    assert len(rows) == 2
    assert rows[1]["symbol"] == ""
    assert rows[1]["unparsed"] is True
    assert rows[1]["text"] == "a line nobody can parse"


def test_the_recap_writes_nothing():
    """Every mutation belongs to an existing owner. The recap reads."""
    import ast

    source = (SCRIPTS_DIR / "away_recap.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    # `append` is deliberately NOT in this list: it is how Python builds a
    # list, and forbidding it would only teach the next person to work around
    # the test. What is forbidden is IO.
    for forbidden in ("write_text", "write_bytes", "mkdir", "unlink", "touch"):
        assert forbidden not in called, f"the recap module must not call {forbidden}"
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    # It does not even reach a store: every input is handed in by the caller,
    # which is what keeps "presentation only" true rather than aspirational.
    assert not {"evidence_ledger", "project_paths", "focus_picks"} & imported
    assert "open(" not in source


# ==========================================================================
# the wiring: the recap must actually be HANDED the day's alerts
# ==========================================================================
@pytest.mark.qt
def test_selecting_the_away_recap_page_hands_it_the_alert_center_backing_list(monkeypatch):
    """Sol C1, as a regression.

    `MainWindow` constructed `AwayRecapPanel` and never called `set_alerts`, so
    a full AWAY day ended in a recap with nothing in it - the diverted alerts
    were counted and the backing list was full, and none of it reached the one
    surface built to show them. The reproduction printed `backing 1
    recap_input 0`.
    """
    from ui.app import MainWindow, PAGE_SPECS
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        center = window.trading_panel.alert_center
        center._auto_mode_now = lambda: "AWAY"
        center.mover_state = lambda *args, **kwargs: "open"
        center.add_alert(
            BounceAlert(
                time_text="10:00:00",
                symbol="AAA",
                side="LONG",
                trigger="[S-TIER] VWAP reclaim",
                timeframe="5m",
                raw_text="[S-TIER] AAA: VWAP reclaim",
            )
        )
        assert center._alerts, "the backing list must fill in AWAY - that half already worked"

        index = [spec.title for spec in PAGE_SPECS].index("AWAY Recap")
        window._select_page(index)

        rows = window.away_recap_panel._alerts
        assert rows, "the AWAY Recap was never handed the Alert Center backing list"
        assert rows[0]["symbol"] == "AAA"
        assert rows[0]["side"] == "LONG"
        # `_alert_rows` reads mappings. Handing it the dataclasses would raise
        # inside the worker thread and the page would stay blank in a way no
        # assertion on `_alerts` could see.
        assert all(hasattr(row, "get") for row in rows)
    finally:
        try:
            window.close()
        except Exception:
            pass


@pytest.mark.qt
def test_the_recap_feed_is_oldest_first_and_carries_the_d1_rows(monkeypatch):
    """The order is the order the day happened - the only ordering nobody has
    to defend - and the D1 feed is part of the day, flagged rather than merged
    away."""
    from ui.app import MainWindow, PAGE_SPECS
    from ui.models.bounce import FOCUS_D1_EVENT_TAG
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        center = window.trading_panel.alert_center
        center._auto_mode_now = lambda: "AWAY"
        center.mover_state = lambda *args, **kwargs: "open"
        for stamp, symbol in (("10:00:00", "AAA"), ("11:00:00", "BBB")):
            center.add_alert(
                BounceAlert(time_text=stamp, symbol=symbol, side="LONG",
                            trigger="VWAP reclaim", raw_text=f"{symbol}: VWAP reclaim")
            )
        d1 = BounceAlert(time_text="10:30:00", symbol="CCC", side="LONG",
                         tag=FOCUS_D1_EVENT_TAG, trigger="MASTER_AVWAP_D1_ZONE: zone1",
                         raw_text="MASTER_AVWAP_D1_ZONE CCC")
        d1.is_d1 = True
        center.add_alert(d1)

        window._select_page([spec.title for spec in PAGE_SPECS].index("AWAY Recap"))
        rows = window.away_recap_panel._alerts

        assert [row["symbol"] for row in rows] == ["AAA", "CCC", "BBB"]
        assert [row["is_d1"] for row in rows] == [False, True, False]
    finally:
        try:
            window.close()
        except Exception:
            pass


# ==========================================================================
# the outlet: the alerts the page is handed must be VISIBLE, and chartable
# ==========================================================================
def _recap_panel():
    from ui.panels.away_recap_panel import AwayRecapPanel

    QApplication.instance() or QApplication([])
    return AwayRecapPanel()


def test_the_days_alerts_are_drawn_not_only_counted():
    """The page was handed the alerts and had nowhere to put them.

    `build_recap` produced `classified_alerts` and `_render` never read it, so
    the only trace of a whole AWAY day's alerts was the word "alert(s)" in the
    summary line. A recap that drops the thing it was opened for is not a
    recap.
    """
    panel = _recap_panel()
    panel._render(
        {
            "summary": "s",
            "classified_alerts": [
                {"symbol": "AAA", "side": "LONG", "tier": "S", "trigger": "VWAP reclaim",
                 "time_text": "10:00:00", "is_d1": False},
                {"symbol": "BBB", "side": "SHORT", "tier": "", "trigger": "zone1",
                 "time_text": "11:30:00", "is_d1": True},
            ],
        }
    )

    assert panel.alerts.rowCount() == 2
    row = [panel.alerts.item(0, column).text() for column in range(panel.alerts.columnCount())]
    assert row[:4] == ["10:00:00", "AAA", "LONG", "S"]
    assert panel.alerts.item(1, 4).text() == "D1", "a D1 row is flagged, never merged away"
    assert panel.alerts.item(0, 4).text() == ""


def test_the_alert_order_is_the_order_the_day_produced():
    """No re-ranking anywhere on this page (the recap's own provenance note)."""
    panel = _recap_panel()
    panel._render(
        {
            "classified_alerts": [
                {"symbol": "AAA", "time_text": "09:00:00"},
                {"symbol": "BBB", "time_text": "10:00:00"},
                {"symbol": "CCC", "time_text": "11:00:00"},
            ]
        }
    )
    assert [panel.alerts.item(index, 1).text() for index in range(3)] == ["AAA", "BBB", "CCC"]


def test_an_empty_alert_list_leaves_an_empty_table_not_a_stale_one():
    panel = _recap_panel()
    panel._render({"classified_alerts": [{"symbol": "AAA", "time_text": "09:00:00"}]})
    panel._render({"classified_alerts": []})
    assert panel.alerts.rowCount() == 0


def test_activating_an_alert_row_asks_the_host_to_chart_it():
    """The page owns no chart. It ASKS, exactly as the Strength Board does, and
    the host opens the Alert Center's existing snapshot popup."""
    panel = _recap_panel()
    panel._render({"classified_alerts": [{"symbol": "AAA", "time_text": "09:00:00"}]})
    seen: list[str] = []
    panel.symbolActivated.connect(seen.append)

    panel._activate_alert(panel.alerts.item(0, 1))

    assert seen == ["AAA"]


def test_activating_a_swing_row_charts_it_too():
    panel = _recap_panel()
    panel._render({"best_swings": [{"rank": 1, "symbol": "VNO", "side": "LONG", "text": "1. VNO"}]})
    seen: list[str] = []
    panel.symbolActivated.connect(seen.append)

    panel._activate_swing(panel.swings.item(0, 1))

    assert seen == ["VNO"]


def test_a_blank_symbol_asks_for_no_chart():
    """Missing data is uncertainty: an empty row must not open a chart for "".

    Since G-P2.1 a blank-symbol row is a SCANNER STATUS row and is hidden and
    counted, so it is revealed here first - otherwise this would pass because
    the row is absent rather than because a blank symbol charts nothing, which
    is a different claim.
    """
    panel = _recap_panel()
    panel._render({"classified_alerts": [{"symbol": "", "time_text": "09:00:00"}]})
    panel._reveal_status_rows()
    assert panel.alerts.rowCount() == 1
    seen: list[str] = []
    panel.symbolActivated.connect(seen.append)

    panel._activate_alert(panel.alerts.item(0, 1))

    assert seen == []


@pytest.mark.qt
def test_the_desk_charts_a_recap_symbol_through_the_one_snapshot_popup():
    """One chart surface for the whole desk. A second chart widget on this page
    would be a second definition of what a symbol looks like."""
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        seen: list[str] = []
        window.trading_panel.alert_center.show_board_symbol = seen.append
        window.away_recap_panel.symbolActivated.emit("AAA")
        assert seen == ["AAA"]
    finally:
        try:
            window.close()
        except Exception:
            pass
