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
from datetime import date, datetime
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


def test_the_desk_is_byte_identical(monkeypatch):
    """Only auto desk should stream signals all day. DESK is untouched."""
    panel = _panel(monkeypatch, mode="DESK")

    for index in range(5):
        panel.add_alert(_alert(f"SYM{index}"))

    assert panel._current_review_alert is not None
    assert len(panel._review_queue) == 4


def test_evening_keeps_its_existing_queue_silently_semantics(monkeypatch):
    """The amendment changes AWAY only. EVENING is for sleeping through the
    morning session, and its queue is what the trader wakes up to."""
    panel = _panel(monkeypatch, mode="EVENING")

    panel.add_alert(_alert("AAA"))
    panel.add_alert(_alert("BBB"))

    assert panel._current_review_alert is not None
    assert len(panel._review_queue) == 1


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
