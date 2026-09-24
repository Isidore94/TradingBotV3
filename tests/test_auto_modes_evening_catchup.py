"""The flip out of EVENING shows ONE catch-up card (trader, 2026-09-23).

The best of the morning: strongest longs, weakest shorts, names strong on
pullbacks (the M5 alerts that fired), best swing setups and the price alerts
that fired. Built off the Qt thread from existing rankings. A row charts
through the board door, never the review queue.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

NOW = datetime(2026, 9, 23, 8, 30)


def _alert(symbol, side="LONG", tier="B", *, d1=False, time_text="07:00:00", trigger="VWAP reclaim"):
    return {
        "symbol": symbol,
        "side": side,
        "tier": tier,
        "trigger": trigger,
        "time_text": time_text,
        "is_d1": d1,
        "cell": "" if d1 else f"{trigger} {side}",
    }


def _section(payload, key):
    return next(section for section in payload["sections"] if section["key"] == key)


def test_the_card_has_the_five_sections_in_order():
    import evening_catchup

    payload = evening_catchup.build_catchup(now=NOW)
    assert [section["key"] for section in payload["sections"]] == [
        "longs",
        "shorts",
        "pullbacks",
        "swing",
        "price",
    ]


def test_strength_puts_held_picks_first_then_the_movers_board():
    import evening_catchup

    persistence = {
        "AAA": {"side": "long", "score": 1.0, "verdict": "held", "detail": "0.2% off HOD"},
        "BBB": {"side": "long", "score": 3.0, "verdict": "held", "detail": "0.1% off HOD"},
        "CCC": {"side": "long", "score": 9.0, "verdict": "faded", "detail": "faded"},
        "SSS": {"side": "short", "score": 2.0, "verdict": "held", "detail": "0.3% off LOD"},
    }
    board = {
        "pop": {
            "long": [{"symbol": "AAA", "day_pct": 4.0}, {"symbol": "MMM", "day_pct": 3.0}],
            "short": [{"symbol": "TTT", "day_pct": -5.0}],
        }
    }
    payload = evening_catchup.build_catchup(
        persistence=persistence, movers_board=board, now=NOW
    )
    assert [row["symbol"] for row in _section(payload, "longs")["rows"]] == ["BBB", "AAA", "MMM"]
    assert [row["symbol"] for row in _section(payload, "shorts")["rows"]] == ["SSS", "TTT"]


def test_pullbacks_rank_the_m5_alerts_by_tier_one_row_per_name():
    import evening_catchup

    alerts = [
        _alert("LOW", tier="C", time_text="06:40:00"),
        _alert("TOP", tier="S", time_text="07:10:00"),
        _alert("TOP", tier="A", time_text="07:20:00"),
        _alert("MID", tier="A", time_text="06:50:00"),
        _alert("D1X", tier="", d1=True),
    ]
    rows = _section(evening_catchup.build_catchup(alerts=alerts, now=NOW), "pullbacks")["rows"]
    assert [row["symbol"] for row in rows] == ["TOP", "MID", "LOW"]
    assert "x2" in rows[0]["text"]


def test_each_section_holds_at_most_five_rows():
    import evening_catchup

    alerts = [_alert(f"S{i}", tier="B") for i in range(9)]
    rows = _section(evening_catchup.build_catchup(alerts=alerts, now=NOW), "pullbacks")["rows"]
    assert len(rows) == 5


def test_swing_lists_the_d1_alerts_then_the_best_scan_rows():
    import evening_catchup

    swing_rows = [
        {"symbol": "LNG", "side": "long", "expected_r": 1.5, "bucket_label": "zone1"},
        {"symbol": "SHT", "side": "short", "expected_r": 0.9, "bucket_label": "zone2"},
    ]
    payload = evening_catchup.build_catchup(
        alerts=[_alert("DDD", d1=True, trigger="MASTER_AVWAP_D1_ZONE: zone1")],
        swing_rows=swing_rows,
        now=NOW,
    )
    rows = _section(payload, "swing")["rows"]
    assert [row["symbol"] for row in rows] == ["DDD", "LNG", "SHT"]
    assert "1.50R" in rows[1]["text"]


def test_price_alerts_only_since_evening_started():
    import evening_catchup

    triggers = [
        {"date": "2026-09-23", "at": "05:00:00", "symbol": "OLD", "side": "above", "level": 1, "last": 2},
        {"date": "2026-09-23", "at": "06:45:00", "symbol": "NEW", "side": "below", "level": 9, "last": 8},
    ]
    payload = evening_catchup.build_catchup(
        price_triggers=triggers, since=datetime(2026, 9, 23, 6, 0), now=NOW
    )
    rows = _section(payload, "price")["rows"]
    assert [row["symbol"] for row in rows] == ["NEW"]
    assert "below" in rows[0]["text"]


def test_the_service_builds_off_the_qt_thread_and_emits_once(monkeypatch):
    import threading

    from ui.services.evening_catchup_service import EveningCatchupService

    threads: list[bool] = []

    def rows():
        threads.append(threading.current_thread() is threading.main_thread())
        return []

    service = EveningCatchupService(
        swing_rows=rows, persistence=lambda now: {}, triggers=lambda now: []
    )
    got: list[dict] = []
    service.ready.connect(got.append)
    assert service.request({"alerts": [_alert("AAA")]}) is True
    deadline = time.monotonic() + 3.0
    while not got and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    assert threads == [False]
    assert len(got) == 1
    assert _section(got[0], "pullbacks")["rows"][0]["symbol"] == "AAA"


def test_a_row_click_asks_for_the_board_chart():
    from ui.widgets.evening_catchup_card import EveningCatchupCard

    import evening_catchup

    card = EveningCatchupCard()
    clicked: list[tuple[str, str]] = []
    card.symbolClicked.connect(lambda s, d: clicked.append((s, d)))
    try:
        card.show_catchup(
            evening_catchup.build_catchup(alerts=[_alert("AAA", side="LONG")], now=NOW)
        )
        buttons = card.row_buttons()
        assert len(buttons) == 1
        buttons[0].click()
    finally:
        card.close()
    assert clicked == [("AAA", "long")]


def test_only_the_flip_out_of_evening_requests_the_card():
    from ui.app import MainWindow

    requested: list[dict] = []
    fake = SimpleNamespace(
        trading_panel=SimpleNamespace(
            alert_center=SimpleNamespace(evening_catchup_snapshot=lambda: {"alerts": []})
        ),
        evening_catchup_service=SimpleNamespace(request=requested.append),
    )
    MainWindow._maybe_request_evening_catchup(fake, "DESK", "EVENING")
    MainWindow._maybe_request_evening_catchup(fake, "AWAY", "DESK")
    assert requested == []
    MainWindow._maybe_request_evening_catchup(fake, "EVENING", "DESK")
    MainWindow._maybe_request_evening_catchup(fake, "EVENING", "OFF")
    assert len(requested) == 2


def test_card_rows_chart_through_the_board_door_never_the_queue(monkeypatch):
    from ui.app import MainWindow
    from ui.widgets import evening_catchup_card

    import evening_catchup

    class _Parentless(evening_catchup_card.EveningCatchupCard):
        def __init__(self, _parent=None):
            super().__init__(None)

    monkeypatch.setattr(evening_catchup_card, "EveningCatchupCard", _Parentless)

    boards: list[tuple[str, str]] = []
    queued: list = []
    center = SimpleNamespace(
        show_board_symbol=lambda symbol, side="": boards.append((symbol, side)),
        _enqueue_review_alert=queued.append,
    )
    fake = SimpleNamespace(
        trading_panel=SimpleNamespace(alert_center=center),
        evening_catchup_card=None,
    )
    fake._chart_recap_row = lambda symbol, side="": MainWindow._chart_recap_row(fake, symbol, side)
    MainWindow._show_evening_catchup(
        fake, evening_catchup.build_catchup(alerts=[_alert("ZZZ", side="SHORT")], now=NOW)
    )
    try:
        fake.evening_catchup_card.row_buttons()[0].click()
    finally:
        fake.evening_catchup_card.close()
    assert boards == [("ZZZ", "short")]
    assert queued == []


def test_the_panel_snapshot_carries_the_diverted_alerts_with_their_tier(monkeypatch):
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: "EVENING")
    panel = AlertCenterPanel()
    panel._auto_mode_cached = None
    panel.on_auto_mode_changed("DESK", "EVENING")
    panel._enqueue_review_alert(
        BounceAlert(
            time_text="06:45:00",
            symbol="NVDA",
            side="LONG",
            trigger="VWAP reclaim",
            timeframe="5m",
            raw_text="[S-TIER] NVDA: VWAP reclaim",
        )
    )
    snapshot = panel.evening_catchup_snapshot()
    assert snapshot["since"] is not None
    assert [(a["symbol"], a["tier"]) for a in snapshot["alerts"]] == [("NVDA", "S")]
    assert isinstance(snapshot["movers_board"], dict)
