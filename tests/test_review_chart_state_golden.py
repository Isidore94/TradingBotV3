"""Golden verdicts for the Alert Center's one display answer, `_review_chart_state`.

Pinned on 2026-09-23 BEFORE the wall-gate leg was added (hard invariant: no
alert-display change without golden results first). Every leg is measured for
real - mover, session VWAP and the D1 SMA trend - off synthetic bars fed
through the panel's own bar accessors at a fixed 11:00 clock.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.models.bounce import FOCUS_D1_EVENT_TAG, BounceAlert  # noqa: E402

FIXTURE = ROOT_DIR / "tests" / "fixtures" / "review_chart_state_golden_v1.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def daily_bars(segments, today: datetime) -> list[dict]:
    """Daily bars from [count, first close, last close] segments, ending yesterday."""
    closes: list[float] = []
    for count, first, last in segments:
        count = int(count)
        for i in range(count):
            step = 0.0 if count == 1 else (float(last) - float(first)) * i / (count - 1)
            closes.append(float(first) + step)
    start = today.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=len(closes)
    )
    return [
        {
            "dt": start + timedelta(days=i),
            "open": c,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": 1000.0,
        }
        for i, c in enumerate(closes)
    ]


def m5_bars(closes, today: datetime) -> list[dict]:
    """Completed M5 bars ending ten minutes before ``today``."""
    count = len(closes)
    return [
        {
            "dt": today - timedelta(minutes=10 * (count - i)),
            "open": c,
            "high": c + 0.05,
            "low": c - 0.05,
            "close": c,
            "volume": 100.0,
        }
        for i, c in enumerate(closes)
    ]


def alert_for(case) -> BounceAlert:
    symbol, side, kind = case["symbol"], case["side"], case["kind"]
    word = side.lower()
    if kind == "d1":
        return BounceAlert(
            time_text="08:25:00",
            symbol=symbol,
            side=side,
            trigger=f"({word}) zone1 reject at AVWAPE",
            timeframe="D1",
            tag=f"d1_flag_{word}",
            raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({word}) zone1 reject at AVWAPE",
            is_d1=True,
        )
    if kind == "focus_d1":
        return BounceAlert(
            time_text="06:35:00",
            symbol=symbol,
            side=side,
            trigger="Focus D1 · New 5-day high",
            timeframe="D1",
            tag=FOCUS_D1_EVENT_TAG,
            raw_text=f"Focus D1: {symbol} new 5-day high",
        )
    return BounceAlert(
        time_text="10:55:00",
        symbol=symbol,
        side=side,
        trigger="[S-TIER] VWAP reclaim",
        timeframe="5m",
        raw_text=f"[S-TIER] {symbol}: VWAP reclaim",
    )


@pytest.fixture()
def golden_panel(monkeypatch):
    QApplication.instance() or QApplication([])
    from ui.panels import alert_center_panel
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    today = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)

    class _At11(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102 - stdlib signature
            return today if tz is None else today.astimezone(tz)

    monkeypatch.setattr(alert_center_panel, "datetime", _At11)
    panel = AlertCenterPanel()
    daily = {case["symbol"]: daily_bars(case["daily"], today) for case in CASES}
    intraday = {case["symbol"]: m5_bars(case["m5"], today) for case in CASES}
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: daily.get(symbol, []))
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda symbol, **kw: intraday.get(symbol, [])
    )
    return panel


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_review_chart_state_matches_the_golden_verdict(golden_panel, case):
    assert golden_panel._review_chart_state(alert_for(case)) == case["expected"]
