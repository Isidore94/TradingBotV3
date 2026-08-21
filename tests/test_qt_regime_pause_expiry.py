"""A stale "holding highs" row leaves the queue, through the real panel.

`test_regime_pause_hold.py` proves the rule. This proves the Alert Center
honours it, and - the load-bearing half - that deleting a queue row deletes
nothing else. The trader's call on 2026-08-21 was explicit: gone from the
queue, kept on disk, so the forward record of whether stale calls were any good
stays measurable.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

OPEN = datetime(2026, 8, 21, 6, 30)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from ui.panels.alert_center_panel import AlertCenterPanel

    made = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    # The movers-only filter is a different rule with its own tests; keep it
    # out of the way so these measure the freshness rule alone.
    monkeypatch.setattr(made, "_review_movers_only", False, raising=False)
    yield made
    made.deleteLater()


def _bar(index: int, high: float, low: float, close: float) -> dict:
    return {
        "dt": OPEN + timedelta(minutes=5 * index),
        "open": (high + low) / 2.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
    }


def _faded() -> list[dict]:
    """A spike into a high, then a long slide - the MRK shape."""
    bars = [_bar(i, 100.5, 99.5, 100.0) for i in range(15)]
    bars.append(_bar(15, 103.0, 101.0, 102.9))
    bars += [_bar(16 + i, 102.0 - i, 101.0 - i, 101.2 - i) for i in range(6)]
    return bars


def _still_climbing() -> list[dict]:
    bars = [_bar(i, 100.5, 99.5, 100.0) for i in range(15)]
    bars += [_bar(15 + i, 103.0 + i, 101.0 + i, 102.9 + i) for i in range(8)]
    return bars


def _regime_alert(symbol="MRK", side="LONG", *, at="08:30:00"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text=at,
        symbol=symbol,
        side=side,
        trigger="M5 regime-pause watch · holding highs",
        timeframe="M5",
        tag="green",
        raw_text=(
            "REGIME PAUSE WATCH (long): SPY paused (-0.30% window) - "
            f"1 swing long still holding highs: {symbol} (1 today). "
            "Recorded as swing-scan evidence, not an entry signal."
        ),
    )


def _feed(panel, bars, *, now):
    """Point the panel at a fixed bar series and a fixed clock."""
    panel._m5_bars_for = lambda symbol, sessions=1: list(bars)  # type: ignore[assignment]
    # Pinned rather than derived: the real _alert_moment stamps the alert on
    # TODAY, so a test that relied on it would only pass on 2026-08-21.
    panel._alert_moment = lambda alert: datetime.strptime(  # type: ignore[assignment]
        f"2026-08-21 {alert.time_text}", "%Y-%m-%d %H:%M:%S"
    )
    import regime_pause_hold

    real = regime_pause_hold.queue_verdict

    def frozen(bars_, side, *, alert_time, minutes=15, tolerance_atr=1.0, **_kwargs):
        return real(
            bars_,
            side,
            alert_time=alert_time,
            now=now,
            minutes=minutes,
            tolerance_atr=tolerance_atr,
        )

    return frozen


def test_a_stale_holding_row_is_removed_from_the_queue(panel, monkeypatch):
    import regime_pause_hold

    alert = _regime_alert()
    panel._enqueue_review_alert(alert)
    panel._enqueue_review_alert(_regime_alert("HTFL"))
    assert panel._current_review_alert is not None

    later = OPEN + timedelta(minutes=200)
    monkeypatch.setattr(
        regime_pause_hold, "queue_verdict", _feed(panel, _faded(), now=later)
    )
    panel._expire_stale_hold_alerts()

    # Both rows are measured against the same faded series, so the queue
    # empties and the pane clears rather than showing a chart that has stopped
    # being true.
    assert [a.symbol for a in panel._review_queue] == []
    assert panel._current_review_alert is None


def test_a_row_still_making_new_highs_survives_past_fifteen_minutes(panel, monkeypatch):
    import regime_pause_hold

    bars = _still_climbing()
    panel._enqueue_review_alert(_regime_alert())
    panel._enqueue_review_alert(_regime_alert("HTFL"))
    later = bars[-1]["dt"] + timedelta(minutes=5)
    monkeypatch.setattr(
        regime_pause_hold, "queue_verdict", _feed(panel, bars, now=later)
    )
    panel._expire_stale_hold_alerts()
    live = [a.symbol for a in panel._review_queue]
    if panel._current_review_alert is not None:
        live.append(panel._current_review_alert.symbol)
    assert "HTFL" in live and "MRK" in live


def test_an_ordinary_alert_is_never_touched_by_this_rule(panel, monkeypatch):
    """Only rows exploded out of a REGIME PAUSE WATCH line expire here."""
    import regime_pause_hold
    from ui.models.bounce import BounceAlert

    ordinary = BounceAlert(
        time_text="06:35:00",
        symbol="NVDA",
        side="LONG",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text="[A-TIER] NVDA: Bounce confirmed",
    )
    panel._enqueue_review_alert(ordinary)
    panel._enqueue_review_alert(_regime_alert())
    later = OPEN + timedelta(minutes=200)
    monkeypatch.setattr(
        regime_pause_hold, "queue_verdict", _feed(panel, _faded(), now=later)
    )
    panel._expire_stale_hold_alerts()
    live = [a.symbol for a in panel._review_queue]
    if panel._current_review_alert is not None:
        live.append(panel._current_review_alert.symbol)
    assert "NVDA" in live


def test_a_symbol_with_no_cached_bars_is_kept(panel, monkeypatch):
    """Uncertainty never deletes: an unwarmed cache is not a fade."""
    panel._enqueue_review_alert(_regime_alert())
    panel._m5_bars_for = lambda symbol, sessions=1: []  # type: ignore[assignment]
    panel._expire_stale_hold_alerts()
    live = [a.symbol for a in panel._review_queue]
    if panel._current_review_alert is not None:
        live.append(panel._current_review_alert.symbol)
    assert "MRK" in live


def test_expiring_a_row_writes_evidence_and_deletes_none(panel, monkeypatch, tmp_path):
    """The trader's call: gone from the queue, kept on disk."""
    import regime_pause_hold

    alert = _regime_alert()
    panel.add_alert(alert)
    history_before = len(panel._alerts)
    events_path = tmp_path / "alert_review_events.jsonl"
    before = events_path.read_text(encoding="utf-8") if events_path.exists() else ""

    later = OPEN + timedelta(minutes=200)
    monkeypatch.setattr(
        regime_pause_hold, "queue_verdict", _feed(panel, _faded(), now=later)
    )
    panel._expire_stale_hold_alerts()

    # The backing alert list is untouched - it is written before any display
    # decision and never consulted by one.
    assert len(panel._alerts) == history_before
    assert any(a.symbol == "MRK" for a in panel._alerts)

    after = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
    assert len(after) > len(before), "the expiry left no evidence behind"
    actions = [
        json.loads(line).get("action")
        for line in after.strip().splitlines()
        if line.strip()
    ]
    assert "hold_expired" in actions


def test_a_kept_row_is_recaptioned_with_what_is_true_now(panel, monkeypatch):
    """A row inside its fifteen minutes but off the high must stop claiming
    'holding highs' in the review header."""
    import regime_pause_hold

    alert = _regime_alert(at="07:50:00")
    panel._enqueue_review_alert(alert)
    bars = _faded()
    # Inside the freshness window, but price has left the high.
    later = datetime(2026, 8, 21, 8, 0)
    monkeypatch.setattr(
        regime_pause_hold, "queue_verdict", _feed(panel, bars, now=later)
    )
    panel._expire_stale_hold_alerts()
    assert alert.trigger.startswith("M5 regime-pause watch")
    assert "holding highs" not in alert.trigger
    assert "ATR off HOD" in alert.trigger
