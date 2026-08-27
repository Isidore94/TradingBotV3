"""A D1 recommendation against its trend is hidden (trader rule 3, 2026-08-27).

MUFG: "recommended to me as a short but it's above all the SMAs and clearly up
trending. Longs should be above the 200 SMA and shorts below the 50 SMA at
least." `test_sma_trend_gate.py` proves the rule; this proves the Alert Center
applies it to D1 recommendations only, folds it into the one display verdict
the other legs already share, and hides without deleting.
"""

from __future__ import annotations

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


def _d1_alert(symbol, side="SHORT"):
    """What the swing scanner's D1 rows look like once they reach the panel."""
    word = side.lower()
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


def _focus_d1_alert(symbol, side="LONG"):
    return BounceAlert(
        time_text="06:35:00",
        symbol=symbol,
        side=side,
        trigger="Focus D1 · New 5-day high",
        timeframe="D1",
        tag=FOCUS_D1_EVENT_TAG,
        raw_text=f"Focus D1: {symbol} new 5-day high",
    )


def _m5_alert(symbol, side="LONG"):
    return BounceAlert(
        time_text="11:30:00",
        symbol=symbol,
        side=side,
        trigger="[S-TIER] VWAP reclaim",
        timeframe="5m",
        raw_text=f"[S-TIER] {symbol}: VWAP reclaim",
    )


def _panel(monkeypatch, *, movers=None, vwaps=None, smas=None):
    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    movers = movers if movers is not None else {}
    vwaps = vwaps if vwaps is not None else {}
    smas = smas if smas is not None else {}
    monkeypatch.setattr(
        panel, "mover_state", lambda symbol, side="": movers.get(str(symbol).upper(), "open")
    )
    monkeypatch.setattr(
        panel, "vwap_state", lambda symbol, side="": vwaps.get(str(symbol).upper(), "open")
    )
    monkeypatch.setattr(
        panel, "sma_trend_state", lambda symbol, side="": smas.get(str(symbol).upper(), "open")
    )
    return panel


def _charted(panel) -> list[str]:
    current = panel._current_review_alert
    return ([current.symbol] if current is not None else []) + [
        queued.symbol for queued in panel._review_queue
    ]


class TestTheTrendLeg:
    def test_a_d1_short_over_its_sma50_is_hidden(self, monkeypatch):
        panel = _panel(monkeypatch, smas={"MUFG": "closed"})
        panel.add_alert(_d1_alert("MUFG", "SHORT"))
        assert _charted(panel) == []
        assert panel.hidden_inside_range_count() == 1

    def test_a_focus_d1_long_under_its_sma200_is_hidden(self, monkeypatch):
        panel = _panel(monkeypatch, smas={"EPD": "closed"})
        panel.add_alert(_focus_d1_alert("EPD", "LONG"))
        assert _charted(panel) == []
        assert panel.hidden_inside_range_count() == 1

    def test_a_d1_row_with_its_trend_charts_normally(self, monkeypatch):
        panel = _panel(monkeypatch, smas={"NVDA": "open"})
        panel.add_alert(_d1_alert("NVDA", "LONG"))
        assert _charted(panel) == ["NVDA"]
        assert panel.chart_review.mover_badge.text() == "MOVING"

    def test_the_trend_leg_is_not_asked_of_an_intraday_alert(self, monkeypatch):
        """The trader's floor is for D1 recommendations. An M5 bounce on a
        name under its 200 is a different trade and still charts."""
        panel = _panel(monkeypatch, smas={"AAA": "closed"})
        panel.add_alert(_m5_alert("AAA", "LONG"))
        assert _charted(panel) == ["AAA"]

    def test_an_unmeasurable_average_still_shows(self, monkeypatch):
        """A name that just listed has no SMA200. Uncertainty shows."""
        panel = _panel(monkeypatch, smas={"NEWCO": "unknown"})
        panel.add_alert(_d1_alert("NEWCO", "LONG"))
        assert _charted(panel) == ["NEWCO"]

    def test_the_hidden_line_names_the_sma(self, monkeypatch):
        panel = _panel(monkeypatch, smas={"MUFG": "closed"})
        panel.add_alert(_d1_alert("MUFG", "SHORT"))
        text = panel.chart_review.hidden_button.text()
        assert "1 hidden" in text and "SMA" in text and "show" in text

    def test_a_revealed_name_says_wrong_side_of_sma(self, monkeypatch):
        panel = _panel(monkeypatch, smas={"MUFG": "closed"})
        panel.add_alert(_d1_alert("MUFG", "SHORT"))
        panel.chart_review.hidden_button.click()
        assert _charted(panel) == ["MUFG"]
        assert panel.chart_review.mover_badge.text() == "wrong side of SMA"

    def test_the_vwap_leg_still_names_itself_first(self, monkeypatch):
        panel = _panel(monkeypatch, vwaps={"AAA": "closed"}, smas={"AAA": "closed"})
        panel.add_alert(_d1_alert("AAA", "LONG"))
        panel.chart_review.hidden_button.click()
        assert panel.chart_review.mover_badge.text() == "wrong side of VWAP"

    def test_it_hides_and_never_deletes_or_records(self, monkeypatch):
        panel = _panel(monkeypatch, smas={"MUFG": "closed"})
        written = []
        monkeypatch.setattr(
            panel, "_record_review_event", lambda action, **kw: written.append(action)
        )
        alert = _d1_alert("MUFG", "SHORT")
        panel.add_alert(alert)
        assert written == []
        assert panel._hidden_inside_range["MUFG"] is alert

    def test_a_row_that_lost_its_trend_while_waiting_is_withheld_at_show_time(
        self, monkeypatch
    ):
        smas = {"AAA": "open", "BBB": "open"}
        panel = _panel(monkeypatch, smas=smas)
        panel.add_alert(_d1_alert("AAA", "LONG"))
        panel.add_alert(_d1_alert("BBB", "LONG"))
        smas["BBB"] = "closed"
        panel._advance_review_queue()
        assert panel._current_review_alert is None
        assert panel.hidden_inside_range_count() == 1


class TestTheMeasurementUsesTheDesksOwnBars:
    def _panel_at_eleven(self, monkeypatch):
        QApplication.instance() or QApplication([])
        from ui.panels import alert_center_panel
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

        monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
        panel = AlertCenterPanel()
        today = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)

        class _At11(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: D102 - stdlib signature
                return today if tz is None else today.astimezone(tz)

        monkeypatch.setattr(alert_center_panel, "datetime", _At11)
        return panel, today

    @staticmethod
    def _daily(today, closes):
        start = today.replace(hour=0, minute=0) - timedelta(days=len(closes))
        return [
            {"dt": start + timedelta(days=i), "open": c, "high": c + 1, "low": c - 1, "close": c}
            for i, c in enumerate(closes)
        ]

    def test_mufg_a_short_above_its_sma50_is_closed(self, monkeypatch):
        panel, today = self._panel_at_eleven(monkeypatch)
        # 250 closes rising to 22; the last daily close is the price (no M5).
        closes = [18.0 + 4.0 * i / 249 for i in range(250)]
        monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: self._daily(today, closes))
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])
        assert panel.sma_trend_state("MUFG", "short") == "closed"
        assert panel.sma_trend_state("MUFG", "long") == "open"

    def test_the_last_completed_m5_close_is_the_price_when_the_bot_has_one(
        self, monkeypatch
    ):
        panel, today = self._panel_at_eleven(monkeypatch)
        closes = [20.0] * 250  # flat: SMA50 = SMA200 = 20
        monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: self._daily(today, closes))
        m5 = [
            {
                "dt": today - timedelta(minutes=10),
                "open": 19.0,
                "high": 19.2,
                "low": 18.8,
                "close": 19.0,
                "volume": 100.0,
            }
        ]
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: m5)
        assert panel.sma_trend_state("AAA", "short") == "open"
        assert panel.sma_trend_state("AAA", "long") == "closed"

    def test_too_little_history_is_unknown_and_a_bad_read_is_unknown(self, monkeypatch):
        panel, today = self._panel_at_eleven(monkeypatch)
        monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: self._daily(today, [20.0] * 60))
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])
        assert panel.sma_trend_state("NEWCO", "long") == "unknown", "no SMA200 yet"
        assert panel.sma_trend_state("NEWCO", "short") == "closed", "the SMA50 exists"

        def _boom(symbol):
            raise OSError("store gone")

        monkeypatch.setattr(panel, "_d1_bars_for", _boom)
        assert panel.sma_trend_state("NEWCO", "long") == "unknown"
        assert panel.sma_trend_state("NEWCO", "WATCH") == "unknown"
