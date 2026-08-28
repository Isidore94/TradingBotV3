"""The review chart hides the wrong side of VWAP, and it checks at show time.

Trader rule 2026-08-27, from the chart it came out of: EPD, a Focus D1 flag
fired on the 06:30 bar, reached the review pane at 07:30 sitting under session
VWAP and fading. "A stock like this really is just wasting my time." Two
rules, both presentation, both defended here the way the movers-only filter is
defended in `test_movers_only_review.py`:

- A long shows only above session VWAP, a short only below it. The predicate
  is the adoption gate's own VWAP leg (`focus_adoption_gate.session_vwap_state`)
  over the cached M5 series - one definition, no second copy.
- The filter is asked again the moment a chart is about to SHOW, not only when
  it was queued. A name that went wrong while it waited is withheld, counted,
  and one click reveals it - exactly like the inside-range names.
- It HIDES. Nothing is deleted, muted, scored or written to the review-learning
  stream. UNKNOWN shows, tagged.
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

from ui.models.bounce import BounceAlert  # noqa: E402


@pytest.fixture(autouse=True)
def _queue_mechanics_only(monkeypatch):
    """Routing off: these tests are about what the QUEUE does with a row.

    Since 2026-08-27 an ordinary intraday alert lists in the M5 alert bar
    instead of queueing a chart (trader rule; `test_qt_m5_alert_bar.py` owns
    that routing and its exemptions). The mechanics below - filters, expiry,
    verbs, badges - are the same for any row the queue holds, so they are
    exercised with the routing switched off rather than rewritten around D1
    fixtures that would drag the D1 feed into every assertion.
    """
    from ui.panels.alert_center_panel import AlertCenterPanel

    monkeypatch.setattr(
        AlertCenterPanel, "_is_m5_review_alert", staticmethod(lambda alert: False)
    )


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


def _panel(monkeypatch, movers=None, vwaps=None):
    """A panel with both legs stubbed per symbol; `vwaps` may be mutated later."""
    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    movers = movers if movers is not None else {}
    vwaps = vwaps if vwaps is not None else {}
    monkeypatch.setattr(
        panel, "mover_state", lambda symbol, side="": movers.get(str(symbol).upper(), "open")
    )
    monkeypatch.setattr(
        panel, "vwap_state", lambda symbol, side="": vwaps.get(str(symbol).upper(), "open")
    )
    return panel


def _charted(panel) -> list[str]:
    current = panel._current_review_alert
    return ([current.symbol] if current is not None else []) + [
        queued.symbol for queued in panel._review_queue
    ]


class TestTheVwapLeg:
    def test_a_long_under_session_vwap_is_hidden(self, monkeypatch):
        panel = _panel(monkeypatch, vwaps={"EPD": "closed"})
        panel.add_alert(_alert("EPD"))

        assert _charted(panel) == []
        assert panel.hidden_inside_range_count() == 1
        assert panel._alerts and panel._alerts[0].symbol == "EPD", "hidden, never deleted"

    def test_a_short_over_session_vwap_is_hidden_too(self, monkeypatch):
        panel = _panel(monkeypatch, vwaps={"EPD": "closed"})
        panel.add_alert(_alert("EPD", "SHORT"))
        assert _charted(panel) == []
        assert panel.hidden_inside_range_count() == 1

    def test_the_right_side_charts_normally(self, monkeypatch):
        panel = _panel(monkeypatch, vwaps={"MOVE": "open"})
        panel.add_alert(_alert("MOVE"))
        assert _charted(panel) == ["MOVE"]
        assert panel.chart_review.mover_badge.text() == "MOVING"

    def test_an_unmeasurable_vwap_still_shows(self, monkeypatch):
        """Missing data is uncertainty, never confirmation."""
        panel = _panel(monkeypatch, vwaps={"NODATA": "unknown"})
        panel.add_alert(_alert("NODATA"))
        assert _charted(panel) == ["NODATA"]
        # The extreme leg is verified, so the chart still says MOVING - the
        # VWAP leg had nothing to say against it.
        assert panel.chart_review.mover_badge.text() == "MOVING"

    def test_one_verified_failure_hides_even_if_the_other_leg_is_unmeasurable(
        self, monkeypatch
    ):
        """Display, not the gate: one measured reason to hide is enough."""
        panel = _panel(monkeypatch, movers={"EPD": "unknown"}, vwaps={"EPD": "closed"})
        panel.add_alert(_alert("EPD"))
        assert _charted(panel) == []
        assert panel.hidden_inside_range_count() == 1

    def test_the_hidden_line_names_both_reasons(self, monkeypatch):
        panel = _panel(monkeypatch, vwaps={"EPD": "closed"})
        panel.add_alert(_alert("EPD"))
        text = panel.chart_review.hidden_button.text()
        assert "1 hidden" in text
        assert "inside yesterday's range" in text
        assert "VWAP" in text
        assert "show" in text

    def test_a_revealed_name_says_wrong_side_of_vwap_on_the_chart(self, monkeypatch):
        panel = _panel(monkeypatch, vwaps={"EPD": "closed"})
        panel.add_alert(_alert("EPD"))
        panel.chart_review.hidden_button.click()
        assert _charted(panel) == ["EPD"]
        assert panel.chart_review.mover_badge.text() == "wrong side of VWAP"

    def test_inside_range_still_wins_the_badge(self, monkeypatch):
        """The extreme leg's own words are unchanged."""
        panel = _panel(monkeypatch, movers={"CHOP": "closed"}, vwaps={"CHOP": "closed"})
        panel.add_alert(_alert("CHOP"))
        panel.chart_review.hidden_button.click()
        assert panel.chart_review.mover_badge.text() == "inside range"

    def test_the_filter_records_nothing_to_the_review_learning_stream(self, monkeypatch):
        panel = _panel(monkeypatch, vwaps={"EPD": "closed"})
        written = []
        monkeypatch.setattr(
            panel, "_record_review_event", lambda action, **kw: written.append(action)
        )
        panel.add_alert(_alert("EPD"))
        assert written == []

    def test_a_deliberate_focus_review_and_an_armed_hit_always_show(self, monkeypatch):
        from ui.models.bounce import CHART_WATCH_TAG, FOCUS_REVIEW_TAG

        panel = _panel(monkeypatch, vwaps={"EPD": "closed", "HIT": "closed"})
        panel.add_alert(_alert("HIT", tag=CHART_WATCH_TAG))
        panel._enqueue_review_alert(_alert("EPD", tag=FOCUS_REVIEW_TAG))
        assert sorted(_charted(panel)) == ["EPD", "HIT"]
        assert panel.hidden_inside_range_count() == 0


class TestShowTimeIsWhenItIsChecked:
    def test_a_name_that_went_wrong_while_it_waited_is_withheld_when_its_turn_comes(
        self, monkeypatch
    ):
        """EPD: right at 06:30 when queued, wrong at 07:30 when shown."""
        vwaps = {"AAA": "open", "EPD": "open", "CCC": "open"}
        panel = _panel(monkeypatch, vwaps=vwaps)
        panel.add_alert(_alert("AAA"))
        panel.add_alert(_alert("EPD"))
        panel.add_alert(_alert("CCC"))
        assert _charted(panel) == ["AAA", "EPD", "CCC"]

        vwaps["EPD"] = "closed"  # an hour passes; EPD slides under VWAP
        panel._advance_review_queue()  # AAA done, next please

        assert panel._current_review_alert.symbol == "CCC", "EPD was skipped over"
        assert _charted(panel) == ["CCC"]
        assert panel.hidden_inside_range_count() == 1
        assert "1 hidden" in panel.chart_review.hidden_button.text()

    def test_a_name_that_came_right_while_it_waited_shows(self, monkeypatch):
        vwaps = {"AAA": "open", "BBB": "open"}
        movers = {"AAA": "open", "BBB": "open"}
        panel = _panel(monkeypatch, movers=movers, vwaps=vwaps)
        panel.add_alert(_alert("AAA"))
        panel.add_alert(_alert("BBB"))
        vwaps["BBB"] = "unknown"  # the measurement lapsed: uncertainty shows
        panel._advance_review_queue()
        assert panel._current_review_alert.symbol == "BBB"

    def test_a_whole_queue_gone_wrong_empties_into_the_hidden_count(self, monkeypatch):
        vwaps = {"AAA": "open", "BBB": "open", "CCC": "open"}
        panel = _panel(monkeypatch, vwaps=vwaps)
        for symbol in ("AAA", "BBB", "CCC"):
            panel.add_alert(_alert(symbol))
        vwaps.update({"BBB": "closed", "CCC": "closed"})

        panel._advance_review_queue()

        assert panel._current_review_alert is None
        assert panel._review_queue == []
        assert panel.hidden_inside_range_count() == 2

    def test_withheld_at_show_time_is_revealed_by_the_same_click(self, monkeypatch):
        vwaps = {"AAA": "open", "BBB": "open"}
        panel = _panel(monkeypatch, vwaps=vwaps)
        panel.add_alert(_alert("AAA"))
        panel.add_alert(_alert("BBB"))
        vwaps["BBB"] = "closed"
        panel._advance_review_queue()
        assert _charted(panel) == []

        panel.chart_review.hidden_button.click()

        assert _charted(panel) == ["BBB"]
        assert panel.hidden_inside_range_count() == 0

    def test_after_the_reveal_nothing_is_rechecked_for_the_session(self, monkeypatch):
        vwaps = {"AAA": "open", "BBB": "open"}
        panel = _panel(monkeypatch, vwaps=vwaps)
        panel.add_alert(_alert("AAA"))
        panel.chart_review.hidden_button.click() if panel.chart_review.hidden_button.isVisible() else None
        panel.reveal_hidden_reviews()
        panel.add_alert(_alert("BBB"))
        vwaps["BBB"] = "closed"

        panel._advance_review_queue()

        assert panel._current_review_alert.symbol == "BBB"
        assert panel.hidden_inside_range_count() == 0

    def test_an_armed_hit_is_never_withheld_at_show_time(self, monkeypatch):
        from ui.models.bounce import CHART_WATCH_TAG

        vwaps = {"AAA": "open", "HIT": "open"}
        panel = _panel(monkeypatch, vwaps=vwaps)
        panel.add_alert(_alert("AAA"))
        panel._review_queue.append(_alert("HIT", tag=CHART_WATCH_TAG))
        vwaps["HIT"] = "closed"
        panel._advance_review_queue()
        assert panel._current_review_alert.symbol == "HIT"


class TestTheMeasurementIsTheGatesOwnOverCachedBars:
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
    def _session(today, closes, *, volume=1000.0):
        start = today.replace(hour=6, minute=30)
        return [
            {
                "dt": start + timedelta(minutes=5 * index),
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": volume,
            }
            for index, close in enumerate(closes)
        ]

    def test_a_close_under_the_session_vwap_is_closed_for_a_long_and_open_for_a_short(
        self, monkeypatch
    ):
        panel, today = self._panel_at_eleven(monkeypatch)
        # Ten bars at 100, then a slide to 97: session VWAP ~99.3, last close 97.
        bars = self._session(today, [100.0] * 10 + [99.0, 98.0, 97.0])
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: bars)

        assert panel.vwap_state("EPD", "long") == "closed"
        assert panel.vwap_state("EPD", "short") == "open"

    def test_a_close_over_the_session_vwap_is_open_for_a_long(self, monkeypatch):
        panel, today = self._panel_at_eleven(monkeypatch)
        bars = self._session(today, [100.0] * 10 + [101.0, 102.0, 103.0])
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: bars)
        assert panel.vwap_state("EPD", "long") == "open"

    def test_no_bars_no_volume_or_no_side_is_unknown(self, monkeypatch):
        panel, today = self._panel_at_eleven(monkeypatch)
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])
        assert panel.vwap_state("EPD", "long") == "unknown"
        # A series with no volume has no VWAP to be on either side of.
        bars = self._session(today, [100.0] * 5 + [97.0], volume=0.0)
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: bars)
        assert panel.vwap_state("EPD", "long") == "unknown"
        assert panel.vwap_state("EPD", "WATCH") == "unknown"

    def test_an_unreadable_measurement_is_unknown_and_therefore_shows(self, monkeypatch):
        panel, _today = self._panel_at_eleven(monkeypatch)

        def _boom(symbol, **kw):
            raise OSError("cache gone")

        monkeypatch.setattr(panel, "_m5_bars_for", _boom)
        assert panel.vwap_state("EPD", "long") == "unknown"

    def test_the_memo_is_keyed_on_the_bars_not_the_clock(self, monkeypatch):
        """A new bar is a new key; the same bars are not re-derived."""
        panel, today = self._panel_at_eleven(monkeypatch)
        bars = self._session(today, [100.0] * 10 + [97.0])
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: list(bars))
        assert panel.vwap_state("EPD", "long") == "closed"
        calls = []
        import regime_pause_hold

        real = regime_pause_hold.session_levels

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(regime_pause_hold, "session_levels", counting)
        assert panel.vwap_state("EPD", "long") == "closed"
        assert calls == [], "same bars, remembered answer"
        bars.append(
            {
                "dt": bars[-1]["dt"] + timedelta(minutes=5),
                "open": 104.0,
                "high": 104.5,
                "low": 103.5,
                "close": 104.0,
                "volume": 5000.0,
            }
        )
        assert panel.vwap_state("EPD", "long") == "open"
        assert calls == [1], "a new bar was measured"
