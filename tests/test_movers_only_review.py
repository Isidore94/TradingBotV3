"""Movers-only chart review and the Focus "moving" flag (trader rule 2026-08-19).

"A long inside yesterday's range is probably chop. Chart review should only
show me longs above the previous day's high and shorts below the previous day's
low. Focus picks that ARE beyond their previous-day extreme should be flagged -
those are the ones actually moving. Inside-range picks appear only when I
deliberately review focus picks."

The rule is presentation. What these tests defend is that it stays that way:

- ONE predicate. The filter reads `focus_adoption_gate.mover_state`, which is
  the same `prev_day_break_state` call the adoption gate makes for its extreme
  leg. A second copy of "beyond yesterday's extreme" would eventually hide a
  name the machine had just adopted.
- It HIDES. Nothing is removed from the feed, the history, any store, any
  watchlist or Focus; nothing is muted; nothing reaches `review_policy.json`
  or the review-learning stream.
- UNKNOWN SHOWS, tagged. Missing data is uncertainty, never confirmation, and
  a data outage must not blank the review queue.
- The count is honest and one click reveals exactly those names.
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

import focus_adoption_gate  # noqa: E402
import prev_day_gate  # noqa: E402


# ---------------------------------------------------------------------------
# Task 1: one predicate, reused
# ---------------------------------------------------------------------------
class TestThePredicateIsTheGatesOwn:
    def test_the_extreme_leg_agrees_with_the_full_gate_everywhere(self):
        """Same inputs, same extreme verdict, through both entry points.

        This is the whole point of factoring rather than re-writing: if these
        ever disagree, the review is hiding names the gate would adopt.
        """
        vwap_values = (None, 9.0, 11.0)
        for side in ("long", "short"):
            for price in (None, 8.0, 10.0, 12.0):
                for prev_high, prev_low in ((None, None), (11.0, 9.0), (10.0, 10.0)):
                    extreme, _reason = focus_adoption_gate.mover_state(
                        side, price, prev_high, prev_low
                    )
                    direct = prev_day_gate.prev_day_break_state(
                        side, price, prev_high, prev_low
                    )
                    assert extreme == direct
                    for vwap in vwap_values:
                        state, reason = focus_adoption_gate.focus_adoption_gate_state(
                            side, price, prev_high, prev_low, vwap
                        )
                        if extreme == focus_adoption_gate.UNKNOWN:
                            # The gate reports "could not measure" before
                            # "measured and failed" - so an UNKNOWN extreme can
                            # never come back as a verified pass.
                            assert state != focus_adoption_gate.OPEN
                        if extreme == focus_adoption_gate.CLOSED:
                            # A verified-inside name can never come back as a
                            # gate pass. (It reads UNKNOWN when the VWAP leg
                            # itself is unmeasurable - "could not measure"
                            # outranks "measured and failed", by design.)
                            assert state != focus_adoption_gate.OPEN, reason

    def test_a_long_above_yesterdays_high_is_a_mover(self):
        state, reason = focus_adoption_gate.mover_state("long", 12.0, 11.0, 9.0)
        assert state == focus_adoption_gate.OPEN
        assert "above yesterday's high" in reason

    def test_a_long_inside_the_range_is_not(self):
        state, reason = focus_adoption_gate.mover_state("long", 10.0, 11.0, 9.0)
        assert state == focus_adoption_gate.CLOSED
        assert "inside yesterday's range" in reason

    def test_a_short_below_yesterdays_low_is_a_mover(self):
        state, _reason = focus_adoption_gate.mover_state("short", 8.0, 11.0, 9.0)
        assert state == focus_adoption_gate.OPEN
        # And the same price on the long side is not.
        assert focus_adoption_gate.mover_state("long", 8.0, 11.0, 9.0)[0] == (
            focus_adoption_gate.CLOSED
        )

    def test_no_prior_session_is_unknown_not_closed(self):
        state, reason = focus_adoption_gate.mover_state("long", 12.0, None, None)
        assert state == focus_adoption_gate.UNKNOWN
        assert "cannot verify" in reason

    def test_no_price_is_unknown(self):
        assert focus_adoption_gate.mover_state("long", None, 11.0, 9.0)[0] == (
            focus_adoption_gate.UNKNOWN
        )

    def test_unknown_is_not_a_mover(self):
        """`is_focus_mover` answers the flag question: verified only."""
        assert focus_adoption_gate.is_focus_mover("long", 12.0, 11.0, 9.0) is True
        assert focus_adoption_gate.is_focus_mover("long", 10.0, 11.0, 9.0) is False
        assert focus_adoption_gate.is_focus_mover("long", 12.0, None, None) is False

    def test_there_is_no_vwap_leg(self):
        """The filter is the extreme leg alone - a mover need not hold VWAP."""
        assert focus_adoption_gate.is_focus_mover("long", 12.0, 11.0, 9.0) is True
        passes, _reason = focus_adoption_gate.passes_focus_adoption_gate(
            "long", 12.0, 11.0, 9.0, 13.0
        )
        assert passes is False, "the full gate still refuses it on VWAP"


# ---------------------------------------------------------------------------
# Tasks 2 and 3: the desk surfaces
# ---------------------------------------------------------------------------
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


def _panel(monkeypatch, states):
    """A panel whose mover measurement is stubbed to `states` per symbol."""
    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    monkeypatch.setattr(
        panel,
        "mover_state",
        lambda symbol, side="": states.get(str(symbol).upper(), "unknown"),
    )
    return panel


class TestTheReviewFilter:
    def test_a_long_inside_yesterdays_range_is_not_queued(self, monkeypatch):
        panel = _panel(monkeypatch, {"CHOP": "closed"})
        panel.add_alert(_alert("CHOP"))

        assert panel._current_review_alert is None
        assert panel._review_queue == []
        assert panel.hidden_inside_range_count() == 1

    def test_a_mover_charts_normally(self, monkeypatch):
        panel = _panel(monkeypatch, {"MOVE": "open"})
        panel.add_alert(_alert("MOVE"))

        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "MOVE"
        assert panel.hidden_inside_range_count() == 0

    def test_a_short_inside_the_range_is_hidden_too(self, monkeypatch):
        panel = _panel(monkeypatch, {"CHOPS": "closed"})
        panel.add_alert(_alert("CHOPS", side="SHORT"))
        assert panel._current_review_alert is None
        assert panel.hidden_inside_range_count() == 1

    def test_unmeasured_names_still_show(self, monkeypatch):
        """A data outage must not blank the review queue."""
        panel = _panel(monkeypatch, {"NODATA": "unknown"})
        panel.add_alert(_alert("NODATA"))

        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "NODATA"
        assert panel.hidden_inside_range_count() == 0
        assert panel.chart_review.mover_badge.text() == "unmeasured"

    def test_a_mover_is_badged_moving(self, monkeypatch):
        panel = _panel(monkeypatch, {"MOVE": "open"})
        panel.add_alert(_alert("MOVE"))
        assert panel.chart_review.mover_badge.text() == "MOVING"

    def test_the_hidden_line_states_the_count_and_is_clickable(self, monkeypatch):
        panel = _panel(monkeypatch, {"AAA": "closed", "BBB": "closed"})
        panel.add_alert(_alert("AAA"))
        panel.add_alert(_alert("BBB"))

        button = panel.chart_review.hidden_button
        assert button.isVisible() or button.text(), "the line must be rendered"
        assert "2 hidden" in button.text()
        assert "inside yesterday's range" in button.text()
        assert "show" in button.text()

    def test_one_click_reveals_them_for_the_session(self, monkeypatch):
        panel = _panel(monkeypatch, {"AAA": "closed", "BBB": "closed"})
        panel.add_alert(_alert("AAA"))
        panel.add_alert(_alert("BBB"))

        panel.chart_review.hidden_button.click()

        charted = [panel._current_review_alert.symbol] + [
            queued.symbol for queued in panel._review_queue
        ]
        assert sorted(charted) == ["AAA", "BBB"]
        assert panel.hidden_inside_range_count() == 0
        assert panel.chart_review.hidden_button.isVisible() is False

        # ...and it stays off for the rest of the session.
        panel.add_alert(_alert("CCC"))
        assert "CCC" in [queued.symbol for queued in panel._review_queue] + [
            panel._current_review_alert.symbol
        ]

    def test_a_revealed_inside_range_name_says_so_on_the_chart(self, monkeypatch):
        panel = _panel(monkeypatch, {"AAA": "closed"})
        panel.add_alert(_alert("AAA"))
        panel.chart_review.hidden_button.click()
        assert panel.chart_review.mover_badge.text() == "inside range"

    def test_the_filter_hides_and_never_deletes(self, monkeypatch):
        """The feed, the history and every store keep the name."""
        panel = _panel(monkeypatch, {"CHOP": "closed"})
        before_ignored = set(panel._ignored_symbols)
        panel.add_alert(_alert("CHOP"))

        assert any(alert.symbol == "CHOP" for alert in panel._alerts), "still in the feed"
        assert panel._ignored_symbols == before_ignored
        assert "CHOP" not in panel._parked_symbols

    def test_the_filter_records_nothing_to_the_review_learning_stream(self, monkeypatch):
        """HARD LINE: presentation only. No review event, ever."""
        panel = _panel(monkeypatch, {"CHOP": "closed"})
        recorded: list = []
        monkeypatch.setattr(
            panel, "_record_review_event", lambda *a, **k: recorded.append((a, k))
        )
        panel.add_alert(_alert("CHOP"))
        assert recorded == []

    def test_a_deliberate_focus_review_shows_everything(self, monkeypatch):
        """The trader asked for their own list; answering with a subset lies."""
        from ui.models.bounce import FOCUS_REVIEW_TAG

        panel = _panel(monkeypatch, {"CHOP": "closed"})
        panel.add_alert(_alert("CHOP", tag=FOCUS_REVIEW_TAG))

        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "CHOP"
        assert panel.hidden_inside_range_count() == 0

    def test_an_armed_chart_watch_hit_always_shows(self, monkeypatch):
        """The exact condition the trader armed and is waiting on."""
        from ui.models.bounce import CHART_WATCH_TAG

        panel = _panel(monkeypatch, {"CHOP": "closed"})
        panel.add_alert(_alert("CHOP", tag=CHART_WATCH_TAG))
        assert panel._current_review_alert is not None
        assert panel._current_review_alert.symbol == "CHOP"

    def test_the_reveal_is_day_scoped(self, monkeypatch):
        """Tomorrow opens filtered again - a reveal is not a preference."""
        panel = _panel(monkeypatch, {"CHOP": "closed"})
        panel.reveal_hidden_reviews()
        assert panel._review_movers_only is False

        panel._ignored_market_date = "1999-01-01"
        panel._refresh_ignored_market_date()
        assert panel._review_movers_only is True
        assert panel.hidden_inside_range_count() == 0


class TestTheMeasurementIsCacheOnly:
    def test_it_reads_the_gates_predicate_over_the_desks_own_bars(self, monkeypatch):
        """No fetch: cached M5 bars and the local daily store, nothing else."""
        QApplication.instance() or QApplication([])
        from ui.panels import alert_center_panel
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

        monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
        panel = AlertCenterPanel()

        # The measurement reads the wall clock (`_measure_mover_state` calls
        # datetime.now()), so the fixture has to pin it. It used to build an
        # 11:00 session against whatever hour the suite happened to run at: a
        # bar stamped 10:50 is in the FUTURE at 07:34, completed_session_bars
        # discarded it, and the assertion read UNKNOWN. The test measured the
        # time of day, not the predicate.
        today = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        class _At11(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: D102 - stdlib signature
                return today if tz is None else today.astimezone(tz)

        monkeypatch.setattr(alert_center_panel, "datetime", _At11)
        d1 = [
            {
                "dt": yesterday.replace(hour=0, minute=0),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
            }
        ]
        m5 = [
            {
                "dt": today - timedelta(minutes=10),
                "open": 111.0,
                "high": 112.0,
                "low": 110.5,
                "close": 111.5,
            }
        ]
        monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: d1)
        monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: m5)

        assert panel.mover_state("AAA", "long") == "open"
        assert panel.mover_state("AAA", "short") == "closed"

    def test_an_unreadable_measurement_is_unknown_and_therefore_shows(self, monkeypatch):
        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

        monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
        panel = AlertCenterPanel()

        def _boom(_symbol):
            raise OSError("store unavailable")

        monkeypatch.setattr(panel, "_d1_bars_for", _boom)
        assert panel.mover_state("AAA", "long") == "unknown"


class TestTheFocusFlag:
    def _focus_panel(self, monkeypatch, states):
        QApplication.instance() or QApplication([])
        from ui.panels.focus_picks_panel import FocusPicksPanel
        from ui.services.focus_service import FocusService

        panel = FocusPicksPanel(FocusService())
        panel.set_mover_source(lambda symbol, side="": states.get(str(symbol).upper(), ""))
        return panel

    def test_a_moving_pick_is_flagged(self, monkeypatch):
        panel = self._focus_panel(monkeypatch, {"AAA": "open"})
        state = panel._live_state_for("AAA", "long")
        assert state["mover"] == "open"

    def test_an_inside_range_pick_is_not(self, monkeypatch):
        panel = self._focus_panel(monkeypatch, {"AAA": "closed"})
        assert panel._live_state_for("AAA", "long")["mover"] == "closed"

    def test_the_chip_renders_the_flag_only_for_a_mover(self, monkeypatch):
        from PySide6.QtWidgets import QLabel

        from ui.panels.focus_picks_panel import FocusStatusChip

        QApplication.instance() or QApplication([])

        def _rendered(state):
            # The chip builds its labels once and hides the ones that do not
            # apply (2026-08-21: it used to be rebuilt per refresh, 105 widget
            # trees at a time). A hidden label is not rendered, so read what is
            # actually shown rather than what exists.
            chip = FocusStatusChip("AAA", tone="long", state=state)
            return " ".join(
                label.text()
                for label in chip.findChildren(QLabel)
                if not label.isHidden()
            )

        assert "MOVING" in _rendered({"mover": "open"})
        for state in ({"mover": "closed"}, {"mover": "unknown"}, {}):
            assert "MOVING" not in _rendered(state), state

    def test_a_panel_with_no_source_shows_no_flag(self, monkeypatch):
        """A bare panel guesses nothing rather than claiming everything moves."""
        QApplication.instance() or QApplication([])
        from ui.panels.focus_picks_panel import FocusPicksPanel
        from ui.services.focus_service import FocusService

        panel = FocusPicksPanel(FocusService())
        assert panel._live_state_for("AAA", "long")["mover"] == ""

    def test_a_failing_source_shows_no_flag(self, monkeypatch):
        def _boom(_symbol, _side=""):
            raise RuntimeError("no measurement")

        panel = self._focus_panel(monkeypatch, {})
        panel.set_mover_source(_boom)
        assert panel._live_state_for("AAA", "long")["mover"] == ""
