"""M5 alerts are a list beside the chart, not a queue in front of it (trader, 2026-08-27).

"A lot of my charts to review are M5 charts... a little sidebar in between the
master AVWAP setups and the chart... the ticker and the alert type... then we
can totally purge M5 alerts from the waiting list and keep those for D1
alerts." Ordering, when asked: "latest at the top, the oldest at the bottom."

What these defend: the bar's order and its two buttons; that the Alert Center
routes intraday alerts to the bar and NOT the review queue while D1 rows,
armed hits, Focus D1 flags and the trader's own charts still queue; that the
routing withholds nothing from the feed or the evidence stream; and that the
bar is the left column of the desk, before the chart.
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

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.models.bounce import (  # noqa: E402
    CHART_WATCH_TAG,
    FOCUS_D1_EVENT_TAG,
    FOCUS_REVIEW_TAG,
    BounceAlert,
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _m5(symbol, side="LONG", *, at="07:09:19", trigger="[S-TIER] VWAP reclaim", tag="green"):
    return BounceAlert(
        time_text=at,
        symbol=symbol,
        side=side,
        trigger=trigger,
        timeframe="5m",
        tag=tag,
        raw_text=f"{trigger} {symbol} ({side.lower()})",
    )


def _d1(symbol, side="LONG"):
    return BounceAlert(
        time_text="08:25:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) zone1 reject at AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{side.lower()}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) zone1 reject at AVWAPE",
        is_d1=True,
    )


class TestTheBar:
    def _bar(self):
        from ui.widgets.m5_alert_bar import M5AlertBar

        return M5AlertBar()

    def test_newest_on_top_oldest_at_the_bottom(self):
        bar = self._bar()
        bar.post(_m5("AAA", at="07:00:00"))
        bar.post(_m5("BBB", at="07:05:00"))
        bar.post(_m5("CCC", at="07:10:00"))
        assert [a.symbol for a in bar.alerts()] == ["CCC", "BBB", "AAA"]
        assert bar.title_label.text() == "M5 alerts (3)"

    def test_every_alert_is_its_own_row_and_the_type_is_on_it(self):
        from ui.widgets.m5_alert_bar import row_text

        bar = self._bar()
        bar.post(_m5("AAA", trigger="[S-TIER] VWAP reclaim"))
        bar.post(_m5("AAA", trigger="M5 regime-pause watch · new HOD", at="07:20:00"))
        assert [a.symbol for a in bar.alerts()] == ["AAA", "AAA"]
        assert row_text(bar.alerts()[0]) == "07:20  ▲ AAA  new HOD"
        assert row_text(bar.alerts()[1]) == "07:09  ▲ AAA  VWAP reclaim"
        assert row_text(_m5("ZZZ", "SHORT", trigger="lrsi_cross_20")).endswith("▼ ZZZ  lrsi_cross_20")

    def test_copy_all_is_one_ticker_per_line_each_once_newest_first(self):
        bar = self._bar()
        bar.post(_m5("AAA", at="07:00:00"))
        bar.post(_m5("BBB", at="07:05:00"))
        bar.post(_m5("AAA", at="07:10:00"))
        text = bar.copy_all()
        assert text == "AAA\nBBB"
        assert QApplication.clipboard().text() == "AAA\nBBB"

    def test_clear_all_empties_the_screen_only(self):
        bar = self._bar()
        bar.post(_m5("AAA"))
        bar.clear_all()
        assert bar.count() == 0
        assert bar.title_label.text() == "M5 alerts"
        assert not bar.copy_button.isEnabled()

    def test_a_click_hands_back_the_alert_and_the_line_goes_away(self):
        """Trader: "after I click on an alert it should go away." """
        bar = self._bar()
        alert = _m5("AAA", at="07:00:00")
        bar.post(alert)
        bar.post(_m5("BBB", at="07:05:00"))
        got = []
        bar.alertActivated.connect(got.append)
        bar._on_item_clicked(bar.list.item(1))  # AAA is the older, lower row
        assert got == [alert]
        assert [a.symbol for a in bar.alerts()] == ["BBB"]
        assert bar.title_label.text() == "M5 alerts (1)"

    def test_the_bar_is_bounded(self):
        from ui.widgets import m5_alert_bar

        bar = self._bar()
        for i in range(m5_alert_bar.MAX_ROWS + 5):
            bar.post(_m5(f"S{i}", at=f"{i % 24:02d}:00:00"))
        assert bar.count() == m5_alert_bar.MAX_ROWS
        assert bar.alerts()[0].symbol == f"S{m5_alert_bar.MAX_ROWS + 4}", "newest kept"


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    made = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(made, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(made, "_auto_mode_now", lambda: "DESK")
    posted = []
    made.m5AlertPosted.connect(posted.append)
    yield made, posted
    made.deleteLater()


def _queued(panel) -> list[str]:
    current = panel._current_review_alert
    return ([current.symbol] if current is not None else []) + [
        a.symbol for a in panel._review_queue
    ]


class TestTheRouting:
    def test_an_m5_alert_goes_to_the_bar_and_never_the_queue(self, panel):
        made, posted = panel
        alert = _m5("NVDA")
        made.add_alert(alert)
        assert posted == [alert]
        assert _queued(made) == []
        # Withheld from nothing: the feed's backing list still has it.
        assert made._alerts and made._alerts[0] is alert

    @pytest.mark.parametrize("trigger", ["lrsi_cross_20", "M5 regime-pause watch · new HOD"])
    def test_every_intraday_kind_is_an_m5_alert(self, panel, trigger):
        made, posted = panel
        made.add_alert(_m5("AAA", "SHORT", trigger=trigger, tag="red"))
        assert [a.symbol for a in posted] == ["AAA"]
        assert _queued(made) == []

    def test_a_d1_row_still_queues(self, panel):
        made, posted = panel
        made.add_alert(_d1("MUFG", "SHORT"))
        assert posted == []
        assert _queued(made) == ["MUFG"]

    def test_a_focus_d1_flag_still_queues(self, panel):
        made, posted = panel
        made.add_alert(_m5("EPD", tag=FOCUS_D1_EVENT_TAG, trigger="Focus D1 · New 5-day high"))
        assert posted == []
        assert _queued(made) == ["EPD"]

    def test_an_armed_hit_still_goes_to_the_front_of_the_queue(self, panel):
        made, posted = panel
        made.add_alert(_d1("AAA"))
        made.add_alert(_m5("HIT", tag=CHART_WATCH_TAG, trigger="New HOD"))
        assert posted == []
        assert "HIT" in _queued(made)

    def test_a_deliberate_focus_review_still_charts(self, panel):
        made, posted = panel
        made._enqueue_review_alert(_m5("FOC", tag=FOCUS_REVIEW_TAG))
        assert posted == []
        assert _queued(made) == ["FOC"]

    def test_the_chart_in_front_still_refreshes_from_its_own_m5_alert(self, panel):
        made, posted = panel
        made.add_alert(_d1("NVDA"))
        assert made._current_review_alert.is_d1
        fresh = _m5("NVDA", trigger="[S-TIER] VWAP reclaim")
        made.add_alert(fresh)
        assert posted == [fresh], "the bar gets it too"
        assert made._current_review_alert is fresh, "and the header in front is current"
        assert _queued(made) == ["NVDA"]

    def test_the_routing_records_nothing(self, panel, monkeypatch):
        made, _posted = panel
        written = []
        monkeypatch.setattr(made, "_record_review_event", lambda action, **kw: written.append(action))
        made.add_alert(_m5("NVDA"))
        assert written == []

    def test_away_still_goes_to_the_recap_not_the_bar(self, panel, monkeypatch):
        made, posted = panel
        monkeypatch.setattr(made, "_auto_mode_now", lambda: "AWAY")
        noted = []
        monkeypatch.setattr(made, "_note_away_recap_alert", noted.append)
        made.add_alert(_m5("NVDA"))
        assert posted == []
        assert [a.symbol for a in noted] == ["NVDA"]

    def test_clicking_a_bar_row_charts_it(self, panel):
        made, posted = panel
        alert = _m5("NVDA")
        made.add_alert(alert)
        made.chart_alert(alert)
        assert made._current_review_alert is alert

    def test_clicking_a_second_bar_row_skips_the_first_instead_of_queueing_it(
        self, panel, monkeypatch
    ):
        """Trader rule 2026-08-27 (second pass): "when I click on an alert in
        the new M5 alert bar and then click to another one, it shouldn't queue
        the old M5 alert in the waiting list. It should just be considered a
        'skip for now' situation"."""
        made, _posted = panel
        written = []
        monkeypatch.setattr(
            made, "_record_review_event", lambda action, **kw: written.append((action, kw))
        )
        first, second = _m5("NVDA"), _m5("AMD")
        made.add_alert(first)
        made.add_alert(second)
        made.chart_alert(first)
        made.chart_alert(second)
        assert made._current_review_alert is second
        assert _queued(made) == ["AMD"], "NVDA is not waiting behind AMD"
        # Two impressions and one terminal action: the chart the trader left
        # is answered rather than stranded as a `shown` with no verb.
        assert [a for a, _ in written] == ["shown", "skip", "shown"]
        skipped = [kw for a, kw in written if a == "skip"][0]
        assert skipped["alert"] is first
        assert skipped["detail"] == {"reason": "clicked_away_from_m5_alert"}

    def test_a_bar_click_still_keeps_a_queued_d1_chart_at_the_head(self, panel):
        made, _posted = panel
        made.add_alert(_d1("MUFG", "SHORT"))
        made.add_alert(_d1("XOM"))
        assert _queued(made) == ["MUFG", "XOM"]
        m5 = _m5("NVDA")
        made.add_alert(m5)
        made.chart_alert(m5)
        assert _queued(made) == ["NVDA", "MUFG", "XOM"], "the D1 in front went back to the head"
        made.chart_alert(_m5("AMD"))
        assert _queued(made) == ["AMD", "MUFG", "XOM"], "and the M5 did not"

    def test_a_refreshed_d1_chart_still_holds_its_place(self, panel):
        """The header in front refreshes from its own M5 alert (test above);
        that does not turn a queued D1 chart into a bar click."""
        made, _posted = panel
        made.add_alert(_d1("NVDA"))
        made.add_alert(_m5("NVDA", trigger="[S-TIER] VWAP reclaim"))
        assert not made._current_review_alert.is_d1
        made.chart_alert(_m5("AMD"))
        assert _queued(made) == ["AMD", "NVDA"]

    def test_the_day_roll_tells_the_bar(self, panel, monkeypatch):
        made, _posted = panel
        rolled = []
        made.m5AlertsDayRolled.connect(lambda: rolled.append(1))
        made._ignored_market_date = "2000-01-01"
        made._refresh_ignored_market_date()
        assert rolled == [1]


def test_the_bar_is_the_left_column_before_the_chart():
    """Trader, second pass the same morning: "move it to the left of the
    visual chart." Bar, then the chart column, then the setups."""
    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        splitter = desk.desk_splitter
        assert splitter is not None
        assert splitter.count() == 3
        # The bar shares its column with "Today's swing picks" (2026-08-31),
        # so the column is what the splitter holds - and the bar is still the
        # first thing in it, still before the chart. That is the trader rule.
        assert splitter.widget(0) is desk.m5_column
        assert desk.m5_column.widget(0) is desk.m5_alert_bar
        assert splitter.widget(1) is desk.alert_center
        assert splitter.widget(2) is desk.master_workspace
        # Wired both ways: alerts flow in, a click flows back.
        alert = _m5("NVDA")
        desk.alert_center.m5AlertPosted.emit(alert)
        assert desk.m5_alert_bar.alerts() == [alert]
    finally:
        desk.shutdown()
        desk.close()
