"""Trader, 2026-09-15: the Master AVWAP setups table cycles and hides.

*"when i click on master avwap setups tab and then I click the veto or like
and claim buttons it should cycle it to the next pick. additionally vetoing it
for the day SHOULD remove it from the list (but the stock should still be
tracked for setup tracker purposes)"*

Two contracts, pinned here:

* a chart opened FROM the setups table carries a `next_pick` callback; the
  Alert Center calls it instead of its waiting list when that chart is vetoed,
  claimed or stepped past, and drops it the moment any other chart takes the
  pane. The table's callback charts the next visible row (skipping the vetoed
  symbol and anything rejected today) and says when the list has run out;
* a symbol vetoed / disliked / parked today is HIDDEN from the table and
  counted; `Show vetoed` brings the rows back in the same order. A day-trade
  pass and an M5 click-away hide nothing. Nothing is deleted or written, and
  the setup tracker never reads the filter.
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
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- helpers
def _rows(*specs):
    from ui.models.setup import SetupRow

    return [SetupRow(symbol=symbol, side=side, score=score) for symbol, side, score in specs]


def _decisions(rejected: dict[str, tuple[str, ...]]):
    from pick_feedback import DayDecisions

    return DayDecisions(
        trade_date="2026-09-15",
        rejected={symbol: tuple((kind, "2026-09-15T10:00:00") for kind in kinds) for symbol, kinds in rejected.items()},
    )


class _Sink:
    """The desk's `chart_symbol` as a recorder: what was charted, with which callback."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def __call__(self, symbol, *, side="", origin="", next_pick=None):
        self.calls.append((symbol, side, next_pick))
        return True

    @property
    def symbols(self) -> list[str]:
        return [symbol for symbol, _side, _cb in self.calls]


def _panel(tmp_path, monkeypatch, rows):
    import chart_snapshot
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])
    panel = MasterAvwapPanel(None, review_events_path=tmp_path / "events.jsonl")
    panel.set_rows(rows)
    return panel


def _click_symbol(panel, proxy_row: int) -> None:
    column = next(c for c, (key, _l) in enumerate(panel.model.COLUMNS) if key == "symbol")
    panel.table.clicked.emit(panel.proxy.index(proxy_row, column))


def _visible(panel) -> list[str]:
    return [panel._row_at_proxy(r).symbol for r in range(panel.proxy.rowCount())]


# --------------------------------------------------------------------------- the snapshot
def test_rejected_symbols_reads_only_the_swing_side_verdicts():
    decisions = _decisions(
        {
            "VETO": ("veto",),
            "DISL": ("dislike",),
            "PARK": ("remove_today",),
            "NOTT": ("not_today",),
            "PASS": ("pass",),
            "CLIK": ("m5_click_away",),
        }
    )
    assert decisions.rejected_symbols() == {"VETO", "DISL", "PARK", "NOTT"}
    from pick_feedback import REJECT_KINDS

    assert decisions.rejected_symbols(REJECT_KINDS) == {"VETO", "DISL", "PARK", "NOTT", "PASS", "CLIK"}


# --------------------------------------------------------------------------- the proxy
def test_the_proxy_hides_rejected_rows_counts_them_and_shows_them_on_request():
    from ui.models.setup_table_model import SetupFilterProxyModel, SetupTableModel

    model = SetupTableModel()
    model.set_rows(_rows(("AAA", "LONG", 90.0), ("BBB", "SHORT", 80.0), ("CCC", "LONG", 70.0)))
    proxy = SetupFilterProxyModel()
    proxy.setSourceModel(model)

    def visible():
        return [model.row_at(proxy.mapToSource(proxy.index(r, 0)).row()).symbol for r in range(proxy.rowCount())]

    assert visible() == ["AAA", "BBB", "CCC"]
    proxy.set_filters(rejected_symbols={"bbb"})
    assert visible() == ["AAA", "CCC"]
    assert proxy.hidden_rejected() == 1
    proxy.set_filters(min_score=0.0)  # a partial call leaves the reject filter alone
    assert visible() == ["AAA", "CCC"]
    proxy.set_filters(show_rejected=True)
    assert visible() == ["AAA", "BBB", "CCC"], "same rows, same order - hidden, never deleted"
    assert proxy.hidden_rejected() == 0
    proxy.set_filters(show_rejected=False, rejected_symbols=())
    assert visible() == ["AAA", "BBB", "CCC"]
    assert len(model.rows()) == 3


# --------------------------------------------------------------------------- the panel
def test_the_decision_snapshot_hides_the_vetoed_row_and_labels_the_box(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch, _rows(("AAA", "LONG", 90.0), ("BBB", "SHORT", 80.0), ("CCC", "LONG", 70.0)))
    try:
        assert _visible(panel) == ["AAA", "BBB", "CCC"]
        assert panel.show_vetoed_toggle.text() == "Show vetoed"
        assert not panel.show_vetoed_toggle.isChecked(), "hidden by default"

        panel._on_day_decisions_ready(_decisions({"BBB": ("veto",), "CCC": ("pass",)}))

        assert _visible(panel) == ["AAA", "CCC"], "a veto hides; a day-trade pass does not"
        assert panel.show_vetoed_toggle.text() == "Show vetoed (1)"
        assert len(panel.model.rows()) == 3, "the row is hidden, never deleted"

        panel.show_vetoed_toggle.setChecked(True)
        assert _visible(panel) == ["AAA", "BBB", "CCC"]
        panel.show_vetoed_toggle.setChecked(False)
        assert _visible(panel) == ["AAA", "CCC"]
    finally:
        panel.close()


def test_the_show_vetoed_choice_is_remembered(tmp_path, monkeypatch):
    import project_paths
    from ui.panels import master_avwap_panel as mod

    saved: dict[str, object] = {}
    monkeypatch.setattr(mod, "save_local_setting", lambda key, value: saved.__setitem__(key, value))
    monkeypatch.setattr(mod, "get_local_setting", lambda key, default=None: saved.get(key, default))
    monkeypatch.setattr(project_paths, "save_local_setting", lambda key, value: saved.__setitem__(key, value))
    panel = _panel(tmp_path, monkeypatch, _rows(("AAA", "LONG", 90.0)))
    try:
        panel.show_vetoed_toggle.setChecked(True)
        assert saved.get(mod.SETTING_SHOW_VETOED) is True
    finally:
        panel.close()
    again = _panel(tmp_path, monkeypatch, _rows(("AAA", "LONG", 90.0)))
    try:
        assert again.show_vetoed_toggle.isChecked()
        assert again.proxy.show_rejected is True
    finally:
        again.close()


def test_a_charted_row_carries_the_way_to_the_next_row(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch, _rows(("AAA", "LONG", 90.0), ("BBB", "SHORT", 80.0), ("CCC", "LONG", 70.0)))
    sink = _Sink()
    panel.set_chart_sink(sink)
    try:
        _click_symbol(panel, 0)
        assert sink.symbols == ["AAA"]
        _symbol, side, advance = sink.calls[-1]
        assert side == "LONG" and callable(advance)

        assert advance() is True
        assert sink.symbols == ["AAA", "BBB"]
        assert sink.calls[-1][1] == "SHORT"
        assert panel.table.currentIndex().row() == 1, "the table's selection follows the walk"

        assert sink.calls[-1][2]() is True
        assert sink.symbols == ["AAA", "BBB", "CCC"]

        assert sink.calls[-1][2]() is False, "nothing after the last row"
        assert sink.symbols == ["AAA", "BBB", "CCC"]
        assert "End of the setups list" in panel.status_label.text()
    finally:
        panel.close()


def test_the_walk_skips_the_vetoed_symbol_and_todays_rejects(tmp_path, monkeypatch):
    panel = _panel(
        tmp_path,
        monkeypatch,
        _rows(("AAA", "LONG", 90.0), ("AAA", "SHORT", 85.0), ("BBB", "SHORT", 80.0), ("CCC", "LONG", 70.0)),
    )
    sink = _Sink()
    panel.set_chart_sink(sink)
    try:
        panel._on_day_decisions_ready(_decisions({"BBB": ("dislike",)}))
        assert _visible(panel) == ["AAA", "AAA", "CCC"]
        _click_symbol(panel, 0)
        assert sink.calls[-1][2]() is True
        assert sink.symbols == ["AAA", "CCC"], "the other AAA side and the disliked BBB are skipped"
    finally:
        panel.close()


def test_the_walk_survives_the_charted_row_leaving_the_table(tmp_path, monkeypatch):
    """The hide filter may catch up with the veto BEFORE the advance runs: the
    row that took the vetoed row's place is then the next one."""
    panel = _panel(tmp_path, monkeypatch, _rows(("AAA", "LONG", 90.0), ("BBB", "SHORT", 80.0), ("CCC", "LONG", 70.0)))
    sink = _Sink()
    panel.set_chart_sink(sink)
    try:
        _click_symbol(panel, 1)  # BBB
        panel._on_day_decisions_ready(_decisions({"BBB": ("veto",)}))
        assert _visible(panel) == ["AAA", "CCC"]
        assert sink.calls[-1][2]() is True
        assert sink.symbols == ["BBB", "CCC"]
    finally:
        panel.close()


def test_a_standalone_panel_without_a_sink_charts_nothing_on_advance(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch, _rows(("AAA", "LONG", 90.0), ("BBB", "SHORT", 80.0)))
    try:
        assert panel._chart_next_pick("AAA", "LONG", 0) is False
    finally:
        panel.close()


# --------------------------------------------------------------------------- the alert center
def _d1_alert(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="08:25:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) zone1 reject at AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{side.lower()}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) zone1 reject",
        is_d1=True,
    )


def _neuter_cohort_merges(rail) -> None:
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True, "added": 0}


@pytest.fixture
def center(tmp_path, monkeypatch):
    """A bare Alert Center with tmp stores and ONE D1 alert waiting in the queue."""
    import pick_feedback
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    pick_feedback.clear_reviewed_today_cache()
    made = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(made, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(made, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(made.chart_review, "_reviewed_symbols", lambda: set())
    monkeypatch.setattr(made.chart_review.capture_rail, "_annotations_path", tmp_path / "trader_annotations.jsonl")
    _neuter_cohort_merges(made.chart_review.capture_rail)
    made.add_alert(_d1_alert("AAPL"))
    made.add_alert(_d1_alert("NVDA", "SHORT"))
    assert made._current_review_alert is not None and made._current_review_alert.symbol == "AAPL"
    yield made
    made.close()


def _chart_from_setups(center, symbol: str, calls: list[str], *, answer: bool = True):
    def advance() -> bool:
        calls.append(symbol)
        return answer

    assert center.chart_symbol(symbol, side="LONG", origin="the Master AVWAP setups", next_pick=advance)
    assert center._current_review_alert.symbol == symbol
    return advance


def test_a_veto_on_a_setups_chart_calls_the_tables_advance_not_the_queue(center):
    calls: list[str] = []
    decisions: list[int] = []
    center.reviewDecisionRecorded.connect(lambda: decisions.append(1))
    _chart_from_setups(center, "XYZ", calls)
    queued_before = [alert.symbol for alert in center._review_queue]

    center._retire_after_veto(center._current_review_alert)

    assert calls == ["XYZ"], "the table's callback was asked for the next row"
    assert [alert.symbol for alert in center._review_queue] == queued_before, "the waiting list was not touched"
    assert decisions == [1], "the setups table is told a decision landed"
    assert center._manual_next_pick is None, "consumed, never called twice"


def test_a_claim_on_a_setups_chart_calls_the_tables_advance(center):
    calls: list[str] = []
    decisions: list[int] = []
    center.reviewDecisionRecorded.connect(lambda: decisions.append(1))
    _chart_from_setups(center, "XYZ", calls)

    center._retire_claimed_review(center._current_review_alert)

    assert calls == ["XYZ"]
    assert center._manual_next_pick is None


def test_the_next_verb_on_a_setups_chart_walks_the_table(center):
    calls: list[str] = []
    _chart_from_setups(center, "XYZ", calls)
    center._advance_review_queue()
    assert calls == ["XYZ"]


def test_when_the_table_has_nothing_left_the_waiting_list_takes_over(center):
    calls: list[str] = []
    _chart_from_setups(center, "XYZ", calls, answer=False)
    center._retire_after_veto(center._current_review_alert)
    assert calls == ["XYZ"]
    assert center._current_review_alert is not None
    assert center._current_review_alert.symbol in {"AAPL", "NVDA"}, "the queue's next chart came up"


def test_a_callback_that_raises_never_costs_the_advance(center):
    def boom() -> bool:
        raise RuntimeError("table gone")

    assert center.chart_symbol("XYZ", side="LONG", origin="the Master AVWAP setups", next_pick=boom)
    center._retire_after_veto(center._current_review_alert)
    assert center._current_review_alert is not None
    assert center._current_review_alert.symbol in {"AAPL", "NVDA"}


def test_any_other_chart_drops_the_tables_callback(center):
    calls: list[str] = []
    _chart_from_setups(center, "XYZ", calls)
    center.chart_symbol("QQQ")  # the lookup box passes no callback
    assert center._manual_next_pick is None
    center._retire_after_veto(center._current_review_alert)
    assert calls == [], "the lookup-box chart did not inherit the table's walk"

    _chart_from_setups(center, "XYZ", calls)
    center._select_review_alert(_d1_alert("TSLA"))  # a feed-row click
    assert center._manual_next_pick is None
    center._retire_after_veto(center._current_review_alert)
    assert calls == []


def test_the_default_chart_symbol_registers_no_callback(center):
    center.chart_symbol("QQQ", side="LONG", origin="the RS/RW board")
    assert center._manual_next_pick is None
