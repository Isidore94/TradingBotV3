"""S11: the desk only FORMATS the night's exit-window numbers.

The Daytrade Tracker "Exit by" column, the M5 alert row hover, the chart review
header line and the Trade Mentor's exit question each show exactly the text
`exit_windows` formats from the published payload. Every compute and file read
in `exit_windows` is made to raise once the cache is warm, so a surface that
computed or read a file on the Qt thread fails here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for path in (ROOT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

import exit_windows as ew  # noqa: E402
from tests.test_s11_exit_windows import eight_events  # noqa: E402

pytestmark = pytest.mark.qt

WINDOW = ("2026-08-01", "2026-09-24")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def payload():
    return ew.build_payload(eight_events(), as_of="2026-09-24", window=WINDOW)


@pytest.fixture
def warm(tmp_path, monkeypatch, payload):
    """A warm cache, then every compute and read in `exit_windows` raises."""
    import json

    ew.reset_cache_for_tests()
    target = tmp_path / "exit_windows.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    ew.warm_cache(target)

    def _boom(*_a, **_k):
        raise AssertionError("the desk computed or read exit windows on the Qt thread")

    for name in ("build_payload", "read_window_rows", "read_payload", "alert_result", "warm_cache"):
        monkeypatch.setattr(ew, name, _boom)
    yield ew.lookup(payload)
    ew.reset_cache_for_tests()


VWAP_LINE = (
    "Exit by: this family usually peaks inside 30 min; "
    "+1R or 60 min has beaten holding by 0.52R (n 5)"
)


# ------------------------------------------------------------ Daytrade Tracker
def test_tracker_exit_by_column_formats_the_payload(app, monkeypatch, payload):
    from ui.panels import daytrade_tracker_panel as dtp

    assert ("exit_by", "Exit by") in dtp.PERFORMANCE_COLUMNS
    monkeypatch.setattr(ew, "build_payload", lambda *a, **k: (_ for _ in ()).throw(AssertionError("computed")))
    panel = dtp.DaytradeTrackerPanel()
    try:
        panel._performance_rows = [
            {"dimension": "bounce_type", "segment": "vwap", "direction": "LONG", "sample_count": 5},
            {"dimension": "bounce_type", "segment": "never_seen", "direction": "LONG", "sample_count": 1},
            {"dimension": "direction", "segment": "LONG", "direction": "LONG", "sample_count": 9},
        ]
        panel._on_held_run_loaded({
            "summaries": {}, "window": {}, "outcome_coverage": {},
            "setup_grades": {}, "exit_windows": payload,
        })
        rows = {row["segment"]: row for row in panel._dimension_tables["bounce_type"][1].rows()}
        assert rows["vwap"]["exit_by"] == "peak <= 60 min 80%; +1R/60m +0.10R vs hold -0.42R"
        assert rows["never_seen"]["exit_by"] == ""
    finally:
        panel.deleteLater()


def test_tracker_worker_read_carries_the_payload(monkeypatch, payload):
    from ui.panels import daytrade_tracker_panel as dtp

    monkeypatch.setattr(ew, "read_payload", lambda *a, **k: payload)
    assert dtp.load_held_run_report()["exit_windows"] is payload


# ------------------------------------------------------------ M5 alert bar
from ui.models.bounce import BounceAlert  # noqa: E402


def _m5(symbol, side, bounce_types):
    return BounceAlert(time_text="07:09:19", symbol=symbol, side=side,
                       trigger="[S-TIER] VWAP reclaim", timeframe="5m", tag="green",
                       raw_text=f"VWAP reclaim {symbol}",
                       payload={"feedback": {"bounce_types": bounce_types}})


def test_m5_row_hover_carries_the_exit_line(app, warm):
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    try:
        bar.post(_m5("AAA", "LONG", "vwap"))
        assert VWAP_LINE in bar.list.item(0).toolTip()
        bar.post(_m5("BBB", "LONG", "never_seen"))
        assert "Exit by" not in bar.list.item(0).toolTip()
        assert bar.count() == 2, "nothing hidden"
    finally:
        bar.deleteLater()


# ------------------------------------------------------------ chart review header
def _review(tmp_path, monkeypatch):
    from ui.widgets.alert_chart_review import AlertChartReview
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_a, **_k: None)
    return AlertChartReview(
        annotations_path=tmp_path / "trader_annotations.jsonl", mentor_context_service=None
    )


def test_chart_review_header_line(app, warm, tmp_path, monkeypatch):
    pane = _review(tmp_path, monkeypatch)
    try:
        pane.set_alert(_m5("AAA", "LONG", "vwap"))
        assert pane.exit_window_label.text() == VWAP_LINE
        d1 = BounceAlert(time_text="09:30:00", symbol="AAA", side="LONG", trigger="D1 wick",
                         timeframe="D1", tag="green", raw_text="D1 wick AAA",
                         payload={"feedback": {"bounce_types": "vwap"}})
        pane.set_alert(d1)
        assert pane.exit_window_label.text() == "", "M5 families only"
        pane.set_alert(_m5("AAA", "LONG", "vwap"))
        pane.clear()
        assert pane.exit_window_label.text() == ""
    finally:
        pane.close()
        pane.deleteLater()


# ------------------------------------------------------------ Trade Mentor
def _question(**overrides):
    base = dict(trade_id="t1", symbol="AAA", direction="long", setup_guess="vwap",
                opened_at="2026-09-24T10:05:00-04:00", trade_date="2026-09-24")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_mentor_exit_question_quotes_the_fact(warm):
    from ui.widgets.trade_mentor_card import _exit_window_quote

    assert _exit_window_quote(_question(), "2026-09-24") == (
        "Fact, not a rule: vwap LONG M5 alerts - "
        "peak <= 60 min 80%; +1R/60m +0.10R vs hold -0.42R (n 5)."
    )
    # A setup the file does not know quotes all M5 alerts on that side.
    assert _exit_window_quote(_question(setup_guess="swing_thing"), "2026-09-24").startswith(
        "Fact, not a rule: all LONG M5 alerts - "
    )
    # A trade held past its entry session is not a day trade: silent.
    assert _exit_window_quote(_question(), "2026-09-25") == ""


def test_mentor_quote_is_silent_without_the_file():
    from ui.widgets.trade_mentor_card import _exit_window_quote

    ew.reset_cache_for_tests()
    try:
        assert _exit_window_quote(_question(), "2026-09-24") == ""
    finally:
        ew.reset_cache_for_tests()
