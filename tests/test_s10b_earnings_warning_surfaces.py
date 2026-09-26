"""S10b: the short-into-earnings warning on every surface a short chart pops up.

Setup table bucket chip + tooltip, the M5 alert row's grade line, the chart
review header, the Movers weak (Rip-weak) row. Annotate only: nothing hidden,
re-ordered or muted, and LONG rows are untouched.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QColor, QHelpEvent, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QToolTip,
)

import earnings_warning  # noqa: E402

pytestmark = pytest.mark.qt

WARNING = "earnings in 4 d - shorts 3-14 d before earnings: -4.2% vs SPY (60 d)"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def warm(tmp_path, monkeypatch):
    """A warm cache: MU reports in 4 days, the stat is -4.25% (two buckets)."""
    earnings_warning.reset_cache_for_tests()
    board = tmp_path / "board.csv"
    fields = ("lookback_days", "horizon_sessions", "side", "factor_key", "value_label",
              "observation_count", "avg_spy_relative_side_return_pct")
    with board.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bucket, value, count in (("3 to < 7", -2.0, 100), ("7 to < 14", -5.0, 300)):
            writer.writerow({"lookback_days": 60, "horizon_sessions": 5, "side": "SHORT",
                             "factor_key": "days_to_next_earnings", "value_label": bucket,
                             "observation_count": count,
                             "avg_spy_relative_side_return_pct": value})
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"symbols": {
        "MU": {"events": [{"earnings_date": "2026-09-30"}]},
        "FAR": {"events": [{"earnings_date": "2026-12-30"}]},
    }}), encoding="utf-8")
    monkeypatch.setattr(earnings_warning, "market_today", lambda: date(2026, 9, 26))
    earnings_warning.warm_cache(board, history)
    yield
    earnings_warning.reset_cache_for_tests()


# ------------------------------------------------------------ setup table
from ui import theme  # noqa: E402
from ui.models.setup import SetupRow  # noqa: E402
from ui.models.setup_table_model import SetupTableModel  # noqa: E402
from ui.widgets.setup_delegate import SetupTableDelegate  # noqa: E402

CELL = QRect(0, 0, 300, 40)


def _column(key):
    return [k for k, _label in SetupTableModel.COLUMNS].index(key)


def _setup(side, days):
    raw = {"symbol": "TEST", "side": side, "priority_bucket": "near_favorite_zone"}
    return SetupRow(symbol="TEST", side=side, score=61.0, bucket="near_favorite_zone",
                    days_to_earnings=days, raw=raw)


def _view(row):
    model = SetupTableModel([row])
    view = QTableView()
    view.setModel(model)
    delegate = SetupTableDelegate(view)
    view.setItemDelegate(delegate)
    option = QStyleOptionViewItem()
    option.initFrom(view)
    option.font = view.font()
    option.rect = QRect(CELL)
    option.state = QStyle.StateFlag.State_Enabled
    return model, view, delegate, option


def _render(row, key="bucket"):
    model, _view_, delegate, option = _view(row)
    image = QImage(CELL.width(), CELL.height(), QImage.Format.Format_ARGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    try:
        delegate.paint(painter, option, model.index(0, _column(key)))
    finally:
        painter.end()
    return image


def _caution_pixels(image):
    wanted = QColor(theme.color("caution")).rgb()
    return sum(1 for y in range(image.height()) for x in range(image.width())
               if QColor(image.pixel(x, y)).rgb() == wanted)


def _hover(row, key="bucket"):
    model, view, delegate, option = _view(row)
    seen = []
    original = QToolTip.showText
    QToolTip.showText = lambda pos, text, *a, **k: seen.append(str(text))
    try:
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(4, 4), QPoint(4, 4))
        delegate.helpEvent(event, view, option, model.index(0, _column(key)))
    finally:
        QToolTip.showText = original
    return seen[-1] if seen else ""


def test_setup_short_within_14_days_paints_an_earnings_chip(app, warm):
    assert _caution_pixels(_render(_setup("SHORT", 5))) > 0
    assert _render(_setup("SHORT", 15)) == _render(_setup("SHORT", None))
    assert _caution_pixels(_render(_setup("SHORT", 15))) == 0


def test_setup_long_row_paints_what_it_paints_today(app, warm):
    assert _render(_setup("LONG", 5)) == _render(_setup("LONG", None))


def test_setup_chip_stays_in_the_bucket_cell(app, warm):
    for key in ("symbol", "side", "score"):
        assert _render(_setup("SHORT", 5), key) == _render(_setup("SHORT", None), key), key


def test_setup_bucket_tooltip_carries_the_warning(app, warm):
    text = _hover(_setup("SHORT", 4))
    assert WARNING in text
    assert "Near" in text, "the bucket's own tooltip is kept"
    assert "earnings in" not in _hover(_setup("LONG", 4))
    assert "earnings in" not in _hover(_setup("SHORT", 20))


def test_setup_bucket_tooltip_plain_wording_without_the_stat(app, tmp_path):
    earnings_warning.reset_cache_for_tests()
    try:
        assert "earnings in 4 d - short into earnings" in _hover(_setup("SHORT", 4))
    finally:
        earnings_warning.reset_cache_for_tests()


# ------------------------------------------------------------ M5 alert bar
from ui.models.bounce import BounceAlert  # noqa: E402


def _m5(symbol, side):
    return BounceAlert(time_text="07:09:19", symbol=symbol, side=side,
                       trigger="[S-TIER] VWAP reclaim", timeframe="5m", tag="green",
                       raw_text=f"VWAP reclaim {symbol}")


def test_m5_short_row_grade_line_carries_the_warning(app, warm):
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    try:
        bar.post(_m5("MU", "SHORT"))
        assert WARNING in bar.list.item(0).toolTip()
        bar.post(_m5("FAR", "SHORT"))
        bar.post(_m5("MU", "LONG"))
        tips = [bar.list.item(i).toolTip() for i in range(bar.list.count())]
        assert sum("earnings in" in tip for tip in tips) == 1
        assert bar.count() == 3, "nothing hidden"
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


def _d1(symbol, side):
    return BounceAlert(time_text="09:30:00", symbol=symbol, side=side, trigger="D1 wick",
                       timeframe="D1", tag="green", raw_text=f"D1 wick {symbol}")


def test_chart_review_header_warns_on_a_short(app, warm, tmp_path, monkeypatch):
    pane = _review(tmp_path, monkeypatch)
    try:
        pane.set_alert(_d1("MU", "SHORT"))
        assert pane.title.text().endswith("⚠ earnings in 4 d")
        assert pane.title.toolTip() == WARNING
        pane.set_alert(_d1("MU", "LONG"))
        assert "earnings" not in pane.title.text()
        assert pane.title.toolTip() == ""
        pane.set_alert(_d1("FAR", "SHORT"))
        assert "earnings" not in pane.title.text()
    finally:
        pane.close()
        pane.deleteLater()


# ------------------------------------------------------------ Movers weak
def _mover(symbol):
    return {"symbol": symbol, "move15_pct": -1.0, "move30_pct": -1.5, "day_pct": -2.0,
            "rvol": 2.0, "vs_spy15_pct": -0.8, "pop_score": -1.0, "dip_score": -1.5,
            "since_start_pct": -0.9, "note": "", "stale": False}


def _rally_board():
    return {
        "as_of": "2026-09-26T10:40:00-04:00",
        "state": {"state": "up_day", "pullback": False, "bounce": False, "rally": True,
                  "extreme_time": "10:15", "start_dt": "2026-09-26T10:15:00-04:00",
                  "spy_from_extreme_pct": 0.4, "spy_day_pct": 0.6},
        "pop": {"long": [_mover("MU")], "short": []},
        "rip": {"long": [_mover("MU")], "short": [_mover("MU"), _mover("FAR")]},
        "dip": {"long": [], "short": []},
        "mine": {"long": [], "short": []},
    }


def test_movers_rip_weak_row_warns_and_nothing_moves(app, warm):
    from ui.widgets.movers_board import MoversBoard

    board = MoversBoard(persist=False)
    try:
        board.update_board(_rally_board())
        board.flush_pending_refresh()
        weak = board.weak.model
        assert [row["symbol"] for row in weak.rows()] == ["MU", "FAR"], "order kept"
        symbol_col = [key for key, _h in weak._columns].index("symbol")
        mu = weak.data(weak.index(0, symbol_col), Qt.ItemDataRole.DisplayRole)
        far = weak.data(weak.index(1, symbol_col), Qt.ItemDataRole.DisplayRole)
        assert mu.startswith("MU ⚠4d")
        assert "⚠" not in far
        assert WARNING in weak.data(weak.index(0, symbol_col), Qt.ItemDataRole.ToolTipRole)
        strong = board.strong.model
        strong_mu = strong.data(strong.index(0, symbol_col), Qt.ItemDataRole.DisplayRole)
        assert "⚠" not in strong_mu, "the long side is silent"
    finally:
        board.deleteLater()
