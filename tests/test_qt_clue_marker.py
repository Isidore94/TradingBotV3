"""Day Recap coach: hindsight clue marking on the desk chart (clue_marker)."""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the chart is PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt, QThreadPool  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
FIRST = datetime(2026, 9, 22, 6, 30, tzinfo=PACIFIC)
COUNT = 40
# The last bar opened 2 minutes ago: it is still forming.
NOW = FIRST + timedelta(minutes=5 * (COUNT - 1) + 2)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _bars(count: int = COUNT, *, naive: bool = False, base: float = 100.0) -> list[dict]:
    rows = []
    for index in range(count):
        stamp = FIRST + timedelta(minutes=5 * index)
        rows.append({
            "dt": stamp.replace(tzinfo=None) if naive else stamp,
            "open": base + index * 0.05,
            "high": base + 0.4 + index * 0.05,
            "low": base - 0.4 + index * 0.05,
            "close": base + 0.2 + index * 0.05,
            "volume": 1000.0 + index,
        })
    return rows


@pytest.fixture()
def chart(qapp):
    from ui.widgets.candle_chart import CandleChart

    widget = CandleChart()
    widget.resize(900, 600)
    widget.set_data(_bars(), timeframe="m5")
    widget.show()
    qapp.processEvents()
    yield widget
    widget.hide()
    widget.deleteLater()


def _click(qapp, chart, x: float, price: float) -> None:
    y = chart._y(price)
    local = chart.mapFromScene(chart.getPlotItem().vb.mapViewToScene(QPointF(x, y)))
    qapp.sendEvent(
        chart.viewport(),
        QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(local), chart.mapToGlobal(local),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        ),
    )


def _wait(qapp, pool, predicate, timeout=5.0) -> None:
    pool.waitForDone(int(timeout * 1000))
    end = time.time() + timeout
    while not predicate() and time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# snapping and time
# ---------------------------------------------------------------------------
def test_snap_never_picks_the_forming_bar():
    from ui.widgets import clue_marker as cm

    bars = _bars()
    assert not cm.is_completed(bars[-1], "M5", NOW)
    assert cm.is_completed(bars[-2], "M5", NOW)
    assert cm.snap_to_completed(bars, COUNT - 1, "M5", NOW) == COUNT - 2
    assert cm.snap_to_completed(bars, COUNT + 5, "M5", NOW) == COUNT - 2
    assert cm.snap_to_completed(bars, 10, "M5", NOW) == 10
    assert cm.snap_to_completed(bars, 10, "M5", FIRST) is None


def test_a_daily_bar_is_complete_only_after_the_close():
    from ui.widgets import clue_marker as cm

    bar = {"dt": datetime(2026, 9, 22), "close": 1.0}
    et = ZoneInfo("America/New_York")
    assert not cm.is_completed(bar, "D1", datetime(2026, 9, 22, 15, 59, tzinfo=et))
    assert cm.is_completed(bar, "D1", datetime(2026, 9, 22, 16, 0, tzinfo=et))


def test_a_naive_bar_time_comes_back_tz_aware():
    from ui.widgets import clue_marker as cm

    stamp = cm.aware_bar_time(datetime(2026, 9, 22, 7, 0))
    assert stamp.tzinfo is not None and stamp.utcoffset() is not None
    assert cm.aware_bar_time("junk") is None


# ---------------------------------------------------------------------------
# clue mode on the chart
# ---------------------------------------------------------------------------
def test_a_click_in_clue_mode_marks_the_completed_bar_with_a_tz_aware_time(qapp, chart):
    from ui.widgets.clue_marker import enable_clue_marking

    seen: list[tuple] = []
    bars_clicked: list[int] = []
    chart.barClicked.connect(bars_clicked.append)
    marker = enable_clue_marking(chart, lambda t, p: seen.append((t, p)), clock=lambda: NOW)

    _click(qapp, chart, 20.0, 101.1)
    assert seen == [] and bars_clicked == [20]  # clue mode is off: a normal click

    marker.set_active(True)
    assert chart.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    _click(qapp, chart, COUNT - 1, 102.0)  # the forming bar
    assert len(seen) == 1
    stamp, price = seen[0]
    assert stamp.tzinfo is not None and stamp == _bars()[COUNT - 2]["dt"]
    assert price == pytest.approx(102.0, rel=1e-3)
    assert bars_clicked == [20]  # the clue click was consumed: no level menu, no pan


def test_escape_leaves_clue_mode_and_restores_the_cursor(qapp, chart):
    from ui.widgets.clue_marker import enable_clue_marking

    seen: list = []
    marker = enable_clue_marking(chart, lambda t, p: seen.append(t), clock=lambda: NOW)
    marker.set_active(True)
    qapp.sendEvent(chart, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not marker.is_active()
    assert chart.viewport().cursor().shape() != Qt.CursorShape.CrossCursor
    _click(qapp, chart, 10.0, 100.5)
    assert seen == []
    marker.toggle()
    assert marker.is_active()
    marker.toggle()
    assert not marker.is_active()


# ---------------------------------------------------------------------------
# the form
# ---------------------------------------------------------------------------
def _payload() -> dict:
    return {
        "session_date": "2026-09-22", "symbol": "AMD", "timeframe": "M5",
        "bar_time": _bars()[10]["dt"], "price": 100.6, "card_id": "card-1",
        "trade_id": "", "pick_id": "",
    }


def test_every_clue_tag_has_plain_words_and_a_unique_glyph():
    import recap_store
    from ui.widgets import clue_marker as cm

    assert set(cm.CLUE_TAG_LABELS) == set(recap_store.CLUE_TAGS)
    assert set(cm.CLUE_TAG_INITIALS) == set(recap_store.CLUE_TAGS)
    assert len(set(cm.CLUE_TAG_INITIALS.values())) == len(cm.CLUE_TAG_INITIALS)
    assert all("_" not in label for label in cm.CLUE_TAG_LABELS.values())


def test_the_form_saves_on_a_worker_thread_and_says_saved(qapp):
    from ui.widgets.clue_marker import ClueForm

    calls: list[tuple[dict, threading.Thread]] = []

    def fake_writer(**fields):
        calls.append((fields, threading.current_thread()))
        return {"id": "rc-1", "kind": "clue", **fields}

    pool = QThreadPool()
    form = ClueForm(writer=fake_writer, pool=pool)
    saved: list[dict] = []
    form.saved.connect(saved.append)
    form.open_for(**_payload())
    assert not form.save_button.isEnabled()  # no tag yet
    form.chips["volume_dry_up"].click()
    form.note.setText("  volume dried up into the pullback ")
    assert form.save_button.isEnabled()
    form.save()
    _wait(qapp, pool, lambda: bool(saved))

    assert form.status.text() == "saved"
    assert len(calls) == 1
    fields, thread = calls[0]
    assert thread is not threading.main_thread()
    assert fields["clue_tag"] == "volume_dry_up"
    assert fields["text"] == "volume dried up into the pullback"
    assert fields["card_id"] == "card-1"
    assert saved[0]["id"] == "rc-1"


def test_a_failed_write_says_not_saved(qapp):
    import recap_store
    from ui.widgets.clue_marker import ClueForm

    def broken_writer(**_fields):
        raise recap_store.RecapWriteError("disk full")

    pool = QThreadPool()
    form = ClueForm(writer=broken_writer, pool=pool)
    failed: list[str] = []
    form.failed.connect(failed.append)
    form.open_for(**_payload())
    form.chips["gap"].click()
    form.save()
    _wait(qapp, pool, lambda: bool(failed))
    assert form.status.text() == "not saved"
    assert "disk full" in failed[0]
    assert form.save_button.isEnabled()  # the trader can try again


def test_the_form_writes_a_real_row_through_record_clue(qapp, tmp_path):
    import recap_store
    from ui.widgets.clue_marker import ClueForm

    target = tmp_path / "recap.jsonl"
    pool = QThreadPool()
    form = ClueForm(writer=functools.partial(recap_store.record_clue, path=target), pool=pool)
    saved: list[dict] = []
    form.saved.connect(saved.append)
    form.open_for(**_payload(), context={"bar": {"close": 100.7}, "spy_close": None})
    form.chips["vwap_reclaim"].click()
    form.save()
    _wait(qapp, pool, lambda: bool(saved))
    rows = recap_store.records_for("2026-09-22", [recap_store.KIND_CLUE], path=target)
    assert len(rows) == 1
    assert rows[0]["clue_tag"] == "vwap_reclaim"
    assert rows[0]["context"] == {"bar": {"close": 100.7}, "spy_close": None}
    assert datetime.fromisoformat(rows[0]["bar_time"]).tzinfo is not None


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def _clue(index: int, tag: str, price: float, text: str = "") -> dict:
    return {
        "id": f"rc-{index}", "symbol": "AMD", "timeframe": "M5",
        "bar_time": _bars()[index]["dt"].isoformat(), "price": price,
        "clue_tag": tag, "text": text,
    }


def test_draw_clues_places_labelled_markers_and_keeps_the_scale(qapp, chart):
    from ui.widgets import clue_marker as cm

    assert cm.draw_clues(chart, []) == 0
    assert not hasattr(chart, "_clue_layer")  # an unmarked chart grows nothing
    before = chart.getPlotItem().vb.viewRange()
    stray = dict(_clue(3, "news", 100.2), bar_time="2026-09-21T09:00:00-04:00")
    drawn = cm.draw_clues(chart, [
        _clue(5, "volume_dry_up", 100.3, "dried up"),
        _clue(15, "vwap_reclaim", 100.9),
        _clue(30, "level_break", 101.6),
        stray,
        dict(_clue(8, "gap", 100.0), price=-1),
    ])
    assert drawn == 3
    assert cm.drawn_clue_count(chart) == 3
    assert chart.getPlotItem().vb.viewRange() == before
    layer = chart._clue_layer
    label = layer.labels[0]
    assert label.toPlainText() == "VD"
    assert "Volume dried up" in label.toolTip() and "dried up" in label.toolTip()
    assert label.pos().x() == 5.0
    assert label.pos().y() == pytest.approx(chart._y(100.3))
    assert cm.draw_clues(chart, [_clue(5, "gap", 100.3)]) == 1
    assert cm.drawn_clue_count(chart) == 1
    assert len(layer.labels) == 3  # pooled, hidden, never rebuilt


def test_draw_clues_matches_naive_chart_bars(qapp):
    from ui.widgets import clue_marker as cm
    from ui.widgets.candle_chart import CandleChart

    widget = CandleChart()
    widget.set_data(_bars(naive=True), timeframe="m5")
    stamp = cm.aware_bar_time(_bars(naive=True)[7]["dt"])
    assert cm.draw_clues(widget, [dict(_clue(7, "news", 100.4), bar_time=stamp.isoformat())]) == 1
    widget.deleteLater()


def test_clues_for_reads_one_symbol(tmp_path):
    import recap_store
    from ui.widgets.clue_marker import clues_for

    target = tmp_path / "recap.jsonl"
    stamp = FIRST + timedelta(minutes=30)
    for symbol in ("AMD", "NVDA", "amd"):
        recap_store.record_clue(
            session_date="2026-09-22", symbol=symbol, timeframe="M5", bar_time=stamp,
            price=10.0, clue_tag="gap", path=target,
        )
    assert [row["symbol"] for row in clues_for("2026-09-22", "amd", path=target)] == ["AMD", "AMD"]
    assert clues_for("2026-09-23", "AMD", path=target) == []


# ---------------------------------------------------------------------------
# the whole flow
# ---------------------------------------------------------------------------
def test_mark_clue_flow_marks_saves_with_context_and_draws(qapp, chart):
    from ui.widgets.clue_marker import drawn_clue_count, mark_clue_flow

    written: list[dict] = []
    threads: list[threading.Thread] = []

    def fake_writer(**fields):
        threads.append(threading.current_thread())
        written.append(fields)
        return {"id": "rc-9", "kind": "clue", **fields, "bar_time": fields["bar_time"].isoformat()}

    loaded_on: list[threading.Thread] = []

    def fake_loader(session, symbol):
        loaded_on.append(threading.current_thread())
        return [_clue(4, "gap", 100.1)]

    pool = QThreadPool()
    spy = _bars(base=500.0)
    flow = mark_clue_flow(
        chart, "2026-09-22", "amd", "M5", trade_id="t-7",
        spy_bars=spy, writer=fake_writer, loader=fake_loader, pool=pool, clock=lambda: NOW,
    )
    flow.load()
    _wait(qapp, pool, lambda: drawn_clue_count(chart) == 1)
    assert drawn_clue_count(chart) == 1
    assert loaded_on and loaded_on[0] is not threading.main_thread()

    flow.set_active(True)
    _click(qapp, chart, 12.0, 100.7)
    assert flow.form.isVisible()
    flow.form.chips["rs_vs_spy"].click()
    flow.form.save_button.click()
    _wait(qapp, pool, lambda: drawn_clue_count(chart) == 2)

    assert threads and threads[0] is not threading.main_thread()
    fields = written[0]
    assert fields["symbol"] == "AMD" and fields["trade_id"] == "t-7"
    assert fields["bar_time"] == _bars()[12]["dt"]
    context = fields["context"]
    assert context["bar"]["close"] == pytest.approx(_bars()[12]["close"])
    assert context["bar"]["volume"] == pytest.approx(_bars()[12]["volume"])
    assert context["spy_close"] == pytest.approx(spy[12]["close"])
    assert drawn_clue_count(chart) == 2

    flow.set_active(False)
    assert not flow.form.isVisible()


def test_missing_spy_is_unknown_not_guessed():
    from ui.widgets.clue_marker import clue_context

    snap = clue_context(_bars()[3], spy_bars=[])
    assert snap["spy_close"] is None
    assert snap["bar"]["high"] == pytest.approx(_bars()[3]["high"])
    assert clue_context(None) == {"bar": None, "spy_close": None}
