"""P1-5 5b: the "Best right now" ranker and strip. Display and ranking only."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import best_now  # noqa: E402


def _alert(symbol, side="LONG", *, grade="B", r=0.5, status="open", entry=10.0, stop=9.5, at="09:40"):
    return {
        "symbol": symbol,
        "side": side,
        "grade": grade,
        "r": r,
        "status": status,
        "entry": entry,
        "stop": stop,
        "received_at": f"2026-09-24 {at}",
    }


def _dip(symbol, score, *, last=20.0, lod=19.2):
    return {"symbol": symbol, "dip_score": score, "last": last, "lod": lod, "since_start_pct": 0.8}


# --- the pure ranker --------------------------------------------------------


def test_d1_with_m5_outranks_m5_alone_which_outranks_dip_strong():
    entries = best_now.rank_best_now(
        [_alert("AAA", grade="A", r=1.2), _alert("BBB", grade="C", r=0.1)],
        [_dip("CCC", 3.0)],
        {("BBB", "LONG"): {"grade": "B", "family": "retest", "claimed": False}},
    )
    assert [e.symbol for e in entries] == ["BBB", "AAA", "CCC"]
    assert [e.tier for e in entries] == [best_now.TIER_D1_M5, best_now.TIER_M5, best_now.TIER_DIP]
    assert entries[0].why.startswith("D1 B + M5 C")
    assert (entries[0].entry, entries[0].stop) == (10.0, 9.5)
    assert (entries[2].entry, entries[2].stop) == (20.0, 19.2)
    assert entries[2].why.startswith("Dip-strong +3.0")


def test_a_d1_name_without_an_m5_alert_is_not_listed():
    entries = best_now.rank_best_now([], [], {("ZZZ", "LONG"): {"grade": "A"}})
    assert entries == []


def test_inside_a_tier_grade_then_live_r_then_time():
    entries = best_now.rank_best_now(
        [
            _alert("LOWR", grade="A", r=0.2, at="09:35"),
            _alert("HIGHR", grade="A", r=1.5, at="09:50"),
            _alert("BGRADE", grade="B", r=3.0, at="09:31"),
            _alert("NODATA", grade="A", r=None, status="unknown", at="09:30"),
        ]
    )
    assert [e.symbol for e in entries] == ["HIGHR", "LOWR", "NODATA", "BGRADE"]


def test_a_stopped_alert_is_left_out_and_a_dip_name_with_an_alert_merges():
    entries = best_now.rank_best_now(
        [_alert("STOP", status="stopped", r=-1.0), _alert("BOTH", r=0.4)],
        [_dip("BOTH", 2.0), _dip("STOP", 1.0)],
    )
    assert [e.symbol for e in entries] == ["BOTH", "STOP"]
    assert entries[0].tier == best_now.TIER_M5 and "dip-strong" in entries[0].why
    assert entries[1].tier == best_now.TIER_DIP


def test_the_limit_and_the_row_text():
    entries = best_now.rank_best_now([_alert(f"S{i}") for i in range(9)], limit=6)
    assert len(entries) == 6
    text = best_now.entry_text(entries[0])
    assert text == "S0 L  M5 B +0.5R\n e 10.00 s 9.50"


def test_diff_rows_names_only_the_changed_rows():
    assert best_now.diff_rows(["a", "b", "c"], ["a", "b", "c"]) == []
    assert best_now.diff_rows(["a", "b", "c"], ["a", "x", "c"]) == [1]
    assert best_now.diff_rows(["a", "b"], ["a", "b", "c"]) == [2]
    assert best_now.diff_rows(["a", "b", "c"], ["a"]) == [1, 2]


def test_dip_strong_rows_reads_the_board_or_nothing():
    assert best_now.dip_strong_rows({"dip": {"long": [_dip("X", 1.0)], "short": []}})[0]["symbol"] == "X"
    assert best_now.dip_strong_rows({}) == []
    assert best_now.dip_strong_rows(None) == []


# --- the strip: diff, never rebuild ----------------------------------------


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _strip(results):
    from ui.widgets.best_now_strip import BestNowStrip

    strip = BestNowStrip(threaded=False)
    strip.set_results_provider(lambda: list(results))
    return strip


def test_an_unchanged_list_touches_no_row_and_builds_no_widget(app):
    results = [_alert("AAA", grade="A", r=1.0), _alert("BBB")]
    strip = _strip(results)
    widgets = strip.row_widgets()
    strip.refresh()
    assert strip.last_changed_rows == 2
    assert [text.split()[0] for text in strip.row_texts()] == ["AAA", "BBB"]
    calls = []
    for row in widgets:
        original = row.set_row
        row.set_row = lambda *a, _o=original, **k: (calls.append(a), _o(*a, **k))
    strip.refresh()
    assert strip.last_changed_rows == 0 and calls == []
    assert strip.row_widgets() == widgets  # the same label objects, never rebuilt


def test_one_changed_row_rewrites_only_that_row(app):
    results = [_alert("AAA", grade="A", r=1.0), _alert("BBB", r=0.5)]
    strip = _strip(results)
    strip.refresh()
    results[1] = _alert("BBB", r=0.8)
    strip.refresh()
    assert strip.last_changed_rows == 1
    assert "+0.8R" in strip.row_texts()[1]


def test_the_movers_board_feeds_the_strip_and_a_day_roll_empties_it(app):
    strip = _strip([])
    strip.set_movers_board({"dip": {"long": [_dip("DIPX", 2.0)]}})
    assert [e.symbol for e in strip.entries()] == ["DIPX"]
    strip.clear_day()
    assert strip.entries() == [] and strip.row_texts() == []


def test_the_strip_never_sets_a_stylesheet(app, monkeypatch):
    from PySide6.QtWidgets import QWidget

    seen = []
    monkeypatch.setattr(QWidget, "setStyleSheet", lambda self, sheet: seen.append(sheet))
    strip = _strip([_alert("AAA")])
    strip.refresh()
    strip.refresh()
    assert seen == []
    import ui

    qss = (Path(ui.__file__).parent / "theme.qss").read_text(encoding="utf-8")
    assert "QFrame#BestNowStrip" in qss and "QLabel#BestNowRow" in qss


def test_the_ranking_runs_off_the_qt_thread(app):
    import threading

    from ui.widgets import best_now_strip as module

    seen = []
    real = module.BestNowStrip._compute

    def spy(results, dips, context, limit):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(results, dips, context, limit)

    strip = module.BestNowStrip(threaded=True)
    strip._compute = spy
    strip.set_results_provider(lambda: [_alert("AAA")])
    strip.refresh()
    deadline = datetime.now().timestamp() + 5
    while not seen and datetime.now().timestamp() < deadline:
        app.processEvents()
    assert seen == [False]


def test_the_timer_waits_for_the_next_bar_boundary():
    from ui.widgets.best_now_strip import BAR_GRACE_SECONDS, ms_to_next_bar

    assert ms_to_next_bar(datetime(2026, 9, 24, 10, 3, 0)) == (120 + BAR_GRACE_SECONDS) * 1000
    assert ms_to_next_bar(datetime(2026, 9, 24, 10, 5, 0)) == (300 + BAR_GRACE_SECONDS) * 1000


# --- the desk: mounted under "Working now", fits both layouts --------------


@pytest.mark.parametrize("layout", ["compact", "classic"])
def test_the_desk_mounts_the_strip_under_working_now_and_it_fits(app, layout):
    from PySide6.QtCore import QObject, Signal

    from ui.panels.trading_desk import TradingDeskPanel
    from ui.widgets.best_now_strip import BestNowStrip

    desk = TradingDeskPanel(workspace_mode="workspace", layout_name=layout)
    try:
        strip = desk.best_now_strip
        assert isinstance(strip, BestNowStrip)
        bar_layout = desk.m5_alert_bar.layout()
        assert bar_layout.itemAt(1).widget() is desk.live_results_strip
        assert bar_layout.itemAt(2).widget() is strip

        # The Movers board reaches it through the desk's hosting seam.
        class _Movers(QObject):
            moversChanged = Signal(dict)

            def board(self):
                return {}

        movers = _Movers()
        desk.attach_movers_service(movers)
        strip._threaded = False
        movers.moversChanged.emit(
            {"dip": {"long": [{"symbol": "DIPLONGNAME", "dip_score": 2.5, "last": 123.45, "lod": 120.1}]}}
        )
        assert [e.symbol for e in strip.entries()] == ["DIPLONGNAME"]

        # It fits the M5 column: rows elide to the width, never widen the column.
        desk.resize(1640, 980)
        desk.show()
        for _ in range(20):
            app.processEvents()
        column = desk.m5_column
        assert strip.width() <= column.width()
        assert strip.minimumSizeHint().width() <= column.minimumWidth()
        row = strip.row_widgets()[0]
        assert row.isVisible() and row.width() <= strip.width()
        assert row.full_text().endswith("e 123.45 s 120.10")
    finally:
        desk.shutdown()
        desk.close()
