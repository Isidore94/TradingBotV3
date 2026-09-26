"""B6 (goal 9, alert signal to noise): the Best-right-now log and its grader,
the alert noise report and the Day Review "Alerts:" truth line. Measurement
only: nothing here changes which alerts show, sound or rank."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

LA = ZoneInfo("America/Los_Angeles")
DAY = "2026-09-24"


def _alert(symbol, side="LONG", *, grade="B", entry=10.0):
    return {"symbol": symbol, "side": side, "grade": grade, "r": 0.5, "status": "open",
            "entry": entry, "stop": entry - 0.5, "received_at": f"{DAY} 09:40"}


def _bars(start: datetime, closes: list[float]) -> list[dict]:
    return [
        {"dt": start + timedelta(minutes=5 * i), "open": c, "high": c, "low": c, "close": c, "volume": 1}
        for i, c in enumerate(closes)
    ]


# --- the log -----------------------------------------------------------------


def test_project_paths_has_the_log_beside_the_movers_log():
    import project_paths

    assert project_paths.BEST_NOW_LOG_FILE.parent == project_paths.MOVERS_DIP_OUTCOMES_FILE.parent
    assert project_paths.BEST_NOW_LOG_FILE.name == "best_now_log.jsonl"


def test_first_appearance_is_logged_once_per_symbol_side_and_day(tmp_path):
    import best_now
    import best_now_outcomes

    path = tmp_path / "best_now_log.jsonl"
    clock = lambda: datetime(2026, 9, 24, 7, 0, 25, tzinfo=LA)  # noqa: E731
    log = best_now_outcomes.BestNowLog(path, clock=clock, trade_date=lambda: DAY)
    results = [_alert("AAA", grade="A", entry=12.5), _alert("BBB", side="SHORT", grade="C")]
    entries = best_now.rank_best_now(results)
    assert len(log.record(entries, results)) == 2
    assert log.record(entries, results) == []
    rows = best_now_outcomes.load_records(path)
    assert [(r["symbol"], r["side"], r["grade"], r["rank"]) for r in rows] == [
        ("AAA", "LONG", "A", 1),
        ("BBB", "SHORT", "C", 2),
    ]
    assert rows[0]["kind"] == "best_now" and rows[0]["trade_date"] == DAY
    assert rows[0]["entry_price_ref"] == 12.5
    assert datetime.fromisoformat(rows[0]["ts"]).tzinfo is not None
    # A restart reads today's keys back: nothing is logged twice.
    again = best_now_outcomes.BestNowLog(path, clock=clock, trade_date=lambda: DAY)
    assert again.record(entries, results) == []


def test_a_failed_log_write_never_raises(tmp_path):
    import best_now
    import best_now_outcomes

    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    log = best_now_outcomes.BestNowLog(blocker / "sub" / "log.jsonl", trade_date=lambda: DAY)
    assert log.record(best_now.rank_best_now([_alert("AAA")]), [_alert("AAA")]) == []


def test_the_strip_logs_on_its_worker_thread_never_the_qt_thread(tmp_path):
    import threading

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from ui.widgets.best_now_strip import BestNowStrip

    seen: list[tuple[bool, list]] = []

    class _Log:
        def record(self, entries, results):
            seen.append((threading.current_thread() is threading.main_thread(), list(entries)))
            return []

    strip = BestNowStrip(threaded=True, log=_Log())
    strip.set_results_provider(lambda: [_alert("AAA")])
    strip.refresh()
    deadline = datetime.now().timestamp() + 5
    while not seen and datetime.now().timestamp() < deadline:
        app.processEvents()
    assert [(on_main, [e.symbol for e in entries]) for on_main, entries in seen] == [(False, ["AAA"])]
    strip.deleteLater()


def test_the_post_close_bar_download_includes_logged_names(tmp_path, monkeypatch):
    import best_now_outcomes
    import day_review_bars
    import project_paths

    path = tmp_path / "best_now_log.jsonl"
    path.write_text(json.dumps({"kind": "best_now", "trade_date": DAY, "symbol": "ZZZQ",
                                "side": "LONG"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(project_paths, "BEST_NOW_LOG_FILE", path)
    assert best_now_outcomes.logged_symbols(DAY) == {"ZZZQ"}
    assert "ZZZQ" in day_review_bars.decided_symbols(DAY, sources=None)


# --- the grader ----------------------------------------------------------------


def _tape():
    start = datetime(2026, 9, 24, 6, 30, tzinfo=LA)
    # AAA +1% per bar; SPY flat; BBB (short) falls 1% per bar; CCC lags SPY.
    return {
        "SPY": _bars(start, [100.0] * 20),
        "AAA": _bars(start, [10.0 * (1 + 0.01 * i) for i in range(20)]),
        "BBB": _bars(start, [10.0 * (1 - 0.01 * i) for i in range(20)]),
        "CCC": _bars(start, [10.0 * (1 - 0.01 * i) for i in range(20)]),
    }


def _row(symbol, side, ts):
    return {"kind": "best_now", "ts": ts, "trade_date": DAY, "symbol": symbol, "side": side,
            "grade": "B", "rank": 1, "entry_price_ref": 10.0}


def test_the_grader_measures_vs_spy_from_the_first_appearance_bar_close():
    import best_now_outcomes

    ts = "2026-09-24T06:40:25-07:00"  # the 06:35 bar has closed; base = its close
    rows = [_row("AAA", "LONG", ts), _row("BBB", "SHORT", ts), _row("CCC", "LONG", ts)]
    summary = best_now_outcomes.summarize(rows, bars_reader=lambda _s: _tape())
    graded = {r["symbol"]: r for r in summary["rows"]}
    assert graded["AAA"]["base_bar"].startswith("2026-09-24 06:35")
    base = 10.0 * 1.01
    assert graded["AAA"]["ret30_pct"] == pytest.approx((10.0 * 1.07 / base - 1) * 100)
    assert graded["AAA"]["excess30_pct"] > 0 and graded["BBB"]["excess30_pct"] > 0
    assert graded["CCC"]["excess30_pct"] < 0
    assert summary["hit_rate"] == pytest.approx(2 / 3) and summary["graded"] == 3
    assert summary["by_horizon"]["60"]["short"]["hit_rate"] == 1.0


def test_no_stored_tape_is_pending_and_a_missing_bar_is_unmeasured():
    import best_now_outcomes

    late = _row("AAA", "LONG", "2026-09-24T07:50:25-07:00")  # +60 is past the tape
    summary = best_now_outcomes.summarize([late], bars_reader=lambda _s: _tape())
    assert summary["by_horizon"]["15"]["graded"] == 1
    assert summary["by_horizon"]["60"]["graded"] == 0
    pending = best_now_outcomes.summarize([late], bars_reader=lambda _s: None)
    assert pending["pending"] == 1 and pending["hit_rate"] is None


def test_the_cli_says_no_data_yet_on_an_empty_log(tmp_path, capsys):
    import best_now_outcomes

    assert best_now_outcomes.main(["--summary", "--path", str(tmp_path / "missing.jsonl")]) == 0
    assert capsys.readouterr().out.strip() == "no data yet"
