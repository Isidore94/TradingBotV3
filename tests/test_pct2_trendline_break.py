"""PCT-2: a trendline break is a tag, a frozen D1 event, and one feed row.

The fixture beside this file is the exact saved D1 alert list emitted by
e183db91 for the frozen scan candidate below.  It was captured before this
packet changes a line of production code.  PCT-2 is additive: that old set
must remain byte for byte identical, with exactly one ``Trendline break`` row
added for the frozen scan candidate.

The runner assertion is deliberately in a child process.  A real Master AVWAP
scan writes many outputs; the child sets ``TRADINGBOTV3_DATA_DIR`` before any
``scripts`` import so this test never writes the session-wide pytest home or a
live store.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURES_DIR = Path(__file__).with_name("fixtures")
GOLDEN_D1_ALERTS = json.loads(
    (FIXTURES_DIR / "pct2_prechange_d1_upgrade_alerts.json").read_text(encoding="utf-8")
)

ARMED_AT = datetime(2026, 9, 10, 10, 0)
FROZEN_LONG_LINE = {
    "type": "H-break",
    "start_date": "2026-09-02",
    "end_date": "2026-09-08",
    "lookback_end": "2026-09-10",
    "current_line_price": 100.0,
    "slope_log_per_bar": 0.0,
    "touch_count": 2,
    "break_date": "",
}
FROZEN_SHORT_LINE = {**FROZEN_LONG_LINE, "type": "L-break"}


def _daily(day: int, *, close: float, high: float | None = None, low: float | None = None) -> dict:
    return {
        "dt": datetime(2026, 9, day),
        "open": close,
        "high": float(high if high is not None else close),
        "low": float(low if low is not None else close),
        "close": float(close),
        "volume": 1_000.0,
    }


def _frozen_watch(*, side: str = "LONG", line: dict | None = None):
    """The persisted scan line is the arm-time evidence, never a live redraw."""
    from chart_watch import D1EventWatch

    return D1EventWatch(
        symbol="TLBR",
        kind="trendline_break",
        armed_at=ARMED_AT,
        side=side,
        trendline_candidate=dict(line or (FROZEN_LONG_LINE if side == "LONG" else FROZEN_SHORT_LINE)),
        trendline_knowledge_at=ARMED_AT,
    )


def test_a_trendline_break_tag_is_present_only_on_the_scan_flag():
    """Item 1 was already built; keep the existing confirmation pinned."""
    from master_avwap_lib.setup_tagging import derive_setup_tag_payload

    tagged = derive_setup_tag_payload(
        {"symbol": "TLBR", "side": "LONG", "trendline_break_recent": True}
    )
    ordinary = derive_setup_tag_payload(
        {"symbol": "TLBR", "side": "LONG", "trendline_break_recent": False}
    )

    assert "TRENDLINE_BREAK" in tagged["setup_tags"]
    assert "TRENDLINE_BREAK" not in ordinary["setup_tags"]


def test_trendline_break_is_an_extension_and_focus_never_auto_arms_it():
    """An extension speaks only through a trader-armed D1 event watch."""
    from chart_watch import D1_EVENT_KINDS, D1_EXTENSION_KINDS, D1_PULLBACK_KINDS

    assert D1_EVENT_KINDS["trendline_break"] == "Trendline break"
    assert "trendline_break" in D1_EXTENSION_KINDS
    assert "trendline_break" not in D1_PULLBACK_KINDS


def test_a_completed_d1_close_crosses_the_frozen_line_only_in_its_setup_direction():
    """Close, not wick, is the evidence.  LONG and SHORT are mirrors."""
    from chart_watch import evaluate_d1_event_watch

    long_watch = _frozen_watch(side="LONG")
    long_hit = evaluate_d1_event_watch(
        long_watch,
        [],
        [_daily(11, close=99.0), _daily(12, close=101.0)],
        now=datetime(2026, 9, 15),
    )
    assert long_hit is not None and long_hit.resolved_side == "long"
    assert long_hit.bar_dt.date().isoformat() == "2026-09-12"

    short_watch = _frozen_watch(side="SHORT")
    short_hit = evaluate_d1_event_watch(
        short_watch,
        [],
        [_daily(11, close=101.0), _daily(12, close=99.0)],
        now=datetime(2026, 9, 15),
    )
    assert short_hit is not None and short_hit.resolved_side == "short"

    # A wick beyond the long line but a close still below it is not a break.
    assert evaluate_d1_event_watch(
        long_watch,
        [],
        [_daily(11, close=99.0), _daily(12, close=99.0, high=105.0, low=98.0)],
        now=datetime(2026, 9, 15),
    ) is None
    # The current session's daily bar is forming at this poll and must wait.
    assert evaluate_d1_event_watch(
        long_watch,
        [],
        [_daily(11, close=99.0), _daily(15, close=101.0)],
        now=datetime(2026, 9, 15, 12, 0),
    ) is None
    # Once the prior completed close is already through, holding through again
    # is not another crossing.
    assert evaluate_d1_event_watch(
        long_watch,
        [],
        [_daily(11, close=101.0), _daily(12, close=102.0)],
        now=datetime(2026, 9, 15),
    ) is None


def test_a_redraw_cannot_replace_the_armed_line_and_a_break_fires_once(tmp_path, monkeypatch):
    """The panel consumes one frozen watch; no redraw can move its 100 line."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda self, symbol, **kwargs: None)
    panel = AlertCenterPanel(d1_event_watches_path=tmp_path / "d1_event_watches.json")
    panel._d1_event_watches = [_frozen_watch()]
    # This is the redrawn scan line.  The price action crosses the ARMED 100,
    # but not this later 120 line.  A correct poll never consults it.
    panel._current_trendline_candidate = lambda _symbol: {
        **FROZEN_LONG_LINE,
        "current_line_price": 120.0,
        "lookback_end": "2026-09-10",
    }
    panel._m5_bars_for = lambda _symbol: []
    panel._d1_bars_for = lambda _symbol: [_daily(11, close=99.0), _daily(12, close=101.0)]

    panel._poll_d1_event_watches(now=datetime(2026, 9, 15))
    first = [alert for alert in panel._alerts if alert.tag == "chart_watch"]
    assert len(first) == 1
    assert panel._d1_event_watches == []

    panel._poll_d1_event_watches(now=datetime(2026, 9, 15) + timedelta(minutes=1))
    assert len([alert for alert in panel._alerts if alert.tag == "chart_watch"]) == 1


def test_an_old_trendline_watch_loads_but_never_confirms_without_its_frozen_line():
    """Missing arm-time evidence is uncertainty, never a reconstructed break."""
    from chart_watch import d1_event_watch_from_dict, evaluate_d1_event_watch

    old = d1_event_watch_from_dict(
        {
            "symbol": "TLBR",
            "kind": "trendline_break",
            "armed_at": ARMED_AT.isoformat(),
            # A real old JSON row has the keys that existed then, and no
            # invented frozen candidate.
        }
    )
    assert old is not None
    assert evaluate_d1_event_watch(
        old,
        [],
        [_daily(11, close=99.0), _daily(12, close=101.0)],
        now=datetime(2026, 9, 15),
    ) is None


RUNNER_CHILD = r'''
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

scratch = Path(sys.argv[1])
scripts = sys.argv[2]
os.environ["TRADINGBOTV3_DATA_DIR"] = str(scratch / "home")
os.environ["LOCALAPPDATA"] = str(scratch / "localappdata")
os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = str(scratch / "diagnostics")
os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, scripts)

import pandas as pd
import project_paths
if "TradingBotData" in str(project_paths.DATA_DIR):
    raise SystemExit(f"unsafe data directory: {project_paths.DATA_DIR}")
from master_avwap_lib import runner

symbol = "TLBR"
end = date.today() - timedelta(days=1)
while end.weekday() >= 5:
    end -= timedelta(days=1)
stamps = list(pd.bdate_range(end=pd.Timestamp(end), periods=120))
rows = []
for index, stamp in enumerate(stamps):
    if index < 90:
        base = 80.0 + index * 0.25
        rows.append((stamp, base, base + 1.2, base - 1.2, base + 0.3))
    else:
        base = 103.0
        wiggle = 0.15 * ((index % 3) - 1)
        rows.append((stamp, base + wiggle, base + 0.30, base - 0.30, base + wiggle))
frame = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
frame["volume"] = 1_000_000
anchor = stamps[90].date().isoformat()

runner.resolve_master_scan_watchlist_paths = lambda **kwargs: ([scratch / "longs.txt"], [scratch / "shorts.txt"], "PCT-2")
runner.load_tickers_from_paths = lambda paths, **kwargs: ([symbol] if "longs" in str(list(paths)[0]) else [])
runner.append_master_avwap_d1_watchlist_symbols = lambda longs, shorts: (longs, shorts, 0)
runner.load_theta_long_symbols = lambda: []
runner.load_scan_earnings_context = lambda symbols: ({symbol: [anchor]}, {symbol: {}})
runner.collect_upcoming_earnings_dates = lambda symbols: {}
runner._maybe_run_setup_tracker_catchup = lambda **kwargs: {}
runner.connect_daily_data_client = lambda **kwargs: None
runner.build_market_regime_snapshot = lambda ib, day: {"label": "mixed"}
runner.fetch_daily_bars = lambda ib, sym, days, **kwargs: frame.copy()
runner.record_theta_picks = lambda *args, **kwargs: None

real_refine = runner.refine_priority_rows_with_directional_filters
def inject_frozen_break(rows, state, ib, **kwargs):
    real_refine(rows, state, ib, **kwargs)
    row = next(row for row in rows if row.get("symbol") == symbol)
    candidate = {
        "type": "H-break", "start_date": "2026-09-02", "end_date": "2026-09-08",
        "lookback_end": "2026-09-10", "current_line_price": 100.0,
        "slope_log_per_bar": 0.0, "touch_count": 2, "break_date": "2026-09-11",
    }
    row.update(trendline_break_recent=True, trendline_break_note="H-break test", trendline_break_candidate=candidate)
    state["symbols"][symbol].update(
        priority_trendline_break_recent=True,
        priority_trendline_break_note="H-break test",
        priority_trendline_break_candidate=candidate,
    )
runner.refine_priority_rows_with_directional_filters = inject_frozen_break

result = runner._run_master_impl(update_setup_tracker=False)
saved = json.loads(Path(project_paths.MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE).read_text(encoding="utf-8"))
row = next(row for row in result["priority_rows"] if row.get("symbol") == symbol)
print("PCT2-RUNNER::" + json.dumps({"row": row, "alerts": saved.get("alerts", [])}, default=str))
'''


def test_the_real_runner_path_adds_one_trendline_feed_row_in_its_own_data_dir(tmp_path):
    """A scan-row field must survive the live runner seam, not a hand dict.

    Golden-first: every pre-change saved D1 row remains byte-identical and the
    one new row uses the break date as its identity.
    """
    scratch = tmp_path / "runner"
    scratch.mkdir()
    child = scratch / "pct2_runner.py"
    child.write_text(RUNNER_CHILD, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(child), str(scratch), str(SCRIPTS_DIR)],
        cwd=str(SCRIPTS_DIR),
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:] + completed.stdout[-4000:]
    line = next((line for line in completed.stdout.splitlines() if line.startswith("PCT2-RUNNER::")), "")
    assert line, completed.stdout[-4000:]
    payload = json.loads(line.split("::", 1)[1])
    assert "TRENDLINE_BREAK" in payload["row"]["setup_tags"]
    matches = [row for row in payload["alerts"] if row.get("event_type") == "trendline_break"]
    champions = [row for row in payload["alerts"] if row.get("event_type") != "trendline_break"]
    assert champions == GOLDEN_D1_ALERTS
    assert len(matches) == 1
    assert matches[0]["label"] == "Trendline break"
    assert matches[0]["break_date"] == "2026-09-11"
