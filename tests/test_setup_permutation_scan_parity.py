r"""P1-4 / 4a - the live scan stamps the permutation key and changes nothing else.

The same synthetic symbol goes through the real `runner._run_master_impl` twice,
each in a child process with a scratch home (the PCT-3 idiom, see
`test_pct3_runner_publishes_compression.py`): once as shipped, once with the two
4a hooks replaced by no-ops. The detector/scoring output - the priority row,
the ai_state entry and every pre-existing `d1_features_history.csv` column -
must be identical; only the appended 4a columns may differ.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutations as sp  # noqa: E402

RESULT_MARKER = "P14-RESULT::"

CHILD_SOURCE = r'''
import csv
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

SCRATCH = Path(sys.argv[1])
SCRIPTS_DIR = sys.argv[2]
DISABLE = sys.argv[3] == "off"
SESSION_COUNT = int(sys.argv[4]) if len(sys.argv) > 4 else 260
TRACKER_SEED = sys.argv[5] if len(sys.argv) > 5 else ""
BARS_SHAPE = sys.argv[6] if len(sys.argv) > 6 else ""
SYMBOL = "PKEY"
os.environ["TRADINGBOTV3_DATA_DIR"] = str(SCRATCH / "home")
os.environ["LOCALAPPDATA"] = str(SCRATCH / "localappdata")
os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = str(SCRATCH / "diag")
os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, SCRIPTS_DIR)

import pandas as pd

import project_paths

if "TradingBotData" in str(project_paths.DATA_DIR):
    raise SystemExit(f"ABORT: live folder {project_paths.DATA_DIR}")

from master_avwap_lib import runner

if DISABLE:
    runner.stamp_permutation_scan_rows = lambda rows, **kwargs: 0
    runner.permutation_ma_distance_columns = lambda *args, **kwargs: {}
    # P11: the appended D1 history columns are switched off the same way.
    if hasattr(runner, "permutation_d1_history_columns"):
        runner.permutation_d1_history_columns = lambda *args, **kwargs: {}
    # P8b: the setup-age column is switched off the same way.
    if hasattr(runner, "permutation_setup_age_columns"):
        runner.permutation_setup_age_columns = lambda *args, **kwargs: 0
    # S6: the trendline columns are switched off the same way.
    if hasattr(runner, "permutation_trendline_columns"):
        runner.permutation_trendline_columns = lambda *args, **kwargs: {}
    # S15: the liquidity, market-cap, sector-rank and earnings-gap columns are switched off the same way.
    if hasattr(runner, "permutation_liquidity_columns"):
        runner.permutation_liquidity_columns = lambda *args, **kwargs: {}
        runner.permutation_earnings_gap_columns = lambda *args, **kwargs: {}
        runner.permutation_sector_rank_columns = lambda *args, **kwargs: 0
        runner.load_permutation_market_caps = lambda *args, **kwargs: {}
    # S15 item 2: the regime columns are switched off the same way.
    if hasattr(runner, "permutation_regime_columns"):
        runner.permutation_regime_columns = lambda *args, **kwargs: 0
        runner.load_permutation_trader_regime = lambda *args, **kwargs: None


def _sessions(count):
    end = date.today() - timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    return list(pd.bdate_range(end=pd.Timestamp(end), periods=count))


stamps = _sessions(SESSION_COUNT)
rows = []
for index, stamp in enumerate(stamps):
    base = 60.0 + index * 0.2 + (1.5 if index % 7 == 0 else 0.0)
    # Longer runs (P11) widen the last 20 ranges so ATR14 has a 252-session range to sit in.
    wide = 1.5 if SESSION_COUNT > 260 and index >= SESSION_COUNT - 20 else 1.0
    if BARS_SHAPE == "trendline_break" and index >= 230:
        # S6: lower highs from session 230 on, then a two-session pop through that falling line.
        if index < SESSION_COUNT - 2:
            step = index - 230
            base = 106.0 - 0.15 * step + 2.0 * (1 - abs(step % 20 - 10) / 10)
        else:
            base = rows[SESSION_COUNT - 3][1] + (4.3 if index == SESSION_COUNT - 1 else 3.0)
        wide = 1.0
    rows.append((stamp, base, base + wide, base - 1.0, base + 0.2))
frame = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
frame["volume"] = 1_000_000
anchor_iso = stamps[230].date().isoformat()

if TRACKER_SEED:
    # P8b: a tracker scoring view whose one setup was first seen `first_seen_offset` sessions back.
    seed = json.loads(Path(TRACKER_SEED).read_text(encoding="utf-8"))
    first_seen = stamps[-1 - int(seed["first_seen_offset"])].date().isoformat()
    seeded = {"schema_version": 1, "setups": {"seed": {
        "setup_id": "seed", "symbol": seed["symbol"], "side": seed["side"], "scan_date": first_seen,
        "setup_family": seed["setup_family"], "setup_status": "OPEN"}}}
    runner.load_setup_tracker_scoring_payload = lambda *a, **k: seeded

runner.resolve_master_scan_watchlist_paths = lambda **kwargs: (
    [SCRATCH / "longs.txt"], [SCRATCH / "shorts.txt"], "test watchlist")
runner.load_tickers_from_paths = lambda paths, **kw: ([SYMBOL] if "longs" in str(list(paths)[0]) else [])
runner.append_master_avwap_d1_watchlist_symbols = lambda longs, shorts: (longs, shorts, 0)
runner.load_theta_long_symbols = lambda *a, **k: []
runner.load_scan_earnings_context = lambda symbols: ({SYMBOL: [anchor_iso]}, {SYMBOL: {}})
runner.collect_upcoming_earnings_dates = lambda symbols: {}
runner._maybe_run_setup_tracker_catchup = lambda **kwargs: {}
runner.connect_daily_data_client = lambda **kwargs: None
runner.build_market_regime_snapshot = lambda ib, day: {"label": "mixed"}
runner.fetch_daily_bars = lambda ib, sym, days, **kw: frame.copy()
runner.record_theta_picks = lambda *a, **k: None

result = runner._run_master_impl(update_setup_tracker=False)
priority = [row for row in (result.get("priority_rows") or []) if row.get("symbol") == SYMBOL]
ai_state = json.loads(Path(project_paths.MASTER_AVWAP_AI_STATE_FILE).read_text(encoding="utf-8"))
history = []
with Path(project_paths.D1_FEATURES_HISTORY_FILE).open("r", encoding="utf-8-sig", newline="") as handle:
    history = list(csv.DictReader(handle))
print("P14-RESULT::" + json.dumps({
    "priority_row": priority[0] if priority else None,
    "ai_state_entry": (ai_state.get("symbols") or {}).get(SYMBOL),
    "history": history,
}, default=str))
'''

#: Values that move between two runs of the same scan (clock, ids), not detector output.
VOLATILE_KEYS = {
    "run_id", "run_timestamp", "generated_at", "updated_at", "scanned_at", "computed_at",
    "daily_bar_status_checked_at", "as_of", "written_at",
}


def _run(tmp_path: Path, mode: str, sessions: int = 260, seed: Path | None = None, bars: str = "") -> dict:
    scratch = tmp_path / f"{mode}-{sessions}{'-seeded' if seed else ''}{'-' + bars if bars else ''}"
    for name in ("home", "localappdata", "diag"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    child = scratch / "child.py"
    child.write_text(CHILD_SOURCE, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(child), str(scratch), str(SCRIPTS_DIR), mode, str(sessions),
         *([str(seed) if seed else "", bars] if bars else [str(seed)] if seed else [])],
        capture_output=True, text=True, timeout=900, env=environment, cwd=str(SCRIPTS_DIR),
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    line = next(
        (text[len(RESULT_MARKER):] for text in reversed(completed.stdout.splitlines())
         if text.startswith(RESULT_MARKER)),
        "",
    )
    assert line, completed.stdout[-4000:]
    return json.loads(line)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("p14-parity")
    return _run(base, "on"), _run(base, "off")


def _strip(value):
    if isinstance(value, dict):
        return {
            key: _strip(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS and key not in sp.SCAN_ROW_COLUMNS and "timestamp" not in key
        }
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def test_the_scan_writes_the_permutation_columns_and_a_key(runs):
    stamped, _plain = runs
    assert stamped["history"], "the scan wrote no feature history"
    row = stamped["history"][-1]
    header = list(row)
    assert header[-len(sp.SCAN_ROW_COLUMNS):] == list(sp.SCAN_ROW_COLUMNS), "4a columns append last"
    assert row["permutation_rule_version"] == sp.PERMUTATION_RULE_VERSION
    assert row["permutation_key"].startswith(f"{sp.PERMUTATION_RULE_VERSION}|")
    assert row["perm_dist_ema21_atr"] not in ("", None)
    assert row["perm_dist_sma200_atr"] not in ("", None)
    # 260 sessions of bars is enough for every support MA, so ma_support is known.
    key = dict(part.split("=", 1) for part in row["permutation_key"].split("|")[3].split(";"))
    assert key["ma_support"] != sp.UNKNOWN
    assert key["ma_order"] != sp.UNKNOWN


def test_detector_and_scoring_output_are_identical_with_and_without_the_stamp(runs):
    stamped, plain = runs
    assert stamped["priority_row"] and plain["priority_row"]
    assert _strip(stamped["priority_row"]) == _strip(plain["priority_row"])
    assert _strip(stamped["ai_state_entry"]) == _strip(plain["ai_state_entry"])
    assert len(stamped["history"]) == len(plain["history"])
    for on_row, off_row in zip(stamped["history"], plain["history"], strict=False):
        assert _strip(on_row) == _strip(off_row)
    # With the hooks off the appended columns are present and blank: nothing else moved.
    assert all(plain["history"][-1][column] == "" for column in sp.SCAN_ROW_COLUMNS)
