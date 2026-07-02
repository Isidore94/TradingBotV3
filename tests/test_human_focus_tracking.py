import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_human_focus_durable_bar_filename_avoids_windows_reserved_device_names():
    from human_focus_tracking import _sanitize_symbol_for_filename

    assert _sanitize_symbol_for_filename("CON") == "CON_"
    assert _sanitize_symbol_for_filename("CON.A") == "CON_.A"
    assert _sanitize_symbol_for_filename("LPT1") == "LPT1_"
    assert _sanitize_symbol_for_filename("BRK.B") == "BRK.B"


def test_snapshot_human_focus_picks_is_idempotent_and_force_merges(tmp_path):
    from human_focus_tracking import snapshot_human_focus_picks

    state_path = tmp_path / "human_focus_snapshot_state.json"
    picks_path = tmp_path / "human_focus_daily_picks.csv"
    first = snapshot_human_focus_picks(
        market_date="2026-06-01",
        focus_map={"long": {"NVDA", "AAPL"}, "short": {"TSLA"}},
        now=datetime(2026, 6, 1, 9, 35),
        snapshot_state_path=state_path,
        daily_picks_path=picks_path,
    )
    second = snapshot_human_focus_picks(
        market_date="2026-06-01",
        focus_map={"long": {"NVDA", "AAPL"}, "short": {"TSLA"}},
        snapshot_state_path=state_path,
        daily_picks_path=picks_path,
    )
    forced = snapshot_human_focus_picks(
        market_date="2026-06-01",
        focus_map={"long": {"NVDA", "AAPL", "MSFT"}, "short": {"TSLA"}},
        force=True,
        snapshot_state_path=state_path,
        daily_picks_path=picks_path,
    )

    rows = _read_csv(picks_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert first["added"] == 3
    assert second["snapshotted"] is False
    assert forced["added"] == 1
    assert {(row["symbol"], row["side"]) for row in rows} == {
        ("AAPL", "LONG"),
        ("MSFT", "LONG"),
        ("NVDA", "LONG"),
        ("TSLA", "SHORT"),
    }
    assert state["last_snapshot_market_date"] == "2026-06-01"


def test_human_focus_outcomes_are_side_adjusted_and_aggregated(tmp_path):
    from human_focus_tracking import snapshot_human_focus_picks, update_human_focus_outcomes

    picks_path = tmp_path / "human_focus_daily_picks.csv"
    outcomes_path = tmp_path / "human_focus_outcomes.csv"
    performance_path = tmp_path / "human_focus_performance.csv"
    snapshot_human_focus_picks(
        market_date="2026-06-01",
        focus_map={"long": {"AAPL"}, "short": {"TSLA"}},
        daily_picks_path=picks_path,
        snapshot_state_path=tmp_path / "state.json",
    )
    dates = pd.date_range("2026-06-01", periods=11, freq="B")
    frames = {
        "AAPL": pd.DataFrame({"datetime": dates, "close": [100, 102, 103, 105, 106, 110, 111, 112, 113, 114, 120]}),
        "TSLA": pd.DataFrame({"datetime": dates, "close": [100, 98, 97, 95, 94, 90, 89, 88, 87, 86, 80]}),
    }

    result = update_human_focus_outcomes(
        reference_date="2026-06-16",
        daily_frames_by_symbol=frames,
        daily_picks_path=picks_path,
        outcomes_path=outcomes_path,
        performance_path=performance_path,
        daily_bars_dir=tmp_path / "daily_bars",
    )

    outcomes = {row["symbol"]: row for row in _read_csv(outcomes_path)}
    performance = _read_csv(performance_path)
    all_h10 = next(
        row
        for row in performance
        if row["side"] == "ALL" and row["horizon_sessions"] == "10"
    )

    assert result["updated_outcomes"] == 2
    assert outcomes["AAPL"]["h1_return"] == "0.020000"
    assert outcomes["TSLA"]["h1_return"] == "0.020000"
    assert outcomes["AAPL"]["h10_return"] == "0.200000"
    assert outcomes["TSLA"]["h10_return"] == "0.200000"
    assert all_h10["sample_count"] == "2"
    assert all_h10["win_rate"] == "1.0000"
    assert all_h10["avg_side_return"] == "0.200000"


def test_matured_outcomes_are_preserved_without_recompute(tmp_path):
    from human_focus_tracking import snapshot_human_focus_picks, update_human_focus_outcomes

    picks_path = tmp_path / "human_focus_daily_picks.csv"
    outcomes_path = tmp_path / "human_focus_outcomes.csv"
    performance_path = tmp_path / "human_focus_performance.csv"
    snapshot_human_focus_picks(
        market_date="2026-06-01",
        focus_map={"long": {"AAPL"}, "short": set()},
        daily_picks_path=picks_path,
        snapshot_state_path=tmp_path / "state.json",
    )
    dates = pd.date_range("2026-06-01", periods=11, freq="B")
    frames = {"AAPL": pd.DataFrame({"datetime": dates, "close": list(range(100, 111))})}

    first = update_human_focus_outcomes(
        reference_date="2026-06-16",
        daily_frames_by_symbol=frames,
        daily_picks_path=picks_path,
        outcomes_path=outcomes_path,
        performance_path=performance_path,
        daily_bars_dir=tmp_path / "daily_bars",
    )
    assert first["updated_outcomes"] == 1
    matured_row = _read_csv(outcomes_path)[0]
    assert matured_row["fully_matured"] == "1"

    # A matured pick must be kept as-is even when no bars are available on a later run.
    second = update_human_focus_outcomes(
        reference_date="2026-06-17",
        daily_frames_by_symbol={},
        daily_picks_path=picks_path,
        outcomes_path=outcomes_path,
        performance_path=performance_path,
        daily_bars_dir=tmp_path / "missing_bars",
    )
    assert second["updated_outcomes"] == 0
    preserved = _read_csv(outcomes_path)
    assert len(preserved) == 1
    assert preserved[0]["h10_return"] == matured_row["h10_return"]


def test_mark_human_focus_rows_sets_priority_and_feature_flags():
    from human_focus_tracking import mark_human_focus_rows

    rows = [
        {"symbol": "AAPL", "side": "LONG"},
        {"symbol": "TSLA", "side": "SHORT"},
        {"symbol": "MSFT", "side": "LONG"},
    ]
    feature_rows = {"AAPL": {}, "TSLA": {}, "MSFT": {}}

    marked = mark_human_focus_rows(
        rows,
        feature_rows,
        focus_map={"long": {"AAPL"}, "short": {"TSLA"}},
    )

    assert marked == 2
    assert rows[0]["human_focus_pick"] is True
    assert rows[0]["human_focus_side"] == "LONG"
    assert rows[1]["human_focus_pick"] is True
    assert rows[2]["human_focus_pick"] is False
    assert feature_rows["AAPL"]["human_focus_pick"] is True
    assert feature_rows["TSLA"]["human_focus_side"] == "SHORT"
