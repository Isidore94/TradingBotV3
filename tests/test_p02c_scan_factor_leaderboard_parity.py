"""P0-2 2c: the scan-factor leaderboard is faster and its output is unchanged.

The golden `p02c_scan_factor_leaderboard_v1` was frozen from the code BEFORE the
speed change (a793f661). Its raw input is a synthetic feature history; the
expected section is every leaderboard row except `generated_at` (a wall clock).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402

GOLDEN = "p02c_scan_factor_leaderboard_v1"


def synthetic_history_records(seed: int = 20260925) -> list[dict]:
    """A deterministic feature history: ties in dates and returns, both sides, reruns."""
    rng = random.Random(seed)
    sessions = pd.bdate_range("2026-06-01", periods=40)
    symbols = [f"S{index:02d}" for index in range(11)] + ["SPY"]
    rows = []
    prices = {symbol: 20.0 + 7.0 * index for index, symbol in enumerate(symbols)}
    for day_index, day in enumerate(sessions):
        runs = 2 if day_index % 5 == 0 else 1
        for run in range(runs):
            run_id = f"r{day_index:03d}{run}"
            for symbol in symbols:
                if symbol != "SPY" and rng.random() < 0.15:
                    continue
                move = rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
                prices[symbol] = max(1.0, round(prices[symbol] * (1 + move / 100.0), 2))
                rows.append(
                    {
                        "symbol": symbol,
                        "side": "SHORT" if symbol in {"S03", "S07", "S11"} else "LONG",
                        "last_trade_date": day.strftime("%Y-%m-%d"),
                        "run_date": day.strftime("%Y-%m-%d"),
                        "run_id": run_id,
                        "run_timestamp": f"{day.strftime('%Y-%m-%d')}T1{run}:00:00",
                        "watchlist_label": "fixture",
                        "last_close": prices[symbol],
                        "market_regime_label": rng.choice(["risk_on", "risk_off"]),
                        "current_band_zone": rng.choice(["upper_1", "vwap", "lower_1"]),
                        "setup_family": rng.choice(["band_bounce", "mid_earnings", ""]),
                        "priority_bucket": rng.choice(
                            ["favorite_setup", "near_favorite_zone", "watch"]
                        ),
                        "spy_above_sma20": rng.choice([True, False, None]),
                        "above_sma200": rng.choice(["true", "false", ""]),
                        "compression_flag": rng.choice([1, 0]),
                        "setup_tags": rng.choice(["a;b", "b", "", "['a', 'c']"]),
                        "priority_score": rng.choice([70, 90, 110, 130, 150]),
                        "relvol": rng.choice([0.4, 0.9, 1.2, 2.0, None]),
                        "dist_sma50_atr": rng.choice([-4.0, -0.5, 0.5, 2.0]),
                    }
                )
    return rows


def leaderboard_without_clock(rows: list[dict]) -> list[dict]:
    cleaned = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "generated_at"}
        cleaned.append(item)
    return json.loads(json.dumps(cleaned, sort_keys=True, default=str))


def _records() -> list[dict]:
    return json.loads(json.dumps(synthetic_history_records(), default=str))


def test_fixture_input_is_the_generator_output():
    from conftest import load_fixture_contract

    generator = load_fixture_contract(GOLDEN)["history_generator"]
    records = _records()
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert generator["record_count"] == len(records)
    assert generator["records_sha256"] == digest


def test_leaderboard_equals_the_pre_change_golden():
    from conftest import load_fixture_contract

    contract = load_fixture_contract(GOLDEN)
    history_df = pd.DataFrame(_records())
    observations = legacy.build_scan_factor_observation_rows(history_df)
    rows = legacy.build_scan_factor_leaderboard_rows(history_df, observations)
    actual = leaderboard_without_clock(rows)
    expected = contract["expected_leaderboard"]
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected, strict=False)):
        contract.assert_matches(got, want, context=f"row {index}")
