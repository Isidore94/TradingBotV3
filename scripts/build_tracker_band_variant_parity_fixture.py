#!/usr/bin/env python3
"""Freeze the tracker-record parity fixture for Phase 0.10 B-2.

**Frozen BEFORE the shadow block exists**, which is the whole point (plan.md
section 5, and the prompt's rule 4). It records what
``build_tracker_setup_record`` + ``recompute_tracker_setup_record`` produce on
the champion's code today - every key, every scenario, the representative and
average R - so that after the band-variant shadow lands, the parity test can
prove every PRE-EXISTING key is byte-identical and only new keys appeared.

A fixture frozen after the edit would prove nothing at all.

The bars are hand-constructed for a synthetic symbol, on the same reasoning as
``build_mixed_unit_avwap_fixture.py``: no vendor revision can move them, and no
desk's store has to be present to re-freeze. They carry a run-up, a pullback
through the anchored bands and a recovery, so scenarios actually resolve -
this record closes six of its eight tradeable scenarios rather than sitting
open and proving nothing.

Run:  .venv/Scripts/python.exe scripts/build_tracker_band_variant_parity_fixture.py
It refuses to overwrite an existing fixture without --force.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "tracker_record_band_variant_parity_v1.json"

SYMBOL = "BNDQ"
ANCHOR_INDEX = 12
ANCHOR_DATE = "2026-07-31"
SCAN_DATE = "2026-08-11"
GENERATED_AT = "2026-08-11T13:20:00-07:00"
#: The scan sees bars through SCAN_DATE; the forward tracking replays the rest.
BARS_AT_SCAN = 20

# (date, open, high, low, close, volume). Bars 0-19 are the mixed-unit fixture's
# price path (a known-good shape, reused rather than re-invented); bars 20-26 are
# the forward tape the scenarios are graded on - a push up, a break down through
# the lower band, and a recovery.
BARS: list[tuple[str, float, float, float, float, float]] = [
    ("2026-07-15", 40.10, 40.86, 39.92, 40.62, 41_284_000.0),
    ("2026-07-16", 40.70, 41.15, 40.41, 41.02, 38_910_000.0),
    ("2026-07-17", 41.05, 41.44, 40.72, 40.88, 44_150_000.0),
    ("2026-07-20", 40.92, 41.60, 40.85, 41.51, 36_720_000.0),
    ("2026-07-21", 41.55, 42.08, 41.30, 41.94, 52_305_000.0),
    ("2026-07-22", 41.90, 42.22, 41.05, 41.28, 61_480_000.0),
    ("2026-07-23", 41.24, 41.66, 40.60, 40.79, 57_940_000.0),
    ("2026-07-24", 40.75, 41.02, 40.11, 40.35, 74_218_900.0),
    ("2026-07-27", 40.40, 41.28, 40.32, 41.19, 49_066_400.0),
    ("2026-07-28", 41.22, 41.95, 41.06, 41.83, 43_512_700.0),
    ("2026-07-29", 41.88, 42.61, 41.74, 42.44, 55_803_100.0),
    ("2026-07-30", 42.50, 43.02, 42.18, 42.31, 47_229_800.0),
    # --- anchor -------------------------------------------------------------
    ("2026-07-31", 42.35, 43.18, 42.20, 43.05, 51_640_200.0),
    ("2026-08-03", 43.10, 43.77, 42.88, 43.62, 45_118_600.0),
    ("2026-08-04", 43.66, 44.30, 43.44, 43.91, 39_874_500.0),
    ("2026-08-05", 43.95, 44.12, 43.02, 43.18, 58_206_300.0),
    ("2026-08-06", 43.20, 43.64, 42.55, 42.77, 62_913_400.0),
    ("2026-08-07", 42.80, 43.51, 42.66, 43.40, 41_007_900.0),
    ("2026-08-10", 43.44, 44.60, 43.38, 44.48, 68_552_100.0),
    # --- scan date; everything below is the forward tape --------------------
    ("2026-08-11", 44.52, 45.31, 44.17, 45.16, 71_336_800.0),
    ("2026-08-12", 45.20, 46.10, 44.90, 45.90, 60_000_000.0),
    ("2026-08-13", 45.95, 46.80, 45.10, 45.30, 55_000_000.0),
    ("2026-08-14", 45.30, 45.60, 43.80, 44.10, 58_000_000.0),
    ("2026-08-17", 44.10, 44.50, 42.90, 43.20, 62_000_000.0),
    ("2026-08-18", 43.20, 44.90, 43.00, 44.70, 57_000_000.0),
    ("2026-08-19", 44.75, 46.50, 44.60, 46.30, 66_000_000.0),
    ("2026-08-20", 46.40, 47.20, 45.90, 46.10, 59_000_000.0),
]

ROW = {
    "symbol": SYMBOL,
    "side": "LONG",
    "priority_bucket": "favorite_setup",
    "score": 72,
    "setup_family": "first_dev_bounce",
    "favorite_signals": ["BOUNCE_UPPER_1"],
    "context_signals": [],
    "bar_status": "completed",
    "view_mode": "STABLE",
}

SHORT_ROW = dict(ROW, side="SHORT", favorite_signals=["BOUNCE_LOWER_1"])

#: Keys read out of the outcome summary and asserted by name, because these are
#: the ones a shadow scenario would silently move if it were not fenced off.
SUMMARY_KEYS = (
    "representative_stop_label",
    "representative_total_r",
    "representative_closed_r",
    "tradeable_scenario_count",
    "open_tradeable_scenario_count",
    "closed_tradeable_scenario_count",
    "avg_total_r",
    "raw_avg_total_r",
    "median_total_r",
    "avg_closed_r",
    "raw_avg_closed_r",
    "open_distortion",
    "best_total_r",
    "worst_total_r",
    "any_target_hit",
    "any_stopped",
)


def _bar_dicts() -> list[dict]:
    return [
        {"date": day, "open": o, "high": h, "low": low, "close": c, "volume": v}
        for day, o, h, low, c, v in BARS
    ]


def _frame(bars: list[dict]):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(bar["date"]),
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }
            for bar in bars
        ]
    )


def _jsonable(value):
    """Plain JSON types, with NaN/Inf collapsed to None.

    A parity fixture is compared key by key, so every value has to survive a
    round trip through JSON unchanged. numpy scalars and Timestamps do not.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if value is None or isinstance(value, str):
        return value
    for caster in (float, int):
        try:
            return _jsonable(caster(value))
        except (TypeError, ValueError):
            continue
    return str(value)


def measure(bars: list[dict], row: dict) -> dict:
    """Build a tracker record, replay it forward, and return the JSON form.

    Importable so the parity test measures with exactly this recipe rather than
    a paraphrase of it - a parity test whose SETUP drifts is not a parity test.
    """
    from master_avwap_lib import legacy, runner

    frame = _frame(bars)
    scan_frame = frame.iloc[:BARS_AT_SCAN]
    vwap, stdev, bands = legacy.calc_anchored_vwap_bands(scan_frame, ANCHOR_INDEX)
    last = bars[BARS_AT_SCAN - 1]
    symbol_entry = {
        "side": row["side"],
        "last_close": last["close"],
        "last_trade_date": SCAN_DATE,
        "atr20": 0.95,
        "current_anchor": {
            "date": ANCHOR_DATE,
            "vwap": float(vwap),
            "stdev": float(stdev),
            "bands": {key: float(value) for key, value in bands.items()},
        },
        "previous_anchor": {},
        # The challenger's shadow blocks, computed the way the scanner computes
        # them (Phase 0.10 B-2 item 1). Absent from the frozen expectations by
        # construction: they did not exist when this fixture was frozen, and the
        # parity test only demands that PRE-EXISTING keys are unchanged. There
        # is no previous anchor on this fixture, so that block states its reason.
        "current_anchor_variant": runner.build_anchor_band_variant_meta(
            scan_frame, ANCHOR_INDEX, ANCHOR_DATE
        ),
        "previous_anchor_variant": runner.build_anchor_band_variant_meta(None, None, ""),
        "entry_feature_snapshot": {},
        "daily_ohlc": [],
    }
    setup = legacy.build_tracker_setup_record(
        dict(row), symbol_entry, {}, GENERATED_AT, None, scan_date=SCAN_DATE
    )
    setup = legacy.recompute_tracker_setup_record(copy.deepcopy(setup), frame)
    summary = legacy._summarize_tracker_setup_outcome(setup)
    return {
        "record": _jsonable(setup),
        "outcome_summary": {key: _jsonable(summary.get(key)) for key in SUMMARY_KEYS},
        "stop_candidate_labels": [
            _jsonable(scenario.get("stop_reference_label"))
            for scenario in setup.get("scenarios", {}).values()
        ],
    }


def build() -> dict:
    bars = _bar_dicts()
    payload = {
        # --- repo-wide fixture contract (plan.md Milestone 3) ----------------
        "schema": "tracker_record_band_variant_parity_v1",
        "feature_version": "tracker_setup_record_pre_band_variant",
        "raw_input_keys": ["bars", "rules_under_test"],
        "raw_input_sha256": "",
        "acquired_at": "2026-08-26T22:00:00-04:00",
        "universe_version": "synthetic_tracker_parity_v1",
        "provider_assumptions": (
            "No provider call. Twenty-seven hand-constructed daily bars for a synthetic "
            "symbol, so no vendor revision can move this fixture and no desk store has to "
            "be present to re-freeze it. Bars 0-19 reuse the mixed-unit fixture's price "
            "path; 20-26 are the forward tape the scenarios are graded on. The scan sees "
            "the first 20 bars and the forward replay sees all of them."
        ),
        "as_of": "2026-08-20T16:00:00-04:00",
        "expected_keys": ["expected"],
        "numeric_tolerance": 1e-09,
        "intentional_difference": (
            "This fixture is frozen on the code as it stands BEFORE the Phase 0.10 B-2 "
            "band-variant shadow block. Its purpose is to fail if that block moves any "
            "pre-existing key. New keys appearing on the record are expected and allowed; "
            "a changed value is not, and neither is a changed scenario list. It is "
            "re-frozen only on a deliberate, documented champion change."
        ),
        # --- the fixture itself ----------------------------------------------
        "fixture": "tracker_record_band_variant_parity_v1",
        "frozen_at": "2026-08-26",
        "symbol": SYMBOL,
        "note": (
            "plan.md Phase 0.10 B-2, golden-fixture-first (plan.md section 5). The shadow "
            "block may add current_anchor_variant / previous_anchor_variant and VARIANT_* "
            "stop scenarios; it may not move representative_total_r, avg_total_r, the "
            "tradeable scenario count, setup_status, or any other key recorded here."
        ),
        "rules_under_test": {
            "anchor_index": ANCHOR_INDEX,
            "anchor_date": ANCHOR_DATE,
            "scan_date": SCAN_DATE,
            "generated_at": GENERATED_AT,
            "bars_at_scan": BARS_AT_SCAN,
            "long_row": ROW,
            "short_row": SHORT_ROW,
            "summary_keys": list(SUMMARY_KEYS),
        },
        "bars": bars,
        "expected": {
            "long": measure(bars, ROW),
            "short": measure(bars, SHORT_ROW),
        },
    }
    payload["raw_input_sha256"] = hashlib.sha256(
        json.dumps(
            {key: payload[key] for key in payload["raw_input_keys"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the tracker-record parity fixture")
    parser.add_argument("--force", action="store_true", help="overwrite an existing fixture")
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        print(f"refusing to overwrite {args.out} (pass --force if you mean it)")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
