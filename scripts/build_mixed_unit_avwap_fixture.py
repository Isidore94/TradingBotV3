#!/usr/bin/env python3
"""Freeze the `mixed_unit_avwap_v1` golden fixture (plan.md R10.V step 1).

**This fixture pins the WRONG answer on purpose.** The durable daily-bar store
mixes two volume units - IB regular-session round lots and Yahoo consolidated
shares - and AVWAP is volume-weighted, so a splice between them re-weights every
level computed across it. R10.V repairs the store. This fixture is what makes
that repair a *visible* change rather than a silent one: it states, in frozen
numbers, exactly what the mixed series produces today and what the clean series
would have produced instead.

Three series over the same twenty bars and the same prices:

* ``shares`` - every bar in Yahoo share units. The correct series.
* ``mixed``  - the same bars with everything from the splice onward divided by
  100, which is the shape measured in 1,179 of the 1,236 files the 2026-08-21
  scan rewrote (median step x0.0088 at 2026-07-29).
* ``lots``   - every bar divided by 100. **The control.** A uniform rescale
  leaves AVWAP and sigma unchanged, because a volume-weighted ratio cancels a
  constant factor; only a SPLICE moves them. That is the numeric argument for
  R10.V refusing IB volume outright instead of converting it, and it is here as
  a measurement rather than as a claim in a document.

The bars are hand-constructed, not sampled, so no vendor revision can move the
fixture, and the symbol is synthetic for the same reason.

Run:  .venv/Scripts/python.exe scripts/build_mixed_unit_avwap_fixture.py
It refuses to overwrite an existing fixture without --force: a golden file that
can be silently regenerated is not golden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "mixed_unit_avwap_v1.json"

SYMBOL = "MIXQ"
ANCHOR_INDEX = 0
SPLICE_INDEX = 12  # the bar where IB round-lot rows begin
LOT_SIZE = 100

# (date, open, high, low, close, share-volume). A gentle advance, a pullback,
# then a push to new highs - enough shape that sigma is not degenerate.
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
    # --- splice: the rows below arrive from IB in round lots -----------------
    ("2026-07-31", 42.35, 43.18, 42.20, 43.05, 51_640_200.0),
    ("2026-08-03", 43.10, 43.77, 42.88, 43.62, 45_118_600.0),
    ("2026-08-04", 43.66, 44.30, 43.44, 43.91, 39_874_500.0),
    ("2026-08-05", 43.95, 44.12, 43.02, 43.18, 58_206_300.0),
    ("2026-08-06", 43.20, 43.64, 42.55, 42.77, 62_913_400.0),
    ("2026-08-07", 42.80, 43.51, 42.66, 43.40, 41_007_900.0),
    ("2026-08-10", 43.44, 44.60, 43.38, 44.48, 68_552_100.0),
    ("2026-08-11", 44.52, 45.31, 44.17, 45.16, 71_336_800.0),
]

BAND_KEYS = ("UPPER_1", "LOWER_1", "UPPER_2", "LOWER_2", "UPPER_3", "LOWER_3")


def _rows(scale: callable) -> list[dict]:
    out = []
    for index, (day, open_, high, low, close, volume) in enumerate(BARS):
        out.append(
            {
                "date": day,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": scale(index, volume),
            }
        )
    return out


def _series() -> dict[str, list[dict]]:
    return {
        "shares": _rows(lambda index, volume: volume),
        "mixed": _rows(
            lambda index, volume: volume / LOT_SIZE if index >= SPLICE_INDEX else volume
        ),
        "lots": _rows(lambda index, volume: volume / LOT_SIZE),
    }


def _measure(rows: list[dict]) -> dict:
    import pandas as pd

    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    frame = pd.DataFrame(rows)
    vwap, stdev, bands = calc_anchored_vwap_bands(frame, ANCHOR_INDEX)
    return {
        "vwap": vwap,
        "stdev": stdev,
        "bands": {key: bands[key] for key in BAND_KEYS},
    }


def build() -> dict:
    series = _series()
    measured = {name: _measure(rows) for name, rows in series.items()}

    shares = measured["shares"]
    mixed = measured["mixed"]
    damage = {
        "vwap_delta": mixed["vwap"] - shares["vwap"],
        "vwap_delta_pct": (mixed["vwap"] - shares["vwap"]) / shares["vwap"] * 100.0,
        "stdev_ratio": mixed["stdev"] / shares["stdev"],
        "upper_2_delta": mixed["bands"]["UPPER_2"] - shares["bands"]["UPPER_2"],
    }

    payload = {
        # --- repo-wide fixture contract (plan.md Milestone 3) ----------------
        "schema": "mixed_unit_avwap_v1",
        "feature_version": "calc_anchored_vwap_bands_running_deviation_v1",
        "raw_input_keys": ["series", "rules_under_test"],
        "raw_input_sha256": "",
        "acquired_at": "2026-08-22T21:00:00-04:00",
        "universe_version": "synthetic_mixed_unit_v1",
        "provider_assumptions": (
            "No provider call. Twenty hand-constructed daily bars for a synthetic symbol, "
            "so no vendor revision can move this fixture. The three series share IDENTICAL "
            "prices and differ only in the volume column: 'shares' is Yahoo consolidated "
            "share volume throughout; 'mixed' divides bar 12 onward by 100, the shape IB "
            "round-lot regular-session volume produces when it is spliced onto Yahoo "
            "history; 'lots' divides every bar by 100 and is the uniform-rescale control. "
            "sigma is the running-deviation variant every band consumer is calibrated to "
            "(plan.md sec 5) and is NOT to be swapped."
        ),
        "as_of": "2026-08-11T16:00:00-04:00",
        "expected_keys": ["expected"],
        "numeric_tolerance": 1e-09,
        "intentional_difference": (
            "This fixture deliberately pins the WRONG answer alongside the right one. The "
            "'mixed' expectations are what the live store produces today; R10.V's backfill "
            "does not change them, because the fixture feeds fixed frames - it changes "
            "which frame the store hands the detector. When an AVWAP-derived fixture is "
            "re-frozen in R10.V step 5, this fixture is the control that explains why it "
            "moved."
        ),
        # --- the fixture itself ----------------------------------------------
        "fixture": "mixed_unit_avwap_v1",
        "frozen_at": "2026-08-22",
        "symbol": SYMBOL,
        "note": (
            "R10.V step 1. Prices are identical across all three series; only volume "
            "differs. The 'lots' control proves a uniform rescale cannot be the mechanism: "
            "AVWAP is a volume-weighted ratio, so a constant factor cancels and the levels "
            "do not move. Only the splice moves them."
        ),
        "rules_under_test": {
            "anchor_index": ANCHOR_INDEX,
            "splice_index": SPLICE_INDEX,
            "lot_size": LOT_SIZE,
            "sigma": "running deviation from the running AVWAP, accumulated volume-weighted",
            "zero_or_negative_volume_bars": "skipped by the band function",
        },
        "series": series,
        "expected": {
            "shares": shares,
            "mixed": mixed,
            "lots": lots_expectation(measured),
            "damage": damage,
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


def lots_expectation(measured: dict) -> dict:
    """The control's own numbers, stored so the test compares against a value.

    Storing them separately (rather than reusing `shares`) is the point: if a
    future change ever DID make a uniform rescale move the levels, this fixture
    would fail rather than quietly agree with itself.
    """
    return measured["lots"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the mixed-unit AVWAP golden fixture")
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
