#!/usr/bin/env python3
"""Freeze the `sector_cohort_v1` golden fixture (plan.md R9.5, sec 5).

The fixture is written BEFORE the detector exists and is the thing the detector
is then built to reproduce. Its inputs are hand-constructed rather than sampled
from a real session, so every case isolates exactly one rule and nothing in it
can drift when a market data vendor revises a bar.

Run:  .venv/Scripts/python.exe scripts/build_sector_cohort_fixture.py
It refuses to overwrite an existing fixture without --force: a golden file that
can be silently regenerated is not golden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "sector_cohort_v1.json"

SESSION = "2026-08-21"
OPEN_ET = datetime(2026, 8, 21, 9, 30)
BARS_PER_SESSION = 78  # 09:30 -> 15:55 ET inclusive, 5-minute bars


def _bar(stamp: datetime, close: float, *, volume: float = 100_000.0) -> dict:
    """A deliberately boring bar: the cohort rule reads closes and opens only."""
    return {
        "dt": stamp.strftime("%Y-%m-%dT%H:%M:00"),
        "open": round(close, 4),
        "high": round(close + 0.02, 4),
        "low": round(close - 0.02, 4),
        "close": round(close, 4),
        "volume": volume,
    }


def _series(open_price: float, path_pct: list[float]) -> list[dict]:
    """``path_pct[i]`` is the cumulative % move from the session open at bar i.

    ``path_pct[0]`` MUST be 0.0: bar 0's open IS the session open and therefore
    the reference every later move is measured against. A non-zero first entry
    silently re-bases the whole series - which is exactly the flaw the first
    draft of this fixture had, and it turned a gap-down case into a gap-up one.
    """
    assert not path_pct or path_pct[0] == 0.0, "path_pct[0] must be 0.0 (it is the open)"
    bars = []
    for index, pct in enumerate(path_pct):
        stamp = OPEN_ET + timedelta(minutes=5 * index)
        bars.append(_bar(stamp, open_price * (1.0 + pct / 100.0)))
    return bars


def _flat_tail(path: list[float], total: int = BARS_PER_SESSION) -> list[float]:
    return path + [path[-1]] * (total - len(path))


def build() -> dict:
    # SPY is flat all session, so every spread below is the ETF's own move.
    # This mirrors 2026-08-21, when SPY closed -0.05% and XLU closed -2.57%.
    spy = _series(766.0, _flat_tail([0.0, -0.05, -0.04, -0.06, -0.05, -0.05]))

    cases = {
        # FIRES: crosses -0.75% at bar 4 and holds, so bar 6 is the third
        # consecutive qualifying bar and the first fire.
        "XLU": {
            "path": _flat_tail([0.0, -0.20, -0.40, -0.60, -0.85, -1.10, -1.40, -1.80, -2.55]),
            "expect_fires": True,
            "expect_first_fire_bar": 6,
            "why": "persists at or beyond -0.75% for >=3 consecutive completed bars",
        },
        # DOES NOT FIRE: touches -0.90% for exactly two bars, then recovers.
        # Two bars is the near-miss the persistence rule exists to reject.
        "XLK": {
            "path": _flat_tail([0.0, -0.10, -0.30, -0.90, -0.95, -0.30, -0.10, 0.05]),
            "expect_fires": False,
            "expect_first_fire_bar": None,
            "why": "only two consecutive qualifying bars; the rule needs three",
        },
        # DOES NOT FIRE: never reaches the threshold at all.
        "XLE": {
            "path": _flat_tail([0.0, -0.10, -0.20, -0.30, -0.40, -0.35, -0.20]),
            "expect_fires": False,
            "expect_first_fire_bar": None,
            "why": "never reaches |spread| >= 0.75%",
        },
        # FIRES LONG SIDE: the mirror case, so the rule cannot be short-only.
        "XLV": {
            "path": _flat_tail([0.0, 0.20, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90]),
            "expect_fires": True,
            "expect_first_fire_bar": 5,
            "why": "the long mirror: +0.75% held for >=3 consecutive bars",
        },
        # DOES NOT FIRE: one violent completed bar that immediately reverses.
        # 31 of the 179 fires measured over 23 sessions sat on the opening bar
        # and were artifacts of exactly this shape, which is why the rule needs
        # three consecutive COMPLETED bars rather than one.
        "XRT": {
            "path": _flat_tail([0.0, -1.60, -0.20, -0.10, -0.05, 0.0]),
            "expect_fires": False,
            "expect_first_fire_bar": None,
            "why": "a single violent first completed bar that reverses; never persists",
        },
    }

    etfs = {}
    expectations = []
    for symbol, case in cases.items():
        etfs[symbol] = _series(100.0, case["path"])
        expectations.append(
            {
                "etf": symbol,
                "fires": case["expect_fires"],
                "first_fire_bar_index": case["expect_first_fire_bar"],
                "why": case["why"],
            }
        )

    payload = {
        # --- repo-wide fixture contract (plan.md Milestone 3) -----------------
        "schema": "sector_cohort_v1",
        "feature_version": "sector_cohort_v1",
        "raw_input_keys": ["benchmark", "etfs"],
        # Filled in below, over the stored sections and in the contract's own
        # canonical byte form (conftest._canonical_json). Hashing anything else
        # produces a fixture that declares one thing and contains another.
        "raw_input_sha256": "",
        "acquired_at": "2026-08-22T00:00:00-04:00",
        "universe_version": "synthetic_sector_cohort_v1",
        "provider_assumptions": (
            "Synthetic regular-session M5 bars, hand-constructed rather than sampled, so no "
            "vendor revision can move a case. Every bar is a completed bar end at a 5-minute "
            "boundary from 09:30 to 15:55 America/New_York. SPY is flat all session, so each "
            "ETF's spread is exactly its own move from the session open. path_pct[0] is 0.0 "
            "in every series because bar 0's open IS the session open."
        ),
        "as_of": "2026-08-21T15:55:00-04:00",
        "expected_keys": ["expected_observations"],
        "numeric_tolerance": 1e-09,
        "intentional_difference": (
            "No live data at all. Real sessions cannot isolate one rule per symbol, and the "
            "two near-miss cases this fixture exists for - two qualifying bars instead of "
            "three, and one violent bar that reverses - would each need a session hunted for "
            "rather than constructed."
        ),
        # --- the fixture itself ----------------------------------------------
        "fixture": "sector_cohort_v1",
        "frozen_at": "2026-08-22",
        "session": SESSION,
        "timezone": "America/New_York",
        "note": (
            "Frozen BEFORE the detector was written (plan.md sec 5, R9.5). Inputs are "
            "hand-constructed so each case isolates one rule and no vendor revision can "
            "move it. SPY is flat throughout, so each ETF's spread is its own move."
        ),
        "rules_under_test": {
            "threshold_pct": 0.75,
            "persistence_bars": 3,
            "completed_bars_only": True,
            "session_only": True,
        },
        "benchmark": {"symbol": "SPY", "bars": spy},
        "etfs": etfs,
        "expected_observations": expectations,
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
    parser = argparse.ArgumentParser(description="Freeze the sector cohort golden fixture")
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
