"""Freeze the P8 parity fixture from the code as it was BEFORE the grid existed.

Phase 0.13 packet P8. The packet's template says the golden fixture is pinned
before the simulator is written. The thing that actually needs pinning is not the
new grid's arithmetic - that has no prior behaviour to preserve - it is the
EXISTING `simulate_m5_close_opportunity`, because P8 adds an optional
`entry_selector` parameter to it and every already-published `m5close_*` row
depends on that function being unchanged when the parameter is absent.

So this script builds the expected rows by importing `outcomes.py` AS IT IS ON
`main` (through `git show`, into a temp directory), not the working copy. That is
what makes the fixture a pin rather than a self-portrait: the numbers come from
code that has never heard of P8.

Usage::

    python scripts/build_setup_entry_timing_fixture.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "setup_entry_timing_parity_v1.json"

#: The three cells the P8 control must reproduce. Rank-1 current-anchor stops at
#: the three shared targets - the same selector every P8 cell carries.
#: THE COMMIT the expected rows are pinned FROM, not a moving branch (R1).
#:
#: This read `main` originally, which was correct on the day it ran and becomes
#: a self-portrait the moment P8 merges: a rerun would then compare the new code
#: against itself and pass no matter what it had broken. `1837b63` is the tip of
#: `main` immediately before P8 was cut - the last commit that had never heard of
#: this grid. Moving it forward is a deliberate act, not a side effect of a
#: merge.
PINNED_BASELINE = "1837b63"

PARITY_RECIPE_IDS = (
    "m5close_current_anchor1_1r_v1",
    "m5close_current_anchor1_2r_v1",
    "m5close_current_anchor1_3r_v1",
)

#: A synthetic session that is deliberately boring: one D1 occurrence, a next
#: session that opens above the trigger, drifts up, pulls back to the trigger and
#: recovers, then trends. Boring is the point - the fixture pins the exit
#: machine's arithmetic, and a dramatic path would pin one branch of it.
SESSION_ONE = datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)
SESSION_TWO = datetime(2026, 8, 4, 13, 30, tzinfo=timezone.utc)
#: Two more sessions so all three targets RESOLVE. A fixture whose 2R and 3R
#: cells both end TRUNCATED pins the entry and the stop but never the exit
#: loop's resolution, which is the half most likely to be broken by accident.
SESSION_THREE = datetime(2026, 8, 5, 13, 30, tzinfo=timezone.utc)
SESSION_FOUR = datetime(2026, 8, 6, 13, 30, tzinfo=timezone.utc)
TRIGGER_LEVEL = 100.0


def _bar(start: datetime, open_: float, high: float, low: float, close: float) -> dict:
    return {
        "interval_start": start,
        "interval_end": start + timedelta(minutes=5),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 10_000,
        "is_complete": True,
        "capture_mode": "LIVE",
    }


def _path(start: datetime, closes: list[float]) -> list[dict]:
    bars = []
    previous = closes[0]
    for index, close in enumerate(closes):
        high = max(previous, close) + 0.10
        low = min(previous, close) - 0.10
        bars.append(_bar(start + timedelta(minutes=5 * index), previous, high, low, close))
        previous = close
    return bars


def build_inputs() -> dict:
    """The raw input, as plain JSON-able values. Hashed into the contract."""
    day_one = [round(100.0 + 0.05 * i, 4) for i in range(78)]
    # Session two: open above the trigger, ease back to it, hold, then trend up.
    day_two = (
        [round(100.6 + 0.04 * i, 4) for i in range(12)]      # drift up
        + [round(101.1 - 0.09 * i, 4) for i in range(12)]    # pull back through
        + [round(100.05 + 0.06 * i, 4) for i in range(54)]   # hold and trend
    )
    return {
        "occurrence": {
            "occurrence_id": "p8-parity-0001",
            "symbol": "PARITY",
            "canonical_setup_id": "AVWAPE_TO_FIRST_DEV",
            "side": "LONG",
            "trigger_at": SESSION_ONE.replace(hour=20, minute=0).isoformat(),
            "entry_price_ref": TRIGGER_LEVEL,
            "stop_price_ref": 98.0,
            "dependency_cluster_id": "p8-parity-cluster",
            # The tracker's bounded stop list travels in `tags` as JSON, which
            # is where `_tracker_geometry` reads it from - the fixture uses the
            # store's own shape rather than a convenient one.
            "tags": json.dumps(
                {
                    "stop_candidates": [
                        {"source_type": "current_anchor", "level": 98.5, "close_failure_limit": 2},
                        {"source_type": "current_anchor", "level": 97.0, "close_failure_limit": 2},
                        {"source_type": "sma", "level": 96.0, "close_failure_limit": 2},
                    ]
                },
                sort_keys=True,
            ),
        },
        "session_one_closes": day_one,
        "session_two_closes": day_two,
        "session_three_closes": [round(103.3 + 0.03 * i, 4) for i in range(78)],
        "session_four_closes": [round(105.7 + 0.03 * i, 4) for i in range(78)],
    }


def materialise(inputs: dict) -> tuple[dict, list[dict]]:
    occurrence = dict(inputs["occurrence"])
    occurrence["trigger_at"] = datetime.fromisoformat(occurrence["trigger_at"])
    bars = _path(SESSION_ONE, list(inputs["session_one_closes"]))
    bars += _path(SESSION_TWO, list(inputs["session_two_closes"]))
    bars += _path(SESSION_THREE, list(inputs["session_three_closes"]))
    bars += _path(SESSION_FOUR, list(inputs["session_four_closes"]))
    return occurrence, bars


def _pre_change_module():
    """`outcomes.py` as `main` has it, imported under its own name.

    Imported from a temp copy of the whole package so its relative imports
    resolve; only `outcomes.py` is replaced with the committed version.
    """
    import importlib.util
    import shutil

    committed = subprocess.run(
        ["git", "show", f"{PINNED_BASELINE}:scripts/research_warehouse/outcomes.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp(prefix="p8-parity-"))
    package = tmp / "research_warehouse"
    shutil.copytree(ROOT / "scripts" / "research_warehouse", package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (package / "outcomes.py").write_text(committed, encoding="utf-8")
    sys.path.insert(0, str(tmp))
    for name in [n for n in list(sys.modules) if n.startswith("research_warehouse")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "research_warehouse.outcomes", package / "outcomes.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["research_warehouse.outcomes"] = module
    spec.loader.exec_module(module)
    return module


def expected_rows(module, occurrence: dict, bars: list[dict]) -> dict:
    as_of = SESSION_FOUR + timedelta(days=40)
    rows = {}
    for recipe_id in PARITY_RECIPE_IDS:
        recipe = next(r for r in module.M5_CLOSE_RECIPES if r.recipe_id == recipe_id)
        row = module.simulate_m5_close_opportunity(
            occurrence, bars, recipe, as_of=as_of, computed_at=as_of, run_id="fixture"
        )
        assert row is not None, f"{recipe_id} produced no row; the fixture cannot pin nothing"
        rows[recipe_id] = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in row.items()
            if key not in {"computed_at", "run_id"}
        }
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    inputs = build_inputs()
    module = _pre_change_module()
    occurrence, bars = materialise(inputs)
    expected = expected_rows(module, occurrence, bars)

    raw_input_keys = ["setup_entry_timing_parity_input_v1"]
    # A single raw_input_key hashes the VALUE directly, not {key: value}.
    digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    payload = {
        "schema": "setup_entry_timing_parity_v1",
        "feature_version": "outcomes_m5_close_opportunity_v1",
        "universe_version": "synthetic-single-symbol-v1",
        "provider_assumptions": (
            "Synthetic RTH M5 bars, no provider. The fixture pins arithmetic, not "
            "a data source."
        ),
        "acquired_at": "2026-09-02T00:00:00+00:00",
        "as_of": (SESSION_FOUR + timedelta(days=40)).isoformat(),
        "raw_input_keys": raw_input_keys,
        # The contract wants TOP-LEVEL keys; the three recipe rows live under one.
        "expected_keys": ["expected"],
        "numeric_tolerance": 1e-9,
        "raw_input_sha256": digest,
        "intentional_difference": "",
        "configuration": {
            "stop_selector": "current_anchor:1",
            "pinned_from": "main, before P8 added the entry_selector parameter",
        },
        "setup_entry_timing_parity_input_v1": inputs,
        "expected": expected,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text, encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(text[:400])
    for recipe_id, row in expected.items():
        print(f"  {recipe_id}: {row['result_state']} gross_r={row['gross_r']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
