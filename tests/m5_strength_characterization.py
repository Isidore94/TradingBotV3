"""Freeze the M5 strength functions before R8 shares them (§9 step 2).

``scripts/strength_scan.py`` is **not edited** by R8 — that is a hard rule in the
spec's §2 and §8. But R8's `weekend_strength` imports its pure functions, and a
shared function is a function two packets can now break. This fixture is the
drift insurance: bit-identical M5 behaviour, frozen against a fixed bar set,
regenerated only with an explicit intended-change note.

The bars are synthetic and deterministic on purpose. A real M5 capture would
drag a date and a data provider into a test whose whole job is to be stable.

Regenerate::

    .venv\\Scripts\\python.exe tests/m5_strength_characterization.py --note "why"
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "m5_strength_functions_v1"

#: Enough bars to satisfy ATR50 (51) with room for the 12-bar body window.
BAR_COUNT = 80


def _synthetic_bars(seed: float, drift: float, noise: float) -> list[dict[str, float]]:
    """A deterministic bar series. No RNG - a golden built on one is a coin flip."""
    bars: list[dict[str, float]] = []
    price = seed
    for index in range(BAR_COUNT):
        # A closed-form wobble: reproducible on any machine and any Python build.
        wobble = math.sin(index * 0.7) * noise
        open_price = price
        close = price * (1.0 + drift) + wobble
        high = max(open_price, close) + abs(wobble) * 0.5 + 0.01
        low = min(open_price, close) - abs(wobble) * 0.5 - 0.01
        bars.append(
            {
                "open": round(open_price, 6),
                "high": round(high, 6),
                "low": round(low, 6),
                "close": round(close, 6),
                "volume": 10_000 + index * 37,
            }
        )
        price = close
    return bars


#: Three shapes with different answers: a mover, a fader, and a flat name.
CASES: dict[str, list[dict[str, float]]] = {
    "trending_up": _synthetic_bars(50.0, 0.004, 0.05),
    "trending_down": _synthetic_bars(50.0, -0.004, 0.05),
    "flat_choppy": _synthetic_bars(50.0, 0.0, 0.12),
}


def capture() -> dict[str, Any]:
    """Every shared pure function's answer on every case."""
    from strength_scan import atr, displaced_close, ema, percentile_cut, strength_score

    measured: dict[str, Any] = {}
    for name, bars in CASES.items():
        closes = [bar["close"] for bar in bars]
        measured[name] = {
            "strength_score": strength_score(bars),
            "atr50": atr(bars, 50),
            "ema15": ema(closes, 15),
            "displaced_close_50": displaced_close(closes, 50),
            "bar_count": len(bars),
        }

    scored = [(name, values["strength_score"]) for name, values in measured.items()]
    scored = [(name, value) for name, value in scored if value is not None]
    measured["_percentile"] = {
        "long_top_50pct": percentile_cut(scored, fraction=0.5, side="long"),
        "short_bottom_50pct": percentile_cut(scored, fraction=0.5, side="short"),
    }

    # Short history must refuse, not approximate.
    measured["_refusals"] = {
        "fifty_bars_is_one_short": strength_score(CASES["trending_up"][:50]),
        "eleven_bars": strength_score(CASES["trending_up"][:11]),
        "empty": strength_score([]),
    }
    return measured


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", default="", help="why the expected output changed")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import FIXTURES_DIR, _canonical_json, validate_fixture_contract

    captured = capture()
    path = FIXTURES_DIR / f"{FIXTURE_NAME}.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    if "measured" in previous and previous["measured"] != captured and not args.note.strip():
        print(
            "REFUSED: the M5 functions' output changed and no --note was given.\n"
            "strength_scan.py is not edited by R8 - if this moved, find out why first.",
            file=sys.stderr,
        )
        return 1

    payload: dict[str, Any] = {
        "schema": "m5_strength_functions/v1",
        "feature_version": "r8-step2-shared-function-drift-insurance",
        "universe_version": "n/a (synthetic bars; no symbol universe is read)",
        "provider_assumptions": (
            "No network and no provider. Bars are closed-form synthetic series, so the "
            "golden is reproducible on any machine without a data vendor or a date."
        ),
        "acquired_at": previous.get("acquired_at") or "2026-08-15T17:30:00-07:00",
        "as_of": "2026-08-15T17:30:00-07:00",
        "numeric_tolerance": 1e-09,
        "intentional_difference": args.note.strip() or previous.get("intentional_difference") or "",
        "raw_input_keys": ["bars"],
        "expected_keys": ["measured"],
        "bars": CASES,
        "measured": captured,
    }
    payload["raw_input_sha256"] = hashlib.sha256(_canonical_json(payload["bars"])).hexdigest()
    validate_fixture_contract(payload, FIXTURE_NAME)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
