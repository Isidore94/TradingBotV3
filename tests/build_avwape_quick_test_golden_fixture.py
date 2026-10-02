"""Build the AVWAPE quick test golden (`tests/fixtures/avwape_quick_test_golden_v1.json`).

    python tests/build_avwape_quick_test_golden_fixture.py <daily_bars dir> <d1_features.csv> <earnings_dates_cache.json>

Every source is only READ (the machine cache's daily bars, the latest scan's d1_features.csv for
the cap and atr20, the earnings-dates cache). The universe: every cached name with `ROW_BARS`
bars ending `AS_OF`, trimmed to `ROW_BARS`. Kept: each name where the rule fired, on either
side, on any of the last `FIRE_SESSIONS` sessions (so the golden holds real positives and the
names that fired earlier are negatives on `AS_OF`), plus `CONTROL_NAMES` seeded names that never
fired. ``expected`` is what `avwape_quick_test.build_rows` returns on them at the time of the build.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "avwape_quick_test_golden_v1.json"
AS_OF = "2026-10-01"
ROW_BARS = 400
FIRE_SESSIONS = 5
CONTROL_NAMES = 12
#: Keeps the fixture well under 2 MB (the AS_OF fires are always kept first).
MAX_NAMES = 60


def _bars(path: Path) -> list[list]:
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["datetime"][:10] <= AS_OF]
    dedup = {row["datetime"][:10]: row for row in rows}
    out = [[day, *(round(float(dedup[day][key]), 4) for key in ("open", "high", "low", "close")),
            float(dedup[day]["volume"] or 0.0)] for day in sorted(dedup)]
    return out[-ROW_BARS:]


def main(argv: list[str]) -> int:
    bars_dir, features_csv, dates_json = (Path(arg).resolve() for arg in argv[1:4])
    scratch = Path(tempfile.mkdtemp(prefix="avwape_quick_test_golden_"))
    os.environ["TRADINGBOTV3_DATA_DIR"] = str(scratch / "home")
    os.environ["LOCALAPPDATA"] = str(scratch / "localappdata")
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "tests"))
    import project_paths

    for name in ("DATA_DIR", "CACHE_DIR", "RUNTIME_DATA_DIR"):
        if str(scratch) not in str(getattr(project_paths, name)):
            raise SystemExit(f"{name} is not in scratch: {getattr(project_paths, name)}")
    import avwape_quick_test as aqt
    from test_avwape_quick_test_golden import _bars as as_dicts
    from test_avwape_quick_test_golden import build

    with features_csv.open(encoding="utf-8-sig", newline="") as handle:
        features = {row["symbol"].strip().upper(): row for row in csv.DictReader(handle) if row.get("symbol")}
    caps = {symbol: float(row["perm_market_cap_m"]) for symbol, row in features.items()
            if row.get("perm_market_cap_m")}
    # The scan's atr20 counts only for a row whose last bar is the completed AS_OF bar (the runner's rule).
    atrs = {symbol: round(float(row["atr20"]), 6) for symbol, row in features.items()
            if row.get("atr20") and str(row.get("last_trade_date") or "")[:10] == AS_OF}
    dates = {symbol.strip().upper(): sorted(entry.get("dates") or ())
             for symbol, entry in json.loads(dates_json.read_text(encoding="utf-8"))["symbols"].items()
             if (entry or {}).get("dates")}

    universe: dict[str, list[list]] = {}
    for path in sorted(bars_dir.glob("*.csv")):
        symbol = path.stem.upper()
        if symbol == "SPY":
            continue
        rows = _bars(path)
        if len(rows) == ROW_BARS and rows[-1][0] == AS_OF:
            universe[symbol] = rows
    fired_now, fired_before = set(), set()
    for symbol, rows in universe.items():
        if symbol not in dates or symbol not in caps:
            continue
        for back in range(FIRE_SESSIONS):
            bars = as_dicts(rows[:len(rows) - back])
            payload = aqt.build_rows(bars_by_symbol={symbol: bars}, earnings_dates_by_symbol={symbol: dates[symbol]},
                                     atr_by_symbol=atrs if back == 0 else None, market_cap_by_symbol=caps,
                                     as_of=bars[-1]["date"])
            if payload["rows"]:
                (fired_now if back == 0 else fired_before).add(symbol)
    kept = sorted(fired_now)
    before = sorted(fired_before - fired_now)
    random.Random("avwape-quick-test-golden-v1").shuffle(before)
    kept += before[:max(0, MAX_NAMES - CONTROL_NAMES - len(kept))]
    controls = sorted(symbol for symbol in universe
                      if symbol in caps and symbol in dates and symbol not in fired_now | fired_before)
    random.Random("avwape-quick-test-golden-v1-controls").shuffle(controls)
    kept += controls[:CONTROL_NAMES]
    raw = {
        "as_of": AS_OF,
        "bars": {symbol: universe[symbol] for symbol in sorted(kept)},
        "spy": _bars(bars_dir / "SPY.csv"),
        "feature_rows": [{"symbol": symbol, "perm_market_cap_m": caps[symbol]} for symbol in sorted(kept)],
        "atr": {symbol: atrs[symbol] for symbol in sorted(kept) if symbol in atrs},
        "earnings_dates": {symbol: dates[symbol] for symbol in sorted(kept)},
    }
    raw = json.loads(json.dumps(raw, sort_keys=True))
    expected = json.loads(json.dumps(build(raw)))
    payload = {
        "schema": "avwape_quick_test_golden_v1",
        "feature_version": "avwape_quick_test.build_rows, first cut (trader, 2026-10-02)",
        "universe_version": (f"{len(kept)} of the {len(universe)} cached names with {ROW_BARS} bars ending {AS_OF}: "
                             f"every {AS_OF} fire, names that fired in the {FIRE_SESSIONS - 1} sessions before, "
                             f"{CONTROL_NAMES} seeded controls"),
        "provider_assumptions": "machine-cache daily bars (read-only), the latest scan's d1_features.csv cap and "
                                "atr20, the earnings-dates cache",
        "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": f"{AS_OF}T16:00:00-04:00",
        "raw_input_keys": ["raw"],
        "expected_keys": ["expected"],
        "numeric_tolerance": 0.0,
        "intentional_difference": "",
        "raw": raw,
        "raw_input_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "expected": expected,
    }
    FIXTURE.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    sides = [row["side"] for row in expected["rows"]]
    print(FIXTURE, FIXTURE.stat().st_size, "names", len(kept), "fired now", len(fired_now),
          "fired before", len(fired_before - fired_now), "LONG", sides.count("LONG"), "SHORT", sides.count("SHORT"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
