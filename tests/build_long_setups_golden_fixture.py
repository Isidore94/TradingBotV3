"""Build the Long leaders golden (`tests/fixtures/long_setups_golden_v1.json`).

    python tests/build_long_setups_golden_fixture.py <daily_bars dir> <d1_features.csv> <long_setups.json>

Every source is only READ (the machine cache's daily bars, the scan's d1_features.csv and its
long_setups.json). The inputs: the 2026-09-25 scan's top Long leaders rows (bars in full), 60
seeded other names (their last 70 bars, only for the RS percentiles), SPY, each row name's
feature facts (gate, sector rank, family, cap), earnings gap and scan ATR. ``expected`` is what
`long_setups.build_rows` returns on them at the time of the build.
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
FIXTURE = ROOT / "tests" / "fixtures" / "long_setups_golden_v1.json"
AS_OF = "2026-09-25"
ROW_NAMES = 26
FILLER_NAMES = 60
FILLER_BARS = 70
#: Enough for a 52-week high over the 120-session lookback (252 + 120).
ROW_BARS = 400
FEATURE_KEYS = ("symbol", "side", "perm_regime_working", "perm_regime_working_rule", "perm_sector_rs_rank_20d",
                "perm_sector_rank_count", "setup_family", "sector", "perm_market_cap_m")


def _bars(path: Path, last: int | None = None) -> list[list]:
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["datetime"][:10] <= AS_OF]
    dedup = {row["datetime"][:10]: row for row in rows}
    out = [[day, *(round(float(dedup[day][key]), 4) for key in ("open", "high", "low", "close")),
            float(dedup[day]["volume"] or 0.0)] for day in sorted(dedup)]
    return out[-last:] if last else out


def main(argv: list[str]) -> int:
    bars_dir, features_csv, payload_json = (Path(arg).resolve() for arg in argv[1:4])
    scratch = Path(tempfile.mkdtemp(prefix="long_setups_golden_"))
    os.environ["TRADINGBOTV3_DATA_DIR"] = str(scratch / "home")
    os.environ["LOCALAPPDATA"] = str(scratch / "localappdata")
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "tests"))
    import project_paths

    for name in ("DATA_DIR", "CACHE_DIR", "RUNTIME_DATA_DIR"):
        if str(scratch) not in str(getattr(project_paths, name)):
            raise SystemExit(f"{name} is not in scratch: {getattr(project_paths, name)}")

    live = json.loads(payload_json.read_text(encoding="utf-8"))
    names = [row["symbol"] for row in live["rows"]][:ROW_NAMES]
    with features_csv.open(encoding="utf-8-sig", newline="") as handle:
        features = [row for row in csv.DictReader(handle) if row["symbol"] in names]
    others = sorted(path.stem.upper() for path in bars_dir.glob("*.csv")
                    if path.stem.upper() not in names and path.stem.upper() != "SPY")
    random.Random("long-setups-golden-v1").shuffle(others)
    bars = {name: _bars(bars_dir / f"{name}.csv", ROW_BARS) for name in names}
    for name in others:
        if len(bars) >= ROW_NAMES + FILLER_NAMES:
            break
        rows = _bars(bars_dir / f"{name}.csv", FILLER_BARS)
        if len(rows) == FILLER_BARS and rows[-1][0] == AS_OF:
            bars[name] = rows
    raw = {
        "as_of": AS_OF,
        "bars": bars,
        "spy": _bars(bars_dir / "SPY.csv"),
        "feature_rows": [{key: row.get(key, "") for key in FEATURE_KEYS} for row in features],
        "earnings": {row["symbol"]: {"gap_date": row["latest_release_gap_date"],
                                     "gap_is_up": (float(row["perm_earnings_gap_atr_signed"] or 0) > 0),
                                     "gap_atr_multiple": abs(float(row["perm_earnings_gap_atr_signed"] or 0))}
                     for row in features if row["latest_release_gap_date"]},
        "atr": {row["symbol"]: round(float(row["atr20"]), 6) for row in features if row["atr20"]},
    }
    raw = json.loads(json.dumps(raw, sort_keys=True))
    from test_long_setups_golden import build

    payload = {
        "schema": "long_setups_golden_v1",
        "feature_version": "long_setups.build_rows before the strength + earnings-AVWAP promotion tier "
                           "(branch claude/p9-strength-avwape-2026-09-27)",
        "universe_version": f"the {AS_OF} scan's first {ROW_NAMES} Long leaders names + {FILLER_NAMES} seeded others",
        "provider_assumptions": "machine-cache daily bars (read-only), the scan's d1_features.csv facts, "
                                "its atr20 and latest_release_gap_date",
        "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": f"{AS_OF}T16:00:00-04:00",
        "raw_input_keys": ["raw"],
        "expected_keys": ["expected"],
        "numeric_tolerance": 0.0,
        "intentional_difference": "",
        "raw": raw,
        "raw_input_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "expected": json.loads(json.dumps(build(raw))),
    }
    FIXTURE.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(FIXTURE, FIXTURE.stat().st_size, len(bars), len(payload["expected"]["rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
