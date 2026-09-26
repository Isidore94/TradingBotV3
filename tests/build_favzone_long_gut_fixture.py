"""Build the favourite-zone long gut golden (`tests/fixtures/favzone_long_gut_v1.json`).

    python tests/build_favzone_long_gut_fixture.py <scratch copy of master_avwap_ai_state.json>

Never point it at the live file: copy it to a scratch folder first. The rows are
the scan's ai_state entries, slimmed (bars, candidates, variants and note text
dropped; floats to 4 dp): every LONG favourite / near / favourite-zone row, every
SHORT favourite, 100 SHORT near rows and 40 unbucketed rows per side (seeded).
The expected block is what the code at the time of the build produces on them.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "favzone_long_gut_v1.json"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

DROP = {
    "feature_row", "daily_ohlc", "setup_candidate", "theta_put_candidate", "theta_pcs_candidate",
    "entry_feature_snapshot", "current_anchor_variant", "previous_anchor_variant",
    "setup_tag_evidence", "setup_tag_roles", "cloud_level_nearby_summary", "hv_level_nearby_summary",
    "hv_level_blocking_summary", "current_nearby_bands", "previous_nearby_bands", "multi_day_patterns",
    "events_all_for_day",
}
DROP_PREFIXES = ("bouncebot_", "pqs_", "cloud_level_", "latest_known_", "relative_avwap_", "distance_from_",
                 "pct_from_", "industry_13w", "return_", "spy_", "symbol_", "weekly_ema8")
ANCHOR_KEYS = ("current_anchor", "previous_anchor", "post_earnings_anchor")


def _round(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def slim(entry: dict) -> dict:
    out = {}
    for key, value in entry.items():
        if key in DROP or key.startswith(DROP_PREFIXES) or key.endswith(("_ratio", "_rule_version")):
            continue
        if key in ANCHOR_KEYS and isinstance(value, dict):
            value = {k: v for k, v in value.items() if not isinstance(v, (list, dict))}
        if value is None or value == "" or value == [] or value == {} or value is False:
            continue
        if key.endswith("_note") and "gate" not in key and isinstance(value, str):
            value = "note"
        out[key] = _round(value)
    return out


def select_rows(state: dict) -> list[dict]:
    rows = []
    for symbol, entry in sorted(state["symbols"].items()):
        if isinstance(entry, dict) and entry.get("priority_score") is not None:
            rows.append({**slim(entry), "symbol": symbol})
    keep = [r for r in rows if (r.get("side") == "LONG" and (r.get("priority_bucket") or r.get("favorite_zone")))
            or r.get("priority_bucket") == "favorite_setup"]
    near_shorts = [r for r in rows if r.get("side") == "SHORT" and r.get("priority_bucket") == "near_favorite_zone"]
    random.Random("favzone-long-gut-near-short").shuffle(near_shorts)
    keep += near_shorts[:100]
    for side in ("LONG", "SHORT"):
        rest = [r for r in rows if r.get("side") == side and r not in keep]
        random.Random(f"favzone-long-gut-{side}").shuffle(rest)
        keep += rest[:40]
    return sorted(keep, key=lambda r: r["symbol"])


def main(argv: list[str]) -> int:
    import os

    scratch = Path(tempfile.mkdtemp(prefix="favzone_gut_"))
    os.environ["TRADINGBOTV3_DATA_DIR"] = str(scratch / "home")
    os.environ["LOCALAPPDATA"] = str(scratch / "localappdata")
    import project_paths

    for name in ("DATA_DIR", "CACHE_DIR", "RUNTIME_DATA_DIR"):
        if str(scratch) not in str(getattr(project_paths, name)):
            raise SystemExit(f"{name} is not in scratch: {getattr(project_paths, name)}")
    source = Path(argv[1])
    if "TradingBotData" in str(source.resolve()):
        raise SystemExit("copy the live ai_state to a scratch folder first")
    state = json.loads(source.read_text(encoding="utf-8"))
    raw = {"rows": select_rows(state)}
    raw = json.loads(json.dumps(raw, sort_keys=True, default=str))

    import favzone_long_gut_replay as golden_module

    with tempfile.TemporaryDirectory() as tmp:
        expected = {"scan": golden_module.observe_scan(raw["rows"], Path(tmp)),
                    "points": golden_module.observe_points()}
    payload = {
        "schema": "favzone_long_gut_v1",
        "feature_version": "master_avwap priority buckets / tiers / D1 triggers / tracker controls before the "
                           "favourite-zone long gut (branch claude/p9-gut-favzone-long-2026-09-26)",
        "universe_version": f"live D1 scan {state.get('run_date')} after close ({len(raw['rows'])} rows)",
        "provider_assumptions": "scratch copy of master_avwap_ai_state.json (run_timestamp "
                                f"{state.get('run_timestamp')}); priority rows rebuilt from priority_* fields",
        "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(state.get("run_timestamp") or "") + "-07:00",
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
    print(FIXTURE, FIXTURE.stat().st_size, len(raw["rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
