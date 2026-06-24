"""Master AVWAP priority-bucket state across scans, for strict D1 Focus Alerts.

The D1 Focus Alerts panel should only surface *genuine upgrades* into the best
buckets (Favorite / High Conviction) — not generic D1/stdev noise, and not
favorite->favorite re-flags. We persist each setup's last bucket by symbol+side
and compare it to the current scan.

Plain Python (no Qt) so the headless `run_master` maintains it. See
GUI_REDESIGN_PLAN.md (Focus Picks + Human Setup Tracker), Step 5.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from project_paths import MASTER_AVWAP_BUCKET_STATE_FILE


UPGRADE_TARGET_BUCKETS = {"favorite_setup", "high_conviction"}


def _norm_bucket(value: Any) -> str:
    return str(value or "").strip().lower()


def _row_bucket(row: dict[str, Any]) -> str:
    return _norm_bucket(row.get("priority_bucket") if row.get("priority_bucket") is not None else row.get("bucket"))


def _key(symbol: Any, side: Any) -> str:
    return f"{str(symbol or '').strip().upper()}|{str(side or '').strip().upper()}"


def is_bucket_upgrade(previous_bucket: Any, current_bucket: Any) -> bool:
    """True only for a genuine upgrade INTO Favorite / High Conviction.

    Upgrades: any non-target bucket (missing/unbucketed/near/tracking) -> target,
    and favorite_setup -> high_conviction. Not upgrades: favorite->favorite,
    hc->hc, hc->favorite (downgrade), or anything ending in a non-target bucket.
    """
    current = _norm_bucket(current_bucket)
    if current not in UPGRADE_TARGET_BUCKETS:
        return False
    previous = _norm_bucket(previous_bucket)
    if previous in UPGRADE_TARGET_BUCKETS:
        return previous == "favorite_setup" and current == "high_conviction"
    return True


def compute_bucket_upgrades(
    rows: Iterable[dict[str, Any]],
    previous_state: dict[str, Any],
) -> list[dict[str, str]]:
    """Return upgrade events for the current scan vs the previous bucket state."""
    upgrades: list[dict[str, str]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        if not symbol or not side:
            continue
        current = _row_bucket(row)
        previous = _norm_bucket((previous_state.get(_key(symbol, side)) or {}).get("bucket"))
        if is_bucket_upgrade(previous, current):
            upgrades.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "previous_bucket": previous,
                    "bucket": current,
                }
            )
    return upgrades


def update_bucket_state(
    state: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    scan_date: Any,
) -> dict[str, Any]:
    """Record each row's current bucket (in place) so the next scan can diff it.

    Pass the FULL set of scanned priority rows (including unbucketed ones, bucket
    "") so a name that drops out of the favorites is recorded and can re-fire on a
    later genuine upgrade.
    """
    now = datetime.now().isoformat(timespec="seconds")
    scan_text = str(scan_date or "")
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        if not symbol or not side:
            continue
        state[_key(symbol, side)] = {
            "bucket": _row_bucket(row),
            "scan_date": scan_text,
            "updated_at": now,
        }
    return state


def load_bucket_state(path: Path = MASTER_AVWAP_BUCKET_STATE_FILE) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_bucket_state(state: dict[str, Any], path: Path = MASTER_AVWAP_BUCKET_STATE_FILE) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # Cloud-synced folders can briefly lock files; state is best-effort.
        pass


def record_scan_bucket_upgrades(
    rows: Iterable[dict[str, Any]],
    scan_date: Any,
    *,
    path: Path = MASTER_AVWAP_BUCKET_STATE_FILE,
) -> list[dict[str, str]]:
    """One-shot for `run_master`: diff against saved state, persist new state,
    and return the genuine bucket upgrades for this scan."""
    rows = list(rows or [])
    previous_state = load_bucket_state(path)
    upgrades = compute_bucket_upgrades(rows, previous_state)
    update_bucket_state(previous_state, rows, scan_date)
    save_bucket_state(previous_state, path)
    return upgrades
