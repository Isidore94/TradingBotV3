"""Append-only trader market-environment annotations for later AI review.

Manual environment choices are observations, not labels that may silently
change detector/scoring behavior.  Every row keeps the bot's contemporaneous
read beside the user's choice so future research can compare the two honestly.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from project_paths import MARKET_ENVIRONMENT_ANNOTATIONS_FILE


SCHEMA_VERSION = "market_environment_annotation_v1"
_write_lock = threading.Lock()


def _json_safe_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key, item in value.items():
        try:
            json.dumps(item)
        except (TypeError, ValueError):
            safe[str(key)] = str(item)
        else:
            safe[str(key)] = item
    return safe


def record_market_environment_annotation(
    *,
    selected_environment: str | None,
    auto_reading: Mapping[str, Any] | None = None,
    session_id: str = "",
    event: str = "manual_selected",
    reason: str = "",
    now: datetime | None = None,
    path: Path = MARKET_ENVIRONMENT_ANNOTATIONS_FILE,
) -> dict[str, Any] | None:
    """Append one user-mode change and return its normalized row.

    ``selected_environment=None`` means the user returned to N/A and handed
    control back to Auto.  A write failure is deliberately non-fatal to live
    scanning, but callers can detect it from the ``None`` return value.
    """

    selected = str(selected_environment or "").strip().lower()
    reading = _json_safe_mapping(auto_reading)
    moment = now or datetime.now().astimezone()
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts": moment.isoformat(timespec="seconds"),
        "trade_date": moment.date().isoformat(),
        "session_id": str(session_id or "").strip(),
        "event": str(event or "manual_selected").strip().lower(),
        "user_mode": selected or "n/a",
        "manual_override_active": bool(selected),
        "reason": str(reason or "").strip(),
        "auto_environment": str(reading.get("env_key") or "").strip().lower(),
        "auto_reading": reading,
    }
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with _write_lock, target.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
    except OSError:
        return None
    return row


def load_market_environment_annotations(
    path: Path = MARKET_ENVIRONMENT_ANNOTATIONS_FILE,
) -> list[dict[str, Any]]:
    """Load valid rows in append order; tolerate a partial/bad JSONL line."""

    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and row.get("schema_version") == SCHEMA_VERSION:
            rows.append(row)
    return rows
