"""Append-only log of the trader's pick verdicts, built for AI review.

Every ★ like (with its origin: which alert timeframe or screen it came from),
every ✕ dislike (with the trader's typed reason), and every unfavorite is one
JSON object per line in ``pick_feedback.jsonl`` (shared home, syncs across
machines). Hand the file to an AI with a prompt like "review my dislikes and
suggest scan/scoring changes" - each row carries enough context (origin,
category, raw alert text / setup row summary) to reason about.

Row shape:
    {"ts": "...", "trade_date": "YYYY-MM-DD", "symbol": "NVDA", "side": "LONG",
     "verdict": "like" | "dislike" | "unfavorite",
     "category": "swing" | "m5" | "",
     "origin": "h1" | "d1" | "m5" | "setups" | "manual" | "",
     "reason": "<why the trader disliked it>",
     "context": "<alert text or setup-row summary>"}

The like origins also feed the human-focus tracker: the daily snapshot tags
each pick's source as e.g. ``focus_swing_h1`` so H1-alert swing picks grade as
their own cohort next to D1-sourced ones.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from project_paths import PICK_FEEDBACK_FILE


PICK_VERDICTS = ("like", "dislike", "unfavorite")
PICK_ORIGINS = ("h1", "d1", "m5", "setups", "manual")


def _trade_date_text() -> str:
    try:
        from market_session import get_market_session_window

        return get_market_session_window().market_date.isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def record_pick_feedback(
    *,
    symbol: object,
    side: object = "",
    verdict: str,
    category: str = "",
    origin: str = "",
    reason: str = "",
    context: str = "",
    now: datetime | None = None,
    path: Path = PICK_FEEDBACK_FILE,
) -> dict[str, Any] | None:
    """Append one verdict row. Returns the row, or None for a blank symbol."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    side_text = str(side or "").strip().upper()
    row = {
        "ts": (now or datetime.now()).isoformat(timespec="seconds"),
        "trade_date": _trade_date_text(),
        "symbol": sym,
        "side": "SHORT" if side_text.startswith("SHORT") else "LONG" if side_text.startswith("LONG") else side_text,
        "verdict": str(verdict or "").strip().lower(),
        "category": str(category or "").strip().lower(),
        "origin": str(origin or "").strip().lower(),
        "reason": str(reason or "").strip(),
        "context": str(context or "").strip(),
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        return None  # cloud-synced folders can briefly lock files; best-effort
    return row


def load_pick_feedback(path: Path = PICK_FEEDBACK_FILE) -> list[dict[str, Any]]:
    """All feedback rows in file order (oldest first). Bad lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        return []
    return rows


def latest_like_origins(
    rows: list[dict[str, Any]] | None = None,
    *,
    path: Path = PICK_FEEDBACK_FILE,
) -> dict[tuple[str, str, str], str]:
    """{(SYMBOL, LONG/SHORT, category): origin} from the most recent like of each pick.

    Used by the daily human-focus snapshot to tag each pick's cohort source
    (e.g. focus_swing_h1) so origins grade separately in the tracker.
    """
    origins: dict[tuple[str, str, str], str] = {}
    for row in rows if rows is not None else load_pick_feedback(path):
        if str(row.get("verdict") or "").lower() != "like":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        category = str(row.get("category") or "").strip().lower()
        origin = str(row.get("origin") or "").strip().lower()
        if symbol and side in {"LONG", "SHORT"} and origin:
            origins[(symbol, side, category)] = origin
    return origins
