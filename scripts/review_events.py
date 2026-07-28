"""Append-only log of every Alert Center review decision, for learning.

The learning triple is (features -> trader action -> outcome). Outcomes are
already tracked (intraday bounce outcomes CSV, human-focus forward returns)
and every bounce alert already carries its feature snapshot plus an
``event_id`` joining it to the 41-column candidates CSV. The missing middle -
what was SHOWN in the review pane and what the trader DID about it - is what
this file captures, one JSON object per line in ``alert_review_events.jsonl``
(shared home, syncs across machines).

Actions (see the Alert Center panel for the emit sites):
    shown           an alert became the active visual review (the impression -
                    the denominator for P(take | shown))
    skip            "Skip for now" - looked at the chart and passed
    remove_today    "Remove for today" / the ✕ dislike's removal
    restore_today   a removed symbol returned to processing
    add_focus       the review pane's type-matched focus add (advances queue)
    toggle_d1_focus / toggle_m5_focus   cross-focus toggles (detail.on)
    favorite        the feed item ★ (detail.on)
    dislike         the feed item ✕ (detail.reason)
    arm_watch / disarm_watch            one-shot session watches (detail.kind)
    watch_fired / watch_expired         how an armed watch actually ended
    arm_level / disarm_level            persistent price-level alerts
                    (detail.direction/level/fill_source: which quick-fill
                    button - vwap, upper_1, hod, ... - produced the price)
    level_fired     a persistent level alert triggered

Every row snapshots the alert's decision-relevant context as structured
fields (tier, bounce types, RRS numbers, session rvol, market environment),
so the log is analyzable standalone; ``event_id`` joins back to the full
candidate row when deeper features are needed. ``dwell_ms`` separates
"considered and passed" from "flushed the queue".

This module must stay import-light (no Qt, no pandas): the GUI calls it on
every click and the offline analysis job imports it headless.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from project_paths import ALERT_REVIEW_EVENTS_FILE

REVIEW_EVENTS_SCHEMA = "review_events_v1"

_TIER_RE = re.compile(r"\[([SABCD])-TIER\]", re.IGNORECASE)
_PROVEN_RE = re.compile(r"\bPROVEN\b")

# context_json keys worth inlining on every row: the numbers the trader is
# implicitly weighing when they act on a chart. Everything else stays behind
# the event_id join.
_CONTEXT_FIELDS = (
    "rrs_spy",
    "rrs_sector",
    "rrs_industry",
    "session_rvol",
    "market_environment",
    "internals_tape",
    "internals_breadth_spread",
    "sector",
    "industry",
)


def _trade_date_text() -> str:
    try:
        from market_session import get_market_session_window

        return get_market_session_window().market_date.isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _as_float(value) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved == resolved else None  # drop NaN


def alert_context_fields(alert) -> dict[str, Any]:
    """Structured decision context from a BounceAlert-shaped object.

    Duck-typed so tests and the offline job can pass plain stand-ins; every
    field degrades to ""/None rather than raising, because a malformed alert
    must never break the click that is being logged.
    """
    fields: dict[str, Any] = {}
    if alert is None:
        return fields
    raw_text = str(getattr(alert, "raw_text", "") or "")
    match = _TIER_RE.search(raw_text)
    fields["tier"] = match.group(1).upper() if match else ""
    fields["proven"] = bool(_PROVEN_RE.search(raw_text))
    fields["banger"] = "BANGER" in raw_text.upper()
    fields["tag"] = str(getattr(alert, "tag", "") or "")
    fields["timeframe"] = str(getattr(alert, "timeframe", "") or "")
    fields["is_d1"] = bool(getattr(alert, "is_d1", False))
    fields["trigger"] = str(getattr(alert, "trigger", "") or "")

    payload = getattr(alert, "payload", None)
    feedback = payload.get("feedback") if isinstance(payload, dict) else None
    feedback = feedback if isinstance(feedback, dict) else {}
    fields["event_id"] = str(feedback.get("event_id") or "")
    fields["bounce_types"] = str(feedback.get("bounce_types") or "")
    fields["entry_price"] = _as_float(feedback.get("entry_price"))
    fields["stop_price"] = _as_float(feedback.get("stop_price"))
    fields["risk_per_share"] = _as_float(feedback.get("risk_per_share"))
    fields["score"] = _as_float(feedback.get("score"))
    fields["is_focus_pick"] = bool(feedback.get("is_focus_pick"))

    context = feedback.get("context_json")
    if isinstance(context, str) and context.strip():
        try:
            context = json.loads(context)
        except (json.JSONDecodeError, ValueError):
            context = None
    if isinstance(context, dict):
        for key in _CONTEXT_FIELDS:
            if key in context:
                fields[key] = context.get(key)

    # Chart-watch hits carry their own payload shape instead of feedback.
    if isinstance(payload, dict) and payload.get("chart_watch_kind"):
        fields["chart_watch_kind"] = str(payload.get("chart_watch_kind") or "")
    return fields


def record_review_event(
    action: str,
    *,
    alert=None,
    symbol: object = "",
    side: object = "",
    detail: dict[str, Any] | None = None,
    dwell_ms: int | None = None,
    queue_len: int | None = None,
    now: datetime | None = None,
    path: Path = ALERT_REVIEW_EVENTS_FILE,
) -> dict[str, Any] | None:
    """Append one decision row. Returns the row, or None when unusable.

    Best-effort like pick_feedback: a cloud-synced folder briefly locking the
    file must never surface as a GUI error, so OSError is swallowed.
    """
    action_text = str(action or "").strip().lower()
    sym = str(symbol or getattr(alert, "symbol", "") or "").strip().upper()
    if not action_text or not sym:
        return None
    side_text = str(side or getattr(alert, "side", "") or "").strip().upper()
    row: dict[str, Any] = {
        "schema": REVIEW_EVENTS_SCHEMA,
        "ts": (now or datetime.now()).isoformat(timespec="seconds"),
        "trade_date": _trade_date_text(),
        "action": action_text,
        "symbol": sym,
        "side": side_text,
    }
    row.update(alert_context_fields(alert))
    if dwell_ms is not None:
        row["dwell_ms"] = max(0, int(dwell_ms))
    if queue_len is not None:
        row["queue_len"] = max(0, int(queue_len))
    if detail:
        row["detail"] = detail
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    except OSError:
        return None
    return row


def load_review_events(path: Path = ALERT_REVIEW_EVENTS_FILE) -> list[dict[str, Any]]:
    """All event rows in file order (oldest first). Bad lines are skipped."""
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
