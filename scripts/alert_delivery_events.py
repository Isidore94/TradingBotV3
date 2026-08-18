"""Machine-local record of what the Alert Center actually DELIVERED.

Packet Phase 1 (``docs/ALERT_CENTER_QUALITY_PACKET.md``). The companion to
``review_events.py``: that module records what the trader did about an alert,
this one records that the alert was put in front of them at all, and whether it
made noise. Without this row the desk's own alerting is unmeasurable - loud
volume, duplicate rate, armed-hit latency, and missed winners all lack a
denominator.

Storage class (trader decision): **machine-local, never the Drive home
folder.** A delivery row is written per alert reaching the feed, which is far
higher volume than a decision row, and the review-event store is cloud-synced.
Files live under the machine-local diagnostics root and are partitioned by
month, so retention is a matter of deleting old files and one session's volume
can never grow an unbounded single file. Nothing here is shared state and
nothing syncs.

Escalation is deliberately NOT decided at write time. The duplicate-rate metric
turns on whether a repeat delivery was a "genuine escalation", and that rule is
a trader judgement that may be revised. So each row stores the *inputs* to the
rule - the tier, whether it was loud, and whether it was an armed condition
firing - and the reader applies the rule. Revising the definition then costs a
re-read, not a re-instrumentation and another month of waiting for data.

Best-effort, exactly like ``review_events`` and ``pick_feedback``: every write
path swallows OSError and lock contention and returns ``None``. A full disk or
a locked file must never surface as a GUI error and must never cost the trader
an alert. Recording is strictly subordinate to delivering.

Import-light on purpose (no Qt, no pandas): the panel calls it on every alert
and the offline audit imports it headless.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from project_paths import get_diagnostics_dir
from review_events import alert_context_fields, get_review_installation_id

DELIVERY_EVENTS_SCHEMA = "alert_delivery_events_v1"
SUPPORTED_DELIVERY_SCHEMAS = frozenset({DELIVERY_EVENTS_SCHEMA})

#: The two Phase 1 actions. ``delivered`` is one alert reaching the feed;
#: ``watch_delivered`` is the visible delivery of a condition the trader armed,
#: which is the only one carrying a latency the sec 17 bound can be checked
#: against.
DELIVERED = "delivered"
WATCH_DELIVERED = "watch_delivered"
DELIVERY_ACTIONS = frozenset({DELIVERED, WATCH_DELIVERED})

#: Typed alert families. The duplicate metric counts repeats of the same typed
#: alert, so the family is part of identity: an M5 bounce and a D1 event on one
#: symbol are two alerts, not one repeated.
TYPE_M5_BOUNCE = "m5_bounce"
TYPE_D1_EVENT = "d1_event"
TYPE_D1_LEVEL = "d1_level"
TYPE_CHART_WATCH = "chart_watch"
TYPE_FOCUS_PICK = "focus_pick"
TYPE_ENTRY_ASSIST = "entry_assist"
TYPE_STATUS = "status"

#: Mirrors of the tag literals in ``ui.models.bounce``. Duplicated rather than
#: imported to keep this module free of the ``ui`` package (the headless audit
#: imports it); ``tests/test_alert_delivery_events.py`` asserts they still match
#: the source of truth, so the duplication cannot drift silently.
CHART_WATCH_TAG = "chart_watch"
MANUAL_CHART_TAG = "manual_chart"
AUTO_PICK_TAG = "auto_pick"
FOCUS_REVIEW_TAG = "focus_review"
FOCUS_D1_EVENT_TAG = "focus_d1_event"
ENTRY_ASSIST_TAG = "entry_assist"

_STORE_DIR_NAME = "alert_delivery_events"
_FILE_STEM = "alert-deliveries"


def delivery_store_dir() -> Path:
    """Machine-local directory holding the delivery shards.

    Derived from ``get_diagnostics_dir()`` on every call rather than captured at
    import: the diagnostics root is overridable by environment variable for
    hermetic test runs, and a module-level constant would freeze whatever value
    happened to be set when the panel first imported this module.
    """

    return get_diagnostics_dir() / _STORE_DIR_NAME


def delivery_store_path(when: date | datetime | None = None) -> Path:
    """The month-partitioned file a row for ``when`` belongs in."""

    moment = when or datetime.now()
    if isinstance(moment, datetime):
        moment = moment.date()
    return delivery_store_dir() / f"{_FILE_STEM}-{moment.strftime('%Y-%m')}.jsonl"


def _text(value: object) -> str:
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def _attr(alert: object, name: str, default: object = "") -> Any:
    """Read one attribute without trusting it.

    The alert is duck-typed, so an attribute can be a property that raises. A
    malformed alert must never cost the trader the alert itself, and this is
    the single choke point where that guarantee is enforced - every read of
    caller-supplied objects in this module goes through here.
    """

    try:
        return getattr(alert, name, default)
    except Exception:
        return default


def alert_type_for(alert: object) -> str:
    """The typed family of an alert, duck-typed from tag and D1 flag.

    Order matters. The tag is checked before ``is_d1`` because a focus-review
    or chart-watch alert can also be a D1 alert, and the surface the trader
    interacted with is the more specific identity.
    """

    tag = _text(_attr(alert, "tag")).lower()
    if tag == ENTRY_ASSIST_TAG:
        return TYPE_ENTRY_ASSIST
    if tag in {CHART_WATCH_TAG, MANUAL_CHART_TAG}:
        return TYPE_CHART_WATCH
    if tag in {AUTO_PICK_TAG, FOCUS_REVIEW_TAG}:
        return TYPE_FOCUS_PICK
    if tag == FOCUS_D1_EVENT_TAG:
        return TYPE_D1_EVENT
    if _attr(alert, "is_d1", False):
        return TYPE_D1_EVENT
    if not _text(_attr(alert, "symbol")):
        return TYPE_STATUS
    return TYPE_M5_BOUNCE


def thesis_anchor_for(alert: object) -> str:
    """The family's natural anchor - what makes this alert *this* thesis.

    Deliberately not the bar timestamp. Anchoring on time would make every
    re-fire a distinct alert, which would drive the duplicate rate to exactly
    zero and quietly render the metric meaningless.
    """

    alert_type = alert_type_for(alert)
    payload = _attr(alert, "payload", None)
    payload = payload if isinstance(payload, dict) else {}

    if alert_type == TYPE_CHART_WATCH:
        return _text(payload.get("chart_watch_kind")).lower()
    if alert_type == TYPE_D1_LEVEL:
        direction = _text(payload.get("direction")).lower()
        level = payload.get("level")
        return f"{direction}@{level}" if level is not None else direction
    if alert_type == TYPE_D1_EVENT:
        anchor = _text(payload.get("bucket") or payload.get("setup_family"))
        if anchor:
            return anchor.lower()
        return _text(_attr(alert, "raw_text")).split(":", 1)[0].lower()
    return _text(_attr(alert, "trigger")).lower()


def alert_event_id(
    alert: object,
    *,
    trade_date: str = "",
    alert_type: str = "",
    anchor: str = "",
) -> str:
    """Typed identity for one alert occurrence, stable across re-deliveries.

    A readable composite rather than a hash: these rows are read by eye in a
    JSONL when a duplicate-rate number looks wrong, and an opaque digest would
    make that debugging session considerably worse.
    """

    symbol = _text(_attr(alert, "symbol")).upper()
    side = _text(_attr(alert, "side")).upper()
    resolved_type = alert_type or alert_type_for(alert)
    resolved_anchor = anchor or thesis_anchor_for(alert)
    day = trade_date or _trade_date_text()
    return "|".join([day, symbol, side, resolved_type, resolved_anchor])


def watch_identity(trade_date: str, symbol: str, side: str, kind: str) -> str:
    """Shared identity for one armed watch, derivable from either store.

    The review log's ``watch_fired`` row carries no explicit watch id, but it
    does carry every part of this tuple. Deriving the identity instead of
    adding a field means the armed-hit metric can join the two stores without
    editing an existing emit site inside the alert panel - the smallest change
    that makes the join possible.
    """

    return "|".join(
        [
            _text(trade_date),
            _text(symbol).upper(),
            _text(side).upper(),
            _text(kind).lower(),
        ]
    )


def _trade_date_text() -> str:
    try:
        from market_session import get_market_session_window

        return get_market_session_window().market_date.isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _machine_name() -> str:
    try:
        import socket

        return socket.gethostname()
    except Exception:
        return ""


def _write_row(row: dict[str, Any], target: Path) -> dict[str, Any] | None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with local_writer_lock(lock_key_for_path(target), timeout_seconds=1.0):
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    except (OSError, LocalLockUnavailable):
        return None
    return row


def _base_row(action: str, now: datetime | None) -> dict[str, Any]:
    moment = now or datetime.now()
    return {
        "schema": DELIVERY_EVENTS_SCHEMA,
        "delivery_record_id": uuid.uuid4().hex,
        "ts": moment.isoformat(timespec="microseconds"),
        "trade_date": _trade_date_text(),
        "installation_id": get_review_installation_id(),
        "machine": _machine_name(),
        "pid": os.getpid(),
        "action": action,
    }


def record_delivery(
    alert: object,
    *,
    loud: bool,
    sounded: bool,
    is_focus: bool = False,
    queue_len: int | None = None,
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """One alert reached the feed. Returns the row, or ``None`` if unusable.

    ``loud`` must be the recorded RESULT of the panel's own
    ``alert_should_sound`` call, never a re-derivation here. A second
    implementation of the loudness rule would drift from the one that actually
    drives the beep, and the metric would then describe a desk that does not
    exist. ``sounded`` is separate because the trader can mute the feed: a loud
    alert that made no noise is a real and different event.
    """

    symbol = _text(_attr(alert, "symbol")).upper()
    if not symbol:
        return None

    try:
        context = alert_context_fields(alert)
    except Exception:
        context = {}

    row = _base_row(DELIVERED, now)
    try:
        alert_type = alert_type_for(alert)
        anchor = thesis_anchor_for(alert)
    except Exception:
        alert_type, anchor = TYPE_M5_BOUNCE, ""
    row.update(
        {
            "symbol": symbol,
            "side": _text(_attr(alert, "side")).upper(),
            "alert_type": alert_type,
            "thesis_anchor": anchor,
            "alert_event_id": alert_event_id(
                alert,
                trade_date=row["trade_date"],
                alert_type=alert_type,
                anchor=anchor,
            ),
            # Escalation inputs. Stored, never judged here - see module docstring.
            "loud": bool(loud),
            "sounded": bool(sounded),
            "is_focus": bool(is_focus),
            "tier": _text(context.get("tier")),
            "is_armed_fire": alert_type == TYPE_CHART_WATCH,
        }
    )
    if queue_len is not None:
        row["queue_len"] = max(0, int(queue_len))
    # Context last, but never over the identity/escalation fields above: a
    # duplicate key there would let a malformed alert rewrite the very fields
    # the duplicate metric is counted on.
    for key, value in context.items():
        row.setdefault(key, value)
    return _write_row(row, Path(path) if path else delivery_store_path(now))


def record_watch_delivery(
    alert: object,
    *,
    watch_id: str,
    fired_to_delivered_ms: int | None = None,
    loud: bool = True,
    sounded: bool = True,
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """A condition the trader armed fired and was visibly delivered.

    ``fired_to_delivered_ms`` is the whole point: ``watch_fired`` in the review
    log records that a watch fired, but nothing records when the trader could
    actually see it, so the sec 17 latency bound has never been checkable.
    """

    identity = _text(watch_id)
    if not identity:
        return None

    row = _base_row(WATCH_DELIVERED, now)
    try:
        alert_type = alert_type_for(alert) if alert is not None else TYPE_CHART_WATCH
    except Exception:
        alert_type = TYPE_CHART_WATCH
    row.update(
        {
            "symbol": _text(_attr(alert, "symbol")).upper(),
            "side": _text(_attr(alert, "side")).upper(),
            "alert_type": alert_type,
            "watch_id": identity,
            "loud": bool(loud),
            "sounded": bool(sounded),
            "is_armed_fire": True,
            "tier": _watch_tier(alert),
        }
    )
    if fired_to_delivered_ms is not None:
        row["fired_to_delivered_ms"] = max(0, int(fired_to_delivered_ms))
    return _write_row(row, Path(path) if path else delivery_store_path(now))


def _watch_tier(alert: object) -> str:
    if alert is None:
        return ""
    try:
        return _text((alert_context_fields(alert) or {}).get("tier"))
    except Exception:
        return ""


def delivery_sources(store_dir: Path | None = None) -> list[Path]:
    """Every readable month shard, oldest first."""

    directory = Path(store_dir) if store_dir is not None else delivery_store_dir()
    try:
        return sorted(
            candidate
            for candidate in directory.glob(f"{_FILE_STEM}-*.jsonl")
            if candidate.is_file()
        )
    except OSError:
        return []


def load_delivery_events(
    store_dir: Path | None = None,
    *,
    sources: Iterable[Path] | None = None,
) -> list[dict[str, Any]]:
    """All delivery rows, oldest first. Unreadable lines are skipped.

    A malformed line is skipped rather than raising: this store is diagnostics,
    and one truncated row from a hard shutdown must not cost the audit the
    other ten thousand.
    """

    paths = list(sources) if sources is not None else delivery_sources(store_dir)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in paths:
        try:
            lines = Path(source).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if _text(row.get("schema")) not in SUPPORTED_DELIVERY_SCHEMAS:
                continue
            record_id = _text(row.get("delivery_record_id"))
            if record_id and record_id in seen:
                continue
            if record_id:
                seen.add(record_id)
            rows.append(row)
    rows.sort(key=lambda item: _text(item.get("ts")))
    return rows
