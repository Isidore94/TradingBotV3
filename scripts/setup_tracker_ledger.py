"""Point-in-time transition ledger for the D1 setup tracker — R10.D (S1–S4).

The tracker is a **snapshot**: one 951 MB JSON that says what every setup looks
like right now. Audit S1 measured what that costs. Between one frozen pair of
payloads, 218 setups changed status, 2,737 CLOSED scenarios changed status or
reason, 1,306 changed their exit date, and 2,618 had their `events` dropped
while status and R stayed identical — including AMCR LONG on 2026-07-28 going
`TIME_STOP @ 46.69, R 0.577` → `TARGET_HIT @ 45.55, R 0.360` on the *same date*.

None of that is visible in the snapshot, because a snapshot has no memory. Ask
it "when did this setup close, and did it ever reopen?" and it can only tell
you what is true now. So the transitions become their own append-only stream,
and the snapshot goes back to being what it is: a convenience that can be
rebuilt, sitting beside an authority that cannot.

**Never by deep-copying the payload.** The obvious implementation — keep the
previous payload and diff the two — means holding two 951 MB dicts. Instead a
tiny sidecar holds one digest per setup (id → short hash of the fields that
define its state), so the diff is a dict comparison over ~10k short strings and
the payload is read exactly once, in the caller's own hands, never copied.

Four event types, and the distinction between the last two is the point:

``initial``     first time this setup is seen at all.
``transition``  a state-bearing field changed.
``reopened``    a setup that had been CLOSED or UNTRADEABLE is OPEN again -
                a transition, but the one worth naming, because S1 measured 35
                CLOSED→OPEN and 1 UNTRADEABLE→OPEN in a single pair.
``tombstone``   the setup left the payload. It is NOT a closure and must never
                be read as one: a setup can vanish because it closed, because
                the tracker pruned it, or because a partial read lost it. The
                row says the setup left and says nothing about why.

Pure and IO-light: digests and diffs here, the ledger write in the caller.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Schema NAME (ground rule 5). A changed meaning is a new name.
SCHEMA_SETUP_TRACKER_EVENT = "setup_tracker_event_v1"

#: Stream name; the ledger segments it by month on its own.
STREAM = "setup_tracker_events"

EVENT_INITIAL = "initial"
EVENT_TRANSITION = "transition"
EVENT_REOPENED = "reopened"
EVENT_TOMBSTONE = "tombstone"

#: Statuses a setup can be `reopened` FROM.
_TERMINAL_STATUSES = frozenset({"CLOSED", "UNTRADEABLE"})

#: The fields whose change constitutes a transition.
#:
#: Deliberately narrow. The payload carries hundreds of fields per setup and
#: most of them move every run - a price, a band, a note - so digesting the
#: whole record would emit a transition for every setup on every run and the
#: stream would say nothing. These are the fields that describe what the setup
#: IS and what happened to it.
STATE_FIELDS: tuple[str, ...] = (
    "setup_status",
    "setup_family",
    "priority_bucket",
    "favorite_zone",
    "side",
    "anchor_date",
    "exit_date",
    "exit_reason",
    "outcome_r",
)

#: Sidecar identity, so a future change of digest inputs is detectable rather
#: than silently producing a run of false transitions.
SIDECAR_SCHEMA = "setup_tracker_digest_sidecar_v1"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value)


def setup_state(setup: Mapping[str, Any]) -> dict[str, str]:
    """The state-bearing fields of one setup, as text."""
    return {field: _text(setup.get(field)) for field in STATE_FIELDS}


def setup_digest(setup: Mapping[str, Any]) -> str:
    """A short, stable digest of one setup's state.

    Short on purpose: the sidecar holds one per setup and is read and written
    on every tracker save, so 16 hex characters over ~10k setups is a ~400 KB
    file rather than a second copy of the tracker. A 64-bit collision across
    10k items is not a risk worth a bigger file.
    """
    payload = json.dumps(setup_state(setup), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_sidecar(
    setups: Mapping[str, Mapping[str, Any]], *, data_session: str = ""
) -> dict[str, Any]:
    """The digest sidecar for a payload's setups."""
    return {
        "schema": SIDECAR_SCHEMA,
        "state_fields": list(STATE_FIELDS),
        "data_session": _text(data_session),
        "digests": {
            str(setup_id): setup_digest(setup or {})
            for setup_id, setup in (setups or {}).items()
        },
        "statuses": {
            str(setup_id): _text((setup or {}).get("setup_status"))
            for setup_id, setup in (setups or {}).items()
        },
    }


def load_sidecar(path: Path | str) -> dict[str, Any]:
    """The previous sidecar, or an empty one. Never raises.

    An unreadable sidecar yields an empty one, which makes the next run emit
    `initial` for everything. That is loud and recoverable; guessing at a
    partial sidecar would emit a wave of false transitions instead, which is
    neither.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SIDECAR_SCHEMA, "digests": {}, "statuses": {}, "state_fields": []}
    if not isinstance(payload, Mapping):
        return {"schema": SIDECAR_SCHEMA, "digests": {}, "statuses": {}, "state_fields": []}
    return {
        "schema": _text(payload.get("schema")),
        "state_fields": list(payload.get("state_fields") or []),
        "data_session": _text(payload.get("data_session")),
        "digests": dict(payload.get("digests") or {}),
        "statuses": dict(payload.get("statuses") or {}),
    }


def save_sidecar(path: Path | str, sidecar: Mapping[str, Any]) -> Path:
    """Write the sidecar atomically beside the tracker."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    import os

    os.replace(tmp, target)
    return target


def diff_setups(
    setups: Mapping[str, Mapping[str, Any]],
    previous: Mapping[str, Any],
    *,
    data_session: str = "",
) -> list[dict[str, Any]]:
    """Events describing what changed since `previous`.

    The payload is read once, in place. Nothing here copies it.

    A sidecar whose `state_fields` differ from this module's is treated as
    absent for the transition question: the digests were computed over a
    different definition, so comparing them would emit a transition for every
    setup and mean nothing. Everything reads `initial` instead, and the reason
    is recorded on the rows.
    """
    prior_digests = dict(previous.get("digests") or {})
    prior_statuses = dict(previous.get("statuses") or {})
    prior_fields = list(previous.get("state_fields") or [])
    fields_changed = bool(prior_digests) and prior_fields != list(STATE_FIELDS)
    if fields_changed:
        prior_digests = {}
        prior_statuses = {}

    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for setup_id, setup in (setups or {}).items():
        key = str(setup_id)
        seen.add(key)
        record = setup or {}
        digest = setup_digest(record)
        state = setup_state(record)
        status = state.get("setup_status", "")
        if key not in prior_digests:
            events.append(
                _event(
                    EVENT_INITIAL,
                    key,
                    record,
                    state,
                    data_session,
                    note=(
                        "the digest sidecar was written over different state fields, "
                        "so no prior state is comparable"
                        if fields_changed
                        else ""
                    ),
                )
            )
            continue
        if prior_digests[key] == digest:
            continue
        prior_status = _text(prior_statuses.get(key))
        kind = (
            EVENT_REOPENED
            if prior_status in _TERMINAL_STATUSES and status == "OPEN"
            else EVENT_TRANSITION
        )
        events.append(
            _event(kind, key, record, state, data_session, previous_status=prior_status)
        )

    for setup_id in prior_digests:
        if setup_id in seen:
            continue
        events.append(
            {
                "event_type": EVENT_TOMBSTONE,
                "setup_id": str(setup_id),
                "previous_status": _text(prior_statuses.get(setup_id)),
                "data_session": _text(data_session),
                # Load-bearing: a tombstone is NOT a closure. A setup can leave
                # because it closed, because the tracker pruned it, or because a
                # partial read lost it, and this row cannot tell those apart.
                "note": (
                    "the setup left the payload; this says nothing about why, and "
                    "must not be read as a closure"
                ),
            }
        )
    return events


def _event(
    kind: str,
    setup_id: str,
    record: Mapping[str, Any],
    state: Mapping[str, str],
    data_session: str,
    *,
    previous_status: str = "",
    note: str = "",
) -> dict[str, Any]:
    event = {
        "event_type": kind,
        "setup_id": setup_id,
        "symbol": _text(record.get("symbol")),
        "side": _text(record.get("side")),
        "scan_date": _text(record.get("scan_date")),
        "data_session": _text(data_session),
        **{f"state_{field}": value for field, value in state.items()},
    }
    if previous_status:
        event["previous_status"] = previous_status
    if note:
        event["note"] = note
    return event


# ---------------------------------------------------------------------------
# completed sessions only (S2)
# ---------------------------------------------------------------------------
def _mark_dates(setup: Mapping[str, Any]) -> Iterable[str]:
    snapshot = setup.get("latest_snapshot")
    if isinstance(snapshot, Mapping):
        stamp = _text(snapshot.get("trade_date"))
        if stamp:
            yield stamp
    for mark in setup.get("daily_marks") or ():
        if isinstance(mark, Mapping):
            stamp = _text(mark.get("trade_date"))
            if stamp:
                yield stamp


def forming_bar_marks(
    setups: Mapping[str, Mapping[str, Any]], data_session: str
) -> dict[str, Any]:
    """Marks dated later than the run's own data vintage (audit S2).

    A tracker run during a session marks the FORMING bar, so a setup carries a
    close that does not exist yet and a scenario can exit on it. The audit
    measured 2,739 such setups on its frozen POST payload (`data_session`
    2026-08-20, marks dated 08-21) and 2,834 on the PRE pair.

    **It does not reproduce on every payload**, and that is the nature of the
    defect rather than a refutation: a payload written after the close, or on a
    Monday reflecting Friday, has nothing forming to mark. Measured on the
    2026-08-24 payload (`data_session` 2026-08-21): 14,043 marks, **zero**
    later than the vintage.

    Counted, named and reported - never silently dropped, and never repaired
    here. Rewriting a mark would be rewriting history (ground rule 5).
    """
    vintage = _text(data_session)
    offenders: dict[str, str] = {}
    marks_seen = 0
    for setup_id, setup in (setups or {}).items():
        for stamp in _mark_dates(setup or {}):
            marks_seen += 1
            if vintage and stamp > vintage:
                current = offenders.get(str(setup_id), "")
                if stamp > current:
                    offenders[str(setup_id)] = stamp
    return {
        "data_session": vintage,
        "marks_seen": marks_seen,
        "setups_with_later_marks": len(offenders),
        "latest_offending_mark": max(offenders.values()) if offenders else "",
        "sample": sorted(offenders)[:10],
        "measured": bool(vintage),
        "note": (
            "a mark dated after the run's data vintage is the FORMING bar: the "
            "close it carries does not exist yet"
            if offenders
            else "no mark is dated after the run's data vintage"
        )
        if vintage
        else "the payload declares no data_session, so this is UNMEASURED",
    }


# ---------------------------------------------------------------------------
# exchange sessions (S3a)
# ---------------------------------------------------------------------------
def _as_date(value: Any) -> date | None:
    text = _text(value)[:10]
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def sessions_between(start: Any, end: Any, calendar: Iterable[date] | None = None) -> int | None:
    """Exchange sessions from `start` to `end`, or None if unmeasurable.

    With a calendar, sessions are counted from it. Without one, business days
    are the fallback and the caller is told which was used - a business-day
    count over a week containing a holiday is close but not exact, and
    presenting it as an exchange-session count would be a number nobody
    measured.
    """
    first, last = _as_date(start), _as_date(end)
    if first is None or last is None:
        return None
    if calendar is not None:
        days = sorted({day for day in calendar if first < day <= last})
        return len(days)
    import numpy as np

    return int(np.busday_count(first, last))


def horizon_drift(
    scan_date: Any, future_scan_date: Any, declared_horizon: Any, *, tolerance: float = 2.0
) -> dict[str, Any]:
    """Does this observation's declared horizon match the sessions it spans?

    Audit S3a, root cause: the future row is chosen as `idx + horizon` into the
    SYMBOL'S OWN scan rows, not into exchange sessions. A symbol that appears on
    a watchlist irregularly therefore has "5 sessions later" land far away.
    Measured over the live file (10,928 rows): horizon 1 → median 1 session,
    horizon 3 → 5, **horizon 5 → 64**, **horizon 10 → 73**, and 42% of rows
    span more than twice their declared horizon.

    This MEASURES and FLAGS; it does not re-select the future row. Changing
    which row is compared would silently redefine every number the tracker has
    ever produced, which is a scoring change and not this packet's to make.
    """
    spanned = sessions_between(scan_date, future_scan_date)
    try:
        declared = int(declared_horizon)
    except (TypeError, ValueError):
        declared = 0
    if spanned is None or declared <= 0:
        return {
            "sessions_spanned": None,
            "stale_horizon": None,
            "basis": "unmeasured: a date could not be read or the horizon is not positive",
        }
    return {
        "sessions_spanned": spanned,
        "stale_horizon": bool(spanned > declared * float(tolerance)),
        "basis": (
            f"business days between the two scan dates; flagged when the span "
            f"exceeds {tolerance:g}x the declared horizon"
        ),
    }
