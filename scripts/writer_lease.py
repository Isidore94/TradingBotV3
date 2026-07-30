"""Layer 3: cross-machine writer protection for shared Drive exports.

WHAT THIS IS - AND WHAT IT IS NOT (plan.md sec 4)
-------------------------------------------------
The home desk and the mini-PC both have the shared folder mounted, and both can
write the same export. This module maintains a lease file next to that export
naming the current writer, its process instance, and a monotonic fencing
generation, so a second machine can see that somebody else is publishing and
degrade honestly instead of clobbering the work.

That is **cross-machine writer protection**. It is emphatically *not*
distributed mutual exclusion and not compare-and-swap. A Google
Drive-synchronized file has no atomic test-and-set across machines: two machines
can each read "free", each write their own lease, and only find out when sync
converges. Nothing here proves clobbering is impossible. What it does is:

* make every *ambiguous* state fail closed instead of fail open, so the
  overwhelmingly common failure - a half-synced, unreadable, stale or
  old-format lease - can no longer authorize an overwrite;
* make ownership provable per *process instance*, so two processes on one
  machine (or a restarted process that inherited a PID) are never conflated;
* carry a fencing generation that is re-read from disk and checked immediately
  before *each* shared replacement, so a writer that lost ownership while
  rendering aborts. The counter has a durable high-water mark beside the lease
  (see :func:`generation_marker_path`), so it keeps increasing across a clean
  release and a process restart instead of resetting to 1.

Two limits worth stating plainly rather than burying:

* the check-then-replace window is small but real. Ownership is re-verified
  from disk immediately before each ``os.replace``, and a writer fenced off
  inside that window still completes its replacement. Closing it entirely needs
  a real compare-and-swap, which a synced file does not have;
* expiry is computed on the writer's clock and judged on the reader's. The
  supported disagreement is :data:`DEFAULT_CLOCK_SKEW_SECONDS` (120 s). A
  machine whose clock is fast by more than the lease's *remaining* TTL plus that
  grace can take over a lease that is still live by the holder's clock. Keep
  both machines on network time; the runbook has the drill.

Real same-machine exclusion is a different problem with a real solution; see
:mod:`local_writer_lock`, which this module takes around every mutation.

IDENTITY
--------
``hostname:pid:instance`` where ``instance`` is generated once per process
start. Hostname alone is never ownership - that is the classic hole this
replaces. A lease is "ours" only when it carries our process-instance id.

WHAT MAY PERMIT ACQUISITION
---------------------------
* the lease file does not exist;
* the lease is complete, valid, and expired past the bounded clock-skew window;
* the lease is complete, valid, unexpired and provably ours (renewal);
* an explicit ``takeover=True``, which is the audited emergency path and is
  never inferred from configuration or environment inside this module.

Everything else blocks: unreadable bytes, a directory in the way, invalid JSON,
a non-object, an unknown schema, a missing/blank/non-string holder, a
missing/unparseable expiry, a current-schema lease missing its instance id or
generation, and any unexpired lease held by somebody else. An old-format
``writer_lease_v1`` lease (no instance id) is never treated as ours: unexpired it
blocks until it expires or is explicitly taken over; expired it is recovered
only by writing a fresh lease through the normal acquisition path.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path

__all__ = [
    "DEFAULT_CLOCK_SKEW_SECONDS",
    "DEFAULT_TTL_MINUTES",
    "GENERATION_MARKER_SCHEMA",
    "LEASE_SCHEMA",
    "LEGACY_LEASE_SCHEMA",
    "MAX_CLOCK_SKEW_SECONDS",
    "LeaseUnavailable",
    "LeaseUnreadable",
    "acquire",
    "assert_still_owned",
    "default_holder_id",
    "holder_of",
    "inspect_lease",
    "machine_holder_id",
    "machine_name",
    "observed_clock_offset_seconds",
    "process_instance_id",
    "release",
    "renew",
    "writer_health_state",
]

LEASE_SCHEMA = "writer_lease_v2"
LEGACY_LEASE_SCHEMA = "writer_lease_v1"
GENERATION_MARKER_SCHEMA = "writer_lease_generation_v1"
DEFAULT_TTL_MINUTES = 10

#: The supported clock disagreement between the two machines. This is a
#: *grace*, not a correction: a machine whose clock is fast by more than the
#: remaining TTL plus this grace can still take over a lease that is live by the
#: holder's clock. Keep both machines inside this window (Windows time service
#: is enough); the runbook documents the drill.
DEFAULT_CLOCK_SKEW_SECONDS = 120

#: A lease may not widen its own grace without bound. Without this cap, one
#: absurd ``clock_skew_seconds`` value in a half-synced lease wedges the writer
#: slot until a human configures an emergency takeover.
MAX_CLOCK_SKEW_SECONDS = 900

#: Generated once per process start. A restarted process - even one that the OS
#: handed the same PID - gets a new one, so it can never claim the dead
#: instance's lease was its own.
_PROCESS_INSTANCE_ID = uuid.uuid4().hex

#: Highest fencing generation this process has seen per lease path. Only ever
#: used to keep the counter monotonic when the on-disk value is unreadable and
#: an audited takeover is rewriting it; the on-disk value always wins when it
#: can be read.
_GENERATION_FLOOR: dict[str, int] = {}
_GENERATION_LOCK = threading.Lock()


class LeaseUnavailable(RuntimeError):
    """This machine may not write: somebody else holds the lease."""


class LeaseUnreadable(LeaseUnavailable):
    """The lease state could not be validated, so writing is refused.

    A subclass of :class:`LeaseUnavailable` on purpose: every publish path in
    the repo already fails closed on that type, so an unverifiable lease takes
    the same safe branch as a lease held by somebody else.
    """


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
def process_instance_id() -> str:
    """Per-process-start id. Two processes never share one; a restart changes it."""
    return _PROCESS_INSTANCE_ID


def machine_name() -> str:
    return socket.gethostname()


def machine_holder_id() -> str:
    """Full writer identity: hostname + PID + process-instance id.

    Hostname alone would make two GUI processes on one PC - or a restarted
    process that inherited a recycled PID - look like a single writer that may
    renew "its own" lease. Both of those are lost-update bugs.
    """
    return f"{machine_name()}:{os.getpid()}:{_PROCESS_INSTANCE_ID[:12]}"


def default_holder_id() -> str:
    """Backwards-compatible alias for :func:`machine_holder_id`."""
    return machine_holder_id()


# ---------------------------------------------------------------------------
# reading + validating on-disk state
# ---------------------------------------------------------------------------
def _parse_ts(value) -> datetime | None:
    """Strict ISO-8601 parse. Only a string can be a timestamp."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def as_instant(value: datetime) -> datetime:
    """Normalize a datetime to naive UTC so any two of them can be compared.

    Timestamps reach this module from three places: this build (which writes an
    explicit UTC field), an older build or a hand edit (naive local wall-clock),
    and a peer build or an operator following the runbook (an ISO string with a
    trailing ``Z`` or a ``+HH:MM`` offset). Mixing those in a bare ``<=`` raises
    ``TypeError: can't compare offset-naive and offset-aware datetimes`` - which
    used to escape the publish path entirely, so the machine neither published
    nor reported a refusal. A naive value is interpreted as local wall-clock,
    which is what every previous build wrote.

    Deliberately arithmetic rather than :meth:`datetime.astimezone`, which the
    Windows CRT refuses for extreme years. A lease claiming to expire in 9999
    must *block* (it does - see the year-3000 case in the tests), not raise an
    unhandled ``OSError`` out of the publish path.
    """
    if value.tzinfo is not None and value.utcoffset() is not None:
        offset = value.utcoffset() or timedelta(0)
        naive = value.replace(tzinfo=None)
    else:
        offset = _local_utc_offset(value)
        naive = value
    try:
        return naive - offset
    except (OverflowError, OSError, ValueError):  # pragma: no cover - year 9999 edges
        return datetime.max if naive > datetime(2000, 1, 1) else datetime.min


def _local_utc_offset(value: datetime) -> timedelta:
    """This machine's UTC offset at ``value``, falling back to right now."""
    for candidate in (value, datetime.now()):
        try:
            offset = candidate.astimezone().utcoffset()
        except (OverflowError, OSError, ValueError):
            continue
        if offset is not None:
            return offset
    return timedelta(0)  # pragma: no cover - astimezone always works for "now"


def _instant_of(payload: dict, utc_key: str, local_key: str) -> datetime | None:
    """Prefer the explicit UTC field; fall back to the naive local one.

    The UTC field is what makes the comparison survive a DST transition: local
    wall-clock jumps an hour twice a year, so a lease written at 01:59 and read
    at 03:01 looks either an hour expired or an hour fresher than it is.
    """
    instant = _parse_ts(payload.get(utc_key))
    if instant is not None:
        return as_instant(instant)
    local = _parse_ts(payload.get(local_key))
    return None if local is None else as_instant(local)


def _valid_holder(value) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _valid_skew(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_CLOCK_SKEW_SECONDS
    return max(0, min(MAX_CLOCK_SKEW_SECONDS, int(value)))


def inspect_lease(lease_path: Path | str) -> dict:
    """Classify the lease on disk, or raise :class:`LeaseUnreadable`.

    ``FileNotFoundError`` - and only that - is "no lease exists"; it returns
    ``{"kind": "missing"}``. Every other read, decode, parse or validation
    failure raises, because a lease this machine cannot understand is not a free
    slot: it may be another machine's live claim, arriving half-synced.
    """
    lease_path = Path(lease_path)
    name = lease_path.name
    try:
        raw = lease_path.read_bytes()
    except FileNotFoundError:
        return {"kind": "missing", "path": lease_path}
    except OSError as exc:
        raise LeaseUnreadable(
            f"lease {name} could not be read ({type(exc).__name__}: {exc}); refusing to "
            "assume it is free"
        ) from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LeaseUnreadable(
            f"lease {name} is not valid UTF-8 ({exc}); refusing to assume it is free"
        ) from exc
    if not text.strip():
        raise LeaseUnreadable(
            f"lease {name} is empty or whitespace-only (a half-written or half-synced "
            "file); refusing to assume it is free"
        )
    try:
        payload = json.loads(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise LeaseUnreadable(
            f"lease {name} is not valid JSON ({exc}); refusing to assume it is free"
        ) from exc
    if not isinstance(payload, dict):
        raise LeaseUnreadable(
            f"lease {name} is a {type(payload).__name__}, not a lease object; refusing to "
            "assume it is free"
        )

    schema = payload.get("schema")
    if schema == LEASE_SCHEMA:
        kind = "current"
    elif schema == LEGACY_LEASE_SCHEMA:
        kind = "legacy"
    else:
        raise LeaseUnreadable(
            f"lease {name} declares schema {schema!r}, which this build cannot validate; "
            "refusing to assume it is free"
        )

    holder = _valid_holder(payload.get("holder"))
    if holder is None:
        raise LeaseUnreadable(
            f"lease {name} has no usable holder ({payload.get('holder')!r}); refusing to "
            "assume it is free"
        )
    expires = _parse_ts(payload.get("expires_at"))
    if expires is None:
        raise LeaseUnreadable(
            f"lease {name} has no usable expiry ({payload.get('expires_at')!r}); refusing "
            "to assume it is free"
        )
    expires_instant = _instant_of(payload, "expires_at_utc", "expires_at")
    if expires_instant is None:  # pragma: no cover - expires parsed above
        raise LeaseUnreadable(
            f"lease {name} has no usable expiry instant; refusing to assume it is free"
        )

    state = {
        "kind": kind,
        "path": lease_path,
        "payload": payload,
        "holder": holder,
        "expires_at": expires,
        "expires_instant": expires_instant,
        "acquired_at": _parse_ts(payload.get("acquired_at")),
        "acquired_instant": _instant_of(payload, "acquired_at_utc", "acquired_at"),
        "clock_skew_seconds": _valid_skew(payload.get("clock_skew_seconds")),
        "instance_id": "",
        "generation": None,
    }

    if kind == "legacy":
        # writer_lease_v1: hostname-only holder, no process instance, no fencing.
        # Valid enough to *block* on, never enough to call ours.
        return state

    instance_id = _valid_holder(payload.get("instance_id"))
    if instance_id is None:
        raise LeaseUnreadable(
            f"lease {name} claims schema {LEASE_SCHEMA} but carries no process instance id; "
            "the state is ambiguous, so writing is refused"
        )
    generation = payload.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise LeaseUnreadable(
            f"lease {name} claims schema {LEASE_SCHEMA} but its fencing generation is "
            f"{generation!r}; the state is ambiguous, so writing is refused"
        )
    state["instance_id"] = instance_id
    state["generation"] = generation
    return state


def _is_live(state: dict, moment: datetime, clock_skew_seconds: int | None = None) -> bool:
    """Is this lease still inside its expiry + bounded clock-skew window?

    The boundary is inclusive: at exactly ``expires_at + skew`` the lease still
    holds. A one-second rounding difference between two machines must never be
    what decides who owns the shared report.

    Both sides are anchored to a real instant first (see :func:`as_instant`), so
    a timezone-aware expiry, a naive one, and a DST boundary all compare
    correctly instead of raising ``TypeError`` out of the publish path.
    """
    if clock_skew_seconds is None:
        skew = state["clock_skew_seconds"]
    else:
        skew = max(0, min(MAX_CLOCK_SKEW_SECONDS, int(clock_skew_seconds)))
    expires = state.get("expires_instant") or as_instant(state["expires_at"])
    try:
        deadline = expires + timedelta(seconds=skew)
    except OverflowError:  # pragma: no cover - an expiry at datetime.max
        return True
    return deadline >= as_instant(moment)


def observed_clock_offset_seconds(state: dict, moment: datetime | None = None) -> float | None:
    """How far *ahead of us* the writer that stamped this lease appears to be.

    Only the "the other machine's clock is ahead" direction is observable from a
    lease: a stamp in our future can only be skew. The opposite direction - our
    own clock running fast - is indistinguishable from an old lease, which is
    exactly why this is reported for the operator rather than corrected for. It
    feeds Layer 5 telemetry so the plan.md sec 6.2 clock-comparison drill has a
    number to read instead of a guess.
    """
    acquired = state.get("acquired_instant")
    if acquired is None:
        return None
    return round((acquired - as_instant(moment or datetime.now())).total_seconds(), 3)


def _is_ours(state: dict, holder: str) -> bool:
    """Can THIS process instance prove the lease is its own?

    Two conditions, both required:

    1. the lease carries our per-process-start instance id - so another process
       on this hostname, and a restarted process with a recycled PID, are both
       excluded;
    2. the lease is labelled either with the exact holder we are asking as, or
       with this process's own derived machine identity - the latter being the
       case where we are relabelling or handing over a lease we ourselves wrote.

    A lease written by this process under some *other* explicit label (another
    logical writer sharing the interpreter) is deliberately not ours.
    """
    if state.get("kind") != "current":
        return False
    if state.get("instance_id") != _PROCESS_INSTANCE_ID:
        return False
    return state.get("holder") == holder or state.get("holder") == machine_holder_id()


def generation_marker_path(lease_path: Path | str) -> Path:
    """Durable high-water mark for the fencing generation, beside the lease.

    ``release`` deletes the lease file, and that used to be the *only* durable
    copy of the counter - so the next process started again at generation 1 and
    a generation-1 publication could be strictly newer than a generation-3 one.
    A marker that outlives the lease keeps the number usable for ordering two
    publications after the fact. It is never a claim of ownership: it carries no
    holder and grants nothing.
    """
    lease_path = Path(lease_path)
    return lease_path.with_name(lease_path.name + ".generation")


def _durable_generation_floor(lease_path: Path) -> int:
    try:
        payload = json.loads(generation_marker_path(lease_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(payload, dict):
        return 0
    value = payload.get("generation")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _generation_floor(lease_path: Path) -> int:
    with _GENERATION_LOCK:
        in_process = _GENERATION_FLOOR.get(str(lease_path), 0)
    return max(in_process, _durable_generation_floor(lease_path))


def _remember_generation(lease_path: Path, generation: int) -> None:
    with _GENERATION_LOCK:
        key = str(lease_path)
        _GENERATION_FLOOR[key] = max(_GENERATION_FLOOR.get(key, 0), int(generation))
    if int(generation) <= _durable_generation_floor(lease_path):
        return
    try:
        from diagnostics.artifact_io import atomic_write_json

        atomic_write_json(
            generation_marker_path(lease_path),
            {
                "schema": GENERATION_MARKER_SCHEMA,
                "generation": int(generation),
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": (
                    "high-water mark only; this file grants nothing and names no holder"
                ),
            },
        )
    except Exception:
        # A missing marker only costs ordering information after a restart; it
        # can never authorize a write, so it must not fail a publication.
        pass


def _write(lease_path: Path, payload: dict) -> None:
    """Atomic replace with guaranteed temp cleanup on every failure path."""
    from diagnostics.artifact_io import atomic_write_json

    atomic_write_json(lease_path, payload)


def _append_takeover_audit(lease_path: Path, record: dict) -> None:
    """Durable, externally visible record that somebody broke a live lease.

    Written beside the lease (so it travels with the shared export) and never
    rewritten by the next renewal - an audit trail a renewal can erase is not an
    audit trail.

    The append happens **before** the lease is replaced, and a failure to write
    it aborts the takeover. The design calls for an *auditable* emergency
    takeover; a takeover that proceeds after its audit record silently failed is
    not auditable, so this fails closed. Over-recording is the safe direction:
    an audit line whose lease write then failed says a takeover was attempted,
    which is true.
    """
    try:
        from diagnostics.artifact_io import append_jsonl

        append_jsonl(lease_path.parent / f"{lease_path.name}.takeover_audit.jsonl", record)
    except Exception as exc:
        raise LeaseUnavailable(
            f"the emergency takeover of lease {lease_path.name} was abandoned because its "
            f"audit record could not be written ({type(exc).__name__}: {exc}); an "
            "unauditable takeover is refused"
        ) from exc


# ---------------------------------------------------------------------------
# acquire / renew / release
# ---------------------------------------------------------------------------
def acquire(
    lease_path: Path | str,
    *,
    holder: str | None = None,
    ttl_minutes: int = DEFAULT_TTL_MINUTES,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
    now: datetime | None = None,
    takeover: bool = False,
    reason: str = "",
) -> dict:
    """Acquire or renew the writer lease, or raise :class:`LeaseUnavailable`.

    ``takeover=True`` is the explicit, audited emergency path and is the only
    thing that overrides a live or unverifiable lease. It is never inferred from
    an environment variable or a configuration file here: an ambient variable
    must not be able to break a live lease, so the decision is made by the
    caller (see :mod:`writer_role`) and passed in.
    """
    lease_path = Path(lease_path)
    holder = holder or machine_holder_id()
    moment = now or datetime.now()
    try:
        with local_writer_lock(lock_key_for_path(lease_path)):
            return _acquire_locked(
                lease_path,
                holder=holder,
                ttl_minutes=ttl_minutes,
                clock_skew_seconds=clock_skew_seconds,
                moment=moment,
                takeover=bool(takeover),
                reason=reason,
            )
    except LocalLockUnavailable as exc:
        raise LeaseUnavailable(
            f"lease {lease_path.name} was not attempted: {exc}"
        ) from exc


def _acquire_locked(
    lease_path: Path,
    *,
    holder: str,
    ttl_minutes: int,
    clock_skew_seconds: int,
    moment: datetime,
    takeover: bool,
    reason: str,
    hold_generation: bool = False,
) -> dict:
    displaced: str | None = None
    carried_previous = ""
    renewed_at = ""
    try:
        state = inspect_lease(lease_path)
    except LeaseUnreadable:
        if not takeover:
            raise
        # An audited takeover is the one defined way out of a lease this build
        # cannot validate (a corrupt file, an unknown schema, a lease claiming
        # to run until the year 3000).
        state = {"kind": "unverifiable", "path": lease_path}

    kind = state["kind"]
    if kind == "missing":
        generation = max(1, _generation_floor(lease_path) + 1)
    elif kind == "unverifiable":
        generation = _generation_floor(lease_path) + 1
        displaced = "(unverifiable lease state)"
    elif kind == "legacy":
        live = _is_live(state, moment, clock_skew_seconds)
        if live and not takeover:
            raise LeaseUnavailable(
                f"lease {lease_path.name} is held by {state['holder']} until "
                f"{state['payload'].get('expires_at')} under the old {LEGACY_LEASE_SCHEMA} "
                "format, which carries no process instance id. It cannot be proved to be "
                "this process's lease, so it is left alone until it expires or is "
                "explicitly taken over."
            )
        if live and takeover:
            displaced = state["holder"]
        generation = max(1, _generation_floor(lease_path) + 1)
    else:  # current schema
        ours = _is_ours(state, holder)
        live = _is_live(state, moment, clock_skew_seconds)
        if live and not ours and not takeover:
            raise LeaseUnavailable(
                f"lease {lease_path.name} is held by {state['holder']} until "
                f"{state['payload'].get('expires_at')} (generation {state['generation']}); "
                "this machine must not overwrite the shared report while that lease stands"
            )
        if live and not ours and takeover:
            displaced = state["holder"]
        if ours and hold_generation:
            # A renewal by the same process instance must NOT fence itself off.
            # ``assert_still_owned`` compares the generation by equality, so a
            # renewal landing between a writer's acquisition and its
            # pre-replacement check used to abort that writer's own publish.
            generation = int(state["generation"])
        else:
            generation = int(state["generation"]) + 1
        if ours:
            carried_previous = str(state["payload"].get("previous_holder") or "")
            if hold_generation:
                renewed_at = moment.isoformat(timespec="seconds")

    if not (hold_generation and renewed_at):
        generation = max(generation, _generation_floor(lease_path) + 1)
    ttl = max(1, int(ttl_minutes))
    skew = max(0, min(MAX_CLOCK_SKEW_SECONDS, int(clock_skew_seconds)))
    expires = moment + timedelta(minutes=ttl)
    payload = {
        "schema": LEASE_SCHEMA,
        "holder": holder,
        "machine": machine_name(),
        "pid": os.getpid(),
        "instance_id": _PROCESS_INSTANCE_ID,
        "generation": generation,
        "acquired_at": moment.isoformat(timespec="seconds"),
        "expires_at": expires.isoformat(timespec="seconds"),
        # UTC anchors alongside the local wall-clock fields above. The naive
        # fields stay for older readers; the comparison uses these, so a DST
        # transition or a machine in a different timezone cannot invent an hour
        # of apparent clock skew.
        "acquired_at_utc": as_instant(moment).replace(tzinfo=timezone.utc).isoformat(timespec="seconds"),
        "expires_at_utc": as_instant(expires).replace(tzinfo=timezone.utc).isoformat(timespec="seconds"),
        "renewed_at": renewed_at,
        "ttl_minutes": ttl,
        "takeover": bool(takeover and displaced is not None),
        "clock_skew_seconds": skew,
        "previous_holder": displaced or carried_previous,
        "takeover_reason": str(reason or "") if displaced is not None else "",
    }
    if payload["takeover"]:
        # Audit first: an unauditable takeover raises here and the live lease on
        # disk is left exactly as it was.
        _append_takeover_audit(
            lease_path,
            {
                "schema": "writer_lease_takeover_v1",
                "at": datetime.now().isoformat(timespec="seconds"),
                "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "lease": lease_path.name,
                "displaced_holder": displaced,
                "new_holder": holder,
                "machine": machine_name(),
                "pid": os.getpid(),
                "instance_id": _PROCESS_INSTANCE_ID,
                "generation": generation,
                "reason": str(reason or ""),
                "note": (
                    "explicit emergency takeover of a live or unverifiable writer lease"
                ),
            },
        )
    _write(lease_path, payload)
    _remember_generation(lease_path, generation)
    return payload


def renew(
    lease_path: Path | str,
    *,
    holder: str | None = None,
    ttl_minutes: int = DEFAULT_TTL_MINUTES,
    now: datetime | None = None,
) -> dict | None:
    """Extend a lease this process already owns; ``None`` if it no longer does.

    Bounded renewal, deliberately: renewing only ever extends the window by one
    TTL from *now*, and it fails (rather than re-acquiring) when ownership has
    been lost, so a woken machine cannot ride a lease it stopped holding while
    it was asleep.

    A renewal keeps the fencing generation it already holds. Advancing it would
    fence off the renewing writer's own in-flight publication, because
    :func:`assert_still_owned` compares the generation by equality. The
    generation orders *changes of ownership*, which a renewal is not.
    """
    lease_path = Path(lease_path)
    holder = holder or machine_holder_id()
    moment = now or datetime.now()
    try:
        with local_writer_lock(lock_key_for_path(lease_path)):
            try:
                state = inspect_lease(lease_path)
            except LeaseUnreadable:
                return None
            if state["kind"] != "current" or not _is_ours(state, holder):
                return None
            if not _is_live(state, moment):
                return None
            return _acquire_locked(
                lease_path,
                holder=holder,
                ttl_minutes=ttl_minutes,
                clock_skew_seconds=state["clock_skew_seconds"],
                moment=moment,
                takeover=False,
                reason="",
                hold_generation=True,
            )
    except LocalLockUnavailable:
        return None


def release(lease_path: Path | str, *, holder: str | None = None) -> bool:
    """Drop a lease this process owns. Never drops anybody else's.

    Wired into GUI shutdown so a clean exit hands the writer slot back
    immediately. A hard kill, where this never runs, is covered by expiry - the
    lease TTL is the backstop, so a killed process can never wedge the writer
    slot permanently.
    """
    lease_path = Path(lease_path)
    holder = holder or machine_holder_id()
    try:
        with local_writer_lock(lock_key_for_path(lease_path)):
            try:
                state = inspect_lease(lease_path)
            except LeaseUnreadable:
                # Questionable state is never destroyed on the way out.
                return False
            if state["kind"] == "missing":
                return True
            if not _is_ours(state, holder):
                return False
            try:
                lease_path.unlink(missing_ok=True)
                return True
            except OSError:
                return False
    except LocalLockUnavailable:
        return False


def holder_of(
    lease_path: Path | str,
    *,
    now: datetime | None = None,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
) -> str | None:
    """Current unexpired holder, or ``None`` when the lease genuinely is free.

    Raises :class:`LeaseUnreadable` when the state cannot be validated. That is
    the honest answer: summarizing an unreadable or half-synced lease as
    "nobody holds it" is how a UI ends up inviting the second machine to
    publish over the first.
    """
    state = inspect_lease(lease_path)
    if state["kind"] == "missing":
        return None
    if not _is_live(state, now or datetime.now(), clock_skew_seconds):
        return None
    return state["holder"]


def assert_still_owned(
    lease_path: Path | str,
    *,
    holder: str,
    generation: int | None = None,
    instance_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Re-verify ownership immediately before replacing shared output.

    Fencing is *enforced* here, not merely recorded: if the generation on disk
    has moved past the one this writer acquired, another instance has fenced us
    off and the replacement must not happen. Anything unverifiable, expired,
    vanished, or belonging to somebody else raises, and the caller aborts with
    the previous verified publication untouched.
    """
    lease_path = Path(lease_path)
    expected_instance = instance_id or _PROCESS_INSTANCE_ID
    state = inspect_lease(lease_path)  # raises LeaseUnreadable on ambiguity
    if state["kind"] == "missing":
        raise LeaseUnavailable(
            f"lease {lease_path.name} disappeared between acquisition and publication; "
            "ownership can no longer be proved"
        )
    if state["kind"] != "current":
        raise LeaseUnreadable(
            f"lease {lease_path.name} was replaced by an old-format lease mid-publish; "
            "ownership can no longer be proved"
        )
    if state["instance_id"] != expected_instance:
        raise LeaseUnavailable(
            f"lease {lease_path.name} is now held by another process instance "
            f"({state['holder']}, instance {state['instance_id']}); this writer was fenced "
            "off while it was rendering"
        )
    if state["holder"] != holder:
        raise LeaseUnavailable(
            f"lease {lease_path.name} is now held by {state['holder']}, not {holder}; "
            "ownership changed between acquisition and publication"
        )
    if generation is not None and int(state["generation"]) != int(generation):
        raise LeaseUnavailable(
            f"lease {lease_path.name} advanced to fencing generation {state['generation']} "
            f"while this writer held generation {generation}; the replacement is refused"
        )
    if not _is_live(state, now or datetime.now()):
        raise LeaseUnavailable(
            f"lease {lease_path.name} expired before the replacement could be made; "
            "ownership must be re-established first"
        )
    return state


# ---------------------------------------------------------------------------
# Layer 5 health surface
# ---------------------------------------------------------------------------
def writer_health_state() -> dict:
    """Read the machine-local writer health artifact (never optimistic).

    Exposed here so every surface has one import for "what is this machine's
    writer status". Missing or corrupt telemetry reports ``healthy=False``;
    it is never rendered as a healthy default.
    """
    from writer_health import read_writer_health

    return read_writer_health()
