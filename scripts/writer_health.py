"""Layer 5: the one health artifact that describes shared-writer reality.

OFF / DESK / AWAY must accurately describe what work and publishing are active
(plan.md sec 5), and a Health panel must never render "fine" for a machine that
has silently stopped publishing. This module owns a single, atomically written,
machine-local artifact - ``<diagnostics>/writer_health.json`` - carrying every
Layer 5 field:

* the configured designated writer, this machine's name and its configured role;
* hostname / PID / per-process-start instance id (the full lease identity);
* the machine-local exclusion state (which primitives were actually held, and
  whether the previous owner died holding one);
* lease holder + instance, acquired / renewed / expires timestamps, and the
  fencing generation;
* the last successful renewal, and the last ownership or configuration failure;
* the read-only reason, if this machine is read-only;
* the emergency-override state and its expiry;
* the owner and generation of the last verified publication.

ABSENT, CORRUPT OR STALE NEVER READS AS HEALTHY
-----------------------------------------------
:func:`read_writer_health` is the reader for every surface. A missing file, a
half-written file, a file that is not a JSON object, one carrying an
unrecognized schema, and one whose ``written_at`` is older than
:data:`MAX_TELEMETRY_AGE_MINUTES` (or unparseable) all return ``healthy=False``
with a status that says which of those it was. It never falls back to a
cheerful default, and it never re-uses a stale in-memory copy: the artifact on
disk is the answer.

The last-verified-publication and last-failure records are *sticky*
(:data:`STICKY_FIELDS`): a refusal does not erase the record of the last good
publication, and a success does not erase the record of the last failure. Only
those history fields carry forward - everything describing the current attempt
is rewritten every time.

The artifact is machine-local (under the diagnostics root, not the shared Drive
folder) precisely because it describes *this* machine's view. Two machines each
keep their own.
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

__all__ = [
    "HEALTH_FILENAME",
    "HEALTH_SCHEMA",
    "MAX_TELEMETRY_AGE_MINUTES",
    "STICKY_FIELDS",
    "health_artifact_path",
    "read_writer_health",
    "write_writer_health",
]

HEALTH_SCHEMA = "writer_health_v1"
HEALTH_FILENAME = "writer_health.json"

#: Older than this and the artifact describes a machine that has stopped
#: reporting, whatever it last said. The Away cadence is hourly, so two missed
#: cycles plus slack is the threshold: enough that a normal hourly gap is not
#: noise, short enough that "stopped publishing at lunchtime" is visible the
#: same afternoon.
MAX_TELEMETRY_AGE_MINUTES = 150

#: Fields that describe the *last time something happened* rather than the
#: current attempt. A refusal must not erase the record of the last good
#: publication, and a success must not erase the record of the last failure -
#: the Layer 5 field list requires one artifact carrying both at once. On the
#: read-only secondary, which refuses on every cycle by design, the blank
#: template used to make the artifact claim the machine had never published.
STICKY_FIELDS = (
    "last_verified_publication",
    "last_failure",
    "last_blocked_at",
    "last_renewal_at",
)

#: Every Layer 5 field, so the artifact has a stable shape whether the publish
#: succeeded, was blocked by configuration, or was blocked by ownership.
_TEMPLATE: dict = {
    "schema": HEALTH_SCHEMA,
    "written_at": "",
    # Layer 1 - configuration
    "designated_writer": "",
    "machine": "",
    "role": "",
    "read_only": True,
    "read_only_reason": "",
    "config_source": "",
    # identity
    "pid": None,
    "instance_id": "",
    "holder_identity": "",
    # Layer 2 - machine-local exclusion
    "local_lock": {
        "key": "",
        "name": "",
        "held": False,
        "mutex": "unavailable",
        "file_lock": "unavailable",
        "abandoned_by_previous_owner": False,
    },
    # Layer 3 - the Drive lease
    "lease_path": "",
    "lease_holder": "",
    "lease_instance_id": "",
    "lease_acquired_at": "",
    "lease_expires_at": "",
    "last_renewal_at": "",
    "fencing_generation": None,
    # Layer 4 - failures
    "last_failure": {"at": "", "kind": "", "message": ""},
    "last_blocked_at": "",
    # emergency takeover
    "emergency_override": {
        "active": False,
        "expires_at": "",
        "reason": "",
        "configured_value": "",
        "rejected_because": "",
    },
    # last verified publication
    "last_verified_publication": {
        "at": "",
        "path": "",
        "holder": "",
        "generation": None,
        "sha256": "",
        "takeover": False,
        "previous_holder": "",
    },
    "status": "unknown",
    "healthy": False,
}


def _diagnostics_dir() -> Path:
    """Machine-local diagnostics root (honors ``TRADINGBOT_DIAGNOSTICS_DIR``)."""
    try:
        from diagnostics.artifact_io import diagnostics_dir

        return diagnostics_dir()
    except Exception:  # pragma: no cover - diagnostics package unavailable
        override = str(os.environ.get("TRADINGBOT_DIAGNOSTICS_DIR") or "").strip()
        if override:
            return Path(override)
        return Path.home() / ".tradingbotv3" / "diagnostics"


def health_artifact_path() -> Path:
    return _diagnostics_dir() / HEALTH_FILENAME


def _merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(_is_blank(item) for item in value.values())
    return False


def _carried_forward(target: Path) -> dict:
    """The sticky history from the previous artifact, if it can be trusted.

    Only the :data:`STICKY_FIELDS` are carried, and only when the current
    snapshot has nothing to say about them. Nothing describing the *current*
    attempt is ever inherited: a stale role, holder or lock state presented as
    current is exactly the kind of cheerful lie this module exists to prevent.
    """
    previous = read_writer_health(path=target)
    if not isinstance(previous, dict) or previous.get("status") in {
        "missing",
        "unreadable",
        "corrupt",
    }:
        return {}
    carried = {}
    for key in STICKY_FIELDS:
        value = previous.get(key)
        if value is not None and not _is_blank(value):
            carried[key] = value
    return carried


def write_writer_health(state: dict, *, path: Path | str | None = None) -> Path | None:
    """Atomically write the health artifact, merged onto the full field set.

    Returns the path, or ``None`` when the artifact could not be written -
    telemetry must never be the thing that fails a publish, but a caller that
    wants to know can check the return value.
    """
    target = Path(path) if path is not None else health_artifact_path()
    incoming = dict(state or {})
    history = {
        key: value
        for key, value in _carried_forward(target).items()
        if _is_blank(incoming.get(key))
    }
    payload = _merge(_merge(_TEMPLATE, history), incoming)
    payload["schema"] = HEALTH_SCHEMA
    payload["written_at"] = datetime.now().isoformat(timespec="seconds")
    payload.setdefault("machine", socket.gethostname())
    if payload.get("pid") is None:
        payload["pid"] = os.getpid()
    try:
        from diagnostics.artifact_io import atomic_write_json

        atomic_write_json(target, payload)
        return target
    except Exception:
        return None


def read_writer_health(*, path: Path | str | None = None) -> dict:
    """Read the health artifact. Absent or corrupt never reads as healthy."""
    import json

    target = Path(path) if path is not None else health_artifact_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {
            "schema": HEALTH_SCHEMA,
            "status": "missing",
            "healthy": False,
            "error": f"no writer health telemetry at {target}",
            "path": str(target),
        }
    except OSError as exc:
        return {
            "schema": HEALTH_SCHEMA,
            "status": "unreadable",
            "healthy": False,
            "error": f"writer health telemetry could not be read: {exc}",
            "path": str(target),
        }
    try:
        payload = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "schema": HEALTH_SCHEMA,
            "status": "corrupt",
            "healthy": False,
            "error": f"writer health telemetry is not valid JSON: {exc}",
            "path": str(target),
        }
    if not isinstance(payload, dict):
        return {
            "schema": HEALTH_SCHEMA,
            "status": "corrupt",
            "healthy": False,
            "error": "writer health telemetry is not a JSON object",
            "path": str(target),
        }
    if payload.get("schema") != HEALTH_SCHEMA:
        return {
            "schema": HEALTH_SCHEMA,
            "status": "corrupt",
            "healthy": False,
            "error": f"unknown writer health schema {payload.get('schema')!r}",
            "path": str(target),
        }
    payload["path"] = str(target)

    # Freshness. The write side swallows every failure by design (telemetry must
    # never fail a publish), and a machine that crashed, went to sleep, or whose
    # diagnostics directory turned read-only keeps its last green artifact on
    # disk forever. Without this check a Health panel renders "published, all
    # fine" for a machine that stopped hours ago - the exact thing this module's
    # contract says it must never do.
    age = _age_minutes(payload.get("written_at"))
    payload["age_minutes"] = age
    if age is None:
        payload["status"] = "corrupt"
        payload["healthy"] = False
        payload["error"] = (
            f"writer health telemetry has no usable written_at ({payload.get('written_at')!r}); "
            "its age cannot be established, so it is not evidence of anything"
        )
    elif age > MAX_TELEMETRY_AGE_MINUTES:
        payload["status"] = f"stale: {payload.get('status') or 'unknown'}"
        payload["healthy"] = False
        payload["error"] = (
            f"writer health telemetry is {age:.0f} minutes old (limit "
            f"{MAX_TELEMETRY_AGE_MINUTES}); this machine has stopped reporting, so its last "
            "state is history, not health"
        )
    return payload


def _age_minutes(written_at) -> float | None:
    """Minutes since ``written_at``; ``None`` when it cannot be established.

    A stamp from the future is treated as age zero rather than as a negative
    age: clock skew is somebody else's problem to report, and "fresher than
    now" must not read as "stale".
    """
    if not isinstance(written_at, str) or not written_at.strip():
        return None
    try:
        stamp = datetime.fromisoformat(written_at.strip())
    except ValueError:
        return None
    now = datetime.now()
    if stamp.tzinfo is not None and stamp.utcoffset() is not None:
        now = now.astimezone()
    delta: timedelta = now - stamp
    return max(0.0, delta.total_seconds() / 60.0)
