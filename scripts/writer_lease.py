"""Single-writer lease for shared mutable exports (plan.md Phase 2.9).

Two machines (home desk + mini-PC) can both write Drive exports like the
away report. Atomic replacement prevents partial files but not lost updates
from two valid writers. A lease file names the current writer and an expiry;
a second machine may read freely but must not overwrite the active writer's
export until the lease expires or is explicitly taken over.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

LEASE_SCHEMA = "writer_lease_v1"
DEFAULT_TTL_MINUTES = 10


def default_holder_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def machine_holder_id() -> str:
    """Stable per-machine identity: renewals survive app restarts."""
    return socket.gethostname()


class LeaseUnavailable(RuntimeError):
    """Another writer holds an unexpired lease."""


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def acquire(
    lease_path: Path | str,
    *,
    holder: str | None = None,
    ttl_minutes: int = DEFAULT_TTL_MINUTES,
    now: datetime | None = None,
    takeover: bool = False,
) -> dict:
    """Acquire or renew the lease. Raises LeaseUnavailable when another
    holder's lease has not expired (unless takeover=True, the explicit
    manual override the plan requires)."""
    lease_path = Path(lease_path)
    holder = holder or machine_holder_id()
    moment = now or datetime.now()
    current = _read(lease_path)
    if current and not takeover:
        expires = _parse_ts(current.get("expires_at"))
        if (
            current.get("holder")
            and current["holder"] != holder
            and expires is not None
            and expires > moment
        ):
            raise LeaseUnavailable(
                f"{lease_path.name} held by {current['holder']} until {current['expires_at']}"
            )
    payload = {
        "schema": LEASE_SCHEMA,
        "holder": holder,
        "acquired_at": moment.isoformat(timespec="seconds"),
        "expires_at": (moment + timedelta(minutes=max(1, int(ttl_minutes)))).isoformat(
            timespec="seconds"
        ),
        "takeover": bool(takeover),
    }
    _write(lease_path, payload)
    return payload


def release(lease_path: Path | str, *, holder: str | None = None) -> bool:
    lease_path = Path(lease_path)
    holder = holder or machine_holder_id()
    current = _read(lease_path)
    if current and current.get("holder") not in ("", None, holder):
        return False  # never release someone else's lease
    try:
        lease_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def holder_of(lease_path: Path | str, *, now: datetime | None = None) -> str | None:
    """Current unexpired holder, or None."""
    current = _read(Path(lease_path))
    if not current:
        return None
    expires = _parse_ts(current.get("expires_at"))
    if expires is None or expires <= (now or datetime.now()):
        return None
    return str(current.get("holder") or "") or None


def _parse_ts(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
