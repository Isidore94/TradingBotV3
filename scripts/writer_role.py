"""Layer 1: which machine is allowed to write the shared exports.

THE AUTHORITY IS EXPLICIT CONFIGURATION, NOT A RACE
---------------------------------------------------
A shared file is not a compare-and-swap lock, so "whoever gets there first owns
it" is not a safety property - it is a description of a race. (This was written
when the folder was Drive-synced; the folder is plain local storage since
decision 0015, but the reasoning holds for any shared mount.) The authority over shared mutable output is therefore a *configured*
designated writer: one machine is named, every other machine is a read-only
secondary that refuses to publish and says why.

The configuration is deliberately **machine-local**: this module reads the
repo's existing local-settings convention (``project_paths.get_local_setting``,
backed by ``LOCAL_SETTINGS_FILE`` under ``%LOCALAPPDATA%``) and environment
variables. A role file living in the shared folder is explicitly *not* consulted,
because it would suffer exactly the sync-convergence problem it was meant to
solve: two machines could read two different versions of "who is the writer".

FAIL CLOSED WHEN UNCONFIGURED
-----------------------------
If no designated writer is configured, :func:`resolve_writer_role` returns a
role whose ``may_publish`` is ``False`` and whose ``reason`` names the
configuration failure. There is no "first machine wins" fallback: the previous
verified report is preserved, and the failure is reported in the publish result,
in the log, and in Health telemetry.

EMERGENCY TAKEOVER
------------------
:func:`resolve_emergency_override` reads an override that is explicit,
time-bounded, and auditable. It is inert unless *all* of the following hold:

* the value is an unambiguous true token (``true``/``yes``/``on``/``1``) or an
  ISO timestamp;
* an expiry is configured (or the value itself is that expiry);
* that expiry is in the future;
* and that expiry is within :data:`MAX_OVERRIDE_WINDOW_HOURS`.

Anything else - ``""``, ``"0"``, ``"false"``, ``"no"``, ``"maybe"``,
``"null"``, ``[]``, ``{}``, a past timestamp, a true token with no expiry, an
expiry a century out - evaluates to *inactive*. A malformed configuration value
must never become an active override by being merely non-empty.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = [
    "CONFIG_OVERRIDE_EXPIRY_KEYS",
    "CONFIG_OVERRIDE_KEYS",
    "CONFIG_ROLE_KEYS",
    "CONFIG_WRITER_KEYS",
    "MAX_OVERRIDE_WINDOW_HOURS",
    "EmergencyOverride",
    "WriterRole",
    "local_machine_name",
    "resolve_emergency_override",
    "resolve_writer_role",
]

#: An emergency takeover must be time-bounded in practice, not just in form.
#: Without a ceiling, ``9999-12-31`` parses, is "in the future", and authorizes
#: breaking the other machine's live lease on every publish for the life of the
#: configuration. An expiry beyond this window is rejected (fail closed) with a
#: message naming the ceiling, rather than silently clamped to something the
#: operator did not write.
MAX_OVERRIDE_WINDOW_HOURS = 12

# Machine-local settings keys. The first spelling is canonical; the rest are
# accepted so a machine configured under an older/alternate name still works.
CONFIG_WRITER_KEYS = (
    "designated_writer",
    "designated_writer_machine",
    "shared_writer_machine",
    "autopilot_designated_writer",
)
CONFIG_ROLE_KEYS = (
    "writer_role",
    "shared_writer_role",
    "autopilot_writer_role",
)
CONFIG_OVERRIDE_KEYS = (
    "writer_emergency_takeover",
    "emergency_takeover",
    "writer_override",
)
CONFIG_OVERRIDE_EXPIRY_KEYS = (
    "writer_emergency_takeover_expires_at",
    "emergency_takeover_expires_at",
    "writer_override_expires_at",
)
CONFIG_OVERRIDE_REASON_KEYS = (
    "writer_emergency_takeover_reason",
    "emergency_takeover_reason",
    "writer_override_reason",
)

ENV_WRITER_KEYS = ("TRADINGBOT_DESIGNATED_WRITER", "TRADINGBOTV3_DESIGNATED_WRITER")
ENV_ROLE_KEYS = ("TRADINGBOT_WRITER_ROLE", "TRADINGBOTV3_WRITER_ROLE")
ENV_OVERRIDE_KEYS = (
    "TRADINGBOT_WRITER_OVERRIDE",
    "TRADINGBOT_LEASE_TAKEOVER",
    "WRITER_LEASE_FORCE",
)
ENV_OVERRIDE_EXPIRY_KEYS = (
    "TRADINGBOT_WRITER_OVERRIDE_EXPIRES_AT",
    "TRADINGBOT_LEASE_TAKEOVER_EXPIRES_AT",
)
ENV_OVERRIDE_REASON_KEYS = ("TRADINGBOT_WRITER_OVERRIDE_REASON",)

#: Role spellings that mean "this machine may publish shared mutable output".
_WRITER_ROLE_WORDS = frozenset(
    {"designated_writer", "designated", "writer", "primary", "master", "desk"}
)
#: Role spellings that mean "read-only".
_SECONDARY_ROLE_WORDS = frozenset(
    {"secondary", "read_only", "readonly", "read-only", "reader", "observer", "standby"}
)
#: The only values that may switch an emergency override on.
_TRUE_WORDS = frozenset({"1", "true", "yes", "y", "on", "enable", "enabled"})

ROLE_DESIGNATED = "designated_writer"
ROLE_SECONDARY = "secondary"
ROLE_UNCONFIGURED = "unconfigured"
ROLE_MISCONFIGURED = "misconfigured"


def local_machine_name() -> str:
    """This machine's name, from the same source the lease identity uses."""
    try:
        return str(socket.gethostname() or "")
    except Exception:  # pragma: no cover - gethostname failing is pathological
        return ""


# ---------------------------------------------------------------------------
# configuration readers
# ---------------------------------------------------------------------------
def _local_setting(key: str):
    """Read one machine-local setting, tolerating an unreadable settings file.

    ``project_paths`` is imported lazily and its attributes are read at call
    time, so a redirected settings location (tests, a relocated profile) is
    honored instead of being frozen at import.
    """
    try:
        import project_paths

        return project_paths.get_local_setting(key, None)
    except Exception:
        return None


def _first_configured(env_keys, config_keys) -> tuple[object, str]:
    """First configured value plus where it came from ('env:X' / 'config:X')."""
    for key in env_keys:
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip(), f"env:{key}"
    for key in config_keys:
        value = _local_setting(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), f"local_settings:{key}"
        if value is not None and not isinstance(value, str):
            return value, f"local_settings:{key}"
    return None, ""


def _normalized(value) -> str:
    return str(value or "").strip().lower()


# ---------------------------------------------------------------------------
# the role
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WriterRole:
    """The resolved answer to 'may this machine write shared output?'."""

    role: str
    machine: str
    designated_writer: str
    may_publish: bool
    reason: str
    source: str

    @property
    def read_only(self) -> bool:
        return not self.may_publish

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "machine": self.machine,
            "designated_writer": self.designated_writer,
            "may_publish": bool(self.may_publish),
            "read_only": bool(self.read_only),
            "read_only_reason": "" if self.may_publish else self.reason,
            "reason": self.reason,
            "config_source": self.source,
        }


def resolve_writer_role(machine: str | None = None) -> WriterRole:
    """Resolve this machine's writer role from machine-local configuration."""
    machine_name = str(machine or local_machine_name() or "")
    configured, writer_source = _first_configured(ENV_WRITER_KEYS, CONFIG_WRITER_KEYS)
    role_value, role_source = _first_configured(ENV_ROLE_KEYS, CONFIG_ROLE_KEYS)
    source = "; ".join(part for part in (writer_source, role_source) if part)

    if not isinstance(configured, str) or not configured.strip():
        return WriterRole(
            role=ROLE_UNCONFIGURED,
            machine=machine_name,
            designated_writer="",
            may_publish=False,
            reason=(
                f"no designated writer is configured on this machine ({machine_name or 'unknown'}); "
                "shared publishing is refused so the last verified report is preserved. Set "
                "'designated_writer' in local_settings.json (or TRADINGBOT_DESIGNATED_WRITER) on "
                "the machine that should publish, and set it to that same machine name here so "
                "this one knows it is a read-only secondary."
            ),
            source=source,
        )

    designated = configured.strip()
    role_word = _normalized(role_value)
    names_match = designated.strip().lower() == machine_name.strip().lower() and bool(machine_name)

    if role_word and role_word not in _WRITER_ROLE_WORDS and role_word not in _SECONDARY_ROLE_WORDS:
        return WriterRole(
            role=ROLE_MISCONFIGURED,
            machine=machine_name,
            designated_writer=designated,
            may_publish=False,
            reason=(
                f"the configured writer role {role_value!r} is not a role this build understands; "
                "shared publishing is refused (fail closed). Use 'designated_writer' or 'secondary'."
            ),
            source=source,
        )

    if role_word in _SECONDARY_ROLE_WORDS:
        return WriterRole(
            role=ROLE_SECONDARY,
            machine=machine_name,
            designated_writer=designated,
            may_publish=False,
            reason=(
                f"this machine ({machine_name or 'unknown'}) is configured as a read-only "
                f"secondary; the designated writer is {designated!r}. It reads shared output and "
                "publishes none of it."
            ),
            source=source,
        )

    if not names_match:
        return WriterRole(
            role=ROLE_SECONDARY,
            machine=machine_name,
            designated_writer=designated,
            may_publish=False,
            reason=(
                f"this machine ({machine_name or 'unknown'}) is not the configured designated "
                f"writer ({designated!r}), so it is read-only for shared output."
            ),
            source=source,
        )

    return WriterRole(
        role=ROLE_DESIGNATED,
        machine=machine_name,
        designated_writer=designated,
        may_publish=True,
        reason=f"this machine ({machine_name}) is the configured designated writer",
        source=source,
    )


# ---------------------------------------------------------------------------
# emergency takeover
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmergencyOverride:
    """An explicit, time-bounded, auditable break-the-lease authorization."""

    active: bool
    expires_at: str
    reason: str
    source: str
    configured_value: str
    rejected_because: str = ""

    def as_dict(self) -> dict:
        return {
            "active": bool(self.active),
            "expires_at": self.expires_at,
            "reason": self.reason,
            "source": self.source,
            "configured_value": self.configured_value,
            "rejected_because": self.rejected_because,
        }


def _parse_timestamp(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_instant(value: datetime) -> datetime:
    """Normalize a timestamp to naive UTC so any two can be compared.

    The runbook asks the operator to hand-write an ISO expiry, and ``3.11+``
    accepts a trailing ``Z`` or ``+02:00``. Comparing that against a naive
    ``datetime.now()`` raised ``TypeError`` *outside* every handler in the
    publish path, so the machine neither published nor reported a refusal and
    Health telemetry kept whatever the previous cycle had written. A naive value
    means local wall-clock, which is what the runbook's own example is.

    Deliberately arithmetic rather than :meth:`datetime.astimezone`, which the
    Windows CRT refuses for extreme years - ``datetime(9999, 12, 31)`` is
    exactly the kind of value an operator typo produces, and it must be
    *rejected*, not turned into an unhandled ``OSError``.
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


def resolve_emergency_override(now: datetime | None = None) -> EmergencyOverride:
    """Resolve the emergency-takeover configuration. Inert unless fully valid."""
    moment = now or datetime.now()
    raw, source = _first_configured(ENV_OVERRIDE_KEYS, CONFIG_OVERRIDE_KEYS)
    expiry_raw, expiry_source = _first_configured(
        ENV_OVERRIDE_EXPIRY_KEYS, CONFIG_OVERRIDE_EXPIRY_KEYS
    )
    reason_raw, _ = _first_configured(ENV_OVERRIDE_REASON_KEYS, CONFIG_OVERRIDE_REASON_KEYS)
    reason = str(reason_raw or "").strip()
    shown = "" if raw is None else str(raw)

    def inert(why: str) -> EmergencyOverride:
        return EmergencyOverride(
            active=False,
            expires_at=str(expiry_raw or ""),
            reason=reason,
            source="; ".join(p for p in (source, expiry_source) if p),
            configured_value=shown,
            rejected_because=why,
        )

    if raw is None:
        return inert("no emergency takeover is configured")

    # Booleans are the only non-string type that may mean "on"; a list, a dict
    # or a number is a malformed value, never a truthy override.
    if isinstance(raw, bool):
        enabled = raw
    elif isinstance(raw, str):
        text = raw.strip().lower()
        if text in _TRUE_WORDS:
            enabled = True
        elif _parse_timestamp(raw) is not None:
            # The value itself is the bound: "override until <timestamp>".
            enabled = True
            if expiry_raw is None:
                expiry_raw, expiry_source = raw, source
        else:
            return inert(f"the configured value {shown!r} is not an explicit true token")
    else:
        return inert(f"the configured value {shown!r} is not a boolean or a timestamp")

    if not enabled:
        return inert(f"the configured value {shown!r} is explicitly off")

    expires = _parse_timestamp(expiry_raw if isinstance(expiry_raw, str) else None)
    if expires is None:
        return inert(
            "an emergency takeover must be time-bounded: set "
            f"{CONFIG_OVERRIDE_EXPIRY_KEYS[0]!r} to an ISO timestamp in the future"
        )
    instant = _as_instant(expires)
    reference = _as_instant(moment)
    if instant <= reference:
        return inert(f"the emergency takeover expired at {expires.isoformat(timespec='seconds')}")
    if instant > reference + timedelta(hours=MAX_OVERRIDE_WINDOW_HOURS):
        return inert(
            f"the emergency takeover expiry {expires.isoformat(timespec='seconds')} is more than "
            f"{MAX_OVERRIDE_WINDOW_HOURS}h away; an override that long is not time-bounded in "
            "any useful sense - it would re-break the other machine's lease on every publish "
            "until somebody remembered to remove it. Set an expiry within "
            f"{MAX_OVERRIDE_WINDOW_HOURS}h and renew it deliberately if the emergency lasts"
        )

    return EmergencyOverride(
        active=True,
        expires_at=expires.isoformat(timespec="seconds"),
        reason=reason or "(no reason recorded)",
        source="; ".join(p for p in (source, expiry_source) if p),
        configured_value=shown,
    )


# ---------------------------------------------------------------------------
# operator CLI
# ---------------------------------------------------------------------------
# The role is deliberately machine-local, which means switching it is a manual
# step on each machine. Hand-editing local_settings.json for that is a bad idea:
# the file also holds API keys and broker tokens, so a slip while editing it
# costs more than a wrong role. These helpers do a read-modify-write through
# project_paths.save_local_setting, which preserves every other key.
def set_role(role: str, writer: str) -> "WriterRole":
    """Write this machine's role + the designated writer, then re-resolve.

    Returns the freshly resolved role so a caller can confirm the write landed
    rather than trusting that it did.
    """
    import project_paths

    normalized = _normalized(role)
    if normalized in _WRITER_ROLE_WORDS:
        role_value = ROLE_DESIGNATED
    elif normalized in _SECONDARY_ROLE_WORDS:
        role_value = ROLE_SECONDARY
    else:
        raise ValueError(
            f"unknown role {role!r}; use {ROLE_DESIGNATED!r} or {ROLE_SECONDARY!r}"
        )
    if not str(writer or "").strip():
        raise ValueError("a designated writer machine name is required")

    project_paths.save_local_setting(CONFIG_WRITER_KEYS[0], str(writer).strip())
    project_paths.save_local_setting(CONFIG_ROLE_KEYS[0], role_value)
    return resolve_writer_role()


def _describe(resolved: "WriterRole") -> str:
    lines = [
        f"machine            : {resolved.machine}",
        f"role               : {resolved.role}",
        f"designated writer  : {resolved.designated_writer or '(none configured)'}",
        f"may publish        : {resolved.may_publish}",
        f"config source      : {resolved.source or '(nothing configured)'}",
    ]
    if not resolved.may_publish:
        lines.append(f"read-only reason   : {resolved.reason}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="writer_role",
        description=(
            "Show or set this machine's shared-output writer role. The role is "
            "machine-local by design: run this on EACH machine. Exactly one "
            "machine should be the designated writer at a time."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--designate-self",
        action="store_true",
        help="make THIS machine the designated writer (it may publish)",
    )
    group.add_argument(
        "--secondary",
        metavar="WRITER_MACHINE",
        help="make this machine a read-only secondary, naming the machine that writes",
    )
    args = parser.parse_args(argv)

    if args.designate_self:
        resolved = set_role(ROLE_DESIGNATED, local_machine_name())
    elif args.secondary:
        resolved = set_role(ROLE_SECONDARY, args.secondary)
    else:
        resolved = resolve_writer_role()

    print(_describe(resolved))

    override = resolve_emergency_override()
    if override.active:
        print(f"EMERGENCY OVERRIDE ACTIVE until {override.expires_at}: {override.reason}")

    if args.designate_self or args.secondary:
        print("\nWritten to machine-local settings. Restart the GUI to pick it up.")
    # Exit non-zero when this machine cannot publish, so an away-day preflight
    # can catch "I forgot to switch the role" instead of discovering it as a
    # missing phone report hours later.
    return 0 if resolved.may_publish else 1


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    import sys

    raise SystemExit(main(sys.argv[1:]))
