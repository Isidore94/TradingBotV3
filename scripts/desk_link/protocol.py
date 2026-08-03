"""Desk Link wire protocol: newline-delimited JSON over a LAN TCP socket.

Both ends are this application, so the framing stays as simple as possible:
one UTF-8 JSON object per line. Every message is an envelope::

    {"v": 1, "type": "<kind>", "ts": "<iso8601 with tz>", "payload": {...}}

The first message on a connection must be the satellite's ``hello`` carrying
the link token; the server answers ``welcome`` (and starts relaying) or
``rejected`` (and closes). Tier 1 satellites are view-only: after the
handshake they send only ``ping`` keepalives.

Security model (docs/MULTI_MACHINE_DESK_PROPOSAL.md): LAN only, single
trader, static link token generated on the main and compared
constant-time. This is deliberately not internet-grade transport — the
server must only ever bind to a private-network interface.
"""

from __future__ import annotations

import hmac
import json
import secrets
from datetime import datetime, timezone
from typing import Any

PROTOCOL_VERSION = 1

# One line = one message. A popup payload (a few thousand M5/D1 bars plus
# levels) is tens to hundreds of KB of JSON; 8 MB leaves headroom without
# letting a corrupt peer make the reader buffer unbounded.
MAX_MESSAGE_BYTES = 8 * 1024 * 1024

TYPE_HELLO = "hello"
TYPE_WELCOME = "welcome"
TYPE_REJECTED = "rejected"
TYPE_PING = "ping"
TYPE_PONG = "pong"
TYPE_STATE_SNAPSHOT = "state_snapshot"
TYPE_ALERT_POPUP = "alert_popup"
# Tier 2 control lease + intents (docs/MULTI_MACHINE_DESK_PROPOSAL.md).
# The lease lives and dies with the satellite's authenticated connection:
# pings renew it implicitly, and the server's idle timeout is the grace
# window - a satellite that sleeps or drops off Wi-Fi loses control when
# its connection is reaped, never silently keeps it.
TYPE_LEASE_REQUEST = "lease_request"   # sat -> main: take control
TYPE_LEASE_RELEASE = "lease_release"   # sat -> main: give control back
TYPE_LEASE_GRANT = "lease_grant"       # main -> sat: you have control
TYPE_LEASE_DENIED = "lease_denied"     # main -> sat: someone else holds it
TYPE_LEASE_REVOKED = "lease_revoked"   # main -> sat: control taken back
TYPE_INTENT = "intent"                 # sat -> main: one decision to apply
TYPE_INTENT_RESULT = "intent_result"   # main -> sat: ack {seq, ok, detail}
# Tier 3 full relay: one generic envelope for every live desk surface.
# payload = {"stream": "<name>", "data": {...}} - adding a stream never
# needs a protocol change, and an unknown stream is skipped, not an error.
TYPE_DESK_STREAM = "desk_stream"


class DeskLinkProtocolError(ValueError):
    """A peer sent something that is not a valid Desk Link message."""


class DeskLinkAuthError(DeskLinkProtocolError):
    """The hello handshake failed (bad token / bad first message)."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def generate_link_token() -> str:
    return secrets.token_urlsafe(24)


def tokens_match(expected: str, presented: str) -> bool:
    return hmac.compare_digest(str(expected or ""), str(presented or ""))


def make_message(message_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": str(message_type),
        "ts": utc_now_iso(),
        "payload": payload if payload is not None else {},
    }


def make_hello(token: str, machine: str) -> dict[str, Any]:
    return make_message(TYPE_HELLO, {"token": str(token), "machine": str(machine), "role": "satellite"})


def make_welcome(server_machine: str) -> dict[str, Any]:
    return make_message(TYPE_WELCOME, {"machine": str(server_machine)})


def make_rejected(reason: str) -> dict[str, Any]:
    return make_message(TYPE_REJECTED, {"reason": str(reason)})


def encode_message(message: dict[str, Any]) -> bytes:
    line = json.dumps(message, separators=(",", ":"), default=str)
    raw = line.encode("utf-8") + b"\n"
    if len(raw) > MAX_MESSAGE_BYTES:
        raise DeskLinkProtocolError(f"message of {len(raw)} bytes exceeds the {MAX_MESSAGE_BYTES}-byte limit")
    return raw


def decode_message(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        if len(line) > MAX_MESSAGE_BYTES:
            raise DeskLinkProtocolError("incoming message exceeds the size limit")
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DeskLinkProtocolError("incoming message is not valid UTF-8") from exc
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DeskLinkProtocolError("incoming message is not valid JSON") from exc
    if not isinstance(message, dict):
        raise DeskLinkProtocolError("incoming message is not a JSON object")
    if message.get("v") != PROTOCOL_VERSION:
        raise DeskLinkProtocolError(f"unsupported protocol version: {message.get('v')!r}")
    if not isinstance(message.get("type"), str) or not message["type"]:
        raise DeskLinkProtocolError("incoming message has no type")
    if not isinstance(message.get("payload"), dict):
        raise DeskLinkProtocolError("incoming message has no payload object")
    return message


def validate_hello(message: dict[str, Any], expected_token: str) -> str:
    """Validate a handshake message; return the satellite's machine name."""
    if message.get("type") != TYPE_HELLO:
        raise DeskLinkAuthError(f"expected hello as the first message, got {message.get('type')!r}")
    payload = message["payload"]
    if not tokens_match(expected_token, str(payload.get("token") or "")):
        raise DeskLinkAuthError("link token mismatch")
    machine = str(payload.get("machine") or "").strip()
    return machine or "unnamed-satellite"
