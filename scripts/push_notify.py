"""Phone/watch push channel over ntfy (https://ntfy.sh or self-hosted).

Why ntfy: plain outbound HTTPS POST - no inbound ports, no server to run at
home, no per-device registration. The trader installs the ntfy app on the
iPhone and subscribes to a private topic; the Apple Watch mirrors iPhone
notifications automatically, so one channel covers both. A self-hosted ntfy
server works by pointing ``push_ntfy_server`` at it (that is the "set up a
server locally" path - still outbound-only from this app's side).

Configuration lives in the machine-local settings file (Settings are per
machine on purpose: only the machine actually watching prices should push):

- ``push_ntfy_topic``   the private topic name; empty = pushes disabled.
- ``push_ntfy_server``  default ``https://ntfy.sh``.
- ``push_ntfy_token``   optional access token for a protected topic.

Every function here is fail-quiet: a push is a convenience on top of the
desk, and a network hiccup at 05:00 must never take down the alert loop that
would retry a minute later.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

DEFAULT_NTFY_SERVER = "https://ntfy.sh"
PUSH_TOPIC_SETTING = "push_ntfy_topic"
PUSH_SERVER_SETTING = "push_ntfy_server"
PUSH_TOKEN_SETTING = "push_ntfy_token"

# ntfy priorities; "urgent" breaks through iOS Focus/sleep modes when the
# subscriber enables critical alerting for the topic - the wake-the-trader
# level Evening mode uses for position price alerts.
PUSH_PRIORITIES = ("min", "low", "default", "high", "urgent")
_REQUEST_TIMEOUT_SECONDS = 10


def load_push_config() -> dict[str, str]:
    """Current push settings; import is deferred so tests can run headless."""
    try:
        from project_paths import get_local_setting

        return {
            "server": str(get_local_setting(PUSH_SERVER_SETTING, DEFAULT_NTFY_SERVER) or DEFAULT_NTFY_SERVER),
            "topic": str(get_local_setting(PUSH_TOPIC_SETTING, "") or "").strip(),
            "token": str(get_local_setting(PUSH_TOKEN_SETTING, "") or "").strip(),
        }
    except Exception:
        return {"server": DEFAULT_NTFY_SERVER, "topic": "", "token": ""}


def push_configured(config: Mapping[str, str] | None = None) -> bool:
    config = config if config is not None else load_push_config()
    return bool(str(config.get("topic") or "").strip())


def build_push_request(
    title: str,
    message: str,
    *,
    config: Mapping[str, str],
    priority: str = "high",
    tags: str = "",
) -> urllib.request.Request | None:
    """The ntfy POST for one notification, or ``None`` when unconfigured."""
    topic = str(config.get("topic") or "").strip().strip("/")
    if not topic:
        return None
    server = str(config.get("server") or DEFAULT_NTFY_SERVER).strip().rstrip("/") or DEFAULT_NTFY_SERVER
    if priority not in PUSH_PRIORITIES:
        priority = "high"
    headers = {
        # Latin-1 is all HTTP headers can carry; tickers and prices fit fine.
        "Title": str(title or "TradingBotV3").encode("ascii", "replace").decode("ascii"),
        "Priority": priority,
        "Content-Type": "text/plain; charset=utf-8",
    }
    if tags:
        headers["Tags"] = str(tags)
    token = str(config.get("token") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(
        f"{server}/{topic}",
        data=str(message or "").encode("utf-8"),
        headers=headers,
        method="POST",
    )


def send_push(
    title: str,
    message: str,
    *,
    priority: str = "high",
    tags: str = "",
    config: Mapping[str, str] | None = None,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """POST one notification; never raises. ``{"ok": bool, "error": str}``.

    ``ok`` False with ``error`` "" means pushes are simply not configured -
    callers may treat that as a silent no-op rather than a failure.
    """
    config = config if config is not None else load_push_config()
    request = build_push_request(title, message, config=config, priority=priority, tags=tags)
    if request is None:
        return {"ok": False, "error": ""}
    opener = opener or urllib.request.urlopen
    try:
        with opener(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            status = int(getattr(response, "status", 200) or 200)
        if 200 <= status < 300:
            return {"ok": True, "error": ""}
        return {"ok": False, "error": f"ntfy returned HTTP {status}"}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = exc.read().decode("utf-8", "replace")
            detail = str(json.loads(body).get("error") or "") if body else ""
        except Exception:
            detail = ""
        error = f"ntfy HTTP {exc.code}" + (f": {detail}" if detail else "")
        logging.warning("Push notification failed: %s", error)
        return {"ok": False, "error": error}
    except Exception as exc:
        logging.warning("Push notification failed: %r", exc)
        return {"ok": False, "error": repr(exc)}
