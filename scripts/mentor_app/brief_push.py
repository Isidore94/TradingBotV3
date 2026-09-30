"""The one optional push a day: the 06:30 PT tape line to the phone. The only mentor_app module
that may import ``push_notify``.

Off by default (setting ``mentor_push_brief``). The line is built from the regime pack only
(``regime_pack.push_line``: mode, D1 env, next econ event, index RRS) and never carries
model text. At most once per PT day: the ``tape_push_sent`` app_state key is written BEFORE
the send, so a crash mid-send loses the line rather than sending it twice. Call it off the
Qt thread (the send is a network call).
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
STATE_KEY = "tape_push_sent"
TITLE = "Trade Mentor tape"
#: The line goes out from 06:30 PT; an app started after 07:30 sends nothing that day.
SEND_FROM = time(6, 30)
SEND_UNTIL = time(7, 30)


def maybe_send(
    store: Any,
    pack: Any,
    *,
    now: datetime,
    enabled: bool | None = None,
    send: Callable[..., Mapping[str, Any]] | None = None,
) -> str:
    """Send today's line if it is due; returns the line sent, or "" when nothing was sent."""
    from mentor_app import settings

    if not (settings.push_brief_enabled() if enabled is None else enabled):
        return ""
    moment = now if now.tzinfo else now.astimezone()
    local = moment.astimezone(PT)
    if local.weekday() >= 5 or not (SEND_FROM <= local.time() < SEND_UNTIL):
        return ""
    today = local.date().isoformat()
    if store.get_state(STATE_KEY) == today:
        return ""
    from mentor_packs import regime_pack

    line = regime_pack.push_line(pack)
    if not store.set_state(STATE_KEY, today):
        return ""  # the day could not be marked: sending could repeat, so nothing is sent
    if send is None:
        import push_notify

        send = push_notify.send_push
    result = send(TITLE, line, priority="default", tags="chart_with_upwards_trend")
    if not (result or {}).get("ok"):
        logging.info("Trade Mentor tape push not delivered: %s", dict(result or {}))
    return line
