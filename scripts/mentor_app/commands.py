"""Slash commands: shortcuts, not the interface. Parsing only; the window acts on the result."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

HELP_TEXT = (
    "**Commands** (or just ask in plain words)\n\n"
    "- `/help` this list\n"
    "- `/quiet 2h` mute the Inbox for a while (`30m`, `1h30m`)\n"
    "- `/remember <text>` keep a note about you that I will recall later\n"
    "- `/tape` the desk right now (works with the brain off)\n"
    "- `/read` give a market read now (Trade Mentor card)\n"
    "- `/pause` no Trade Mentor questions for the rest of today\n"
    "- `/pick SYM` pick assessment (coming in Phase 2)\n"
    "- `/vetoes` veto challenges (coming in Phase 3)\n"
)
MAX_QUIET = timedelta(hours=12)


@dataclass(frozen=True)
class CommandResult:
    action: str
    reply: str = ""
    arg: Any = None


def parse_duration(text: str) -> timedelta | None:
    """``2h``, ``30m``, ``1h30m`` or bare minutes; None when unreadable or zero."""
    raw = str(text or "").strip().lower().replace(" ", "")
    if not raw:
        return None
    if raw.isdigit():
        minutes = int(raw)
        return timedelta(minutes=minutes) if minutes > 0 else None
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?", raw)
    if not match or not any(match.groups()):
        return None
    total = timedelta(hours=int(match.group(1) or 0), minutes=int(match.group(2) or 0))
    return total if total > timedelta(0) else None


def handle(text: str) -> CommandResult | None:
    """A command's result, or None when ``text`` is ordinary chat."""
    stripped = str(text or "").strip()
    if not stripped.startswith("/"):
        return None
    head, _, rest = stripped.partition(" ")
    name, rest = head[1:].lower(), rest.strip()
    if name in ("help", "?"):
        return CommandResult("help", HELP_TEXT)
    if name == "quiet":
        duration = parse_duration(rest or "1h")
        if duration is None:
            return CommandResult("error", "Try `/quiet 2h` or `/quiet 30m`.")
        duration = min(duration, MAX_QUIET)
        return CommandResult("quiet", "", duration)
    if name == "remember":
        if not rest:
            return CommandResult("error", "Try `/remember I stop after two losses`.")
        return CommandResult("remember", "", rest)
    if name == "tape":
        return CommandResult("tape")
    if name == "read":
        return CommandResult("read")
    if name == "pause":
        return CommandResult("pause")
    if name == "pick":
        symbol = rest.split()[0].upper() if rest else ""
        return CommandResult("stub", f"`/pick {symbol or 'SYM'}` is coming in Phase 2.")
    if name == "vetoes":
        return CommandResult("stub", "`/vetoes` is coming in Phase 3.")
    return CommandResult("error", f"I don't know `/{name}`. Type `/help`.")
