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
    "- `/remember <text>` keep a note about you that I will recall later (start it with `rule:` for a rule)\n"
    "- `/memory` what I loaded at start (night digests and your notes), with ids\n"
    "- `/recall <text>` search what we said before (plain text search when the brain is off)\n"
    "- `/forget <id>` retire a note (it is kept, never deleted); `/keep <id>` says it is still true\n"
    "- `/tape` the tape: Auto mode, D1, last night's read, econ, sectors (read aloud when the brain is up)\n"
    "- `/read` give a market read now (Trade Mentor card)\n"
    "- `/pause` no Trade Mentor questions for the rest of today\n"
    "- `/pick SYM` what the desk knows about a pick, narrated (or tap a Focus chip)\n"
    "- `/debate SYM [long|short]` a bull case and a bear case from the same evidence, side by side (you decide)\n"
    "- `/news SYM [days]` the stored headlines for a stock (title, source, time, link; 3 days unless you say)\n"
    "- `/vetoes [YYYY-MM-DD]` the last session's vetoes (or that one's), each with its slice and any challenge\n"
    "- `/check short NVDA 400 stop 3.20 entry 3.05` check a trade before you take it (advice only; it never orders)\n"
    "- `/book` your open positions by account (Questrade when it can be read, else the journal; read-only)\n"
    "- `/mirror [weeks]` your own record in cuts with n: likes vs the scan, vetoes, journal, regime (6 weeks"
    " unless you say)\n"
    "- `/tilt` today's patterns after a loss (observations with leg ids) and how often they led to a red rest"
    " of day\n"
    "- `/scorecard` how the challenges have done by kind, with n, and how the app itself is doing\n"
    "- `/hypotheses` the night's queries into the shadow permutation grid, each with its cell and grade\n"
    "- `/ai off [2h|4h|tonight]` pause every local-AI use of the GPU host (default: until 06:00);"
    " `/ai on` resumes; `/ai` says which\n"
)
AI_USAGE = "Try `/ai off 2h`, `/ai off tonight`, `/ai on` or `/ai`."
#: `/ai off` words for "until I resume".
FOREVER_WORDS = ("until_resumed", "forever", "indefinitely", "resume")
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
    if name in ("forget", "keep"):
        from mentor_app.memory import parse_note_id

        note_id = parse_note_id(rest)
        if note_id is None:
            return CommandResult("error", f"Try `/{name} 12` (the note id `/memory` shows).")
        return CommandResult(name, "", note_id)
    if name == "memory":
        return CommandResult("memory")
    if name == "recall":
        if not rest:
            return CommandResult("error", "Try `/recall NVDA`.")
        return CommandResult("recall", "", rest)
    if name == "tape":
        return CommandResult("tape")
    if name == "read":
        return CommandResult("read")
    if name == "pause":
        return CommandResult("pause")
    if name == "pick":
        parts = rest.upper().split()
        symbol = parts[0] if parts else ""
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            return CommandResult("error", "Try `/pick NVDA` (or `/pick NVDA short`).")
        side = parts[1] if len(parts) > 1 and parts[1] in ("LONG", "SHORT") else ""
        return CommandResult("pick", "", (symbol, side))
    if name == "debate":
        parts = rest.upper().split()
        symbol = parts[0] if parts else ""
        side = parts[1] if len(parts) > 1 else ""
        if (not symbol or not symbol.replace(".", "").replace("-", "").isalnum() or len(parts) > 2
                or side not in ("", "LONG", "SHORT")):
            return CommandResult("error", "Try `/debate NVDA` (or `/debate NVDA short`).")
        return CommandResult("debate", "", (symbol, side))
    if name == "news":
        from news_feed import clean_symbol

        parts = rest.split()
        symbol = clean_symbol(parts[0]) if parts else ""
        days_text = parts[1].lower().removesuffix("d") if len(parts) > 1 else "3"
        if not symbol or len(parts) > 2 or not days_text.isdigit() or not 1 <= int(days_text) <= 14:
            return CommandResult("error", "Try `/news NVDA` or `/news NVDA 7` (1 to 14 days).")
        return CommandResult("news", "", (symbol, int(days_text)))
    if name == "vetoes":
        if rest and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", rest):
            return CommandResult("error", "Try `/vetoes` or `/vetoes 2026-09-29`.")
        return CommandResult("vetoes", "", rest)
    if name == "check":
        from mentor_app.gate import parse_check

        request = parse_check(rest)
        if request is None:
            return CommandResult("error", "Try `/check short NVDA 400 stop 3.20 entry 3.05` (size, stop, entry are optional).")
        return CommandResult("check", "", request)
    if name == "scorecard":
        return CommandResult("scorecard")
    if name in ("hypotheses", "hyp"):
        if rest:
            return CommandResult("error", "Try `/hypotheses` (no arguments).")
        return CommandResult("hypotheses")
    if name == "mirror":
        if rest and not (rest.isdigit() and 1 <= int(rest) <= 52):
            return CommandResult("error", "Try `/mirror` or `/mirror 8` (1 to 52 weeks).")
        return CommandResult("mirror", "", int(rest) if rest else 6)
    if name == "tilt":
        if rest:
            return CommandResult("error", "Try `/tilt` (no arguments).")
        return CommandResult("tilt")
    if name == "book":
        if rest:
            return CommandResult("error", "Try `/book` (no arguments).")
        return CommandResult("book")
    if name == "ai":
        return _ai_command(rest)
    return CommandResult("error", f"I don't know `/{name}`. Type `/help`.")


def _ai_command(rest: str) -> CommandResult:
    """``/ai`` status, ``/ai on``, ``/ai off [2h|4h|tonight|30m|forever]`` (Pause AI, not `/pause`)."""
    words = rest.lower().split()
    if not words:
        return CommandResult("ai_status")
    if words == ["on"]:
        return CommandResult("ai_on")
    if words[0] != "off" or len(words) > 2:
        return CommandResult("error", AI_USAGE)
    arg = words[1] if len(words) > 1 else "tonight"
    if arg in FOREVER_WORDS:
        return CommandResult("ai_off", "", "until_resumed")
    if arg in ("tonight", "06:00", "6am"):
        return CommandResult("ai_off", "", "tonight")
    duration = parse_duration(arg)
    if duration is None:
        return CommandResult("error", AI_USAGE)
    return CommandResult("ai_off", "", duration)
