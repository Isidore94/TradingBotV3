"""The Trade Mentor app's settings and the GPU time-share clock. Qt-free.

Every value is read through ``project_paths.get_local_setting``. The night AI owns
the 5080 from 22:00 to 06:00 PT (its ai_jobs window) on tunnel port 11435; the app
stops using the model 15 minutes before that window opens and stays off until it
closes, on its own port (11436 by default).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import project_paths

SSH_ALIAS_KEY = "ai_remote_gpu_ssh_alias"
NIGHT_TUNNEL_PORT_KEY = "ai_remote_gpu_tunnel_port"
DEFAULT_NIGHT_TUNNEL_PORT = 11435

TUNNEL_PORT_KEY = "mentor_tunnel_port"
DEFAULT_TUNNEL_PORT = 11436
#: Used when the saved app port collides with the night's port.
FALLBACK_TUNNEL_PORT = 11437

MODEL_KEY = "mentor_model"
KEEP_ALIVE_KEY = "mentor_keep_alive"
DEFAULT_KEEP_ALIVE: Any = -1
PROACTIVE_PER_DAY_KEY = "mentor_proactive_per_day"
DEFAULT_PROACTIVE_PER_DAY = 6

#: Ollama's own port on the GPU host; the tunnel forwards to it.
REMOTE_OLLAMA_PORT = 11434
#: The app hands the GPU back this many minutes before the night window opens.
PRE_WINDOW_MINUTES = 15
EMBED_MODEL = "nomic-embed-text"
PREFETCH_SCOPE_KEY = "mentor_prefetch_scope"
PREFETCH_SCOPES = ("liked", "all")


def _setting(key: str, default: Any = None) -> Any:
    return project_paths.get_local_setting(key, default)


def _as_port(raw: Any, default: int) -> int:
    if isinstance(raw, bool):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if 0 < value < 65536 else default


def ssh_alias() -> str:
    return str(_setting(SSH_ALIAS_KEY, "") or "").strip()


def night_tunnel_port() -> int:
    return _as_port(_setting(NIGHT_TUNNEL_PORT_KEY, DEFAULT_NIGHT_TUNNEL_PORT), DEFAULT_NIGHT_TUNNEL_PORT)


def tunnel_port() -> int:
    """The app's local tunnel port; never the night's, so the two never fight over a bind."""
    port = _as_port(_setting(TUNNEL_PORT_KEY, DEFAULT_TUNNEL_PORT), DEFAULT_TUNNEL_PORT)
    night = night_tunnel_port()
    if port == night:
        port = DEFAULT_TUNNEL_PORT if DEFAULT_TUNNEL_PORT != night else FALLBACK_TUNNEL_PORT
    return port


def endpoint() -> str:
    """Base URL of the app's own tunnel (Ollama native API)."""
    return f"http://127.0.0.1:{tunnel_port()}"


def mentor_model() -> str:
    """The app's chat model; defaults to the night's medium model tag."""
    chosen = str(_setting(MODEL_KEY, "") or "").strip()
    if chosen:
        return chosen
    import ai_summary

    return ai_summary.local_model("medium")


def keep_alive() -> Any:
    raw = _setting(KEEP_ALIVE_KEY, DEFAULT_KEEP_ALIVE)
    if isinstance(raw, bool) or raw is None or raw == "":
        return DEFAULT_KEEP_ALIVE
    return raw


def context_tokens() -> int:
    import ai_summary

    return ai_summary.local_context_tokens()


def proactive_per_day() -> int:
    raw = _setting(PROACTIVE_PER_DAY_KEY, DEFAULT_PROACTIVE_PER_DAY)
    if isinstance(raw, bool):
        return DEFAULT_PROACTIVE_PER_DAY
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_PROACTIVE_PER_DAY


def prefetch_scope() -> str:
    """``liked`` (default): prefetch only the trader's liked picks; ``all``: then the rest of Focus when idle."""
    raw = str(_setting(PREFETCH_SCOPE_KEY, "liked") or "").strip().lower()
    return raw if raw in PREFETCH_SCOPES else "liked"


PUSH_BRIEF_KEY = "mentor_push_brief"


def push_brief_enabled() -> bool:
    """The one 06:30 PT tape line to the phone; off unless the setting is exactly true."""
    return _setting(PUSH_BRIEF_KEY, False) is True


LIKED_SOURCES_KEY = "mentor_liked_sources"
DEFAULT_LIKED_SOURCES = "claims,likes,favorites"


def liked_sources() -> frozenset[str]:
    """Which stores make a liked pick: ``claims,likes,favorites`` (+ opt-in ``likes_strength_board``)."""
    from mentor_app.pick_jobs import parse_liked_sources

    return parse_liked_sources(_setting(LIKED_SOURCES_KEY, DEFAULT_LIKED_SOURCES))


FRONTIER_ENABLED_KEY = "mentor_frontier_enabled"
FRONTIER_MODEL_KEY = "mentor_frontier_model"
FRONTIER_CAP_KEY = "mentor_frontier_daily_usd_cap"
#: The current Sonnet id per the claude-api skill (model table cached 2026-06-24).
DEFAULT_FRONTIER_MODEL = "claude-sonnet-5"
DEFAULT_FRONTIER_CAP_USD = 2.00


def frontier_enabled() -> bool:
    """The metered frontier switch; off unless the setting is exactly true."""
    return _setting(FRONTIER_ENABLED_KEY, False) is True


def frontier_model() -> str:
    return str(_setting(FRONTIER_MODEL_KEY, "") or "").strip() or DEFAULT_FRONTIER_MODEL


def frontier_daily_cap_usd() -> float:
    """The daily USD cap; an unreadable or negative value is the default."""
    raw = _setting(FRONTIER_CAP_KEY, DEFAULT_FRONTIER_CAP_USD)
    if isinstance(raw, bool):
        return DEFAULT_FRONTIER_CAP_USD
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_FRONTIER_CAP_USD
    return value if value >= 0 else DEFAULT_FRONTIER_CAP_USD


def gpu_block_reason(now: datetime | None = None) -> str:
    """"" while the app may use the model; otherwise why it may not.

    Blocked while Pause AI is on, inside the ai_jobs night window and for
    PRE_WINDOW_MINUTES before it.
    """
    import ai_pause
    from ai_jobs.window import in_offhours_window

    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    paused = ai_pause.reason(moment)
    if paused:
        return paused
    if in_offhours_window(moment):
        return "the night AI owns the GPU until its window closes (06:00 PT)"
    if in_offhours_window(moment + timedelta(minutes=PRE_WINDOW_MINUTES)):
        return f"the night AI starts within {PRE_WINDOW_MINUTES} minutes; the model is handed back"
    return ""
