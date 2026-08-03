"""Machine-local startup role for the one Trading Desk launcher.

The role is intentionally selected before MainWindow construction.  A main
desk owns TWS/scanner services; a satellite desk consumes Desk Link instead.
Changing that ownership in a live process would violate the one-owner rule,
so Settings persists the choice and restarts through ``launch_gui.py``.
"""

from __future__ import annotations

from project_paths import get_local_setting, save_local_setting

DESK_ROLE_SETTING = "trading_desk_role"
ROLE_MAIN = "main"
ROLE_SATELLITE = "satellite"
VALID_DESK_ROLES = (ROLE_MAIN, ROLE_SATELLITE)


def normalize_desk_role(value: object) -> str:
    role = str(value or "").strip().lower()
    return role if role in VALID_DESK_ROLES else ROLE_MAIN


def saved_desk_role() -> str:
    return normalize_desk_role(get_local_setting(DESK_ROLE_SETTING, ROLE_MAIN))


def save_desk_role(role: str) -> str:
    normalized = normalize_desk_role(role)
    save_local_setting(DESK_ROLE_SETTING, normalized)
    return normalized


def startup_desk_role(*, explicit: str | None = None, legacy_satellite: bool = False) -> str:
    """Role for this launch; explicit compatibility flags also become the default."""
    if legacy_satellite:
        return save_desk_role(ROLE_SATELLITE)
    if explicit is not None:
        return save_desk_role(explicit)
    return saved_desk_role()
