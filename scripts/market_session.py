from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, tzinfo
import zoneinfo

from project_paths import get_local_setting


NYSE_TIMEZONE_NAME = "America/New_York"
DEFAULT_LOCAL_FALLBACK_TIMEZONE = "America/Los_Angeles"
LOCAL_MARKET_TIMEZONE_SETTING_KEYS = (
    "market_local_timezone",
    "market_timezone",
)
LOCAL_MARKET_TIMEZONE_ENV = "TRADINGBOT_MARKET_TIMEZONE"
NYSE_REGULAR_OPEN = dt_time(hour=9, minute=30)
NYSE_REGULAR_CLOSE = dt_time(hour=16, minute=0)


@dataclass(frozen=True)
class MarketSessionWindow:
    local_timezone_name: str
    market_timezone_name: str
    market_date: date
    open_local: datetime
    close_local: datetime

    @property
    def last_hour_start_local(self) -> datetime:
        return self.close_local - timedelta(hours=1)

    @property
    def open_label(self) -> str:
        return self.open_local.strftime("%H:%M")

    @property
    def close_label(self) -> str:
        return self.close_local.strftime("%H:%M")

    @property
    def last_hour_start_label(self) -> str:
        return self.last_hour_start_local.strftime("%H:%M")

    @property
    def session_label(self) -> str:
        return f"{self.open_label}-{self.close_label}"

    @property
    def last_hour_label(self) -> str:
        return f"{self.last_hour_start_label}-{self.close_label}"


def _resolve_configured_timezone_name(explicit_name: str | None = None) -> str | None:
    candidates = [explicit_name, os.environ.get(LOCAL_MARKET_TIMEZONE_ENV)]
    for setting_key in LOCAL_MARKET_TIMEZONE_SETTING_KEYS:
        candidates.append(get_local_setting(setting_key))

    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def _coerce_zoneinfo(name: str | None) -> zoneinfo.ZoneInfo | None:
    value = str(name or "").strip()
    if not value:
        return None
    try:
        return zoneinfo.ZoneInfo(value)
    except Exception:
        return None


def get_market_local_timezone(local_timezone_name: str | None = None) -> tuple[tzinfo, str]:
    configured_name = _resolve_configured_timezone_name(local_timezone_name)
    configured_tz = _coerce_zoneinfo(configured_name)
    if configured_tz is not None and configured_name:
        return configured_tz, configured_name

    system_tz = datetime.now().astimezone().tzinfo
    system_name = getattr(system_tz, "key", None) or getattr(system_tz, "zone", None)
    if system_name:
        system_zoneinfo = _coerce_zoneinfo(system_name)
        if system_zoneinfo is not None:
            return system_zoneinfo, system_name
    if system_tz is not None:
        return system_tz, system_name or str(system_tz)

    fallback = zoneinfo.ZoneInfo(DEFAULT_LOCAL_FALLBACK_TIMEZONE)
    return fallback, DEFAULT_LOCAL_FALLBACK_TIMEZONE


def _normalize_reference_datetime(
    reference: datetime | date | None,
    local_tz: tzinfo,
) -> datetime:
    if reference is None:
        return datetime.now().astimezone(local_tz)
    if isinstance(reference, date) and not isinstance(reference, datetime):
        return datetime.combine(reference, dt_time(hour=12), tzinfo=local_tz)
    if reference.tzinfo is None:
        return reference.replace(tzinfo=local_tz)
    return reference.astimezone(local_tz)


def get_market_local_now(local_timezone_name: str | None = None) -> datetime:
    local_tz, _ = get_market_local_timezone(local_timezone_name)
    return datetime.now().astimezone(local_tz)


def get_market_session_window(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> MarketSessionWindow:
    local_tz, local_name = get_market_local_timezone(local_timezone_name)
    local_reference = _normalize_reference_datetime(reference, local_tz)
    ny_tz = zoneinfo.ZoneInfo(NYSE_TIMEZONE_NAME)
    ny_reference = local_reference.astimezone(ny_tz)
    market_date = ny_reference.date()
    open_market = datetime.combine(market_date, NYSE_REGULAR_OPEN, tzinfo=ny_tz)
    close_market = datetime.combine(market_date, NYSE_REGULAR_CLOSE, tzinfo=ny_tz)
    return MarketSessionWindow(
        local_timezone_name=local_name,
        market_timezone_name=NYSE_TIMEZONE_NAME,
        market_date=market_date,
        open_local=open_market.astimezone(local_tz),
        close_local=close_market.astimezone(local_tz),
    )


def get_market_session_labels(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> tuple[str, str]:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    return session.open_label, session.close_label


def get_last_hour_window_labels(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> tuple[str, str]:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    return session.last_hour_start_label, session.close_label


def get_default_setup_tracker_refresh_slot(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> str:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    return session.last_hour_start_label


def _round_up_to_hour(value: datetime) -> datetime:
    rounded = value.replace(minute=0, second=0, microsecond=0)
    if value.minute or value.second or value.microsecond:
        rounded += timedelta(hours=1)
    return rounded


def get_default_hourly_scan_schedule(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> tuple[str, ...]:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    slots: list[str] = []
    cursor = _round_up_to_hour(session.open_local)
    while cursor <= session.close_local:
        slots.append(cursor.strftime("%H:%M"))
        cursor += timedelta(hours=1)
    return tuple(slots)


def get_default_stop_time_label(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
    grace_minutes: int = 30,
) -> str:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    stop_dt = session.close_local + timedelta(minutes=max(0, int(grace_minutes)))
    return stop_dt.strftime("%H:%M")


def is_within_regular_market_session(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> bool:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    local_tz, _ = get_market_local_timezone(local_timezone_name)
    local_reference = _normalize_reference_datetime(reference, local_tz)
    return session.open_local <= local_reference <= session.close_local


def get_market_session_open_naive(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> datetime:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    return session.open_local.replace(tzinfo=None)


def get_market_session_close_naive(
    reference: datetime | date | None = None,
    local_timezone_name: str | None = None,
) -> datetime:
    session = get_market_session_window(reference=reference, local_timezone_name=local_timezone_name)
    return session.close_local.replace(tzinfo=None)
