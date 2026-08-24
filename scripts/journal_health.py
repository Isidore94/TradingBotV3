"""Is the Questrade credential chain alive? (AI-P4, review 2026-08-24 §5)

Questrade issues **single-use** refresh tokens: every successful refresh returns
the next one, and the old one dies with it. So the chain has exactly one thread
holding it, and once that thread breaks — a refresh that half-succeeded, a token
used twice, three days of disuse — nothing reconnects it but the trader pasting
a fresh token from the Questrade portal. There is no retry that fixes it.

That is why this needs a surface rather than a log line. On 2026-08-24 the chain
had been dead since 2026-08-19: **0 of 142 Questrade session days covered**, 56
identical `500 Server Error ... oauth2/token` rows, one whole broker (including
a TFSA) missing from the journal, from walk-away analysis, and from everything
the AI layer reads. Nothing on the desk said so. It was found by opening the
SQLite database by hand.

Deliberately Qt-free and free of :mod:`ui`, so both the Journal Health tab and
:mod:`operations_audit` (which renders inside System Health, frozen) can read the
same verdict without either importing the other's world.

**Honesty rules, which are the whole point of the module:**

* An absent setting is ``not_configured`` — never ``ok``. "Nobody set this up"
  and "this is working" are different answers and must never render the same.
* A database it cannot read is ``unknown`` — never ``ok``. Missing data is
  uncertainty, never confirmation (plan.md sec 5).
* Nothing here writes a number it did not measure (Phase 0.7 ground rule 6):
  with no recorded refresh, the age is ``None`` and stays unstated rather than
  being reported as zero.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

#: How long since the last SUCCESSFUL refresh before the chain is presumed
#: broken even with no error recorded.
#:
#: Three days because that is Questrade's own refresh-token lifetime: a chain
#: unused for longer than this is dead whether or not anything has tried it yet.
#: The threshold is stated in the rendered text rather than left implicit, so a
#: reader can disagree with it without reading this file.
QUESTRADE_STALE_AFTER_DAYS = 3

#: What a broken chain looks like in a coverage row's message. Matched loosely
#: on purpose: the failure arrives as a 400, a 401 or a 500 depending on how the
#: chain broke, and all three are the same problem with the same fix.
_OAUTH_FAILURE = re.compile(r"oauth2?/token|grant_type=refresh_token", re.IGNORECASE)

STATE_OK = "ok"
STATE_DEAD = "dead"
STATE_STALE = "stale"
STATE_NOT_CONFIGURED = "not_configured"
STATE_UNKNOWN = "unknown"

#: The states that must show the trader something. ``ok`` and
#: ``not_configured`` are quiet: one is fine, and the other is a machine that
#: was never asked to do this.
ALERTING_STATES = frozenset({STATE_DEAD, STATE_STALE})

REFRESH_TOKEN_SETTING = "journal_questrade_refresh_token"
EXPIRES_AT_SETTING = "journal_questrade_expires_at"

#: One sentence, in the trader's terms, naming the surface that fixes it.
REPAIR_ACTION = (
    "Get a new refresh token from the Questrade portal (Apps -> your app -> "
    "generate) and paste it into Journal > Health > 'Questrade refresh token', "
    "then Save. Questrade refresh tokens are single-use, so a broken chain "
    "cannot heal itself and no retry will fix it."
)


def _default_settings_reader() -> Callable[[str, Any], Any]:
    from project_paths import get_local_setting

    return get_local_setting


def _parse_stamp(text: str) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _age_days(moment: datetime, stamp: datetime) -> float:
    """Days between two datetimes that may not agree about tzinfo.

    `journal_importers` writes the stamp with a bare `datetime.now()`, so it is
    naive machine-local. Callers differ: the Journal tab passes nothing and gets
    a naive `datetime.now()`, while `operations_audit` passes a timezone-AWARE
    moment. Subtracting those raises, which is how this first ran.

    Normalized by ATTACHING the caller's zone to the naive side, never by
    stripping the aware side (CLAUDE.md, `_gate_moment`): both values are the
    same wall clock on this desk, so attaching preserves the instant while
    stripping would silently discard an offset. The reverse case - an aware
    stamp against a naive caller - is normalized the same direction.
    """
    if (moment.tzinfo is None) != (stamp.tzinfo is None):
        if moment.tzinfo is not None:
            stamp = stamp.replace(tzinfo=moment.tzinfo)
        else:
            moment = moment.replace(tzinfo=stamp.tzinfo)
    return round((moment - stamp).total_seconds() / 86400.0, 2)


def _last_oauth_failure(db_path: Path) -> dict[str, str] | None:
    """The newest Questrade auth failure, or None.

    Returns None for "no such failure"; raises for "could not look", because
    the caller must tell those apart.

    ``day`` and ``recorded_at`` are DIFFERENT dates and the distinction is
    load-bearing. A backfill spans a year, so one broken chain writes a FAILED
    row for every session day in the range - the oldest of which can be a year
    back. ``day`` is the trading day left uncovered; ``recorded_at`` is when
    the attempt failed. Reporting ``day`` as though it were the failure date
    tells the trader their chain broke in 2025, which is how this function
    read on its first run.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT day, message, updated_at FROM import_coverage "
            "WHERE broker = 'QUESTRADE' AND status = 'FAILED' "
            "ORDER BY updated_at DESC, day DESC LIMIT 200"
        ).fetchall()
    for day, message, updated in rows:
        if _OAUTH_FAILURE.search(str(message or "")):
            return {
                "day": str(day or ""),
                "message": str(message or ""),
                "recorded_at": str(updated or ""),
            }
    return None


def questrade_chain_health(
    *,
    now: datetime | None = None,
    db_path: Path | None = None,
    settings_reader: Callable[[str, Any], Any] | None = None,
) -> dict[str, Any]:
    """Classify the Questrade credential chain. Never raises."""
    moment = now or datetime.now()
    read_setting = settings_reader or _default_settings_reader()

    def _verdict(state: str, headline: str, **extra: Any) -> dict[str, Any]:
        return {
            "state": state,
            "headline": headline,
            "action": REPAIR_ACTION if state in ALERTING_STATES else "",
            "stale_after_days": QUESTRADE_STALE_AFTER_DAYS,
            "last_refresh_at": None,
            "days_since_refresh": None,
            "last_failure_day": None,
            "last_failure_at": None,
            "last_failure": None,
            **extra,
        }

    try:
        token = str(read_setting(REFRESH_TOKEN_SETTING, "") or "").strip()
    except Exception:  # noqa: BLE001 - an unreadable setting is unknown, not absent
        return _verdict(
            STATE_UNKNOWN,
            "Questrade credentials could not be read, so whether the chain is "
            "alive is UNKNOWN.",
        )
    if not token:
        return _verdict(
            STATE_NOT_CONFIGURED,
            "Questrade is not configured on this machine - no refresh token is "
            "set. This is not a failure; it is a broker this desk was never "
            "asked to import.",
        )

    refreshed_at = _parse_stamp(read_setting(EXPIRES_AT_SETTING, "") or "")
    age_days: float | None = None
    if refreshed_at is not None:
        age_days = _age_days(moment, refreshed_at)

    path = Path(db_path) if db_path is not None else None
    if path is None:
        from project_paths import JOURNAL_DB_FILE

        path = Path(JOURNAL_DB_FILE)

    failure: dict[str, str] | None = None
    if path.exists():
        try:
            failure = _last_oauth_failure(path)
        except Exception:  # noqa: BLE001
            return _verdict(
                STATE_UNKNOWN,
                "The journal database could not be read, so whether the "
                "Questrade chain is alive is UNKNOWN. It is not therefore fine.",
                last_refresh_at=refreshed_at.isoformat() if refreshed_at else None,
                days_since_refresh=age_days,
            )
    else:
        return _verdict(
            STATE_UNKNOWN,
            "No journal database exists yet, so the Questrade chain's health is "
            "UNKNOWN - nothing has tried to use it.",
            last_refresh_at=refreshed_at.isoformat() if refreshed_at else None,
            days_since_refresh=age_days,
        )

    common = {
        "last_refresh_at": refreshed_at.isoformat() if refreshed_at else None,
        "days_since_refresh": age_days,
        "last_failure_day": failure["day"] if failure else None,
        "last_failure_at": failure["recorded_at"] if failure else None,
        "last_failure": failure["message"] if failure else None,
    }

    # A recorded auth failure outranks a fresh-looking stamp: the stamp only
    # records the last refresh that WORKED, so a chain that broke this morning
    # still carries this morning's timestamp.
    if failure is not None:
        age_text = (
            f" The last successful refresh was {age_days} day(s) ago."
            if age_days is not None
            else ""
        )
        # The date shown is when the attempt FAILED, not the trading day it
        # left uncovered - one broken chain marks a year of days failed, and
        # naming the oldest of those would report a 2025 outage for a chain
        # that broke last week.
        when = failure["recorded_at"] or "an unrecorded time"
        return _verdict(
            STATE_DEAD,
            f"The Questrade refresh chain is BROKEN - the import failed to "
            f"authenticate (last attempt {when}). No Questrade trade has "
            f"reached the journal since.{age_text}",
            **common,
        )
    if age_days is None:
        return _verdict(
            STATE_UNKNOWN,
            "A Questrade token is configured but no successful refresh has ever "
            "been recorded, so the chain's health is UNKNOWN.",
            **common,
        )
    if age_days > QUESTRADE_STALE_AFTER_DAYS:
        return _verdict(
            STATE_STALE,
            f"The Questrade chain has not refreshed successfully for "
            f"{age_days} day(s). Questrade refresh tokens expire after "
            f"{QUESTRADE_STALE_AFTER_DAYS} days, so it is probably already dead.",
            **common,
        )
    return _verdict(
        STATE_OK,
        f"The Questrade chain refreshed successfully {age_days} day(s) ago.",
        **common,
    )
