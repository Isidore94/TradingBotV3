"""The coverage ledger: which (broker, account, day) the journal actually has.

R7 §9 step 6, root causes A3/A5/A6, invariant I2.

Before this, ``import_runs`` recorded that an import ran and how many rows it
wrote. It did not record **which days it looked at**, so a gap was structurally
undetectable: a day the import never covered and a day with genuinely no trades
looked identical, and a failed EOD slot left a hole nothing would ever go back
for. The trader's report - "the journal misses trades" - is that hole.

THE HONESTY RULE (I2)

A day is ``COVERED`` only when an import actually spanned it and said so.
Nothing here infers coverage from a row count, from a neighbouring day, or from
an import that "probably" included it. The four states:

* ``COVERED``    - an import spanned this day successfully. It may hold zero
                   executions; that is a fact, not a gap.
* ``FAILED``     - an import tried and could not. Carries the error and an
                   attempt count, so a permanently broken day stops being
                   retried forever but never stops being visible.
* ``PENDING``    - claimed by an import that has not reported back.
* ``NO_SESSION`` - the market was closed. Not a gap, and never retried.

The IBKR **socket** importer never marks coverage. It can only see the current
TWS session, so a successful socket pull says nothing about any other day - and
a ledger that recorded it as coverage would be lying in exactly the way this
module exists to stop.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from typing import Any

import market_calendar
from project_paths import get_local_setting

COVERED = "COVERED"
FAILED = "FAILED"
PENDING = "PENDING"
NO_SESSION = "NO_SESSION"
COVERAGE_STATUSES = frozenset({COVERED, FAILED, PENDING, NO_SESSION})

#: How much history one night's self-heal may try to repair. Two months of
#: sessions is enough to close any realistic gap without turning a nightly job
#: into an unbounded backfill that hammers a broker API until morning.
DEFAULT_MAX_DAYS_PER_NIGHT = 62

#: After this many failures a day stops being retried automatically. It stays
#: FAILED and stays visible - the trader repairs it from the Health tab. A day
#: that fails deterministically would otherwise consume the whole nightly
#: budget every night, forever, and starve the days that could succeed.
DEFAULT_MAX_ATTEMPTS_PER_DAY = 5

#: Where the ledger starts. Per broker, falling back to a global setting.
INCEPTION_SETTING = "journal_inception_date"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def is_trading_day(day: date) -> bool:
    """Is ``day`` a session? Fail **open** when the calendar cannot answer.

    ``market_calendar`` refuses to extrapolate outside 2000-2032. An unknown day
    is treated as a session, so it shows up as a gap that wants covering rather
    than being quietly filed as a market holiday. Under I2 the failure mode has
    to be "visible work" and never "invisible hole".
    """
    try:
        return market_calendar.is_session(day)
    except market_calendar.SessionCalendarError:
        return True


def inception_date(broker: str) -> date | None:
    """The first day this broker's ledger is expected to cover.

    Per broker (``journal_inception_date_questrade``) with a global fallback
    (``journal_inception_date``). Unset means "no horizon declared", and the
    caller decides - `find_gaps` will simply not look further back than the
    range it is given.
    """
    broker_key = f"{INCEPTION_SETTING}_{str(broker or '').strip().lower()}"
    for key in (broker_key, INCEPTION_SETTING):
        raw = get_local_setting(key, "")
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            return _as_date(text)
        except ValueError:
            continue
    return None


def mark_coverage(
    store: Any,
    *,
    broker: str,
    account_number: str,
    day: Any,
    status: str,
    source: str = "",
    import_run_id: int | None = None,
    message: str = "",
) -> None:
    """Record one (broker, account, day) outcome.

    ``attempts`` only ever counts failures, and only ever goes up. A day that
    failed four times and then succeeded keeps its history of four - that is the
    number the Health tab needs to show a day that is fighting back.
    """
    normalized = str(status or "").strip().upper()
    if normalized not in COVERAGE_STATUSES:
        raise ValueError(f"unsupported coverage status: {status!r}")
    day_value = _as_date(day)
    # A closed market is never a gap, whatever the caller believes.
    if not is_trading_day(day_value):
        normalized = NO_SESSION
    with store.connection() as conn:
        _mark_one(
            conn,
            broker=broker,
            account_number=account_number,
            day=day_value,
            status=normalized,
            source=source,
            import_run_id=import_run_id,
            message=message,
        )


def _mark_one(
    conn: sqlite3.Connection,
    *,
    broker: str,
    account_number: str,
    day: date,
    status: str,
    source: str,
    import_run_id: int | None,
    message: str,
) -> None:
    conn.execute(
        """
        INSERT INTO import_coverage(
            broker, account_number, day, status, source, import_run_id, attempts, message, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(broker, account_number, day) DO UPDATE SET
            status = excluded.status,
            source = excluded.source,
            import_run_id = excluded.import_run_id,
            attempts = import_coverage.attempts + (CASE WHEN excluded.status = 'FAILED' THEN 1 ELSE 0 END),
            message = excluded.message,
            updated_at = excluded.updated_at
        """,
        (
            str(broker or "").upper(),
            str(account_number or ""),
            day.isoformat(),
            status,
            str(source or ""),
            import_run_id,
            1 if status == FAILED else 0,
            str(message or "")[:500],
            _now_iso(),
        ),
    )


def mark_range(
    store: Any,
    *,
    broker: str,
    account_number: str,
    start: Any,
    end: Any,
    status: str,
    source: str = "",
    import_run_id: int | None = None,
    message: str = "",
) -> int:
    """Mark every day in an inclusive range. Non-session days become NO_SESSION.

    This is what an importer calls after a chunk succeeds or fails: the chunk's
    own span, and nothing else. A failed chunk marks only its own days, which is
    the whole point of A5's fix - one bad chunk used to discard the entire pull.
    """
    normalized = str(status or "").strip().upper()
    if normalized not in COVERAGE_STATUSES:
        raise ValueError(f"unsupported coverage status: {status!r}")
    first, last = _as_date(start), _as_date(end)
    if last < first:
        return 0
    marked = 0
    with store.connection() as conn:
        cursor = first
        while cursor <= last:
            _mark_one(
                conn,
                broker=broker,
                account_number=account_number,
                day=cursor,
                status=normalized if is_trading_day(cursor) else NO_SESSION,
                source=source,
                import_run_id=import_run_id,
                message=message,
            )
            marked += 1
            cursor += timedelta(days=1)
    return marked


def coverage_rows(
    store: Any,
    *,
    broker: str = "",
    account_number: str = "",
    start: Any = None,
    end: Any = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if str(broker or "").strip():
        clauses.append("broker = ?")
        params.append(str(broker).upper())
    if str(account_number or "").strip():
        clauses.append("account_number = ?")
        params.append(str(account_number))
    if start is not None:
        clauses.append("day >= ?")
        params.append(_as_date(start).isoformat())
    if end is not None:
        clauses.append("day <= ?")
        params.append(_as_date(end).isoformat())
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with store.connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM import_coverage {where} ORDER BY broker, account_number, day", params
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def find_gaps(
    store: Any,
    *,
    broker: str,
    account_number: str,
    start: Any,
    end: Any,
) -> list[date]:
    """Session days in the range that are not COVERED.

    A day with no ledger row at all is a gap: absence of evidence is not
    coverage. ``NO_SESSION`` days are skipped, and a ``FAILED`` day is a gap -
    it is exactly the day that needs another attempt.
    """
    first, last = _as_date(start), _as_date(end)
    if last < first:
        return []
    known = {
        _as_date(row["day"]): str(row["status"] or "")
        for row in coverage_rows(
            store, broker=broker, account_number=account_number, start=first, end=last
        )
    }
    gaps: list[date] = []
    cursor = first
    while cursor <= last:
        status = known.get(cursor, "")
        if status not in {COVERED, NO_SESSION} and is_trading_day(cursor):
            gaps.append(cursor)
        cursor += timedelta(days=1)
    return gaps


def attempts_for(store: Any, *, broker: str, account_number: str, day: Any) -> int:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT attempts FROM import_coverage WHERE broker = ? AND account_number = ? AND day = ?",
            (str(broker or "").upper(), str(account_number or ""), _as_date(day).isoformat()),
        ).fetchone()
    return int(row[0]) if row else 0


def known_accounts(store: Any, broker: str = "") -> list[tuple[str, str]]:
    """(broker, account_number) pairs the ledger or the account table knows."""
    clause = "WHERE broker = ?" if str(broker or "").strip() else ""
    params = [str(broker).upper()] if clause else []
    with store.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT broker, account_number FROM accounts {clause}
            UNION
            SELECT broker, account_number FROM import_coverage {clause}
            ORDER BY broker, account_number
            """,
            params * 2,
        ).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def self_heal(
    store: Any,
    fetch: Callable[[str, str, date], int],
    *,
    accounts: Iterable[tuple[str, str]] | None = None,
    today: date | None = None,
    lookback_days: int = 365,
    max_days_per_night: int = DEFAULT_MAX_DAYS_PER_NIGHT,
    max_attempts_per_day: int = DEFAULT_MAX_ATTEMPTS_PER_DAY,
) -> dict[str, Any]:
    """Repair gaps and retry failures, oldest first, within a night's budget.

    ``fetch(broker, account_number, day)`` imports one day and returns how many
    executions it wrote; raising marks the day FAILED with the message. The
    callback keeps this module free of any broker knowledge, which is what makes
    it testable without a network.

    Oldest first on purpose: the oldest gap is the one most likely to fall out
    of a broker's retention window, so it is the one that gets tonight's budget.

    A day that has failed ``max_attempts_per_day`` times is skipped and counted
    as ``exhausted``. It stays FAILED and stays visible; it just stops eating
    the budget every night.
    """
    reference = today or date.today()
    horizon_start = reference - timedelta(days=max(1, int(lookback_days)))
    # Yesterday: today is still being traded, and marking it covered would
    # freeze a partial day as complete.
    horizon_end = reference - timedelta(days=1)

    pairs = list(accounts) if accounts is not None else known_accounts(store)
    summary: dict[str, Any] = {
        "attempted": [],
        "repaired": [],
        "failed": [],
        "exhausted": [],
        "budget": int(max_days_per_night),
        "budget_exhausted": False,
    }
    if not pairs:
        return summary

    work: list[tuple[date, str, str]] = []
    for broker, account_number in pairs:
        start = horizon_start
        declared = inception_date(broker)
        if declared and declared > start:
            start = declared
        for day in find_gaps(
            store, broker=broker, account_number=account_number, start=start, end=horizon_end
        ):
            work.append((day, str(broker), str(account_number)))
    work.sort()

    budget = max(0, int(max_days_per_night))
    for day, broker, account_number in work:
        if len(summary["attempted"]) >= budget:
            summary["budget_exhausted"] = True
            break
        if attempts_for(store, broker=broker, account_number=account_number, day=day) >= max_attempts_per_day:
            summary["exhausted"].append({"broker": broker, "account": account_number, "day": day.isoformat()})
            continue
        summary["attempted"].append({"broker": broker, "account": account_number, "day": day.isoformat()})
        try:
            count = int(fetch(broker, account_number, day))
        except Exception as exc:  # noqa: BLE001 - the reason is the payload
            mark_coverage(
                store,
                broker=broker,
                account_number=account_number,
                day=day,
                status=FAILED,
                source="self_heal",
                message=str(exc),
            )
            summary["failed"].append(
                {"broker": broker, "account": account_number, "day": day.isoformat(), "message": str(exc)}
            )
            continue
        mark_coverage(
            store,
            broker=broker,
            account_number=account_number,
            day=day,
            status=COVERED,
            source="self_heal",
            message=f"{count} execution(s)",
        )
        summary["repaired"].append(
            {"broker": broker, "account": account_number, "day": day.isoformat(), "executions": count}
        )
    return summary
