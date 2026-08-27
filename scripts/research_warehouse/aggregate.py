"""Session table and deterministic aggregation (plan sec 5.4, Phase 4).

Two jobs live here, both EOD build-job work:

* publish ``trading_session`` rows from the versioned exchange calendar;
* derive ``bar_derived`` - M15/M30/H1 from canonical M5, and W1 from canonical
  D1 - under an explicit ``aggregation_contract_id``.

The contract is the point. A derived bar is only meaningful next to the rule
that produced it, so every row records the contract, how many constituents were
expected, how many arrived, and whether it is an end-of-session stub. The v1
RTH contract (09:30-16:00 ET, session-anchored) yields 26 M15, 13 M30, and 7 H1
bars per full session - six full hours plus a 15:30-16:00 stub that carries its
true 30-minute duration and must never be compared with a full hour as
equivalent. Half days use the half-day variant of the same contract.

H1 boundaries deliberately match IB's native ``useRTH=1`` hourly bars, which is
what makes the sentinel derived-vs-native parity check meaningful.

Rules that are not negotiable here:

* **completed bars only** - a bucket whose interval has not closed is not
  published at all, and a week is not published until its final session closes;
* **missing constituents are visible** - a short bucket is PARTIAL with its
  real counts, never silently averaged into a full bar;
* **provider-native D1 stays canonical** - W1 is derived from it, and an
  intraday-derived D1 would be a validation variant, never a replacement
  (LD-24).

Derived aggregates are RTH-only in v1. The session-scope stays in the contract
id, so an ETH variant later is an additive contract rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

try:  # package import
    from . import exchange_calendar as xcal
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import exchange_calendar as xcal  # type: ignore
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION  # type: ignore
    from store import ResearchStore  # type: ignore

QUALITY_COMPLETE = "COMPLETE"
QUALITY_PARTIAL = "PARTIAL"

TIMEFRAME_MINUTES = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}
#: H4 activates post-slice with the H4 feature series (sec 19.4).
SLICE_TIMEFRAMES = ("M15", "M30", "H1")

CONTRACT_PREFIX = "xnys_rth"
CONTRACT_VERSION = "v1"

# Worst-first ordering for input_capture_mode_worst: a bar built from any
# BACKFILL constituent is not LIVE evidence, whatever the others were.
_CAPTURE_MODE_RANK = {"LIVE": 0, "DELAYED": 1, "BACKFILL": 2, "RECONSTRUCTED": 3, "": 4}


def aggregation_contract_id(timeframe: str, *, half_day: bool = False) -> str:
    """The contract that produced a bar: session scope, timeframe, version."""
    scope = f"{CONTRACT_PREFIX}_half" if half_day else CONTRACT_PREFIX
    return f"{scope}_{str(timeframe).lower()}_{CONTRACT_VERSION}"


W1_CONTRACT_ID = f"{CONTRACT_PREFIX}_w1_from_d1_{CONTRACT_VERSION}"


@dataclass
class AggregateReport:
    dataset: str = "bar_derived"
    status: str = "OK"  # OK | DISABLED | NOTHING_TO_DERIVE
    sessions: int = 0
    rows_published: int = 0
    rows_quarantined: int = 0
    by_timeframe: dict = field(default_factory=dict)
    stubs: int = 0
    partial: int = 0
    skipped_forming: int = 0


@dataclass
class SessionTableReport:
    dataset: str = "trading_session"
    status: str = "OK"  # OK | DISABLED | ALREADY_PUBLISHED
    rows: int = 0
    half_days: int = 0
    holidays_skipped: int = 0


def session_buckets(session: xcal.TradingSession, timeframe: str):
    """Session-anchored buckets for one timeframe.

    Yields (start, end, expected_m5_constituents, is_stub). The final bucket is
    truncated at the session close; that stub keeps its true duration instead of
    being padded or dropped.
    """
    minutes = TIMEFRAME_MINUTES.get(str(timeframe).upper())
    if minutes is None:
        raise ValueError(f"unsupported derived timeframe {timeframe!r}; known: {sorted(TIMEFRAME_MINUTES)}")
    step = timedelta(minutes=minutes)
    start = session.rth_open_at
    while start < session.rth_close_at:
        end = min(start + step, session.rth_close_at)
        span = int((end - start).total_seconds() // 60)
        yield start, end, span // 5, span < minutes
        start = end


def _worst_capture_mode(modes) -> str:
    worst = ""
    rank = -1
    for mode in modes:
        value = _CAPTURE_MODE_RANK.get(str(mode or ""), 4)
        if value > rank:
            rank, worst = value, str(mode or "")
    return worst


def derive_session_bars(
    m5_rows,
    session: xcal.TradingSession,
    timeframe: str,
    *,
    as_of: datetime,
    computed_at: datetime | None = None,
    run_id: str = "",
) -> list[dict]:
    """Aggregate one symbol-session of M5 bars into one derived timeframe.

    ``m5_rows`` are canonical ``bar_m5`` rows for a single symbol. Only
    completed, RTH constituents are used: a forming M5 bar is preview, and an
    ETH bar belongs to a different (future) contract.
    """
    stamp = computed_at or utc_now()
    usable = []
    for row in m5_rows:
        start = row.get("interval_start")
        if start is None or not row.get("is_complete", True):
            continue
        if start.tzinfo is None:
            continue
        start = start.astimezone(timezone.utc)
        if not (session.rth_open_at <= start < session.rth_close_at):
            continue
        usable.append((start, row))
    usable.sort(key=lambda item: item[0])

    rows: list[dict] = []
    symbol = str(usable[0][1].get("symbol")) if usable else ""
    for start, end, expected, is_stub in session_buckets(session, timeframe):
        if end > as_of:
            continue  # the bucket has not closed: preview, never evidence
        members = [row for stamp_, row in usable if start <= stamp_ < end]
        if not members:
            continue  # no constituents at all: a gap, not a zero-volume bar
        highs = [row.get("high") for row in members if row.get("high") is not None]
        lows = [row.get("low") for row in members if row.get("low") is not None]
        rows.append(
            {
                "symbol": symbol or str(members[0].get("symbol") or ""),
                "timeframe": str(timeframe).upper(),
                "aggregation_contract_id": aggregation_contract_id(timeframe, half_day=session.is_half_day),
                "interval_start": start,
                "interval_end": end,
                "session_id": session.session_id,
                "open": members[0].get("open"),
                "high": max(highs) if highs else None,
                "low": min(lows) if lows else None,
                "close": members[-1].get("close"),
                "volume": int(sum(int(row.get("volume") or 0) for row in members)),
                "is_stub": bool(is_stub),
                "stub_duration_min": int((end - start).total_seconds() // 60) if is_stub else None,
                "constituent_count": len(members),
                "constituent_expected": expected,
                "is_complete": len(members) == expected,
                "quality": QUALITY_COMPLETE if len(members) == expected else QUALITY_PARTIAL,
                "event_at": end,
                "computed_at": stamp,
                "input_capture_mode_worst": _worst_capture_mode(row.get("capture_mode") for row in members),
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    return rows


def build_trading_sessions(
    store: ResearchStore | None,
    start: date,
    end: date,
    *,
    now: datetime | None = None,
    run_id: str = "",
    job_id: str = "trading_session",
) -> SessionTableReport:
    """Publish the calendar's sessions for a date range. Idempotent."""
    report = SessionTableReport()
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    years = {day.year for day in (start, end)}
    known = set()
    for year in sorted(years):
        table = store.read_table("trading_session", f"year={year}", columns=["session_id"])
        known.update(str(value) for value in table.column("session_id").to_pylist())

    rows = []
    day = start
    while day <= end:
        session = xcal.trading_session(day)
        if session is None:
            report.holidays_skipped += 1
        elif session.session_id not in known:
            if session.is_half_day:
                report.half_days += 1
            rows.append(
                {
                    "session_id": session.session_id,
                    "exchange_calendar": session.exchange_calendar,
                    "session_date": session.session_date,
                    "rth_open_at": session.rth_open_at,
                    "rth_close_at": session.rth_close_at,
                    "eth_open_at": session.eth_open_at,
                    "eth_close_at": session.eth_close_at,
                    "is_half_day": session.is_half_day,
                    "expected_m5_bars_rth": session.expected_m5_bars_rth,
                    "expected_m1_bars_rth": session.expected_m1_bars_rth,
                    "calendar_version": session.calendar_version,
                    "observed_at": stamp,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
        day += timedelta(days=1)

    if not rows:
        report.status = "ALREADY_PUBLISHED"
        return report
    report.rows = store.publish("trading_session", rows, job_id=job_id).rows_published
    return report


def build_derived_bars(
    store: ResearchStore | None,
    session_dates,
    *,
    timeframes=SLICE_TIMEFRAMES,
    symbols=None,
    as_of: datetime | None = None,
    now: datetime | None = None,
    run_id: str = "",
    job_id: str = "bar_derived",
) -> AggregateReport:
    """Derive M15/M30/H1 from canonical M5 for the given sessions."""
    report = AggregateReport()
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    cutoff = as_of or stamp
    wanted = {str(symbol).strip().upper() for symbol in (symbols or [])}

    rows: list[dict] = []
    for day in session_dates or []:
        session = xcal.trading_session(day)
        if session is None:
            continue
        report.sessions += 1
        partition = f"month={session.rth_open_at:%Y-%m}"
        # Both narrowings run in ARROW, not in Python. This used to be
        # `read_table(partition).to_pylist()` followed by exactly the two
        # filters below, which meant a whole month of M5 bars became Python
        # dicts so that one session of them could be used: 8.7M rows / 15.4 GB
        # on 2026-08-27, against a largest session of 588,778 rows. The
        # predicates are unchanged - same half-open session window, same exact
        # symbol match - so the derived rows are identical; only the peak moves.
        m5 = store.read_rows(
            "bar_m5",
            partition,
            symbols=sorted(wanted) if wanted else None,
            interval_start_range=(session.rth_open_at, session.rth_close_at),
        )
        by_symbol: dict[str, list[dict]] = {}
        for row in m5:
            symbol = str(row.get("symbol") or "")
            if row.get("interval_start") is None:
                continue
            by_symbol.setdefault(symbol, []).append(row)

        for timeframe in timeframes:
            existing = _existing_keys(store, timeframe, session)
            for symbol, symbol_rows in sorted(by_symbol.items()):
                for derived in derive_session_bars(
                    symbol_rows, session, timeframe, as_of=cutoff, computed_at=stamp, run_id=run_id
                ):
                    if (derived["symbol"], derived["interval_start"]) in existing:
                        continue
                    if derived["is_stub"]:
                        report.stubs += 1
                    if not derived["is_complete"]:
                        report.partial += 1
                    report.by_timeframe[timeframe] = report.by_timeframe.get(timeframe, 0) + 1
                    rows.append(derived)

    if not rows:
        report.status = "NOTHING_TO_DERIVE"
        return report
    published = store.publish("bar_derived", rows, job_id=job_id)
    report.rows_published = published.rows_published
    report.rows_quarantined = published.rows_quarantined
    return report


def _existing_keys(store: ResearchStore, timeframe: str, session: xcal.TradingSession) -> set:
    partition = f"timeframe={str(timeframe).upper()}/month={session.rth_open_at:%Y-%m}"
    table = store.read_table("bar_derived", partition, columns=["symbol", "interval_start"])
    keys = set()
    for symbol, start in zip(table.column("symbol").to_pylist(), table.column("interval_start").to_pylist()):
        if start is None:
            continue
        keys.add((str(symbol), start if start.tzinfo else start.replace(tzinfo=timezone.utc)))
    return keys


def build_weekly_bars(
    store: ResearchStore | None,
    weeks,
    *,
    symbols=None,
    as_of: datetime | None = None,
    now: datetime | None = None,
    run_id: str = "",
    job_id: str = "bar_derived_w1",
) -> AggregateReport:
    """Derive W1 from canonical D1 (LD-24). Provider W1 is a validation variant.

    ``weeks`` is any iterable of dates; each is resolved to its Monday-Sunday
    exchange week. A week is published only once its final session has closed -
    a forming weekly bar is never evidence - and a short week (holiday) is
    flagged through ``is_stub`` plus its real session counts.
    """
    report = AggregateReport(dataset="bar_derived")
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    cutoff = as_of or stamp
    wanted = {str(symbol).strip().upper() for symbol in (symbols or [])}

    rows: list[dict] = []
    seen_weeks = set()
    for day in weeks or []:
        monday, sunday = xcal.exchange_week(day)
        if monday in seen_weeks:
            continue
        seen_weeks.add(monday)
        sessions = xcal.week_sessions(monday)
        if not sessions:
            continue
        if sessions[-1].rth_close_at > cutoff:
            report.skipped_forming += 1
            continue  # the week has not finished trading yet
        report.sessions += 1

        d1 = store.read_table("bar_d1", f"year={monday.year}").to_pylist()
        if sunday.year != monday.year:  # a week straddling New Year
            d1 += store.read_table("bar_d1", f"year={sunday.year}").to_pylist()
        session_dates = {session.session_date for session in sessions}
        by_symbol: dict[str, list[dict]] = {}
        for row in d1:
            symbol = str(row.get("symbol") or "")
            if wanted and symbol not in wanted:
                continue
            day_value = row.get("session_date")
            if isinstance(day_value, datetime):
                day_value = day_value.date()
            if day_value in session_dates:
                by_symbol.setdefault(symbol, []).append((day_value, row))

        existing = _existing_keys(store, "W1", sessions[0])
        for symbol, entries in sorted(by_symbol.items()):
            entries.sort(key=lambda item: item[0])
            members = [row for _day, row in entries]
            if not members:
                continue
            interval_start = sessions[0].rth_open_at
            interval_end = sessions[-1].rth_close_at
            if (symbol, interval_start) in existing:
                continue
            highs = [row.get("high") for row in members if row.get("high") is not None]
            lows = [row.get("low") for row in members if row.get("low") is not None]
            short_week = len(sessions) < 5
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": "W1",
                    "aggregation_contract_id": W1_CONTRACT_ID,
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "session_id": sessions[-1].session_id,  # completes at the week's final close
                    "open": members[0].get("open"),
                    "high": max(highs) if highs else None,
                    "low": min(lows) if lows else None,
                    "close": members[-1].get("close"),
                    "volume": int(sum(int(row.get("volume") or 0) for row in members)),
                    # A holiday-shortened week is flagged, never quietly
                    # compared with a full week as equivalent.
                    "is_stub": short_week,
                    "stub_duration_min": None,
                    "constituent_count": len(members),
                    "constituent_expected": len(sessions),
                    "is_complete": len(members) == len(sessions),
                    "quality": QUALITY_COMPLETE if len(members) == len(sessions) else QUALITY_PARTIAL,
                    "event_at": interval_end,
                    "computed_at": stamp,
                    "input_capture_mode_worst": _worst_capture_mode(
                        row.get("capture_mode") for row in members
                    ),
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
            if short_week:
                report.stubs += 1
            report.by_timeframe["W1"] = report.by_timeframe.get("W1", 0) + 1

    if not rows:
        report.status = "NOTHING_TO_DERIVE"
        return report
    published = store.publish("bar_derived", rows, job_id=job_id)
    report.rows_published = published.rows_published
    report.rows_quarantined = published.rows_quarantined
    return report


def native_h1_boundaries(session: xcal.TradingSession) -> list[tuple[datetime, datetime]]:
    """IB's native ``useRTH=1`` hourly boundaries for one session.

    Stated independently of the derivation so the sentinel parity check
    compares two things rather than one thing with itself.
    """
    bounds = []
    start = session.rth_open_at
    while start < session.rth_close_at:
        end = min(start + timedelta(hours=1), session.rth_close_at)
        bounds.append((start, end))
        start = end
    return bounds


__all__ = [
    "AggregateReport",
    "SLICE_TIMEFRAMES",
    "SessionTableReport",
    "TIMEFRAME_MINUTES",
    "W1_CONTRACT_ID",
    "aggregation_contract_id",
    "build_derived_bars",
    "build_trading_sessions",
    "build_weekly_bars",
    "derive_session_bars",
    "native_h1_boundaries",
    "session_buckets",
]
