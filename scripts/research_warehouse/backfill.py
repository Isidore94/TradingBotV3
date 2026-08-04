"""Nightly/weekly backfill jobs and the one-time yfinance seed (plan Phase 3b).

Everything here is *net-new* provider traffic, so everything here goes through
the shared pacer (:mod:`pacer`) and none of it may touch a champion path. The
jobs are deliberately provider-agnostic: a job receives a ``fetcher`` callable
and drives it. That keeps the scheduling, chunking, resume, and gap-recording
logic - the parts that can actually go wrong quietly - fully testable offline,
and leaves the socket work in one thin adapter (:mod:`ib_capture`).

What each job is for (sec 5.2, LD-02/LD-03):

* ``run_nightly_backfill`` - ETH-inclusive (``useRTH=0``) M5/M1 for the active
  cohort, filling what the RTH-scoped tee could not see. Premarket extremes are
  first-class trader levels and ETH history not captured forward is lost
  permanently, so raw capture is ETH-inclusive from the first backfill onward.
* ``run_weekly_universe_sweep`` - the Saturday full-universe M5 "1 W" pass.
* ``run_yahoo_seed`` - the one-time 60-day full-universe M5 seed, trickled over
  several nights with a per-symbol completion ledger, chunked with backoff.
  Never one bulk scrape: an unofficial-ban risk that leaves a biased partial
  archive is exactly risk R11.

Three properties hold for all of them and are pinned by tests:

* **Idempotent and resumable.** A symbol/session already in the lake is not
  re-requested, so a job interrupted by the ~23:45 ET TWS restart resumes with
  no duplicate and no hole.
* **Reconnect-tolerant.** A dropped connection is a pause, not a failure: the
  job re-checks the connection through the injected ``is_connected`` callable
  and continues where it stopped.
* **Absence is recorded.** Anything not collected leaves a ``collection_gap``
  row with an honest reason, never a silent hole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:  # package import
    from . import exchange_calendar as xcal
    from . import pacer as pacer_mod
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import exchange_calendar as xcal  # type: ignore
    import pacer as pacer_mod  # type: ignore
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION  # type: ignore
    from store import ResearchStore  # type: ignore

CAPTURE_MODE_BACKFILL = "BACKFILL"
PROVIDER_IBKR = "IBKR"
PROVIDER_YAHOO = "YAHOO"

QUALITY_COMPLETE = "COMPLETE"
#: Reserved for work the capture policy never intended to collect. A pacing
#: shortfall is *intended-but-not-collected* and must never borrow this reason
#: (sec 5.4: policy absence is distinct from MISSING/NO_RESPONSE/TIMED_OUT).
NOT_COLLECTED_BY_POLICY = "NOT_COLLECTED_BY_POLICY"
#: Intended work the pacer or the run's own budget could not get to in time.
REASON_TIMED_OUT = "TIMED_OUT"
REASON_NO_RESPONSE = "NO_RESPONSE"
#: Written into ``collection_gap.resolution`` when a later run fills the gap.
RESOLUTION_BACKFILLED = "BACKFILLED"

#: Longest a single request may block waiting for a capture slot. ``None``
#: resolves to the pacer's own window: an exhausted token bucket refills after
#: at most one window, so any shorter cap turns a normal refill wait into a
#: spurious denial and a false gap row.
DEFAULT_MAX_WAIT_SECONDS = None

# Job outcomes per symbol, recorded rather than inferred.
SYMBOL_OK = "OK"
SYMBOL_ALREADY_HAVE = "ALREADY_HAVE"
SYMBOL_NO_RESPONSE = "NO_RESPONSE"
SYMBOL_PACED_OUT = "PACED_OUT"
SYMBOL_DISCONNECTED = "DISCONNECTED"
SYMBOL_ERROR = "ERROR"

SEED_LEDGER_NAME = "yahoo_m5_seed_ledger.jsonl"
#: yfinance serves ~60 days of 5-minute history; the seed's whole purpose.
YAHOO_M5_WINDOW_DAYS = 60


@dataclass
class BackfillReport:
    job: str = ""
    status: str = "OK"  # OK | DISABLED | NO_COHORT | STOPPED
    requested: int = 0
    rows_published: int = 0
    rows_quarantined: int = 0
    rows_duplicate: int = 0
    gaps_recorded: int = 0
    gaps_resolved: int = 0
    seconds_waited: float = 0.0
    by_outcome: dict = field(default_factory=dict)
    stopped_reason: str = ""

    def note(self, outcome: str) -> None:
        self.by_outcome[outcome] = self.by_outcome.get(outcome, 0) + 1


@dataclass
class FetchResult:
    """What a provider adapter returns for one (symbol, window) request."""

    bars: list = field(default_factory=list)
    error_code: int = 0
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code and bool(self.bars)


def _session_dates(start: date, end: date):
    """Weekday sessions in [start, end]; holidays are handled by the provider.

    A holiday simply returns no bars, which becomes an honest gap row rather
    than a guess about which days the exchange was open.
    """
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def _bar_rows(
    symbol: str,
    bars,
    *,
    timeframe: str,
    provider: str,
    observed_at: datetime,
    run_id: str,
    session_id_for,
    phase_for,
    interval: timedelta,
):
    rows = []
    for bar in bars or []:
        start = bar.get("interval_start") if isinstance(bar, dict) else getattr(bar, "interval_start", None)
        if start is None:
            continue
        if start.tzinfo is None:
            continue  # a naive timestamp is uncertainty, never a guess
        start = start.astimezone(timezone.utc)
        get = (lambda name: bar.get(name)) if isinstance(bar, dict) else (lambda name: getattr(bar, name, None))
        end = get("interval_end") or (start + interval)
        rows.append(
            {
                "symbol": symbol,
                "interval_start": start,
                "interval_end": end,
                "session_id": session_id_for(start),
                "session_phase": phase_for(start),
                "open": _float(get("open")),
                "high": _float(get("high")),
                "low": _float(get("low")),
                "close": _float(get("close")),
                "volume": int(_float(get("volume")) or 0),
                "vwap": _float(get("vwap")),
                "trade_count": _int(get("trade_count")),
                "provider": provider,
                "is_complete": True,
                "quality": QUALITY_COMPLETE,
                "source_hash": "",
                "event_at": end,
                "observed_at": observed_at,
                # Backfilled rows are never LIVE: the AS_OBSERVED filter must
                # exclude them from coverage, latency, and promotion evidence.
                "capture_mode": CAPTURE_MODE_BACKFILL,
                "revision_id": "",
                "supersedes_revision_id": "",
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    return rows


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _session_helpers():
    try:
        from .bar_archive import session_context
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        from bar_archive import session_context  # type: ignore

    cache: dict[date, object] = {}

    def context_for(stamp: datetime):
        key = stamp.date()
        if key not in cache:
            cache[key] = session_context(stamp)
        return cache[key]

    return (
        lambda stamp: context_for(stamp).session_id,
        lambda stamp: context_for(stamp).phase_of(stamp),
    )


def already_captured(store: ResearchStore, dataset: str, symbol: str, day: date) -> bool:
    """Has this (symbol, session day) already been archived by a *backfill*?

    Deliberately not "has any bar": the RTH-scoped tee archives the whole
    watchlist cohort every session as ``capture_mode=LIVE``/``DELAYED``, and
    treating those rows as "already have" would make the ETH-inclusive nightly
    job skip exactly the symbols it exists to extend (LD-03, review defect D6).
    Only a prior ``BACKFILL`` row for the day means this job already ran.
    """
    known, backfilled = archive_state(store, dataset, [symbol], [day])
    del known
    return (symbol, day) in backfilled


def _session_date_of(session_id, stamp: datetime) -> date:
    """The exchange session a bar belongs to - never its UTC calendar date.

    ETH runs to 20:00 ET, which is 01:00 UTC the *next* day under EST, so the
    final hour of a winter session is stored under tomorrow's UTC date. Bucketing
    by ``interval_start.date()`` therefore both failed to mark the session that
    was backfilled and marked the *following* session as covered, which skipped
    its request outright and lost its ETH bars permanently (review defect D22).
    ``bar_m5.session_id`` is a frozen sec-7.1 column that already carries the
    right answer; it is only re-derived when a row somehow lacks one.
    """
    text = str(session_id or "")
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[-10:])
        except ValueError:
            pass
    try:
        from .bar_archive import session_context
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        from bar_archive import session_context  # type: ignore
    return session_context(stamp).session_date


def _bar_partitions(days) -> set[str]:
    """Month partitions that can hold bars for these sessions.

    A session's own ETH tail can land in the next month (31 January's 19:00 ET
    bars are 1 February in UTC), so the following day's partition is read too -
    without it the per-bar dedupe cannot see those rows and republishes them.
    """
    partitions: set[str] = set()
    for day in days or []:
        partitions.add(f"month={day:%Y-%m}")
        partitions.add(f"month={day + timedelta(days=1):%Y-%m}")
    return partitions


def archive_state(store: ResearchStore, dataset: str, symbols, days):
    """One pass over the relevant month partitions.

    Returns ``(known_bar_keys, backfilled_sessions)``: the per-bar
    ``(symbol, interval_start)`` set that makes ETH rows publishable alongside
    the tee's RTH rows without duplicating them (the ``bar_archive`` pattern),
    and the ``(symbol, session_date)`` set that says a backfill already covered
    the session.
    """
    wanted = {str(symbol).strip().upper() for symbol in (symbols or [])}
    known: set[tuple[str, datetime]] = set()
    backfilled: set[tuple[str, date]] = set()
    for partition in sorted(_bar_partitions(days)):
        table = store.read_table(
            dataset, partition, columns=["symbol", "interval_start", "capture_mode", "session_id"]
        )
        for name, start, mode, session_id in zip(
            table.column("symbol").to_pylist(),
            table.column("interval_start").to_pylist(),
            table.column("capture_mode").to_pylist(),
            table.column("session_id").to_pylist(),
        ):
            symbol = str(name)
            if start is None or (wanted and symbol not in wanted):
                continue
            stamp = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
            known.add((symbol, stamp))
            if str(mode or "") == CAPTURE_MODE_BACKFILL:
                backfilled.add((symbol, _session_date_of(session_id, stamp)))
    return known, backfilled


def run_backfill(
    store: ResearchStore | None,
    cohort,
    *,
    fetcher,
    job: str,
    days,
    dataset: str = "bar_m5",
    timeframe: str = "M5",
    interval: timedelta = timedelta(minutes=5),
    provider: str = PROVIDER_IBKR,
    use_rth: bool = False,
    pacer=None,
    is_connected=None,
    now: datetime | None = None,
    clock=None,
    sleep=None,
    time_budget_seconds: float = 0.0,
    max_wait_seconds: float | None = DEFAULT_MAX_WAIT_SECONDS,
    run_id: str = "",
    job_id: str = "",
    max_requests: int | None = None,
) -> BackfillReport:
    """Drive one backfill pass over (symbol, session) pairs.

    ``fetcher(symbol, day, *, timeframe, use_rth)`` returns a
    :class:`FetchResult`. Every call is gated by the pacer, so capture can
    never crowd a champion; a pacing error backs capture off and the remaining
    work is left for the next run rather than hammered.

    Time is read from ``clock`` (default: the real UTC clock) on **every** pacer
    interaction. Freezing one stamp for a whole run would freeze the token
    bucket's 10-minute window with it, capping any single invocation at roughly
    one window's allowance (review defect D7). ``time_budget_seconds`` is how
    long the run as a whole may *block* waiting for slots - 0 keeps the old
    non-blocking behaviour, and a nightly job passes a real budget so it can
    work through its cohort while still yielding instantly to champions and to
    error 162/366.
    """
    report = BackfillReport(job=job)
    if store is None:
        report.status = "DISABLED"
        return report
    symbols = [str(symbol).strip().upper() for symbol in (cohort or []) if str(symbol).strip()]
    if not symbols:
        report.status = "NO_COHORT"
        return report

    arbiter = pacer or pacer_mod.get_pacer()
    tick = clock or utc_now
    stamp = now or tick()  # the run stamp: gap detected_at, seed provenance
    session_id_for, phase_for = _session_helpers()
    session_days = list(days or [])
    requests_left = max_requests if max_requests is not None else len(symbols) * max(1, len(session_days))
    budget_left = max(0.0, float(time_budget_seconds or 0.0))
    wait_cap = (
        float(max_wait_seconds)
        if max_wait_seconds is not None
        else getattr(arbiter, "window", timedelta(seconds=600)).total_seconds()
    )
    missed: list[tuple[str, date, str]] = []
    captured: list[tuple[str, date]] = []

    known, backfilled = archive_state(store, dataset, symbols, session_days)

    for symbol in symbols:
        for day in session_days:
            if requests_left <= 0:
                # Our own cap stopped intended work: that is a timeout, not
                # policy absence (sec 5.4).
                missed.append((symbol, day, REASON_TIMED_OUT))
                continue
            if is_connected is not None and not is_connected():
                # A TWS restart is a pause, not a failure: stop cleanly, record
                # what was not collected, and resume on the next run.
                report.note(SYMBOL_DISCONNECTED)
                missed.append((symbol, day, REASON_NO_RESPONSE))
                report.status = "STOPPED"
                report.stopped_reason = "provider disconnected"
                continue
            if (symbol, day) in backfilled:
                report.note(SYMBOL_ALREADY_HAVE)
                continue

            key = f"{symbol}|{timeframe}|{day.isoformat()}|{int(use_rth)}"
            wait = min(wait_cap, budget_left) if budget_left > 0 else 0.0
            started = tick()
            decision = arbiter.acquire(key=key, timeout=wait, sleep=sleep, now=tick)
            waited = max(0.0, (tick() - started).total_seconds())
            report.seconds_waited += waited
            budget_left = max(0.0, budget_left - waited)
            if not decision.granted:
                report.note(SYMBOL_PACED_OUT)
                missed.append((symbol, day, REASON_TIMED_OUT))
                continue
            requests_left -= 1
            report.requested += 1
            observed_at = tick()
            try:
                result = fetcher(symbol, day, timeframe=timeframe, use_rth=use_rth)
            except Exception as exc:  # a provider adapter blowing up is data, not a crash
                arbiter.note_error(0, str(exc), capture=True, now=tick())
                report.note(SYMBOL_ERROR)
                missed.append((symbol, day, REASON_NO_RESPONSE))
                continue
            if result is None:
                result = FetchResult()
            if result.error_code or result.error_message:
                # Tagged capture=True: this error is handled by the pacer and
                # never reaches the champion's Yahoo-only circuit breaker (R1).
                arbiter.note_error(result.error_code, result.error_message, capture=True, now=tick())
                report.note(SYMBOL_PACED_OUT if pacer_mod.is_pacing_error(result.error_code, result.error_message) else SYMBOL_ERROR)
                missed.append((symbol, day, REASON_NO_RESPONSE))
                continue
            if not result.bars:
                report.note(SYMBOL_NO_RESPONSE)
                missed.append((symbol, day, REASON_NO_RESPONSE))
                continue

            arbiter.note_capture_success(now=tick())
            rows = _bar_rows(
                symbol,
                result.bars,
                timeframe=timeframe,
                provider=provider,
                observed_at=observed_at,
                run_id=run_id or job,
                session_id_for=session_id_for,
                phase_for=phase_for,
                interval=interval,
            )
            if not rows:
                report.note(SYMBOL_NO_RESPONSE)
                missed.append((symbol, day, REASON_NO_RESPONSE))
                continue

            # Per-bar dedupe: the tee's RTH rows must neither block this
            # request nor be duplicated by its ETH-inclusive answer (D6).
            fresh = []
            for row in rows:
                bar_key = (row["symbol"], row["interval_start"])
                if bar_key in known:
                    report.rows_duplicate += 1
                    continue
                known.add(bar_key)
                fresh.append(row)
            backfilled.add((symbol, day))
            captured.append((symbol, day))
            report.note(SYMBOL_OK)
            if not fresh:
                continue
            published = store.publish(dataset, fresh, job_id=job_id or job)
            report.rows_published += published.rows_published
            report.rows_quarantined += published.rows_quarantined

    report.gaps_recorded = _record_missed(
        store,
        missed,
        timeframe=timeframe,
        detected_at=stamp,
        run_id=run_id or job,
        use_rth=use_rth,
        interval=interval,
    )
    report.gaps_resolved = resolve_gaps(
        store,
        captured,
        timeframe=timeframe,
        resolved_at=stamp,
        run_id=run_id or job,
        use_rth=use_rth,
        interval=interval,
    )
    return report


def open_gap_keys(store: ResearchStore, partitions) -> dict:
    """Unresolved ``collection_gap`` rows, keyed by (symbol, timeframe, gap_start).

    The lake is append-only, so a gap is "closed" by a superseding row carrying
    ``resolved_at``; the current state of a gap is its latest ``detected_at``
    row, exactly as ``latest_outcomes`` does for the outcome path (BD-53).
    """
    latest: dict[tuple[str, str, datetime], dict] = {}
    for partition in sorted(set(partitions)):
        for row in store.read_table("collection_gap", partition).to_pylist():
            start = row.get("gap_start")
            if start is not None and start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            key = (str(row.get("symbol")), str(row.get("timeframe")), start)
            current = latest.get(key)
            if current is None or _stamp_of(row, "detected_at") >= _stamp_of(current, "detected_at"):
                latest[key] = row
    return {key: row for key, row in latest.items() if row.get("resolved_at") is None}


def _stamp_of(row, column) -> datetime:
    value = row.get(column)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _gap_window(day: date, *, use_rth: bool = False, interval: timedelta = timedelta(minutes=5)):
    """The collection interval for one (session, scope), and its bar count.

    ``gap_start``/``gap_end`` are the session's own boundaries for the scope the
    job actually requested - ETH for ``useRTH=0`` capture, RTH otherwise - so
    ``expected_bars`` is the count expected *across that interval*, which is what
    the column means (BD-62/D18). A non-session day has no interval to name and
    expects no bars, so it keeps the calendar day and a count of zero: a holiday
    is an honest absence, not a shortfall.
    """
    session = xcal.trading_session(day)
    if session is None:
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        return start, start + timedelta(days=1), 0
    start, end = session.window(extended=not use_rth)
    minutes = int(interval.total_seconds() // 60)
    return start, end, session.expected_bars(minutes, extended=not use_rth)


def _record_missed(
    store: ResearchStore,
    missed,
    *,
    timeframe: str,
    detected_at: datetime,
    run_id: str,
    use_rth: bool = False,
    interval: timedelta = timedelta(minutes=5),
) -> int:
    """Append a gap row per miss, skipping ones already open.

    Without the dedupe every re-run inflated ``collection_gap`` with a second
    copy of a gap that was already recorded and still open (review defect D7).
    """
    if not missed:
        return 0
    partitions = {f"month={day:%Y-%m}" for _symbol, day, _reason in missed}
    already = set(open_gap_keys(store, partitions))
    rows = []
    for symbol, day, reason in missed:
        start, end, expected = _gap_window(day, use_rth=use_rth, interval=interval)
        key = (symbol, timeframe, start)
        if key in already:
            continue
        already.add(key)
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "gap_start": start,
                "gap_end": end,
                "expected_bars": int(expected),
                "reason": reason,
                "detected_at": detected_at,
                "resolved_at": None,
                "resolution": None,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    if not rows:
        return 0
    return store.publish("collection_gap", rows, job_id=run_id).rows_published


def resolve_gaps(
    store: ResearchStore,
    captured,
    *,
    timeframe: str,
    resolved_at: datetime,
    run_id: str,
    resolution: str = RESOLUTION_BACKFILLED,
    use_rth: bool = False,
    interval: timedelta = timedelta(minutes=5),
) -> int:
    """Close the open gaps that this run actually filled.

    An immutable lake cannot edit the original row, so the closure is a
    superseding row at the same grain carrying ``resolved_at``/``resolution``;
    readers take the latest ``detected_at`` per key. Nothing set these columns
    before, so a gap once recorded stayed open forever (review defect D7).
    """
    if not captured:
        return 0
    partitions = {f"month={day:%Y-%m}" for _symbol, day in captured}
    open_gaps = open_gap_keys(store, partitions)
    rows = []
    closed = set()
    for symbol, day in captured:
        start, _end, _expected = _gap_window(day, use_rth=use_rth, interval=interval)
        key = (symbol, timeframe, start)
        row = open_gaps.get(key)
        if row is None or key in closed:
            continue
        closed.add(key)
        superseding = dict(row)
        superseding.update(
            {
                "detected_at": resolved_at,
                "resolved_at": resolved_at,
                "resolution": resolution,
                "run_id": run_id,
            }
        )
        rows.append(superseding)
    if not rows:
        return 0
    return store.publish("collection_gap", rows, job_id=run_id).rows_published


def run_nightly_backfill(
    store: ResearchStore | None,
    cohort,
    *,
    fetcher,
    session_date: date | None = None,
    lookback_days: int = 1,
    **kwargs,
) -> BackfillReport:
    """ETH-inclusive M5 backfill for the active cohort (LD-03)."""
    end = session_date or (utc_now().date() - timedelta(days=1))
    start = end - timedelta(days=max(0, lookback_days - 1))
    return run_backfill(
        store,
        cohort,
        fetcher=fetcher,
        job="nightly_backfill",
        days=list(_session_dates(start, end)),
        use_rth=False,  # ETH-inclusive: premarket extremes are first-class
        **kwargs,
    )


def run_weekly_universe_sweep(
    store: ResearchStore | None,
    universe,
    *,
    fetcher,
    week_ending: date | None = None,
    **kwargs,
) -> BackfillReport:
    """The Saturday full-universe M5 pass over the week's sessions."""
    end = week_ending or utc_now().date()
    start = end - timedelta(days=6)
    return run_backfill(
        store,
        universe,
        fetcher=fetcher,
        job="weekly_universe_sweep",
        days=list(_session_dates(start, end)),
        use_rth=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The one-time yfinance 60-day M5 seed (R11: trickled, resumable, never bulk)
# ---------------------------------------------------------------------------
@dataclass
class SeedReport:
    status: str = "OK"  # OK | DISABLED | COMPLETE | NO_COHORT
    symbols_attempted: int = 0
    symbols_completed: int = 0
    symbols_failed: int = 0
    rows_published: int = 0
    remaining: int = 0


def seed_ledger_path(spool_dir: Path) -> Path:
    return Path(spool_dir) / SEED_LEDGER_NAME


def load_seed_ledger(spool_dir: Path) -> dict:
    """Per-symbol completion state, so a trickled seed resumes exactly."""
    path = seed_ledger_path(spool_dir)
    state: dict[str, dict] = {}
    if not path.exists():
        return state
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        symbol = str(record.get("symbol") or "").upper()
        if symbol:
            state[symbol] = record
    return state


def _append_seed_ledger(spool_dir: Path, record: dict) -> None:
    path = seed_ledger_path(spool_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")


def run_yahoo_seed(
    store: ResearchStore | None,
    universe,
    *,
    fetcher,
    spool_dir: Path,
    batch_size: int = 50,
    window_days: int = YAHOO_M5_WINDOW_DAYS,
    now: datetime | None = None,
    run_id: str = "yahoo_m5_seed",
    job_id: str = "yahoo_m5_seed",
    backoff=None,
) -> SeedReport:
    """One trickled batch of the 60-day M5 seed. Call it nightly until done.

    The seed is provider=YAHOO, capture_mode=BACKFILL, and costs zero IB
    budget - but yfinance has an unofficial ban risk, so it goes out in small
    batches with backoff and a per-symbol ledger. A symbol that fails is
    retried on a later night; the ledger is what makes a partial archive
    resumable instead of quietly biased (R11).
    """
    report = SeedReport()
    if store is None:
        report.status = "DISABLED"
        return report
    symbols = [str(symbol).strip().upper() for symbol in (universe or []) if str(symbol).strip()]
    if not symbols:
        report.status = "NO_COHORT"
        return report

    stamp = now or utc_now()
    ledger = load_seed_ledger(spool_dir)
    pending = [symbol for symbol in symbols if ledger.get(symbol, {}).get("status") != "COMPLETE"]
    report.remaining = len(pending)
    if not pending:
        report.status = "COMPLETE"
        return report

    session_id_for, phase_for = _session_helpers()
    end = stamp.date()
    start = end - timedelta(days=window_days)
    for symbol in pending[: max(1, int(batch_size))]:
        report.symbols_attempted += 1
        try:
            result = fetcher(symbol, start, end)
        except Exception as exc:
            result = FetchResult(error_message=str(exc))
        if result is None:
            result = FetchResult()
        if not result.ok:
            report.symbols_failed += 1
            _append_seed_ledger(
                spool_dir,
                {
                    "symbol": symbol,
                    "status": "FAILED",
                    "error": result.error_message,
                    "at": stamp.isoformat(),
                },
            )
            if backoff is not None:
                backoff(symbol, result.error_message)
            continue
        rows = _bar_rows(
            symbol,
            result.bars,
            timeframe="M5",
            provider=PROVIDER_YAHOO,
            observed_at=stamp,
            run_id=run_id,
            session_id_for=session_id_for,
            phase_for=phase_for,
            interval=timedelta(minutes=5),
        )
        published = store.publish("bar_m5", rows, job_id=job_id) if rows else None
        if published is not None:
            report.rows_published += published.rows_published
        report.symbols_completed += 1
        _append_seed_ledger(
            spool_dir,
            {
                "symbol": symbol,
                "status": "COMPLETE",
                "rows": published.rows_published if published else 0,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "at": stamp.isoformat(),
            },
        )
    report.remaining = max(0, report.remaining - report.symbols_completed)
    return report


__all__ = [
    "BackfillReport",
    "CAPTURE_MODE_BACKFILL",
    "DEFAULT_MAX_WAIT_SECONDS",
    "FetchResult",
    "REASON_NO_RESPONSE",
    "REASON_TIMED_OUT",
    "RESOLUTION_BACKFILLED",
    "PROVIDER_IBKR",
    "PROVIDER_YAHOO",
    "SEED_LEDGER_NAME",
    "SYMBOL_ALREADY_HAVE",
    "SYMBOL_DISCONNECTED",
    "SYMBOL_ERROR",
    "SYMBOL_NO_RESPONSE",
    "SYMBOL_OK",
    "SYMBOL_PACED_OUT",
    "SeedReport",
    "YAHOO_M5_WINDOW_DAYS",
    "already_captured",
    "archive_state",
    "load_seed_ledger",
    "open_gap_keys",
    "resolve_gaps",
    "run_backfill",
    "run_nightly_backfill",
    "run_weekly_universe_sweep",
    "run_yahoo_seed",
    "seed_ledger_path",
]
