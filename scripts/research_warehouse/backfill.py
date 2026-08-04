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
    from . import pacer as pacer_mod
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import pacer as pacer_mod  # type: ignore
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION  # type: ignore
    from store import ResearchStore  # type: ignore

CAPTURE_MODE_BACKFILL = "BACKFILL"
PROVIDER_IBKR = "IBKR"
PROVIDER_YAHOO = "YAHOO"

QUALITY_COMPLETE = "COMPLETE"
NOT_COLLECTED_BY_POLICY = "NOT_COLLECTED_BY_POLICY"

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
    gaps_recorded: int = 0
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
    """Has this (symbol, session day) already been archived?"""
    table = store.read_table(dataset, f"month={day:%Y-%m}", columns=["symbol", "interval_start"])
    for name, start in zip(table.column("symbol").to_pylist(), table.column("interval_start").to_pylist()):
        if str(name) != symbol or start is None:
            continue
        stamp = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        if stamp.date() == day:
            return True
    return False


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
    run_id: str = "",
    job_id: str = "",
    max_requests: int | None = None,
) -> BackfillReport:
    """Drive one backfill pass over (symbol, session) pairs.

    ``fetcher(symbol, day, *, timeframe, use_rth)`` returns a
    :class:`FetchResult`. Every call is gated by the pacer, so capture can
    never crowd a champion; a pacing error backs capture off and the remaining
    work is left for the next run rather than hammered.
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
    stamp = now or utc_now()
    session_id_for, phase_for = _session_helpers()
    session_days = list(days or [])
    requests_left = max_requests if max_requests is not None else len(symbols) * max(1, len(session_days))
    missed: list[tuple[str, date, str]] = []

    for symbol in symbols:
        for day in session_days:
            if requests_left <= 0:
                missed.append((symbol, day, NOT_COLLECTED_BY_POLICY))
                continue
            if is_connected is not None and not is_connected():
                # A TWS restart is a pause, not a failure: stop cleanly, record
                # what was not collected, and resume on the next run.
                report.note(SYMBOL_DISCONNECTED)
                missed.append((symbol, day, "NO_RESPONSE"))
                report.status = "STOPPED"
                report.stopped_reason = "provider disconnected"
                continue
            if already_captured(store, dataset, symbol, day):
                report.note(SYMBOL_ALREADY_HAVE)
                continue

            key = f"{symbol}|{timeframe}|{day.isoformat()}|{int(use_rth)}"
            decision = arbiter.try_acquire(key=key, now=stamp)
            if not decision.granted:
                report.note(SYMBOL_PACED_OUT)
                missed.append((symbol, day, NOT_COLLECTED_BY_POLICY))
                continue
            requests_left -= 1
            report.requested += 1
            try:
                result = fetcher(symbol, day, timeframe=timeframe, use_rth=use_rth)
            except Exception as exc:  # a provider adapter blowing up is data, not a crash
                arbiter.note_error(0, str(exc), capture=True, now=stamp)
                report.note(SYMBOL_ERROR)
                missed.append((symbol, day, "NO_RESPONSE"))
                continue
            if result is None:
                result = FetchResult()
            if result.error_code or result.error_message:
                # Tagged capture=True: this error is handled by the pacer and
                # never reaches the champion's Yahoo-only circuit breaker (R1).
                arbiter.note_error(result.error_code, result.error_message, capture=True, now=stamp)
                report.note(SYMBOL_PACED_OUT if pacer_mod.is_pacing_error(result.error_code, result.error_message) else SYMBOL_ERROR)
                missed.append((symbol, day, "NO_RESPONSE"))
                continue
            if not result.bars:
                report.note(SYMBOL_NO_RESPONSE)
                missed.append((symbol, day, "NO_RESPONSE"))
                continue

            arbiter.note_capture_success(now=stamp)
            rows = _bar_rows(
                symbol,
                result.bars,
                timeframe=timeframe,
                provider=provider,
                observed_at=stamp,
                run_id=run_id or job,
                session_id_for=session_id_for,
                phase_for=phase_for,
                interval=interval,
            )
            if not rows:
                report.note(SYMBOL_NO_RESPONSE)
                missed.append((symbol, day, "NO_RESPONSE"))
                continue
            published = store.publish(dataset, rows, job_id=job_id or job)
            report.rows_published += published.rows_published
            report.rows_quarantined += published.rows_quarantined
            report.note(SYMBOL_OK)

    report.gaps_recorded = _record_missed(store, missed, timeframe=timeframe, detected_at=stamp, run_id=run_id or job)
    return report


def _record_missed(store: ResearchStore, missed, *, timeframe: str, detected_at: datetime, run_id: str) -> int:
    if not missed:
        return 0
    rows = []
    for symbol, day, reason in missed:
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "gap_start": start,
                "gap_end": start + timedelta(days=1),
                "expected_bars": 0,
                "reason": reason,
                "detected_at": detected_at,
                "resolved_at": None,
                "resolution": None,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
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
    "FetchResult",
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
    "load_seed_ledger",
    "run_backfill",
    "run_nightly_backfill",
    "run_weekly_universe_sweep",
    "run_yahoo_seed",
    "seed_ledger_path",
]
