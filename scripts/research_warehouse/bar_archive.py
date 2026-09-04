"""M5 tee archive, scan coverage, and collection gaps (plan Phase 3).

**Capture by interception.** BounceBot already fetches "5 D / 5 mins" bars for
its watchlist cohort and keeps them in memory under
``latest_bars["<SYM>|5 D|5 mins"]``. The tee reads that mapping *after* the
champion is done with it and archives what is already there. That is the whole
mechanism, and it is why the tee costs zero provider requests: nothing in this
module fetches, connects, retries, paces, or caches, and no capture code sits
inside a champion fetch path (risk R3).

Consequences that follow from that design and are pinned by tests:

* champion timing is untouched - the tee cannot delay, queue, or reorder a
  fetch it never participates in;
* the tee cannot trip the champion's Yahoo circuit breaker, because it makes no
  request that could fail;
* re-running a capture is idempotent - a bar already archived for
  (symbol, interval_start) is not appended again, so every production cycle can
  tee the same in-memory cache safely.

Only completed bars are archived: a bar whose interval has not closed at the
observation time is preview, never evidence (plan.md sec 5, decision 0007).
Absence is recorded as explicitly as presence - a symbol outside the capture
cohort produces a ``NOT_COLLECTED_BY_POLICY`` gap row, never a silent hole and
never ``MISSING``.

Phase 3b adds the shared IB pacer, the nightly/weekly backfill jobs, and the
yfinance seed to this module. They are deliberately absent here: this file has
no provider client at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:  # package import
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION  # type: ignore
    from store import ResearchStore  # type: ignore

# The key shape BounceBot uses for its M5 cohort cache.
M5_TEE_KEY_SUFFIX = "|5 D|5 mins"
M5_INTERVAL = timedelta(minutes=5)
DEFAULT_EXCHANGE_CALENDAR = "XNYS"
RTH_M5_BARS = 78  # 09:30-16:00 ET at 5 minutes

QUALITY_COMPLETE = "COMPLETE"
QUALITY_PARTIAL = "PARTIAL"
QUALITY_MISSING = "MISSING"
NOT_COLLECTED_BY_POLICY = "NOT_COLLECTED_BY_POLICY"

PHASE_PRE = "PRE"
PHASE_RTH = "RTH"
PHASE_POST = "POST"

CAPTURE_LIVE = "LIVE"
CAPTURE_DELAYED = "DELAYED"
CAPTURE_BACKFILL = "BACKFILL"


#: Resolved ONCE. ``Path(__file__).resolve()`` is a real-path syscall (~200 us
#: on the desk), and until 2026-09-03 it ran once per cached bar per tee tick -
#: see BD-96 for the measurement (91% of the desk's GIL samples).
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
_MARKET_SESSION_MODULE = None


def _ensure_scripts_on_path() -> None:
    import sys

    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)


def _market_session_module():
    """Wrapped read of the champion's own session helper - never a second one.

    Imported once and memoized: the import machinery is cheap on a hit, but the
    path check in front of it was not, and this is called per session lookup.
    """
    global _MARKET_SESSION_MODULE
    if _MARKET_SESSION_MODULE is not None:
        return _MARKET_SESSION_MODULE
    _ensure_scripts_on_path()
    try:
        import market_session
    except ImportError:  # pragma: no cover - packaged import
        from scripts import market_session  # type: ignore
    _MARKET_SESSION_MODULE = market_session
    return market_session


@dataclass(frozen=True)
class SessionContext:
    """RTH boundaries for one exchange session, in UTC."""

    session_id: str
    session_date: date
    rth_open_at: datetime
    rth_close_at: datetime
    market_timezone: str

    def phase_of(self, moment: datetime) -> str:
        if moment < self.rth_open_at:
            return PHASE_PRE
        if moment >= self.rth_close_at:
            return PHASE_POST
        return PHASE_RTH


def session_context(
    reference: datetime | date | None = None,
    *,
    exchange_calendar: str = DEFAULT_EXCHANGE_CALENDAR,
) -> SessionContext:
    market_session = _market_session_module()
    window = market_session.get_market_session_window(reference)
    return SessionContext(
        session_id=f"{exchange_calendar}-{window.market_date.isoformat()}",
        session_date=window.market_date,
        rth_open_at=window.open_local.astimezone(timezone.utc),
        rth_close_at=window.close_local.astimezone(timezone.utc),
        market_timezone=window.market_timezone_name,
    )


def market_local_timezone():
    tz, _name = _market_session_module().get_market_local_timezone()
    return tz


@dataclass
class CaptureReport:
    dataset: str = "bar_m5"
    status: str = "OK"  # OK | DISABLED | NOTHING_TO_CAPTURE
    symbols: int = 0
    rows_published: int = 0
    rows_quarantined: int = 0
    forming_skipped: int = 0
    duplicates_skipped: int = 0
    unparsable_skipped: int = 0
    #: Symbols whose newest cached bar was already behind the caller's
    #: high-water mark, so the tee never walked their list (BD-96).
    symbols_unchanged: int = 0


@dataclass
class CoverageReport:
    dataset: str = "scan_coverage"
    status: str = "OK"
    risk_set_id: str = ""
    rows: int = 0


@dataclass
class GapReport:
    dataset: str = "collection_gap"
    status: str = "OK"
    rows: int = 0
    by_reason: dict = field(default_factory=dict)
    #: Bars actually short, per reason. The stored ``expected_bars`` column is
    #: the count expected across the gap interval (D18); the shortfall is a
    #: property of this run's observation and lives here.
    missing_bars_by_reason: dict = field(default_factory=dict)


def extract_tee_bars(latest_bars) -> dict:
    """Pull the M5 cohort out of BounceBot's in-memory bar cache.

    Read-only and copy-free of intent: the champion owns that dict, and the tee
    only looks at the ``|5 D|5 mins`` keys it already populated.
    """
    cohort: dict[str, list] = {}
    for key, bars in dict(latest_bars or {}).items():
        text = str(key)
        if not text.endswith(M5_TEE_KEY_SUFFIX) or not bars:
            continue
        symbol = text[: -len(M5_TEE_KEY_SUFFIX)].strip().upper()
        if symbol:
            cohort[symbol] = list(bars)
    return cohort


def _bar_field(bar, *names):
    for name in names:
        if isinstance(bar, dict):
            if name in bar:
                return bar[name]
        elif hasattr(bar, name):
            return getattr(bar, name)
    return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bar_start(bar, market_tz):
    """The bar's interval start as an aware UTC timestamp, or None.

    IB returns naive local timestamps (``formatDate=1``); the champion reads
    them in the configured market-local zone, so the tee uses that same zone
    rather than inventing one. A timestamp that cannot be read is skipped, not
    guessed - missing data is uncertainty.
    """
    raw = _bar_field(bar, "dt", "datetime", "date", "time")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        stamp = raw
    else:
        text = str(raw).strip()
        stamp = None
        for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d"):
            try:
                stamp = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if stamp is None:
            try:
                stamp = datetime.fromisoformat(text)
            except ValueError:
                return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=market_tz)
    return stamp.astimezone(timezone.utc)


def _source_hash(symbol: str, start: datetime, values) -> str:
    material = "|".join([symbol, start.isoformat(), *(f"{value}" for value in values)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def capture_m5_tee(
    store: ResearchStore | None,
    bars_by_symbol,
    *,
    now: datetime | None = None,
    provider: str = "IBKR",
    capture_mode: str = CAPTURE_LIVE,
    session: SessionContext | None = None,
    market_tz=None,
    run_id: str = "",
    job_id: str = "m5_tee",
    spool=None,
    seen=None,
    high_water=None,
) -> CaptureReport:
    """Archive already-fetched M5 bars into ``bar_m5``. Zero provider cost.

    ``bars_by_symbol`` is either BounceBot's raw ``latest_bars`` mapping or an
    already-extracted {symbol: bars} mapping. Bars may be the champion's
    ``IbBar`` records, dicts, or anything exposing the same fields.

    Live capture passes a ``spool`` (the GUI-owned writer, sec 8.4) so the
    session never writes the lake directly; the EOD build job seals those
    segments. Pass ``seen`` - a caller-held set of (symbol, interval_start) -
    to keep de-duplication working while the lake is unreachable.

    ``high_water`` is the cheaper de-duplication the live desk uses (BD-96): a
    caller-held ``{symbol: newest interval_start already captured}``. A bar at
    or before its symbol's mark is a duplicate, and a symbol whose NEWEST bar
    is at or before the mark is skipped without walking its list at all - the
    champion's cache is a rolling window that only ever grows at the end
    (``_dedupe_bars`` sorts it), so the last bar answers for the whole list.
    The mapping is advanced in place for every bar this call spools.

    **De-duplication happens before any per-bar work.** The old order parsed,
    hashed and session-tagged every bar of every symbol and THEN dropped it as
    a duplicate; on 2026-09-03 that was 346k bars every 60 s and a full core.
    """
    report = CaptureReport()
    if store is None and spool is None:
        report.status = "DISABLED"
        return report
    observed_at = now or utc_now()
    cohort = bars_by_symbol or {}
    if any(str(key).endswith(M5_TEE_KEY_SUFFIX) for key in cohort):
        cohort = extract_tee_bars(cohort)
    cohort = {str(symbol).strip().upper(): bars for symbol, bars in cohort.items() if bars}
    if not cohort:
        report.status = "NOTHING_TO_CAPTURE"
        return report

    tz = market_tz or market_local_timezone()
    context = session or session_context(observed_at)
    # One session lookup per session DATE, not per bar. The cache holds five
    # sessions of bars, so this is five lookups instead of 346k (BD-96).
    contexts: dict[date, SessionContext] = {context.session_date: context}
    rows: list[dict] = []
    partitions_needed: set[str] = set()
    candidates: list[tuple[str, datetime, object]] = []
    watermark = high_water if isinstance(high_water, dict) else None

    # Pass 1 - identity only. Parse the timestamp, drop forming bars and
    # anything already captured, and stage the rest. No price, hash or
    # session work happens for a bar that is about to be thrown away.
    for symbol, bars in sorted(cohort.items()):
        report.symbols += 1
        mark = watermark.get(symbol) if watermark is not None else None
        if mark is not None and bars:
            newest = _bar_start(bars[-1], tz)
            if newest is not None and newest <= mark:
                report.symbols_unchanged += 1
                continue
        for bar in bars:
            start = _bar_start(bar, tz)
            if start is None:
                report.unparsable_skipped += 1
                continue
            if start + M5_INTERVAL > observed_at:
                # The interval has not closed yet: preview, never evidence.
                report.forming_skipped += 1
                continue
            if (mark is not None and start <= mark) or (seen is not None and (symbol, start) in seen):
                report.duplicates_skipped += 1
                continue
            partitions_needed.add(f"month={start:%Y-%m}")
            candidates.append((symbol, start, bar))

    known = _known_bar_keys(store, partitions_needed) if store is not None and candidates else set()

    # Pass 2 - the real work, only for bars that will actually be published.
    staged: list[tuple[str, datetime, dict]] = []
    for symbol, start, bar in candidates:
        open_ = _as_float(_bar_field(bar, "open"))
        high = _as_float(_bar_field(bar, "high"))
        low = _as_float(_bar_field(bar, "low"))
        close = _as_float(_bar_field(bar, "close"))
        if None in (open_, high, low, close):
            # Unreadable is reported as unreadable, never folded into
            # "duplicate" because a readable twin happened to come first.
            report.unparsable_skipped += 1
            continue
        if (symbol, start) in known:
            report.duplicates_skipped += 1
            continue
        known.add((symbol, start))
        end = start + M5_INTERVAL
        volume = _as_float(_bar_field(bar, "volume")) or 0.0
        bar_day = start.astimezone(tz).date()
        bar_session = contexts.get(bar_day)
        if bar_session is None:
            bar_session = contexts[bar_day] = session_context(start)
        row = {
            "symbol": symbol,
            "interval_start": start,
            "interval_end": end,
            "session_id": bar_session.session_id,
            "session_phase": bar_session.phase_of(start),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            # As provided by the provider. IB historical TRADES volume is
            # in round lots while Yahoo is in shares; that difference is a
            # sentinel-parity check against `provider`, never a rewrite of
            # captured evidence (plan sec 7.1, 9.1).
            "volume": int(volume),
            "vwap": _as_float(_bar_field(bar, "vwap", "average", "wap")),
            "trade_count": _as_int(_bar_field(bar, "trade_count", "barCount", "bar_count")),
            "provider": provider,
            "is_complete": True,
            "quality": QUALITY_COMPLETE,
            "source_hash": _source_hash(symbol, start, (open_, high, low, close, volume)),
            "event_at": end,
            "observed_at": observed_at,
            "capture_mode": capture_mode,
            "revision_id": "",
            "supersedes_revision_id": "",
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
        }
        staged.append((symbol, start, row))

    for symbol, start, row in staged:
        if seen is not None:
            seen.add((symbol, start))
        if watermark is not None:
            current = watermark.get(symbol)
            if current is None or start > current:
                watermark[symbol] = start
        rows.append(row)

    if not rows:
        report.status = "NOTHING_TO_CAPTURE"
        return report
    if spool is not None:
        # Live path: the session spools; the EOD build job seals. M5 capture is
        # PROTECTED - it is never shed, whatever the spool cap says.
        report.rows_published = spool.write("bar_m5", rows, now=observed_at)
        report.status = "SPOOLED"
        return report
    result = store.publish("bar_m5", rows, job_id=job_id)
    report.rows_published = result.rows_published
    report.rows_quarantined = result.rows_quarantined
    return report


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _known_bar_keys(store: ResearchStore, partitions) -> set:
    known: set[tuple[str, datetime]] = set()
    for partition in sorted(partitions):
        table = store.read_table("bar_m5", partition, columns=["symbol", "interval_start"])
        for symbol, start in zip(
            table.column("symbol").to_pylist(), table.column("interval_start").to_pylist()
        ):
            stamp = start if start is None or start.tzinfo else start.replace(tzinfo=timezone.utc)
            known.add((str(symbol), stamp))
    return known


# ---------------------------------------------------------------------------
# scan_coverage - the denominator down-payment (sec 13, LD-21)
# ---------------------------------------------------------------------------
def coverage_context_from_manifest(manifest: dict) -> dict:
    """Risk-set identity taken straight from the run manifest.

    ``risk_set_id`` IS the manifest's ``run_id``, so coverage rows and run
    manifests reconcile by construction instead of by a fuzzy time join.
    """
    payload = manifest or {}
    scheduled = payload.get("started_at") or ""
    stamp = None
    if scheduled:
        try:
            stamp = datetime.fromisoformat(str(scheduled).replace("Z", "+00:00"))
        except ValueError:
            stamp = None
    if stamp is not None and stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return {
        "risk_set_id": str(payload.get("run_id") or ""),
        "run_kind": str(payload.get("job_type") or ""),
        "scheduled_at": stamp,
    }


def record_scan_coverage(
    store: ResearchStore | None,
    statuses,
    *,
    risk_set_id: str,
    run_kind: str,
    scheduled_at: datetime,
    provider: str = "",
    bar_source: str = "",
    observed_at: datetime | None = None,
    run_id: str = "",
    job_id: str = "scan_coverage",
) -> CoverageReport:
    """One row per (risk_set, symbol): what was assigned, and what came back.

    ``statuses`` maps symbol -> ``scan_status`` string, or symbol -> dict with
    ``scan_status`` plus optional ``provider``, ``bar_source``, and
    ``family_status_map``. Unevaluated is never recorded as rejected or as "no
    setup" (sec 13); every rung keeps its own status value.
    """
    report = CoverageReport(risk_set_id=risk_set_id)
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = observed_at or utc_now()
    existing = store.read_table(
        "scan_coverage",
        f"month={scheduled_at:%Y-%m}",
        columns=["risk_set_id", "symbol"],
    )
    already = {
        (str(risk), str(symbol))
        for risk, symbol in zip(
            existing.column("risk_set_id").to_pylist(), existing.column("symbol").to_pylist()
        )
    }
    rows = []
    for raw_symbol, value in sorted((statuses or {}).items()):
        symbol = str(raw_symbol).strip().upper()
        if not symbol or (risk_set_id, symbol) in already:
            continue
        detail = value if isinstance(value, dict) else {"scan_status": value}
        rows.append(
            {
                "risk_set_id": risk_set_id,
                "scheduled_at": scheduled_at,
                "run_kind": run_kind,
                "symbol": symbol,
                "scan_status": str(detail.get("scan_status") or "NOT_ASSIGNED"),
                "provider": str(detail.get("provider") or provider),
                "bar_source": str(detail.get("bar_source") or bar_source),
                "family_status_map": str(detail.get("family_status_map") or "{}"),
                "observed_at": stamp,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id or risk_set_id,
            }
        )
    if not rows:
        report.status = "ALREADY_RECORDED"
        return report
    result = store.publish("scan_coverage", rows, job_id=job_id)
    report.rows = result.rows_published
    return report


def reconcile_scan_coverage(store: ResearchStore | None, manifest: dict, *, symbol_counter="symbols_processed") -> dict:
    """Compare coverage rows with the run manifest that produced them.

    The Phase-3 exit criterion. A mismatch is reported, never repaired: the
    manifest and the lake are both evidence, and silently "fixing" one of them
    would destroy the discrepancy a reader needs to see.
    """
    context = coverage_context_from_manifest(manifest)
    outcome = {
        "risk_set_id": context["risk_set_id"],
        "run_kind": context["run_kind"],
        "manifest_symbols": None,
        "coverage_rows": 0,
        "matched": False,
        "provider_lookups": 0,
        "reason": "",
    }
    counters = (manifest or {}).get("counters") or {}
    manifest_symbols = counters.get(symbol_counter)
    outcome["manifest_symbols"] = int(manifest_symbols) if isinstance(manifest_symbols, (int, float)) else None
    outcome["provider_lookups"] = int(
        sum(
            int(value)
            for key, value in counters.items()
            if str(key).startswith("provider.") and str(key).endswith(".lookup")
        )
    )
    if store is None:
        outcome["reason"] = "warehouse disabled"
        return outcome
    if not outcome["risk_set_id"]:
        outcome["reason"] = "manifest has no run_id"
        return outcome
    scheduled = context["scheduled_at"] or utc_now()
    table = store.read_table("scan_coverage", f"month={scheduled:%Y-%m}", columns=["risk_set_id", "symbol"])
    rows = [
        symbol
        for risk, symbol in zip(
            table.column("risk_set_id").to_pylist(), table.column("symbol").to_pylist()
        )
        if str(risk) == outcome["risk_set_id"]
    ]
    outcome["coverage_rows"] = len(rows)
    if outcome["manifest_symbols"] is None:
        outcome["reason"] = f"manifest has no {symbol_counter} counter"
        return outcome
    outcome["matched"] = outcome["coverage_rows"] == outcome["manifest_symbols"]
    if not outcome["matched"]:
        outcome["reason"] = (
            f"{outcome['coverage_rows']} coverage rows vs {outcome['manifest_symbols']} in the run manifest"
        )
    return outcome


# ---------------------------------------------------------------------------
# collection_gap - absence recorded as explicitly as presence (sec 5.4)
# ---------------------------------------------------------------------------
def record_collection_gaps(
    store: ResearchStore | None,
    *,
    session: SessionContext,
    timeframe: str = "M5",
    captured_counts=None,
    policy_symbols=None,
    expected_bars: int = RTH_M5_BARS,
    detected_at: datetime | None = None,
    run_id: str = "",
    job_id: str = "collection_gap",
) -> GapReport:
    """Record what the session did NOT collect, with the right reason each time.

    ``captured_counts`` maps a cohort symbol to how many completed bars were
    archived; anything short of ``expected_bars`` is a PARTIAL gap and zero is
    MISSING. ``policy_symbols`` are symbols the capture policy never intended to
    collect intraday - they get ``NOT_COLLECTED_BY_POLICY``, which is a
    different fact from missing data and must never be conflated with it.
    """
    report = GapReport()
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = detected_at or utc_now()
    existing = store.read_table(
        "collection_gap",
        f"month={session.rth_open_at:%Y-%m}",
        columns=["symbol", "timeframe", "gap_start"],
    )
    already = {
        (str(symbol), str(frame), start.replace(tzinfo=timezone.utc) if start and not start.tzinfo else start)
        for symbol, frame, start in zip(
            existing.column("symbol").to_pylist(),
            existing.column("timeframe").to_pylist(),
            existing.column("gap_start").to_pylist(),
        )
    }
    rows = []

    def _add(symbol: str, reason: str, missing: int):
        key = (symbol, timeframe, session.rth_open_at)
        if key in already:
            return
        already.add(key)
        report.by_reason[reason] = report.by_reason.get(reason, 0) + 1
        report.missing_bars_by_reason[reason] = (
            report.missing_bars_by_reason.get(reason, 0) + int(missing)
        )
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "gap_start": session.rth_open_at,
                "gap_end": session.rth_close_at,
                # The count expected across [gap_start, gap_end] - which is the
                # whole session - not this run's shortfall. Storing the
                # shortfall under a column documented as the expected count
                # made the Health coverage tile sum an ambiguous number
                # (review defect D18); the shortfall is in the report instead.
                "expected_bars": int(expected_bars),
                "reason": reason,
                "detected_at": stamp,
                "resolved_at": None,
                "resolution": None,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )

    for raw_symbol, count in sorted((captured_counts or {}).items()):
        symbol = str(raw_symbol).strip().upper()
        captured = int(count or 0)
        if captured >= expected_bars:
            continue
        _add(symbol, QUALITY_MISSING if captured == 0 else QUALITY_PARTIAL, expected_bars - captured)

    captured_symbols = {str(symbol).strip().upper() for symbol in (captured_counts or {})}
    for raw_symbol in sorted({str(symbol).strip().upper() for symbol in (policy_symbols or [])}):
        if raw_symbol and raw_symbol not in captured_symbols:
            _add(raw_symbol, NOT_COLLECTED_BY_POLICY, expected_bars)

    if not rows:
        report.status = "NO_GAPS"
        return report
    result = store.publish("collection_gap", rows, job_id=job_id)
    report.rows = result.rows_published
    return report


def captured_bar_counts(store: ResearchStore | None, session: SessionContext, symbols=None) -> dict:
    """Completed M5 bars archived for each symbol in one session."""
    if store is None:
        return {}
    table = store.read_table("bar_m5", f"month={session.rth_open_at:%Y-%m}", columns=["symbol", "interval_start"])
    wanted = {str(symbol).strip().upper() for symbol in (symbols or [])}
    counts: dict[str, int] = {symbol: 0 for symbol in wanted}
    for symbol, start in zip(table.column("symbol").to_pylist(), table.column("interval_start").to_pylist()):
        if start is None:
            continue
        stamp = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        if not (session.rth_open_at <= stamp < session.rth_close_at):
            continue
        name = str(symbol)
        if wanted and name not in wanted:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


__all__ = [
    "CAPTURE_BACKFILL",
    "CAPTURE_DELAYED",
    "CAPTURE_LIVE",
    "CaptureReport",
    "CoverageReport",
    "GapReport",
    "M5_TEE_KEY_SUFFIX",
    "NOT_COLLECTED_BY_POLICY",
    "RTH_M5_BARS",
    "SessionContext",
    "capture_m5_tee",
    "captured_bar_counts",
    "coverage_context_from_manifest",
    "extract_tee_bars",
    "market_local_timezone",
    "reconcile_scan_coverage",
    "record_collection_gaps",
    "record_scan_coverage",
    "session_context",
]
