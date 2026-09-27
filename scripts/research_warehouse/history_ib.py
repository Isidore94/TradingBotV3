"""Five-year IB M30 history, with H1 and H4 derived from it (P10, 2026-09-27).

Yahoo serves H1 for ~730 days and M30 for ~60, so older intraday history comes
from IB. Only M30 is requested (TRADES, useRTH=1, split-adjusted as IB serves
it); H1 (``h1_from_m30_rth_v1``) and H4 (``h4_rth_0930_1330_v1``) are derived
from it into ``bar_derived_history``. The reader picks one intraday basis per
symbol (``history_reader.intraday_basis``); the Yahoo series stays stored as a
cross-check.

Rules this module keeps (pinned by ``tests/test_history_ib.py``):

* never in US market hours: no request between 09:15 and 16:15 ET on a trading
  day (06:15-13:15 PT), so it never competes with the desk's IB traffic. The
  run pauses there and resumes on its own;
* client id 1011 through an ``IbPacer`` capped at 45 requests / 10 minutes,
  backing off on 162/366/420. It survives the nightly TWS restart by building
  a fresh connection;
* resumable and idempotent: one ledger line per (symbol, window) is written
  only after its rows are sealed; a sealed window is never requested again,
  and a bar already stored is never re-published;
* the same row checks and quarantine as ``history`` (plus off-grid starts):
  bars outside RTH or on a non-session go to ``_quarantine`` with a reason;
* flags, never drops: missing and partial sessions, and sessions where IB's
  derived H1 and Yahoo's native H1 closes differ by more than 0.5%;
* a split recorded after the series was first pulled starts a new revision
  (``supersedes_revision_id``) and the whole symbol is pulled again.

Network lives only in :class:`IbHistoryFetcher`; the job takes any fetcher,
so the tests run offline.
"""

from __future__ import annotations

import contextlib
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime

import pyarrow.dataset as pads

try:  # package import
    from . import exchange_calendar as xcal
    from . import history as hist
    from . import history_reader as reader
    from . import ib_capture
    from . import pacer as pacer_mod
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import exchange_calendar as xcal  # type: ignore
    import history as hist  # type: ignore
    import history_reader as reader  # type: ignore
    import ib_capture  # type: ignore
    import pacer as pacer_mod  # type: ignore
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION  # type: ignore

PROVIDER_IBKR = "IBKR"
IB_ADJUSTMENT_VERSION = "ibkr_trades_split_v1"
IB_START = date(2021, 9, 27)
M30 = timedelta(minutes=30)
#: IB API 9.81 reports US-stock historical volume in round lots of 100 shares.
VOLUME_LOT = 100
#: Capture requests per 10-minute window: under IB's 60 so the desk keeps headroom.
CAPTURE_PER_WINDOW = 45
REQUEST_TIMEOUT_SECONDS = 180.0
#: Measured 2026-09-27: a cold 1 Y M30 request takes ~14 s.
SECONDS_PER_REQUEST = 14.0
MARKET_GUARD_START = dtime(9, 15)  # 06:15 PT
MARKET_GUARD_END = dtime(16, 15)  # 13:15 PT
BATCH_SYMBOLS = 40
#: A window whose first bar is this far past its start marks the listing date.
LISTING_SLACK_DAYS = 7
MAX_ATTEMPTS = 6
RETRY_SLEEP_SECONDS = 60.0
RETRY_SLEEP_MAX_SECONDS = 900.0
NO_CONTRACT_RETRY_DAYS = 7
MISMATCH_TOLERANCE = 0.005
LEDGER_NAME = "ib_m30"
PRIORITY = ("SPY", "QQQ", "IWM", "DIA")

OFF_GRID = "OFF_GRID_INTERVAL"
FLAG_PARTIAL_SESSION = "PARTIAL_SESSION"
FLAG_IB_YAHOO_MISMATCH = "IB_YAHOO_H1_MISMATCH"

# fetch outcomes
OK = "OK"
NO_DATA = "NO_DATA"
NO_CONTRACT = "NO_CONTRACT"
RETRY = "RETRY"
ERROR = "ERROR"
# ledger statuses that close a window
DONE = "DONE"
EMPTY = "EMPTY"
EMPTY_INFERRED = "EMPTY_INFERRED"
CLOSED_STATUSES = frozenset({DONE, EMPTY, EMPTY_INFERRED})


# ---------------------------------------------------------------------------
# windows, symbols, schedule (pure)
# ---------------------------------------------------------------------------
def _add_year(day: date) -> date:
    try:
        return day.replace(year=day.year + 1)
    except ValueError:  # 29 February
        return day.replace(year=day.year + 1, day=28)


def ib_windows(start: date, through: date) -> list[tuple[date, date]]:
    """Fixed one-year windows from ``start``; the last one ends at ``through``."""
    out = []
    first = start
    while first <= through:
        following = _add_year(first)
        out.append((first, min(following - timedelta(days=1), through)))
        first = following
    return out


def duration_for(first: date, last: date) -> str:
    """The smallest IB duration that reaches back from ``last`` to ``first``."""
    days = (last - first).days + 1
    if days <= 28:
        return "1 M"
    if days <= 88:
        return "3 M"
    if days <= 178:
        return "6 M"
    return "1 Y"


def ib_symbol(symbol: str) -> str:
    """IB writes share classes with a space: BRK.B -> BRK B."""
    return str(symbol).strip().upper().replace(".", " ").replace("-", " ")


def ib_request(symbol: str, end: date, duration: str) -> dict:
    return {
        "symbol": ib_symbol(symbol),
        "endDateTime": f"{end:%Y%m%d} 16:00:00 {ib_capture.EXCHANGE_TZ_NAME}",
        "durationStr": duration,
        "barSizeSetting": ib_capture.BAR_SIZES["M30"],
        "whatToShow": "TRADES",
        "useRTH": 1,
        "formatDate": 2,
        "keepUpToDate": False,
    }


def market_guard_until(moment: datetime) -> datetime | None:
    """When ``moment`` is inside 09:15-16:15 ET of a trading day, the guard's end."""
    local = moment.astimezone(xcal.EXCHANGE_TZ)
    if xcal.trading_session(local.date()) is None:
        return None
    begin = datetime.combine(local.date(), MARKET_GUARD_START, xcal.EXCHANGE_TZ)
    end = datetime.combine(local.date(), MARKET_GUARD_END, xcal.EXCHANGE_TZ)
    if begin <= local < end:
        return end.astimezone(timezone.utc)
    return None


def order_symbols(store, symbols, *, through: date | None) -> list[str]:
    """Indexes and sector ETFs first, then 20-day median dollar volume, highest first."""
    names = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    names = [s for s in names if not s.startswith("^")]
    first = [s for s in (*PRIORITY, *hist.sector_etfs()) if s in names]
    rest = [s for s in names if s not in first]
    liquidity: dict[str, float] = {}
    if rest and through is not None:
        series = reader.read_d1(rest, start=through - timedelta(days=45), end=through, store=store)
        for symbol, frame in series.items():
            tail = frame.tail(20)
            values = [float(c) * float(v) for c, v in zip(tail["close"], tail["volume"], strict=False) if c and v]
            if values:
                liquidity[symbol] = statistics.median(values)
    rest.sort(key=lambda s: (-liquidity.get(s, -1.0), s))
    return first + rest


# ---------------------------------------------------------------------------
# rows (pure)
# ---------------------------------------------------------------------------
def ib_row_problem(row: dict) -> str | None:
    """``history.row_problem`` plus: an M30 bar must start on :00 or :30 ET."""
    start = row.get("interval_start")
    if isinstance(start, datetime):
        local = start.astimezone(xcal.EXCHANGE_TZ)
        if local.minute not in (0, 30) or local.second or local.microsecond:
            return OFF_GRID
    return hist.row_problem(row)


def m30_rows(symbol, bars, *, first: date, last: date, revision_id, supersedes, observed_at, run_id) -> list[dict]:
    """Parsed IB bars inside [first, last] (ET dates) -> ``bar_m30`` rows, one per start."""
    rows: dict[datetime, dict] = {}
    for bar in bars:
        start = bar["interval_start"].astimezone(timezone.utc)
        day = start.astimezone(xcal.EXCHANGE_TZ).date()
        if not first <= day <= last:
            continue
        session = xcal.session_for(start)
        if session is not None and session.rth_close_at + hist.SETTLE > observed_at:
            continue  # not final yet
        if session is None:
            end, phase, session_id = start + M30, "CLOSED", ""
        else:
            end = min(start + M30, session.rth_close_at)
            phase, session_id = session.phase_of(start), session.session_id
        values = [hist._num(bar.get(name)) for name in ("open", "high", "low", "close")]
        lots = hist._num(bar.get("volume"))
        volume = None if lots is None else int(round(lots * VOLUME_LOT))
        rows.setdefault(
            start,
            {
                "symbol": symbol,
                "interval_start": start,
                "interval_end": end,
                "session_id": session_id,
                "session_phase": phase,
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "volume": volume,
                "vwap": hist._num(bar.get("vwap")),
                "trade_count": bar.get("trade_count"),
                "provider": PROVIDER_IBKR,
                "is_complete": True,
                "quality": hist.QUALITY_COMPLETE,
                "source_hash": hist._source_hash(symbol, start, [*values, volume]),
                "adjustment_version": IB_ADJUSTMENT_VERSION,
                "event_at": end,
                "observed_at": observed_at,
                "capture_mode": hist.CAPTURE_BACKFILL,
                "revision_id": revision_id,
                "supersedes_revision_id": supersedes,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            },
        )
    return [rows[key] for key in sorted(rows)]


def h1_buckets(session) -> list[tuple[datetime, datetime]]:
    """09:30-10:30 ... 15:30-16:00 ET; a half day stops at its 13:00 close."""
    out = []
    begin = session.rth_open_at
    while begin < session.rth_close_at:
        out.append((begin, min(begin + timedelta(hours=1), session.rth_close_at)))
        begin += timedelta(hours=1)
    return out


def derive_h1_rows(m30: list[dict], *, provider, source_revision_id, computed_at, run_id) -> list[dict]:
    """H1 bars from one symbol's M30 bars under ``h1_from_m30_rth_v1``."""
    by_session: dict[str, list[dict]] = {}
    for row in m30:
        if row.get("session_id"):
            by_session.setdefault(row["session_id"], []).append(row)
    out = []
    for session_id, rows in sorted(by_session.items()):
        session = xcal.trading_session(date.fromisoformat(session_id.split("-", 1)[1]))
        if session is None:
            continue
        for begin, end in h1_buckets(session):
            members = sorted((r for r in rows if begin <= r["interval_start"] < end), key=lambda r: r["interval_start"])
            if not members:
                continue
            span = int((end - begin).total_seconds() // 60)
            expected = math.ceil(span / 30)
            complete = len(members) == expected
            out.append(
                {
                    "symbol": members[0]["symbol"],
                    "timeframe": "H1",
                    "aggregation_contract_id": reader.H1_FROM_M30_CONTRACT_ID,
                    "interval_start": begin,
                    "interval_end": end,
                    "session_id": session_id,
                    "open": members[0]["open"],
                    "high": max(r["high"] for r in members),
                    "low": min(r["low"] for r in members),
                    "close": members[-1]["close"],
                    "volume": int(sum(int(r["volume"] or 0) for r in members)),
                    "is_stub": span < 60,
                    "stub_duration_min": span if span < 60 else None,
                    "constituent_count": len(members),
                    "constituent_expected": expected,
                    "is_complete": complete,
                    "quality": hist.QUALITY_COMPLETE if complete else "PARTIAL",
                    "event_at": end,
                    "computed_at": computed_at,
                    "input_capture_mode_worst": hist.CAPTURE_BACKFILL,
                    "provider": provider,
                    "source_revision_id": source_revision_id,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
    return out


def session_flags(symbol, m30: list[dict], *, detected_at, run_id, reference_days: set | None = None) -> list[dict]:
    """Sessions with no M30 bar, or fewer than a full session's, between the first and last bar.

    With ``reference_days`` (SPY's IB sessions) a day SPY has no bars either is
    not flagged: that is a market closure the calendar lacks (2025-01-09), not a gap.
    """
    if not m30:
        return []
    counts: dict[date, int] = {}
    for row in m30:
        day = row["interval_start"].astimezone(xcal.EXCHANGE_TZ).date()
        counts[day] = counts.get(day, 0) + 1
    flags = []
    for session in xcal.sessions_between(min(counts), max(counts)):
        day = session.session_date
        expected = session.expected_bars(30)
        have = counts.get(day, 0)
        if have == 0 and reference_days and day not in reference_days:
            continue
        if have == 0:
            flags.append(hist.flag_row("bar_m30", symbol, hist.FLAG_MISSING_SESSION, day, "no IBKR M30 bar for an exchange session", detected_at=detected_at, run_id=run_id))
        elif have < expected:
            flags.append(hist.flag_row("bar_m30", symbol, FLAG_PARTIAL_SESSION, day, f"IBKR M30 {have} of {expected} bars", detected_at=detected_at, run_id=run_id))
    return flags


def mismatch_flags(symbol, ib_h1: list[dict], yahoo_h1: dict, *, detected_at, run_id, tolerance=MISMATCH_TOLERANCE) -> list[dict]:
    """Sessions where an IB-derived H1 close and Yahoo's native H1 close differ by more than ``tolerance``."""
    worst: dict[date, tuple] = {}
    for row in ib_h1:
        if not row.get("is_complete"):
            continue
        theirs = yahoo_h1.get(row["interval_start"])
        ours = hist._num(row.get("close"))
        if not theirs or not ours:
            continue
        gap = ours / theirs - 1.0
        day = row["interval_start"].astimezone(xcal.EXCHANGE_TZ).date()
        if abs(gap) > tolerance and (day not in worst or abs(gap) > abs(worst[day][0])):
            worst[day] = (gap, row["interval_start"], ours, theirs)
    flags = []
    for day, (gap, start, ours, theirs) in sorted(worst.items()):
        clock = start.astimezone(xcal.EXCHANGE_TZ).strftime("%H:%M")
        detail = f"H1 close differs {gap:+.2%} at {clock} ET (IBKR {ours:.4f} vs YAHOO {theirs:.4f})"
        flags.append(hist.flag_row("bar_derived_history", symbol, FLAG_IB_YAHOO_MISMATCH, day, detail, detected_at=detected_at, run_id=run_id, interval_start=start))
    return flags


# ---------------------------------------------------------------------------
# the fetcher (the only network code)
# ---------------------------------------------------------------------------
@dataclass
class IbFetch:
    status: str = OK
    bars: list = field(default_factory=list)
    code: int = 0
    message: str = ""


def classify(bars, error, pacer) -> IbFetch:
    """An IB answer -> an outcome. Only pacing errors back the pacer off."""
    if error:
        code, message = error if isinstance(error, (tuple, list)) else (0, str(error))
        code, text = int(code or 0), str(message or "")
        lower = text.lower()
        if code == 162 and ("no data" in lower or "returned no data" in lower):
            return IbFetch(status=NO_DATA, code=code, message=text)
        if code == 200:  # no security definition / ambiguous contract
            return IbFetch(status=NO_CONTRACT, code=code, message=text)
        pacer.note_error(code, text, capture=True)
        if pacer_mod.is_pacing_error(code, text) or code in (ib_capture.TIMEOUT_ERROR_CODE, 504, 1100, 1300):
            return IbFetch(status=RETRY, code=code, message=text)
        return IbFetch(status=ERROR, code=code, message=text)
    parsed = [bar for bar in (ib_capture.parse_bar(raw, interval=M30) for raw in bars or []) if bar]
    if not parsed:
        return IbFetch(status=NO_DATA, message="no bars")
    pacer.note_capture_success()
    dropped = len(bars) - len(parsed)
    return IbFetch(status=OK, bars=parsed, message=f"{dropped} unreadable bars dropped" if dropped else "")


class IbHistoryFetcher:
    """Client 1011 through the pacer; a dead connection is rebuilt, never reused."""

    def __init__(self, transport_factory=None, *, spec=None, pacer=None, timeout=REQUEST_TIMEOUT_SECONDS, sleep=time.sleep, acquire_timeout=900.0):
        self.spec = spec or ib_capture.backfill_connection_spec()
        self.spec.validate()
        self.pacer = pacer or pacer_mod.IbPacer(capture_allowance=CAPTURE_PER_WINDOW)
        self.factory = transport_factory or ib_capture.build_ib_transport
        self.timeout = float(timeout)
        self.sleep = sleep
        self.acquire_timeout = float(acquire_timeout)
        self.transport = None
        self.requests = 0

    def _connected(self) -> bool:
        current = self.transport
        if current is not None and current.is_connected():
            return True
        if current is not None:
            with contextlib.suppress(Exception):
                current.disconnect()
        self.transport = None
        try:
            self.transport = self.factory(self.spec)
        except Exception:  # noqa: BLE001 - TWS down (nightly restart): retried later
            return False
        for _ in range(50):
            if self.transport.is_connected():
                self.sleep(1.0)  # let the handshake's farm notices land
                return True
            self.sleep(0.2)
        return False

    def fetch(self, symbol: str, *, end: date, duration: str) -> IbFetch:
        request = ib_request(symbol, end, duration)
        decision = self.pacer.acquire(key=f"{request['symbol']}|{end}|{duration}", timeout=self.acquire_timeout, sleep=self.sleep)
        if not decision.granted:
            return IbFetch(status=RETRY, message=f"pacer: {decision.reason}")
        if not self._connected():
            return IbFetch(status=RETRY, message="TWS not reachable")
        self.requests += 1
        try:
            bars, error = self.transport.request_historical(timeout=self.timeout, **request)
        except Exception as exc:  # noqa: BLE001 - a dropped socket is retried
            self.pacer.note_error(0, str(exc), capture=True)
            return IbFetch(status=RETRY, message=str(exc))
        return classify(bars, error, self.pacer)

    def close(self) -> None:
        if self.transport is not None:
            with contextlib.suppress(Exception):
                self.transport.disconnect()
        self.transport = None


# ---------------------------------------------------------------------------
# lake state
# ---------------------------------------------------------------------------
@dataclass
class IbState:
    revision_id: str
    first_observed: datetime
    starts: set


def ib_states(store, symbols) -> dict[str, IbState]:
    """Each symbol's current IBKR ``bar_m30`` revision and its stored bar starts."""
    table = reader._scan(
        store,
        "bar_m30",
        symbols=sorted(symbols),
        extra_filter=pads.field("provider") == PROVIDER_IBKR,
        columns=["symbol", "interval_start", "revision_id", "supersedes_revision_id", "observed_at"],
    )
    if not table.num_rows:
        return {}
    frame = table.to_pandas()
    out = {}
    for symbol, rows in frame.groupby("symbol"):
        revision = reader._current_revision(rows)
        rows = rows[rows["revision_id"] == revision]
        out[str(symbol)] = IbState(
            revision_id=revision,
            first_observed=rows["observed_at"].min().to_pydatetime(),
            starts={stamp.to_pydatetime() for stamp in rows["interval_start"]},
        )
    return out


def _current_ib_m30(store, symbols) -> dict[str, list[dict]]:
    table = reader._scan(
        store,
        "bar_m30",
        symbols=sorted(symbols),
        extra_filter=pads.field("provider") == PROVIDER_IBKR,
        columns=["symbol", "interval_start", "session_id", "open", "high", "low", "close", "volume",
                 "revision_id", "supersedes_revision_id", "observed_at"],
    )
    if not table.num_rows:
        return {}
    frame = table.to_pandas()
    out = {}
    for symbol, rows in frame.groupby("symbol"):
        revision = reader._current_revision(rows)
        rows = rows[rows["revision_id"] == revision].drop_duplicates("interval_start").sort_values("interval_start")
        records = rows.drop(columns=["supersedes_revision_id", "observed_at"]).to_dict("records")
        for record in records:
            record["interval_start"] = record["interval_start"].to_pydatetime()
            record["symbol"] = str(symbol)
            record["volume"] = hist._volume(record["volume"])
        out[str(symbol)] = records
    return out


def _yahoo_h1_closes(store, symbols) -> dict[str, dict]:
    table = reader._scan(
        store,
        "bar_h1",
        symbols=sorted(symbols),
        extra_filter=pads.field("provider") == hist.PROVIDER_YAHOO,
        columns=["symbol", "interval_start", "close", "revision_id", "supersedes_revision_id", "observed_at"],
    )
    if not table.num_rows:
        return {}
    frame = table.to_pandas()
    out = {}
    for symbol, rows in frame.groupby("symbol"):
        revision = reader._current_revision(rows)
        rows = rows[rows["revision_id"] == revision]
        out[str(symbol)] = {stamp.to_pydatetime(): float(close) for stamp, close in zip(rows["interval_start"], rows["close"], strict=False)}
    return out


def _split_days(store, symbols) -> dict[str, set]:
    actions = reader.read_corporate_actions(sorted(symbols), store=store)
    out: dict[str, set] = {}
    for symbol, kind, day in zip(actions["symbol"], actions["action_type"], actions["ex_date"], strict=False):
        if kind == "SPLIT":
            out.setdefault(str(symbol), set()).add(day)
    return out


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------
def _window_closed(record: dict | None, revision: str, last: date) -> bool:
    if not record or record.get("status") not in CLOSED_STATUSES or record.get("revision") != revision:
        return False
    try:
        return date.fromisoformat(str(record.get("through"))) >= last
    except ValueError:
        return False


def plan(store, symbols, *, now: datetime | None = None, start: date = IB_START) -> dict:
    """Dry run: the windows still owed per the ledger, and the time they take."""
    stamp = now or utc_now()
    through = hist.last_completed_session(stamp)
    names = order_symbols(store, symbols, through=through)
    memory = hist.HistoryLedger(store.root, LEDGER_NAME).latest()
    windows = ib_windows(start, through) if through else []
    owed, per_symbol = 0, {}
    for symbol in names:
        count = 0
        for first, last in windows:
            record = memory.get(f"{symbol}|{first.isoformat()}")
            count += not _window_closed(record, (record or {}).get("revision"), last)
        per_symbol[symbol] = count
        owed += count
    return {
        "status": "OK",
        "dry_run": True,
        "symbols": len(names),
        "through": through.isoformat() if through else None,
        "windows_per_symbol": len(windows),
        "requests_owed": owed,
        "hours_estimate": round(owed * max(SECONDS_PER_REQUEST, 600.0 / CAPTURE_PER_WINDOW) / 3600.0, 1),
        "first_symbols": names[:20],
        "symbols_done": sum(1 for count in per_symbol.values() if count == 0),
    }


class _Stop(Exception):
    """The run's deadline passed."""


def run_ib_backfill(
    store,
    symbols,
    *,
    fetcher,
    now: datetime | None = None,
    start: date = IB_START,
    max_hours: float | None = None,
    clock=None,
    sleep=time.sleep,
    lock=contextlib.nullcontext,
    log=print,
    batch_symbols: int = BATCH_SYMBOLS,
    run_id: str = "",
) -> hist.JobReport:
    """Pull IB M30 windows newest first, seal them per batch, derive H1/H4, flag."""
    clock = clock or utc_now
    stamp = now or clock()
    run_id = run_id or f"history_ib_m30_{stamp:%Y%m%dT%H%M%SZ}"
    report = hist.JobReport(job="ib_m30_backfill")
    through = hist.last_completed_session(stamp)
    deadline = clock() + timedelta(hours=float(max_hours)) if max_hours else None
    ledger = hist.HistoryLedger(store.root, LEDGER_NAME)
    memory = ledger.latest()
    names = order_symbols(store, symbols, through=through)
    report.symbols = len(names)
    if through is None:
        report.notes.append("no completed session yet")
        return report
    windows = list(reversed(ib_windows(start, through)))

    def _wait_ready(flush) -> None:
        """Block through market hours; raise _Stop at the deadline."""
        while True:
            moment = clock()
            if deadline is not None and moment >= deadline:
                raise _Stop()
            lift = market_guard_until(moment)
            if lift is None:
                return
            flush()
            target = lift if deadline is None else min(lift, deadline)
            log(f"ib m30: market hours - paused until {target.astimezone(xcal.EXCHANGE_TZ):%H:%M} ET")
            sleep(max(1.0, min(300.0, (target - moment).total_seconds())))

    def _fetch(symbol, first, last, flush) -> IbFetch:
        pause = RETRY_SLEEP_SECONDS
        result = IbFetch(status=ERROR, message="not attempted")
        for _attempt in range(MAX_ATTEMPTS):
            _wait_ready(flush)
            result = fetcher.fetch(symbol, end=last, duration=duration_for(first, last))
            report.note("requests")
            if result.status != RETRY:
                return result
            log(f"ib m30 {symbol} {first}: {result.message} - retry in {pause:.0f}s")
            sleep(pause)
            pause = min(pause * 2, RETRY_SLEEP_MAX_SECONDS)
        return IbFetch(status=ERROR, code=result.code, message=f"gave up after {MAX_ATTEMPTS} tries: {result.message}")

    for batch in hist._batches(names, batch_symbols):
        states = ib_states(store, batch)
        splits = _split_days(store, batch)
        pending_rows: list[dict] = []
        pending_ledger: list[dict] = []
        no_contract: list[str] = []
        bad = hist.quarantined_keys(store)

        def _flush():
            nonlocal pending_rows, pending_ledger
            report.add("bar_m30", hist._publish(store, "bar_m30", pending_rows, lock=lock, job_id=run_id, validate=ib_row_problem))
            for record in pending_ledger:  # after the seal: a crash leaves the window owed
                ledger.append(record)
                memory[record["key"]] = record
            pending_rows, pending_ledger = [], []

        stopped = False
        try:
            for symbol in batch:
                if _recent_no_contract(memory.get(f"{symbol}|*"), stamp):
                    report.note("NO_CONTRACT_SKIPPED")
                    continue
                state = states.get(symbol)
                supersedes = ""
                if state is not None:
                    pulled = state.first_observed.astimezone(xcal.EXCHANGE_TZ).date()
                    if any(pulled < day <= through for day in splits.get(symbol, ())):
                        supersedes = state.revision_id
                        report.repulled.append(symbol)
                        report.note("REBASED")
                        state = None
                revision = state.revision_id if state else f"{PROVIDER_IBKR}:{symbol}:{run_id}"
                have = set(state.starts) if state else set()
                listed_after: date | None = None
                for first, last in windows:
                    key = f"{symbol}|{first.isoformat()}"
                    if _window_closed(memory.get(key), revision, last):
                        continue
                    base = {"key": key, "sym": symbol, "revision": revision, "through": last.isoformat(), "at": stamp.isoformat(), "run_id": run_id}
                    if listed_after is not None and last < listed_after:
                        pending_ledger.append({**base, "status": EMPTY_INFERRED, "rows": 0})
                        continue
                    result = _fetch(symbol, first, last, _flush)
                    if result.status == NO_CONTRACT:
                        no_contract.append(symbol)
                        pending_ledger.append({"key": f"{symbol}|*", "sym": symbol, "status": NO_CONTRACT, "message": result.message, "at": stamp.isoformat(), "run_id": run_id})
                        report.note(NO_CONTRACT)
                        break
                    if result.status == ERROR:
                        report.note(ERROR)
                        report.notes.append(f"{symbol} {first}: {result.message}")
                        log(f"ib m30 {symbol} {first}..{last}: ERROR {result.message}")
                        continue
                    if result.status == NO_DATA:
                        listed_after = first
                        pending_ledger.append({**base, "status": EMPTY, "rows": 0})
                        report.note(EMPTY)
                        continue
                    built = m30_rows(symbol, result.bars, first=first, last=last, revision_id=revision, supersedes=supersedes, observed_at=stamp, run_id=run_id)
                    built = [r for r in built if r["interval_start"] not in have and hist._row_key("bar_m30", r) not in bad]
                    have.update(r["interval_start"] for r in built)
                    pending_rows.extend(built)
                    days = [b["interval_start"].astimezone(xcal.EXCHANGE_TZ).date() for b in result.bars]
                    if days and min(days) > first + timedelta(days=LISTING_SLACK_DAYS):
                        listed_after = min(days)
                    pending_ledger.append({**base, "status": DONE, "rows": len(built)})
                    report.note(DONE)
                    log(f"ib m30 {symbol} {first}..{last}: {len(built)} new bars")
        except _Stop:
            stopped = True
        _flush()
        if no_contract:
            hist._no_data_flags(store, "bar_m30", no_contract, now=stamp, run_id=run_id, lock=lock, report=report)
        whole = {s for s in batch if _pull_whole(memory, s, windows)}
        derive_and_flag(store, batch, now=stamp, run_id=run_id, lock=lock, report=report, flag_symbols=whole)
        log(f"ib m30 batch {batch[0]}..{batch[-1]} sealed: {report.rows_published.get('bar_m30', 0)} bars so far")
        if stopped:
            report.status = "PARTIAL"
            report.notes.append("stopped at --max-hours; the next run resumes")
            break
    if report.by_outcome.get(ERROR):
        report.status = "PARTIAL"
    return report


def _recent_no_contract(record: dict | None, now: datetime) -> bool:
    return bool(record) and record.get("status") == NO_CONTRACT and hist._recent(record, now, NO_CONTRACT_RETRY_DAYS)


def _pull_whole(memory: dict, symbol: str, windows) -> bool:
    """Every window of the symbol is closed in the ledger (under its recorded revision)."""
    for first, last in windows:
        record = memory.get(f"{symbol}|{first.isoformat()}")
        if not _window_closed(record, (record or {}).get("revision"), last):
            return False
    return True


def derive_and_flag(store, symbols, *, now, run_id, lock=contextlib.nullcontext, report=None, flag_symbols=None) -> hist.JobReport:
    """H1 and H4 from each symbol's current IBKR M30, then session and cross-check flags.

    Idempotent: only derivations and flags not already stored are written.
    ``flag_symbols`` limits the session flags to symbols whose pull is whole,
    so a window still owed is never flagged as missing sessions.
    """
    report = report or hist.JobReport(job="ib_derive")
    series = _current_ib_m30(store, symbols)
    if not series:
        return report
    names = sorted(series)
    done = hist.existing_keys(store, "bar_derived_history", names, ["symbol", "timeframe", "interval_start", "source_revision_id"])
    yahoo = _yahoo_h1_closes(store, names)
    spy = series.get("SPY") or _current_ib_m30(store, ["SPY"]).get("SPY") or []
    reference = {row["interval_start"].astimezone(xcal.EXCHANGE_TZ).date() for row in spy}
    derived, flags = [], []
    for symbol, rows in series.items():
        revision = rows[0]["revision_id"]
        h1 = derive_h1_rows(rows, provider=PROVIDER_IBKR, source_revision_id=revision, computed_at=now, run_id=run_id)
        h4 = hist.derive_h4_rows([r for r in h1 if r["is_complete"]], provider=PROVIDER_IBKR, source_revision_id=revision, computed_at=now, run_id=run_id)
        for bar in h1 + h4:
            if (symbol, bar["timeframe"], bar["interval_start"], revision) not in done:
                derived.append(bar)
        if flag_symbols is None or symbol in flag_symbols:
            days = None if symbol == "SPY" else reference
            flags.extend(session_flags(symbol, rows, detected_at=now, run_id=run_id, reference_days=days))
        flags.extend(mismatch_flags(symbol, h1, yahoo.get(symbol, {}), detected_at=now, run_id=run_id))
    report.add("bar_derived_history", hist._publish(store, "bar_derived_history", derived, lock=lock, job_id=run_id))
    known = hist.existing_keys(store, "history_quality_flag", names, ["dataset", "symbol", "check", "flag_date"])
    fresh = []
    for row in flags:
        key = (row["dataset"], row["symbol"], row["check"], row["flag_date"])
        if key not in known:
            known.add(key)
            fresh.append(row)
            report.note(row["check"])
    report.add("history_quality_flag", hist._publish(store, "history_quality_flag", fresh, lock=lock, job_id=run_id))
    return report


def ib_coverage(store) -> dict:
    """IBKR intraday in the lake: symbols, bars, span, and the ledger's window tally."""
    table = reader._scan(store, "bar_m30", extra_filter=pads.field("provider") == PROVIDER_IBKR, columns=["symbol", "interval_start"])
    out = {"symbols": 0, "bars": int(table.num_rows), "first": None, "last": None, "windows": {}}
    if table.num_rows:
        import pyarrow.compute as pc

        out["symbols"] = len(pc.unique(table.column("symbol")))
        span = pc.min_max(table.column("interval_start"))
        out["first"] = span["min"].as_py().astimezone(xcal.EXCHANGE_TZ).date().isoformat()
        out["last"] = span["max"].as_py().astimezone(xcal.EXCHANGE_TZ).date().isoformat()
    tally: dict[str, int] = {}
    for record in hist.HistoryLedger(store.root, LEDGER_NAME).latest().values():
        status = str(record.get("status") or "")
        tally[status] = tally.get(status, 0) + 1
    out["windows"] = dict(sorted(tally.items()))
    return out


__all__ = [
    "CAPTURE_PER_WINDOW",
    "IB_START",
    "IbFetch",
    "IbHistoryFetcher",
    "classify",
    "derive_and_flag",
    "derive_h1_rows",
    "ib_coverage",
    "ib_windows",
    "market_guard_until",
    "order_symbols",
    "plan",
    "run_ib_backfill",
]
