"""Five-year provider history into the lake: backfill, top-up, quality (P10).

Writes only the P10 datasets (``bar_d1_history``, ``bar_h1``, ``bar_m30``,
``bar_derived_history``, ``corporate_action``, ``earnings_date``,
``history_quality_flag``) through ``ResearchStore.publish`` - the same seal,
manifest and quarantine as every other dataset. Read it back with
``history_reader``.

Rules this module keeps (pinned by ``tests/test_history_backfill.py``):

* completed sessions only - a session is stored once its close is an hour old;
* idempotent - a row already in the symbol's current revision is never
  re-published, so a re-run adds nothing;
* never edit in place - when the provider's adjustment basis moves (a new split,
  or stored closes that no longer match), the symbol's whole history is pulled
  again as a NEW revision whose ``supersedes_revision_id`` names the old one;
* dirty rows (NaN, non-positive, OHLC out of order, negative volume, a date the
  exchange was closed) go to ``_quarantine`` with a reason, never dropped.
  Rows that are NaN in every price column are a batch download's alignment
  padding (before a listing, after a delisting), not provider rows, and are
  not stored at all;
* absence is recorded: a symbol the provider returned nothing for gets a
  ``PROVIDER_NO_DATA`` flag, and a symbol with no earnings dates an
  ``EARNINGS_NOT_AVAILABLE`` flag - never a guess.

Survivorship: Yahoo serves today's listed names only. Delisted names are
missing from this history, and the coverage report says so.

Network lives only in :class:`YahooClient`; every job takes an injected client,
so the tests run offline.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:  # package import
    from . import exchange_calendar as xcal
    from . import history_reader as reader
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import exchange_calendar as xcal  # type: ignore
    import history_reader as reader  # type: ignore
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION  # type: ignore
    from store import ResearchStore  # type: ignore

PROVIDER_YAHOO = "YAHOO"
ADJUSTMENT_VERSION = "yahoo_split_v1"
CAPTURE_BACKFILL = "BACKFILL"
QUALITY_COMPLETE = "COMPLETE"
D1_START = date(2018, 1, 1)
#: A session is final this long after its close (late prints, volume settle).
SETTLE = timedelta(hours=1)
D1_TOPUP_DAYS = 21  # ~10+ sessions plus holidays
INTRADAY_TOPUP_PERIOD = "5d"
EARNINGS_LIMIT = 40  # ~10 years of quarters; covers 2018+
EARNINGS_REFRESH_DAYS = 7
#: Earnings refreshes per night: the universe cycles in about a week at ~1/s.
EARNINGS_TOPUP_LIMIT = 300
NO_DATA_RETRY_DAYS = 7
#: Median close disagreement that means the provider re-based the series.
REBASE_TOLERANCE = 0.01
JUMP_THRESHOLD = 0.40

INTRADAY = {
    # timeframe: (dataset, yahoo interval, full-history period, minutes)
    "H1": ("bar_h1", "1h", "730d", 60),
    "M30": ("bar_m30", "30m", "60d", 30),
}
H4_BUCKETS = ((9, 30), (13, 30))  # 09:30-13:30 and 13:30-close, ET

PRIORITY_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA", "^VIX")
MACRO_SYMBOLS = ("TLT", "HYG", "USO", "GLD")
_SECTOR_FALLBACK = ("XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU")

LEDGER_DIR = "_history_ledger"
QUARANTINE_MEMORY_DATASETS = frozenset({"bar_d1_history", "bar_h1", "bar_m30"})
SURVIVORSHIP_NOTE = (
    "Yahoo serves currently listed symbols only: delisted names are absent, so "
    "any backtest over this history carries survivorship bias."
)

# Quarantine reasons (the publish validator's verdicts).
BAD_PRICE = "PRICE_NOT_POSITIVE_FINITE"
BAD_OHLC = "OHLC_INCONSISTENT"
BAD_VOLUME = "VOLUME_MISSING_OR_NEGATIVE"
NOT_A_SESSION = "NOT_AN_EXCHANGE_SESSION"
OUTSIDE_RTH = "OUTSIDE_RTH"

# Series-check names (history_quality_flag.check).
FLAG_MISSING_SESSION = "MISSING_SESSION"
FLAG_DUPLICATE = "DUPLICATE_SESSION"
FLAG_STALE = "STALE_REPEAT_BAR"
FLAG_JUMP = "UNEXPLAINED_JUMP"
FLAG_NO_DATA = "PROVIDER_NO_DATA"
FLAG_NO_EARNINGS = "EARNINGS_NOT_AVAILABLE"
FLAG_REVISED = "PROVIDER_REBASED"


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------
def _project_paths():
    try:
        from scripts import project_paths
    except ImportError:  # pragma: no cover
        import project_paths  # type: ignore
    return project_paths


def sector_etfs() -> tuple[str, ...]:
    try:
        try:
            from scripts.group_rrs import SECTOR_ETFS
        except ImportError:  # pragma: no cover
            from group_rrs import SECTOR_ETFS  # type: ignore
        return tuple(sorted(set(SECTOR_ETFS.values())))
    except Exception:  # noqa: BLE001 - a missing module still leaves the known 11
        return tuple(sorted(_SECTOR_FALLBACK))


def industry_etfs(path: Path | None = None) -> list[str]:
    """The distinct industry proxy ETFs the market prep and group tape use (read-only)."""
    target = Path(path) if path is not None else Path(_project_paths().INDUSTRY_ETF_MAP_FILE)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    refs = data.get("yahoo_industryKey_to_ref") if isinstance(data, dict) else None
    if not isinstance(refs, dict):
        return []
    return sorted({str(ref.get("etf") or "").strip().upper() for ref in refs.values() if isinstance(ref, dict)} - {""})


def read_symbol_file(path: Path) -> list[str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        symbol = line.split("#", 1)[0].strip().upper()
        if symbol:
            out.append(symbol)
    return out


def priority_symbols() -> list[str]:
    """Indexes, VIX and the sector ETFs: the regime builder's inputs, pulled first."""
    return list(dict.fromkeys([*PRIORITY_SYMBOLS, *sector_etfs()]))


def etf_symbols(industry_path: Path | None = None) -> set[str]:
    return set(priority_symbols()) | set(industry_etfs(industry_path)) | set(MACRO_SYMBOLS)


def history_universe(
    store: ResearchStore | None = None,
    *,
    universe_file: Path | None = None,
    industry_path: Path | None = None,
) -> list[str]:
    """Priority names first, then universe_all + every legacy bar_d1 symbol + ETFs."""
    names = set(read_symbol_file(universe_file or _project_paths().UNIVERSE_ALL_FILE))
    if store is not None:
        table = reader._scan(store, reader.LEGACY_D1_DATASET)
        if table.num_rows:
            names.update(str(value).upper() for value in set(table.column("symbol").to_pylist()) if value)
    names |= etf_symbols(industry_path)
    first = priority_symbols()
    return first + sorted(names - set(first))


def yahoo_ticker(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


# ---------------------------------------------------------------------------
# the provider adapter (the only network code)
# ---------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """The provider failed a whole request after every retry."""


class YahooClient:
    """yfinance, batched, paced and retried. Returns {symbol: DataFrame}."""

    def __init__(self, *, pause: float = 2.0, retries: int = 4, backoff: float = 10.0, sleep=time.sleep):
        self.pause = pause
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.requests = 0

    def _download(self, tickers, *, attempts: int | None = None, **kwargs):
        try:
            from scripts import yahoo_download
        except ImportError:  # pragma: no cover - scripts/ directly on sys.path
            import yahoo_download  # type: ignore

        last: Exception | None = None
        tries = max(1, int(attempts or self.retries))
        for attempt in range(tries):
            if self.requests:
                self.sleep(self.pause)
            self.requests += 1
            try:
                # The one in-process door to yfinance.download (RULES: shared dict).
                data = yahoo_download.download(
                    tickers,
                    auto_adjust=False,
                    actions=True,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    prepost=False,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - provider failure is retried, then reported
                last = exc
                data = None
            if data is not None and not data.empty:
                return data
            if attempt + 1 < tries:
                self.sleep(self.backoff * (2**attempt))
        raise ProviderError(f"yfinance returned nothing for {len(tickers)} tickers: {last}")

    def fetch_bars(self, symbols, *, interval: str, start: date | None = None, period: str | None = None) -> dict:
        symbols = list(symbols)
        by_ticker = {yahoo_ticker(symbol): symbol for symbol in symbols}
        kwargs = {"interval": interval}
        if start is not None:
            kwargs["start"] = start.isoformat()
        else:
            kwargs["period"] = period or "max"
        out = split_download(self._download(list(by_ticker), **kwargs), by_ticker)
        missing = [ticker for ticker, symbol in by_ticker.items() if symbol not in out]
        if missing and len(missing) < len(by_ticker):
            # One smaller second pass: batch downloads drop tickers under load.
            try:
                second = self._download(missing, attempts=1, **kwargs)
                out.update(split_download(second, {t: by_ticker[t] for t in missing}))
            except ProviderError:
                pass
        return out

    def fetch_earnings(self, symbol: str, *, limit: int = EARNINGS_LIMIT):
        import yfinance as yf

        try:
            from yfinance.exceptions import YFRateLimitError
        except ImportError:  # pragma: no cover
            YFRateLimitError = ()  # noqa: N806
        last: Exception | None = None
        for attempt in range(self.retries):
            if self.requests:
                self.sleep(self.pause)
            self.requests += 1
            try:
                return yf.Ticker(yahoo_ticker(symbol)).get_earnings_dates(limit=limit)
            except YFRateLimitError as exc:
                last = exc  # throttled: back off and ask again
            except Exception as exc:  # noqa: BLE001 - no page / no dates: recorded as a gap
                last = exc
                if attempt >= 1:
                    return None
            self.sleep(self.backoff * (2**attempt))
        raise ProviderError(f"earnings dates for {symbol}: {last}")


def split_download(data, by_ticker: dict) -> dict:
    """Per-symbol frames from a (possibly multi-ticker) yfinance download."""
    out: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out
    multi = isinstance(data.columns, pd.MultiIndex)
    for ticker, symbol in by_ticker.items():
        if multi:
            if ticker not in data.columns.get_level_values(0):
                continue
            frame = data[ticker]
        else:
            frame = data
        frame = frame.dropna(how="all", subset=[c for c in ("Open", "High", "Low", "Close") if c in frame.columns])
        if not frame.empty:
            out[symbol] = frame
    return out


# ---------------------------------------------------------------------------
# row checks and row builders (pure)
# ---------------------------------------------------------------------------
def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_problem(row: dict) -> str | None:
    """The publish validator: a reason sends the row to ``_quarantine``."""
    prices = [_num(row.get(name)) for name in ("open", "high", "low", "close")]
    if any(price is None or price <= 0 for price in prices):
        return BAD_PRICE
    open_, high, low, close = prices
    if high < max(open_, close, low) or low > min(open_, close):
        return BAD_OHLC
    volume = row.get("volume")
    if volume is None or _num(volume) is None or float(volume) < 0:
        return BAD_VOLUME
    if not row.get("session_id"):
        return NOT_A_SESSION
    if row.get("session_phase") not in (None, "RTH"):
        return OUTSIDE_RTH
    return None


def _volume(value):
    number = _num(value)
    return None if number is None else int(round(number))


def last_completed_session(now: datetime) -> date | None:
    day = now.astimezone(xcal.EXCHANGE_TZ).date()
    for _ in range(10):
        session = xcal.trading_session(day)
        if session is not None and session.rth_close_at + SETTLE <= now:
            return day
        day -= timedelta(days=1)
    return None


def new_revision_id(symbol: str, run_id: str) -> str:
    return f"{PROVIDER_YAHOO}:{symbol}:{run_id}"


def _padding(bar) -> bool:
    """NaN in every price column: a batch download's alignment row, not a bar."""
    return all(_num(bar.get(name)) is None for name in ("Open", "High", "Low", "Close"))


def d1_rows(symbol, frame, *, revision_id, supersedes="", observed_at, run_id, through: date | None) -> list[dict]:
    rows = []
    for stamp, bar in frame.iterrows():
        if _padding(bar):
            continue
        day = pd.Timestamp(stamp).date()
        if through is not None and day > through:
            continue  # forming or unsettled session: never stored
        session = xcal.trading_session(day)
        rows.append(
            {
                "symbol": symbol,
                "session_id": session.session_id if session else "",
                "session_date": day,
                "open": _num(bar.get("Open")),
                "high": _num(bar.get("High")),
                "low": _num(bar.get("Low")),
                "close": _num(bar.get("Close")),
                "volume": _volume(bar.get("Volume")),
                "adjustment_version": ADJUSTMENT_VERSION,
                "corporate_action_id": None,
                "provider": PROVIDER_YAHOO,
                "quality": QUALITY_COMPLETE,
                "is_complete": True,
                "event_at": session.rth_close_at if session else datetime.combine(day, datetime.min.time(), timezone.utc),
                "observed_at": observed_at,
                "capture_mode": CAPTURE_BACKFILL,
                "revision_id": revision_id,
                "supersedes_revision_id": supersedes,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    return rows


def _source_hash(symbol, start, values) -> str:
    material = "|".join([symbol, start.isoformat(), *(f"{value}" for value in values)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def intraday_rows(symbol, frame, *, minutes, revision_id, supersedes="", observed_at, run_id) -> list[dict]:
    """Native bars of completed sessions only; interval_end clipped at the close."""
    rows = []
    for stamp, bar in frame.iterrows():
        if _padding(bar):
            continue
        start = pd.Timestamp(stamp)
        if start.tzinfo is None:
            start = start.tz_localize(xcal.EXCHANGE_TZ)
        start = start.to_pydatetime().astimezone(timezone.utc)
        session = xcal.session_for(start)
        if session is not None and session.rth_close_at + SETTLE > observed_at:
            continue  # the session is not final yet
        if session is None:
            end, phase, session_id = start + timedelta(minutes=minutes), None, ""
        else:
            end = min(start + timedelta(minutes=minutes), session.rth_close_at)
            phase, session_id = session.phase_of(start), session.session_id
        values = [_num(bar.get(name)) for name in ("Open", "High", "Low", "Close")]
        volume = _volume(bar.get("Volume"))
        rows.append(
            {
                "symbol": symbol,
                "interval_start": start,
                "interval_end": end,
                "session_id": session_id,
                "session_phase": phase or "CLOSED",
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "volume": volume,
                "vwap": None,
                "trade_count": None,
                "provider": PROVIDER_YAHOO,
                "is_complete": True,
                "quality": QUALITY_COMPLETE,
                "source_hash": _source_hash(symbol, start, [*values, volume]),
                "adjustment_version": ADJUSTMENT_VERSION,
                "event_at": end,
                "observed_at": observed_at,
                "capture_mode": CAPTURE_BACKFILL,
                "revision_id": revision_id,
                "supersedes_revision_id": supersedes,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    return rows


def action_rows(symbol, frame, *, observed_at, run_id, through: date | None = None) -> list[dict]:
    rows = []
    for column, kind in (("Stock Splits", "SPLIT"), ("Dividends", "DIVIDEND")):
        if column not in frame.columns:
            continue
        for stamp, value in frame[column].items():
            number = _num(value)
            if not number:
                continue
            day = pd.Timestamp(stamp).date()
            if through is not None and day > through:
                continue
            session = xcal.trading_session(day)
            rows.append(
                {
                    "symbol": symbol,
                    "action_type": kind,
                    "ex_date": day,
                    "value": number,
                    "corporate_action_id": f"{PROVIDER_YAHOO}:{symbol}:{kind}:{day.isoformat()}",
                    "provider": PROVIDER_YAHOO,
                    "event_at": session.rth_open_at if session else datetime.combine(day, datetime.min.time(), timezone.utc),
                    "observed_at": observed_at,
                    "capture_mode": CAPTURE_BACKFILL,
                    "revision_id": "",
                    "supersedes_revision_id": "",
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
    return rows


def h4_contract_buckets(session) -> list[tuple[datetime, datetime]]:
    """09:30-13:30 and 13:30-close (ET); a half day keeps only the clipped first."""
    out = []
    for hour, minute in H4_BUCKETS:
        start = datetime.combine(
            session.session_date, datetime.min.time().replace(hour=hour, minute=minute), xcal.EXCHANGE_TZ
        ).astimezone(timezone.utc)
        if start >= session.rth_close_at:
            continue
        end = min(start + timedelta(hours=4), session.rth_close_at)
        out.append((start, end))
    return out


def derive_h4_rows(h1_rows: list[dict], *, provider, source_revision_id, computed_at, run_id) -> list[dict]:
    """H4 bars from one symbol's H1 bars under contract ``h4_rth_0930_1330_v1``."""
    by_session: dict[str, list[dict]] = {}
    for row in h1_rows:
        if row.get("session_id"):
            by_session.setdefault(row["session_id"], []).append(row)
    out = []
    for session_id, rows in sorted(by_session.items()):
        session = xcal.trading_session(date.fromisoformat(session_id.split("-", 1)[1]))
        if session is None:
            continue
        for start, end in h4_contract_buckets(session):
            members = sorted(
                (row for row in rows if start <= row["interval_start"] < end), key=lambda row: row["interval_start"]
            )
            expected = math.ceil((end - start).total_seconds() / 3600)
            if not members:
                continue
            complete = len(members) == expected
            span = int((end - start).total_seconds() // 60)
            out.append(
                {
                    "symbol": members[0]["symbol"],
                    "timeframe": "H4",
                    "aggregation_contract_id": reader.H4_CONTRACT_ID,
                    "interval_start": start,
                    "interval_end": end,
                    "session_id": session_id,
                    "open": members[0]["open"],
                    "high": max(row["high"] for row in members),
                    "low": min(row["low"] for row in members),
                    "close": members[-1]["close"],
                    "volume": int(sum(int(row["volume"] or 0) for row in members)),
                    "is_stub": span < 240,
                    "stub_duration_min": span if span < 240 else None,
                    "constituent_count": len(members),
                    "constituent_expected": expected,
                    "is_complete": complete,
                    "quality": QUALITY_COMPLETE if complete else "PARTIAL",
                    "event_at": end,
                    "computed_at": computed_at,
                    "input_capture_mode_worst": CAPTURE_BACKFILL,
                    "provider": provider,
                    "source_revision_id": source_revision_id,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
    return out


def earnings_rows(symbol, frame, *, observed_at, run_id, since: date = D1_START) -> list[dict]:
    rows = []
    if frame is None or len(frame) == 0:
        return rows
    for stamp, record in frame.iterrows():
        moment = pd.Timestamp(stamp)
        if moment.tzinfo is None:
            moment = moment.tz_localize(xcal.EXCHANGE_TZ)
        local = moment.tz_convert(xcal.EXCHANGE_TZ)
        day = local.date()
        if day < since:
            continue
        clock = (local.hour, local.minute)
        if clock == (0, 0):
            tod, at = "UNKNOWN", None
        else:
            tod = "BMO" if clock < (9, 30) else ("AMC" if clock >= (16, 0) else "DURING")
            at = local.to_pydatetime().astimezone(timezone.utc)
        rows.append(
            {
                "symbol": symbol,
                "earnings_date": day,
                "time_of_day": tod,
                "earnings_at": at,
                "eps_estimate": _num(record.get("EPS Estimate")),
                "eps_reported": _num(record.get("Reported EPS")),
                "surprise_pct": _num(record.get("Surprise(%)")),
                "source": "yahoo_earnings_dates",
                "observed_at": observed_at,
                "capture_mode": CAPTURE_BACKFILL,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    return rows


def flag_row(dataset, symbol, check, flag_date, detail, *, detected_at, run_id, interval_start=None) -> dict:
    return {
        "dataset": dataset,
        "symbol": symbol,
        "check": check,
        "flag_date": flag_date,
        "interval_start": interval_start,
        "detail": detail,
        "detected_at": detected_at,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# lake state
# ---------------------------------------------------------------------------
@dataclass
class SeriesState:
    revision_id: str = ""
    closes: dict = field(default_factory=dict)  # key -> close, current revision only
    first_key: object = None


def current_state(store, dataset, symbols, *, key: str, lo: date | None = None) -> dict[str, SeriesState]:
    """Each symbol's current YAHOO revision and its stored closes (from ``lo``)."""
    table = reader._scan(store, dataset, symbols=sorted(symbols), lo=lo)
    if not table.num_rows:
        return {}
    frame = table.select(
        ["symbol", key, "close", "provider", "revision_id", "supersedes_revision_id", "observed_at"]
    ).to_pandas()
    frame = frame[frame["provider"] == PROVIDER_YAHOO]
    out = {}
    for symbol, rows in frame.groupby("symbol"):
        revision = reader._current_revision(rows)
        rows = rows[rows["revision_id"] == revision]
        out[str(symbol)] = SeriesState(
            revision_id=revision,
            closes=dict(zip(rows[key], rows["close"], strict=False)),
            first_key=rows[key].min(),
        )
    return out


def rebased(state: SeriesState, rows: list[dict], key: str) -> bool:
    """True when fetched closes disagree with stored ones: the basis moved."""
    diffs = []
    for row in rows:
        old = state.closes.get(row[key])
        new = _num(row.get("close"))
        if old and new:
            diffs.append(abs(new / old - 1.0))
    if not diffs:
        return False
    diffs.sort()
    return diffs[len(diffs) // 2] > REBASE_TOLERANCE


def existing_keys(store, dataset, symbols, columns, *, lo: date | None = None) -> set:
    table = reader._scan(store, dataset, symbols=sorted(symbols), lo=lo)
    if not table.num_rows:
        return set()
    values = [table.column(name).to_pylist() for name in columns]
    return set(zip(*values, strict=False))


# ---------------------------------------------------------------------------
# ledger (resume + retry memory), kept beside the lake
# ---------------------------------------------------------------------------
class HistoryLedger:
    def __init__(self, root: Path, name: str):
        self.path = Path(root) / LEDGER_DIR / f"{name}.jsonl"

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def latest(self) -> dict[str, dict]:
        state: dict[str, dict] = {}
        if not self.path.exists():
            return state
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            key = str(record.get("symbol") or record.get("key") or "")
            if key:
                state[key] = record
        return state


def _recent(record: dict | None, now: datetime, days: int) -> bool:
    if not record:
        return False
    try:
        stamp = datetime.fromisoformat(str(record.get("at")))
    except ValueError:
        return False
    return now - stamp < timedelta(days=days)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
@dataclass
class JobReport:
    job: str = ""
    status: str = "OK"
    symbols: int = 0
    rows_published: dict = field(default_factory=dict)
    rows_quarantined: dict = field(default_factory=dict)
    by_outcome: dict = field(default_factory=dict)
    repulled: list = field(default_factory=list)
    failed_batches: int = 0
    notes: list = field(default_factory=list)

    def note(self, outcome: str, count: int = 1) -> None:
        self.by_outcome[outcome] = self.by_outcome.get(outcome, 0) + count

    def add(self, dataset: str, result) -> None:
        if result is None:
            return
        self.rows_published[dataset] = self.rows_published.get(dataset, 0) + result.rows_published
        self.rows_quarantined[dataset] = self.rows_quarantined.get(dataset, 0) + result.rows_quarantined


def _row_key(dataset: str, row: dict) -> str:
    moment = row.get("session_date") if dataset == "bar_d1_history" else row.get("interval_start")
    return f"{dataset}|{row.get('symbol')}|{moment.isoformat() if hasattr(moment, 'isoformat') else moment}"


def _publish(store, dataset, rows, *, lock, job_id, validate=None):
    """Seal rows; remember quarantined bar keys so a re-run does not re-offer them."""
    if not rows:
        return None
    with lock():
        result = store.publish(dataset, rows, job_id=job_id, validate=validate)
    if result.dirty and dataset in QUARANTINE_MEMORY_DATASETS:
        ledger = HistoryLedger(store.root, "quarantined")
        for item in result.dirty:
            ledger.append({"key": _row_key(dataset, item.row), "reason": item.reason, "run_id": job_id})
    return result


def quarantined_keys(store) -> set:
    return set(HistoryLedger(store.root, "quarantined").latest())


def _batches(items, size):
    items = list(items)
    for index in range(0, len(items), max(1, size)):
        yield items[index : index + size]


def _no_data_flags(store, dataset, symbols, *, now, run_id, lock, report):
    known = existing_keys(store, "history_quality_flag", symbols, ["dataset", "symbol", "check"])
    rows = [
        flag_row(dataset, symbol, FLAG_NO_DATA, now.date(), "provider returned no bars", detected_at=now, run_id=run_id)
        for symbol in symbols
        if (dataset, symbol, FLAG_NO_DATA) not in known
    ]
    report.add("history_quality_flag", _publish(store, "history_quality_flag", rows, lock=lock, job_id=run_id))


def run_d1(
    store: ResearchStore,
    symbols,
    *,
    client,
    now: datetime | None = None,
    run_id: str = "",
    mode: str = "backfill",
    start: date = D1_START,
    batch_size: int = 100,
    lock=contextlib.nullcontext,
    log=print,
) -> JobReport:
    """D1 + splits/dividends. ``backfill`` pulls from ``start`` for symbols with no
    history yet; ``topup`` pulls the last ~10 sessions for the rest and re-pulls
    any symbol whose basis moved."""
    stamp = now or utc_now()
    run_id = run_id or f"history_d1_{mode}_{stamp:%Y%m%dT%H%M%SZ}"
    report = JobReport(job=f"d1_{mode}")
    through = last_completed_session(stamp)
    ledger = HistoryLedger(store.root, "d1")
    memory = ledger.latest()
    symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    report.symbols = len(symbols)
    have = current_state(store, "bar_d1_history", symbols, key="session_date", lo=stamp.date() - timedelta(days=60))
    fresh = [s for s in symbols if s not in have and not (memory.get(s, {}).get("status") == "NO_DATA" and _recent(memory.get(s), stamp, NO_DATA_RETRY_DAYS))]
    stale = [s for s in symbols if s in have] if mode == "topup" else []
    repull: list[str] = []

    def _ingest(batch, frames, *, full: bool):
        states = have if not full else {}
        bad = quarantined_keys(store)
        actions = existing_keys(store, "corporate_action", batch, ["symbol", "action_type", "ex_date"])
        rows, action_out, no_data = [], [], []
        for symbol in batch:
            frame = frames.get(symbol)
            if frame is None or frame.empty:
                no_data.append(symbol)
                ledger.append({"symbol": symbol, "status": "NO_DATA", "at": stamp.isoformat(), "run_id": run_id})
                continue
            fetched_actions = action_rows(symbol, frame, observed_at=stamp, run_id=run_id, through=through)
            new_actions = [a for a in fetched_actions if (symbol, a["action_type"], a["ex_date"]) not in actions]
            state = states.get(symbol)
            if state is None:
                prior = have.get(symbol)
                revision = new_revision_id(symbol, run_id)
                built = d1_rows(
                    symbol, frame, revision_id=revision, supersedes=prior.revision_id if prior else "",
                    observed_at=stamp, run_id=run_id, through=through,
                )
                outcome = "REPULLED" if prior else "OK"
            else:
                built = d1_rows(symbol, frame, revision_id=state.revision_id, observed_at=stamp, run_id=run_id, through=through)
                new_split = any(a["action_type"] == "SPLIT" for a in new_actions)
                if new_split or rebased(state, built, "session_date"):
                    repull.append(symbol)
                    report.note("REBASED")
                    continue
                built = [row for row in built if row["session_date"] not in state.closes]
                outcome = "OK"
            built = [row for row in built if _row_key("bar_d1_history", row) not in bad]
            rows.extend(built)
            action_out.extend(new_actions)
            report.note(outcome)
            ledger.append({"symbol": symbol, "status": outcome, "rows": len(built), "at": stamp.isoformat(), "run_id": run_id})
        report.add("bar_d1_history", _publish(store, "bar_d1_history", rows, lock=lock, job_id=run_id, validate=row_problem))
        report.add("corporate_action", _publish(store, "corporate_action", action_out, lock=lock, job_id=run_id))
        if no_data:
            report.note("NO_DATA", len(no_data))
            _no_data_flags(store, "bar_d1_history", no_data, now=stamp, run_id=run_id, lock=lock, report=report)

    def _pull(batch, *, full):
        try:
            if full:
                frames = client.fetch_bars(batch, interval="1d", start=start)
            else:
                frames = client.fetch_bars(batch, interval="1d", start=stamp.date() - timedelta(days=D1_TOPUP_DAYS))
        except ProviderError as exc:
            report.failed_batches += 1
            report.notes.append(f"batch {batch[0]}..{batch[-1]}: {exc}")
            log(f"d1 batch failed {batch[0]}..{batch[-1]}: {exc}")
            return
        _ingest(batch, frames, full=full)
        log(f"d1 {'full' if full else 'recent'} {batch[0]}..{batch[-1]} ok: {report.rows_published.get('bar_d1_history', 0)} rows so far")

    for batch in _batches(fresh, batch_size):
        _pull(batch, full=True)
    for batch in _batches(stale, batch_size):
        _pull(batch, full=False)
    for batch in _batches(repull, batch_size):
        report.repulled.extend(batch)
        _pull(batch, full=True)
    if report.failed_batches:
        report.status = "PARTIAL"
    return report


def run_intraday(
    store: ResearchStore,
    symbols,
    timeframe: str,
    *,
    client,
    now: datetime | None = None,
    run_id: str = "",
    mode: str = "backfill",
    batch_size: int = 100,
    lock=contextlib.nullcontext,
    log=print,
) -> JobReport:
    """Native H1/M30 bars, then (for H1) the H4 derivation. Same revision rules as D1."""
    dataset, interval, full_period, minutes = INTRADAY[timeframe]
    stamp = now or utc_now()
    run_id = run_id or f"history_{timeframe.lower()}_{mode}_{stamp:%Y%m%dT%H%M%SZ}"
    report = JobReport(job=f"{timeframe.lower()}_{mode}")
    ledger = HistoryLedger(store.root, timeframe.lower())
    memory = ledger.latest()
    symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    report.symbols = len(symbols)
    have = current_state(store, dataset, symbols, key="interval_start", lo=stamp.date() - timedelta(days=20))
    fresh = [s for s in symbols if s not in have and not (memory.get(s, {}).get("status") == "NO_DATA" and _recent(memory.get(s), stamp, NO_DATA_RETRY_DAYS))]
    stale = [s for s in symbols if s in have] if mode == "topup" else []
    repull: list[str] = []

    def _pull(batch, *, full):
        try:
            frames = client.fetch_bars(batch, interval=interval, period=full_period if full else INTRADAY_TOPUP_PERIOD)
        except ProviderError as exc:
            report.failed_batches += 1
            report.notes.append(f"batch {batch[0]}..{batch[-1]}: {exc}")
            log(f"{timeframe} batch failed {batch[0]}..{batch[-1]}: {exc}")
            return
        rows, no_data, touched = [], [], {}
        bad = quarantined_keys(store)
        for symbol in batch:
            frame = frames.get(symbol)
            if frame is None or frame.empty:
                no_data.append(symbol)
                ledger.append({"symbol": symbol, "status": "NO_DATA", "at": stamp.isoformat(), "run_id": run_id})
                continue
            state = None if full else have.get(symbol)
            if state is None:
                prior = have.get(symbol)
                revision = new_revision_id(symbol, run_id)
                built = intraday_rows(
                    symbol, frame, minutes=minutes, revision_id=revision,
                    supersedes=prior.revision_id if prior else "", observed_at=stamp, run_id=run_id,
                )
                outcome = "REPULLED" if prior else "OK"
                if prior is not None:
                    built = carry_forward(store, dataset, symbol, prior, built, observed_at=stamp, run_id=run_id) + built
            else:
                built = intraday_rows(symbol, frame, minutes=minutes, revision_id=state.revision_id, observed_at=stamp, run_id=run_id)
                if rebased(state, built, "interval_start"):
                    repull.append(symbol)
                    report.note("REBASED")
                    continue
                built = [row for row in built if row["interval_start"] not in state.closes]
                revision = state.revision_id
                outcome = "OK"
            built = [row for row in built if _row_key(dataset, row) not in bad]
            rows.extend(built)
            touched[symbol] = revision
            report.note(outcome)
            ledger.append({"symbol": symbol, "status": outcome, "rows": len(built), "at": stamp.isoformat(), "run_id": run_id})
        report.add(dataset, _publish(store, dataset, rows, lock=lock, job_id=run_id, validate=row_problem))
        if no_data:
            report.note("NO_DATA", len(no_data))
            _no_data_flags(store, dataset, no_data, now=stamp, run_id=run_id, lock=lock, report=report)
        if timeframe == "H1" and touched:
            derive_h4(store, touched, lo=None if full else stamp.date() - timedelta(days=10), now=stamp, run_id=run_id, lock=lock, report=report)
        log(f"{timeframe} {'full' if full else 'recent'} {batch[0]}..{batch[-1]} ok: {report.rows_published.get(dataset, 0)} rows so far")

    for batch in _batches(fresh, batch_size):
        _pull(batch, full=True)
    for batch in _batches(stale, batch_size):
        _pull(batch, full=False)
    for batch in _batches(repull, batch_size):
        report.repulled.extend(batch)
        _pull(batch, full=True)
    if report.failed_batches:
        report.status = "PARTIAL"
    return report


CARRIED_ADJUSTMENT = f"{ADJUSTMENT_VERSION}+carried"
CAPTURE_RECONSTRUCTED = "RECONSTRUCTED"


def carry_forward(store, dataset, symbol, prior: SeriesState, built: list[dict], *, observed_at, run_id) -> list[dict]:
    """Bars older than the provider's window, re-based into the new revision.

    The provider only serves the last 730 (H1) / 60 (M30) days, so a split
    re-pull would otherwise shrink the readable history. Older bars of the old
    revision are copied into the new one with prices divided (volume
    multiplied) by the measured basis ratio, marked RECONSTRUCTED and
    ``yahoo_split_v1+carried``. The old revision's rows are never touched.
    """
    ratios = []
    for row in built:
        old = prior.closes.get(row["interval_start"])
        new = _num(row.get("close"))
        if old and new:
            ratios.append(old / new)
    if not ratios or not built:
        return []
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    first_new = min(row["interval_start"] for row in built)
    table = reader._scan(store, dataset, symbols=[symbol], hi=first_new.date())
    if not table.num_rows:
        return []
    out = []
    for row in table.to_pylist():
        if row["revision_id"] != prior.revision_id or row["interval_start"] >= first_new:
            continue
        carried = dict(row)
        for name in ("open", "high", "low", "close"):
            carried[name] = row[name] / ratio if row[name] is not None else None
        carried["volume"] = None if row["volume"] is None else int(round(row["volume"] * ratio))
        carried.update(
            adjustment_version=CARRIED_ADJUSTMENT,
            capture_mode=CAPTURE_RECONSTRUCTED,
            observed_at=observed_at,
            revision_id=built[0]["revision_id"],
            supersedes_revision_id=prior.revision_id,
            run_id=run_id,
            source_hash=_source_hash(symbol, row["interval_start"], [carried["close"], carried["volume"]]),
        )
        out.append(carried)
    return out


def derive_h4(store, revisions: dict, *, lo, now, run_id, lock, report) -> None:
    """H4 for each symbol's current H1 revision; existing derivations are skipped."""
    symbols = sorted(revisions)
    table = reader._scan(store, "bar_h1", symbols=symbols, lo=lo)
    if not table.num_rows:
        return
    done = existing_keys(
        store, "bar_derived_history", symbols, ["symbol", "interval_start", "source_revision_id"], lo=lo
    )
    rows_by_symbol: dict[str, list[dict]] = {}
    for row in table.to_pylist():
        if row["revision_id"] == revisions.get(row["symbol"]) and row["provider"] == PROVIDER_YAHOO:
            rows_by_symbol.setdefault(row["symbol"], []).append(row)
    out = []
    for symbol, rows in rows_by_symbol.items():
        for bar in derive_h4_rows(rows, provider=PROVIDER_YAHOO, source_revision_id=revisions[symbol], computed_at=now, run_id=run_id):
            if (symbol, bar["interval_start"], bar["source_revision_id"]) not in done:
                out.append(bar)
    report.add("bar_derived_history", _publish(store, "bar_derived_history", out, lock=lock, job_id=run_id))


def run_earnings(
    store: ResearchStore,
    symbols,
    *,
    client,
    now: datetime | None = None,
    run_id: str = "",
    refresh_days: int = EARNINGS_REFRESH_DAYS,
    flush_every: int = 50,
    max_symbols: int | None = None,
    lock=contextlib.nullcontext,
    log=print,
) -> JobReport:
    """Earnings dates, ~1 request/s, resumable: a symbol refreshed within
    ``refresh_days`` is skipped. ETFs and indexes have none and are not asked.
    ``max_symbols`` caps one run (the nightly slice), oldest refresh first."""
    stamp = now or utc_now()
    run_id = run_id or f"history_earnings_{stamp:%Y%m%dT%H%M%SZ}"
    report = JobReport(job="earnings")
    ledger = HistoryLedger(store.root, "earnings")
    memory = ledger.latest()
    skip = etf_symbols()
    todo = [
        s for s in (str(x).strip().upper() for x in symbols)
        if s and s not in skip and not s.startswith("^") and not _recent(memory.get(s), stamp, refresh_days)
    ]
    if max_symbols is not None:
        todo = sorted(todo, key=lambda s: str((memory.get(s) or {}).get("at") or ""))[: max(0, int(max_symbols))]
    report.symbols = len(todo)
    pending_rows: list[dict] = []
    pending_missing: list[str] = []

    def _flush():
        nonlocal pending_rows, pending_missing
        if pending_rows:
            names = sorted({row["symbol"] for row in pending_rows})
            known = existing_keys(store, "earnings_date", names, ["symbol", "earnings_date", "source"])
            fresh = [r for r in pending_rows if (r["symbol"], r["earnings_date"], r["source"]) not in known]
            report.add("earnings_date", _publish(store, "earnings_date", fresh, lock=lock, job_id=run_id))
        if pending_missing:
            known = existing_keys(store, "history_quality_flag", pending_missing, ["dataset", "symbol", "check"])
            flags = [
                flag_row("earnings_date", s, FLAG_NO_EARNINGS, stamp.date(), "provider has no earnings dates", detected_at=stamp, run_id=run_id)
                for s in pending_missing
                if ("earnings_date", s, FLAG_NO_EARNINGS) not in known
            ]
            report.add("history_quality_flag", _publish(store, "history_quality_flag", flags, lock=lock, job_id=run_id))
        pending_rows, pending_missing = [], []

    for index, symbol in enumerate(todo, start=1):
        try:
            frame = client.fetch_earnings(symbol)
        except ProviderError as exc:
            report.note("ERROR")
            report.notes.append(f"{symbol}: {exc}")
            continue
        rows = earnings_rows(symbol, frame, observed_at=stamp, run_id=run_id)
        if rows:
            pending_rows.extend(rows)
            report.note("OK")
            status = "OK"
        else:
            pending_missing.append(symbol)
            report.note("NO_DATA")
            status = "NO_DATA"
        ledger.append({"symbol": symbol, "status": status, "rows": len(rows), "at": stamp.isoformat(), "run_id": run_id})
        if index % flush_every == 0:
            _flush()
            log(f"earnings {index}/{len(todo)}")
    _flush()
    return report


# ---------------------------------------------------------------------------
# quality: series checks and the coverage report
# ---------------------------------------------------------------------------
def series_flags(symbol, frame, *, reference_days: set, splits: set, detected_at, run_id, dataset="bar_d1_history") -> list[dict]:
    """Missing sessions (vs sessions the reference series traded), stale repeats,
    and >40% day moves with no split within three days."""
    flags = []
    if frame.empty:
        return flags
    days = list(frame["session_date"])
    present = set(days)
    first, last = days[0], days[-1]
    for session in xcal.sessions_between(first, last):
        day = session.session_date
        if day not in present and (not reference_days or day in reference_days):
            flags.append(flag_row(dataset, symbol, FLAG_MISSING_SESSION, day, "no bar for an exchange session", detected_at=detected_at, run_id=run_id))
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy()
    closes = frame["close"].to_numpy()
    for index in range(1, len(frame)):
        day = days[index]
        if (values[index] == values[index - 1]).all():
            flags.append(flag_row(dataset, symbol, FLAG_STALE, day, "OHLCV identical to the previous session", detected_at=detected_at, run_id=run_id))
        previous = closes[index - 1]
        if previous and abs(closes[index] / previous - 1.0) > JUMP_THRESHOLD:
            near = any(abs((day - split).days) <= 3 for split in splits)
            if not near:
                move = closes[index] / previous - 1.0
                flags.append(flag_row(dataset, symbol, FLAG_JUMP, day, f"close moved {move:+.1%} with no split", detected_at=detected_at, run_id=run_id))
    return flags


def run_quality(store: ResearchStore, *, symbols=None, now: datetime | None = None, run_id: str = "", lock=contextlib.nullcontext) -> JobReport:
    """Series checks over the provider D1 history; only new flags are written."""
    stamp = now or utc_now()
    run_id = run_id or f"history_quality_{stamp:%Y%m%dT%H%M%SZ}"
    report = JobReport(job="quality")
    series = {s: f for s, f in reader.read_d1(symbols, store=store).items() if set(f["source_dataset"]) == {reader.D1_DATASET}}
    report.symbols = len(series)
    reference = set(series["SPY"]["session_date"]) if "SPY" in series else set()
    actions = reader.read_corporate_actions(list(series), store=store)
    splits: dict[str, set] = {}
    for symbol, kind, day in zip(actions["symbol"], actions["action_type"], actions["ex_date"], strict=False):
        if kind == "SPLIT":
            splits.setdefault(symbol, set()).add(day)
    raw = reader._scan(store, reader.D1_DATASET, symbols=sorted(series))
    rows = []
    if raw.num_rows:
        dup = raw.select(["symbol", "session_date", "revision_id"]).to_pandas()
        dup = dup[dup.duplicated(["symbol", "session_date", "revision_id"], keep="first")]
        for symbol, day in zip(dup["symbol"], dup["session_date"], strict=False):
            rows.append(flag_row(reader.D1_DATASET, symbol, FLAG_DUPLICATE, day, "same session twice in one revision", detected_at=stamp, run_id=run_id))
    for symbol, frame in series.items():
        rows.extend(series_flags(symbol, frame, reference_days=reference, splits=splits.get(symbol, set()), detected_at=stamp, run_id=run_id))
    known = existing_keys(store, "history_quality_flag", sorted(series), ["dataset", "symbol", "check", "flag_date"])
    fresh, seen = [], set()
    for row in rows:
        key = (row["dataset"], row["symbol"], row["check"], row["flag_date"])
        if key in known or key in seen:
            continue
        seen.add(key)
        fresh.append(row)
    for row in fresh:
        report.note(row["check"])
    report.add("history_quality_flag", _publish(store, "history_quality_flag", fresh, lock=lock, job_id=run_id))
    return report


def coverage_report(store: ResearchStore, *, symbols=None) -> dict:
    """Per symbol: first/last date, sessions, flags; plus dataset totals."""
    series = reader.read_d1(symbols, store=store)
    flags = reader.read_quality_flags(store=store)
    flag_counts: dict[tuple, int] = {}
    for symbol, check in zip(flags["symbol"], flags["check"], strict=False):
        flag_counts[(symbol, check)] = flag_counts.get((symbol, check), 0) + 1
    per_symbol = {}
    years = set()
    for symbol, frame in series.items():
        if frame.empty:
            continue
        first, last = frame["session_date"].iloc[0], frame["session_date"].iloc[-1]
        years.update(range(first.year, last.year + 1))
        per_symbol[symbol] = {
            "first": first.isoformat(),
            "last": last.isoformat(),
            "sessions": int(len(frame)),
            "source": str(frame["source_dataset"].iloc[0]),
            "provider": str(frame["provider"].iloc[0]),
            "missing_sessions": flag_counts.get((symbol, FLAG_MISSING_SESSION), 0),
            "stale": flag_counts.get((symbol, FLAG_STALE), 0),
            "jumps": flag_counts.get((symbol, FLAG_JUMP), 0),
        }
    datasets = {}
    live = store.manifest.resolve()
    for name in ("bar_d1_history", "bar_h1", "bar_m30", "bar_derived_history", "corporate_action", "earnings_date", "history_quality_flag"):
        entries = [entry for entry in live.entries if entry.dataset == name]
        datasets[name] = {"files": len(entries), "rows": sum(entry.row_count for entry in entries)}
    quarantined: dict[str, int] = {}
    for entry in store.manifest.quarantine_entries():
        if entry.dataset in datasets:
            quarantined[entry.dataset] = quarantined.get(entry.dataset, 0) + entry.row_count
    earnings = reader.read_earnings_dates(store=store)
    history_symbols = [s for s, row in per_symbol.items() if row["source"] == reader.D1_DATASET]
    return {
        "symbols": len(per_symbol),
        "history_symbols": len(history_symbols),
        "legacy_only_symbols": len(per_symbol) - len(history_symbols),
        "years_covered": [min(years), max(years)] if years else [],
        "datasets": datasets,
        "quarantined_rows": quarantined,
        "earnings_symbols": len(earnings),
        "earnings_dates": sum(len(days) for days in earnings.values()),
        "flags": {check: int((flags["check"] == check).sum()) for check in sorted(set(flags["check"]))},
        "survivorship": SURVIVORSHIP_NOTE,
        "per_symbol": per_symbol,
    }


def run_topup(store: ResearchStore, *, client, symbols=None, now: datetime | None = None, lock=contextlib.nullcontext, log=print) -> dict:
    """The daily keep-it-flowing pass: D1 (+ re-pulls, + new names), H1/H4, M30,
    earnings when a week old, then the series checks. Idempotent."""
    stamp = now or utc_now()
    names = list(symbols) if symbols is not None else history_universe(store)
    out: dict = {"started_at": stamp.isoformat(), "symbols": len(names)}
    out["d1"] = vars(run_d1(store, names, client=client, now=stamp, mode="topup", lock=lock, log=log))
    for timeframe in ("H1", "M30"):
        out[timeframe.lower()] = vars(run_intraday(store, names, timeframe, client=client, now=stamp, mode="topup", lock=lock, log=log))
    out["earnings"] = vars(
        run_earnings(store, names, client=client, now=stamp, max_symbols=EARNINGS_TOPUP_LIMIT, lock=lock, log=log)
    )
    out["quality"] = vars(run_quality(store, now=stamp, lock=lock))
    coverage = coverage_report(store)
    coverage.pop("per_symbol", None)
    out["coverage"] = coverage
    return out


__all__ = [
    "ADJUSTMENT_VERSION",
    "D1_START",
    "HistoryLedger",
    "JobReport",
    "ProviderError",
    "SURVIVORSHIP_NOTE",
    "YahooClient",
    "coverage_report",
    "derive_h4_rows",
    "history_universe",
    "priority_symbols",
    "row_problem",
    "run_d1",
    "run_earnings",
    "run_intraday",
    "run_quality",
    "run_topup",
]
