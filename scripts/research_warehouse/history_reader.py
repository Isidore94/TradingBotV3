"""Read the lake's provider history for backtests (P10, 2026-09-27).

Pure reads over the manifest-resolved files: nothing here writes, fetches or
touches Qt, so never call it on the Qt thread (a full-universe read is seconds).

The contract every reader keeps: ONE consistent series per symbol.

* D1 comes from ``bar_d1_history`` - the preferred provider (YAHOO, then IBKR),
  its latest revision only (a revision named in another row's
  ``supersedes_revision_id`` is never read). A symbol with no history rows falls
  back to the legacy ``bar_d1`` rows, and the ``provider`` / ``source_dataset``
  columns say so. Providers are never mixed inside one symbol's series.
* Intraday (H1, M30 native; H4 derived from H1) is completed bars only, with
  ``interval_start`` tz-aware in America/New_York.
* Earnings dates are the recorded ones; a symbol with none is absent, never
  guessed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads

try:  # package import
    from .schemas import dataset_spec
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    from schemas import dataset_spec  # type: ignore
    from store import ResearchStore  # type: ignore

D1_DATASET = "bar_d1_history"
LEGACY_D1_DATASET = "bar_d1"
INTRADAY_DATASETS = {"H1": "bar_h1", "M30": "bar_m30"}
DERIVED_DATASET = "bar_derived_history"
H4_CONTRACT_ID = "h4_rth_0930_1330_v1"
CORPORATE_ACTION_DATASET = "corporate_action"
EARNINGS_DATASET = "earnings_date"
QUALITY_FLAG_DATASET = "history_quality_flag"
#: First provider present wins for a symbol; anything else sorts after these.
PROVIDER_PREFERENCE = ("YAHOO", "IBKR")
MARKET_TZ = "America/New_York"

D1_COLUMNS = [
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
    "adjustment_version",
    "revision_id",
    "source_dataset",
]
INTRADAY_COLUMNS = [
    "interval_start",
    "interval_end",
    "session_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
    "revision_id",
]


class LakeNotConfigured(RuntimeError):
    """No research lake is configured, so there is no history to read."""


def _open(store: ResearchStore | None) -> ResearchStore:
    if store is not None:
        return store
    opened = ResearchStore.open()
    if opened is None:
        raise LakeNotConfigured("research_store_dir is not configured; there is no lake to read.")
    return opened


def _norm_symbols(symbols) -> list[str] | None:
    if symbols is None:
        return None
    if isinstance(symbols, str):
        symbols = [symbols]
    return sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})


def _as_day(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _partition_in_range(partition: str, lo: date | None, hi: date | None) -> bool:
    """Skip year/month partitions wholly outside [lo, hi]; keep anything else."""
    for piece in str(partition).split("/"):
        key, _, value = piece.partition("=")
        try:
            if key == "year":
                first, last = date(int(value), 1, 1), date(int(value), 12, 31)
            elif key == "month":
                year, month = (int(part) for part in value.split("-"))
                first = date(year, month, 1)
                last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            else:
                continue
        except ValueError:
            return True
        if (lo is not None and last < lo) or (hi is not None and first > hi):
            return False
    return True


def _scan(
    store: ResearchStore,
    dataset: str,
    *,
    symbols=None,
    lo: date | None = None,
    hi: date | None = None,
    extra_filter=None,
    partition_prefix: str = "",
) -> pa.Table:
    """Manifest-resolved rows, narrowed in Arrow before pandas sees them."""
    spec = dataset_spec(dataset)
    entries = [
        entry
        for entry in store.manifest.resolve(dataset=dataset).entries
        if entry.partition.startswith(partition_prefix) and _partition_in_range(entry.partition, lo, hi)
    ]
    if not entries:
        return spec.schema.empty_table()
    paths = []
    for entry in entries:
        path = store.root / entry.file_path
        if not path.exists():
            raise FileNotFoundError(f"{entry.file_path} is manifest-live but missing; restore before reading.")
        paths.append(str(path))
    predicate = extra_filter
    if symbols:
        clause = pads.field("symbol").isin(list(symbols))
        predicate = clause if predicate is None else predicate & clause
    column = spec.time_column
    if lo is not None or hi is not None:
        is_date = pa.types.is_date(spec.schema.field(column).type)
        for bound, upper in ((lo, False), (hi, True)):
            if bound is None:
                continue
            if is_date:
                scalar = pc.scalar(bound)
                clause = pads.field(column) <= scalar if upper else pads.field(column) >= scalar
            else:
                # Dates bound whole exchange days; UTC start of day minus a day is safe.
                edge = datetime.combine(bound + timedelta(days=2 if upper else -1), datetime.min.time(), timezone.utc)
                scalar = pc.scalar(edge)
                clause = pads.field(column) < scalar if upper else pads.field(column) >= scalar
            predicate = clause if predicate is None else predicate & clause
    dataset_obj = pads.dataset(paths, schema=spec.schema, format="parquet")
    return dataset_obj.to_table(filter=predicate)


def _provider_rank(provider: str) -> int:
    name = str(provider or "").upper()
    return PROVIDER_PREFERENCE.index(name) if name in PROVIDER_PREFERENCE else len(PROVIDER_PREFERENCE)


def _current_revision(frame: pd.DataFrame) -> str:
    """The revision nobody supersedes; the most recently observed one on a tie."""
    revisions = [str(value or "") for value in frame["revision_id"].unique()]
    superseded = {str(value) for value in frame["supersedes_revision_id"].dropna().unique() if str(value)}
    live = [revision for revision in revisions if revision not in superseded] or revisions
    if len(live) == 1:
        return live[0]
    seen = frame[frame["revision_id"].isin(live)].groupby("revision_id")["observed_at"].max()
    return str(seen.idxmax())


def _one_series(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """One provider, one revision, one row per ``key`` (first observed wins)."""
    providers = sorted(frame["provider"].fillna("").unique(), key=lambda name: (_provider_rank(name), name))
    chosen = frame[frame["provider"].fillna("") == providers[0]]
    revision = _current_revision(chosen)
    chosen = chosen[chosen["revision_id"].fillna("") == revision]
    chosen = chosen.sort_values([key, "observed_at"], kind="stable").drop_duplicates(key, keep="first")
    return chosen.sort_values(key, kind="stable").reset_index(drop=True)


def available_symbols(*, store: ResearchStore | None = None) -> list[str]:
    """Every symbol with daily bars in the lake (history or legacy ``bar_d1``)."""
    lake = _open(store)
    found: set[str] = set()
    for dataset in (D1_DATASET, LEGACY_D1_DATASET):
        table = _scan(lake, dataset)
        if table.num_rows:
            found.update(str(value) for value in pc.unique(table.column("symbol")).to_pylist() if value)
    return sorted(found)


def read_d1(symbols=None, start=None, end=None, *, store: ResearchStore | None = None) -> dict[str, pd.DataFrame]:
    """Daily bars per symbol, one consistent series each (see module docstring).

    ``start``/``end`` are inclusive session dates. Columns: ``D1_COLUMNS``.
    """
    lake = _open(store)
    wanted = _norm_symbols(symbols)
    if wanted == []:
        return {}
    lo, hi = _as_day(start), _as_day(end)
    out: dict[str, pd.DataFrame] = {}

    history = _scan(lake, D1_DATASET, symbols=wanted, lo=lo, hi=hi).to_pandas()
    if not history.empty:
        for symbol, frame in history.groupby("symbol", sort=True):
            series = _one_series(frame, "session_date")
            series["source_dataset"] = D1_DATASET
            out[str(symbol)] = series[D1_COLUMNS].reset_index(drop=True)

    missing = None if wanted is None else [symbol for symbol in wanted if symbol not in out]
    if missing is None or missing:
        legacy = _scan(lake, LEGACY_D1_DATASET, symbols=missing, lo=lo, hi=hi).to_pandas()
        if not legacy.empty:
            legacy = legacy[~legacy["symbol"].isin(list(out))]
            for symbol, frame in legacy.groupby("symbol", sort=True):
                series = _one_series(frame, "session_date")
                series["source_dataset"] = LEGACY_D1_DATASET
                out[str(symbol)] = series[D1_COLUMNS].reset_index(drop=True)
    return dict(sorted(out.items()))


def _completed(frame: pd.DataFrame, now: datetime | None) -> pd.DataFrame:
    cutoff = pd.Timestamp(now or datetime.now(timezone.utc))
    if cutoff.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    mask = frame["interval_end"] <= cutoff
    if "is_complete" in frame.columns:
        mask &= frame["is_complete"].fillna(False).astype(bool)
    return frame[mask]


def _to_market_time(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("interval_start", "interval_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(MARKET_TZ)
    return frame


def read_intraday(
    timeframe: str,
    symbols=None,
    start=None,
    end=None,
    *,
    store: ResearchStore | None = None,
    now: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Completed H1 / H4 / M30 bars per symbol, ``interval_start`` in ET.

    ``start``/``end`` are inclusive exchange dates. H4 is derived from H1 under
    contract ``H4_CONTRACT_ID`` (09:30-13:30 and 13:30-16:00 ET).
    """
    frame_name = str(timeframe or "").strip().upper()
    lake = _open(store)
    wanted = _norm_symbols(symbols)
    lo, hi = _as_day(start), _as_day(end)
    out: dict[str, pd.DataFrame] = {}
    if wanted == [] and frame_name in {*INTRADAY_DATASETS, "H4"}:
        return out
    if frame_name in INTRADAY_DATASETS:
        table = _scan(lake, INTRADAY_DATASETS[frame_name], symbols=wanted, lo=lo, hi=hi).to_pandas()
        if table.empty:
            return out
        table = _completed(table, now)
        for symbol, frame in table.groupby("symbol", sort=True):
            series = _one_series(frame, "interval_start")
            out[str(symbol)] = series[INTRADAY_COLUMNS].copy()
    elif frame_name == "H4":
        table = _scan(
            lake,
            DERIVED_DATASET,
            symbols=wanted,
            lo=lo,
            hi=hi,
            extra_filter=pads.field("aggregation_contract_id") == H4_CONTRACT_ID,
            partition_prefix="timeframe=H4/",
        ).to_pandas()
        if table.empty:
            return out
        table = _completed(table, now)
        for symbol, frame in table.groupby("symbol", sort=True):
            providers = sorted(frame["provider"].fillna("").unique(), key=lambda name: (_provider_rank(name), name))
            frame = frame[frame["provider"].fillna("") == providers[0]]
            # The newest derivation names the current source revision of H1.
            current = frame.sort_values("computed_at").iloc[-1]["source_revision_id"]
            frame = frame[frame["source_revision_id"] == current]
            frame = frame.sort_values(["interval_start", "computed_at"]).drop_duplicates("interval_start", keep="first")
            frame = frame.rename(columns={"source_revision_id": "revision_id"})
            out[str(symbol)] = frame[INTRADAY_COLUMNS + ["is_stub", "constituent_count"]].reset_index(drop=True)
    else:
        raise ValueError(f"unsupported timeframe {timeframe!r}; use H1, H4 or M30")
    for symbol, frame in out.items():
        frame = _to_market_time(frame.reset_index(drop=True))
        if lo is not None:
            frame = frame[frame["interval_start"].dt.date >= lo]
        if hi is not None:
            frame = frame[frame["interval_start"].dt.date <= hi]
        out[symbol] = frame.reset_index(drop=True)
    return {symbol: frame for symbol, frame in sorted(out.items()) if not frame.empty}


def read_earnings_dates(symbols=None, *, store: ResearchStore | None = None) -> dict[str, list[date]]:
    """Recorded earnings dates per symbol, ascending, de-duplicated across sources."""
    lake = _open(store)
    table = _scan(lake, EARNINGS_DATASET, symbols=_norm_symbols(symbols))
    out: dict[str, set] = {}
    for symbol, day in zip(table.column("symbol").to_pylist(), table.column("earnings_date").to_pylist(), strict=False):
        if symbol and day is not None:
            out.setdefault(str(symbol), set()).add(day)
    return {symbol: sorted(days) for symbol, days in sorted(out.items())}


EARNINGS_EVENT_COLUMNS = ["symbol", "earnings_date", "time_of_day", "eps_estimate", "eps_reported",
                          "surprise_pct", "source"]


def read_earnings_events(symbols=None, *, store: ResearchStore | None = None) -> pd.DataFrame:
    """One row per (symbol, earnings date): a row carrying EPS facts wins, then the first observed.
    Columns: ``EARNINGS_EVENT_COLUMNS`` (BMO/AMC, EPS estimate/actual, surprise %)."""
    lake = _open(store)
    frame = _scan(lake, EARNINGS_DATASET, symbols=_norm_symbols(symbols)).to_pandas()
    if frame.empty:
        return pd.DataFrame(columns=EARNINGS_EVENT_COLUMNS)
    frame["_no_eps"] = frame["eps_reported"].isna() & frame["surprise_pct"].isna()
    frame = frame.sort_values(["symbol", "earnings_date", "_no_eps", "observed_at"], kind="stable")
    frame = frame.drop_duplicates(["symbol", "earnings_date"], keep="first")
    return frame[EARNINGS_EVENT_COLUMNS].reset_index(drop=True)


def read_corporate_actions(symbols=None, *, store: ResearchStore | None = None) -> pd.DataFrame:
    """Splits and dividends as recorded (symbol, action_type, ex_date, value, provider)."""
    lake = _open(store)
    frame = _scan(lake, CORPORATE_ACTION_DATASET, symbols=_norm_symbols(symbols)).to_pandas()
    columns = ["symbol", "action_type", "ex_date", "value", "provider"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.sort_values(["symbol", "ex_date", "action_type", "observed_at"])
    return frame.drop_duplicates(["symbol", "action_type", "ex_date", "provider"])[columns].reset_index(drop=True)


def read_quality_flags(dataset=None, symbols=None, *, store: ResearchStore | None = None) -> pd.DataFrame:
    """Series-quality findings, so a backtest can exclude or inspect flagged bars."""
    lake = _open(store)
    frame = _scan(lake, QUALITY_FLAG_DATASET, symbols=_norm_symbols(symbols)).to_pandas()
    if dataset is not None and not frame.empty:
        frame = frame[frame["dataset"] == dataset]
    columns = ["dataset", "symbol", "check", "flag_date", "interval_start", "detail"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.sort_values(["symbol", "flag_date"])[columns].reset_index(drop=True)


__all__ = [
    "D1_COLUMNS",
    "H4_CONTRACT_ID",
    "INTRADAY_COLUMNS",
    "LakeNotConfigured",
    "available_symbols",
    "read_corporate_actions",
    "read_d1",
    "read_earnings_dates",
    "read_earnings_events",
    "read_intraday",
    "read_quality_flags",
]
