"""Read the lake's provider history for backtests (P10, 2026-09-27).

Pure reads over the manifest-resolved files: nothing here writes, fetches or
touches Qt, so never call it on the Qt thread (a full-universe read is seconds).

The contract every reader keeps: ONE consistent series per symbol.

* D1 comes from ``bar_d1_history`` - the preferred provider (YAHOO, then IBKR),
  its latest revision only (a revision named in another row's
  ``supersedes_revision_id`` is never read). A symbol with no history rows falls
  back to the legacy ``bar_d1`` rows, and the ``provider`` / ``source_dataset``
  columns say so. Providers are never mixed inside one symbol's series.
* Intraday is completed bars only, with ``interval_start`` tz-aware in
  America/New_York. Each symbol has ONE intraday basis provider for M30, H1
  and H4 alike: the provider whose native intraday series spans the longest
  (IBKR on a tie). Yahoo basis: native H1 and M30, H4 derived from H1. IBKR
  basis: native M30 (5 years), H1 and H4 derived from that M30. The other
  provider's rows stay stored as a cross-check, never mixed in.
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
#: H1 built from two RTH M30 bars: 09:30-10:30 ... 15:30-16:00 (half days clipped).
H1_FROM_M30_CONTRACT_ID = "h1_from_m30_rth_v1"
CORPORATE_ACTION_DATASET = "corporate_action"
EARNINGS_DATASET = "earnings_date"
QUALITY_FLAG_DATASET = "history_quality_flag"
#: First provider present wins for a symbol; anything else sorts after these.
PROVIDER_PREFERENCE = ("YAHOO", "IBKR")
#: Intraday basis tie-break: IBKR first (its M30 is the only 5-year intraday).
INTRADAY_PROVIDER_PREFERENCE = ("IBKR", "YAHOO")
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
    columns: list[str] | None = None,
) -> pa.Table:
    """Manifest-resolved rows, narrowed in Arrow before pandas sees them."""
    spec = dataset_spec(dataset)
    entries = [
        entry
        for entry in store.manifest.resolve(dataset=dataset).entries
        if entry.partition.startswith(partition_prefix) and _partition_in_range(entry.partition, lo, hi)
    ]
    if not entries:
        table = spec.schema.empty_table()
        return table.select(columns) if columns else table
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
    return dataset_obj.to_table(filter=predicate, columns=columns)


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


def intraday_basis(symbols=None, *, store: ResearchStore | None = None) -> dict[str, str]:
    """Each symbol's intraday basis provider: the longest native intraday span.

    Span is last minus first ``interval_start`` over ``bar_m30`` and ``bar_h1``
    together, per provider; a tie goes to IBKR. A symbol with no native
    intraday rows is absent.
    """
    lake = _open(store)
    wanted = _norm_symbols(symbols)
    if wanted == []:
        return {}
    spans: dict[tuple[str, str], list] = {}
    for dataset in INTRADAY_DATASETS.values():
        table = _scan(lake, dataset, symbols=wanted, columns=["symbol", "provider", "interval_start"])
        if not table.num_rows:
            continue
        grouped = table.group_by(["symbol", "provider"]).aggregate(
            [("interval_start", "min"), ("interval_start", "max")]
        )
        for symbol, provider, lo, hi in zip(
            grouped.column("symbol").to_pylist(),
            grouped.column("provider").to_pylist(),
            grouped.column("interval_start_min").to_pylist(),
            grouped.column("interval_start_max").to_pylist(),
            strict=False,
        ):
            key = (str(symbol), str(provider or ""))
            old = spans.get(key)
            spans[key] = [lo, hi] if old is None else [min(old[0], lo), max(old[1], hi)]
    best: dict[str, tuple] = {}
    order = INTRADAY_PROVIDER_PREFERENCE
    for (symbol, provider), (lo, hi) in spans.items():
        name = provider.upper()
        tie = order.index(name) if name in order else len(order)
        rank = (-(hi - lo).total_seconds(), tie, provider)
        if symbol not in best or rank < best[symbol][0]:
            best[symbol] = (rank, provider)
    return {symbol: provider for symbol, (_rank, provider) in sorted(best.items())}


def _pick_provider(frame: pd.DataFrame, basis: str | None) -> pd.DataFrame:
    """Rows of the basis provider only (maybe none); without a basis, the D1 preference order."""
    names = frame["provider"].fillna("")
    if basis is not None:
        return frame[names == basis]
    providers = sorted(names.unique(), key=lambda name: (_provider_rank(name), name))
    return frame[names == providers[0]]


def _derived_series(frame: pd.DataFrame, basis: str | None) -> pd.DataFrame:
    frame = _pick_provider(frame, basis)
    if frame.empty:
        return frame
    # The newest derivation names the current source revision.
    current = frame.sort_values("computed_at").iloc[-1]["source_revision_id"]
    frame = frame[frame["source_revision_id"] == current]
    frame = frame.sort_values(["interval_start", "computed_at"]).drop_duplicates("interval_start", keep="first")
    frame = frame.rename(columns={"source_revision_id": "revision_id"})
    return frame[INTRADAY_COLUMNS + ["is_stub", "constituent_count"]].reset_index(drop=True)


def _scan_derived(lake, timeframe: str, contract: str, *, symbols, lo, hi) -> pd.DataFrame:
    return _scan(
        lake,
        DERIVED_DATASET,
        symbols=symbols,
        lo=lo,
        hi=hi,
        extra_filter=pads.field("aggregation_contract_id") == contract,
        partition_prefix=f"timeframe={timeframe}/",
    ).to_pandas()


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

    ``start``/``end`` are inclusive exchange dates. Every timeframe of a symbol
    comes from its :func:`intraday_basis` provider. H4 is contract
    ``H4_CONTRACT_ID`` (09:30-13:30 and 13:30-16:00 ET); an IBKR-basis H1 is
    contract ``H1_FROM_M30_CONTRACT_ID``.
    """
    frame_name = str(timeframe or "").strip().upper()
    if frame_name not in {*INTRADAY_DATASETS, "H4"}:
        raise ValueError(f"unsupported timeframe {timeframe!r}; use H1, H4 or M30")
    lake = _open(store)
    wanted = _norm_symbols(symbols)
    lo, hi = _as_day(start), _as_day(end)
    out: dict[str, pd.DataFrame] = {}
    if wanted == []:
        return out
    basis = intraday_basis(wanted, store=lake)
    if frame_name in INTRADAY_DATASETS:
        table = _scan(lake, INTRADAY_DATASETS[frame_name], symbols=wanted, lo=lo, hi=hi).to_pandas()
        if not table.empty:
            table = _completed(table, now)
            for symbol, frame in table.groupby("symbol", sort=True):
                name = str(symbol)
                if frame_name == "H1" and basis.get(name) == "IBKR":
                    continue  # an IBKR-basis H1 is the M30-derived one, below
                chosen = _pick_provider(frame, basis.get(name))
                if not chosen.empty:
                    out[name] = _one_series(chosen, "interval_start")[INTRADAY_COLUMNS].copy()
        if frame_name == "H1":
            ib_names = [s for s, provider in basis.items() if provider == "IBKR"]
            if ib_names:
                derived = _scan_derived(lake, "H1", H1_FROM_M30_CONTRACT_ID, symbols=ib_names, lo=lo, hi=hi)
                if not derived.empty:
                    derived = _completed(derived, now)
                    for symbol, frame in derived.groupby("symbol", sort=True):
                        out[str(symbol)] = _derived_series(frame, "IBKR")
    else:
        table = _scan_derived(lake, "H4", H4_CONTRACT_ID, symbols=wanted, lo=lo, hi=hi)
        if table.empty:
            return out
        table = _completed(table, now)
        for symbol, frame in table.groupby("symbol", sort=True):
            out[str(symbol)] = _derived_series(frame, basis.get(str(symbol)))
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
    "H1_FROM_M30_CONTRACT_ID",
    "H4_CONTRACT_ID",
    "INTRADAY_COLUMNS",
    "LakeNotConfigured",
    "available_symbols",
    "intraday_basis",
    "read_corporate_actions",
    "read_d1",
    "read_earnings_dates",
    "read_intraday",
    "read_quality_flags",
]
