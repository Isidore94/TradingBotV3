"""Setup permutation keys (WISHLIST P1-4 / 4b): the scratch-only backfill.

    python scripts/setup_permutation_backfill.py --scratch <dir> \
        --features <copy of d1_features_history.csv> \
        (--horizons <copy of master_avwap_session_horizon_outcomes.csv> | --daily-bars <copy of daily_bars dir>) \
        [--m5-outcomes <copy of intraday_bounce_outcomes.csv>] [--m5-stamps <copy>] [--review-events <copy>] \
        [--scan-reports <copy dir>] [--environment <copy of the d1 environment store>] \
        --out <scratch>/permutation_outcomes.parquet

Writes ONE long table, one row per episode x horizon, with every facet as it
was on the scan date (``f_<facet>`` columns):

- **swing**: the session's last scan row per (symbol, side, scan date) - the
  `session_horizon_outcomes` v2 grain, joined on the same ``scan_row_id`` - at
  1/3/5/10 exchange sessions. ``win`` = the v2 ``favorable`` flag; ``r`` = the
  side return in ATR20 units of the scan row (close to close, so the v2
  gap-aware execution convention and the literal one book the same number).
  With ``--daily-bars`` the v2 build is re-run over the whole history
  (``window_sessions=None``) instead of reading the 30-session file.
- **m5**: one row per MEASURED M5 episode (`held_run_score.build_episodes`),
  family = its bounce type, ``win`` = the level held 30 minutes, ``r`` = the
  episode's MFE_R (held_run_score's MFE rule). Its facets come from the
  PREVIOUS session's last scan row for the name and side - the D1 picture the
  trader had before the alert, never the same day's later scan. With
  ``--m5-stamps`` the live alert-time stamp (`m5_setup_key_stamp`, same rule)
  is joined on event_id and wins; no usable stamp record = the recompute.

Refuses any input or output inside a live store: copy first.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutations as sp  # noqa: E402  (pure; imports no store path)

POPULATION_SWING = "swing"
POPULATION_M5 = "m5"
SWING_HORIZONS = (1, 3, 5, 10)
#: The M5 population has one horizon: the first 30 minutes (held_run_score).
M5_HORIZON = 0
BACKFILL_VERSION = "setup_permutation_backfill.v1"

BASE_COLUMNS = (
    "population", "episode_id", "symbol", "side", "family", "session", "horizon",
    "win", "r", "r_unit", "outcome_kind", "permutation_rule_version", "backfill_version",
)


def facet_column(name: str) -> str:
    return f"f_{name}"


def output_columns() -> list[str]:
    return [*BASE_COLUMNS, *(facet_column(name) for name in sp.FACETS)]


# --- live-store refusal


class LiveStoreRefused(RuntimeError):
    """An input or output is inside a live store."""


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path))).rstrip("\\/")


def _is_under(path: Path | str, root: Path | str) -> bool:
    child, parent = _norm(path), _norm(root)
    return child == parent or child.startswith(parent + os.sep)


def live_roots() -> list[Path]:
    roots = [Path(r"C:\TradingBotData"), Path(r"\\MINI-PC\Trading Bot Data"),
             Path.home() / "AppData" / "Local" / "TradingBotV3"]
    original = os.environ.get("LOCALAPPDATA")
    if original:
        roots.append(Path(original) / "TradingBotV3")
    return roots


def refuse_live(paths: Iterable[Path | str | None], roots: list[Path] | None = None) -> None:
    roots = roots if roots is not None else live_roots()
    for item in paths:
        if item is None:
            continue
        for root in roots:
            if _is_under(item, root):
                raise LiveStoreRefused(f"{item} is inside live store {root}; copy it to scratch first")


# --- small parsing helpers


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number or number in (float("inf"), float("-inf")) else number


def _truthy(value: Any) -> bool:
    return _text(value).lower() in {"true", "1", "1.0", "yes"}


def _side(value: Any) -> str:
    text = _text(value).upper()
    return text if text in {"LONG", "SHORT"} else ""


# --- pass 1: which scan row speaks for each (symbol, side, session)


def _scan_date(row: Mapping[str, Any]) -> str:
    for column in ("last_trade_date", "run_date"):
        text = _text(row.get(column))[:10]
        if text:
            try:
                date.fromisoformat(text)
            except ValueError:
                continue
            return text
    return ""


def scan_row_id(row: Mapping[str, Any]) -> str:
    """`legacy._scan_factor_row_id`'s shape: symbol:scan_date:run_id (or run_timestamp)."""
    suffix = _text(row.get("run_id")) or _text(row.get("run_timestamp"))
    return f"{_text(row.get('symbol')).upper()}:{_scan_date(row)}:{suffix}".rstrip(":")


def _read_rows(path: Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def session_representatives(features_path: Path) -> dict[int, tuple[str, str, str]]:
    """``{input row index: (symbol, side, session)}`` for each session's LAST valid scan row."""
    return representatives_of(_read_rows(features_path))


def representatives_of(rows: Iterable[Mapping[str, Any]]) -> dict[int, tuple[str, str, str]]:
    """The rule over rows in file order (shared with the live M5 stamp, `m5_setup_key_stamp`).

    The v2 build's choice: valid rows (symbol, date, close > 0), ordered by
    run_timestamp, run_id, input order; the last one per (symbol, side, date).
    """
    best: dict[tuple[str, str, str], tuple[tuple[str, str, int], int]] = {}
    for index, row in enumerate(rows):
        symbol = _text(row.get("symbol")).upper()
        session = _scan_date(row)
        close = _number(row.get("last_close"))
        side = _side(row.get("side")) or "LONG"
        if not symbol or not session or close is None or close <= 0:
            continue
        order = (_text(row.get("run_timestamp")), _text(row.get("run_id")), index)
        key = (symbol, side, session)
        current = best.get(key)
        if current is None or order >= current[0]:
            best[key] = (order, index)
    return {index: key for key, (_order, index) in best.items()}


# --- ctx lookups over whole copied stores


@dataclass
class ContextStores:
    reports_dir: Path | None = None
    review_events: Path | None = None
    m5_outcomes: Path | None = None
    environment: Path | None = None
    _triggers: dict[str, dict] | None = field(default=None, repr=False)
    _m5: dict[str, dict] | None = field(default=None, repr=False)
    _labels: dict[str, str] | None = field(default=None, repr=False)
    _trigger_times: dict[str, dict] | None = field(default=None, repr=False)
    _triggers_start: str | None = field(default=None, repr=False)
    _m5_start: str | None = field(default=None, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def paths(self) -> list[Path | None]:
        return [self.reports_dir, self.review_events, self.m5_outcomes, self.environment]

    def _index(self) -> None:
        import setup_permutation_context as spc

        if self.review_events is not None and self._triggers is None:
            events = []
            source = Path(self.review_events)
            files = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
            for item in files:
                with item.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            parsed = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(parsed, dict) and parsed.get("action") == "watch_fired":
                            events.append(parsed)
            by_session: dict[str, list] = {}
            for event in events:
                by_session.setdefault(_text(event.get("trade_date"))[:10], []).append(event)
            self._triggers = {day: spc.entry_triggers(rows, day) for day, rows in by_session.items()}
            self._trigger_times = {
                day: spc.entry_trigger_checkpoints(rows, day) for day, rows in by_session.items()
            }
            self._triggers_start = spc.watch_fired_coverage_start(events)
        if self.m5_outcomes is not None and self._m5 is None:
            by_session = {}
            first_day = None
            for row in _read_rows(Path(self.m5_outcomes)):
                day = _text(row.get("trade_date"))[:10]
                if day and (first_day is None or day < first_day):
                    first_day = day
                if _text(row.get("event_type")) == "registered":
                    by_session.setdefault(day, []).append(row)
            self._m5_start = first_day
            self._m5 = {day: spc.m5_bounce_types(rows, day) for day, rows in by_session.items()}
        if self.environment is not None and self._labels is None:
            import d1_environment_store

            self._labels = d1_environment_store.labels_by_session(path=self.environment)

    def session_context(self, session: str):
        import setup_permutation_context as spc

        if session in self._cache:
            return self._cache[session]
        self._index()
        slots = (
            spc.load_discovery_slots(session, reports_dir=self.reports_dir)
            if self.reports_dir is not None
            else None
        )
        # Before a source's first logged session its silence is unknown, never "none".
        triggers_known = self._triggers is not None and spc.covered(session, self._triggers_start)
        m5_known = self._m5 is not None and spc.covered(session, self._m5_start)
        context = spc.SessionContext(
            slots=slots,
            triggers=(self._triggers or {}).get(session, {}) if triggers_known else None,
            m5=(self._m5 or {}).get(session, {}) if m5_known else None,
            environment=(self._labels or {}).get(session, sp.UNKNOWN) if self._labels is not None else sp.UNKNOWN,
            trigger_times=(self._trigger_times or {}).get(session, {}) if triggers_known else None,
        )
        self._cache[session] = context
        return context


# --- pass 2: the facets of every representative row


@dataclass
class KeyedRow:
    symbol: str
    side: str
    session: str
    family: str
    scan_row_id: str
    last_close: float | None
    atr20: float | None
    run_id: str
    run_timestamp: str
    last_trade_date: str
    priority_bucket: str
    facets: dict[str, str]
    rule_version: str


def key_scan_row(row: Mapping[str, Any], symbol: str, side: str, session: str, context: Any) -> "sp.PermutationKey":
    """One representative scan row's key; ``context`` is a `SessionContext` for ``session``."""
    view = dict(row)
    view["side"] = side
    view.setdefault("run_date", session)
    if not _text(view.get("run_date")):
        view["run_date"] = session
    stamped = bool(_text(row.get("permutation_rule_version")))
    return sp.facets_for_row(sp.scan_row_view(view, has_ma_columns=stamped), context.ctx_for(symbol, side))


def key_representatives(
    features_path: Path, representatives: Mapping[int, tuple[str, str, str]], stores: ContextStores
) -> dict[tuple[str, str, str], KeyedRow]:
    keyed: dict[tuple[str, str, str], KeyedRow] = {}
    for index, row in enumerate(_read_rows(features_path)):
        identity = representatives.get(index)
        if identity is None:
            continue
        symbol, side, session = identity
        key = key_scan_row(row, symbol, side, session, stores.session_context(session))
        keyed[identity] = KeyedRow(
            symbol=symbol,
            side=side,
            session=session,
            family=key.family,
            scan_row_id=scan_row_id(row),
            last_close=_number(row.get("last_close")),
            atr20=_number(row.get("atr20")),
            run_id=_text(row.get("run_id")),
            run_timestamp=_text(row.get("run_timestamp")),
            last_trade_date=_text(row.get("last_trade_date")),
            priority_bucket=_text(row.get("priority_bucket")),
            facets=key.as_dict(),
            rule_version=key.permutation_rule_version,
        )
    return keyed


# --- outcomes


def _swing_row(keyed: KeyedRow, horizon: int, favorable: Any, side_return_pct: Any, entry_close: Any) -> dict:
    ret = _number(side_return_pct)
    close = _number(entry_close) or keyed.last_close
    r = None
    if ret is not None and close and keyed.atr20 and keyed.atr20 > 0:
        r = ret / (keyed.atr20 / close * 100.0)
    return {
        "population": POPULATION_SWING,
        "episode_id": keyed.scan_row_id,
        "symbol": keyed.symbol,
        "side": keyed.side,
        "family": keyed.family,
        "session": keyed.session,
        "horizon": int(horizon),
        "win": _truthy(favorable),
        "r": r,
        "r_unit": "atr20",
        "outcome_kind": "favorable_direction_session_v2",
        "permutation_rule_version": keyed.rule_version,
        "backfill_version": BACKFILL_VERSION,
        **{facet_column(name): value for name, value in keyed.facets.items()},
    }


def swing_rows_from_horizons(horizons_path: Path, keyed: Mapping[tuple[str, str, str], KeyedRow]) -> list[dict]:
    by_id = {row.scan_row_id: row for row in keyed.values()}
    out = []
    for row in _read_rows(horizons_path):
        horizon = int(_number(row.get("horizon_sessions")) or 0)
        if horizon not in SWING_HORIZONS or not _truthy(row.get("measured")):
            continue
        match = by_id.get(_text(row.get("scan_row_id")))
        if match is None:
            continue
        out.append(_swing_row(match, horizon, row.get("favorable"), row.get("side_return_pct"), row.get("entry_close")))
    return out


def swing_rows_from_daily_bars(
    bars_dir: Path, keyed: Mapping[tuple[str, str, str], KeyedRow], *, last_completed: date
) -> list[dict]:
    import pandas as pd

    from master_avwap_lib.session_horizon_outcomes import build_session_horizon_observation_rows

    history = pd.DataFrame([
        {"symbol": row.symbol, "side": row.side, "last_trade_date": row.last_trade_date or row.session,
         "run_date": row.session, "run_id": row.run_id, "run_timestamp": row.run_timestamp,
         "last_close": row.last_close, "setup_family": row.family, "priority_bucket": row.priority_bucket}
        for row in keyed.values()
    ])
    cache: dict[str, dict | None] = {}

    def closes_for(symbol: str):
        if symbol not in cache:
            path = Path(bars_dir) / f"{symbol}.csv"
            closes = None
            if path.is_file():
                closes = {}
                for bar in _read_rows(path):
                    close = _number(bar.get("close"))
                    try:
                        day = date.fromisoformat(_text(bar.get("datetime"))[:10])
                    except ValueError:
                        continue
                    if close and close > 0:
                        closes[day] = close
            cache[symbol] = closes
        return cache[symbol]

    build = build_session_horizon_observation_rows(
        history, closes_for, horizons=SWING_HORIZONS, last_completed_session=last_completed, window_sessions=None,
    )
    by_id = {row.scan_row_id: row for row in keyed.values()}
    out = []
    for row in build.rows:
        match = by_id.get(_text(row.get("scan_row_id")))
        if match is None or not row.get("measured"):
            continue
        out.append(_swing_row(match, row["horizon_sessions"], row.get("favorable"), row.get("side_return_pct"),
                              row.get("entry_close")))
    return out


def previous_session_text(session: str) -> str:
    """The session before ``session`` ("" outside the calendar): the D1 picture an M5 alert had."""
    import market_calendar

    try:
        return market_calendar.previous_session(date.fromisoformat(_text(session)[:10])).isoformat()
    except Exception:  # noqa: BLE001 - outside the calendar: no D1 picture
        return ""


def live_stamp_facets(record: Mapping[str, Any] | None) -> dict[str, str] | None:
    """The facets an M5 sidecar record stamped at alert time, or None when it holds no usable key."""
    import m5_setup_key_stamp

    if not isinstance(record, Mapping) or record.get("status") != m5_setup_key_stamp.STATUS_STAMPED:
        return None
    if record.get("permutation_rule_version") != sp.PERMUTATION_RULE_VERSION:
        return None
    facets = record.get("facets")
    if not isinstance(facets, Mapping):
        return None
    return {name: _text(facets.get(name)) or sp.UNKNOWN for name in sp.FACETS}


def m5_rows(
    m5_path: Path,
    keyed: Mapping[tuple[str, str, str], KeyedRow],
    *,
    as_of: date,
    stamps: Mapping[str, Mapping[str, Any]] | None = None,
    counts: dict[str, int] | None = None,
) -> list[dict]:
    """One row per measured M5 episode. ``stamps`` (the live sidecar, by event_id) wins when usable."""
    import held_run_score

    unknown = {name: sp.UNKNOWN for name in sp.FACETS}
    tally = counts if counts is not None else {}
    tally.setdefault("m5_live_stamped", 0)
    tally.setdefault("m5_live_stamp_disagreed", 0)
    out = []
    previous: dict[str, str] = {}
    for episode in held_run_score.build_episodes(_read_rows(m5_path), as_of=as_of):
        if not episode.measured:
            continue
        side = _side(episode.direction)
        session = _text(episode.trade_date)[:10]
        if not side or not session:
            continue
        if session not in previous:
            previous[session] = previous_session_text(session)
        d1 = keyed.get((episode.symbol, side, previous[session]))
        facets = d1.facets if d1 else unknown
        live = live_stamp_facets((stamps or {}).get(episode.event_id))
        if live is not None:
            tally["m5_live_stamped"] += 1
            tally["m5_live_stamp_disagreed"] += int(live != facets)
            facets = live
        out.append({
            "population": POPULATION_M5,
            "episode_id": episode.event_id,
            "symbol": episode.symbol,
            "side": side,
            "family": episode.bounce_type or sp.UNKNOWN,
            "session": session,
            "horizon": M5_HORIZON,
            "win": bool(episode.held),
            "r": episode.mfe_r,
            "r_unit": "mfe_r",
            "outcome_kind": held_run_score.HELD_RUN_OUTCOME_KIND,
            "permutation_rule_version": sp.PERMUTATION_RULE_VERSION,
            "backfill_version": BACKFILL_VERSION,
            **{facet_column(name): value for name, value in facets.items()},
        })
    return out


# --- the whole backfill


@dataclass
class BackfillResult:
    rows: list[dict]
    counts: dict[str, int]


def build_permutation_outcomes(
    features: Path,
    *,
    horizons: Path | None = None,
    daily_bars: Path | None = None,
    m5_outcomes: Path | None = None,
    stores: ContextStores | None = None,
    last_completed: date | None = None,
    m5_stamps: Path | None = None,
) -> BackfillResult:
    """Every population row. Refuses live paths before it opens anything."""
    stores = stores or ContextStores()
    refuse_live([features, horizons, daily_bars, m5_outcomes, m5_stamps, *stores.paths()])
    if horizons is None and daily_bars is None:
        raise ValueError("give --horizons or --daily-bars for the swing outcomes")
    representatives = session_representatives(Path(features))
    keyed = key_representatives(Path(features), representatives, stores)
    if horizons is not None:
        swing = swing_rows_from_horizons(Path(horizons), keyed)
    else:
        import market_calendar

        finished = last_completed or market_calendar.last_completed_session(datetime.now(market_calendar.MARKET_TZ))
        swing = swing_rows_from_daily_bars(Path(daily_bars), keyed, last_completed=finished)
    stamp_counts: dict[str, int] = {}
    m5 = []
    if m5_outcomes:
        import m5_setup_key_stamp

        stamps = m5_setup_key_stamp.read_stamps(Path(m5_stamps)) if m5_stamps else None
        m5 = m5_rows(Path(m5_outcomes), keyed, as_of=last_completed or date.today(), stamps=stamps,
                     counts=stamp_counts)
    return BackfillResult(
        rows=[*swing, *m5],
        counts={"scan_rows_keyed": len(keyed), "swing_rows": len(swing), "m5_rows": len(m5), **stamp_counts},
    )


def write_parquet(rows: list[dict], out: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    refuse_live([out])
    columns = output_columns()
    schema = pa.schema([
        pa.field(name, pa.int32() if name == "horizon" else pa.bool_() if name == "win"
                 else pa.float64() if name == "r" else pa.string())
        for name in columns
    ])
    table = pa.Table.from_pylist([{name: row.get(name) for name in columns} for row in rows], schema=schema)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    pq.write_table(table, temp)
    os.replace(temp, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--horizons", type=Path)
    parser.add_argument("--daily-bars", type=Path)
    parser.add_argument("--m5-outcomes", type=Path)
    parser.add_argument("--m5-stamps", type=Path, help="copy of m5_setup_key_stamps.jsonl (joined on event_id)")
    parser.add_argument("--review-events", type=Path)
    parser.add_argument("--scan-reports", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--last-completed", type=date.fromisoformat)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    stores = ContextStores(reports_dir=args.scan_reports, review_events=args.review_events,
                           m5_outcomes=args.m5_outcomes, environment=args.environment)
    everything = [args.scratch, args.features, args.horizons, args.daily_bars, args.out, args.m5_stamps,
                  *stores.paths()]
    roots = live_roots()  # taken before LOCALAPPDATA is pointed at scratch
    try:
        refuse_live(everything, roots)
    except LiveStoreRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    # Scratch roots before any project_paths import, and abort if one still resolves live.
    os.environ["TRADINGBOTV3_DATA_DIR"] = str(Path(args.scratch) / "data")
    os.environ["LOCALAPPDATA"] = str(Path(args.scratch) / "localappdata")
    import project_paths

    try:
        refuse_live([project_paths.DATA_DIR, project_paths.PERSISTENT_DATA_DIR, project_paths.LOCAL_SETTINGS_DIR],
                    roots)
    except LiveStoreRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = build_permutation_outcomes(
        args.features, horizons=args.horizons, daily_bars=args.daily_bars, m5_outcomes=args.m5_outcomes,
        stores=stores, last_completed=args.last_completed, m5_stamps=args.m5_stamps,
    )
    write_parquet(result.rows, args.out)
    print(json.dumps({"out": str(args.out), **result.counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
