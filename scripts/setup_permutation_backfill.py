"""Setup permutation keys (WISHLIST P1-4 / 4b): the scratch-only backfill.

    python scripts/setup_permutation_backfill.py --scratch <dir> \
        --features <copy of d1_features_history.csv> \
        (--horizons <copy of master_avwap_session_horizon_outcomes.csv> | --daily-bars <copy of daily_bars dir>) \
        [--spy-bars <copy of SPY daily bars, csv or parquet>] \
        [--m5-outcomes <copy of intraday_bounce_outcomes.csv>] [--m5-candidates <copy of intraday_bounce_candidates.csv>] \
        [--m5-stamps <copy>] [--review-events <copy>] \
        [--scan-reports <copy dir>] [--environment <copy of the d1 environment store>] \
        --out <scratch>/permutation_outcomes.parquet

Writes ONE long table, one row per episode x horizon, with every facet as it
was on the scan date (``f_<facet>`` columns):

- **swing**: the session's last scan row per (symbol, side, scan date) - the
  `session_horizon_outcomes` v2 grain, joined on the same ``scan_row_id`` - at
  1/3/5/10 exchange sessions. ``win`` = the side return beat SPY's side
  return over the same sessions (`setup_grades.tape_result`, S4); a row whose
  tape is unknown (no SPY close, immature, unmeasured) is left out, never a
  loss. ``r`` = the side return in ATR20 units of the scan row (close to
  close, so the v2 gap-aware execution convention and the literal one book the
  same number). With ``--daily-bars`` the v2 build is re-run over the whole
  history (``window_sessions=None``) instead of reading the 30-session file;
  SPY comes from ``--spy-bars``, else ``<daily-bars>/SPY.csv``. S15 item 4:
  ``raw_win`` = the side return > 0 (not vs SPY) and ``regime_working`` /
  ``regime_working_rule`` = the entry day's `long_regime_working` verdict, from
  the row's live stamp, else recomputed (``--structural-regime`` copy of the
  journal first, then SPY above a rising 20-day through the session).
- **m5**: one row per MEASURED M5 episode (`held_run_score.build_episodes`),
  family = its bounce type, ``win`` = the level held 30 minutes, ``r`` = the
  episode's MFE_R (held_run_score's MFE rule). Its facets come from the
  PREVIOUS session's last scan row for the name and side - the D1 picture the
  trader had before the alert, never the same day's later scan. With
  ``--m5-stamps`` the live alert-time stamp (`m5_setup_key_stamp`, same rule)
  is joined on event_id and wins; no usable stamp record = the recompute.
  The M5-native facets (P11, ``f_m5_*``) join the same way: the stamp's M5
  part wins, else the registered row's entry time, RVOL and bounce type (the
  VWAP distance and SPY state need the live stamp and are unknown otherwise).
  Horizon name ``held30``.
- **m5, horizon ``bracket_1r``** (S3, with ``--m5-candidates``): one row per
  DECIDED alert - every ``confirmed`` event id in the candidates log joined to
  the outcome log - ``win`` = +1R before -1R on the first decisive row
  (`setup_grades.bracket_results`), ``r`` = the final row's close R (None
  while unsettled). Facets are joined exactly as for ``held30``; an alert with
  no D1 scan row keeps its row with unknown D1 facets.

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
#: Both M5 horizons are same-session (horizon 0, no embargo); ``horizon_name`` tells them apart.
M5_HORIZON = 0
HORIZON_HELD30 = "held30"
HORIZON_BRACKET_1R = "bracket_1r"
BRACKET_OUTCOME_KIND = "bracket_1r_first_decisive"
SWING_OUTCOME_KIND = "tape_relative_session_v2"
#: Rows per pandas chunk when reading the multi-GB M5 logs.
CSV_CHUNK_ROWS = 250_000
BACKFILL_VERSION = "setup_permutation_backfill.v1"

BASE_COLUMNS = (
    "population", "episode_id", "symbol", "side", "family", "session", "horizon", "horizon_name",
    "win", "r", "r_unit", "outcome_kind", "permutation_rule_version", "backfill_version",
    # S15 item 4: the raw side return won (> 0, not vs SPY) and the entry day's long regime verdict.
    "raw_win", "regime_working", "regime_working_rule",
)


def facet_column(name: str) -> str:
    return f"f_{name}"


def output_columns() -> list[str]:
    """D1 facets then the M5-native facets (P11): the m5 population is searched on the union."""
    return [*BASE_COLUMNS, *(facet_column(name) for name in (*sp.FACETS, *sp.M5_FACETS))]


def swing_horizon_name(horizon: int) -> str:
    return f"{int(horizon)}_sessions"


def _unknown_m5() -> dict[str, str]:
    return {facet_column(name): sp.UNKNOWN for name in sp.M5_FACETS}


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
    #: `setup_permutations.long_regime_working` on the entry day: the scan row's stamp, else recomputed.
    regime_working: str = sp.UNKNOWN
    regime_rule: str = sp.UNKNOWN


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
            **_stamped_regime(row),
        )
    return keyed


# --- S15 item 4: the entry day's long regime ("working" or not)

_WORKING, _WORKING_RULE = "perm_regime_working", "perm_regime_working_rule"


def _stamped_regime(row: Mapping[str, Any]) -> dict[str, str]:
    """The verdict the live scan stamped on the row (S15 item 2); unknown on older rows."""
    verdict = _text(row.get(_WORKING)).lower()
    if verdict not in ("yes", "no"):
        return {}
    return {"regime_working": verdict, "regime_rule": _text(row.get(_WORKING_RULE)) or sp.UNKNOWN}


def label_entry_regimes(
    keyed: Mapping[tuple[str, str, str], KeyedRow],
    spy_closes: Mapping[str, float],
    trader_rows: list[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Fill the regime verdict of every row the live scan did not stamp; returns counts by rule.

    Same definition as the stamp (`setup_permutations.long_regime_working`): the trader's
    structural regime on the session (``trader_rows`` = a copy of the journal's
    `structural_regime` rows), else SPY above a rising 20-day from SPY closes through the
    session (the entry is the session's close, so its bar is complete).
    """
    import structural_regime

    days = sorted(spy_closes or {})
    counts: dict[str, int] = {}
    for row in keyed.values():
        if row.regime_working == sp.UNKNOWN:
            segment = structural_regime.regime_at(trader_rows, row.session) if trader_rows else None
            upto = [day for day in days if day <= row.session]
            trend = (None, None)
            if upto and upto[-1] == row.session:
                trend = sp.spy_trend([spy_closes[day] for day in upto[-30:]])
            row.regime_working, row.regime_rule = sp.long_regime_working(
                (segment or {}).get("regime"), *trend)
        counts[row.regime_rule] = counts.get(row.regime_rule, 0) + 1
    return counts


# --- outcomes


def _swing_row(keyed: KeyedRow, horizon: int, win: bool, side_return_pct: Any, entry_close: Any) -> dict:
    ret = _number(side_return_pct)
    close = _number(entry_close) or keyed.last_close
    raw_win = None if ret is None else ret > 0
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
        "horizon_name": swing_horizon_name(horizon),
        "win": bool(win),
        "r": r,
        "r_unit": "atr20",
        "outcome_kind": SWING_OUTCOME_KIND,
        "permutation_rule_version": keyed.rule_version,
        "backfill_version": BACKFILL_VERSION,
        "raw_win": raw_win,
        "regime_working": keyed.regime_working,
        "regime_working_rule": keyed.regime_rule,
        **{facet_column(name): value for name, value in keyed.facets.items()},
        **_unknown_m5(),
    }


def read_spy_closes(path: Path | None) -> dict[str, float]:
    """``{iso date: close}`` from a copied SPY daily-bar file (csv or parquet); {} when absent."""
    if path is None or not Path(path).is_file():
        return {}
    import pandas as pd

    frame = pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)
    column = next((c for c in ("datetime", "date") if c in frame.columns), None)
    if column is None or "close" not in frame.columns:
        return {}
    days = pd.to_datetime(frame[column].astype(str).str[:10], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    return {day.date().isoformat(): float(close) for day, close in zip(days, closes, strict=False)
            if not pd.isna(day) and not pd.isna(close) and close > 0}


def _tape_win(keyed: KeyedRow, horizon_row: Mapping[str, Any], spy_closes: Mapping[str, float],
              tally: dict[str, int]) -> bool | None:
    """True / False = the side return beat SPY's side return; None = unknown (the row is left out)."""
    import setup_grades

    row = dict(horizon_row)
    row["measured"] = "true" if _truthy(horizon_row.get("measured")) else "false"
    outcome = setup_grades.tape_result({"side": keyed.side}, row, spy_closes)
    if outcome == setup_grades.UNKNOWN:
        tally["swing_tape_unknown"] = tally.get("swing_tape_unknown", 0) + 1
        return None
    return outcome == setup_grades.WIN


def swing_rows_from_horizons(
    horizons_path: Path,
    keyed: Mapping[tuple[str, str, str], KeyedRow],
    spy_closes: Mapping[str, float] | None = None,
    counts: dict[str, int] | None = None,
) -> list[dict]:
    by_id = {row.scan_row_id: row for row in keyed.values()}
    tally = counts if counts is not None else {}
    out = []
    for row in _read_rows(horizons_path):
        horizon = int(_number(row.get("horizon_sessions")) or 0)
        if horizon not in SWING_HORIZONS or not _truthy(row.get("measured")):
            continue
        match = by_id.get(_text(row.get("scan_row_id")))
        if match is None:
            continue
        win = _tape_win(match, row, spy_closes or {}, tally)
        if win is None:
            continue
        out.append(_swing_row(match, horizon, win, row.get("side_return_pct"), row.get("entry_close")))
    return out


def swing_rows_from_daily_bars(
    bars_dir: Path,
    keyed: Mapping[tuple[str, str, str], KeyedRow],
    *,
    last_completed: date,
    spy_closes: Mapping[str, float] | None = None,
    counts: dict[str, int] | None = None,
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
    tally = counts if counts is not None else {}
    out = []
    for row in build.rows:
        match = by_id.get(_text(row.get("scan_row_id")))
        if match is None or not row.get("measured"):
            continue
        win = _tape_win(match, row, spy_closes or {}, tally)
        if win is None:
            continue
        out.append(_swing_row(match, row["horizon_sessions"], win, row.get("side_return_pct"),
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


def live_m5_facets(record: Mapping[str, Any] | None) -> dict[str, str] | None:
    """The M5-native facets a sidecar record stamped at alert time (P11), or None when it holds none."""
    if not isinstance(record, Mapping) or record.get("m5_rule_version") != sp.M5_PERMUTATION_RULE_VERSION:
        return None
    facets = record.get("m5_facets")
    if not isinstance(facets, Mapping):
        return None
    return {name: _text(facets.get(name)) or sp.UNKNOWN for name in sp.M5_FACETS}


def _registered_inputs(rows: Iterable[Mapping[str, Any]], sink: dict[str, dict], tz: Any) -> Iterable:
    """Pass rows through, keeping the M5 inputs of each event's registered (alert-time) row."""
    import m5_setup_key_stamp

    for row in rows:
        if _text(row.get("event_type")) == "registered":
            event_id = _text(row.get("event_id"))
            if event_id and event_id not in sink:
                sink[event_id] = m5_setup_key_stamp.alert_inputs(row, tz)
        yield row


def _m5_event_facets(
    event_id: str,
    symbol: str,
    side: str,
    session: str,
    *,
    keyed: Mapping[tuple[str, str, str], KeyedRow],
    stamps: Mapping[str, Mapping[str, Any]] | None,
    registered: Mapping[str, dict],
    previous: dict[str, str],
    tally: dict[str, int],
) -> tuple[dict[str, str], dict[str, str]]:
    """(D1 facets, M5 facets) for one alert: the live stamp when usable, else the recompute."""
    if session not in previous:
        previous[session] = previous_session_text(session)
    d1 = keyed.get((symbol, side, previous[session]))
    facets = d1.facets if d1 else {name: sp.UNKNOWN for name in sp.FACETS}
    record = (stamps or {}).get(event_id)
    live = live_stamp_facets(record)
    if live is not None:
        tally["m5_live_stamped"] = tally.get("m5_live_stamped", 0) + 1
        tally["m5_live_stamp_disagreed"] = tally.get("m5_live_stamp_disagreed", 0) + int(live != facets)
        facets = live
    m5_facets = live_m5_facets(record)
    if m5_facets is not None:
        tally["m5_live_m5_stamped"] = tally.get("m5_live_m5_stamped", 0) + 1
    else:
        m5_facets = sp.m5_facets_for(registered.get(event_id), side).as_dict()
    return facets, m5_facets


def confirmed_event_ids(candidates_path: Path) -> set[str]:
    """Every event id the candidates log ever marked ``confirmed`` (pandas chunks, two columns)."""
    import pandas as pd

    out: set[str] = set()
    for chunk in pd.read_csv(candidates_path, usecols=["event_id", "event_type"], chunksize=CSV_CHUNK_ROWS,
                             dtype=str, keep_default_na=False):
        confirmed = chunk.loc[chunk["event_type"].str.strip().str.lower() == "confirmed", "event_id"]
        out.update(text.strip() for text in confirmed if text.strip())
    return out


#: The outcome-log columns the bracket and the M5 facet recompute read.
BRACKET_COLUMNS = frozenset({
    "event_id", "event_type", "logged_at", "trade_date", "symbol", "direction", "entry_time", "entry_price",
    "risk_per_share", "bars_elapsed", "close_r", "target_1r_hit", "target_2r_hit", "stop_hit", "eod_close",
    "context_json",
})
_FLAG_TEXT = ("true", "1", "1.0", "yes")


def _decided_outcome_rows(outcomes_path: Path, keep: set[str]) -> Iterable[dict]:
    """Outcome-log rows of ``keep`` events that can matter to the bracket: registered, final or flagged.

    An unflagged update row never decides (the flags are cumulative), so it is
    dropped in the chunk; context_json is kept only on registered rows.
    """
    import pandas as pd

    for chunk in pd.read_csv(outcomes_path, usecols=lambda column: column in BRACKET_COLUMNS,
                             chunksize=CSV_CHUNK_ROWS, dtype=str, keep_default_na=False):
        chunk = chunk[chunk["event_id"].str.strip().isin(keep)]
        if chunk.empty:
            continue
        kind = chunk["event_type"].str.strip().str.lower()
        flagged = False
        for column in ("stop_hit", "target_1r_hit", "target_2r_hit"):
            if column in chunk.columns:
                flagged = flagged | chunk[column].str.strip().str.lower().isin(_FLAG_TEXT)
        chunk = chunk[kind.isin(("registered", "final")) | flagged]
        if "context_json" in chunk.columns:
            chunk = chunk.assign(context_json=chunk["context_json"].where(kind.loc[chunk.index] == "registered", ""))
        yield from chunk.to_dict("records")


def m5_bracket_rows(
    outcomes_path: Path,
    candidates_path: Path,
    keyed: Mapping[tuple[str, str, str], KeyedRow],
    *,
    stamps: Mapping[str, Mapping[str, Any]] | None = None,
    counts: dict[str, int] | None = None,
) -> list[dict]:
    """S3: one ``bracket_1r`` row per decided confirmed alert, keyed or not."""
    import m5_setup_key_stamp
    import setup_grades

    tally = counts if counts is not None else {}
    confirmed = confirmed_event_ids(Path(candidates_path))
    registered: dict[str, dict] = {}
    symbols: dict[str, str] = {}
    tz = m5_setup_key_stamp._local_tz()
    rows = []
    for row in _registered_inputs(_decided_outcome_rows(Path(outcomes_path), confirmed), registered, tz):
        event_id = _text(row.get("event_id"))
        symbols.setdefault(event_id, _text(row.get("symbol")).upper())
        rows.append(row)
    results = setup_grades.bracket_results(rows)
    ignored: dict[str, int] = {}  # the held30 pass already counts the live stamps
    previous: dict[str, str] = {}
    out = []
    for result in results:
        if result["result"] not in (setup_grades.WIN, setup_grades.LOSS):
            tally["m5_bracket_undecided"] = tally.get("m5_bracket_undecided", 0) + 1
            continue
        event_id = result["event_id"]
        side, session, symbol = _side(result["side"]), _text(result["trade_date"])[:10], symbols.get(event_id, "")
        if not side or not session or not symbol:
            continue
        facets, m5_facets = _m5_event_facets(
            event_id, symbol, side, session, keyed=keyed, stamps=stamps, registered=registered,
            previous=previous, tally=ignored,
        )
        out.append({
            "population": POPULATION_M5,
            "episode_id": event_id,
            "symbol": symbol,
            "side": side,
            "family": result["bounce_type"] or sp.UNKNOWN,
            "session": session,
            "horizon": M5_HORIZON,
            "horizon_name": HORIZON_BRACKET_1R,
            "win": result["result"] == setup_grades.WIN,
            "r": result["eod_r"],
            "r_unit": "close_r",
            "outcome_kind": BRACKET_OUTCOME_KIND,
            "permutation_rule_version": sp.PERMUTATION_RULE_VERSION,
            "backfill_version": BACKFILL_VERSION,
            **{facet_column(name): value for name, value in facets.items()},
            **{facet_column(name): value for name, value in m5_facets.items()},
        })
    tally["m5_confirmed_events"] = len(confirmed)
    return out


def m5_rows(
    m5_path: Path,
    keyed: Mapping[tuple[str, str, str], KeyedRow],
    *,
    as_of: date,
    stamps: Mapping[str, Mapping[str, Any]] | None = None,
    counts: dict[str, int] | None = None,
) -> list[dict]:
    """One row per measured M5 episode. ``stamps`` (the live sidecar, by event_id) wins when usable.

    D1 facets and M5-native facets are joined separately: each comes from the
    live stamp when it holds one, else from the recompute (the previous
    session's scan row; the registered row's entry time, RVOL and bounce type).
    """
    import held_run_score
    import m5_setup_key_stamp

    tally = counts if counts is not None else {}
    tally.setdefault("m5_live_stamped", 0)
    tally.setdefault("m5_live_stamp_disagreed", 0)
    tally.setdefault("m5_live_m5_stamped", 0)
    out = []
    previous: dict[str, str] = {}
    registered: dict[str, dict] = {}
    tz = m5_setup_key_stamp._local_tz()
    rows = _registered_inputs(_read_rows(m5_path), registered, tz)
    for episode in held_run_score.build_episodes(rows, as_of=as_of):
        if not episode.measured:
            continue
        side = _side(episode.direction)
        session = _text(episode.trade_date)[:10]
        if not side or not session:
            continue
        facets, m5_facets = _m5_event_facets(
            episode.event_id, episode.symbol, side, session, keyed=keyed, stamps=stamps,
            registered=registered, previous=previous, tally=tally,
        )
        out.append({
            "population": POPULATION_M5,
            "episode_id": episode.event_id,
            "symbol": episode.symbol,
            "side": side,
            "family": episode.bounce_type or sp.UNKNOWN,
            "session": session,
            "horizon": M5_HORIZON,
            "horizon_name": HORIZON_HELD30,
            "win": bool(episode.held),
            "r": episode.mfe_r,
            "r_unit": "mfe_r",
            "outcome_kind": held_run_score.HELD_RUN_OUTCOME_KIND,
            "permutation_rule_version": sp.PERMUTATION_RULE_VERSION,
            "backfill_version": BACKFILL_VERSION,
            **{facet_column(name): value for name, value in facets.items()},
            **{facet_column(name): value for name, value in m5_facets.items()},
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
    m5_candidates: Path | None = None,
    spy_bars: Path | None = None,
    spy_closes: Mapping[str, float] | None = None,
    structural_regime: Path | None = None,
) -> BackfillResult:
    """Every population row. Refuses live paths before it opens anything.

    SPY closes for the tape-relative swing win: ``spy_closes``, else ``spy_bars``,
    else ``<daily_bars>/SPY.csv``. With none, every swing row's tape is unknown
    and none is written. ``structural_regime`` is a copy of the trade journal: the
    trader's regime for rows the live scan did not stamp (S15 item 4).
    """
    stores = stores or ContextStores()
    refuse_live([features, horizons, daily_bars, m5_outcomes, m5_stamps, m5_candidates, spy_bars,
                 structural_regime, *stores.paths()])
    if horizons is None and daily_bars is None:
        raise ValueError("give --horizons or --daily-bars for the swing outcomes")
    representatives = session_representatives(Path(features))
    keyed = key_representatives(Path(features), representatives, stores)
    if spy_closes is None:
        spy_source = spy_bars or (Path(daily_bars) / "SPY.csv" if daily_bars is not None else None)
        spy_closes = read_spy_closes(spy_source)
    trader_rows = None
    if structural_regime is not None:
        import setup_permutation_context as spc

        trader_rows = spc.load_structural_regime_rows(path=Path(structural_regime))
    regime_counts = label_entry_regimes(keyed, spy_closes, trader_rows)
    swing_counts: dict[str, int] = {"swing_tape_unknown": 0}
    if horizons is not None:
        swing = swing_rows_from_horizons(Path(horizons), keyed, spy_closes, swing_counts)
    else:
        import market_calendar

        finished = last_completed or market_calendar.last_completed_session(datetime.now(market_calendar.MARKET_TZ))
        swing = swing_rows_from_daily_bars(Path(daily_bars), keyed, last_completed=finished, spy_closes=spy_closes,
                                           counts=swing_counts)
    stamp_counts: dict[str, int] = {}
    m5: list[dict] = []
    bracket: list[dict] = []
    if m5_outcomes:
        import m5_setup_key_stamp

        stamps = m5_setup_key_stamp.read_stamps(Path(m5_stamps)) if m5_stamps else None
        m5 = m5_rows(Path(m5_outcomes), keyed, as_of=last_completed or date.today(), stamps=stamps,
                     counts=stamp_counts)
        if m5_candidates:
            bracket = m5_bracket_rows(Path(m5_outcomes), Path(m5_candidates), keyed, stamps=stamps,
                                      counts=stamp_counts)
    return BackfillResult(
        rows=[*swing, *m5, *bracket],
        counts={"scan_rows_keyed": len(keyed), "swing_rows": len(swing), "spy_closes": len(spy_closes),
                **swing_counts, "m5_rows": len(m5), "m5_bracket_rows": len(bracket), **stamp_counts,
                **{f"regime_rule_{rule}": count for rule, count in sorted(regime_counts.items())}},
    )


def write_parquet(rows: list[dict], out: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    refuse_live([out])
    columns = output_columns()
    schema = pa.schema([
        pa.field(name, pa.int32() if name == "horizon" else pa.bool_() if name in ("win", "raw_win")
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
    parser.add_argument("--m5-candidates", type=Path,
                        help="copy of intraday_bounce_candidates.csv: adds the bracket_1r horizon (S3)")
    parser.add_argument("--spy-bars", type=Path, help="copy of SPY daily bars (csv or parquet) for the tape win")
    parser.add_argument("--structural-regime", type=Path,
                        help="copy of trade_journal.sqlite3: the trader's regime for unstamped rows (S15 item 4)")
    parser.add_argument("--review-events", type=Path)
    parser.add_argument("--scan-reports", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--last-completed", type=date.fromisoformat)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    stores = ContextStores(reports_dir=args.scan_reports, review_events=args.review_events,
                           m5_outcomes=args.m5_outcomes, environment=args.environment)
    everything = [args.scratch, args.features, args.horizons, args.daily_bars, args.out, args.m5_stamps,
                  args.m5_candidates, args.spy_bars, args.structural_regime, *stores.paths()]
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
        m5_candidates=args.m5_candidates, spy_bars=args.spy_bars, structural_regime=args.structural_regime,
    )
    write_parquet(result.rows, args.out)
    print(json.dumps({"out": str(args.out), **result.counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
