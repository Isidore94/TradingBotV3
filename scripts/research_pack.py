"""Research pack: a read-only export of the desk's setup evidence for a frontier model.

START HERE (for an AI session on the desk PC)
---------------------------------------------
1. See what exists and what is missing:

       .venv\\Scripts\\python.exe scripts\\research_pack.py status

   Every source is listed with its location, whether it is reachable, row
   counts, date coverage and freshness. ``rows = unknown`` means the source
   could not be read; it never means zero.

2. Write a pack to a scratch folder (never a live store; the tool refuses):

       .venv\\Scripts\\python.exe scripts\\research_pack.py export --out %TEMP%\\research_pack

   Optional: ``--as-of 2026-09-01`` rebuilds the pack as it was known at the
   end of that day; ``--format csv``; ``--floor 30`` (episodes a cell needs
   before it is ranked); ``--d1-recipe <recipe_id>`` (headline swing recipe).

3. Read ``manifest.json`` first, then ``setup_summary.csv``, then the tables:

   * ``d1_occurrences``  one row per D1 Master AVWAP setup occurrence (lake
     ``setup_occurrence``, latest revision) with its traits, point-in-time
     daily features (``feat_*``, ``feat_basis`` says how they were known),
     market context by timeframe (``ctx_*``) and the headline recipe's outcome
     (``hl_*``: net R, R at 1/2/3/5/10/18 sessions, MFE/MAE).
   * ``d1_outcomes``     every recipe's outcome per occurrence (latest
     ``computed_at``), with ``matured`` evaluated at ``as_of``.
   * ``m5_alerts``       one row per M5 bounce alert (event_id) with the
     context known when it fired, grade (``tier``) and R at 1/3/6/12 bars and
     at the final (end-of-day hold) mark.
   * ``journal_trades``  the trader's real trades (no account numbers), each
     joined to the D1 occurrence and the M5 alert it most plausibly came
     from. ``d1_match``/``m5_match`` record the match quality; unmatched stays
     unmatched.
   * ``setup_summary.csv`` n, episodes, win rate, mean/median by family,
     side and trait. ``rank`` is empty for any cell under the floor: a thin
     cell is shown, never ranked. Everything is EXPLORATORY.
   * ``trading_plan.md`` the trader's own plan (the history snapshot in force
     at ``--as-of`` when given) and ``plan_challenges.json`` - the night's
     challenges to it for the 7 days ending that date, each with the trader's
     answer (accepted / rejected + reason) or open / expired.
   * ``market_regime_table.jsonl`` + ``regime_journal.json`` (manifest section
     ``market_regimes``): the full multi-timeframe regime table and the trader's
     own regime journal, plus ``setup_grades_by_regime.json`` when a stored
     file exists (otherwise the section says it is absent).

Point-in-time rules
-------------------
* A journal trade only joins a D1 setup whose trigger session closed BEFORE
  the trade's session, and an M5 alert logged before the trade opened.
* D1 features come from the latest daily snapshot on or before the trigger
  session. ``feat_basis`` is ``as_observed`` when that snapshot was computed
  by the trigger, ``reconstructed`` when it was computed later from the same
  bars, and ``missing`` when none exists.
* M5 context is the ``registered`` row's context, written when the alert fired.
* ``--as-of`` hides every revision, outcome, alert row and trade written
  after that moment.
* Ranks use the mean of R clipped to +/-10 (``rank_basis``); ``mean`` and
  ``median`` stay raw. One alert with a tiny stop can otherwise read +300R.

Safety
------
Every source is opened read-only (SQLite ``mode=ro``, Parquet through the lake
manifest, CSV/JSONL as plain reads). The output folder is refused when it sits
inside any live store: the shared home, ``%LOCALAPPDATA%\\TradingBotV3``, the
research lake, the AI store, ``C:\\TradingBotData``, ``\\\\MINI-PC\\Trading
Bot Data`` or the repo. No detector module is imported. Like every script, the
``project_paths`` import runs the desk's standard path housekeeping.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pyarrow as pa  # noqa: E402
import pyarrow.csv as pacsv  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from scripts import journal_exposure  # noqa: E402
from scripts.research_warehouse import exchange_calendar as xcal  # noqa: E402
from scripts.research_warehouse.outcomes import OUTCOME_DEFINITION_ID  # noqa: E402
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
ET = xcal.EXCHANGE_TZ
PACK_SCHEMA = "research_pack_v1"
EVIDENCE_TIER = "EXPLORATORY"
DEFAULT_D1_RECIPE = "m5close_current_anchor1_2r_v1"
DEFAULT_FLOOR = 30
DEFAULT_JOURNAL_FLOOR = 10
DEFAULT_D1_MAX_SESSIONS = 5
DEFAULT_M5_WINDOW_MINUTES = 120
TIER_MATCH_SECONDS = 600
#: R is clipped to this band for RANKING only; a tiny stop makes one alert worth +300R.
RANK_CLIP_R = 10.0

#: Live roots written by the desk; an output folder under any of them is refused.
HARD_PROTECTED_ROOTS = (Path(r"C:\TradingBotData"), Path(r"\\MINI-PC\Trading Bot Data"))

D1_FEATURE_COLUMNS = (
    "close",
    "atr14",
    "dist_sma50_atr",
    "dist_sma100_atr",
    "dist_sma200_atr",
    "spy_regime_state",
    "band1_rejection_strength",
    "favorite_zone_residence_bars",
    "first_dev_touch_order",
    "second_band_streak",
    "anchor_knowledge",
)
CHECKPOINTS = ("r_at_s1", "r_at_s2", "r_at_s3", "r_at_s5", "r_at_s10", "r_at_s18")
OUTCOME_COLUMNS = (
    "occurrence_id", "recipe_id", "outcome_definition_id", "entry_at", "entry_price", "stop_price",
    "stop_distance", "r_at_15m", "r_at_30m", "r_at_60m", "r_at_120m", "r_at_eod", *CHECKPOINTS,
    "mfe_r", "mae_r", "time_to_mfe_min", "first_hit", "gross_r", "net_r", "result_state",
    "path_kind", "maturity_at", "censor_reason", "computed_at", "input_capture_mode_worst",
)
CONTEXT_TIMEFRAMES = ("D1", "H4", "H1", "M30", "M5")
M5_MILESTONES = {"1_bar": "r_1bar", "3_bar": "r_3bar", "6_bar": "r_6bar", "12_bar": "r_12bar"}
M5_COLUMNS = (
    "event_id", "event_type", "logged_at", "trade_date", "symbol", "direction", "entry_time",
    "entry_price", "stop_price", "risk_per_share", "close_r", "mfe_r", "mae_r", "target_1r_hit",
    "target_2r_hit", "stop_hit", "status", "context_json", "outcome_mode", "eod_move_pct",
    "mfe_pct", "mae_pct",
)
_EVENT_ID = re.compile(r"^(?P<symbol>[^_]+)_(?P<dir>long|short)_(?P<day>\d{8})_(?P<hms>\d{2}_\d{2}_\d{2})_(?P<types>.+)$")
#: ``AAOI18JUN26P120.00`` - the other option spelling in the journal.
_ALT_OPTION = re.compile(r"^(?P<root>[A-Z][A-Z.]{0,5})\d{2}[A-Z]{3}\d{2}(?P<right>[CP])\d")


class LiveStoreWriteRefused(RuntimeError):
    """The requested output folder is inside a live store."""


@dataclass
class Sources:
    """Where each input lives. ``None`` or a missing path means unknown."""

    lake_root: Path | None
    journal_db: Path | None
    bounce_outcomes_csv: Path | None
    bounces_csv: Path | None
    #: name -> (path, date key) for status-only evidence files.
    extra_files: dict = field(default_factory=dict)
    ai_store_root: Path | None = None
    #: Extra roots to refuse writing under (tests; other machines).
    protected: tuple = ()
    #: P1-4 4d: the setup-permutation report, copied into the pack as-is (facts only).
    setup_keys_report: Path | None = None
    #: P1-7: the trading plan, its history and the week's challenges / answers.
    trading_plan: Path | None = None
    plan_history_dir: Path | None = None
    plan_challenges: Path | None = None
    plan_answers: Path | None = None
    #: S17.2: the multi-timeframe regime table and the setups-by-regime grades (S16.3).
    market_regime_table: Path | None = None
    regime_grades: Path | None = None


def _pp():
    from scripts import project_paths

    return project_paths


def resolve_sources() -> Sources:
    """Every input from ``project_paths`` and the configured lake / AI store."""
    paths = _pp()
    try:
        from scripts.research_warehouse import config as lake_config

        lake = lake_config.get_research_store_dir()
    except Exception:
        lake = None
    try:
        from scripts.ai_jobs import store as ai_store

        ai_root = ai_store.get_ai_store_dir()
    except Exception:
        ai_root = None
    return Sources(
        lake_root=lake,
        journal_db=Path(paths.JOURNAL_DB_FILE),
        bounce_outcomes_csv=Path(paths.INTRADAY_BOUNCE_OUTCOMES_FILE),
        bounces_csv=Path(paths.INTRADAY_BOUNCES_FILE),
        extra_files={
            "setup_points_log": (Path(paths.SETUP_POINTS_LOG_FILE), "scan_date"),
            "master_avwap_tier_outcomes": (Path(paths.MASTER_AVWAP_TIER_OUTCOMES_FILE), "scan_date"),
            "human_focus_outcomes": (Path(paths.HUMAN_FOCUS_OUTCOMES_FILE), "trade_date"),
            "claimed_picks": (Path(paths.CLAIMED_PICKS_FILE), "session_date"),
            "alert_review_events_legacy": (Path(paths.ALERT_REVIEW_EVENTS_FILE), "trade_date"),
            "alert_review_events": (Path(paths.ALERT_REVIEW_EVENTS_DIR), "trade_date"),
        },
        ai_store_root=ai_root,
        setup_keys_report=Path(paths.SETUP_PERMUTATION_REPORT_FILE),
        trading_plan=Path(paths.TRADING_PLAN_FILE),
        plan_history_dir=Path(paths.TRADING_PLAN_HISTORY_DIR),
        plan_challenges=Path(paths.PLAN_CHALLENGES_FILE),
        plan_answers=Path(paths.PLAN_CHALLENGE_ANSWERS_FILE),
        market_regime_table=Path(paths.MARKET_REGIME_TABLE_FILE),
        # S16.3 keeps its per-regime grades on the tracker worker; a stored file is
        # read when a build names one, otherwise the section says it is absent.
        regime_grades=(
            Path(paths.SETUP_GRADES_BY_REGIME_FILE)
            if getattr(paths, "SETUP_GRADES_BY_REGIME_FILE", None) else None
        ),
    )


# ---------------------------------------------------------------------------
# refuse-live-write
# ---------------------------------------------------------------------------
def protected_roots(sources: Sources) -> list[Path]:
    paths = _pp()
    roots: list[Path] = [
        Path(paths.PERSISTENT_DATA_DIR),
        Path(paths.SHARED_HOME_DIR),
        Path(paths.DATA_DIR),
        Path(paths.OUTPUT_DIR),
        Path(paths.LOCAL_SETTINGS_DIR),
        Path(paths.ROOT_DIR),
        ROOT_DIR,
        *HARD_PROTECTED_ROOTS,
    ]
    for extra in (sources.lake_root, sources.ai_store_root, *sources.protected):
        if extra:
            roots.append(Path(extra))
    return roots


def _norm(path: Path) -> str:
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except OSError:
        resolved = Path(os.path.abspath(Path(path).expanduser()))
    return os.path.normcase(str(resolved)).rstrip("\\/")


def _inside(child: Path, parent: Path) -> bool:
    c, p = _norm(child), _norm(parent)
    return c == p or c.startswith(p + os.sep)


def assert_safe_output_dir(out_dir: Path, sources: Sources) -> Path:
    for root in protected_roots(sources):
        if _inside(out_dir, root):
            raise LiveStoreWriteRefused(
                f"Refusing to write the research pack to {out_dir}: it is inside the live store {root}. "
                "Pass --out pointing at a scratch folder (default: the system temp folder)."
            )
    return Path(out_dir)


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.gettempdir()) / "tradingbotv3_research_pack" / stamp


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _exists(path: Path | None) -> bool:
    try:
        return bool(path) and Path(path).exists()
    except OSError:
        return False


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _aware(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else None


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat() if value.tzinfo else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value if value not in ("",) else None


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC)
    except OSError:
        return None


def _sessions_after(start: date, end: date) -> int:
    """Trading sessions in (start, end]."""
    count, day = 0, start + timedelta(days=1)
    while day <= end:
        if xcal.is_trading_day(day):
            count += 1
        day += timedelta(days=1)
    return count


def _bucket(value: float | None, edges=(-4, -2, -1, 0, 1, 2, 4)) -> str:
    if value is None:
        return "unknown"
    low = None
    for edge in edges:
        if value <= edge:
            return f"<={edge}" if low is None else f"({low},{edge}]"
        low = edge
    return f">{edges[-1]}"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def _status_row(source, kind, location, *, reachable, rows=None, files=None, date_min=None, date_max=None,
                freshness=None, note=""):
    age = None
    if isinstance(freshness, datetime):
        age = round((datetime.now(UTC) - freshness).total_seconds() / 3600.0, 1)
        freshness = freshness.isoformat()
    return {
        "source": source, "kind": kind, "location": str(location) if location else "",
        "reachable": reachable, "rows": rows, "files": files, "date_min": date_min, "date_max": date_max,
        "freshness": freshness, "age_hours": age, "note": note,
    }


def _lake_status(root: Path | None) -> list[dict]:
    if not _exists(root) or not _exists(Path(root) / "manifest_log.jsonl"):
        return [_status_row("research_lake", "parquet lake", root, reachable=False, note="lake root or manifest not reachable")]
    try:
        store = ResearchStore(Path(root))
        entries = store.manifest.resolve().entries
        written = [entry.written_at for entry in store.manifest.read_entries() if entry.written_at]
    except Exception as exc:  # noqa: BLE001 - status reports, never raises
        return [_status_row("research_lake", "parquet lake", root, reachable=False, note=f"manifest unreadable: {exc}")]
    rows = [
        _status_row("research_lake", "parquet lake", root, reachable=True, rows=sum(e.row_count for e in entries),
                    files=len(entries), freshness=_aware(max(written)) if written else None,
                    note="manifest-resolved; datasets below")
    ]
    by_dataset: dict[str, list] = {}
    for entry in entries:
        by_dataset.setdefault(entry.dataset, []).append(entry)
    for name in sorted(by_dataset):
        group = by_dataset[name]
        mins = [str(e.min_ts)[:10] for e in group if e.min_ts]
        maxs = [str(e.max_ts)[:10] for e in group if e.max_ts]
        stamps = [e.written_at for e in group if e.written_at]
        rows.append(_status_row(
            f"lake:{name}", "lake dataset", Path(root) / name, reachable=True,
            rows=sum(e.row_count for e in group), files=len(group),
            date_min=min(mins) if mins else None, date_max=max(maxs) if maxs else None,
            freshness=_aware(max(stamps)) if stamps else None,
        ))
    return rows


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _journal_status(path: Path | None) -> dict:
    if not _exists(path):
        return _status_row("trade_journal", "sqlite", path, reachable=False)
    try:
        connection = _connect_ro(path)
        try:
            count, low, high, updated = connection.execute(
                "SELECT COUNT(*), MIN(trade_date), MAX(trade_date), MAX(updated_at) FROM trades"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return _status_row("trade_journal", "sqlite", path, reachable=False, note=f"unreadable: {exc}")
    return _status_row("trade_journal", "sqlite", path, reachable=True, rows=count, date_min=low, date_max=high,
                       freshness=_aware(updated) or _mtime(path), note="table trades")


def _csv_status(name: str, path: Path | None, date_column: str) -> dict:
    if not _exists(path):
        return _status_row(name, "csv", path, reachable=False)
    rows, low, high = 0, None, None
    try:
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=1 << 24),
            convert_options=pacsv.ConvertOptions(
                include_columns=[date_column], include_missing_columns=True,
                column_types={date_column: pa.string()},
            ),
        )
        for batch in reader:
            rows += batch.num_rows
            values = [v for v in batch.column(0).to_pylist() if v]
            if values:
                low = min(low, min(values)) if low else min(values)
                high = max(high, max(values)) if high else max(values)
    except (pa.ArrowInvalid, OSError) as exc:
        return _status_row(name, "csv", path, reachable=False, note=f"unreadable: {exc}")
    return _status_row(name, "csv", path, reachable=True, rows=rows, date_min=low and low[:10],
                       date_max=high and high[:10], freshness=_mtime(path))


def _jsonl_files_status(name: str, path: Path | None, date_key: str) -> dict:
    if not _exists(path):
        return _status_row(name, "jsonl", path, reachable=False)
    files = sorted(Path(path).glob("*.jsonl")) if Path(path).is_dir() else [Path(path)]
    rows, low, high, machines = 0, None, None, set()
    newest = None
    for item in files:
        stamp = _mtime(item)
        newest = max(newest, stamp) if newest and stamp else (stamp or newest)
        try:
            with item.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows += 1
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    value = str(record.get(date_key) or record.get("ts") or "")[:10]
                    if value:
                        low = min(low, value) if low else value
                        high = max(high, value) if high else value
                    if record.get("machine"):
                        machines.add(str(record["machine"]))
        except OSError as exc:
            return _status_row(name, "jsonl", path, reachable=False, note=f"unreadable: {exc}")
    note = f"machines: {', '.join(sorted(machines))}" if machines else ""
    return _status_row(name, "jsonl", path, reachable=True, rows=rows, files=len(files), date_min=low,
                       date_max=high, freshness=newest, note=note)


def _ai_store_status(root: Path | None) -> dict:
    if not _exists(root):
        return _status_row("ai_store", "files", root, reachable=False, note="AI store not configured or not reachable")
    counts, newest, total = [], None, 0
    for sub in ("digests", "facts", "briefs", "retros"):
        folder = Path(root) / sub
        if not folder.exists():
            continue
        items = [item for item in folder.rglob("*") if item.is_file()]
        total += len(items)
        counts.append(f"{sub}={len(items)}")
        for item in items:
            stamp = _mtime(item)
            if stamp and (newest is None or stamp > newest):
                newest = stamp
    return _status_row("ai_store", "files", root, reachable=True, rows=total, files=total, freshness=newest,
                       note=" ".join(counts))


def collect_status(sources: Sources) -> list[dict]:
    rows = _lake_status(sources.lake_root)
    rows.append(_journal_status(sources.journal_db))
    rows.append(_csv_status("m5_bounce_outcomes", sources.bounce_outcomes_csv, "trade_date"))
    rows.append(_csv_status("bounces_csv", sources.bounces_csv, "trade_date"))
    for name, (path, date_key) in sorted(sources.extra_files.items()):
        if path and str(path).lower().endswith(".csv"):
            rows.append(_csv_status(name, path, date_key))
        else:
            rows.append(_jsonl_files_status(name, path, date_key))
    rows.append(_ai_store_status(sources.ai_store_root))
    return rows


def render_status(rows: list[dict]) -> str:
    header = f"{'source':34} {'ok':3} {'rows':>10} {'from':10} {'to':10} {'age h':>7}  location / note"
    lines = [header, "-" * len(header)]
    for row in rows:
        count = "unknown" if row["rows"] is None else str(row["rows"])
        age = "" if row["age_hours"] is None else f"{row['age_hours']:.1f}"
        lines.append(
            f"{row['source'][:34]:34} {'yes' if row['reachable'] else 'NO':3} {count:>10} "
            f"{(row['date_min'] or '-'):10} {(row['date_max'] or '-'):10} {age:>7}  {row['location']}"
            + (f"  [{row['note']}]" if row["note"] else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# D1 tables
# ---------------------------------------------------------------------------
def _revision_number(revision_id: Any) -> int:
    text = str(revision_id or "")
    try:
        return int(text[4:]) if text.startswith("rev-") else 0
    except ValueError:
        return 0


def _visible(row: dict, column: str, as_of: datetime | None) -> bool:
    if as_of is None:
        return True
    stamp = row.get(column)
    return stamp is None or stamp <= as_of


def _pick_feature(candidates: list[dict], trigger_at: datetime) -> tuple[dict | None, str]:
    """Latest snapshot session on or before the trigger session; prefer one computed by then."""
    if not candidates:
        return None, "missing"
    best_session = max(row["session_date"] for row in candidates)
    same = [row for row in candidates if row["session_date"] == best_session]
    observed = [row for row in same if row.get("computed_at") is not None and row["computed_at"] <= trigger_at]
    pool, basis = (observed, "as_observed") if observed else (same, "reconstructed")
    chosen = max(pool, key=lambda row: (str(row.get("feature_set_version") or ""),
                                        row.get("computed_at") or datetime.min.replace(tzinfo=UTC)))
    return chosen, basis


def build_d1_tables(store: ResearchStore, *, as_of: datetime | None = None,
                    recipe_id: str = DEFAULT_D1_RECIPE) -> tuple[list[dict], list[dict]]:
    """(occurrence rows, outcome rows) as known at ``as_of`` (``None`` = now)."""
    clock = as_of or datetime.now(UTC)
    latest: dict[str, dict] = {}
    for row in store.read_table("setup_occurrence").to_pylist():
        if not _visible(row, "computed_at", as_of) or (row.get("trigger_at") and row["trigger_at"] > clock):
            continue
        identity = str(row.get("occurrence_id") or "")
        current = latest.get(identity)
        if current is None or _revision_number(row.get("revision_id")) > _revision_number(current.get("revision_id")):
            latest[identity] = row
    if not latest:
        return [], []
    ids = sorted(latest)

    outcomes: dict[tuple, dict] = {}
    for row in store.read_rows("outcome_path", columns=list(OUTCOME_COLUMNS)):
        if str(row.get("occurrence_id")) not in latest or not _visible(row, "computed_at", as_of):
            continue
        key = (row["occurrence_id"], row.get("recipe_id"), row.get("outcome_definition_id"))
        current = outcomes.get(key)
        if current is None or (row.get("computed_at") or clock) >= (current.get("computed_at") or clock):
            outcomes[key] = row

    symbols = sorted({str(row.get("symbol") or "") for row in latest.values()} - {""})
    features_by_symbol: dict[str, list[dict]] = {}
    for row in store.read_rows("feature_snapshot_daily", symbols=symbols,
                               columns=["symbol", "session_date", "computed_at", "feature_set_version", *D1_FEATURE_COLUMNS]):
        if _visible(row, "computed_at", as_of):
            features_by_symbol.setdefault(str(row["symbol"]), []).append(row)

    context: dict[tuple, dict] = {}
    for row in store.read_rows("setup_market_context", occurrence_ids=ids,
                               columns=["occurrence_id", "timeframe", "env_key", "source", "computed_at"]):
        if not _visible(row, "computed_at", as_of):
            continue
        key = (row["occurrence_id"], row.get("timeframe"))
        if key not in context or (row.get("computed_at") or clock) >= (context[key].get("computed_at") or clock):
            context[key] = row

    occurrence_rows = []
    for identity in ids:
        occ = latest[identity]
        trigger_at = occ.get("trigger_at")
        session = trigger_at.astimezone(ET).date() if trigger_at else None
        try:
            tags = json.loads(occ.get("tags") or "{}")
        except ValueError:
            tags = {}
        tags = tags if isinstance(tags, dict) else {}
        candidates = [row for row in features_by_symbol.get(str(occ.get("symbol")), ())
                      if session is not None and row.get("session_date") and row["session_date"] <= session]
        feature, basis = _pick_feature(candidates, trigger_at) if trigger_at else (None, "missing")
        record = {
            "occurrence_id": identity,
            "symbol": occ.get("symbol"),
            "family": occ.get("canonical_setup_id"),
            "side": occ.get("side"),
            "structural_timeframe": occ.get("structural_timeframe"),
            "session_date": session.isoformat() if session else None,
            "trigger_at": _iso(trigger_at),
            "known_at": _iso(occ.get("computed_at")),
            "episode_id": occ.get("dependency_cluster_id"),
            "lifecycle_status_now": occ.get("status"),
            "entry_price_ref": occ.get("entry_price_ref"),
            "stop_price_ref": occ.get("stop_price_ref"),
            "priority_bucket": tags.get("priority_bucket"),
            "anchor_date": tags.get("anchor_date"),
            "rescan_count": tags.get("rescan_count"),
            "detector_version": occ.get("detector_version"),
            "revision_id": occ.get("revision_id"),
            "feat_basis": basis,
            "feat_session_date": feature["session_date"].isoformat() if feature else None,
        }
        for column in D1_FEATURE_COLUMNS:
            record[f"feat_{column}"] = feature.get(column) if feature else None
        for timeframe in CONTEXT_TIMEFRAMES:
            ctx = context.get((identity, timeframe))
            usable = ctx and ctx.get("source") != "insufficient_completed_bars" and ctx.get("env_key")
            record[f"ctx_{timeframe}"] = ctx["env_key"] if usable else "unknown"
        headline = outcomes.get((identity, recipe_id, OUTCOME_DEFINITION_ID))
        matured = bool(headline) and headline.get("maturity_at") is not None and headline["maturity_at"] <= clock
        record["hl_recipe_id"] = recipe_id
        record["hl_result_state"] = headline.get("result_state") if headline else None
        record["hl_matured"] = matured if headline else None
        for column in ("net_r", "gross_r", "mfe_r", "mae_r", *CHECKPOINTS):
            record[f"hl_{column}"] = headline.get(column) if headline and matured else None
        occurrence_rows.append(record)

    outcome_rows = []
    for (identity, _recipe, _definition), row in sorted(outcomes.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = {column: _iso(row.get(column)) if isinstance(row.get(column), datetime) else row.get(column)
                  for column in OUTCOME_COLUMNS}
        record["matured"] = row.get("maturity_at") is not None and row["maturity_at"] <= clock
        record["family"] = latest[identity].get("canonical_setup_id")
        record["side"] = latest[identity].get("side")
        outcome_rows.append(record)
    return occurrence_rows, outcome_rows


# ---------------------------------------------------------------------------
# M5 alerts
# ---------------------------------------------------------------------------
def _entry_at(naive_text: str, reference: datetime | None) -> datetime | None:
    """The alert CSV writes entry_time on the logging clock without an offset; borrow logged_at's."""
    try:
        naive = datetime.fromisoformat(str(naive_text or "").strip())
    except ValueError:
        return None
    if naive.tzinfo is not None:
        return naive
    if reference is None:
        return None
    return naive.replace(tzinfo=reference.tzinfo)


def _read_tiers(path: Path | None) -> dict[tuple, list[dict]]:
    if not _exists(path):
        return {}
    tiers: dict[tuple, list[dict]] = {}
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("trade_date"), str(row.get("symbol") or "").upper(), str(row.get("direction") or "").lower())
            tiers.setdefault(key, []).append(row)
    return tiers


def _tier_for(alert: dict, registered_at: datetime | None, tiers: dict) -> tuple[dict | None, int | None]:
    if registered_at is None:
        return None, None
    key = (alert["trade_date"], alert["symbol"], alert["direction"])
    best, best_gap = None, None
    for row in tiers.get(key, ()):
        try:
            moment = datetime.fromisoformat(f"{row['trade_date']}T{row['time_local']}").replace(tzinfo=registered_at.tzinfo)
        except (ValueError, KeyError, TypeError):
            continue
        gap = abs((moment - registered_at).total_seconds())
        if gap <= TIER_MATCH_SECONDS and (best_gap is None or gap < best_gap):
            best, best_gap = row, gap
    return best, (int(best_gap) if best_gap is not None else None)


def build_m5_alerts(outcomes_csv: Path | None, bounces_csv: Path | None, *, as_of: datetime | None) -> list[dict]:
    """One row per M5 bounce alert: context at fire time, milestone R, final mark."""
    if not _exists(outcomes_csv):
        return []
    reader = pacsv.open_csv(
        outcomes_csv,
        read_options=pacsv.ReadOptions(block_size=1 << 24),
        convert_options=pacsv.ConvertOptions(
            include_columns=list(M5_COLUMNS), include_missing_columns=True,
            column_types={name: pa.string() for name in M5_COLUMNS},
        ),
    )
    events: dict[str, dict] = {}
    for batch in reader:
        for row in batch.to_pylist():
            kind = row.get("event_type")
            if kind not in ("registered", "final", *M5_MILESTONES):
                continue
            logged = _aware(row.get("logged_at"))
            if as_of is not None and (logged is None or logged > as_of):
                continue
            state = events.setdefault(str(row.get("event_id")), {"_rows": {}})
            previous = state["_rows"].get(kind)
            if previous is None or (logged and previous[0] and logged >= previous[0]):
                state["_rows"][kind] = (logged, row)

    tiers = _read_tiers(bounces_csv)
    alerts = []
    for event_id in sorted(events):
        kinds = events[event_id]["_rows"]
        base_logged, base = kinds.get("registered") or min(kinds.values(), key=lambda item: item[0] or datetime.max.replace(tzinfo=UTC))
        match = _EVENT_ID.match(event_id)
        types = match.group("types").split("-") if match else []
        direction = str(base.get("direction") or (match.group("dir") if match else "")).lower()
        try:
            context = json.loads(base.get("context_json") or "{}")
        except ValueError:
            context = {}
        context = context if isinstance(context, dict) else {}
        symbol_context = context.get("symbol_context") if isinstance(context.get("symbol_context"), dict) else {}
        entry_at = _entry_at(base.get("entry_time"), base_logged)
        alert = {
            "event_id": event_id,
            "symbol": str(base.get("symbol") or "").upper(),
            "trade_date": base.get("trade_date"),
            "direction": direction,
            "side": direction.upper() if direction in ("long", "short") else None,
            "family": match.group("types") if match else None,
            "bounce_types": "|".join(types) if types else None,
            "n_types": len(types) or None,
            "known_at": _iso(base_logged) if "registered" in kinds else None,
            "entry_at": _iso(entry_at),
            "entry_hour_et": entry_at.astimezone(ET).hour if entry_at else None,
            "entry_price": _float(base.get("entry_price")),
            "stop_price": _float(base.get("stop_price")),
            "risk_per_share": _float(base.get("risk_per_share")),
            "market_environment": context.get("market_environment") or None,
            "sector": context.get("sector") or None,
            "industry": context.get("industry") or None,
            "watchlist_bias": context.get("watchlist_bias") or None,
            "rrs_spy": _float(context.get("rrs_spy")),
            "rrs_spy_signal": context.get("rrs_spy_signal") or None,
            "rrs_sector_signal": context.get("rrs_sector_signal") or None,
            "rrs_industry_signal": context.get("rrs_industry_signal") or None,
            "move_ratio": _float(symbol_context.get("move_ratio")),
            "excess_move_ratio": _float(symbol_context.get("excess_move_ratio")),
            "context_json": base.get("context_json") or None,
            "outcome_mode": base.get("outcome_mode") or None,
        }
        tier_row, gap = _tier_for(alert, base_logged if "registered" in kinds else None, tiers)
        alert["tier"] = (tier_row.get("tier") or None) if tier_row else None
        alert["composite_r"] = _float(tier_row.get("composite_r")) if tier_row else None
        alert["tier_match"] = f"within_{gap}s" if tier_row else "unmatched"
        for kind, column in M5_MILESTONES.items():
            alert[column] = _float(kinds[kind][1].get("close_r")) if kind in kinds else None
        final = kinds.get("final")
        final_row = final[1] if final else {}
        alert["final_status"] = final_row.get("status") if final else None
        alert["final_logged_at"] = _iso(final[0]) if final else None
        resolved = bool(final) and final_row.get("status") != "unresolved"
        alert["close_r_final"] = _float(final_row.get("close_r")) if resolved else None
        alert["mfe_r"] = _float(final_row.get("mfe_r")) if resolved else None
        alert["mae_r"] = _float(final_row.get("mae_r")) if resolved else None
        alert["target_1r_hit"] = _bool(final_row.get("target_1r_hit")) if resolved else None
        alert["target_2r_hit"] = _bool(final_row.get("target_2r_hit")) if resolved else None
        alert["stop_hit"] = _bool(final_row.get("stop_hit")) if resolved else None
        alert["eod_move_pct"] = _float(final_row.get("eod_move_pct")) if resolved else None
        alerts.append(alert)
    return alerts


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------
JOURNAL_COLUMNS = (
    "trade_id", "broker", "symbol", "security_type", "currency", "direction", "status", "opened_at",
    "closed_at", "trade_date", "quantity_opened", "quantity_closed", "average_entry_price",
    "average_exit_price", "net_pnl", "pnl_usd", "net_pnl_usd", "auto_tag_summary",
)


def load_journal_trades(db_path: Path | None, *, as_of: datetime | None) -> list[dict]:
    """Trades (no account identifiers) with their annotation tags, as known at ``as_of``."""
    if not _exists(db_path):
        return []
    connection = _connect_ro(db_path)
    connection.row_factory = sqlite3.Row
    try:
        available = {row[1] for row in connection.execute("PRAGMA table_info(trades)")}
        columns = [column for column in JOURNAL_COLUMNS if column in available]
        has_annotations = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trade_annotations'"
        ).fetchone()
        select = ", ".join(f"t.{column}" for column in columns)
        if has_annotations:
            query = (f"SELECT {select}, a.setup_tags AS setup_tags, a.tag_status AS tag_status "
                     "FROM trades t LEFT JOIN trade_annotations a ON a.trade_id = t.trade_id")
        else:
            query = f"SELECT {select} FROM trades t"
        raw = [dict(row) for row in connection.execute(query)]
    finally:
        connection.close()

    trades = []
    for row in raw:
        opened = _aware(row.get("opened_at"))
        if as_of is not None and (opened is None or opened > as_of):
            continue
        closed = _aware(row.get("closed_at"))
        closed_known = closed is not None and (as_of is None or closed <= as_of)
        pnl = row.get("net_pnl_usd")
        if pnl is None:
            pnl = row.get("pnl_usd")
        status = str(row.get("status") or "")
        if not closed_known and status.startswith("CLOSED"):
            status = "OPEN_AT_AS_OF"
        done = status == "CLOSED" and closed_known
        underlying, side, basis = _trade_setup_side(row)
        trades.append({
            "trade_id": row.get("trade_id"),
            "broker": row.get("broker"),
            "symbol": row.get("symbol"),
            "underlying": underlying,
            "security_type": row.get("security_type"),
            "direction": row.get("direction"),
            "setup_side": side,
            "side_basis": basis,
            "status": status,
            "opened_at": _iso(opened) or row.get("opened_at"),
            "closed_at": (_iso(closed) if closed_known else None),
            "session_date": opened.astimezone(ET).date().isoformat() if opened else None,
            "hold_minutes": round((closed - opened).total_seconds() / 60.0, 1) if (done and opened) else None,
            "quantity_opened": row.get("quantity_opened"),
            "average_entry_price": row.get("average_entry_price"),
            "average_exit_price": row.get("average_exit_price") if done else None,
            "net_pnl_usd": pnl if done else None,
            "currency": row.get("currency"),
            "win": (pnl > 0) if (done and pnl is not None) else None,
            "auto_tag_summary": row.get("auto_tag_summary"),
            "setup_tags": row.get("setup_tags"),
            "tag_status": row.get("tag_status"),
            "_opened": opened,
        })
    return trades


def _trade_setup_side(trade: dict) -> tuple[str, str | None, str]:
    """(underlying, LONG/SHORT/None, basis) - a sold put is a LONG setup."""
    symbol = str(trade.get("symbol") or "").upper().strip()
    exposure = journal_exposure.classify_exposure(trade)
    contracts = exposure.contracts
    underlying = contracts[0].underlying if contracts and contracts[0].underlying else symbol
    bias = exposure.market_bias
    if bias in (journal_exposure.BIAS_BULLISH, journal_exposure.BIAS_BULLISH_OR_NEUTRAL):
        return underlying, "LONG", f"exposure:{bias}"
    if bias in (journal_exposure.BIAS_BEARISH, journal_exposure.BIAS_BEARISH_OR_NEUTRAL):
        return underlying, "SHORT", f"exposure:{bias}"
    alt = _ALT_OPTION.match(symbol)
    ownership = str(trade.get("direction") or "").upper()
    if alt:
        right = alt.group("right")
        if ownership == "LONG":
            return alt.group("root"), ("LONG" if right == "C" else "SHORT"), "alt_option_symbol"
        if ownership == "SHORT":
            return alt.group("root"), ("LONG" if right == "P" else "SHORT"), "alt_option_symbol"
        return alt.group("root"), None, "unknown"
    if ownership in ("LONG", "SHORT") and re.fullmatch(r"[A-Z][A-Z.]{0,5}", symbol):
        return underlying, ownership, "ownership_unverified_instrument"
    return underlying, None, "unknown"


def match_journal_trades(trades: list[dict], d1_occurrences: list[dict], m5_alerts: list[dict], *,
                         d1_max_sessions: int = DEFAULT_D1_MAX_SESSIONS,
                         m5_window_minutes: int = DEFAULT_M5_WINDOW_MINUTES) -> list[dict]:
    """Join each trade to the setup known before it opened. Unmatched stays unmatched."""
    d1_by_symbol: dict[str, list[dict]] = {}
    for occ in d1_occurrences:
        if occ.get("session_date"):
            d1_by_symbol.setdefault(str(occ.get("symbol") or "").upper(), []).append(occ)
    m5_by_key: dict[tuple, list[tuple[datetime, dict]]] = {}
    for alert in m5_alerts:
        known = _aware(alert.get("known_at"))
        if known is None:
            continue
        key = (alert.get("symbol"), known.astimezone(ET).date().isoformat())
        m5_by_key.setdefault(key, []).append((known, alert))

    matched = []
    for trade in trades:
        row = {key: value for key, value in trade.items() if not key.startswith("_")}
        opened = trade.get("_opened") or _aware(trade.get("opened_at"))
        session = date.fromisoformat(trade["session_date"]) if trade.get("session_date") else None
        underlying = str(trade.get("underlying") or "").upper()
        side = trade.get("setup_side")

        d1 = {"d1_occurrence_id": None, "d1_family": None, "d1_side": None, "d1_session_date": None,
              "d1_sessions_before": None, "d1_candidates": 0, "d1_candidate_families": None, "d1_match": "unmatched"}
        if session is not None:
            window = []
            for occ in d1_by_symbol.get(underlying, ()):
                trigger_session = date.fromisoformat(occ["session_date"])
                if trigger_session >= session:
                    continue
                gap = _sessions_after(trigger_session, session)
                if 1 <= gap <= d1_max_sessions:
                    window.append((gap, occ))
            same_side = [item for item in window if side is None or item[1].get("side") == side]
            if same_side:
                best_gap = min(gap for gap, _ in same_side)
                nearest = sorted((occ for gap, occ in same_side if gap == best_gap),
                                 key=lambda occ: (str(occ.get("family")), str(occ.get("occurrence_id"))))
                best = nearest[0]
                quality = "next_session" if best_gap == 1 else f"within_{best_gap}_sessions"
                d1.update({
                    "d1_occurrence_id": best["occurrence_id"], "d1_family": best.get("family"),
                    "d1_side": best.get("side"), "d1_session_date": best.get("session_date"),
                    "d1_sessions_before": best_gap, "d1_candidates": len(nearest),
                    "d1_candidate_families": "|".join(sorted({str(occ.get("family")) for occ in nearest})),
                    "d1_match": quality if side is not None else f"{quality}_side_unknown",
                })
            elif window:
                d1["d1_match"] = "unmatched_opposite_side"

        m5 = {"m5_event_id": None, "m5_family": None, "m5_tier": None, "m5_minutes_before": None,
              "m5_candidates": 0, "m5_match": "unmatched"}
        if opened is not None and session is not None:
            pool = []
            for known, alert in m5_by_key.get((underlying, session.isoformat()), ()):
                minutes = (opened - known).total_seconds() / 60.0
                if 0 <= minutes <= m5_window_minutes and (side is None or alert.get("side") == side):
                    pool.append((minutes, alert))
            if pool:
                minutes, best = min(pool, key=lambda item: (item[0], str(item[1].get("event_id"))))
                band = "within_15m" if minutes <= 15 else ("within_60m" if minutes <= 60 else f"within_{m5_window_minutes}m")
                m5.update({
                    "m5_event_id": best["event_id"], "m5_family": best.get("family"), "m5_tier": best.get("tier"),
                    "m5_minutes_before": round(minutes, 1), "m5_candidates": len(pool),
                    "m5_match": band if side is not None else f"{band}_side_unknown",
                })
        row.update(d1)
        row.update(m5)
        matched.append(row)
    return matched


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def rank_cells(cells: list[dict], floor: int) -> list[dict]:
    """Rank by ``rank_value`` (else mean) within (source, recipe, trait); cells under ``floor`` episodes get no rank."""
    groups: dict[tuple, list[dict]] = {}
    for cell in cells:
        cell["floor"] = floor
        cell["below_floor"] = (cell.get("n_episodes") or 0) < floor or cell.get("mean") is None
        cell["rank"] = None
        if not cell["below_floor"]:
            groups.setdefault((cell.get("source"), cell.get("recipe_id"), cell.get("trait")), []).append(cell)
    for group in groups.values():
        for position, cell in enumerate(sorted(group, key=lambda c: -float(_rank_value(c))), start=1):
            cell["rank"] = position
    return cells


def _rank_value(cell: dict) -> float:
    value = cell.get("rank_value")
    return cell["mean"] if value is None else value


def _cell(source, family, side, recipe_id, trait, value, metric, samples, episodes, extras, clip=None):
    values = [v for v in samples if v is not None]
    clipped = [max(-clip, min(clip, v)) for v in values] if clip else values
    cell = {
        "source": source, "family": family, "side": side, "recipe_id": recipe_id, "trait": trait,
        "trait_value": value, "metric": metric, "n": len(values), "n_episodes": len(episodes - {None}),
        "wins": sum(1 for v in values if v > 0),
        "win_rate": (sum(1 for v in values if v > 0) / len(values)) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "rank_value": statistics.fmean(clipped) if clipped else None,
        "rank_basis": f"mean clipped to +/-{clip:g}" if clip else "mean",
        "evidence_tier": EVIDENCE_TIER,
    }
    for name, numbers in extras.items():
        numbers = [v for v in numbers if v is not None]
        cell[name] = statistics.fmean(numbers) if numbers else None
    return cell


def _grouped_cells(source, rows, *, metric, value_of, episode_of, traits, extras, recipe_id=None, clip=None):
    buckets: dict[tuple, dict] = {}
    for row in rows:
        metric_value = value_of(row)
        if metric_value is None:
            continue
        for trait, trait_values in traits(row):
            for trait_value in trait_values:
                for family, side in ((row.get("family"), row.get("side")), ("*", "*")):
                    key = (family, side, trait, trait_value)
                    bucket = buckets.setdefault(key, {"samples": [], "episodes": set(), "extras": {k: [] for k in extras}})
                    bucket["samples"].append(metric_value)
                    bucket["episodes"].add(episode_of(row))
                    for name, getter in extras.items():
                        bucket["extras"][name].append(getter(row))
    return [
        _cell(source, family, side, recipe_id, trait, value, metric, b["samples"], b["episodes"], b["extras"], clip)
        for (family, side, trait, value), b in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0]))
    ]


def summarize(d1_occurrences: list[dict], m5_alerts: list[dict], journal: list[dict], *,
              floor: int = DEFAULT_FLOOR, journal_floor: int = DEFAULT_JOURNAL_FLOOR) -> list[dict]:
    def d1_traits(row):
        return [
            ("all", ["all"]),
            ("priority_bucket", [row.get("priority_bucket") or "unknown"]),
            ("spy_regime_state", [row.get("feat_spy_regime_state") or "unknown"]),
            ("dist_sma50_atr", [_bucket(row.get("feat_dist_sma50_atr"))]),
            ("dist_sma200_atr", [_bucket(row.get("feat_dist_sma200_atr"))]),
            ("anchor_knowledge", [row.get("feat_anchor_knowledge") or "unknown"]),
            ("ctx_H1", [row.get("ctx_H1") or "unknown"]),
            ("ctx_M30", [row.get("ctx_M30") or "unknown"]),
        ]

    def m5_traits(row):
        hour = row.get("entry_hour_et")
        return [
            ("all", ["all"]),
            ("bounce_type", (row.get("bounce_types") or "unknown").split("|")),
            ("tier", [row.get("tier") or "unknown"]),
            ("market_environment", [row.get("market_environment") or "unknown"]),
            ("rrs_spy_signal", [row.get("rrs_spy_signal") or "unknown"]),
            ("sector", [row.get("sector") or "unknown"]),
            ("entry_hour_et", [str(hour) if hour is not None else "unknown"]),
        ]

    recipe = next((row.get("hl_recipe_id") for row in d1_occurrences if row.get("hl_recipe_id")), None)
    d1_cells = _grouped_cells(
        "d1_setups", d1_occurrences, metric="hl_net_r", recipe_id=recipe, clip=RANK_CLIP_R,
        value_of=lambda r: r.get("hl_net_r") if r.get("hl_result_state") not in (None, "NO_TRIGGER", "TRUNCATED", "OPEN") else None,
        episode_of=lambda r: r.get("episode_id"), traits=d1_traits,
        extras={"mean_r_at_s5": lambda r: r.get("hl_r_at_s5"), "mean_r_at_s10": lambda r: r.get("hl_r_at_s10"),
                "mean_mfe_r": lambda r: r.get("hl_mfe_r"), "mean_mae_r": lambda r: r.get("hl_mae_r")},
    )
    m5_cells = _grouped_cells(
        "m5_alerts", m5_alerts, metric="close_r_final", clip=RANK_CLIP_R,
        value_of=lambda r: r.get("close_r_final"),
        episode_of=lambda r: (r.get("symbol"), r.get("trade_date"), r.get("direction")), traits=m5_traits,
        extras={"mean_r_12bar": lambda r: r.get("r_12bar"), "mean_mfe_r": lambda r: r.get("mfe_r"),
                "mean_mae_r": lambda r: r.get("mae_r")},
    )
    journal_rows = [dict(row, family=row.get("d1_family") or row.get("m5_family") or "unmatched",
                         side=row.get("setup_side") or "unknown") for row in journal]
    journal_cells = _grouped_cells(
        "journal_trades", journal_rows, metric="net_pnl_usd",
        value_of=lambda r: r.get("net_pnl_usd") if r.get("win") is not None else None,
        episode_of=lambda r: r.get("trade_id"),
        traits=lambda r: [("all", ["all"]), ("d1_match", [str(r.get("d1_match")).split("_")[0]]),
                          ("m5_match", [str(r.get("m5_match")).split("_")[0]]),
                          ("security_type", [r.get("security_type") or "unknown"])],
        extras={"mean_hold_minutes": lambda r: r.get("hold_minutes")},
    )
    return rank_cells(d1_cells, floor) + rank_cells(m5_cells, floor) + rank_cells(journal_cells, journal_floor)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
PLAN_WEEK_DAYS = 7


def export_plan(sources: Sources, out_dir: Path, *, as_of: datetime | None = None,
                today: date | None = None) -> dict:
    """Copy the plan and write the week's challenges with their answers (P1-7 7c).

    Read-only on every source. With ``as_of`` the plan is the newest history
    snapshot taken by then, and later challenges and answers are hidden.
    """
    from scripts import plan_challenges, trading_plan

    info: dict[str, Any] = {"plan": {"status": "missing", "file": None, "source": None},
                            "challenges": {"status": "missing", "file": None, "rows": None}}
    plan_text = None
    if as_of is not None:
        chosen = trading_plan.snapshot_at(as_of, sources.plan_history_dir) if sources.plan_history_dir else None
        if chosen is not None:
            plan_text = chosen.read_text(encoding="utf-8")
            info["plan"]["source"] = f"snapshot {chosen.name}"
    elif _exists(sources.trading_plan):
        plan_text = Path(sources.trading_plan).read_text(encoding="utf-8")
        info["plan"]["source"] = "current"
    if plan_text is not None:
        (out_dir / "trading_plan.md").write_text(plan_text, encoding="utf-8")
        info["plan"].update(status="ok", file="trading_plan.md")

    if _exists(sources.plan_challenges):
        end = (as_of.astimezone(ET).date() if as_of is not None else (today or datetime.now(ET).date()))
        rows = plan_challenges.week_rows(
            end, days=PLAN_WEEK_DAYS, as_of=as_of, challenges_file=Path(sources.plan_challenges),
            answers_file=Path(sources.plan_answers) if sources.plan_answers else Path(out_dir / "_none"),
        )
        by_status: dict[str, int] = {}
        for row in rows:
            by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        payload = {"window_end": end.isoformat(), "window_days": PLAN_WEEK_DAYS,
                   "by_status": by_status, "challenges": rows}
        (out_dir / "plan_challenges.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        info["challenges"].update(status="ok", file="plan_challenges.json", rows=len(rows), by_status=by_status,
                                  window_end=end.isoformat())
    return info


def _visible_regime_row(row: dict, as_of: datetime | None) -> bool:
    """A table row is visible at ``as_of`` once it was computed by then (else its session closed)."""
    if as_of is None:
        return True
    stamp = _aware(row.get("computed_at"))
    if stamp is not None:
        return stamp <= as_of
    session = str(row.get("session_date") or "")[:10]
    return bool(session) and session <= as_of.astimezone(ET).date().isoformat()


def _regime_journal_rows(db_path: Path | None, as_of: datetime | None) -> list[dict] | None:
    """The trader's regime journal (``structural_regime``), read-only; None when unreadable."""
    if not _exists(db_path):
        return None
    connection = _connect_ro(db_path)
    try:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM structural_regime ORDER BY segment_id")]
    finally:
        connection.close()
    if as_of is not None:
        rows = [row for row in rows if (_aware(row.get("entered_at")) or as_of) <= as_of]
    return rows


def export_market_regimes(sources: Sources, out_dir: Path, *, as_of: datetime | None = None) -> dict:
    """S17.2: the full regime table, the trader's regime journal and per-regime grades.

    Read-only on every source; ``as_of`` hides table rows computed later and
    journal rows typed later. A missing source is named, never faked.
    """
    from scripts import structural_regime

    info: dict[str, Any] = {}
    notes: dict[str, str] = {}
    rows: list[dict] | None = None
    if _exists(sources.market_regime_table):
        rows = []
        for line in Path(sources.market_regime_table).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and _visible_regime_row(row, as_of):
                rows.append(row)
        with (out_dir / "market_regime_table.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        low, high = _date_span(rows, "session_date")
        info["table"] = {"status": "ok", "file": "market_regime_table.jsonl", "rows": len(rows),
                         "symbols": sorted({str(row.get("symbol") or "") for row in rows}),
                         "date_min": low, "date_max": high}
    else:
        info["table"] = {"status": "missing", "file": None, "rows": None}
    try:
        journal = _regime_journal_rows(sources.journal_db, as_of)
    except sqlite3.Error as exc:
        journal = None
        notes["journal"] = f"regime journal unreadable: {exc}"
    if journal is None:
        info["journal"] = {"status": "missing", "file": None, "rows": None}
    else:
        payload = {"rows": journal, "timeline": structural_regime.effective_segments(journal),
                   "vocabulary": list(structural_regime.VOCABULARY), "source": "trader"}
        (out_dir / "regime_journal.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        info["journal"] = {"status": "ok", "file": "regime_journal.json", "rows": len(journal)}
    if _exists(sources.regime_grades):
        copied = out_dir / "setup_grades_by_regime.json"
        copied.write_bytes(Path(sources.regime_grades).read_bytes())
        info["grades"] = {"status": "ok", "file": copied.name}
    else:
        info["grades"] = {"status": "missing", "file": None}
        notes["grades"] = ("setups-by-regime grades are not in this pack: no stored file exists "
                           "(S16.3 computes them on the tracker worker)")
    info["notes"] = notes
    info["how_to_read"] = ("One table row per session and symbol: env_key on M5, M30, H1, H4 (close), D1 and W, "
                           "three intraday snapshots and the structure facts. The journal is the trader's own "
                           "regime; the table is the machine's check beside it, never over it.")
    return info


def _write_table(rows: list[dict], out_dir: Path, name: str, fmt: str) -> dict:
    if not rows:
        return {"status": "empty", "rows": 0, "file": None}
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    table = pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in rows])
    filename = f"{name}.{fmt}"
    if fmt == "parquet":
        pq.write_table(table, out_dir / filename, compression="zstd")
    else:
        pacsv.write_csv(table, out_dir / filename)
    return {"status": "ok", "rows": len(rows), "file": filename, "columns": columns}


def _date_span(rows: Iterable[dict], column: str) -> tuple[str | None, str | None]:
    values = sorted(str(row[column])[:10] for row in rows if row.get(column))
    return (values[0], values[-1]) if values else (None, None)


def export_pack(sources: Sources, out_dir: Path | None = None, *, as_of: datetime | None = None,
                fmt: str = "parquet", floor: int = DEFAULT_FLOOR, journal_floor: int = DEFAULT_JOURNAL_FLOOR,
                d1_recipe: str = DEFAULT_D1_RECIPE, d1_max_sessions: int = DEFAULT_D1_MAX_SESSIONS,
                m5_window_minutes: int = DEFAULT_M5_WINDOW_MINUTES) -> dict:
    """Write the pack; returns its manifest. Refuses any output folder inside a live store."""
    if fmt not in ("parquet", "csv"):
        raise ValueError("fmt must be 'parquet' or 'csv'")
    target = assert_safe_output_dir(Path(out_dir) if out_dir else default_output_dir(), sources)
    target.mkdir(parents=True, exist_ok=True)
    notes: dict[str, str] = {}

    d1_occ: list[dict] | None = None
    d1_out: list[dict] | None = None
    if _exists(sources.lake_root) and _exists(Path(sources.lake_root) / "manifest_log.jsonl"):
        try:
            d1_occ, d1_out = build_d1_tables(ResearchStore(Path(sources.lake_root)), as_of=as_of, recipe_id=d1_recipe)
        except Exception as exc:  # noqa: BLE001 - one bad source must not sink the pack
            notes["d1"] = f"lake read failed: {exc}"
    m5: list[dict] | None = None
    if _exists(sources.bounce_outcomes_csv):
        try:
            m5 = build_m5_alerts(sources.bounce_outcomes_csv, sources.bounces_csv, as_of=as_of)
        except Exception as exc:  # noqa: BLE001
            notes["m5"] = f"alert read failed: {exc}"
    trades: list[dict] | None = None
    if _exists(sources.journal_db):
        try:
            trades = load_journal_trades(sources.journal_db, as_of=as_of)
        except Exception as exc:  # noqa: BLE001
            notes["journal"] = f"journal read failed: {exc}"
    matched = (match_journal_trades(trades, d1_occ or [], m5 or [], d1_max_sessions=d1_max_sessions,
                                    m5_window_minutes=m5_window_minutes) if trades is not None else None)
    cells = summarize(d1_occ or [], m5 or [], matched or [], floor=floor, journal_floor=journal_floor)

    tables: dict[str, dict] = {}
    for name, rows, date_column in (
        ("d1_occurrences", d1_occ, "session_date"),
        ("d1_outcomes", d1_out, "entry_at"),
        ("m5_alerts", m5, "trade_date"),
        ("journal_trades", matched, "session_date"),
    ):
        if rows is None:
            tables[name] = {"status": "missing", "rows": None, "file": None}
            continue
        info = _write_table(rows, target, name, fmt)
        info["date_min"], info["date_max"] = _date_span(rows, date_column)
        tables[name] = info
    summary_info = _write_table(cells, target, "setup_summary", "csv")
    tables["setup_summary"] = summary_info
    # P1-4 4d: the setup-keys report travels whole; a missing one is said, not faked.
    if _exists(sources.setup_keys_report):
        copied = target / "setup_permutation_report.json"
        copied.write_bytes(Path(sources.setup_keys_report).read_bytes())
        tables["setup_permutation_report"] = {"status": "ok", "rows": None, "file": copied.name}
    else:
        tables["setup_permutation_report"] = {"status": "missing", "rows": None, "file": None}
    try:
        plan_info = export_plan(sources, target, as_of=as_of)
    except Exception as exc:  # noqa: BLE001 - one bad source must not sink the pack
        plan_info = {}
        notes["trading_plan"] = f"plan read failed: {exc}"
    try:
        regimes_info = export_market_regimes(sources, target, as_of=as_of)
    except Exception as exc:  # noqa: BLE001 - one bad source must not sink the pack
        regimes_info = {}
        notes["market_regimes"] = f"regime read failed: {exc}"

    def _count(rows, column):
        return sum(1 for row in rows or [] if row.get(column))

    manifest = {
        "schema": PACK_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat() if as_of else None,
        "evidence_tier": EVIDENCE_TIER,
        "output_dir": str(target),
        "parameters": {"format": fmt, "floor_episodes": floor, "journal_floor_trades": journal_floor,
                       "d1_headline_recipe": d1_recipe, "outcome_definition_id": OUTCOME_DEFINITION_ID,
                       "d1_max_sessions": d1_max_sessions, "m5_window_minutes": m5_window_minutes},
        "tables": tables,
        "trading_plan": plan_info,
        "market_regimes": regimes_info,
        "join": {
            "journal_trades": len(matched or []),
            "d1_matched": _count(matched, "d1_occurrence_id"),
            "m5_matched": _count(matched, "m5_event_id"),
            "either_matched": sum(1 for r in matched or [] if r.get("d1_occurrence_id") or r.get("m5_event_id")),
            "m5_alerts_with_tier": sum(1 for r in m5 or [] if r.get("tier")),
        },
        "point_in_time": [
            "Journal trades join only D1 setups whose trigger session closed before the trade's session "
            f"(1..{d1_max_sessions} sessions) and M5 alerts logged 0..{m5_window_minutes} min before the trade opened.",
            "feat_basis: as_observed = snapshot computed by the trigger; reconstructed = computed later from bars "
            "up to the trigger session; missing = none (unknown, not zero).",
            "m5 context is the registered row's context_json, written when the alert fired.",
            "Outcomes (hl_*, d1_outcomes, r_*bar, close_r_final) are labels measured after the fact.",
            "lifecycle_status_now is today's detector state, not a point-in-time trait.",
            "--as-of hides revisions, outcomes, alert rows and trades written after that moment.",
        ],
        "sources": collect_status(sources),
        "notes": notes,
        "how_to_read": "Start with setup_summary.csv (rank is empty under the floor), then join tables on "
                       "occurrence_id / event_id. See scripts/research_pack.py --help.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_as_of(text: str | None) -> datetime | None:
    if not text:
        return None
    if len(text) == 10:
        day = date.fromisoformat(text)
        return datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=ET).astimezone(UTC)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--as-of with a time needs a timezone offset")
    return parsed.astimezone(UTC)


def _apply_overrides(sources: Sources, args) -> Sources:
    for attr in ("lake_root", "journal_db", "bounce_outcomes_csv", "bounces_csv"):
        value = getattr(args, attr, None)
        if value:
            setattr(sources, attr, Path(value))
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research_pack", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lake", dest="lake_root", help="override the research lake root")
    common.add_argument("--journal-db", dest="journal_db", help="override trade_journal.sqlite3")
    common.add_argument("--bounce-outcomes", dest="bounce_outcomes_csv", help="override intraday_bounce_outcomes.csv")
    common.add_argument("--bounces", dest="bounces_csv", help="override intraday_bounces.csv")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", parents=[common], help="list every source and its coverage")
    status.add_argument("--json", action="store_true")
    export = sub.add_parser("export", parents=[common], help="write the research pack to a scratch folder")
    export.add_argument("--out", help="output folder (default: system temp); live stores are refused")
    export.add_argument("--as-of", help="YYYY-MM-DD (end of day ET) or ISO time with offset")
    export.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    export.add_argument("--floor", type=int, default=DEFAULT_FLOOR, help="episodes a cell needs to be ranked")
    export.add_argument("--journal-floor", type=int, default=DEFAULT_JOURNAL_FLOOR)
    export.add_argument("--d1-recipe", default=DEFAULT_D1_RECIPE)
    export.add_argument("--d1-max-sessions", type=int, default=DEFAULT_D1_MAX_SESSIONS)
    export.add_argument("--m5-window-min", type=int, default=DEFAULT_M5_WINDOW_MINUTES)
    args = parser.parse_args(argv)

    if args.command == "export":
        # Refuse before any source is resolved or read.
        out = Path(args.out) if args.out else default_output_dir()
        bare = Sources(None, None, None, None, {}, None, ())
        try:
            assert_safe_output_dir(out, bare)
        except LiveStoreWriteRefused as exc:
            print(str(exc), file=sys.stderr)
            return 2
    sources = _apply_overrides(resolve_sources(), args)
    if args.command == "status":
        rows = collect_status(sources)
        print(json.dumps(rows, indent=2, default=str) if args.json else render_status(rows))
        return 0
    try:
        manifest = export_pack(
            sources, out, as_of=_parse_as_of(args.as_of), fmt=args.format, floor=args.floor,
            journal_floor=args.journal_floor, d1_recipe=args.d1_recipe, d1_max_sessions=args.d1_max_sessions,
            m5_window_minutes=args.m5_window_min,
        )
    except LiveStoreWriteRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Research pack written to {manifest['output_dir']}")
    for name, info in manifest["tables"].items():
        span = f" {info.get('date_min')}..{info.get('date_max')}" if info.get("date_min") else ""
        rows = "unknown" if info["rows"] is None else info["rows"]
        print(f"  {name:16} {info['status']:8} rows={rows}{span}")
    join = manifest["join"]
    print(f"  journal join: {join['d1_matched']} D1 / {join['m5_matched']} M5 of {join['journal_trades']} trades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
