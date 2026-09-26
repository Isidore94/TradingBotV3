"""Setup permutation keys (WISHLIST P1-4 / 4c): the search.

    python scripts/setup_permutation_search.py --outcomes <scratch>/permutation_outcomes.parquet \
        --ledger-root <warehouse root or scratch> --out <path>/permutation_report.json

For each population (swing, m5 - never pooled), each horizon and each family x
side: a baseline, then single facets, pairs and triples, never deeper.

- Statistics per cell: n, sessions, wins, win rate, Wilson 95% lower bound,
  mean R. Win rate first (decision 0016).
- Floors as in `setup_grades`: n >= 30 and sessions >= 10 in the selection
  window. Floors are counts, not outcomes, so each depth's grid is fixed
  before any outcome is read.
- Hold-out: the population's last 20 sessions, never used for selection; the
  selection sessions whose outcome window reaches into it are embargoed too.
- Every grid (one per population x horizon x family x side x depth) is
  registered in `research_warehouse.trial_ledger` BEFORE its outcomes are read.
  A grid of more than 10 cells (the k > 10 rule) must pass hold-out on the 99%
  Wilson bound AND beat the grid's median hold-out win rate.
- Depth 2 and 3 extend the best BEAM_WIDTH cells of the depth above by one
  facet; a deeper key must beat its best parent on hold-out or it is not
  reported.
- "no key found" is a first-class answer.
- Horizons: swing 1/3/5/10 sessions; m5 ``0`` (``held30``, the level held 30
  minutes) and ``bracket_1r`` (+1R before -1R, S3). Every horizon block,
  family and key carries ``horizon_name``.
- A population x horizon whose selection window has under
  MIN_SELECTION_SESSIONS sessions is not searched: the block says
  ``refused`` and why (S4; F11 was a 5-session window).
- S14 study families (`long_study_families.STUDY_FAMILIES`, rows the backfill
  copies under the study name) are searched FIRST, on their own small grid of
  `STUDY_SEARCH_FACETS` (spy_trend, trend20, htf_trend_4h), and reported in
  each horizon's ``study_families`` block, never in ``families`` (so no
  verdict, chip or narration reads them). Promotion is ask-first.
- Each run also keeps a dated copy in `permutation_report_history/` and writes
  `permutation_verdicts.json` (P12, `setup_permutation_verdicts.py`), both beside --out.

Shadow only: the report ranks and annotates. Nothing reads it for a score, a
filter or an alert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import long_study_families as lsf  # noqa: E402
import setup_permutations as sp  # noqa: E402

REPORT_SCHEMA = "setup_permutation_report_v1"
SEARCH_VERSION = "setup_permutation_search.v1"
MIN_N = 30
MIN_SESSIONS = 10
HOLDOUT_SESSIONS = 20
HOLDOUT_MIN_N = 10
#: Fewer selection sessions than this and the population x horizon is not published (S4).
MIN_SELECTION_SESSIONS = 20
#: Horizon names that are their own report key; every other horizon is keyed by its session count.
NAMED_HORIZONS = ("bracket_1r",)
M5_HELD30_NAME = "held30"
MAX_DEPTH = 3
BEAM_WIDTH = 10
#: Above this many cells a grid must clear the 99% bound and the grid median (the k > 10 rule).
K_RULE = 10
REJECTED_SHOWN = 5
Z95 = 1.959963984540054
Z99 = 2.5758293035489
AUTHORIZATION = "WISHLIST.md P1-4 4c (trader 2026-09-24: 'keep going in order, finish it all')"

VERDICT_KEY = "key_found"
VERDICT_NONE = "no_key_found"
VERDICT_THIN = "too_little_data"
#: Beside the report, so a scratch --out never writes into the live history.
HISTORY_DIR_NAME = "permutation_report_history"
VERDICTS_FILE_NAME = "permutation_verdicts.json"

Cell = tuple[tuple[str, str], ...]


def horizon_key(row: Mapping[str, Any]) -> str:
    """The report key of a row's horizon: a named one (``bracket_1r``), else its session count."""
    name = str(row.get("horizon_name") or "").strip()
    return name if name in NAMED_HORIZONS else str(int(row.get("horizon") or 0))


def horizon_name(population: str, key: str) -> str:
    """What a horizon key means in words: ``held30``, ``bracket_1r`` or ``<n>_sessions``."""
    text = str(key)
    if text in NAMED_HORIZONS:
        return text
    if population == "m5" and text == "0":
        return M5_HELD30_NAME
    return f"{text}_sessions"


def horizon_sort_key(key: Any) -> tuple[int, int, str]:
    """Session counts first, in order; named horizons after them."""
    text = str(key)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def wilson_lower_bound(wins: int, n: int, z: float = Z95) -> float | None:
    if n <= 0:
        return None
    phat = wins / n
    denominator = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denominator)


@dataclass
class Stats:
    n: int = 0
    wins: int = 0
    sessions: int = 0
    mean_r: float | None = None

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.n if self.n else None

    @property
    def wilson_lb(self) -> float | None:
        return wilson_lower_bound(self.wins, self.n)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "sessions": self.sessions,
            "wins": self.wins,
            "win_rate": _round(self.win_rate),
            "wilson_lb": _round(self.wilson_lb),
            "mean_r": _round(self.mean_r),
        }


def _round(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(float(value), places)


def stats_for(rows: Sequence[Mapping[str, Any]]) -> Stats:
    rs = [float(row["r"]) for row in rows if _finite(row.get("r"))]
    return Stats(
        n=len(rows),
        wins=sum(1 for row in rows if bool(row.get("win"))),
        sessions=len({row.get("session") for row in rows}),
        mean_r=(sum(rs) / len(rs)) if rs else None,
    )


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(number) or math.isinf(number))


def cell_label(cell: Cell) -> str:
    return " + ".join(f"{name}={value}" for name, value in cell)


def facet_names(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("f_") and key[2:] not in names:
                names.append(key[2:])
        break
    return names


# --- grids: counts only, fixed before outcomes


class Index:
    """Row indices per (facet, value), so a cell's members are one set intersection."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> None:
        self.rows = rows
        self.by_value: dict[tuple[str, str], set[int]] = defaultdict(set)
        for index, row in enumerate(rows):
            for name in names:
                value = row.get(f"f_{name}")
                if value and value != sp.UNKNOWN:
                    self.by_value[(name, str(value))].add(index)

    def members(self, cell: Cell) -> list[Mapping[str, Any]]:
        sets = [self.by_value.get(part, set()) for part in cell]
        if not sets:
            return list(self.rows)
        common = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return [self.rows[index] for index in sorted(common)]


def single_cells(index: Index) -> list[Cell]:
    return sorted((part,) for part in index.by_value)


def extend_cells(parents: Sequence[Cell], index: Index) -> list[Cell]:
    cells = set()
    for parent in parents:
        used = {name for name, _value in parent}
        members = set.intersection(*(index.by_value.get(part, set()) for part in parent))
        if not members:
            continue
        for part, rows in index.by_value.items():
            if part[0] not in used and members & rows:
                cells.add(tuple(sorted((*parent, part))))
    return sorted(cells)


def floored(cells: Sequence[Cell], index: Index) -> list[Cell]:
    """Cells meeting n and session floors in the selection rows (counts only)."""
    out = []
    for cell in cells:
        members = index.members(cell)
        if len(members) >= MIN_N and len({row.get("session") for row in members}) >= MIN_SESSIONS:
            out.append(cell)
    return out


# --- the trial ledger


def grid_trial(
    *, population: str, horizon: int | str, family: str, side: str, depth: int, cells: Sequence[Cell],
    selection_window: tuple[str, str], holdout_window: tuple[str, str],
) -> dict[str, Any]:
    payload = json.dumps([list(map(list, cell)) for cell in cells], sort_keys=True)
    digest = hashlib.sha256(
        f"{payload}|{selection_window}|{holdout_window}|{MIN_N}|{MIN_SESSIONS}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "trial_id": f"{SEARCH_VERSION}:{population}:{family}:{side}:h{horizon}:d{depth}:{digest}",
        "family": f"{family}:{side}",
        "question": (
            f"Which depth-{depth} facet key of {family} {side} ({population}, horizon "
            f"{horizon_name(population, str(horizon))}) beats the "
            "family baseline win rate, measured on sessions before the hold-out and confirmed on it?"
        ),
        "failure_mode": "A lucky facet that wins in selection and not on the last 20 sessions.",
        "declared_cells": [cell_label(cell) for cell in cells],
        "declared_cell_count": len(cells),
        "declared_floors": {"min_n": MIN_N, "min_sessions": MIN_SESSIONS, "holdout_min_n": HOLDOUT_MIN_N},
        "declared_window": {
            "kind": "selection_then_holdout",
            "selection": list(selection_window),
            "holdout": list(holdout_window),
            "holdout_sessions": HOLDOUT_SESSIONS,
        },
        "authorization": AUTHORIZATION,
        "analysis_unit": "episode_horizon",
        "status": "registered",
        "outcome": "",
        "registered_by": SEARCH_VERSION,
    }


def register_grid(ledger_root: Path, trial: Mapping[str, Any]) -> str:
    from research_warehouse import trial_ledger

    trial_ledger.register(ledger_root, trial)
    ids = {str(row.get("trial_id") or "") for row in trial_ledger.load(ledger_root)}
    if trial["trial_id"] not in ids:
        raise RuntimeError(f"grid {trial['trial_id']} is not in the ledger; its results stay unread")
    return str(trial["trial_id"])


# --- one family x side x horizon


@dataclass
class CellResult:
    cell: Cell
    depth: int
    selection: Stats
    holdout: Stats
    k: int
    parents: list[Cell] = field(default_factory=list)
    passed: bool = False
    reason: str = ""


def _holdout_pass(result: CellResult, base_sel: Stats, base_hold: Stats, grid_median: float | None,
                  best_parent_holdout: float | None) -> tuple[bool, str]:
    sel_lb = result.selection.wilson_lb
    if sel_lb is None or base_sel.win_rate is None or sel_lb <= base_sel.win_rate:
        return False, "selection: lower bound does not beat the baseline"
    if result.holdout.n < HOLDOUT_MIN_N or base_hold.win_rate is None:
        return False, f"hold-out: under {HOLDOUT_MIN_N} episodes"
    rate = result.holdout.win_rate or 0.0
    if result.k > K_RULE:
        bound = wilson_lower_bound(result.holdout.wins, result.holdout.n, Z99) or 0.0
        if bound <= base_hold.win_rate:
            return False, "hold-out: 99% bound does not beat the hold-out baseline (k > 10)"
        if grid_median is not None and rate <= grid_median:
            return False, "hold-out: does not beat the grid median (k > 10)"
    elif rate <= base_hold.win_rate:
        return False, "hold-out: does not beat the hold-out baseline"
    if best_parent_holdout is not None and rate <= best_parent_holdout:
        return False, "hold-out: does not beat its best parent"
    return True, "passed"


def search_group(
    selection: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    register,
) -> dict[str, Any]:
    """Baseline, then depth 1..3. ``register(depth, cells) -> trial_id`` runs before each depth's outcomes."""
    base_sel, base_hold = stats_for(selection), stats_for(holdout)
    out: dict[str, Any] = {
        "baseline": base_sel.as_dict(),
        "holdout_baseline": base_hold.as_dict(),
        "trial_ids": [],
        "cells_tested": 0,
        "keys": [],
        "rejected": 0,
        "top_rejected": [],
    }
    if base_sel.n < MIN_N or base_sel.sessions < MIN_SESSIONS:
        out["verdict"] = VERDICT_THIN
        return out
    sel_index, hold_index = Index(selection, names), Index(holdout, names)
    results: dict[Cell, CellResult] = {}
    parents: list[Cell] = []
    for depth in range(1, MAX_DEPTH + 1):
        candidates = single_cells(sel_index) if depth == 1 else extend_cells(parents, sel_index)
        grid = floored(candidates, sel_index)
        if not grid:
            break
        out["trial_ids"].append(register(depth, grid))  # registered BEFORE any outcome below is read
        out["cells_tested"] += len(grid)
        level = []
        for cell in grid:
            result = CellResult(
                cell=cell, depth=depth, k=len(grid),
                selection=stats_for(sel_index.members(cell)),
                holdout=stats_for(hold_index.members(cell)),
                parents=[tuple(sub) for sub in combinations(cell, depth - 1)] if depth > 1 else [],
            )
            level.append(result)
        rates = [r.holdout.win_rate for r in level if r.holdout.n >= HOLDOUT_MIN_N and r.holdout.win_rate is not None]
        median = statistics.median(rates) if rates else None
        for result in level:
            parent_rates = [
                results[parent].holdout.win_rate for parent in result.parents
                if parent in results and results[parent].holdout.win_rate is not None
            ]
            result.passed, result.reason = _holdout_pass(
                result, base_sel, base_hold, median, max(parent_rates) if parent_rates else None
            )
            results[result.cell] = result
        ranked = sorted(level, key=lambda r: (r.selection.wilson_lb or 0.0, r.selection.n), reverse=True)
        parents = [r.cell for r in ranked[:BEAM_WIDTH]]
    keys = [r for r in results.values() if r.passed]
    rejected = [r for r in results.values() if not r.passed]
    out["rejected"] = len(rejected)
    # The strongest-looking cells that failed, with why: a lucky facet stays visible, never silent.
    rejected.sort(key=lambda r: (r.selection.wilson_lb or 0.0, r.selection.n), reverse=True)
    out["top_rejected"] = [
        {"label": cell_label(r.cell), "depth": r.depth, "facets": {name: value for name, value in r.cell},
         "selection": r.selection.as_dict(), "holdout": r.holdout.as_dict(), "reason": r.reason}
        for r in rejected[:REJECTED_SHOWN]
    ]
    keys.sort(key=lambda r: (r.selection.wilson_lb or 0.0, r.holdout.win_rate or 0.0), reverse=True)
    for rank, result in enumerate(keys, start=1):
        sel_rate = result.selection.win_rate or 0.0
        out["keys"].append({
            "rank": rank,
            "depth": result.depth,
            "facets": {name: value for name, value in result.cell},
            "label": cell_label(result.cell),
            "selection": result.selection.as_dict(),
            "lift_pp": _round((sel_rate - (base_sel.win_rate or 0.0)) * 100.0, 2),
            "holdout": {**result.holdout.as_dict(), "passed": True, "reason": result.reason},
            "grid_cells": result.k,
        })
    out["verdict"] = VERDICT_KEY if keys else VERDICT_NONE
    return out


# --- the whole report


def split_sessions(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], tuple[str, str], tuple[str, str]]:
    sessions = sorted({str(row.get("session") or "") for row in rows if row.get("session")})
    holdout = set(sessions[-HOLDOUT_SESSIONS:])
    selection = [day for day in sessions if day not in holdout]
    sel_window = (selection[0], selection[-1]) if selection else ("", "")
    hold_sorted = sorted(holdout)
    hold_window = (hold_sorted[0], hold_sorted[-1]) if hold_sorted else ("", "")
    return holdout, sel_window, hold_window


def embargoed_sessions(rows: Sequence[Mapping[str, Any]], holdout: set[str], horizon: int) -> set[str]:
    """Selection sessions whose outcome window (session + horizon) reaches into the hold-out.

    Counted in the population's own scan sessions, so a missing day only makes
    the embargo longer, never shorter. Horizon 0 (M5, same session) embargoes nothing.
    """
    sessions = sorted({str(row.get("session") or "") for row in rows if row.get("session")})
    if not holdout or horizon <= 0:
        return set()
    first_holdout = min(sessions.index(day) for day in holdout if day in sessions)
    return {day for index, day in enumerate(sessions)
            if day not in holdout and index + int(horizon) >= first_holdout}


def build_report(rows: Sequence[Mapping[str, Any]], *, ledger_root: Path, source: str = "") -> dict[str, Any]:
    by_population: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_population[str(row.get("population") or "")][horizon_key(row)].append(row)
    names = facet_names(rows)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "search_version": SEARCH_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "permutation_rule_version": sp.PERMUTATION_RULE_VERSION,
        "source": source,
        "floors": {"min_n": MIN_N, "min_sessions": MIN_SESSIONS, "holdout_sessions": HOLDOUT_SESSIONS,
                   "holdout_min_n": HOLDOUT_MIN_N, "max_depth": MAX_DEPTH, "beam_width": BEAM_WIDTH,
                   "k_rule": K_RULE, "min_selection_sessions": MIN_SELECTION_SESSIONS},
        "note": "Shadow only. Populations are never pooled. Nothing ranks, filters or alerts on this.",
        "populations": {},
    }
    for population in sorted(by_population):
        horizons_out: dict[str, Any] = {}
        for horizon in sorted(by_population[population], key=horizon_sort_key):
            population_rows = by_population[population][horizon]
            name = horizon_name(population, horizon)
            holdout_days, sel_window, hold_window = split_sessions(population_rows)
            # Embargo: a selection row whose outcome is measured inside the hold-out leaks it.
            embargo = embargoed_sessions(population_rows, holdout_days, int(population_rows[0].get("horizon") or 0))
            selection_days = {str(row.get("session") or "") for row in population_rows if row.get("session")}
            selection_days -= holdout_days | embargo
            block: dict[str, Any] = {
                "horizon_name": name,
                "embargoed_sessions": sorted(embargo),
                "selection_window": list(sel_window),
                "selection_sessions": len(selection_days),
                "holdout_window": list(hold_window),
                "families": {},
                "study_families": {},
            }
            horizons_out[str(horizon)] = block
            groups: dict[tuple[str, str], list] = defaultdict(list)
            for row in population_rows:
                groups[(str(row.get("family") or sp.UNKNOWN), str(row.get("side") or sp.UNKNOWN))].append(row)
            refused = len(selection_days) < MIN_SELECTION_SESSIONS
            if refused:
                # S4: a short selection window proves nothing; no grid is registered and no outcome is read.
                block["refused"] = True
                block["refused_reason"] = (
                    f"{population} {name}: selection window has {len(selection_days)} sessions "
                    f"({sel_window[0] or '-'} to {sel_window[1] or '-'}), under {MIN_SELECTION_SESSIONS}; "
                    "not searched, not published"
                )
            families: dict[str, Any] = {}
            study: dict[str, Any] = {}
            study_names = [name for name in lsf.STUDY_SEARCH_FACETS if name in names]
            # S14: the study keys first, each on its own three-facet grid.
            ordered = sorted(groups.items(), key=lambda item: (item[0][0] not in lsf.STUDY_FAMILIES, item[0]))
            for (family, side), members in ordered:
                is_study = family in lsf.STUDY_FAMILIES
                target = study if is_study else families
                selection = [row for row in members
                             if row.get("session") not in holdout_days and row.get("session") not in embargo]
                holdout = [row for row in members if row.get("session") in holdout_days]
                if refused:
                    target[f"{family} {side}"] = {
                        "family": family, "side": side, "horizon_name": name, "verdict": VERDICT_THIN,
                        "refused_reason": block["refused_reason"],
                        "baseline": {"n": len(selection), "sessions": len({r.get("session") for r in selection})},
                        "holdout_baseline": {"n": len(holdout), "sessions": len({r.get("session") for r in holdout})},
                        "trial_ids": [], "cells_tested": 0, "keys": [], "rejected": 0, "top_rejected": [],
                    }
                    continue

                def register(depth, cells, *, _family=family, _side=side, _population=population,
                             _horizon=horizon, _sel_window=sel_window, _hold_window=hold_window):
                    return register_grid(ledger_root, grid_trial(
                        population=_population, horizon=_horizon, family=_family, side=_side, depth=depth,
                        cells=cells, selection_window=_sel_window, holdout_window=_hold_window,
                    ))

                result = search_group(selection, holdout, names=study_names if is_study else names,
                                      register=register)
                for key in result["keys"]:
                    key["horizon_name"] = name
                target[f"{family} {side}"] = {"family": family, "side": side, "horizon_name": name, **result}
                if is_study:
                    target[f"{family} {side}"]["facets_searched"] = list(study_names)
            block["families"] = families
            block["study_families"] = study
        report["populations"][population] = {"horizons": horizons_out}
    data_day = report_data_date(report)
    report["data_date"] = data_day.isoformat() if data_day else ""
    return report


def read_outcomes(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def write_report(report: Mapping[str, Any], out: Path) -> Path:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temp, target)
    return target


# --- report history (P12): one file per data date, read by setup_permutation_verdicts

HISTORY_KEEP_DAYS = 600
_HISTORY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def _content_hash(report: Mapping[str, Any]) -> str:
    """Hash of the report without its run stamp, so a rerun on the same data matches."""
    body = {key: value for key, value in report.items() if key != "generated_at"}
    text = json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def history_files(history_dir: Path) -> list[tuple[date, Path]]:
    """`(data date, path)` for every history file, oldest first."""
    out: list[tuple[date, Path]] = []
    try:
        entries = list(Path(history_dir).iterdir())
    except OSError:
        return out
    for entry in entries:
        match = _HISTORY_FILE_RE.match(entry.name)
        if not match:
            continue
        try:
            out.append((date.fromisoformat(match.group(1)), entry))
        except ValueError:
            continue
    out.sort()
    return out


def report_data_date(report: Mapping[str, Any]) -> date | None:
    """The last session the search used: `data_date`, else the newest hold-out window end."""
    candidates = [str(report.get("data_date") or "")]
    for block in ((report.get("populations") or {}).values()):
        for hz in ((block or {}).get("horizons") or {}).values():
            window = (hz or {}).get("holdout_window") or []
            candidates.append(str(window[-1] if window else ""))
    days = []
    for text in candidates:
        try:
            days.append(date.fromisoformat(text[:10]))
        except ValueError:
            continue
    return max(days) if days else None


def append_history(report: Mapping[str, Any], history_dir: Path, *, today: date | None = None) -> Path | None:
    """Copy the report to `<history_dir>/<data date>.json` unless it matches the newest copy.

    Named by the report's data date (the run date only when it has none), so a
    rerun on the same data REPLACES that date's file instead of adding a report.
    Returns the file written, or None when the content is unchanged. Files older
    than HISTORY_KEEP_DAYS are pruned. Raises on I/O failure; the report itself
    is already written by then.
    """
    folder = Path(history_dir)
    run_day = today or datetime.now().astimezone().date()
    data_day = report_data_date(report) or run_day
    digest = _content_hash(report)
    existing = history_files(folder)
    if existing:
        try:
            newest = json.loads(existing[-1][1].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            newest = None
        if isinstance(newest, dict) and _content_hash(newest) == digest:
            return None
    target = write_report(report, folder / f"{data_day.isoformat()}.json")
    cutoff = run_day - timedelta(days=HISTORY_KEEP_DAYS)
    for day, path in history_files(folder):
        if day >= cutoff:
            break
        path.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--ledger-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--history-dir", type=Path, default=None,
                        help="default: permutation_report_history/ beside --out")
    parser.add_argument("--verdicts-out", type=Path, default=None,
                        help="default: permutation_verdicts.json beside --out")
    args = parser.parse_args(argv)
    rows = read_outcomes(args.outcomes)
    report = build_report(rows, ledger_root=args.ledger_root, source=str(args.outcomes))
    write_report(report, args.out)
    history_dir = args.history_dir or Path(args.out).parent / HISTORY_DIR_NAME
    append_history(report, history_dir)
    # P12: the verdicts over the newest two reports, in the same run (never the Qt thread).
    import setup_permutation_verdicts

    setup_permutation_verdicts.publish(history_dir, args.verdicts_out or Path(args.out).parent / VERDICTS_FILE_NAME)
    found = sum(
        len(fam["keys"])
        for pop in report["populations"].values()
        for hz in pop["horizons"].values()
        for fam in (*hz["families"].values(), *hz.get("study_families", {}).values())
    )
    print(json.dumps({"out": str(args.out), "keys": found}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
