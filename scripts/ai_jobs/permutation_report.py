"""The `permutation_report` night slot: the setup-permutation backfill and search, every night.

Stage 1, deterministic, no model. It copies the live inputs the backfill reads
into a private staging folder (the backfill refuses live paths, and a copy is
one consistent snapshot), then runs `setup_permutation_backfill` and
`setup_permutation_search` as library calls - their statistics, floors,
hold-out and trial-ledger pre-registration untouched - with the live research
lake as the ledger root.

Nothing is published until the search has returned. Then, in order: the
outcomes parquet (`SETUP_PERMUTATION_OUTCOMES_FILE`, which the report's
``source`` names), the report (temp-and-rename), its dated history copy and the
verdicts - the same files the by-hand run wrote. A failed step leaves the last
good report untouched and returns ``failed`` with the reason.

Shadow only: nothing reads the report for a score, a filter or an alert.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from ai_jobs import ledger

_log = logging.getLogger(__name__)

#: Measured 2026-09-30 on scratch copies of the live inputs: copy 3 s, backfill 77 s,
#: search 39 s (2.0 min). The reserve leaves room for a cold disk and a longer ledger.
RESERVE_MINUTES = 10.0
STAGING_PREFIX = "tbv3_permutation_report_"


@dataclass(frozen=True)
class Inputs:
    """The live files the backfill and the search read (never written here)."""

    features: Path
    daily_bars: Path
    m5_outcomes: Path
    m5_candidates: Path
    m5_stamps: Path
    journal_db: Path
    environment: Path
    review_events_dir: Path
    review_events_file: Path
    scan_reports: Path


@dataclass(frozen=True)
class Outputs:
    outcomes: Path
    report: Path
    history_dir: Path
    verdicts: Path


def live_inputs() -> Inputs:
    import project_paths as pp

    return Inputs(
        features=Path(pp.D1_FEATURES_HISTORY_FILE),
        daily_bars=Path(pp.DAILY_BARS_CACHE_DIR),
        m5_outcomes=Path(pp.INTRADAY_BOUNCE_OUTCOMES_FILE),
        m5_candidates=Path(pp.INTRADAY_BOUNCE_CANDIDATES_FILE),
        m5_stamps=Path(pp.M5_SETUP_KEY_STAMPS_FILE),
        journal_db=Path(pp.JOURNAL_DB_FILE),
        environment=Path(pp.D1_ENVIRONMENT_FILE),
        review_events_dir=Path(pp.ALERT_REVIEW_EVENTS_DIR),
        review_events_file=Path(pp.ALERT_REVIEW_EVENTS_FILE),
        # Same folder as `master_avwap_lib.scan_manifest.scan_reports_dir()`, spelled
        # here so no AI job imports a detector package (pinned by the slot test).
        scan_reports=Path(pp.get_diagnostics_dir()) / "scan_reports",
    )


def live_outputs() -> Outputs:
    import project_paths as pp

    return Outputs(
        outcomes=Path(pp.SETUP_PERMUTATION_OUTCOMES_FILE),
        report=Path(pp.SETUP_PERMUTATION_REPORT_FILE),
        history_dir=Path(pp.SETUP_PERMUTATION_REPORT_HISTORY_DIR),
        verdicts=Path(pp.SETUP_PERMUTATION_VERDICTS_FILE),
    )


def ledger_root() -> Path:
    """The live research lake: the trial ledger every grid is registered in first."""
    from research_warehouse import config

    root = config.get_research_store_dir()
    if root is None:
        raise RuntimeError("the research warehouse is not configured, so no grid can be registered")
    return Path(root)


# --- staging: one private copy of every input


def _copy_journal(source: Path, target: Path) -> None:
    """A read-only SQLite backup, so rows still in the WAL are in the copy."""
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def stage_inputs(inputs: Inputs, folder: Path) -> dict[str, Path | None]:
    """Copy what exists into ``folder``; a missing optional input stays None."""
    folder.mkdir(parents=True, exist_ok=True)
    if not inputs.features.is_file():
        raise FileNotFoundError(f"no scan history at {inputs.features}")
    if not (inputs.daily_bars / "SPY.csv").is_file():
        raise FileNotFoundError(f"no SPY daily bars in {inputs.daily_bars}")
    staged: dict[str, Path | None] = {}

    def copy_file(name: str, source: Path) -> None:
        staged[name] = None
        if source.is_file():
            staged[name] = Path(shutil.copy2(source, folder / source.name))

    copy_file("features", inputs.features)
    staged["daily_bars"] = Path(shutil.copytree(inputs.daily_bars, folder / "daily_bars"))
    copy_file("m5_outcomes", inputs.m5_outcomes)
    copy_file("m5_candidates", inputs.m5_candidates)
    copy_file("m5_stamps", inputs.m5_stamps)
    copy_file("environment", inputs.environment)
    staged["journal_db"] = None
    if inputs.journal_db.is_file():
        _copy_journal(inputs.journal_db, folder / "trade_journal.sqlite3")
        staged["journal_db"] = folder / "trade_journal.sqlite3"
    events = folder / "review_events"
    staged["review_events"] = None
    if inputs.review_events_dir.is_dir():
        shutil.copytree(inputs.review_events_dir, events)
        staged["review_events"] = events
    if inputs.review_events_file.is_file():
        events.mkdir(exist_ok=True)
        shutil.copy2(inputs.review_events_file, events / inputs.review_events_file.name)
        staged["review_events"] = events
    staged["scan_reports"] = None
    if inputs.scan_reports.is_dir():
        staged["scan_reports"] = Path(shutil.copytree(inputs.scan_reports, folder / "scan_reports"))
    return staged


# --- the two library calls


def run_backfill(staged: Mapping[str, Path | None], last_completed: date | None) -> tuple[list[dict], dict]:
    """`build_permutation_outcomes` over the staged copies: (rows, counts)."""
    import setup_permutation_backfill as bf

    m5 = staged.get("m5_outcomes")
    stores = bf.ContextStores(reports_dir=staged.get("scan_reports"), review_events=staged.get("review_events"),
                              m5_outcomes=m5, environment=staged.get("environment"))
    bars = staged["daily_bars"]
    result = bf.build_permutation_outcomes(
        staged["features"], daily_bars=bars, m5_outcomes=m5, stores=stores, last_completed=last_completed,
        m5_stamps=staged.get("m5_stamps") if m5 else None,
        m5_candidates=staged.get("m5_candidates") if m5 else None,
        spy_bars=Path(bars) / "SPY.csv", structural_regime=staged.get("journal_db"),
    )
    return result.rows, result.counts


def _publish_file(source: Path, target: Path) -> Path:
    """Copy beside ``target`` then rename over it, so a reader never sees half a file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, target)
    return target


def _count_keys(report: Mapping[str, Any]) -> int:
    return sum(
        len(family.get("keys") or ())
        for population in (report.get("populations") or {}).values()
        for horizon in (population.get("horizons") or {}).values()
        for family in (*(horizon.get("families") or {}).values(), *(horizon.get("study_families") or {}).values())
    )


def _failed(reason: str) -> dict[str, Any]:
    return {"status": ledger.STATUS_FAILED, "model": "", "outputs": [],
            "reason": f"{reason}; the last good permutation report is unchanged"}


def run_permutation_report(
    *,
    session_date: str = "",
    inputs: Inputs | None = None,
    outputs: Outputs | None = None,
    trial_root: Path | None = None,
    staging_parent: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """One night's backfill and search. Never raises; a failure publishes nothing."""
    import setup_permutation_backfill as bf
    import setup_permutation_search as ps

    inputs = inputs or live_inputs()
    outputs = outputs or live_outputs()
    try:
        root = Path(trial_root) if trial_root is not None else ledger_root()
    except Exception as exc:  # noqa: BLE001 - a missing lake is a reason, not a crash
        return _failed(f"no trial ledger: {exc}")
    try:
        last_completed = date.fromisoformat(str(session_date)[:10]) if session_date else None
    except ValueError:
        last_completed = None
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=staging_parent))
    try:
        started = time.perf_counter()
        try:
            staged = stage_inputs(inputs, staging / "in")
        except Exception as exc:  # noqa: BLE001
            return _failed(f"could not copy the inputs: {exc}")
        try:
            rows, counts = run_backfill(staged, last_completed)
            outcomes = bf.write_parquet(rows, staging / "permutation_outcomes.parquet")
        except Exception as exc:  # noqa: BLE001
            _log.warning("Permutation backfill failed.", exc_info=True)
            return _failed(f"the backfill failed: {exc}")
        backfilled = time.perf_counter()
        try:
            import regime_join

            segments = regime_join.read_segments(staged.get("journal_db")) if staged.get("journal_db") else None
            report = ps.build_report(ps.read_outcomes(outcomes), ledger_root=root, source=str(outputs.outcomes),
                                     segments=segments)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Permutation search failed.", exc_info=True)
            return _failed(f"the search failed: {exc}")
        searched = time.perf_counter()
        try:
            _publish_file(outcomes, outputs.outcomes)
            ps.write_report(report, outputs.report)
        except OSError as exc:
            return _failed(f"the report could not be written: {exc}")
        written = [str(outputs.outcomes), str(outputs.report)]
        timing = (f"copy+backfill {backfilled - started:.0f}s, search {searched - backfilled:.0f}s; "
                  f"{counts.get('swing_rows', 0)} swing and {counts.get('m5_rows', 0)} m5 rows")
        summary = f"{_count_keys(report)} key(s), data date {report.get('data_date') or '?'}"
        try:
            history = ps.append_history(report, outputs.history_dir)
            import setup_permutation_verdicts

            setup_permutation_verdicts.publish(outputs.history_dir, outputs.verdicts)
        except Exception as exc:  # noqa: BLE001 - the report itself is already published
            return {"status": ledger.STATUS_DEGRADED, "model": "", "outputs": written,
                    "reason": f"{summary}; report written but history/verdicts failed: {exc}; {timing}"}
        if history is not None:
            written.append(str(history))
        written.append(str(outputs.verdicts))
        return {"status": ledger.STATUS_OK, "model": "", "outputs": written, "reason": f"{summary}; {timing}"}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "Inputs",
    "Outputs",
    "RESERVE_MINUTES",
    "live_inputs",
    "live_outputs",
    "run_backfill",
    "run_permutation_report",
    "stage_inputs",
]
