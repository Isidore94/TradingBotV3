"""Forward tracking for vetoed names, so a veto can be graded later.

A veto says "not for today". Whether that was right is only knowable forward,
and only if the name is being watched after the trader walks away from it.
This module turns veto annotations into pick rows in the human-focus column
schema so the existing outcome math grades them - ``human_focus_veto`` with
one sub-cohort per reason code, e.g. ``human_focus_veto_incoming_trendline``.

CAPTURE-SIDE ONLY (plan.md sec 5, and the packet that built this). Producing
these rows changes nothing the desk decides: no veto mutes an alert, filters a
scan, moves a score, or reorders a queue, and this packet ships no consumer of
the resulting numbers. It exists so the question "are my vetoes any good?"
becomes answerable at all.

Its own files, on purpose. ``human_focus_daily_picks.csv`` is keyed
(trade_date, symbol, side) with no source column, so a veto row for a name
that is also a focus pick that day would collide with the focus row and
suppress it. Veto rows therefore live in ``veto_cohort_picks.csv`` and are
graded into ``veto_cohort_outcomes.csv`` / ``veto_cohort_performance.csv``.
This module is the sole writer of those three files.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from human_focus_tracking import (
    HUMAN_FOCUS_DAILY_PICK_COLUMNS,
    VETO_SOURCE_PREFIX,
)
from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from project_paths import (
    MASTER_AVWAP_DAILY_BARS_DIR,
    TRADER_ANNOTATIONS_FILE,
    VETO_COHORT_OUTCOMES_FILE,
    VETO_COHORT_PERFORMANCE_FILE,
    VETO_COHORT_PICKS_FILE,
)
from ui.annotations.store import EVENT_VETO, load_annotations

_SIDES = ("LONG", "SHORT")


def veto_cohort_source(reason_code: str) -> str:
    """``veto_<reason_code>`` - the cohort a veto with this reason grades in."""
    code = str(reason_code or "").strip().lower()
    if not code:
        raise ValueError("reason_code is required to build a veto cohort source")
    return f"{VETO_SOURCE_PREFIX}_{code}"


def _pick_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Matches human_focus_tracking's key so the outcome math agrees."""
    side = str(row.get("side") or "").strip().upper()
    return (
        str(row.get("trade_date") or "").strip(),
        str(row.get("symbol") or "").strip().upper(),
        "SHORT" if side.startswith("SHORT") else "LONG",
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=HUMAN_FOCUS_DAILY_PICK_COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {column: row.get(column, "") for column in HUMAN_FOCUS_DAILY_PICK_COLUMNS}
                )
        os.replace(tmp, target)
    except OSError:
        return False
    return True


def veto_pick_rows(
    annotations: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """(cohort rows, skipped-for-no-side count) from veto annotations.

    A cohort row needs a side: the forward return is side-adjusted, and
    ``human_focus_tracking`` silently reads a blank side as LONG. Guessing
    would manufacture a directional claim the trader never made, so a veto
    with no side is counted and skipped rather than tracked. The count is
    returned so the caller can say so instead of quietly dropping it.

    First veto of a (date, symbol, side) wins. A second veto that day is still
    recorded in full in the annotation log; the cohort just grades the name
    once, against the reason that made the trader walk away first.
    """
    timestamp = (now or datetime.now()).isoformat(timespec="seconds")
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped_no_side = 0
    for annotation in annotations:
        if str(annotation.get("event_type") or "") != EVENT_VETO:
            continue
        symbol = str(annotation.get("symbol") or "").strip().upper()
        reason = str(annotation.get("reason_code") or "").strip().lower()
        side = str(annotation.get("side") or "").strip().upper()
        trade_date = str(annotation.get("session_date") or "").strip()
        if not symbol or not reason or not trade_date:
            continue
        if side not in _SIDES:
            skipped_no_side += 1
            continue
        key = (trade_date, symbol, side)
        if key in rows:
            continue
        rows[key] = {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "source": veto_cohort_source(reason),
            "snapshotted_at": timestamp,
            "active_at_snapshot": "1",
        }
    ordered = sorted(rows.values(), key=lambda row: (_pick_key(row)[0], _pick_key(row)[2], _pick_key(row)[1]))
    return ordered, skipped_no_side


def merge_veto_cohort_picks(
    *,
    annotations_path: Path = TRADER_ANNOTATIONS_FILE,
    picks_path: Path = VETO_COHORT_PICKS_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bring the veto cohort file up to date with the annotation log.

    Cheap and idempotent by design: it reads a small JSONL, merges what is
    missing, and never rewrites or removes an existing row. Idempotence is
    what makes it safe to call at capture time - a merge lost to a concurrent
    writer is recovered by the next call rather than gone.
    """
    annotations = load_annotations(annotations_path, event_types=(EVENT_VETO,))
    candidates, skipped_no_side = veto_pick_rows(annotations, now=now)
    existing = _read_rows(Path(picks_path))
    by_key = {_pick_key(row): dict(row) for row in existing if _pick_key(row)[1]}
    added = 0
    for row in candidates:
        key = _pick_key(row)
        if key in by_key:
            continue
        by_key[key] = row
        added += 1
    result = {
        "added": added,
        "total_rows": len(by_key),
        "skipped_no_side": skipped_no_side,
        "written": True,
    }
    if not added:
        return result
    merged = sorted(by_key.values(), key=lambda row: (_pick_key(row)[0], _pick_key(row)[2], _pick_key(row)[1]))
    try:
        with local_writer_lock(lock_key_for_path(Path(picks_path)), timeout_seconds=1.0):
            result["written"] = _write_rows(Path(picks_path), merged)
    except LocalLockUnavailable:
        result["written"] = False
    return result


def update_veto_cohort_outcomes(
    *,
    reference_date: Any = None,
    picks_path: Path = VETO_COHORT_PICKS_FILE,
    outcomes_path: Path = VETO_COHORT_OUTCOMES_FILE,
    performance_path: Path = VETO_COHORT_PERFORMANCE_FILE,
    daily_bars_dir: Path = MASTER_AVWAP_DAILY_BARS_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Grade the veto cohort with the existing human-focus outcome math.

    NOT wired to any timer, scan, or market-hours path by this packet. It
    reads daily bars off disk, so it belongs on a runner or an offline call,
    never on the GUI thread. It is here so the evidence is computable the day
    somebody wants it - deciding when to run it is a separate change.
    """
    from human_focus_tracking import update_human_focus_outcomes

    return update_human_focus_outcomes(
        reference_date=reference_date,
        daily_picks_path=Path(picks_path),
        outcomes_path=Path(outcomes_path),
        performance_path=Path(performance_path),
        daily_bars_dir=Path(daily_bars_dir),
        now=now,
    )
