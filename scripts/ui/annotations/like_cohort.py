"""Forward-grading for LIKE claims — R10.F (C1, C3).

Audit C1: **52 `like_claim` rows over 2 sessions, and no `like_cohort_*` file
exists.** The veto trio has graded the trader's rejections forward since the
cohort packet shipped; the other half of the same decision — the charts they
LIKED — has never been graded at all. So "were my vetoes any good?" became
answerable and "were my likes any good?" did not, which is the more interesting
question of the two and the one the trader actually asked for.

This mirrors `veto_cohort` deliberately and closely. The two cohorts are read
side by side, so a difference between them must come from the data and never
from two implementations that drifted apart:

* the same `_pick_key`, so the outcome math agrees;
* the same first-of-the-day rule, so a name claimed twice is graded once;
* the same **sideless refusal** — a claim with no side is counted and named,
  never graded, because `human_focus_tracking` reads a blank side as LONG and
  grading one would manufacture a directional claim the trader never made;
* the same delegate, `update_human_focus_outcomes`, reached through path
  parameters rather than reimplemented.

What differs, and why:

* the cohort **source is the claimed setup id**, not a reason code. A like says
  "I think this is a `post_earnings_52w_break`", so the cohort that means
  anything is the one per claimed family.
* stamps carry **UTC plus an explicit `session_date`** (ground rule 7). The veto
  trio predates that rule and stamps market-local; carrying both here makes the
  ET/PT question moot rather than answered differently in two places.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project_paths import (
    LIKE_COHORT_OUTCOMES_FILE,
    LIKE_COHORT_PERFORMANCE_FILE,
    LIKE_COHORT_PICKS_FILE,
    MASTER_AVWAP_DAILY_BARS_DIR,
    TRADER_ANNOTATIONS_FILE,
)
from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from ui.annotations.store import EVENT_LIKE_CLAIM, like_mode_of, load_annotations

#: The mirror trio, re-exported from `project_paths` where the veto trio also
#: lives. Every reader addresses them by CONSTANT.

#: Cohort namespace. `human_focus_tracking` groups on this prefix.
LIKE_COHORT_PREFIX = "like"

_SIDES = ("LONG", "SHORT")

PICK_COLUMNS = [
    "trade_date",
    "symbol",
    "side",
    "source",
    "snapshotted_at",
    "active_at_snapshot",
    # Ground rule 7: UTC and the market session, both, on every row.
    "claimed_at_utc",
    "session_date",
    # P9: HOW the like was made - `claimed` (Alt+K, a digit, a why) or `quick`
    # (one key, no claim). The COHORT is unchanged: a quick like still grades
    # under `like_unclaimed`, which is where an unnamed like already went. This
    # column is what lets a later rollup split the two WITHOUT rewriting a row,
    # and the split matters because they are different statements: a claimed
    # like says which setup, a quick like says only that something was good.
    "like_mode",
    # P10: WHICH SCREEN the like was made on - the Master AVWAP setups table, the
    # chart-review pane, the Focus panel, the M5 alert bar, or the capture rail
    # itself. Trader, 2026-09-02: a star in Master AVWAP and a like in chart
    # review are the SAME thing, so this is a COLUMN and never a second cohort.
    # It rides here so a later rollup can ask whether the trader judges better
    # from one screen than another - a question the column answers and two
    # cohorts would destroy.
    "surface",
]


def like_cohort_source(setup_id: Any) -> str:
    """`post_earnings_52w_break` -> `like_post_earnings_52w_break`.

    A claim with no setup id becomes `like_unclaimed` rather than being
    dropped: the trader liked the chart and declined to name it, which is a
    real answer and a cohort worth watching on its own.
    """
    claimed = str(setup_id or "").strip().lower()
    return f"{LIKE_COHORT_PREFIX}_{claimed}" if claimed else f"{LIKE_COHORT_PREFIX}_unclaimed"


def _pick_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Matches `human_focus_tracking._pick_key` so the outcome math agrees."""
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
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PICK_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in PICK_COLUMNS})
        import os

        os.replace(tmp, target)
        return True
    except OSError:
        return False


def like_pick_rows(
    annotations: list[dict[str, Any]], *, now: datetime | None = None
) -> tuple[list[dict[str, Any]], int]:
    """(cohort rows, skipped-for-no-side count) from LIKE annotations.

    The sideless refusal is the veto cohort's rule, kept verbatim: a forward
    return is side-adjusted and a blank side reads as LONG downstream, so a
    claim without one is counted and named rather than graded into a direction
    the trader never expressed.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp_local = moment.astimezone().isoformat(timespec="seconds")
    stamp_utc = moment.astimezone(timezone.utc).isoformat(timespec="seconds")

    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped_no_side = 0
    for annotation in annotations:
        if str(annotation.get("event_type") or "") != EVENT_LIKE_CLAIM:
            continue
        symbol = str(annotation.get("symbol") or "").strip().upper()
        side = str(annotation.get("side") or "").strip().upper()
        trade_date = str(annotation.get("session_date") or "").strip()
        if not symbol or not trade_date:
            continue
        if side not in _SIDES:
            skipped_no_side += 1
            continue
        key = (trade_date, symbol, side)
        if key in rows:
            # First claim of the day wins, exactly as the veto cohort does. The
            # annotation log keeps every claim in full; the cohort grades the
            # name once.
            continue
        rows[key] = {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "source": like_cohort_source(annotation.get("claimed_setup_id")),
            "like_mode": like_mode_of(annotation),
            "surface": str(annotation.get("surface") or ""),
            "snapshotted_at": stamp_local,
            "active_at_snapshot": "1",
            "claimed_at_utc": stamp_utc,
            "session_date": trade_date,
        }
    ordered = sorted(
        rows.values(), key=lambda row: (_pick_key(row)[0], _pick_key(row)[2], _pick_key(row)[1])
    )
    return ordered, skipped_no_side


def merge_like_cohort_picks(
    *,
    annotations_path: Path = TRADER_ANNOTATIONS_FILE,
    picks_path: Path = LIKE_COHORT_PICKS_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bring the like cohort file up to date with the annotation log.

    Idempotent: it never rewrites or removes an existing row, so a merge lost
    to a concurrent writer is recovered by the next call rather than gone.

    Called from TWO places since 2026-09-01 - the capture rail at click time
    and the nightly job - so it takes the writer lock the veto merge has always
    taken. Losing the lock returns ``written: False`` and the next call
    recovers, which is the same contract as before and the reason it is safe to
    run at capture time at all.
    """
    annotations = load_annotations(annotations_path, event_types=(EVENT_LIKE_CLAIM,))
    candidates, skipped_no_side = like_pick_rows(annotations, now=now)
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
    merged = sorted(
        by_key.values(), key=lambda row: (_pick_key(row)[0], _pick_key(row)[2], _pick_key(row)[1])
    )
    try:
        with local_writer_lock(lock_key_for_path(Path(picks_path)), timeout_seconds=1.0):
            result["written"] = _write_rows(Path(picks_path), merged)
    except LocalLockUnavailable:
        result["written"] = False
    return result


def update_like_cohort_outcomes(
    *,
    reference_date: Any = None,
    picks_path: Path = LIKE_COHORT_PICKS_FILE,
    outcomes_path: Path = LIKE_COHORT_OUTCOMES_FILE,
    performance_path: Path = LIKE_COHORT_PERFORMANCE_FILE,
    daily_bars_dir: Path = MASTER_AVWAP_DAILY_BARS_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Grade the like cohort forward.

    Delegates to `human_focus_tracking.update_human_focus_outcomes` through its
    path parameters - the same close-to-close, side-adjusted math the focus
    picks and the veto cohort already use. Reimplementing it here is exactly
    how the two cohorts would come to disagree for reasons that had nothing to
    do with the trader's decisions.

    The performance rollup that comes back carries R10.C's robust half, because
    `build_human_focus_performance_rows` routes through `evidence_stats`.
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
