"""Forward-grading for NOT-TODAY and DISLIKE — P5.

The other three verdicts are already graded forward. The veto cohort answers
"was I right to throw that chart away", the like cohort "was I right to endorse
it", and the pass cohort "did the one issue I passed on matter". These two were
never graded at all:

* **not_today** — the trader throwing back ONE auto-adopted pick for ONE
  session. 223 of them on the live log.
* **dislike** — the name itself, not the day. 34 of them, and they carry the
  most information-dense free text the trader ever writes.

`pick_feedback.py` has kept them distinct on purpose since packet R2, and this
module keeps them distinct too: **they are separate cohorts and are never
pooled**, because a same-day pass and a judgement on the name are different
claims and averaging them would teach the wrong lesson about both.

Two deliberate refusals.

**`unfavorite` is not graded here.** It is a membership change - a name coming
off a list - not a verdict about the setup, and the live rows carry no side at
all. Grading a sideless row would manufacture a direction the trader never
expressed.

**The free-text `reason` is carried and never coded.** It rides on the picks
CSV as written, because the whole value of those 34 dislikes is the sentence.
Turning them into machine categories is a separate decision with a vocabulary
behind it, and inventing one here would destroy the thing worth reading.

The grading itself goes through `human_focus_tracking.update_human_focus_outcomes`
- the one owner of the forward-return arithmetic, the maturity rule and the
performance rollup - exactly as the veto, like and pass cohorts do.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from project_paths import (
    MASTER_AVWAP_DAILY_BARS_DIR,
    PICK_FEEDBACK_FILE,
    REJECTION_COHORT_OUTCOMES_FILE,
    REJECTION_COHORT_PERFORMANCE_FILE,
    REJECTION_COHORT_PICKS_FILE,
)

_log = logging.getLogger(__name__)

#: Cohort namespace. The DOUBLE underscore is load-bearing:
#: `human_focus_tracking.COHORT_BASE_BY_SOURCE_PREFIX` claims these with the
#: prefix `focus_`, which matches `source.startswith("focus__")` and therefore
#: cannot reach `focus_swing`, `focus_m5` or `focus_pick`.
REJECTION_COHORT_PREFIX = "focus_"

#: The verdicts graded here, and the only ones. `like` belongs to the like
#: cohort; `unfavorite` is a membership change, not a verdict, and carries no
#: side on the live log.
GRADED_VERDICTS = ("not_today", "dislike")

#: The lanes those verdicts actually arrive on, measured 2026-09-02 on the live
#: log: `not_today` is M5 (223 rows) and `dislike` is swing (34). Recorded here
#: because it is the reason the category belongs in the cohort name, not a
#: constraint - a `swing/not_today` row would simply grade under its own name.
OBSERVED_LANES = {"not_today": "m5", "dislike": "swing"}

_SIDES = ("LONG", "SHORT")

PICK_COLUMNS = [
    "trade_date",
    "symbol",
    "side",
    "source",
    "snapshotted_at",
    "active_at_snapshot",
    # Ground rule 7: UTC and the market session, both, on every row.
    "recorded_at_utc",
    "session_date",
    "verdict",
    "category",
    "origin",
    # The trader's own words, carried verbatim and NEVER coded by machine.
    "reason",
]


def rejection_cohort_source(verdict: Any, category: Any = "") -> str:
    """`(not_today, m5)` -> `focus__m5_not_today`. Never pooled with `dislike`.

    THE CATEGORY IS PART OF THE IDENTITY (R1). Dropping it made a false claim
    about the live log: `not_today` is recorded on M5 rows (223) and `dislike`
    on swing rows (34), so a cohort named for the verdict alone reads as
    "the trader's not-today record" when it is really "the trader's INTRADAY
    not-today record". A cohort that names the wrong population cannot be
    compared with anything, and rows are never rewritten, so the name has to be
    right the first time it is written.

    The DOUBLE underscore stays immediately after the prefix -
    `focus__m5_not_today`, not `focus_m5__not_today` - because
    `COHORT_BASE_BY_SOURCE_PREFIX` matches `startswith("focus_" + "_")`. The
    second spelling would fall through to `focus_m5` if that prefix ever
    existed, and silently grade these rows as somebody else's cohort.
    """
    text = str(verdict or "").strip().lower() or "unstated"
    lane = str(category or "").strip().lower() or "unstated"
    return f"{REJECTION_COHORT_PREFIX}_{lane}_{text}"


def _pick_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Matches `human_focus_tracking.pick_key_with_source`.

    The source is part of the identity: a name can be thrown back for the day
    AND disliked outright, and those are two verdicts about one name that must
    not collapse into one row.
    """
    side = str(row.get("side") or "").strip().upper()
    return (
        str(row.get("trade_date") or "").strip(),
        str(row.get("symbol") or "").strip().upper(),
        "SHORT" if side.startswith("SHORT") else "LONG",
        str(row.get("source") or "").strip(),
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
    import os

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PICK_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in PICK_COLUMNS})
        os.replace(tmp, target)
        return True
    except OSError:
        return False


def load_rejection_feedback(path: Path = PICK_FEEDBACK_FILE) -> list[dict[str, Any]]:
    """Every `not_today` / `dislike` row in the feedback log.

    Read here rather than through `pick_feedback.latest_like_origins`, which
    filters to likes by design. A corrupt line is skipped, never fatal: one bad
    row must not make the rest of the record unreadable.
    """
    rows: list[dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return rows
    try:
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and str(row.get("verdict") or "") in GRADED_VERDICTS:
                    rows.append(row)
    except OSError as exc:
        _log.debug("Pick feedback unreadable for the rejection cohort: %s", exc)
    return rows


def rejection_pick_rows(
    feedback_rows: list[dict[str, Any]], *, now: datetime | None = None
) -> tuple[list[dict[str, Any]], int]:
    """(cohort rows, skipped-for-no-side count) from feedback verdicts.

    Dated by `trade_date` - the session the verdict is ABOUT - never by `ts`,
    which is when it was typed. A verdict entered on Saturday about Friday
    belongs to Friday, and every other reader of this log already agrees.

    The sideless refusal is the other cohorts' rule, kept verbatim.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp_local = moment.astimezone().isoformat(timespec="seconds")
    stamp_utc = moment.astimezone(timezone.utc).isoformat(timespec="seconds")

    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    skipped_no_side = 0
    for entry in feedback_rows:
        verdict = str(entry.get("verdict") or "").strip().lower()
        if verdict not in GRADED_VERDICTS:
            continue
        symbol = str(entry.get("symbol") or "").strip().upper()
        trade_date = str(entry.get("trade_date") or "").strip()[:10]
        side = str(entry.get("side") or "").strip().upper()
        if not symbol or not trade_date:
            continue
        if side not in _SIDES:
            skipped_no_side += 1
            continue
        source = rejection_cohort_source(verdict, entry.get("category"))
        key = (trade_date, symbol, side, source)
        if key in rows:
            # First verdict of the day wins, as in every sibling cohort: the
            # log keeps them all and the cohort grades the name once.
            continue
        rows[key] = {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "source": source,
            "snapshotted_at": stamp_local,
            "active_at_snapshot": "1",
            "recorded_at_utc": stamp_utc,
            "session_date": trade_date,
            "verdict": verdict,
            "category": str(entry.get("category") or ""),
            "origin": str(entry.get("origin") or ""),
            "reason": str(entry.get("reason") or ""),
        }
    return list(rows.values()), skipped_no_side


def merge_rejection_cohort_picks(
    *,
    feedback_path: Path = PICK_FEEDBACK_FILE,
    picks_path: Path = REJECTION_COHORT_PICKS_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bring the rejection cohort file up to date with the feedback log.

    Idempotent: never rewrites or removes an existing row, so a merge lost to a
    concurrent writer is recovered by the next call rather than gone.
    """
    candidates, skipped_no_side = rejection_pick_rows(
        load_rejection_feedback(Path(feedback_path)), now=now
    )
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
    merged = sorted(by_key.values(), key=_pick_key)
    try:
        with local_writer_lock(lock_key_for_path(Path(picks_path)), timeout_seconds=1.0):
            result["written"] = _write_rows(Path(picks_path), merged)
    except LocalLockUnavailable:
        result["written"] = False
    return result


def update_rejection_cohort_outcomes(
    *,
    reference_date: Any = None,
    picks_path: Path = REJECTION_COHORT_PICKS_FILE,
    outcomes_path: Path = REJECTION_COHORT_OUTCOMES_FILE,
    performance_path: Path = REJECTION_COHORT_PERFORMANCE_FILE,
    daily_bars_dir: Path = MASTER_AVWAP_DAILY_BARS_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Grade the rejection cohort forward through the ONE existing grader."""
    from human_focus_tracking import pick_key_with_source, update_human_focus_outcomes

    return update_human_focus_outcomes(
        reference_date=reference_date,
        daily_picks_path=Path(picks_path),
        outcomes_path=Path(outcomes_path),
        performance_path=Path(performance_path),
        daily_bars_dir=Path(daily_bars_dir),
        now=now,
        pick_key=pick_key_with_source,
    )


__all__ = [
    "GRADED_VERDICTS",
    "PICK_COLUMNS",
    "REJECTION_COHORT_PREFIX",
    "load_rejection_feedback",
    "merge_rejection_cohort_picks",
    "rejection_cohort_source",
    "rejection_pick_rows",
    "update_rejection_cohort_outcomes",
]
