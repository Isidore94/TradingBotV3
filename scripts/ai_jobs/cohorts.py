"""Nightly grading of the veto cohort.

``update_veto_cohort_outcomes`` has existed since the cohort packet shipped and
had **zero callers** — the picks accumulated on every veto commit and nothing
ever graded them, so "are my vetoes any good?" stayed computable-but-unanswered.
This is the caller.

Deterministic, not a model job. It reads daily bars off disk and does
close-to-close side-adjusted return math; no LLM is involved, nothing is sent
anywhere, and the output is two CSVs. It lives in ``ai_jobs`` because that is
where the desk's overnight slate runs, not because it is an AI job.

**It informs nothing the desk decides.** No score, no watchlist, no Focus
entry, no alert, no queue order, and nothing in ``review_policy.json`` reads
these files. The cohort exists so a question becomes answerable.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

#: Sides the outcome math can grade. Matches ``veto_cohort._SIDES``.
GRADEABLE_SIDES = ("LONG", "SHORT")


def _read_pick_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        _log.debug("Veto cohort picks unreadable at %s.", path, exc_info=True)
        return []


def partition_by_gradeable_side(
    rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(gradeable, ungradeable) — split on an EXPLICIT side, never a guess.

    This is the whole reason the job does not simply hand the picks file to
    the outcome math. ``human_focus_tracking._side_label`` reads anything that
    is not "SHORT..." as LONG, blank included, so a row with no side would be
    graded as a long — manufacturing a directional claim the trader never made
    and folding a fabricated return into a cohort average.

    ``veto_pick_rows`` already refuses to WRITE a sideless row, so in a healthy
    file this partition finds nothing. A row here therefore means legacy data
    or a hand edit, which is exactly when silently defaulting would be worst.
    """
    gradeable: list[dict[str, Any]] = []
    ungradeable: list[dict[str, Any]] = []
    for row in rows:
        side = str(row.get("side") or "").strip().upper()
        (gradeable if side in GRADEABLE_SIDES else ungradeable).append(row)
    return gradeable, ungradeable


def run_veto_cohort_grading(
    *,
    session_date: str = "",
    now: datetime | None = None,
    picks_path: Path | None = None,
    outcomes_path: Path | None = None,
    performance_path: Path | None = None,
    daily_bars_dir: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Grade every gradeable veto pick. Idempotent; never destructive.

    Idempotent because the underlying math is: a fully matured pick is skipped
    on re-read, and both CSVs are rewritten in full from the merged set rather
    than appended to. Running twice in one night produces identical files.

    A failure leaves the previous ``veto_cohort_outcomes.csv`` and
    ``veto_cohort_performance.csv`` exactly as they were —
    ``human_focus_tracking._write_csv_rows`` stages to ``<name>.tmp`` and
    ``os.replace``s, and swallows OSError, so a half-written file can never
    land. An unreadable bar store simply produces no new outcome rows.
    """
    from project_paths import (
        MASTER_AVWAP_DAILY_BARS_DIR,
        VETO_COHORT_OUTCOMES_FILE,
        VETO_COHORT_PERFORMANCE_FILE,
        VETO_COHORT_PICKS_FILE,
    )
    from ui.annotations.veto_cohort import update_veto_cohort_outcomes

    picks = Path(picks_path or VETO_COHORT_PICKS_FILE)
    outcomes = Path(outcomes_path or VETO_COHORT_OUTCOMES_FILE)
    performance = Path(performance_path or VETO_COHORT_PERFORMANCE_FILE)
    bars_dir = Path(daily_bars_dir or MASTER_AVWAP_DAILY_BARS_DIR)

    rows = _read_pick_rows(picks)
    if not rows:
        return {
            "status": "skipped",
            "reason": f"no veto cohort picks yet at {picks.name}",
            "picks": 0,
        }

    gradeable, ungradeable = partition_by_gradeable_side(rows)
    if not gradeable:
        return {
            "status": "skipped",
            "reason": (
                f"{len(ungradeable)} veto pick(s) carry no side and none can be "
                "graded; a side is never assumed"
            ),
            "picks": len(rows),
            "skipped_no_side": len(ungradeable),
        }

    staged: Path | None = None
    try:
        source = picks
        if ungradeable:
            # Only sideless rows force a staged copy, so the healthy path
            # touches no extra file. The copy is what keeps the guarantee
            # honest: the outcome math never sees a row it would guess about.
            staged = picks.with_name(picks.name + ".gradeable.tmp")
            _write_pick_subset(staged, rows[0].keys(), gradeable)
            source = staged
        result = update_veto_cohort_outcomes(
            reference_date=None,
            picks_path=source,
            outcomes_path=outcomes,
            performance_path=performance,
            daily_bars_dir=bars_dir,
            now=now,
        )
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                _log.debug("Could not remove %s.", staged, exc_info=True)

    reason = (
        f"graded {len(gradeable)} pick(s); "
        f"{result.get('updated_outcomes', 0)} outcome row(s) updated, "
        f"{result.get('performance_rows', 0)} cohort(s)"
    )
    if ungradeable:
        # Counted and named, never graded and never silently dropped.
        reason += f"; {len(ungradeable)} skipped for no side"
    return {
        "status": "ok",
        "reason": reason,
        "picks": len(rows),
        "graded": len(gradeable),
        "skipped_no_side": len(ungradeable),
        "outcome_rows": result.get("outcome_rows", 0),
        "updated_outcomes": result.get("updated_outcomes", 0),
        "performance_rows": result.get("performance_rows", 0),
        "outputs": [str(outcomes), str(performance)],
    }


def run_like_cohort_grading(
    *,
    session_date: str = "",
    now: datetime | None = None,
    picks_path: Path | None = None,
    outcomes_path: Path | None = None,
    performance_path: Path | None = None,
    daily_bars_dir: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Grade every gradeable LIKE claim. R10.F (C1, C3).

    The mirror of :func:`run_veto_cohort_grading`, and deliberately the same
    shape: audit C1 found 52 `like_claim` rows over 2 sessions and **no**
    `like_cohort_*` file, so the trader's rejections had a forward record and
    their endorsements did not.

    It merges the picks from the annotation log first, because unlike the veto
    cohort - which writes a pick row at capture time - nothing has ever written
    a like pick. The first run therefore grades the whole history retroactively.

    Deterministic and idempotent: no model is called, and running twice in one
    night produces identical files.
    """
    from ui.annotations.like_cohort import (
        LIKE_COHORT_OUTCOMES_FILE,
        LIKE_COHORT_PERFORMANCE_FILE,
        LIKE_COHORT_PICKS_FILE,
        merge_like_cohort_picks,
        update_like_cohort_outcomes,
    )
    from project_paths import MASTER_AVWAP_DAILY_BARS_DIR

    picks = Path(picks_path or LIKE_COHORT_PICKS_FILE)
    outcomes = Path(outcomes_path or LIKE_COHORT_OUTCOMES_FILE)
    performance = Path(performance_path or LIKE_COHORT_PERFORMANCE_FILE)
    bars_dir = Path(daily_bars_dir or MASTER_AVWAP_DAILY_BARS_DIR)

    merged = merge_like_cohort_picks(picks_path=picks, now=now)
    rows = _read_pick_rows(picks)
    if not rows:
        return {
            "status": "skipped",
            "reason": (
                f"no like cohort picks yet at {picks.name}"
                + (
                    f"; {merged['skipped_no_side']} claim(s) carry no side"
                    if merged.get("skipped_no_side")
                    else ""
                )
            ),
            "picks": 0,
            "skipped_no_side": merged.get("skipped_no_side", 0),
        }

    gradeable, ungradeable = partition_by_gradeable_side(rows)
    if not gradeable:
        return {
            "status": "skipped",
            "reason": (
                f"{len(ungradeable)} like pick(s) carry no side and none can be "
                "graded; a side is never assumed"
            ),
            "picks": len(rows),
            "skipped_no_side": len(ungradeable),
        }

    staged: Path | None = None
    try:
        source = picks
        if ungradeable:
            staged = picks.with_name(picks.name + ".gradeable.tmp")
            _write_pick_subset(staged, rows[0].keys(), gradeable)
            source = staged
        result = update_like_cohort_outcomes(
            reference_date=None,
            picks_path=source,
            outcomes_path=outcomes,
            performance_path=performance,
            daily_bars_dir=bars_dir,
            now=now,
        )
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                _log.debug("Could not remove %s.", staged, exc_info=True)

    reason = (
        f"merged {merged.get('added', 0)} new claim(s); graded {len(gradeable)} pick(s); "
        f"{result.get('updated_outcomes', 0)} outcome row(s) updated, "
        f"{result.get('performance_rows', 0)} cohort(s)"
    )
    if ungradeable or merged.get("skipped_no_side"):
        # Counted and named, never graded and never silently dropped.
        reason += (
            f"; {len(ungradeable) + merged.get('skipped_no_side', 0)} skipped for no side"
        )
    return {
        "status": "ok",
        "reason": reason,
        "picks": len(rows),
        "graded": len(gradeable),
        "merged": merged.get("added", 0),
        "skipped_no_side": len(ungradeable) + merged.get("skipped_no_side", 0),
        "outcome_rows": result.get("outcome_rows", 0),
        "updated_outcomes": result.get("updated_outcomes", 0),
        "performance_rows": result.get("performance_rows", 0),
        "outputs": [str(outcomes), str(performance)],
    }


def _write_pick_subset(path: Path, columns: Any, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def _run_cohort_grading(
    *,
    label: str,
    merge,
    grade,
    picks: Path,
    outcomes: Path,
    performance: Path,
    bars_dir: Path,
    now,
) -> dict[str, Any]:
    """The shape both P5 cohorts share with the veto and like slots.

    Merge first, then grade only the rows that carry a side. A pick with no
    side is COUNTED AND NAMED, never graded: forward returns are side-adjusted
    and a blank side reads as LONG downstream, so grading one would manufacture
    a direction the trader never expressed.

    Deterministic and idempotent - no model is called, and running twice in one
    night produces identical files.
    """
    merged = merge(picks_path=picks, now=now)
    rows = _read_pick_rows(picks)
    if not rows:
        return {
            "status": "skipped",
            "reason": (
                f"no {label} cohort picks yet at {picks.name}"
                + (
                    f"; {merged['skipped_no_side']} row(s) carry no side"
                    if merged.get("skipped_no_side")
                    else ""
                )
            ),
            "picks": 0,
            "skipped_no_side": merged.get("skipped_no_side", 0),
        }

    gradeable, ungradeable = partition_by_gradeable_side(rows)
    if not gradeable:
        return {
            "status": "skipped",
            "reason": (
                f"{len(ungradeable)} {label} pick(s) carry no side and none can "
                "be graded; a side is never assumed"
            ),
            "picks": len(rows),
            "skipped_no_side": len(ungradeable),
        }

    staged = None
    try:
        source = picks
        if ungradeable:
            staged = picks.with_name(picks.name + ".gradeable.tmp")
            _write_pick_subset(staged, rows[0].keys(), gradeable)
            source = staged
        result = grade(
            reference_date=None,
            picks_path=source,
            outcomes_path=outcomes,
            performance_path=performance,
            daily_bars_dir=bars_dir,
            now=now,
        )
    finally:
        if staged is not None:
            try:
                staged.unlink()
            except OSError:
                pass

    return {
        "status": "ok",
        "picks": len(rows),
        "added": merged.get("added", 0),
        "skipped_no_side": len(ungradeable),
        "outcome_rows": result.get("outcome_rows", 0),
        "performance_rows": result.get("performance_rows", 0),
        "reason": (
            f"{len(gradeable)} {label} pick(s) graded; "
            f"{result.get('performance_rows', 0)} performance row(s)"
            + (f"; {len(ungradeable)} skipped for no side" if ungradeable else "")
        ),
    }


def run_pass_cohort_grading(
    *,
    now=None,
    picks_path=None,
    outcomes_path=None,
    performance_path=None,
    daily_bars_dir=None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Grade every day-trade PASS forward. P5.

    The third verdict. The veto cohort grades what was thrown away and the like
    cohort what was endorsed; a pass is neither - it is "I like this name but
    not this setup" - and until now nothing measured whether the one issue the
    trader passed on actually mattered.

    A pass with k reason codes produces k+1 rows: one per code and one pooled
    `pass_all`. The code cohorts therefore OVERLAP and must never be summed;
    only `pass_all` counts passes. See `ui.annotations.pass_cohort`.
    """
    from project_paths import MASTER_AVWAP_DAILY_BARS_DIR
    from ui.annotations.pass_cohort import (
        PASS_COHORT_OUTCOMES_FILE,
        PASS_COHORT_PERFORMANCE_FILE,
        PASS_COHORT_PICKS_FILE,
        merge_pass_cohort_picks,
        update_pass_cohort_outcomes,
    )

    return _run_cohort_grading(
        label="pass",
        merge=merge_pass_cohort_picks,
        grade=update_pass_cohort_outcomes,
        picks=Path(picks_path or PASS_COHORT_PICKS_FILE),
        outcomes=Path(outcomes_path or PASS_COHORT_OUTCOMES_FILE),
        performance=Path(performance_path or PASS_COHORT_PERFORMANCE_FILE),
        bars_dir=Path(daily_bars_dir or MASTER_AVWAP_DAILY_BARS_DIR),
        now=now,
    )


def run_rejection_cohort_grading(
    *,
    now=None,
    picks_path=None,
    outcomes_path=None,
    performance_path=None,
    daily_bars_dir=None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Grade NOT-TODAY and DISLIKE forward. P5.

    The fourth and fifth verdicts, and the last two that had no forward record
    at all: 223 not-todays and 34 dislikes on the live log. They are separate
    cohorts whose numbers are never combined into a verdict - a same-day
    throwback and a judgement on the name are different claims. The family's
    BASE row does pool both (every cohort family gets one) and must not be read
    as either verdict; the Weekend Prep table labels it. See `rejection_cohort`.
    """
    from project_paths import (
        MASTER_AVWAP_DAILY_BARS_DIR,
        REJECTION_COHORT_OUTCOMES_FILE,
        REJECTION_COHORT_PERFORMANCE_FILE,
        REJECTION_COHORT_PICKS_FILE,
    )
    from rejection_cohort import (
        merge_rejection_cohort_picks,
        update_rejection_cohort_outcomes,
    )

    return _run_cohort_grading(
        label="rejection",
        merge=merge_rejection_cohort_picks,
        grade=update_rejection_cohort_outcomes,
        picks=Path(picks_path or REJECTION_COHORT_PICKS_FILE),
        outcomes=Path(outcomes_path or REJECTION_COHORT_OUTCOMES_FILE),
        performance=Path(performance_path or REJECTION_COHORT_PERFORMANCE_FILE),
        bars_dir=Path(daily_bars_dir or MASTER_AVWAP_DAILY_BARS_DIR),
        now=now,
    )

