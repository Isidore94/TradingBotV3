"""Forward-grading for day-trade PASSES — P5.

A pass is the verdict the trader described on 2026-08-31: *"I really like this
stock for a daytrade but it has this ONE issue"*. It is not a veto — the veto
cohort already grades what was thrown away — and it is not a like. Its own
vocabulary family (`pass_reasons_v*.json`) exists precisely so the two are
never folded together, and this module is what makes the third verdict
answerable: **did the issue I passed on actually matter?**

Three things shape it, and each is a rule rather than a preference.

**A pass is MULTI-SELECT, so one pass grades in several cohorts.** A pass with
k reason codes is written into k code cohorts (`pass_v<version>_<code>`) AND
into the pooled `pass_all`. That means:

    THE CODE COHORTS ARE NOT INDEPENDENT AND MUST NEVER BE SUMMED.

Their sample counts overlap by construction, `pass_all` is the only row whose
n is a count of passes, and the performance CSV says so in its own column so a
reader who never opens this file still cannot add them up by accident.

**Cohort identity on write is (vocab_version, reason_code)**, exactly as the
veto cohort does it. A code is stable within one vocabulary; the guarantee it
never changes meaning is a rule in the vocabulary file, not something this
module can verify, and the cost of trusting it wrongly is two judgements
averaged into one number. Rows are never rewritten, and pooling equivalent
definitions is a rollup-time reading, never a write-time decision.

**The day-trade question is intraday, so the daily grades are not enough.**
Beside the h1/h3/h5/h10 forward sessions every focus cohort gets, a pass with
an M5 bar sidecar also carries the SAME-SESSION result: entry at the first
completed M5 close after the pass, stop at the session extreme on the pass
side, target 2R, stop-first on an ambiguous bar. Without a sidecar those
columns are BLANK — never zero, because a pass the desk held no bars for is an
unmeasured one, and a zero would read as "it went nowhere".
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from project_paths import (
    MASTER_AVWAP_DAILY_BARS_DIR,
    PASS_COHORT_OUTCOMES_FILE,
    PASS_COHORT_PERFORMANCE_FILE,
    PASS_COHORT_PICKS_FILE,
    TRADER_ANNOTATIONS_FILE,
)
from ui.annotations.store import EVENT_PASS, load_annotations

_log = logging.getLogger(__name__)

#: Cohort namespace. `human_focus_tracking` groups on this prefix.
PASS_COHORT_PREFIX = "pass"
#: The pooled cohort every pass joins, whatever its codes. The ONLY row in the
#: family whose `n` is a count of passes rather than of (pass, code) pairs.
PASS_ALL_SOURCE = f"{PASS_COHORT_PREFIX}_all"

_SIDES = ("LONG", "SHORT")

#: Same 2R target and stop-first convention the setup scoreboard grades by.
INTRADAY_TARGET_R = 2.0

PICK_COLUMNS = [
    "trade_date",
    "symbol",
    "side",
    "source",
    "snapshotted_at",
    "active_at_snapshot",
    # Ground rule 7: UTC and the market session, both, on every row.
    "passed_at_utc",
    "session_date",
    # What this row is, said on the row: `pass_all` is one pass, a code row is
    # one (pass, code) pair, and `reason_code_count` is how many cohorts this
    # single pass entered.
    "reason_code",
    "reason_code_count",
    "vocab_version",
    # The same-session grade, when the desk was holding bars. BLANK otherwise.
    "intraday_entry_at",
    "intraday_entry_price",
    "intraday_stop_price",
    "intraday_risk_per_share",
    "intraday_first_hit",
    "intraday_close_r",
    "intraday_bar_count",
    # WHY an intraday grade is missing. "No sidecar" and "a sidecar that
    # cannot reach the entry bar" are different facts, and a single blank
    # would read as the first when today it is always the second.
    "intraday_unmeasured_reason",
]

#: Printed into the performance CSV so the overlap travels with the numbers.
OVERLAP_NOTE = (
    "A pass with k reason codes appears in k code cohorts AND in pass_all, so "
    "the code cohorts OVERLAP and must never be summed. Only pass_all's n is a "
    "count of passes."
)


def pass_cohort_source(reason_code: Any, vocab_version: Any = None) -> str:
    """``pass_v<version>_<code>`` — the cohort one coded pass grades in.

    Identity is (vocab_version, reason_code), mirroring the veto cohort. With
    no version the historical unversioned form is returned, which is what keeps
    any row already on disk grading where it was filed; rows are never
    rewritten.
    """
    code = str(reason_code or "").strip().lower()
    if not code:
        return f"{PASS_COHORT_PREFIX}_uncoded"
    try:
        version = int(vocab_version)
    except (TypeError, ValueError):
        return f"{PASS_COHORT_PREFIX}_{code}"
    return f"{PASS_COHORT_PREFIX}_v{version}_{code}"


def _pick_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Matches `human_focus_tracking.pick_key_with_source`.

    The SOURCE is part of the identity here, unlike the veto and like cohorts:
    one pass legitimately produces several rows for one (date, symbol, side),
    and without the source they would collapse into one and k of the k+1
    cohorts would vanish.
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


# ---------------------------------------------------------------------------
# the same-session grade
# ---------------------------------------------------------------------------
def _bar_time(bar) -> datetime | None:
    value = bar.get("dt") if hasattr(bar, "get") else None
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _after(bar_stamp: datetime, passed_at: datetime) -> bool:
    """Is this bar at or after the pass? Compared on ONE clock.

    The sidecar stores bar times NAIVE (they are market-local by construction)
    while `created_at` carries an offset. CLAUDE.md's rule for this seam:
    normalize by ATTACHING market-local to the naive side, NEVER by stripping
    the aware side - dropping an offset silently reinterprets the timestamp and
    would pick the wrong entry bar by whole hours.
    """
    if bar_stamp.tzinfo is None and passed_at.tzinfo is not None:
        bar_stamp = bar_stamp.replace(tzinfo=passed_at.tzinfo)
    elif bar_stamp.tzinfo is not None and passed_at.tzinfo is None:
        passed_at = passed_at.replace(tzinfo=bar_stamp.tzinfo)
    return bar_stamp >= passed_at


def intraday_pass_outcome(bars, *, side: str, passed_at: datetime | None) -> dict[str, Any]:
    """The same-session result of the day trade the trader passed on.

    Entry is the FIRST COMPLETED M5 CLOSE AFTER the pass, never the bar the
    pass happened inside: the trader could not have traded a bar that had not
    finished, and entering on it would be reading a price they never saw.

    The stop is the session extreme on the pass side taken from the bars UP TO
    AND INCLUDING the entry bar - the low for a long, the high for a short -
    which is the level a trader watching that chart had in front of them. The
    target is 2R. A bar that touches both is scored STOP FIRST, the same
    convention the warehouse and the setup scoreboard use: assuming the good
    fill on an ambiguous bar is how a backtest flatters itself.

    Empty when the bars cannot answer it. Never a zero.
    """
    def blank(reason: str) -> dict[str, Any]:
        return {
            "intraday_entry_at": "",
            "intraday_entry_price": "",
            "intraday_stop_price": "",
            "intraday_risk_per_share": "",
            "intraday_first_hit": "",
            "intraday_close_r": "",
            "intraday_bar_count": "",
            "intraday_unmeasured_reason": reason,
        }

    rows = [bar for bar in (bars or ()) if hasattr(bar, "get")]
    if not rows:
        return blank("no_sidecar_bars")
    is_long = str(side or "").strip().upper() != "SHORT"

    entry_index = None
    for index, bar in enumerate(rows):
        stamp = _bar_time(bar)
        if stamp is None:
            continue
        if passed_at is None or _after(stamp, passed_at):
            entry_index = index
            break
    if entry_index is None:
        # MEASURED on the live desk, 2026-09-01: the sidecar is written from
        # the bars the desk was ALREADY HOLDING when the pass was recorded, so
        # by construction every bar in it starts before the pass. The entry bar
        # this rule asks for - the first completed close AFTER the pass - is
        # therefore never inside it, and the honest answer today is "not
        # measured", said out loud rather than left as an ambiguous blank.
        #
        # Whether the entry should instead be the last completed close AT the
        # pass - the price the trader was actually looking at when they passed -
        # is a definition change, and the trader's to make.
        return blank("sidecar_ends_before_the_entry_bar")

    entry_bar = rows[entry_index]
    entry_price = _as_float(entry_bar.get("close"))
    if entry_price is None:
        return blank("entry_bar_has_no_close")

    seen = rows[: entry_index + 1]
    extremes = [
        value
        for value in (_as_float(bar.get("low" if is_long else "high")) for bar in seen)
        if value is not None
    ]
    if not extremes:
        return blank("no_session_extreme_before_entry")
    stop_price = min(extremes) if is_long else max(extremes)
    risk = (entry_price - stop_price) if is_long else (stop_price - entry_price)
    if risk is None or risk <= 0:
        # No risk means no R. A stop at or through the entry is not a trade
        # anybody took, and dividing by it would manufacture one.
        return blank("stop_at_or_through_entry")

    target = entry_price + INTRADAY_TARGET_R * risk if is_long else entry_price - INTRADAY_TARGET_R * risk
    first_hit = ""
    close_r = None
    for bar in rows[entry_index + 1 :]:
        high = _as_float(bar.get("high"))
        low = _as_float(bar.get("low"))
        if high is None or low is None:
            continue
        hit_stop = low <= stop_price if is_long else high >= stop_price
        hit_target = high >= target if is_long else low <= target
        if hit_stop:
            # STOP FIRST on a bar that touches both.
            first_hit, close_r = "STOP", -1.0
            break
        if hit_target:
            first_hit, close_r = "TARGET", float(INTRADAY_TARGET_R)
            break
    if close_r is None:
        last_close = _as_float(rows[-1].get("close"))
        if last_close is None:
            return blank("last_bar_has_no_close")
        move = (last_close - entry_price) if is_long else (entry_price - last_close)
        first_hit, close_r = "SESSION_CLOSE", move / risk

    entry_stamp = _bar_time(entry_bar)
    return {
        "intraday_entry_at": entry_stamp.isoformat() if entry_stamp else "",
        "intraday_entry_price": round(entry_price, 4),
        "intraday_stop_price": round(stop_price, 4),
        "intraday_risk_per_share": round(risk, 4),
        "intraday_first_hit": first_hit,
        "intraday_close_r": round(float(close_r), 4),
        "intraday_bar_count": len(rows),
        "intraday_unmeasured_reason": "",
    }


# ---------------------------------------------------------------------------
# the picks
# ---------------------------------------------------------------------------
def pass_pick_rows(
    annotations: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    annotations_path: Any = None,
) -> tuple[list[dict[str, Any]], int]:
    """(cohort rows, skipped-for-no-side count) from PASS annotations.

    One pass yields k+1 rows: one per reason code, plus `pass_all`. Every one
    of them carries the same forward outcome, because the forward return of a
    name from a date does not depend on why it was passed on - what differs is
    which cohort the row is filed in.

    The sideless refusal is the veto and like cohorts' rule, kept verbatim: a
    forward return is side-adjusted and a blank side reads as LONG downstream,
    so a pass without one is counted and named rather than graded into a
    direction the trader never expressed.

    First pass of the day wins per (date, symbol, side, source), exactly as the
    other cohorts do: the annotation log keeps every pass in full, and the
    cohort grades the name once.
    """
    from ui.annotations import pass_bars

    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp_local = moment.astimezone().isoformat(timespec="seconds")
    stamp_utc = moment.astimezone(timezone.utc).isoformat(timespec="seconds")

    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    skipped_no_side = 0
    for annotation in annotations:
        if str(annotation.get("event_type") or "") != EVENT_PASS:
            continue
        symbol = str(annotation.get("symbol") or "").strip().upper()
        side = str(annotation.get("side") or "").strip().upper()
        trade_date = str(annotation.get("session_date") or "").strip()
        if not symbol or not trade_date:
            continue
        if side not in _SIDES:
            skipped_no_side += 1
            continue

        codes = [
            str(code or "").strip().lower()
            for code in (annotation.get("reason_codes") or [])
            if str(code or "").strip()
        ]
        version = annotation.get("vocab_version")
        sidecar = pass_bars.read_pass_bars(annotation, annotations_path=annotations_path)
        passed_at = None
        created = str(annotation.get("created_at") or "").strip()
        if created:
            try:
                passed_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                passed_at = None
        intraday = intraday_pass_outcome(
            (sidecar or {}).get("bars") or (), side=side, passed_at=passed_at
        )

        sources = [pass_cohort_source(code, version) for code in codes] + [PASS_ALL_SOURCE]
        for source, code in zip(sources, [*codes, ""]):
            key = (trade_date, symbol, side, source)
            if key in rows:
                continue
            rows[key] = {
                "trade_date": trade_date,
                "symbol": symbol,
                "side": side,
                "source": source,
                "snapshotted_at": stamp_local,
                "active_at_snapshot": "1",
                "passed_at_utc": stamp_utc,
                "session_date": trade_date,
                "reason_code": code,
                # How many cohorts this one pass entered. On a `pass_all` row it
                # is the pass's own code count, which is what makes the overlap
                # readable from the file itself.
                "reason_code_count": len(codes),
                "vocab_version": "" if version is None else version,
                **intraday,
            }
    return list(rows.values()), skipped_no_side


def merge_pass_cohort_picks(
    *,
    annotations_path: Path = TRADER_ANNOTATIONS_FILE,
    picks_path: Path = PASS_COHORT_PICKS_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bring the pass cohort file up to date with the annotation log.

    Idempotent: it never rewrites or removes an existing row, so a merge lost
    to a concurrent writer is recovered by the next call rather than gone. That
    is what makes it safe to run at capture time as well as nightly, exactly
    like the veto and like merges.
    """
    annotations = load_annotations(Path(annotations_path), event_types=(EVENT_PASS,))
    candidates, skipped_no_side = pass_pick_rows(
        annotations, now=now, annotations_path=annotations_path
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


def update_pass_cohort_outcomes(
    *,
    reference_date: Any = None,
    picks_path: Path = PASS_COHORT_PICKS_FILE,
    outcomes_path: Path = PASS_COHORT_OUTCOMES_FILE,
    performance_path: Path = PASS_COHORT_PERFORMANCE_FILE,
    daily_bars_dir: Path = MASTER_AVWAP_DAILY_BARS_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Grade the pass cohort forward through the ONE existing grader.

    `human_focus_tracking.update_human_focus_outcomes` already owns the
    forward-return arithmetic, the maturity rule and the performance rollup;
    reimplementing any of it here would be a second definition to keep in step.
    The only thing this passes is a wider row identity - see
    `pick_key_with_source` for why a multi-select verdict needs one.
    """
    from human_focus_tracking import pick_key_with_source, update_human_focus_outcomes

    result = update_human_focus_outcomes(
        reference_date=reference_date,
        daily_picks_path=Path(picks_path),
        outcomes_path=Path(outcomes_path),
        performance_path=Path(performance_path),
        daily_bars_dir=Path(daily_bars_dir),
        now=now,
        pick_key=pick_key_with_source,
    )
    _stamp_overlap_note(Path(performance_path))
    return result


def _stamp_overlap_note(performance_path: Path) -> None:
    """Put `OVERLAP_NOTE` on every row of the pass rollup.

    The packet said the CSV must state the overlap and it did not - the note
    lived in the module, the Weekend Prep label and the AI scope, all of which
    are places a person reads. A CSV is read by a person too, and by whatever
    reads it next; a file whose code cohorts cannot be summed must say so ON the
    file rather than only wherever it happens to be displayed.

    Done here rather than in `HUMAN_FOCUS_PERFORMANCE_COLUMNS` because that list
    is shared by the veto, like and rejection rollups, and only the PASS cohort
    is multi-select - a note about overlapping cohorts on the veto file would be
    false.

    WRITTEN THE WAY EVERY OTHER WRITER IN THIS MODULE WRITES (R2): under the
    file's own writer lock, through a temp file and `os.replace`. The first
    version rewrote the rollup IN PLACE and unlocked - a window in which the
    file on disk is neither the old one nor the new one, over a file the nightly
    slot and the desk both read, which is the exact shape of the 2026-08-27
    feature-history corruption.

    Never allowed to cost the grading it follows: it runs AFTER the rollup is
    safely written, and every failure leaves that file exactly as the grader
    wrote it. A LOCK it cannot take, or a file it cannot parse, is REPORTED -
    returning the reason rather than swallowing it - because the slot has
    already succeeded by then and a silent no-op here reads as "the note is on
    the file".
    """
    import os

    target = Path(performance_path)
    try:
        if not target.is_file():
            return
        with target.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        _log.warning("pass cohort overlap note not stamped: %s", exc)
        return
    if not rows or "overlap_note" in rows[0]:
        return

    fieldnames = [*rows[0].keys(), "overlap_note"]
    try:
        with local_writer_lock(lock_key_for_path(target), timeout_seconds=1.0):
            tmp = target.with_name(target.name + ".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row, "overlap_note": OVERLAP_NOTE})
            os.replace(tmp, target)
    except LocalLockUnavailable:
        _log.warning(
            "pass cohort overlap note not stamped: another writer holds %s", target
        )
    except OSError as exc:
        _log.warning("pass cohort overlap note not stamped: %s", exc)



__all__ = [
    "OVERLAP_NOTE",
    "PASS_ALL_SOURCE",
    "PASS_COHORT_PREFIX",
    "PICK_COLUMNS",
    "intraday_pass_outcome",
    "merge_pass_cohort_picks",
    "pass_cohort_source",
    "pass_pick_rows",
    "update_pass_cohort_outcomes",
]
