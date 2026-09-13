"""Weekly, monthly and quarterly story packs — WISHLIST 10D step 3.

The daily story answers "what happened that day". These three answer "what has
been happening", and they are built from the daily stories rather than from the
journal again, so a session cannot be told one way in the day pack and another
way in the week's.

**A session is named exactly once in a period, and every period says what it is
missing.** That is harder than it sounds at the month boundary. ISO week 36 of
2026 starts on Monday 31 August and week 40 ends on Friday 2 October, so the
five weeks that touch September hold 24 sessions between them while September
itself has 21. A month built by concatenating its weeks over-counts by three
and imports sessions from two other months.

So the rule is the one the packet names - **monthly from weekly plus uncovered
days**:

* each ISO week belongs to ONE month, the month of its Thursday (the ISO
  rule). September 2026 owns weeks 36-39; week 40 belongs to October;
* a month pack is created only for a month that OWNS at least one week. A
  session in a week another month owns is not smuggled in;
* the month takes, from its own weeks, only the sessions that fall inside it,
  and then adds the **uncovered days**: sessions of that month that its weeks
  did not contribute, which is how 28-30 September (in October's week 40) get
  home;
* `weeks` on a month lists every week that TOUCHES it, because that is the
  provenance a reader wants - not the ownership rule above it.

A quarter is the same shape one level up: it is built from the month packs that
exist, and a month nobody ever wrote a story for is absent rather than empty.

**Rebuilt only when an input changed.** Every pack carries `inputs_hash` over
the stories it was built from; a run compares that hash with the one in the file
already on disk and rewrites nothing when they match. Two runs over unchanged
stories leave every file byte-identical, and changing one session's story
rebuilds its week, its month and its quarter - and nothing else.

Deterministic: no model, no clock of its own (`now` is injected), and nothing
here reaches a detector, score, gate, alert, watchlist, Focus list, the review
queue or `review_policy.json` (plan.md sec 5). The local-AI narration of these
packs is a later packet; this one only counts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

KIND_WEEKLY = "weekly"
KIND_MONTHLY = "monthly"
KIND_QUARTERLY = "quarterly"

#: How many missing sessions a coverage note NAMES before it counts the rest.
#: A quarter with a two-month hole should say how big the hole is, not print
#: forty dates; a week with two missing days must name both.
COVERAGE_NOTE_NAMES = 12

#: How many completed daily bars a story's measured part reads.
INDEX_BAR_LIMIT = 60

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------
def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parts = [int(part) for part in str(value or "")[:10].split("-")]
        return date(parts[0], parts[1], parts[2])
    except Exception:  # noqa: BLE001 - an undated story belongs to no period
        return None


def _is_session(day: date) -> bool:
    from market_calendar import is_session

    try:
        return bool(is_session(day))
    except Exception:  # noqa: BLE001 - outside the calendar's validated range
        return False


def _sessions_in(start: date, end: date) -> list[str]:
    out: list[str] = []
    cursor = start
    while cursor <= end:
        if _is_session(cursor):
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _week_id(day: date) -> str:
    year, week, _weekday = day.isocalendar()
    return f"{year}-W{week:02d}"


def _week_bounds(period_id: str) -> tuple[date, date]:
    year, week = int(period_id[:4]), int(period_id[6:])
    return date.fromisocalendar(year, week, 1), date.fromisocalendar(year, week, 7)


def _week_owner_month(period_id: str) -> str:
    """The month a week BELONGS to: the month of its Thursday (the ISO rule)."""
    year, week = int(period_id[:4]), int(period_id[6:])
    thursday = date.fromisocalendar(year, week, 4)
    return f"{thursday.year}-{thursday.month:02d}"


def _month_id(day: date) -> str:
    return f"{day.year}-{day.month:02d}"


def _month_bounds(period_id: str) -> tuple[date, date]:
    year, month = int(period_id[:4]), int(period_id[5:7])
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, end - timedelta(days=1)


def _quarter_id(month_id: str) -> str:
    year, month = int(month_id[:4]), int(month_id[5:7])
    return f"{year}-Q{(month - 1) // 3 + 1}"


def _quarter_bounds(period_id: str) -> tuple[date, date]:
    year, quarter = int(period_id[:4]), int(period_id[6:])
    first = 3 * (quarter - 1) + 1
    start = date(year, first, 1)
    end = date(year + 1, 1, 1) if first + 3 > 12 else date(year, first + 3, 1)
    return start, end - timedelta(days=1)


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _story_dict(story: Any) -> dict[str, Any]:
    if is_dataclass(story) and not isinstance(story, type):
        return asdict(story)
    return dict(story or {})


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _coverage_note(period_id: str, covered: Sequence[str], missing: Sequence[str]) -> str:
    if not missing:
        return (
            f"{period_id}: all {len(covered)} exchange session(s) in the period are "
            "covered by a daily story."
        )
    named = list(missing[:COVERAGE_NOTE_NAMES])
    rest = len(missing) - len(named)
    tail = ", ".join(named) + (f" and {rest} more" if rest > 0 else "")
    return (
        f"{period_id}: {len(covered)} of {len(covered) + len(missing)} exchange "
        f"session(s) covered; no daily story for {tail}."
    )


def _pack(
    *,
    kind: str,
    period_id: str,
    covered: Sequence[str],
    expected: Sequence[str],
    open_theses: Sequence[Mapping[str, Any]],
    digests: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    covered_sorted = sorted(dict.fromkeys(covered))
    expected_sorted = sorted(dict.fromkeys(expected))
    missing = [day for day in expected_sorted if day not in set(covered_sorted)]
    pack: dict[str, Any] = {
        "kind": kind,
        "period_id": period_id,
        "sessions_covered": covered_sorted,
        "sessions_expected": expected_sorted,
        "sessions_missing": missing,
        "complete": not missing,
        "coverage_note": _coverage_note(period_id, covered_sorted, missing),
        # Carried forward into every period a thesis is still open in: a
        # question the trader has not answered does not stop being open
        # because a week ended.
        "open_theses": [dict(row) for row in open_theses or ()],
    }
    pack.update(dict(extra or {}))
    pack["inputs_hash"] = _digest(
        {
            "kind": kind,
            "period_id": period_id,
            "sessions": covered_sorted,
            "expected": expected_sorted,
            "stories": {day: digests.get(day, "") for day in covered_sorted},
            "open_theses": [dict(row) for row in open_theses or ()],
        }
    )
    return pack


def build_rollups(
    daily_stories: Iterable[Any], *, open_theses: Sequence[Mapping[str, Any]] = ()
) -> dict[str, list[dict[str, Any]]]:
    """Weekly, monthly and quarterly packs from the daily stories handed in.

    PURE: it reads the exchange calendar and nothing else, and it writes
    nothing. `run_market_story_rollups` is what puts the result on disk.
    """
    stories: dict[str, dict[str, Any]] = {}
    for raw in daily_stories or ():
        story = _story_dict(raw)
        day = _as_date(story.get("session_date"))
        if day is None:
            continue
        stories[day.isoformat()] = story
    digests = {day: _digest(story) for day, story in stories.items()}
    days = sorted(stories)

    # -- weekly ---------------------------------------------------------
    by_week: dict[str, list[str]] = {}
    for day in days:
        parsed = _as_date(day)
        if parsed is None:
            continue
        by_week.setdefault(_week_id(parsed), []).append(day)

    weekly: list[dict[str, Any]] = []
    for period_id in sorted(by_week):
        start, end = _week_bounds(period_id)
        weekly.append(
            _pack(
                kind=KIND_WEEKLY,
                period_id=period_id,
                covered=by_week[period_id],
                expected=_sessions_in(start, end),
                open_theses=open_theses,
                digests=digests,
                extra={
                    "starts": start.isoformat(),
                    "ends": end.isoformat(),
                    "owner_month": _week_owner_month(period_id),
                    # The week is the level that keeps the WORDS: a month that
                    # repeated them would store the same thought three times.
                    "sessions": [_day_block(stories[day]) for day in by_week[period_id]],
                },
            )
        )

    # -- monthly: from the weeks it OWNS, plus its own uncovered days ----
    owned: dict[str, list[str]] = {}
    for period_id in sorted(by_week):
        owned.setdefault(_week_owner_month(period_id), []).append(period_id)

    monthly: list[dict[str, Any]] = []
    for month_id in sorted(owned):
        start, end = _month_bounds(month_id)
        from_weeks = [
            day
            for week_id in owned[month_id]
            for day in by_week[week_id]
            if _month_id_of(day) == month_id
        ]
        uncovered = [day for day in days if _month_id_of(day) == month_id and day not in set(from_weeks)]
        touching = sorted(
            week_id
            for week_id in by_week
            if _weeks_touch_month(week_id, month_id)
        )
        monthly.append(
            _pack(
                kind=KIND_MONTHLY,
                period_id=month_id,
                covered=from_weeks + uncovered,
                expected=_sessions_in(start, end),
                open_theses=open_theses,
                digests=digests,
                extra={
                    "weeks": touching,
                    "weeks_owned": list(owned[month_id]),
                    "sessions_from_uncovered_weeks": sorted(uncovered),
                },
            )
        )

    # -- quarterly: from the month packs that exist ----------------------
    by_quarter: dict[str, list[dict[str, Any]]] = {}
    for pack in monthly:
        by_quarter.setdefault(_quarter_id(str(pack["period_id"])), []).append(pack)

    quarterly: list[dict[str, Any]] = []
    for quarter_id in sorted(by_quarter):
        start, end = _quarter_bounds(quarter_id)
        months = sorted(str(pack["period_id"]) for pack in by_quarter[quarter_id])
        covered = [
            day for pack in by_quarter[quarter_id] for day in pack["sessions_covered"]
        ]
        quarterly.append(
            _pack(
                kind=KIND_QUARTERLY,
                period_id=quarter_id,
                covered=covered,
                expected=_sessions_in(start, end),
                open_theses=open_theses,
                digests=digests,
                extra={"months": months},
            )
        )

    return {KIND_WEEKLY: weekly, KIND_MONTHLY: monthly, KIND_QUARTERLY: quarterly}


def _month_id_of(day: str) -> str:
    return str(day)[:7]


def _weeks_touch_month(week_id: str, month_id: str) -> bool:
    start, end = _week_bounds(week_id)
    first, last = _month_bounds(month_id)
    return start <= last and end >= first


def _day_block(story: Mapping[str, Any]) -> dict[str, Any]:
    """One session inside a weekly pack: the words, and how to read them."""
    said = [dict(row) for row in (story.get("trader_said") or ())]
    return {
        "session_date": str(story.get("session_date") or ""),
        "entry_count": len(said),
        "entries": [
            {
                "entry_id": str(row.get("entry_id") or ""),
                "text": str(row.get("text") or ""),
                "written_after_the_session": bool(row.get("written_after_the_session")),
            }
            for row in said
        ],
        "measured": [dict(row) for row in (story.get("measured") or ())],
        "notes": [str(note) for note in (story.get("notes") or ())],
    }


# ---------------------------------------------------------------------------
# the nightly slot
# ---------------------------------------------------------------------------
def run_market_story_rollups(
    *,
    session_date: str = "",
    now: datetime | None = None,
    stories: Iterable[Any] | None = None,
    out_dir: Path | None = None,
    theses_path: Path | None = None,
    journal_dir: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Rebuild the packs whose inputs changed. Deterministic, no model.

    A journal with nothing in it is `skipped` with a reason, never a failure:
    the first night after this lands has no story to roll up and should say so
    rather than record a broken job.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    target = Path(out_dir) if out_dir is not None else _default_out_dir()

    if stories is None:
        stories = _stories_from_journal(journal_dir)
    listed = list(stories or ())
    if not listed:
        return {
            "status": "skipped",
            "model": "",
            "reason": "no daily stories exist yet; there is nothing to roll up",
            "outputs": [],
            "rebuilt": 0,
            "cached": 0,
        }

    open_theses = _open_theses(moment, path=theses_path)
    packs = build_rollups(listed, open_theses=open_theses)

    rebuilt = 0
    cached = 0
    outputs: list[str] = []
    for kind, rows in packs.items():
        for pack in rows:
            path = target / kind / f"{pack['period_id']}.json"
            if _unchanged(path, str(pack.get("inputs_hash") or "")):
                cached += 1
                continue
            body = dict(pack)
            body["built_at"] = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
            _write(path, body)
            rebuilt += 1
            outputs.append(str(path))

    return {
        "status": "ok",
        "model": "",
        "reason": (
            f"{rebuilt} pack(s) rebuilt, {cached} unchanged, over "
            f"{len(listed)} daily story/stories"
        ),
        "outputs": outputs,
        "rebuilt": rebuilt,
        "cached": cached,
        "session_date": str(session_date or ""),
    }


def _default_out_dir() -> Path:
    from project_paths import MARKET_STORY_ROLLUPS_DIR

    return Path(MARKET_STORY_ROLLUPS_DIR)


def _unchanged(path: Path, inputs_hash: str) -> bool:
    if not inputs_hash:
        return False
    try:
        existing = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(existing.get("inputs_hash") or "") == inputs_hash


def _write(path: Path, pack: Mapping[str, Any]) -> None:
    """Temp-and-rename: a half-written pack never replaces a readable one."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(dict(pack), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(target)


def _open_theses(moment: datetime, *, path: Path | None = None) -> list[dict[str, Any]]:
    try:
        import market_thesis

        rows = market_thesis.read_rows(path)
        return market_thesis.active_theses(rows, as_of=moment.date())
    except Exception:  # noqa: BLE001 - a missing sidecar is no open thesis
        _log.debug("Market theses unreadable for the rollups.", exc_info=True)
        return []


def _stories_from_journal(journal_dir: Path | None = None) -> list[Any]:
    """Every session the journal has words for, as a daily story.

    Headless: it reads the ledger directly rather than through the Qt service,
    and it selects each entry's session the ONE way this desk selects it
    (`market_journal.session_of_entry`), so the nightly rollup and the desk's
    Story pane can never disagree about which day a note belongs to.
    """
    try:
        import market_journal
        import market_story
        from evidence_ledger import EvidenceLedger

        ledger = (
            EvidenceLedger(
                stream=market_journal.STREAM,
                schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
                directory=Path(journal_dir),
            )
            if journal_dir is not None
            else EvidenceLedger(
                stream=market_journal.STREAM,
                schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
            )
        )
        result = ledger.read()
        entries = market_journal.resolve_entries(result.rows)
    except Exception:  # noqa: BLE001 - an unreadable journal is no story
        _log.debug("Market journal unreadable for the rollups.", exc_info=True)
        return []
    if not entries:
        return []

    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in entries:
        by_session.setdefault(market_journal.session_of_entry(row), []).append(row)
    by_session.pop("", None)

    bars = load_index_bars(market_story.BENCHMARKS)
    return [
        market_story.build_daily_story(
            session, entries=rows, index_bars=bars
        )
        for session, rows in sorted(by_session.items())
    ]


def load_index_bars(symbols: Iterable[str], *, limit: int = INDEX_BAR_LIMIT) -> dict[str, list[dict[str, Any]]]:
    """Completed daily OHLC for the benchmarks, from the DURABLE store.

    READS, never fetches - the same parquet-per-symbol store the cohort grades
    use. It lives here rather than in `market_story` because that module is
    pure by contract, and here rather than in the Qt service because the
    nightly slot must not import Qt; both callers share this one reader so the
    desk and the overnight run measure the same bars.

    A symbol with no readable file is simply absent, which the story reports as
    unmeasured with a reason. Missing data is uncertainty, never a zero.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        import pandas as pd

        from human_focus_tracking import (
            MASTER_AVWAP_DAILY_BARS_DIR,
            _load_durable_daily_frame,
        )
    except Exception:  # noqa: BLE001 - no pandas, no measured part
        return out
    for raw in symbols or ():
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        try:
            frame = _load_durable_daily_frame(symbol, Path(MASTER_AVWAP_DAILY_BARS_DIR))
            if frame is None or getattr(frame, "empty", True):
                continue
            work = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
            if "datetime" not in work.columns:
                for candidate in ("date", "time", "timestamp"):
                    if candidate in work.columns:
                        work["datetime"] = work[candidate]
                        break
            if "datetime" not in work.columns or "close" not in work.columns:
                continue
            work = work.tail(int(limit))
            bars: list[dict[str, Any]] = []
            for _index, record in work.iterrows():
                stamp = pd.to_datetime(record["datetime"], errors="coerce")
                if stamp is None or pd.isna(stamp):
                    continue
                bar: dict[str, Any] = {"dt": stamp.date().isoformat()}
                for name in ("open", "high", "low", "close"):
                    if name in work.columns:
                        bar[name] = record[name]
                bars.append(bar)
            if bars:
                out[symbol] = bars
        except Exception:  # noqa: BLE001 - one unreadable symbol is one unmeasured cell
            _log.debug("Daily bars unreadable for %s.", symbol, exc_info=True)
    return out
