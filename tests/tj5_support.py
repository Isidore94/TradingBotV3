r"""Hand-built fixtures for the TJ-5 (Week Review) red tests. NOT a test module.

Nothing here is produced by the code under test: `ai_jobs/week_review_narration.py`
and the Week Review page do not exist on this branch. The five day PACKS are
built by TJ-4's already-merged `day_review_pack.build_pack` (the OLD code that
TJ-5 only READS), every read verdict is a constant out of
`market_read_grades` rather than a typed string, and every expected number in a
test is hand-counted in that test's own docstring from the tables below.

THE WEEK is Monday **2026-09-14** to Friday **2026-09-18** - five real regular
sessions with no holiday in them, and the week whose Friday the live
`C:\TradingBotData\day_review\sessions\` folder actually reaches (it holds
2026-09-16, -17 and -18 and ZERO `pack.json`, measured 2026-09-20). That is why
`PACKED_SESSIONS` is three of five by default: the first thing the trader will
see on a Saturday is a week with fewer than five packs, and a week page that
cannot say so honestly is the defect this fixture exists to catch.

WHAT IS MODELLED AS IT REALLY IS
--------------------------------
* A session with no pack is ABSENT from disk - `read_pack` answers `None`. It is
  never a pack full of zeroes, and it must never be counted as a quiet day.
* A pack's `report_card` hook is PRESENT and EMPTY (`{}`) when nobody built a
  card for that session - that is exactly what TJ-4 ships and what 100% of the
  live sessions carry today.
* A read verdict may be `right`, `wrong`, `flat`, `pending <date>` or
  `unmeasured:<reason>`. Only the first two are resolved; everything else is
  `unresolved` and is never zero.
* The AI job ledger holds SKIPPED rows for the same job all night long; only
  `ok` / `failed` / `degraded` decide anything (`day_report_card._slot_verdicts`).

THE HAND-COUNTED TABLES
-----------------------
Reads per session (``READS_BY_SESSION``)::

    2026-09-14   2 right   1 wrong   0 other
    2026-09-15   1 right   0 wrong   1 other (pending)
    2026-09-16   1 right   1 wrong   0 other
    2026-09-17   0 right   2 wrong   0 other
    2026-09-18   1 right   0 wrong   1 other (unmeasured)
    ---------------------------------------------------
    all five     5 right   4 wrong   2 other      n = 11
    PACKED only  2 right   3 wrong   1 other      n =  6   (09-16, -17, -18)

Report-card counts per session (``CARD_COUNTS``), the integers TJ-12's lines
carry and the week re-cut pools::

    session      did_well n/measured/runs   missed n/measured/runs
    2026-09-14          4 / 3 / 2                  6 / 5 / 1
    2026-09-15          2 / 2 / 1                  3 / 2 / 0
    2026-09-16          5 / 4 / 3                  4 / 4 / 2
    2026-09-17          1 / 1 / 0                  7 / 6 / 3
    2026-09-18          3 / 3 / 2                  2 / 1 / 1
    ------------------------------------------------------------
    all five           15 /13 / 8                 22 /18 / 7
    PACKED only         9 / 8 / 5                 13 /11 / 6
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")

#: Monday to Friday, 2026-09-14 .. 2026-09-18. Checked against the exchange
#: calendar by `test_tj5_week_strip.test_the_week_is_five_real_exchange_sessions`
#: rather than trusted here.
WEEK: tuple[str, ...] = (
    "2026-09-14",
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-18",
)
FRIDAY = WEEK[-1]

#: The three sessions the LIVE store has folders for (2026-09-20). Three of five.
PACKED_SESSIONS: tuple[str, ...] = ("2026-09-16", "2026-09-17", "2026-09-18")

#: ISO week id, computed from the stdlib rather than typed, so a wrong week
#: number cannot hide in a string.
WEEK_ID = "{0}-W{1:02d}".format(*date.fromisoformat(FRIDAY).isocalendar()[:2])

#: Saturday night, 22:00 ET - inside the shipped 18:30-08:00 window, on the
#: evening of a non-session day whose next day is also not a session, which is
#: `runner.night_kind()`'s definition of the SATURDAY night.
SATURDAY_NIGHT = datetime(2026, 9, 19, 22, 0, tzinfo=EASTERN)
#: Saturday AFTERNOON - the moment TJ-13A item 1 exists for. Outside the window.
SATURDAY_AFTERNOON = datetime(2026, 9, 19, 14, 0, tzinfo=EASTERN)

READS_BY_SESSION: dict[str, tuple[tuple[str, str], ...]] = {}
CARD_COUNTS: dict[str, dict[str, tuple[int, int, int]]] = {
    "2026-09-14": {"did_well": (4, 3, 2), "missed": (6, 5, 1)},
    "2026-09-15": {"did_well": (2, 2, 1), "missed": (3, 2, 0)},
    "2026-09-16": {"did_well": (5, 4, 3), "missed": (4, 4, 2)},
    "2026-09-17": {"did_well": (1, 1, 0), "missed": (7, 6, 3)},
    "2026-09-18": {"did_well": (3, 3, 2), "missed": (2, 1, 1)},
}


def _verdicts() -> dict[str, str]:
    import market_read_grades as grades

    return {
        "right": grades.VERDICT_RIGHT,
        "wrong": grades.VERDICT_WRONG,
        "pending": f"{grades.PENDING_PREFIX} 2026-09-16",
        "unmeasured": f"{grades.UNMEASURED_PREFIX}:no_tape",
    }


def _fill_reads() -> None:
    """The table in this module's docstring, as (read_id suffix, verdict) pairs."""
    if READS_BY_SESSION:
        return
    v = _verdicts()
    READS_BY_SESSION.update(
        {
            "2026-09-14": (("a", v["right"]), ("b", v["right"]), ("c", v["wrong"])),
            "2026-09-15": (("a", v["right"]), ("b", v["pending"])),
            "2026-09-16": (("a", v["right"]), ("b", v["wrong"])),
            "2026-09-17": (("a", v["wrong"]), ("b", v["wrong"])),
            "2026-09-18": (("a", v["right"]), ("b", v["unmeasured"])),
        }
    )


def tally_over(sessions: Sequence[str]) -> dict[str, int]:
    """``{"right", "wrong", "unresolved", "n"}`` over ``sessions``. Counting only."""
    _fill_reads()
    import market_read_grades as grades

    right = wrong = other = 0
    for session in sessions:
        for _suffix, verdict in READS_BY_SESSION.get(session, ()):
            if verdict == grades.VERDICT_RIGHT:
                right += 1
            elif verdict == grades.VERDICT_WRONG:
                wrong += 1
            else:
                other += 1
    return {"right": right, "wrong": wrong, "unresolved": other, "n": right + wrong + other}


# ---------------------------------------------------------------------------
# the trader's own rows, one prediction and one note per session
# ---------------------------------------------------------------------------
def session_moment(session: str, hour: int = 10, minute: int = 5) -> datetime:
    day = date.fromisoformat(str(session)[:10])
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=PACIFIC)


def entries_for(session: str) -> list[dict[str, Any]]:
    """One Mentor row that CALLED something and one older note, for `session`.

    Through `tj4_support`, so the rows are what TJ-14A's writer really produces:
    `mentor.observation` and `mentor.prediction` as two separate keys on the
    first, and `mentor` PRESENT and EMPTY on the second.
    """
    import tj4_support as tj4

    clicked = tj4.observing_and_predicting_entry(stamp=session_moment(session))
    clicked = {**clicked, "entry_id": f"{session}-call"}
    noted = tj4.observation_only_entry(stamp=session_moment(session, hour=12))
    noted = {**noted, "entry_id": f"{session}-note"}
    return [clicked, noted]


def read_rows_for(session: str) -> list[dict[str, Any]]:
    """The session's graded reads, in the shape the Day Review payload carries.

    The VERDICT strings are `market_read_grades` constants; the counts are the
    table in this module's docstring. Nothing here grades anything.
    """
    _fill_reads()
    rows: list[dict[str, Any]] = []
    for index, (suffix, verdict) in enumerate(READS_BY_SESSION.get(session, ())):
        rows.append(
            {
                "read_id": f"{session}-{suffix}",
                "entry_id": f"{session}-call",
                "session_date": session,
                "timeframe": "M5" if index % 2 == 0 else "D1",
                "direction": "up" if index % 2 == 0 else "down",
                "horizon": "rest_of_day",
                "created_at": session_moment(session).isoformat(),
                "text": f"{session} read {suffix}",
                "observation": f"{session} observation {suffix}",
                "verdict": verdict,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# the day inputs TJ-12's card is built from, and the card itself
# ---------------------------------------------------------------------------
def walkaway_for(session: str):
    """A real `WalkawayDay` whose two tables carry this session's counts.

    `day_report_card.did_well_line` counts the `liked_not_traded` table and
    `missed_line` the `rejected` one, and both read `real_miss` off each row
    through `real_miss.RUN` / `real_miss.NO_RUN`. So the rows are built to the
    (n, measured, runs) triples in ``CARD_COUNTS`` and nothing is typed.
    """
    import real_miss
    import walkaway_day

    def _rows(name: str, n: int, measured: int, runs: int):
        out = []
        for index in range(n):
            if index < runs:
                verdict = real_miss.RUN
            elif index < measured:
                verdict = real_miss.NO_RUN
            else:
                verdict = "unmeasured"
            out.append(
                walkaway_day.WalkawayRow(
                    decision_id=(session, f"SYM{index}", "LONG", "chart_review", name, "annotations", "D1"),
                    time=session_moment(session),
                    symbol=f"SYM{index}",
                    side="LONG",
                    category="chart_review",
                    what_you_did="like" if name == "liked_not_traded" else "veto",
                    ran_after_pct=float(index),
                    state="measured" if index < measured else "unmeasured",
                    real_miss=verdict,
                    reason="extended",
                )
            )
        return tuple(out)

    counts = CARD_COUNTS[session]
    return walkaway_day.WalkawayDay(
        liked_not_traded=_rows("liked_not_traded", *counts["did_well"]),
        rejected=_rows("rejected", *counts["missed"]),
        traded_left_early=(),
        claimed_d1=(),
        earlier_calls=(),
        skill={
            "session": {
                "window_sessions": 1,
                "cells": [],
                "overlapping": (),
                "sentence": f"{session}: too few to call.",
            }
        },
        sentences={
            "liked_not_traded": f"You liked {counts['did_well'][0]} and left them.",
            "rejected": f"You rejected {counts['missed'][0]}.",
        },
        money={"n": 0, "net": None, "line": "too few to call (n=0)"},
    )


def day_inputs_for(session: str, *, ledger_path: Any = None) -> dict[str, Any]:
    """Exactly what `day_report_card.build` takes for ONE session.

    ``freshness.session`` is THIS session, never blank: `how_fresh` filters the
    ledger tail on it, and an empty one pools every night the tail holds
    (`day_report_card._slot_verdicts`, the `if session and ...` guard).
    """
    return {
        "session": session,
        "walkaway": walkaway_for(session),
        "your_reads": None,
        "congruence": (),
        "trades": [],
        "origin_lanes": {},
        "origin_lanes_read": (),
        "freshness": {
            "session": session,
            "story_written_at": "",
            "fills_current_to": "",
            "reads_graded_through": session,
            "ledger_path": ledger_path,
        },
    }


def card_for(session: str, *, ledger_path: Any = None):
    """TJ-12's built `ReportCard` for one session, through its own owner."""
    import day_report_card

    return day_report_card.build(day_inputs_for(session, ledger_path=ledger_path))


# ---------------------------------------------------------------------------
# the packs on disk
# ---------------------------------------------------------------------------
def pack_for(
    session: str, *, with_card: bool = True, ledger_path: Any = None, **overrides
) -> dict[str, Any]:
    """One TJ-4 day pack for `session`, through `day_review_pack.build_pack`."""
    import day_review_pack

    kwargs: dict[str, Any] = {
        "entries": entries_for(session),
        "forecast": {},
        "story": None,
        "environment": (),
        "d1_label": "trending_up",
        "internals": (),
        "walkaway": walkaway_for(session),
        "reads": read_rows_for(session),
        "congruence": (),
        "trades": [],
        "report_card": card_for(session, ledger_path=ledger_path) if with_card else None,
    }
    kwargs.update(overrides)
    return day_review_pack.build_pack(
        session,
        now=datetime.fromisoformat(f"{session}T16:30:00-04:00"),
        **kwargs,
    )


def write_week(
    root: Path,
    *,
    sessions: Sequence[str] = PACKED_SESSIONS,
    with_card: bool = True,
    narrated: Sequence[str] | None = None,
    ledger_path: Any = None,
) -> dict[str, dict[str, Any]]:
    """Write one pack per session in `sessions`, and a TJ-4 narration for each
    session in `narrated`. Sessions outside `sessions` are ABSENT from disk.
    """
    import day_review_pack

    packs: dict[str, dict[str, Any]] = {}
    for session in sessions:
        pack = pack_for(session, with_card=with_card, ledger_path=ledger_path)
        day_review_pack.write_pack(pack, root=Path(root))
        packs[session] = pack
    for session in narrated or ():
        write_day_narration(Path(root), session, packs.get(session))
    return packs


def write_day_narration(root: Path, session: str, pack: Mapping[str, Any] | None) -> Path:
    """A TJ-4 day story on disk, in `day_review_narration`'s own shape."""
    import ai_jobs.day_review_narration as day_narration

    path = day_narration.narration_path(session, root=Path(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": day_narration.SCHEMA,
        "session_date": session,
        "generated_at": f"{session}T23:10:00+00:00",
        "inputs_hash": str((pack or {}).get("inputs_hash") or ""),
        "prompt_version": day_narration.PROMPT_VERSION,
        "model": "gemma3:12b-tbv3ctx-64k",
        "graded": {"reads_graded": 1, "reads_in_pack": len(read_rows_for(session))},
        "narration": {
            "headline": f"{session}: the tape held its opening range.",
            "what_happened": f"{session} narrated.",
            "what_you_thought": "You expected continuation.",
            "were_you_right": [],
            "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
            "process": "You sat on your hands.",
            "sources": [],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the AI job ledger
# ---------------------------------------------------------------------------
def write_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """An AI-jobs ledger holding exactly `rows`, JSONL, in order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def ledger_row(job: str, session: str, status: str, **extra) -> dict[str, Any]:
    return {
        "job": job,
        "session_date": session,
        "status": status,
        "at": f"{session}T23:00:00+00:00",
        "reason": "",
        **extra,
    }


# ---------------------------------------------------------------------------
# the two contrast packs TJ-5's tendencies and misses come from
# ---------------------------------------------------------------------------
def prediction_contrast_pack(*, reportable_cells: int = 5, n_start: int = 40) -> dict[str, Any]:
    """A TJ-16 pack whose `by_hour` table holds `reportable_cells` cells.

    Each cell's ``n`` is distinct and descending from `n_start`, so
    `prediction_contrast.tendencies`' SIZE ordering (n desc, then name) has one
    answer. One extra cell sits UNDER `evidence_stats.MIN_REPORTABLE_N` with
    ``reportable`` False, and must never be offered.
    """
    import evidence_stats

    cells = []
    for index in range(reportable_cells):
        n = n_start - index
        right = n // 2
        cells.append(
            {
                "key": 7 + index,
                "n": n,
                "right": right,
                "wrong": n - right,
                "rate": right / n,
                "low": 0.3,
                "reportable": True,
            }
        )
    thin = evidence_stats.MIN_REPORTABLE_N - 5
    cells.append(
        {
            "key": 6,
            "n": thin,
            "right": thin,
            "wrong": 0,
            "rate": 1.0,
            "low": 0.1,
            "reportable": False,
        }
    )
    return {
        "schema": "prediction_contrast_v1",
        "session_date": FRIDAY,
        "horizons": {"rest_of_day": {"tables": {"by_hour": cells}}},
        "notes": [],
    }


def miss_contrast_pack() -> dict[str, Any]:
    """A TJ-15 pack with one reportable feature and one named as too thin."""
    return {
        "schema": "miss_contrast_v1",
        "session_date": FRIDAY,
        "groups": [
            {
                "verdict": "veto",
                "reason": "extended",
                "n": 34,
                "runs": 12,
                "rate": 12 / 34,
                "reportable": True,
            }
        ],
        "features": [
            {
                "feature": "compression_calibration.auc",
                "auc": 0.62,
                "n": 31,
                "reportable": True,
            }
        ],
        "thin_features": ["gap_pct"],
        "excluded_by_timeframe": {"M5": 4},
        "notes": [],
    }


def sessions_missing(sessions: Sequence[str] = PACKED_SESSIONS) -> tuple[str, ...]:
    """The week's sessions with no pack, in week order. Named, never padded."""
    have = set(sessions)
    return tuple(day for day in WEEK if day not in have)


def next_weekday(day: str, count: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=count)).isoformat()
