"""Hand-built fixtures for the TJ-12 red tests. NOT a test module.

Nothing here is produced by the code TJ-12 will build. Every number is either
counted by hand from the plan written beside it, or computed by the OWNER that
TJ-12 is forbidden to duplicate:

* the read ledger goes through `market_journal.build_entry` ->
  `market_read_grades.read_rows` -> `grade_read` -> `append_grades`, exactly as
  TJ-16's `tj16_support` does, and the tally comes back out of
  `prediction_ledger.your_reads`;
* the skill window comes out of `walkaway_day._skill_window`, so its cells, its
  Wilson bounds and its sentence are the real ones;
* the walk-away rows are real `walkaway_day.WalkawayRow` instances;
* the congruence lines are the real `market_read_grades.congruence_lines`
  output, built from a read and a desk label.

THE READ LEDGER (:func:`one_session_of_clicks`)
-----------------------------------------------
ONE falling session (the tape closes 2.00 BELOW the anchor, three flat-band
widths, so `down` is right and `up` is wrong), four hourly clicks::

    07:00  down   last hour: down   D1 label: compressed
    08:00  down   last hour: down   D1 label: compressed
    09:00  down   last hour: unmeasured
    10:00  up     last hour: unmeasured

Counted by hand:

===========================  =====  =====  ====================
population                   right      n  rate
===========================  =====  =====  ====================
the trader                       3      4  0.75
`always up`                      0      4  0.00
`same as the last hour`          2      2  1.00  (two stamps unmeasured)
`with the D1 environment`        0      0  no rate  (label is not a direction)
===========================  =====  =====  ====================

That is the shape the honest wording has to survive: the trader has MORE right
answers than the best baseline (3 against 2) and a WORSE rate (0.75 against
1.00) on a SMALLER base. `prediction_ledger.your_reads` picks the baseline with
the most `right`, which is `same as the last hour` here. A card that turned
"3 beats 2" into a compliment would be wrong, and the tests say so.

THE SKILL WINDOW (:func:`skill_window`)
---------------------------------------
One side (`LONG`), three families, every name decided `liked_or_claimed` and
every horizon CLOSED, so the fractions are exactly:

===========  =====  =========  ======  ==================================
family       runs   measured   rate    Wilson lower bound (the ONE, z 1.96)
===========  =====  =========  ======  ==================================
`steady`        55        100    0.55   ~0.4524  <- the best BOUND
`flashy`        18         30    0.60   ~0.4232
`tiny`           9         10    0.90   under `MIN_REPORTABLE_N`, never named
===========  =====  =========  ======  ==================================

`flashy` wins on the RATE and `steady` on the BOUND; `tiny` wins on both and is
under the floor. A card that ranked on the rate names `flashy`; a card that
ignored the floor names `tiny`. The bounds are not written down here - they are
read back out of `walkaway_day`'s own cells, because the ONE Wilson is its
`swing_headline` import and not a copy in a test.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tj16_support as fx  # noqa: E402  - the shipped TJ-16 tape and writers

PACIFIC = ZoneInfo("America/Los_Angeles")

#: The session every fixture here is about. A Friday, a regular close.
SESSION = "2026-09-18"

#: The previous exchange session, for the week re-cut.
PRIOR_SESSION = "2026-09-17"


# ---------------------------------------------------------------------------
# the read ledger
# ---------------------------------------------------------------------------
#: (hour, direction, last_hour_spy). See the module docstring for the counts.
CLICK_PLAN: tuple[tuple[int, str, str], ...] = (
    (7, "down", "down"),
    (8, "down", "down"),
    (9, "down", "unmeasured"),
    (10, "up", "unmeasured"),
)

#: A D1 label that is NOT a direction, so `with_the_d1_environment` has no
#: answer on any of these stamps and leaves its own fraction rather than
#: counting a guess against itself.
NO_DIRECTION_LABEL = "compressed"


def one_session_of_clicks(
    root: Path, *, session: str = SESSION, plan: Sequence[tuple[int, str, str]] = CLICK_PLAN
) -> Path:
    """Write ONE session's graded clicks through the store's own writer."""
    grades = fx.graded_session(
        session,
        [
            {
                "hour": hour,
                "direction": direction,
                "confidence": "medium",
                "last_hour_spy": last_hour,
                "d1_environment": NO_DIRECTION_LABEL,
            }
            for hour, direction, last_hour in plan
        ],
        rising=False,
        band_multiple=3.0,
    )
    fx.write_ledger(Path(root), {session: grades})
    return Path(root)


#: The prior session: SIX clicks, five of them `up` on a falling tape, so ONE
#: right of six. Deliberately a DIFFERENT size from `CLICK_PLAN`'s four, because
#: a week that averaged the two days' rates instead of pooling their counts
#: would only be caught by days of unequal length:
#:
#:     pooled   (3 + 1) right of (4 + 6) = 4/10 = 0.400
#:     averaged (0.750 + 0.1667) / 2             = 0.458
PRIOR_CLICK_PLAN: tuple[tuple[int, str, str], ...] = (
    (7, "up", "down"),
    (8, "up", "down"),
    (9, "up", "unmeasured"),
    (10, "up", "unmeasured"),
    (11, "up", "unmeasured"),
    (12, "down", "unmeasured"),
)

#: Hand-counted, and re-checked by the owner in the tests.
WEEK_READS_RIGHT = 4
WEEK_READS_N = 10


def your_reads_for(root: Path, session: str = SESSION) -> dict[str, Any]:
    """The OWNER's answer for one session - never a hand-typed tally."""
    import prediction_ledger

    return prediction_ledger.your_reads(session, root=Path(root))


# ---------------------------------------------------------------------------
# the skill window (TJ-11's base rates)
# ---------------------------------------------------------------------------
#: family -> (runs, measured). Hand-chosen; see the module docstring.
FAMILY_PLAN: dict[str, tuple[int, int]] = {
    "steady": (55, 100),
    "flashy": (18, 30),
    "tiny": (9, 10),
}

BEST_BOUND_FAMILY = "steady"
BEST_RATE_FAMILY = "flashy"
UNDER_FLOOR_FAMILY = "tiny"


def _skill_maps(session: str = SESSION):
    """``(names, decided, verdicts)`` for `walkaway_day._skill_window`."""
    import real_miss
    import walkaway_day

    names: dict[tuple[str, str, str], str] = {}
    decided: dict[tuple[str, str, str], str] = {}
    verdicts: dict[tuple[str, str, str], tuple[str, str]] = {}
    for family, (runs, measured) in FAMILY_PLAN.items():
        for index in range(measured):
            key = (session, f"{family.upper()}{index:03d}", "LONG")
            names[key] = family
            decided[key] = "liked_or_claimed"
            verdicts[key] = (
                real_miss.RUN if index < runs else real_miss.NO_RUN,
                walkaway_day.POOL_MEASURED,
            )
    return names, decided, verdicts


def skill_window(session: str = SESSION) -> dict[str, Any]:
    """`walkaway_day`'s OWN window over the plan above. Real cells, real bounds."""
    import evidence_stats
    import walkaway_day

    names, decided, verdicts = _skill_maps(session)
    return walkaway_day._skill_window(
        names,
        decided,
        verdicts,
        window_sessions=evidence_stats.LATELY_SESSIONS,
        label="Of this scan",
    )


def family_cell(window: Mapping[str, Any], family: str, population: str = "liked_or_claimed"):
    """One cell out of a skill window, by family and population."""
    for cell in window.get("cells") or ():
        if cell.get("setup_family") == family and cell.get("population") == population:
            return cell
    return None


# ---------------------------------------------------------------------------
# the walk-away rows
# ---------------------------------------------------------------------------
def _stamp(session: str, hour: int) -> datetime:
    day = date.fromisoformat(session)
    return datetime(day.year, day.month, day.day, hour, 5, tzinfo=PACIFIC)


def walkaway_row(
    symbol: str,
    *,
    what_you_did: str,
    real_miss_verdict: str = "",
    reason: str = "",
    session: str = SESSION,
    hour: int = 7,
    instrument: str = "stock",
    **extra: Any,
):
    """A REAL `walkaway_day.WalkawayRow`, not a look-alike dict."""
    import walkaway_day

    return walkaway_day.WalkawayRow(
        decision_id=(session, symbol, "LONG", what_you_did, "", "", ""),
        time=_stamp(session, hour),
        symbol=symbol,
        side="LONG",
        category="d1_scan",
        what_you_did=what_you_did,
        real_miss=real_miss_verdict,
        reason=reason,
        instrument=instrument,
        **extra,
    )


#: Four likes: two real runs, one no-run, one the desk could not measure. So the
#: `Did well` line is **2 runs of 3 measured, 4 considered** - and the
#: unmeasured one is NEVER read as a zero.
LIKED_PLAN: tuple[tuple[str, str], ...] = (
    ("AAA", "run"),
    ("BBB", "run"),
    ("CCC", "no_run"),
    ("DDD", "unmeasured:no_completed_bar_after_the_stamp"),
)

#: Five vetoes: three real misses, one no-run, one unmeasured; THREE of them
#: share the coded reason `extended`, which is what `_reason_clause` names.
REJECTED_PLAN: tuple[tuple[str, str, str], ...] = (
    ("EEE", "run", "extended"),
    ("FFF", "run", "extended"),
    ("GGG", "run", "extended"),
    ("HHH", "no_run", "late_in_the_day"),
    ("III", "unmeasured:atr_unreadable", ""),
)


def walkaway_day_fixture(session: str = SESSION, *, with_skill: bool = True):
    """A REAL `walkaway_day.WalkawayDay` with the plans above and its sentences."""
    import walkaway_day

    liked = tuple(
        walkaway_row(symbol, what_you_did="like", real_miss_verdict=verdict, session=session)
        for symbol, verdict in LIKED_PLAN
    )
    rejected = tuple(
        walkaway_row(
            symbol,
            what_you_did="veto",
            real_miss_verdict=verdict,
            reason=reason,
            session=session,
        )
        for symbol, verdict, reason in REJECTED_PLAN
    )
    day = {
        "liked_not_traded": liked,
        "rejected": rejected,
        "traded_left_early": (),
        "claimed_d1": (),
        "earlier_calls": (),
    }
    window = skill_window(session) if with_skill else None
    return walkaway_day.WalkawayDay(
        liked_not_traded=liked,
        rejected=rejected,
        traded_left_early=(),
        claimed_d1=(),
        earlier_calls=(),
        skill={"session": window, "lately": window} if window else None,
        sentences=walkaway_day._sentences(day),
        money=walkaway_day._money(()),
    )


# ---------------------------------------------------------------------------
# the congruence lines (TJ-10's own output)
# ---------------------------------------------------------------------------
#: Three D1 likes, all LONG. The read is `up` and the crowd leans up, so the
#: picks line AGREES on the direction - and three is far under
#: `MIN_REPORTABLE_N`, so its verdict is `too_few` and the card must print that
#: as "too few to call", never as agreement.
D1_LIKE_SYMBOLS: tuple[str, ...] = ("AAA", "BBB", "CCC")


def congruence_lines(session: str = SESSION, *, decisions: Sequence[Mapping[str, Any]] | None = None):
    """The REAL `market_read_grades.congruence_lines` for a D1 up-read."""
    import market_read_grades as grades

    read = {
        "read_id": "read-d1-1",
        "direction": "up",
        "timeframe": "D1",
        "horizon": "next_5_sessions",
        "source": grades.SOURCE_CLICK,
    }
    if decisions is None:
        decisions = [
            {
                "symbol": symbol, "side": "LONG", "timeframe": "D1",
                "verdict": "like", "capture_id": f"cap-{symbol}",
                "session_date": session,
            }
            for symbol in D1_LIKE_SYMBOLS
        ]
    return grades.congruence_lines(
        session=session,
        d1_read=read,
        d1_label="trending_up",
        decisions=list(decisions),
        claims=(),
        trades=(),
    )


# ---------------------------------------------------------------------------
# the trades (TJ-9's Process line)
# ---------------------------------------------------------------------------
#: Four closed trades. Two were preceded by a decision (PLANNED), two were not
#: (UNPLANNED); one of those carries a date-only fill, which `trade_origin`
#: refuses to place at all and calls `unmeasured`. An OLD row has its label keys
#: PRESENT and EMPTY - never absent.
def trades_fixture(session: str = SESSION) -> list[dict[str, Any]]:
    day = date.fromisoformat(session)

    def _fill(hour: int) -> str:
        return datetime(day.year, day.month, day.day, hour, 45, tzinfo=PACIFIC).isoformat()

    return [
        {
            "trade_id": "T1", "symbol": "AAA", "side": "long", "status": "closed",
            "opened_at": _fill(7), "closed_at": _fill(12),
            "setup": "steady", "tag_status": "confirmed",
            "label_provenance": "claimed_before_entry",
            "instrument": "stock",
        },
        {
            "trade_id": "T2", "symbol": "BBB", "side": "long", "status": "closed",
            "opened_at": _fill(8), "closed_at": _fill(12),
            "setup": "steady", "tag_status": "confirmed",
            "label_provenance": "same_session",
            "instrument": "stock",
        },
        {
            "trade_id": "T3", "symbol": "ZZZ", "side": "long", "status": "closed",
            "opened_at": _fill(9), "closed_at": _fill(12),
            # An old row: the keys are PRESENT and EMPTY.
            "setup": "", "tag_status": "", "label_provenance": "",
            "instrument": "stock",
        },
        {
            "trade_id": "T4", "symbol": "YYY", "side": "long", "status": "closed",
            # A broker file is blind to time: midnight market-local, so
            # `trade_origin` refuses to place it and answers `unmeasured`.
            "opened_at": f"{session}T00:00:00-04:00",
            "closed_at": f"{session}T00:00:00-04:00",
            "setup": "", "tag_status": "", "label_provenance": "",
            "instrument": "option",
        },
    ]


def origin_lanes(session: str = SESSION) -> dict[str, list[dict[str, Any]]]:
    """The four lanes `trade_origin.planned_state` reads, as the desk holds them.

    Only AAA and BBB were spoken about before their first fill, so exactly two
    of the four trades are `planned`.
    """
    day = date.fromisoformat(session)

    def _before(hour: int) -> str:
        return datetime(day.year, day.month, day.day, hour, 0, tzinfo=PACIFIC).isoformat()

    return {
        "decisions": [
            {"symbol": "AAA", "side": "long", "created_at": _before(6), "verdict": "like"},
        ],
        "claims": [
            {"symbol": "BBB", "side": "long", "claimed_at": _before(7),
             "claimed_setup_id": "steady"},
        ],
        "focus_adds": [],
        "armed": [],
    }


# ---------------------------------------------------------------------------
# the freshness facts
# ---------------------------------------------------------------------------
LEDGER_SESSION = SESSION


def ledger_rows(*, failed_slot: str = "day_review_narration") -> list[dict[str, Any]]:
    """A night's ledger tail: one slot that failed LAST, one that recovered.

    `market_story_narration` failed and then succeeded, so it is NOT named; the
    failed slot's LAST row for the session is a failure, so it IS.
    """
    import ai_jobs.ledger as ledger

    def _row(job: str, status: str, minute: int) -> dict[str, Any]:
        moment = datetime(2026, 9, 18, 22, minute, tzinfo=PACIFIC)
        return {
            "schema": ledger.LEDGER_SCHEMA,
            "job": job,
            "status": status,
            "session_date": LEDGER_SESSION,
            "model": "",
            "started_at": moment.isoformat(timespec="seconds"),
            "finished_at": (moment + timedelta(minutes=1)).isoformat(timespec="seconds"),
            "duration_seconds": 60.0,
            "reason": "",
            "outputs": [],
            "tokens": {},
            "error": "",
        }

    return [
        _row("market_story_narration", ledger.STATUS_FAILED, 10),
        _row("market_story_narration", ledger.STATUS_OK, 20),
        _row(failed_slot, ledger.STATUS_OK, 30),
        _row(failed_slot, ledger.STATUS_FAILED, 40),
    ]


def write_ledger_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """The ledger as JSONL, through the store's own append."""
    import ai_jobs.ledger as ledger

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        ledger.append_row(row, path=target)
    return target


def freshness_facts(
    *,
    ledger_path: Any = None,
    story_written_at: str = "2026-09-18T22:31:00-07:00",
    fills_current_to: str = "2026-09-17",
    reads_graded_through: str = SESSION,
    session: str = SESSION,
) -> dict[str, Any]:
    """What `How fresh` rests on. Each value is a FACT, never a rate."""
    return {
        "session": session,
        "story_written_at": story_written_at,
        "fills_current_to": fills_current_to,
        "reads_graded_through": reads_graded_through,
        "ledger_path": ledger_path,
    }


# ---------------------------------------------------------------------------
# the whole day's inputs
# ---------------------------------------------------------------------------
def day_inputs(
    root: Path,
    *,
    session: str = SESSION,
    with_trades: bool = True,
    with_reads: bool = True,
    with_walkaway: bool = True,
    ledger_path: Any = None,
) -> dict[str, Any]:
    """One session's inputs for `day_report_card.build`, all already READ.

    `build` is PURE: every file in here was opened by the Day Review worker
    before the card was asked for. A key that is missing is an input the desk
    did not have, and the matching line has to SAY so.
    """
    return {
        "session": session,
        "walkaway": walkaway_day_fixture(session) if with_walkaway else None,
        "your_reads": your_reads_for(root, session) if with_reads else None,
        "congruence": congruence_lines(session),
        "trades": trades_fixture(session) if with_trades else [],
        "origin_lanes": origin_lanes(session),
        "freshness": freshness_facts(session=session, ledger_path=ledger_path),
    }
