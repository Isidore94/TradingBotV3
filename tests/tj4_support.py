"""Hand-built fixtures for the TJ-4 red tests. NOT a test module.

The precedent is `tests/tj10_support.py`, and this module builds ON it rather
than beside it: TJ-4's pack is assembled out of TJ-10's read rows, so the two
must be measured against the same tape or the pack's `reads` section would be
graded by a second ruler.

Nothing here is produced by the code under test. `day_review_pack.py` and
`ai_jobs/day_review_narration.py` do not exist yet; every read row below comes
from TJ-10's own `read_rows` + `grade_read` (the OLD code), every bar is written
from a level chosen by hand, and every expected number in the tests is derived
from those levels by arithmetic written out in the test itself.

THE SESSION is `tj10_support.SESSION` - Friday **2026-09-18**, a real
regular-close session. Its tape rises from 100.0 (the first completed bar after
the 07:02 Pacific stamp) to 102.0 at the close, so `SESSION_MOVE` is +2.00 and
`tj10_support.atr_for(SESSION_MOVE, 3.0)` is the ATR that puts that move three
flat-band widths up - `right` for an `up` call and `wrong` for a `down` one,
whatever value the builder gave the band.

WHAT IS MODELLED AS IT REALLY IS
--------------------------------
* A machine row is a REAL `auto_mode_flip` entry (`market_journal.MACHINE_ORIGINS`),
  not a dict with a `[desk]` prefix in its text.
* A Mentor row carries `observation` AND `prediction` as two separate keys, as
  TJ-14A writes them; an older row carries `mentor == {}` - the key PRESENT and
  EMPTY, which is 13 of the live September rows.
* A session with no pasted forecast has `forecast == {}`, which is what
  `DayReviewService._forecast` answers, not `None` and not a missing key.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")

SESSION = fx.SESSION
SESSION_MOVE = fx.SESSION_MOVE
AFTER_THE_CLOSE = fx.AFTER_THE_CLOSE

#: 02:00 Eastern on the Monday morning after the Friday session - inside every
#: shipped night window, and a FIXED instant, so no test here builds a slate or
#: a verdict out of the clock this machine happens to be on.
OVERNIGHT = datetime(2026, 9, 19, 2, 0, tzinfo=ZoneInfo("America/New_York"))

#: The forecast the trader actually pasted, kept under `tests/fixtures/day_review/`
#: (a `tests/fixtures/*.json` file directly under that folder would owe the
#: Milestone-3 provenance contract; this is text, and it lives one level down).
FORECAST_FIXTURE = ROOT_DIR / "tests" / "fixtures" / "day_review" / "forecast_2026-09-17.md"


# ---------------------------------------------------------------------------
# the trader's own rows
# ---------------------------------------------------------------------------
def observing_and_predicting_entry(
    *,
    direction: str = "up",
    horizon: str = "rest_of_day",
    timeframe: str = "M5",
    observation: str = "Breadth is better than yesterday and SPY is over its VWAP.",
    because: str = "breadth",
    stamp: datetime | None = None,
) -> dict[str, Any]:
    """ONE Mentor row that both SAW something and CALLED something.

    TJ-14A keeps the two apart at the writer: `mentor.observation` is what the
    trader saw and `mentor.prediction` is the call they clicked. A pack that
    folded them into one item would let the story quote a description as a
    prediction, which is exactly what TJ-4's amendment forbids.
    """
    return fx.mentor_entry(
        direction=direction,
        horizon=horizon,
        timeframe=timeframe,
        text=observation,
        because=because,
        created_at=stamp or fx.STAMP,
    )


def observation_only_entry(
    *,
    text: str = "Everything I like is extended; I am sitting on my hands.",
    timeframe: str = "M5",
    stamp: datetime | None = None,
) -> dict[str, Any]:
    """An older Mentor row: `mentor` PRESENT and EMPTY, no click at all."""
    return fx.old_entry("empty", text=text, timeframe=timeframe, created_at=stamp or fx.STAMP)


def session_moment(session: str, hour: int = 10, minute: int = 5) -> datetime:
    """A Pacific stamp inside `session`'s regular hours."""
    from datetime import date as _date

    day = _date.fromisoformat(str(session)[:10])
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=PACIFIC)


def d1_click_entry(
    session: str,
    *,
    direction: str = "up",
    because: str = "higher lows since the gap",
    text: str = "",
    hour: int = 10,
) -> dict[str, Any]:
    """A D1 row with a CLICKED next-5-sessions call - a stated thesis."""
    return fx.mentor_entry(
        direction=direction,
        horizon="next_5_sessions",
        timeframe="D1",
        text=text,
        because=because,
        created_at=session_moment(session, hour),
        session=session,
    )


def d1_note_entry(session: str, *, text: str, hour: int = 11) -> dict[str, Any]:
    """A D1 row with WORDS and no click - `mentor` present and empty."""
    return fx.old_entry(
        "empty",
        text=text,
        timeframe="D1",
        session=session,
        created_at=session_moment(session, hour),
    )


def m5_note_entry(session: str, *, text: str, hour: int = 7) -> dict[str, Any]:
    """An M5 row, which the rolling D1 view may never treat as a thesis."""
    return fx.old_entry(
        "empty",
        text=text,
        timeframe="M5",
        session=session,
        created_at=session_moment(session, hour),
    )


def machine_entry(*, session: str = SESSION) -> dict[str, Any]:
    """A REAL desk-written row - the one kind that may never reach the pack."""
    import market_journal

    return market_journal.build_entry(
        text="Auto Pilot mode changed from DESK to EVENING.",
        session_date=session,
        timeframe="M5",
        symbols=["SPY"],
        origin=market_journal.ORIGIN_AUTO_MODE_FLIP,
        now=fx.STAMP + timedelta(hours=1),
    )


def forecast_entry(*, session: str = SESSION) -> dict[str, Any]:
    """The pasted brief, as the journal stores it: somebody ELSE's words."""
    import market_journal

    return market_journal.build_entry(
        text=FORECAST_FIXTURE.read_text(encoding="utf-8"),
        session_date=session,
        timeframe="D1",
        origin=market_journal.ORIGIN_EXTERNAL_FORECAST,
        now=fx.STAMP - timedelta(hours=1),
    )


def forecast_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The shape `DayReviewService._forecast` hands the page, built here."""
    import forecast_brief

    return {
        "entry_id": str(entry.get("entry_id") or ""),
        "text": str(entry.get("text") or ""),
        "created_at": str(entry.get("created_at") or ""),
        "source_model": "",
        "brief": forecast_brief.parse(str(entry.get("text") or "")),
    }


# ---------------------------------------------------------------------------
# TJ-10's rows, measured against TJ-10's own tape
# ---------------------------------------------------------------------------
def graded_reads(entries, *, session: str = SESSION, now: datetime | None = None):
    """`(reads, grades)` through the REAL grader - never a hand-typed verdict.

    The read rows carry the extra fields the Day Review payload adds to them
    (`verdict` and `observation`), because that payload is what the pack is
    assembled from.
    """
    import market_read_grades as grades

    moment = now or AFTER_THE_CLOSE
    rows = grades.read_rows(entries, session=session)
    tape = fx.session_tape()
    out_reads: list[dict[str, Any]] = []
    out_grades: list[dict[str, Any]] = []
    by_entry = {str(row.get("entry_id") or ""): row for row in entries or ()}
    for row in rows:
        grade = grades.grade_read(
            row,
            m5_bars=tape,
            daily_bars=(),
            atr=fx.atr_for(SESSION_MOVE, 3.0),
            now=moment,
            context={"availability": "available", "internals": {}},
        )
        entry = by_entry.get(str(row.get("entry_id") or "")) or {}
        mentor = entry.get("mentor") if isinstance(entry.get("mentor"), Mapping) else {}
        out_reads.append({
            **{key: value for key, value in row.items() if key != "context"},
            "verdict": grade["verdict"],
            "move_atr": grade["move_atr"],
            "checkpoints": grade["checkpoints"],
            "flat_band_rule": grade["flat_band_rule"],
            "grader_gap": grade["grader_gap"],
            "observation": str((mentor or {}).get("observation") or ""),
        })
        out_grades.append(grade)
    return out_reads, out_grades


def congruence(reads, *, session: str = SESSION, d1_label: str = "trending_up"):
    """TJ-10's three (or four) lines, through the real builder."""
    import market_read_grades as grades

    d1, d1_note = grades.select_read(reads, timeframe="D1")
    m5, m5_note = grades.select_read(reads, timeframe="M5")
    return grades.congruence_lines(
        session=session,
        d1_read=d1,
        d1_label=d1_label,
        decisions=(),
        claims=(),
        trades=(),
        d1_note=d1_note,
        m5_read=m5,
        m5_note=m5_note,
    )


# ---------------------------------------------------------------------------
# the rest of the pack's inputs
# ---------------------------------------------------------------------------
def daily_story(entries, *, session: str = SESSION):
    """`market_story.build_daily_story` over the SAME entries. Deterministic."""
    import market_story

    return market_story.build_daily_story(
        session,
        entries=entries,
        index_bars={"SPY": _daily_index_bars(session)},
        benchmarks=("SPY",),
    )


def _daily_index_bars(session: str) -> list[dict[str, Any]]:
    """Two completed daily bars for SPY: the day before, and the session."""
    return [
        {"dt": "2026-09-17", "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1_000_000},
        {"dt": session, "open": 100.0, "high": 102.5, "low": 99.5,
         "close": 102.0, "volume": 1_000_000},
    ]


def regime_shifts(*, session: str = SESSION) -> list[dict[str, Any]]:
    """The day's regime-shift rows, through `market_context_ledger`'s builder."""
    import market_context_ledger as ledger

    return [
        ledger.regime_shift_event(
            from_regime="neutral", to_regime="risk_on", source=ledger.SOURCE_AUTO,
            session_date=session, detail="SPY reclaimed its opening range",
            spy_day_pct=0.8,
        ),
        ledger.regime_shift_event(
            from_regime="risk_on", to_regime="neutral", source=ledger.SOURCE_USER,
            session_date=session, detail="trader forced neutral into the close",
            spy_day_pct=0.4,
        ),
    ]


def internals_marks(*, session: str = SESSION) -> list[dict[str, Any]]:
    """The open, ONE Mentor hour and the close - TJ-14A's v2 context each time.

    Built through `trade_mentor_context.build_context`, the ONE builder, so the
    pack cannot be given a shape the live card never writes.
    """
    import trade_mentor_context

    marks = []
    for kind, moment in (
        ("open", datetime(2026, 9, 18, 6, 30, tzinfo=PACIFIC)),
        ("mentor", datetime(2026, 9, 18, 7, 0, tzinfo=PACIFIC)),
        ("close", datetime(2026, 9, 18, 13, 0, tzinfo=PACIFIC)),
    ):
        marks.append({
            "kind": kind,
            "at": moment.isoformat(),
            "context": trade_mentor_context.build_context(
                now=moment,
                m5_bars={"SPY": fx.session_tape()},
                d1_bars={"SPY": _daily_index_bars(session)},
                sources={"m5": "fixture", "d1": "fixture"},
            ),
        })
    return marks


def walkaway_day_value(*, session: str = SESSION):
    """A real `WalkawayDay` with one rejected row, a skill block and sentences.

    Two closed horizons out of two, one of them a real miss: the skill cell's
    rate is 1/2, written out here so a pack that recomputed it would disagree.
    """
    import walkaway_day

    row = walkaway_day.WalkawayRow(
        decision_id=(session, "ABCL", "LONG", "chart_review", "veto", "annotations", "D1"),
        time=datetime(2026, 9, 18, 10, 5, tzinfo=PACIFIC),
        symbol="ABCL",
        side="LONG",
        category="chart_review",
        what_you_did="veto",
        ran_after_pct=4.0,
        state="measured",
        real_miss="real_miss",
        reason="extended",
    )
    return walkaway_day.WalkawayDay(
        liked_not_traded=(),
        rejected=(row,),
        traded_left_early=(),
        claimed_d1=(),
        earlier_calls=(),
        skill={
            "session": {
                "window_sessions": 1,
                "cells": [{
                    "population": "rejected", "side": "LONG", "setup_family": "",
                    "n": 2, "measured": 2, "pending": 0, "runs": 1, "rate": 0.5,
                }],
                "overlapping": (),
                "sentence": "This session: rejected 1 of 2 ran (n=2).",
            },
        },
        sentences={"rejected": "You rejected 1 name and it ran."},
        money={"n": 0, "net": None, "line": "too few to call (n=0)"},
    )


def trades(*, session: str = SESSION) -> list[dict[str, Any]]:
    """Two closed trades, as the shared trade journal hands them over."""
    return [
        {"trade_id": "t-1", "symbol": "ABCL", "direction": "LONG", "status": "closed",
         "trade_date": session, "closed_at": f"{session}T13:00:00-07:00",
         "realized_pnl": 180.0, "quantity": 100},
        {"trade_id": "t-2", "symbol": "ERAS", "direction": "SHORT", "status": "closed",
         "trade_date": session, "closed_at": f"{session}T12:10:00-07:00",
         "realized_pnl": -60.0, "quantity": 200},
    ]


# ---------------------------------------------------------------------------
# the whole set of inputs, as one call
# ---------------------------------------------------------------------------
def pack_inputs(*, with_forecast: bool = True, with_machine_row: bool = True) -> dict[str, Any]:
    """Every keyword `day_review_pack.build_pack` takes, built from real code."""
    clicked = observing_and_predicting_entry()
    noted = observation_only_entry()
    entries = [clicked, noted]
    forecast: dict[str, Any] = {}
    if with_forecast:
        pasted = forecast_entry()
        entries.insert(0, pasted)
        forecast = forecast_payload(pasted)
    if with_machine_row:
        entries.append(machine_entry())
    reads, _grades = graded_reads(entries)
    return {
        "entries": entries,
        "forecast": forecast,
        "story": daily_story(entries),
        "environment": regime_shifts(),
        "d1_label": "trending_up",
        "internals": internals_marks(),
        "walkaway": walkaway_day_value(),
        "reads": reads,
        "congruence": congruence(reads),
        "trades": trades(),
    }


def build(session: str = SESSION, *, now: datetime | None = None, **overrides):
    """`day_review_pack.build_pack` over :func:`pack_inputs`, with overrides."""
    import day_review_pack

    kwargs = pack_inputs(
        with_forecast=overrides.pop("with_forecast", True),
        with_machine_row=overrides.pop("with_machine_row", True),
    )
    kwargs.update(overrides)
    return day_review_pack.build_pack(session, now=now or AFTER_THE_CLOSE, **kwargs)


def list_sections(pack: Mapping[str, Any]) -> dict[str, list]:
    """Every section of the pack that is a LIST of items carrying source ids."""
    out: dict[str, list] = {}
    for name, value in (pack or {}).items():
        if name in ("reads", "congruence", "trader_said", "environment",
                    "measured", "internals", "trades"):
            out[name] = list(value or ())
    return out
