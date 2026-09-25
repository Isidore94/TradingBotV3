"""TJ-11 blocker 2 - a pooled rate counts only names whose horizon has CLOSED.

Reviewer NO-GO, 2026-09-19: the first build kept an EARLY run inside an open
five-session horizon as `run` (correct for the row - a run cannot be taken
back) but also pooled it, while every no-run of the same unfinished window was
held back as `unmeasured:horizon_open`. So the numerator could grow while the
denominator could not, and every open-horizon cell came out 100% by
construction. Measured on a copy of the live stores: the lately LONG rejected
cell shipped **34% (30/87)** where the closed-horizon-only truth was
**26% (20/77)**.

The rule this file pins: a name enters a pooled rate only when its horizon has
closed - runs and no-runs alike. An open-horizon name is in NEITHER half and is
counted and printed as its own number, `pending P`. The ROW may still show its
early run; a rate may not.

Plain dicts only: no store, no clock, no network.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SESSION = "2026-09-18"
#: Flat pre-history: every bar spans exactly 2.00 around a 100.00 close, so
#: Wilder ATR(14) is exactly 2.00 and one ATR is exactly 2%.
ATR = 2.0


def _sessions_ending(session: str, count: int) -> list[str]:
    from market_calendar import previous_session

    cursor = date.fromisoformat(session)
    out = [cursor.isoformat()]
    for _ in range(count - 1):
        cursor = previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _sessions_after(session: str, count: int) -> list[str]:
    from market_calendar import is_session

    cursor = date.fromisoformat(session)
    out: list[str] = []
    while len(out) < count:
        cursor += timedelta(days=1)
        if is_session(cursor):
            out.append(cursor.isoformat())
    return out


def _flat_daily(session: str = SESSION, *, bars: int = 15, close: float = 100.0) -> list[dict]:
    return [
        {"dt": f"{day}T00:00:00", "open": close, "high": close + 1, "low": close - 1, "close": close}
        for day in _sessions_ending(session, bars)
    ]


def _daily_after(session: str, rows) -> list[dict]:
    days = _sessions_after(session, len(rows))
    return [
        {"dt": f"{day}T00:00:00", "open": o, "high": h, "low": low, "close": c}
        for day, (o, h, low, c) in zip(days, rows, strict=False)
    ]


#: Day one is +3% (1.5 ATR) with nothing against it: a run, and a run that is
#: already decided on the FIRST session of a five-session horizon.
_RUN_DAY = (100.0, 103.0, 99.9, 102.5)
_QUIET_DAY = (102.5, 103.0, 102.0, 102.5)
#: -1.5% (0.75 ATR) against before anything in favour: a no-run.
_NO_RUN_DAY = (100.0, 100.5, 98.5, 98.6)
_FLAT_DAY = (98.6, 99.0, 98.0, 98.5)


def _daily(*, run: bool, sessions_after: int, session: str = SESSION) -> list[dict]:
    """`sessions_after` closed sessions after the scan, running or not."""
    first = _RUN_DAY if run else _NO_RUN_DAY
    rest = _QUIET_DAY if run else _FLAT_DAY
    after = [first] + [rest] * max(0, sessions_after - 1)
    return _flat_daily(session) + _daily_after(session, after)


def _scan_rows(names, side: str = "LONG", family: str = "general", session: str = SESSION):
    return [
        {
            "symbol": symbol, "side": side, "scan_date": session,
            "horizon_sessions": str(horizon), "setup_family": family,
        }
        for symbol in names
        for horizon in (1, 3, 5, 10)
    ]


def _build(*, scan_rows=(), daily_bars=None, now, decisions=()):
    from walkaway_day import build

    return build(
        SESSION,
        sources={
            "decisions": tuple(decisions),
            "preference": (),
            "outcomes": (),
            "scan_rows": tuple(scan_rows),
            "earlier_decisions": (),
        },
        bars={},
        now=now,
        daily_bars=daily_bars or {},
    )


def _cell(day, population: str, *, window: str = "session", side: str = "LONG", family: str = ""):
    cells = [
        cell for cell in day.skill[window]["cells"]
        if cell["population"] == population and cell["side"] == side
        and cell.get("setup_family", "") == family
    ]
    assert len(cells) == 1, (population, day.skill[window]["cells"])
    return cells[0]


# -- the rule ----------------------------------------------------------------


def test_an_early_run_inside_an_open_horizon_is_pending_and_not_a_rate(tmp_path):
    """Ten names: five ran on day one, five did not, none has closed.

    Before the fix this cell read 100% (5/5) - every no-run held back and every
    early run counted. It now reads nothing at all, because nothing has
    finished.
    """
    runners = [f"R{i}" for i in range(5)]
    quiet = [f"Q{i}" for i in range(5)]
    daily = {name: _daily(run=True, sessions_after=1) for name in runners}
    daily |= {name: _daily(run=False, sessions_after=1) for name in quiet}

    day = _build(
        scan_rows=_scan_rows(runners + quiet),
        daily_bars=daily,
        # 2026-09-21 has closed; the five-session horizon (2026-09-25) has not.
        now=datetime(2026, 9, 22, 8, 0),
    )

    cell = _cell(day, "untouched")
    assert (cell["n"], cell["measured"], cell["pending"]) == (10, 0, 10)
    assert cell["runs"] == 0
    assert cell["rate"] is None
    assert cell["low"] is None and cell["high"] is None
    assert cell["reportable"] is False


def test_the_pooled_rate_is_the_closed_horizons_and_nothing_else():
    """40 closed names at 25%, plus 20 open ones that all ran on day one.

    The censored answer would be 30/50 = 60%. The rule's answer is 10/40 = 25%,
    with the 20 unfinished names printed as `pending`.
    """
    from swing_headline import wilson_lower_bound

    closed_runs = [f"C{i:02d}" for i in range(10)]
    closed_quiet = [f"D{i:02d}" for i in range(30)]
    open_runs = [f"O{i:02d}" for i in range(20)]
    daily = {name: _daily(run=True, sessions_after=5) for name in closed_runs}
    daily |= {name: _daily(run=False, sessions_after=5) for name in closed_quiet}
    daily |= {name: _daily(run=True, sessions_after=1) for name in open_runs}

    # The closed names were scanned five sessions earlier, so their horizon has
    # run out; the open ones were scanned on the selected session.
    rows = _scan_rows(closed_runs + closed_quiet, session="2026-09-11")
    rows += _scan_rows(open_runs, session=SESSION)

    day = _build(
        scan_rows=rows, daily_bars={
            **{name: _flat_daily("2026-09-11") + _daily_after(
                "2026-09-11", [_RUN_DAY] + [_QUIET_DAY] * 4) for name in closed_runs},
            **{name: _flat_daily("2026-09-11") + _daily_after(
                "2026-09-11", [_NO_RUN_DAY] + [_FLAT_DAY] * 4) for name in closed_quiet},
            **{name: _daily(run=True, sessions_after=1) for name in open_runs},
        },
        now=datetime(2026, 9, 22, 8, 0),
    )

    cell = _cell(day, "untouched", window="lately")
    assert (cell["n"], cell["measured"], cell["pending"]) == (60, 40, 20)
    assert cell["runs"] == 10
    assert cell["rate"] == pytest.approx(0.25)
    assert cell["low"] == pytest.approx(wilson_lower_bound(10, 40))
    assert cell["low"] < 0.25 < cell["high"]
    assert cell["reportable"] is True


def test_a_session_whose_horizons_are_all_open_names_no_rate_and_says_pending():
    names = [f"S{i:02d}" for i in range(35)]
    day = _build(
        scan_rows=_scan_rows(names),
        daily_bars={name: _daily(run=True, sessions_after=1) for name in names},
        now=datetime(2026, 9, 22, 8, 0),
    )

    sentence = day.skill["session"]["sentence"]
    assert "measured 0" in sentence, sentence
    assert "pending 35" in sentence, sentence
    assert "%" not in sentence, sentence
    assert day.skill["session"]["overlapping"] == ()


def test_every_cell_always_prints_measured_and_pending():
    """Not only when they differ: what a rate was computed over IS the rate."""
    names = [f"S{i:02d}" for i in range(40)]
    day = _build(
        scan_rows=_scan_rows(names, session="2026-09-11"),
        daily_bars={
            name: _flat_daily("2026-09-11")
            + _daily_after("2026-09-11", [_RUN_DAY] + [_QUIET_DAY] * 4)
            for name in names
        },
        now=datetime(2026, 9, 22, 8, 0),
    )

    sentence = day.skill["lately"]["sentence"]
    assert "n 40, measured 40, pending 0" in sentence, sentence


def test_the_row_still_shows_its_early_run_even_though_the_rate_does_not():
    """The row is a fact about one name; the rate is a claim about a group."""
    decision = {
        "session_date": SESSION, "symbol": "R0", "side": "LONG",
        "verdict": "veto", "source": "annotations", "timeframe": "D1",
        "stamp": "2026-09-18T13:00:00-04:00", "category": "chart_review",
        "decision_session": SESSION,
    }
    day = _build(
        decisions=[decision],
        scan_rows=_scan_rows(["R0"]),
        daily_bars={"R0": _daily(run=True, sessions_after=1)},
        now=datetime(2026, 9, 22, 8, 0),
    )

    row = day.rejected[0]
    assert row.real_miss == "run"
    assert row.state.startswith("pending "), row.state
    cell = _cell(day, "rejected")
    assert (cell["measured"], cell["pending"], cell["runs"]) == (0, 1, 0)
