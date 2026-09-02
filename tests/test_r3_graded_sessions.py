"""R3 item 4 - the synthesis counter was reading a LIST as a COUNT.

`human_focus_tracking` writes `matured_horizons` as the horizons that matured,
joined with commas: `"1,3,5"`, `"1,3"`, `"1"`. `synthesis._matured` did
`int(float(value))`, which raises on every one of those except a bare `"1"` -
and a raise was swallowed as "not matured".

So the counter saw only the rows with EXACTLY ONE matured horizon. That is why
it read 2 on 2026-09-01 and 1 on 2026-09-02: **it went DOWN as more evidence
accrued**, because rows whose value grew from "1" to "1,3" stopped parsing.

Measured on the live files on 2026-09-02: 176 matured veto rows and 53 matured
like rows across FOUR distinct trade_dates, and the counter returned 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _row(trade_date: str, matured: str) -> dict:
    return {
        "trade_date": trade_date,
        "symbol": "AAA",
        "side": "LONG",
        "source": "focus_pick",
        "matured_horizons": matured,
        "fully_matured": "0",
    }


def test_a_comma_separated_horizon_list_counts_as_matured():
    """The shape the writer actually produces."""
    from ai_jobs import synthesis

    assert synthesis._matured(_row("2026-08-20", "1,3,5")) is True
    assert synthesis._matured(_row("2026-08-20", "1,3")) is True
    assert synthesis._matured(_row("2026-08-20", "1")) is True
    assert synthesis._matured(_row("2026-08-20", "10")) is True


def test_nothing_matured_is_still_nothing():
    from ai_jobs import synthesis

    assert synthesis._matured(_row("2026-08-20", "")) is False
    assert synthesis._matured(_row("2026-08-20", "0")) is False
    assert synthesis._matured(_row("2026-08-20", "   ")) is False
    assert synthesis._matured({"trade_date": "2026-08-20"}) is False


def test_five_graded_dates_count_as_five():
    """The packet's fixture: five distinct dates, in the real field shape."""
    from ai_jobs import synthesis

    veto = [
        _row("2026-08-20", "1,3,5"),
        _row("2026-08-21", "1,3"),
        _row("2026-08-27", "1,3,5"),
        _row("2026-08-31", "1"),
        # Two rows on one date must not count twice, and an ungraded row on a
        # sixth date must not count at all.
        _row("2026-08-31", "1,3"),
        _row("2026-09-01", ""),
    ]
    like = [_row("2026-08-28", "1,3,5,10")]

    assert synthesis.graded_sessions(veto, like) == 5


def test_the_count_can_never_fall_as_a_row_matures_further():
    """The symptom that gave the bug away: 2 on one night, 1 the next.

    A row whose horizons grow from "1" to "1,3" is MORE graded, not less. Any
    predicate that can flip to False as evidence accrues is wrong by
    construction.
    """
    from ai_jobs import synthesis

    early = [_row("2026-08-20", "1"), _row("2026-08-21", "1")]
    later = [_row("2026-08-20", "1,3"), _row("2026-08-21", "1,3,5")]

    assert synthesis.graded_sessions(early, []) == 2
    assert synthesis.graded_sessions(later, []) >= synthesis.graded_sessions(early, [])
    assert synthesis.graded_sessions(later, []) == 2
