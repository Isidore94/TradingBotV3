"""R4 A8 - the daily floors are measured on CLOSED bars, over enough of them.

Three defects, one packet item:

* the daily download had no completed-bar filter, so today's FORMING daily bar
  went straight into the 100 and 200 SMA and the floor a row was greyed against
  moved on every refresh - most at 09:31, when today's "close" is nine minutes
  of trading;
* `DAILY_FETCH_PERIOD` was `1y`, ~252 sessions against a 200-close requirement,
  which leaves ~52 sessions of slack for a listing date, a provider gap and a
  holiday run to spend;
* `autopilot_core._frame_rows` coerced a missing volume to `0.0`, and a zero is
  a measurement - it reaches the relative volume as "this bar traded nothing",
  which is a claim about the tape rather than about the download.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _Frame:
    """The per-symbol frame `_frame_rows` reads, with a settable Volume."""

    def __init__(self, rows):
        self._rows = rows
        self.empty = not rows

    def iterrows(self):
        for row in self._rows:
            yield row["dt"], {
                "Open": row["open"],
                "High": row["high"],
                "Low": row["low"],
                "Close": row["close"],
                "Volume": row["volume"],
            }


def _row(volume, *, when=datetime(2026, 7, 2, 9, 35)):
    return {
        "dt": when,
        "open": 10.0,
        "high": 10.5,
        "low": 9.5,
        "close": 10.2,
        "volume": volume,
    }


# ---------------------------------------------------------------------------
# A missing volume is None
# ---------------------------------------------------------------------------


def test_a_missing_volume_is_blank_and_never_a_measured_zero():
    from autopilot_core import _frame_rows

    rows = _frame_rows(_Frame([_row(None), _row(float("nan")), _row(-5.0), _row(900.0)]))

    assert [row["volume"] for row in rows] == [None, None, None, 900.0]


def test_a_blank_volume_makes_the_relative_volume_blank_rather_than_zero():
    """The whole point of A8's third defect, measured end to end.

    A zero would rank the symbol at the bottom of a volume filter it was never
    eligible for; a blank says "not measured", which is what happened.
    """
    from strength_scan import relative_volume

    bars = []
    start = datetime(2026, 6, 1, 6, 30)
    for session in range(16):
        day = start + timedelta(days=session)
        for index in range(78):
            bars.append(
                {
                    "dt": day + timedelta(minutes=5 * index),
                    "volume": 1000.0,
                }
            )
    assert relative_volume(bars) == pytest.approx(1.0)

    bars[-1]["volume"] = None
    assert relative_volume(bars) is None


def test_a_frame_row_with_a_readable_volume_is_unchanged():
    """The fix must not turn a real zero-volume bar into a blank.

    A bar that genuinely printed no volume is data; only an unreadable one is a
    gap. Zero is kept, negative is not - a negative volume is not a quantity.
    """
    from autopilot_core import _frame_rows

    rows = _frame_rows(_Frame([_row(0.0)]))
    assert rows[0]["volume"] == 0.0
    assert not math.isnan(rows[0]["volume"])


# ---------------------------------------------------------------------------
# The forming daily bar
# ---------------------------------------------------------------------------


def _daily_rows(last_day: date, count: int = 5) -> list[dict]:
    return [
        {
            "dt": datetime.combine(last_day - timedelta(days=count - 1 - index), datetime.min.time()),
            "close": 100.0 + index,
        }
        for index in range(count)
    ]


def test_todays_forming_daily_bar_never_reaches_the_sma():
    """Wednesday 2026-09-02 at 11:00 ET: today's row is nine minutes old."""
    from zoneinfo import ZoneInfo

    from ui.services.strength_board_service import _completed_daily_rows

    et = ZoneInfo("America/New_York")
    rows = _daily_rows(date(2026, 9, 2))
    kept = _completed_daily_rows(rows, now=datetime(2026, 9, 2, 11, 0, tzinfo=et))

    assert [row["dt"].date() for row in kept][-1] == date(2026, 9, 1)
    assert len(kept) == len(rows) - 1


def test_after_the_close_todays_bar_is_a_completed_bar():
    """The filter is "closed", not "not today" - or the board would lag a day."""
    from zoneinfo import ZoneInfo

    from ui.services.strength_board_service import _completed_daily_rows

    et = ZoneInfo("America/New_York")
    rows = _daily_rows(date(2026, 9, 2))
    kept = _completed_daily_rows(rows, now=datetime(2026, 9, 2, 20, 0, tzinfo=et))

    assert [row["dt"].date() for row in kept][-1] == date(2026, 9, 2)
    assert len(kept) == len(rows)


def test_a_row_with_no_readable_date_is_kept_rather_than_guessed_at():
    from zoneinfo import ZoneInfo

    from ui.services.strength_board_service import _completed_daily_rows

    et = ZoneInfo("America/New_York")
    rows = [{"dt": None, "close": 10.0}]
    kept = _completed_daily_rows(rows, now=datetime(2026, 9, 2, 11, 0, tzinfo=et))

    assert kept == rows


# ---------------------------------------------------------------------------
# The fetch window
# ---------------------------------------------------------------------------


def test_the_daily_window_holds_more_than_a_years_slack_over_the_200_sma():
    """~504 sessions against a 200-close requirement, not ~252."""
    from ui.services import strength_board_service

    assert strength_board_service.DAILY_FETCH_PERIOD == "2y"


def test_the_daily_download_asks_for_the_declared_period():
    """The constant is only worth pinning if the call actually uses it."""
    from ui.services.strength_board_service import DAILY_FETCH_PERIOD, _daily_closes

    asked: list[dict] = []

    def _downloader(chunk, **kwargs):
        asked.append(kwargs)
        raise RuntimeError("no data needed; the ASK is what is under test")

    _daily_closes(["AAA"], _downloader, chunk_size=10, now=datetime(2026, 9, 2, 20, 0))

    assert asked and asked[0]["period"] == DAILY_FETCH_PERIOD
    assert asked[0]["interval"] == "1d"
