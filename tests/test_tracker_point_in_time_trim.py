"""R10.V step 7 - a recompute may not see a bar that came after its session.

The S2 defect, reproduced from the frozen pre/post tracker pair: the payload's
`data_session` said 2026-08-20 while **2,739 setups carried a `latest_snapshot`
dated 2026-08-21**, and 452 scenario exit events sat on that same bar. The cause
is the catch-up holding today's daily frames while replaying a past session, so a
recompute FOR 08-20 could advance a setup on 08-21's bar.

Evidence-only by construction: the recompute writes the tracker payload, which
nothing in the live decision path reads back as a signal.

The rule the tests pin: **an unparseable session does not trim.** This is a
point-in-time guard, not a filter, and silently emptying every frame because a
stamp was malformed would be a far worse failure than the one it prevents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402


def _frame(days: int = 5, *, start: str = "2026-08-17"):
    stamps = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {
            "datetime": stamps,
            "open": [10.0 + i for i in range(days)],
            "high": [10.5 + i for i in range(days)],
            "low": [9.5 + i for i in range(days)],
            "close": [10.2 + i for i in range(days)],
            "volume": [1_000_000.0] * days,
        }
    )


def _dates(frame) -> list[str]:
    return [stamp.date().isoformat() for stamp in pd.to_datetime(frame["datetime"])]


def test_bars_after_the_session_are_dropped():
    trimmed = master_avwap._trim_daily_frame_to_session(_frame(5), "2026-08-19")
    assert _dates(trimmed) == ["2026-08-17", "2026-08-18", "2026-08-19"]


def test_the_session_itself_is_kept():
    """`<=`, not `<`: the session being recomputed is the session we have."""
    trimmed = master_avwap._trim_daily_frame_to_session(_frame(5), "2026-08-21")
    assert _dates(trimmed)[-1] == "2026-08-21"


def test_a_frame_that_needs_no_trimming_is_returned_unchanged():
    frame = _frame(3)
    assert master_avwap._trim_daily_frame_to_session(frame, "2026-08-25") is frame


def test_the_provenance_attribute_survives_a_trim():
    frame = master_avwap._set_daily_bar_source(_frame(5), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    trimmed = master_avwap._trim_daily_frame_to_session(frame, "2026-08-19")
    assert master_avwap._get_daily_bar_source(trimmed) == "yahoo"


def test_a_full_timestamp_is_read_as_its_date():
    trimmed = master_avwap._trim_daily_frame_to_session(_frame(5), "2026-08-19T16:00:00")
    assert _dates(trimmed)[-1] == "2026-08-19"


@pytest.mark.parametrize("session", ["", None, "not-a-date", "2026-13-45"])
def test_an_unparseable_session_does_not_trim(session):
    """A malformed stamp must not empty every frame in the run."""
    frame = _frame(5)
    trimmed = master_avwap._trim_daily_frame_to_session(frame, session)
    assert len(trimmed) == 5


def test_a_compact_date_is_understood():
    """`20260819` is a date, not a malformed stamp, and trims like one."""
    trimmed = master_avwap._trim_daily_frame_to_session(_frame(5), "20260819")
    assert _dates(trimmed)[-1] == "2026-08-19"


def test_an_empty_or_column_less_frame_is_handled():
    assert master_avwap._trim_daily_frame_to_session(None, "2026-08-19") is None
    empty = pd.DataFrame()
    assert master_avwap._trim_daily_frame_to_session(empty, "2026-08-19") is empty
    no_column = pd.DataFrame({"close": [1.0]})
    assert len(master_avwap._trim_daily_frame_to_session(no_column, "2026-08-19")) == 1


def test_a_session_before_every_bar_leaves_nothing_to_recompute():
    """The caller skips the setup rather than marking it from nothing."""
    trimmed = master_avwap._trim_daily_frame_to_session(_frame(5), "2026-01-01")
    assert trimmed is not None and trimmed.empty


def test_unparseable_bar_stamps_are_dropped_not_kept():
    frame = _frame(3)
    frame.loc[1, "datetime"] = pd.NaT
    trimmed = master_avwap._trim_daily_frame_to_session(frame, "2026-08-19")
    assert _dates(trimmed) == ["2026-08-17", "2026-08-19"]


def test_the_recompute_applies_the_trim_before_it_computes_anything():
    """The guard has to sit before BOTH the indicator frame and the record.

    An indicator frame built from untrimmed bars would carry the future into a
    record built from trimmed ones, which is the same defect wearing a hat.
    """
    import inspect

    source = inspect.getsource(master_avwap.update_setup_tracker_from_scan)
    trim_at = source.index("_trim_daily_frame_to_session(df, target_scan_date)")
    indicators_at = source.index("compute_indicator_frame(df)")
    record_at = source.index("recompute_tracker_setup_record(")
    assert trim_at < indicators_at < record_at


def test_the_indicator_cache_is_keyed_by_session_as_well_as_symbol():
    """The frame it is built from is now session-dependent."""
    import inspect

    source = inspect.getsource(master_avwap.update_setup_tracker_from_scan)
    assert "indicator_key = (symbol, target_scan_date)" in source
    assert "indicator_frame_cache.get(indicator_key)" in source
