"""R10.A / MAJOR-4 - a measurement may not come from the forming bar.

`_update_pending_bounce_outcomes` gets its frame from a request with an empty
`endDateTime`, so the last row is the bar still forming. Measuring it wrote a
forming bar into `last_measured`, and when the session ended without another
update that forming bar became the basis of a **final**. plan.md sec 5:
completed bars only for state transitions; a forming bar is preview.

The blast radius is one caller. `_rows_after_bounce_entry_for_session` is called
from `_update_pending_bounce_outcomes` and nowhere else, and that function writes
outcome rows only - no detector, score, gate or alert reads it. The test below
asserts that rather than trusting the grep that established it.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _Host:
    pass


def _host():
    from bounce_bot_lib.legacy import BounceBot

    host = _Host.__new__(_Host)
    host.OUTCOME_BAR_MINUTES = BounceBot.OUTCOME_BAR_MINUTES
    host._naive_market_local = BounceBot._naive_market_local
    host._completed_session_rows = BounceBot._completed_session_rows.__get__(host, _Host)
    host._rows_after_bounce_entry_for_session = (
        BounceBot._rows_after_bounce_entry_for_session.__get__(host, _Host)
    )
    return host


def _frame(count: int, *, start="2026-08-21 07:00:00"):
    stamps = pd.date_range(start, periods=count, freq="5min")
    return pd.DataFrame(
        {
            "datetime": stamps,
            "open": [100.0] * count,
            "high": [100.5] * count,
            "low": [99.5] * count,
            "close": [100.2] * count,
            "volume": [1000.0] * count,
        }
    )


STATE = {"trade_date": "2026-08-21", "entry_time": "2026-08-21T06:55:00"}
ENTRY = datetime(2026, 8, 21, 6, 55)


def test_the_forming_bar_is_dropped():
    """07:20 has not finished at 07:23. The last COMPLETED bar starts 07:15."""
    host = _host()
    rows = host._rows_after_bounce_entry_for_session(
        STATE, _frame(5), ENTRY, now=datetime(2026, 8, 21, 7, 23)
    )
    assert list(rows["datetime"].dt.strftime("%H:%M")) == ["07:00", "07:05", "07:10", "07:15"]


def test_a_bar_that_just_closed_is_kept():
    """The boundary is INCLUSIVE - a strict `<` discards the bar that just closed."""
    host = _host()
    rows = host._rows_after_bounce_entry_for_session(
        STATE, _frame(5), ENTRY, now=datetime(2026, 8, 21, 7, 25)
    )
    assert list(rows["datetime"].dt.strftime("%H:%M"))[-1] == "07:20"


def test_a_finished_session_keeps_every_bar():
    host = _host()
    rows = host._rows_after_bounce_entry_for_session(
        STATE, _frame(5), ENTRY, now=datetime(2026, 8, 21, 14, 0)
    )
    assert len(rows) == 5


def test_an_undateable_bar_is_dropped_because_it_is_not_known_to_be_complete():
    """The shared rule's stance, and the right one: uncertainty is not confirmation.

    In the live path such a row cannot reach here - the caller has already
    coerced and dropped it - so this pins the rule rather than a live case.
    """
    host = _host()
    rows = _frame(3)
    rows["datetime"] = ["not", "a", "time"]
    assert len(host._completed_session_rows(rows, now=datetime(2026, 8, 21, 7, 23))) == 0


def test_a_failure_of_the_cut_itself_returns_the_frame_whole(monkeypatch):
    """A machine that cannot tell the time must not stop measuring altogether."""
    import builtins

    host = _host()
    rows = _frame(3)
    real_import = builtins.__import__

    def angry(name, *args, **kwargs):
        if name == "completed_bars":
            raise ImportError("no clock today")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", angry)
    assert len(host._completed_session_rows(rows, now=datetime(2026, 8, 21, 7, 23))) == 3


def test_an_empty_frame_stays_empty():
    host = _host()
    assert host._completed_session_rows(pd.DataFrame(), now=datetime(2026, 8, 21, 7, 23)).empty


def test_the_forbidden_timezone_spelling_is_gone_from_this_path():
    """plan.md sec 5: `astimezone`, never `replace(tzinfo=None)` on its own."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    for function in (
        BounceBot._rows_after_bounce_entry_for_session,
        BounceBot._append_bounce_outcome_row,
    ):
        source = inspect.getsource(function)
        assert "replace(tzinfo=None)" not in source, function.__name__

    helper = inspect.getsource(BounceBot._naive_market_local)
    assert "astimezone(" in helper, "the conversion is explicit before the drop"


def test_only_the_outcome_writer_reads_this_helper():
    """The proof MAJOR-4's authorization was conditional on."""
    import inspect

    from bounce_bot_lib import legacy

    source = inspect.getsource(legacy)
    calls = source.count("_rows_after_bounce_entry_for_session(")
    assert calls == 2, f"one definition and one caller, found {calls} occurrences"
    caller = inspect.getsource(legacy.BounceBot._update_pending_bounce_outcomes)
    assert "_rows_after_bounce_entry_for_session(" in caller
    # ...and that caller writes outcome rows and nothing else.
    assert "_append_bounce_outcome_row(" in caller
    for forbidden in ("alert", "tier", "score", "watchlist", "focus"):
        assert forbidden not in caller.lower(), f"the caller touches {forbidden}"
