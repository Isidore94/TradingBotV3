"""R10.0b §1.3: pin the durable daily-bar source to Yahoo.

Interim measure while the cliff packet (R10.V) is built. The store is mixed:
1,227 of 1,737 measurable parquet files carry a >20x volume step because IB
returns regular-session volume in round lots (`useRTH=1`, `whatToShow="TRADES"`,
`master_avwap_lib/legacy.py:15245-15256`) while Yahoo returns the full
consolidated session in shares. The ratio is symbol-dependent (SPY 1.0x, TSLA
56x, AAPL 81x, A 162x, NVDA 188x), so no constant converts one into the other -
which is exactly why this is a pin and not a rescale.

`_IBKR_HISTORICAL_YAHOO_ONLY` is a circuit-breaker *state*, flipped by repeated
IB failures and cleared each scan. This is a *setting*, read at the same seam,
and the two are independent: either one alone sends daily bars to Yahoo.

Intraday bars are deliberately untouched - the trader authorized the daily seam.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_latch():
    master_avwap.reset_ibkr_historical_failure_circuit()
    yield
    master_avwap.reset_ibkr_historical_failure_circuit()


def _yahoo_frame():
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-08-20", "2026-08-21"]),
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
            "close": [1.0, 1.0], "volume": [45_479_200, 5_286_472],
        }
    )


class _FakeIb:
    """Stands in for a live IBApi. Touching it at all is a test failure."""

    def __init__(self):
        self.data, self.ready, self.calls = {}, {}, []

    def reqHistoricalData(self, *a, **k):
        self.calls.append(a)
        raise AssertionError("the pin must return before any IB request is made")


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------
def test_the_default_is_unchanged_behaviour():
    """Absent key = auto = whatever the scan does today. No silent policy change."""
    with mock.patch.object(master_avwap, "get_local_setting", return_value=None):
        assert master_avwap.daily_bars_source_pin() == "auto"
    with mock.patch.object(master_avwap, "get_local_setting", return_value="auto"):
        assert master_avwap.daily_bars_source_pin() == "auto"


@pytest.mark.parametrize("raw", ["yahoo", "YAHOO", "  Yahoo  "])
def test_the_pin_is_read_case_and_space_insensitively(raw):
    with mock.patch.object(master_avwap, "get_local_setting", return_value=raw):
        assert master_avwap.daily_bars_source_pin() == "yahoo"


def test_an_unrecognised_value_falls_back_to_auto_rather_than_guessing():
    """A typo must not silently pin, and must not silently un-pin either."""
    with mock.patch.object(master_avwap, "get_local_setting", return_value="yahooo"):
        assert master_avwap.daily_bars_source_pin() == "auto"


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------
def test_the_pin_returns_yahoo_without_ever_calling_ib():
    ib = _FakeIb()
    with mock.patch.object(master_avwap, "get_local_setting", return_value="yahoo"), \
            mock.patch.object(master_avwap, "fetch_daily_bars_from_yahoo",
                              return_value=_yahoo_frame()) as yahoo:
        out = master_avwap._fetch_live_daily_bars(ib, "NVDA", 30)
    assert len(out) == 2
    assert yahoo.call_count == 1
    assert ib.calls == [], "a pinned scan must spend no IB budget on daily bars"


def test_without_the_pin_the_ib_path_is_still_attempted():
    """The pin is the only thing this change adds; auto behaviour is untouched.

    The IB branch catches its own failures and falls back to Yahoo, which is the
    production behaviour and not what is under test here - so this asserts that
    the request was *reached*, not that an exception escaped.
    """
    ib = _FakeIb()
    with mock.patch.object(master_avwap, "get_local_setting", return_value="auto"), \
            mock.patch.object(master_avwap, "fetch_daily_bars_from_yahoo",
                              return_value=_yahoo_frame()):
        master_avwap._fetch_live_daily_bars(ib, "NVDA", 30)
    assert ib.calls, "with no pin, the IB daily request must still be made"


def test_the_pin_is_independent_of_the_failure_circuit():
    """Either one alone routes to Yahoo; neither implies the other."""
    ib = _FakeIb()
    with mock.patch.object(master_avwap, "get_local_setting", return_value="yahoo"), \
            mock.patch.object(master_avwap, "fetch_daily_bars_from_yahoo",
                              return_value=_yahoo_frame()):
        master_avwap._fetch_live_daily_bars(ib, "NVDA", 30)
    assert master_avwap._ibkr_historical_yahoo_only() is False, \
        "the setting must not flip the circuit-breaker state"


def test_it_says_which_source_is_pinned_once_per_scan_not_once_per_symbol(caplog):
    """1,500 symbols must not produce 1,500 identical lines."""
    import logging

    ib = _FakeIb()
    with mock.patch.object(master_avwap, "get_local_setting", return_value="yahoo"), \
            mock.patch.object(master_avwap, "fetch_daily_bars_from_yahoo",
                              return_value=_yahoo_frame()):
        with caplog.at_level(logging.INFO):
            for symbol in ("NVDA", "AAPL", "TSLA", "A"):
                master_avwap._fetch_live_daily_bars(ib, symbol, 30)
    pinned = [r for r in caplog.records if "daily_bars_source" in r.getMessage()]
    assert len(pinned) == 1, f"expected one line, got {len(pinned)}"
    assert "yahoo" in pinned[0].getMessage()

    # ...and the next scan says it again, because the latch resets with the circuit.
    caplog.clear()
    master_avwap.reset_ibkr_historical_failure_circuit()
    with mock.patch.object(master_avwap, "get_local_setting", return_value="yahoo"), \
            mock.patch.object(master_avwap, "fetch_daily_bars_from_yahoo",
                              return_value=_yahoo_frame()):
        with caplog.at_level(logging.INFO):
            master_avwap._fetch_live_daily_bars(ib, "NVDA", 30)
    assert len([r for r in caplog.records if "daily_bars_source" in r.getMessage()]) == 1


def test_intraday_bars_are_deliberately_not_pinned():
    """The trader authorized the daily seam. Intraday keeps its own behaviour."""
    import inspect

    source = inspect.getsource(master_avwap._fetch_live_intraday_bars)
    assert "daily_bars_source_pin" not in source
