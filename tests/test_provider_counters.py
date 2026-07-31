"""Provider request/cache/throttle/failure counters (plan.md sec 6.3 bullet 9).

Before this module existed the string "provider_calls" appeared exactly once in
the repo - as a made-up example counter in a manifest test - and the Health
page's provider dimension was a permanent UNKNOWN.  These tests pin the three
layers: the counter store itself, the fetch-boundary instrumentation in the
legacy scan (counting only - behaviour must be untouched), and the manifest
flush that carries the counts to the operations audit.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnostics import provider_counters  # noqa: E402
import master_avwap_lib.legacy as legacy  # noqa: E402


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
def test_record_snapshot_totals_and_reset():
    provider_counters.reset()
    provider_counters.record("daily_bars", "request")
    provider_counters.record("daily_bars", "cache_hit", 3)
    provider_counters.record("intraday_bars", "request")
    provider_counters.record("ibkr_historical", "throttle")

    assert provider_counters.snapshot() == {
        "daily_bars.cache_hit": 3,
        "daily_bars.request": 1,
        "ibkr_historical.throttle": 1,
        "intraday_bars.request": 1,
    }
    assert provider_counters.totals() == {
        "request": 2,
        "cache_hit": 3,
        "failure": 0,
        "throttle": 1,
    }
    provider_counters.reset()
    assert provider_counters.snapshot() == {}


def test_record_never_raises_on_junk():
    provider_counters.reset()
    provider_counters.record(None, None)  # type: ignore[arg-type]
    provider_counters.record("x", "y", "not-a-number")  # type: ignore[arg-type]
    # The malformed increment is dropped; the well-formed key survives.
    provider_counters.record("daily_bars", "request")
    assert provider_counters.snapshot()["daily_bars.request"] == 1
    provider_counters.reset()


class _FakeRecorder:
    def __init__(self):
        self.counters: dict[str, int] = {}

    def set_counter(self, key, value):
        self.counters[key] = value


def test_flush_stamps_captured_even_when_all_zero():
    provider_counters.reset()
    recorder = _FakeRecorder()
    provider_counters.flush_to_manifest(recorder)
    # "Measured and zero" must be distinguishable from "not measured".
    assert recorder.counters == {"provider.captured": 1}

    provider_counters.record("daily_bars", "request", 2)
    provider_counters.record("daily_bars", "failure")
    provider_counters.flush_to_manifest(recorder)
    assert recorder.counters["provider.daily_bars.request"] == 2
    assert recorder.counters["provider.daily_bars.failure"] == 1
    provider_counters.reset()


# ---------------------------------------------------------------------------
# the fetch boundary (counting only; fetch behaviour itself is untouched and
# stays covered by test_master_avwap_setups)
# ---------------------------------------------------------------------------
def _recent_daily_frame(rows: int = 40) -> pd.DataFrame:
    end = datetime.now().date()
    dates = pd.bdate_range(end=end, periods=rows)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
        }
    )


def test_fetch_daily_bars_counts_cache_hit_without_a_live_call():
    provider_counters.reset()
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "daily_bars"
        legacy._DAILY_BAR_FRAME_CACHE.clear()
        legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
        legacy._DAILY_BAR_LIVE_FAILURE_AT.clear()
        with (
            patch.object(legacy, "DAILY_BARS_CACHE_DIR", cache_dir),
            patch.object(legacy, "_fetch_live_daily_bars") as live_fetch,
        ):
            legacy._write_cached_daily_bar_frame("AL", _recent_daily_frame())
            frame = legacy.fetch_daily_bars(None, "AL", 30)

    assert not frame.empty
    live_fetch.assert_not_called()
    counts = provider_counters.snapshot()
    assert counts.get("daily_bars.cache_hit") == 1
    assert "daily_bars.request" not in counts
    provider_counters.reset()


def test_fetch_daily_bars_counts_request_and_failure_on_live_miss():
    provider_counters.reset()
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "daily_bars"
        legacy._DAILY_BAR_FRAME_CACHE.clear()
        legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
        legacy._DAILY_BAR_LIVE_FAILURE_AT.clear()
        with (
            patch.object(legacy, "DAILY_BARS_CACHE_DIR", cache_dir),
            patch.object(
                legacy,
                "_fetch_live_daily_bars",
                return_value=legacy._empty_daily_bar_frame(),
            ),
        ):
            frame = legacy.fetch_daily_bars(None, "NOPE", 30)

    assert frame.empty
    counts = provider_counters.snapshot()
    assert counts.get("daily_bars.request") == 1
    assert counts.get("daily_bars.failure") == 1
    assert "daily_bars.cache_hit" not in counts
    provider_counters.reset()


def test_throttle_is_only_counted_from_a_real_pacing_signal():
    provider_counters.reset()
    legacy.reset_ibkr_historical_failure_circuit()

    # A pacing-class error is a throttle event.
    legacy._record_ibkr_historical_result(
        "AAPL", succeeded=False, errors=[{"code": 162, "message": "pacing violation"}]
    )
    # A plain timeout is an ordinary failure, never throttling.
    legacy._record_ibkr_historical_result("MSFT", succeeded=False, timed_out=True)
    # A success counts nothing.
    legacy._record_ibkr_historical_result("NVDA", succeeded=True)

    counts = provider_counters.snapshot()
    assert counts.get("ibkr_historical.throttle") == 1
    assert counts.get("ibkr_historical.failure") == 1
    provider_counters.reset()
    legacy.reset_ibkr_historical_failure_circuit()
