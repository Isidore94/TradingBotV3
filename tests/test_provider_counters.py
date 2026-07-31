"""Provider telemetry v2 (plan.md sec 6.3 bullet 9): honest boundary accounting.

v1 stamped provider.captured unconditionally and let the audit grade an empty
block HEALTHY; one wrapper "request" could hide an IBKR attempt plus a Yahoo
fallback attempt; failure ratios divided mixed endpoints by unrelated totals.
These tests pin the v2 contract:

* distinct concepts (lookup / cache_hit / attempt.<provider> /
  success.<provider> / failure.<provider> / throttle.<provider> /
  fallback_used / refresh_unusable);
* completeness declared, not assumed (expected vs instrumented families);
* capture failures observable, never silently swallowed under a schema stamp;
* per-run isolation with a visible orphan bucket;
* runner flush on both successful and failed scans.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnostics import provider_counters  # noqa: E402
import master_avwap_lib.legacy as legacy  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_counters():
    provider_counters.reset()
    yield
    provider_counters.reset()


class _FakeRecorder:
    def __init__(self):
        self.counters: dict[str, object] = {}
        self.outputs: dict[str, str] = {}

    def set_counter(self, key, value):
        self.counters[key] = value


# ---------------------------------------------------------------------------
# the store: concepts, isolation, capture errors
# ---------------------------------------------------------------------------
def test_distinct_concepts_and_provider_qualified_keys():
    provider_counters.begin_run()
    provider_counters.record("daily_bars", "lookup")
    provider_counters.record("daily_bars", "cache_hit")
    provider_counters.record("daily_bars", "attempt", "ibkr")
    provider_counters.record("daily_bars", "failure", "ibkr")
    provider_counters.record("daily_bars", "attempt", "yahoo")
    provider_counters.record("daily_bars", "success", "yahoo")
    provider_counters.record("daily_bars", "fallback_used")

    assert provider_counters.snapshot() == {
        "daily_bars.attempt.ibkr": 1,
        "daily_bars.attempt.yahoo": 1,
        "daily_bars.cache_hit": 1,
        "daily_bars.failure.ibkr": 1,
        "daily_bars.fallback_used": 1,
        "daily_bars.lookup": 1,
        "daily_bars.success.yahoo": 1,
    }


def test_measured_true_zero_flushes_schema_and_declarations():
    provider_counters.begin_run()
    recorder = _FakeRecorder()
    provider_counters.flush_to_manifest(recorder)

    assert recorder.counters["provider.schema_version"] == provider_counters.SCHEMA_VERSION
    assert recorder.counters["provider.capture_errors"] == 0
    assert recorder.counters["provider.orphan_events"] == 0
    assert recorder.outputs["provider_families_expected"] == ",".join(
        provider_counters.FAMILIES_EXPECTED
    )
    assert recorder.outputs["provider_families_instrumented"] == ",".join(
        provider_counters.FAMILIES_INSTRUMENTED
    )


def test_capture_helper_failure_is_observable_not_silent():
    provider_counters.begin_run()
    # Malformed events are dropped AND counted - never silently lost.
    provider_counters.record("", "lookup")
    provider_counters.record("daily_bars", "")
    provider_counters.record("daily_bars", "lookup", n="junk")  # type: ignore[arg-type]

    recorder = _FakeRecorder()
    provider_counters.flush_to_manifest(recorder)
    assert recorder.counters["provider.capture_errors"] == 3


def test_late_worker_lands_in_the_orphan_bucket_never_the_next_run():
    provider_counters.begin_run()
    provider_counters.record("daily_bars", "lookup")
    first = _FakeRecorder()
    provider_counters.flush_to_manifest(first)
    assert first.counters["provider.daily_bars.lookup"] == 1

    # A worker that outlived run 1 records between runs.
    provider_counters.record("daily_bars", "attempt", "ibkr")
    assert provider_counters.orphan_snapshot() == {"daily_bars.attempt.ibkr": 1}

    provider_counters.begin_run()
    provider_counters.record("intraday_bars", "lookup")
    second = _FakeRecorder()
    provider_counters.flush_to_manifest(second)

    # The next run's manifest carries ONLY its own events, plus a visible
    # orphan count - never the late worker's counter merged in.
    assert "provider.daily_bars.attempt.ibkr" not in second.counters
    assert second.counters["provider.intraday_bars.lookup"] == 1
    assert second.counters["provider.orphan_events"] == 1


# ---------------------------------------------------------------------------
# the fetch boundary (counting only; behaviour covered elsewhere)
# ---------------------------------------------------------------------------
def _recent_daily_frame(rows: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range(end=datetime.now().date(), periods=rows)
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


def test_cache_hit_counts_lookup_but_no_outbound_attempt():
    provider_counters.begin_run()
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
    assert counts.get("daily_bars.lookup") == 1
    assert counts.get("daily_bars.cache_hit") == 1
    assert not any(".attempt." in key for key in counts)


def test_ibkr_failure_then_yahoo_fallback_success_counts_both_attempts():
    """A logical lookup that costs one IBKR attempt AND one Yahoo attempt must
    show exactly that - v1's single wrapper 'request' hid the second call."""
    provider_counters.begin_run()

    class _FakeIB:
        data: dict = {}
        ready: dict = {}
        request_errors: dict = {}

        def reqHistoricalData(self, *args, **kwargs):
            return None

        def cancelHistoricalData(self, *args, **kwargs):
            return None

    yahoo_frame = legacy._set_daily_bar_source(
        legacy._normalize_daily_bar_frame(_recent_daily_frame()),
        legacy.DAILY_BAR_SOURCE_YAHOO,
    )
    legacy.reset_ibkr_historical_failure_circuit()
    with (
        patch.object(legacy, "DAILY_BAR_IBKR_TIMEOUT_SEC", 0.01),
        patch.object(legacy, "DAILY_BAR_IBKR_POLL_INTERVAL_SEC", 0.005),
        patch.object(legacy, "fetch_daily_bars_from_yahoo", return_value=yahoo_frame) as yahoo,
    ):
        frame = legacy._fetch_live_daily_bars(_FakeIB(), "AL", 30)

    yahoo.assert_called_once()
    assert not frame.empty
    counts = provider_counters.snapshot()
    assert counts.get("daily_bars.attempt.ibkr") == 1
    assert counts.get("daily_bars.failure.ibkr") == 1
    assert counts.get("daily_bars.fallback_used") == 1
    # The Yahoo attempt/success themselves are counted inside the (patched)
    # yahoo fetcher, so they are asserted by the dedicated Yahoo tests below.
    legacy.reset_ibkr_historical_failure_circuit()


def test_yahoo_failure_counts_attempt_and_failure():
    provider_counters.begin_run()
    with patch.object(legacy.yf, "download", side_effect=RuntimeError("boom")):
        frame = legacy.fetch_daily_bars_from_yahoo("AL", 30)

    assert frame.empty
    counts = provider_counters.snapshot()
    assert counts.get("daily_bars.attempt.yahoo") == 1
    assert counts.get("daily_bars.failure.yahoo") == 1
    assert "daily_bars.success.yahoo" not in counts


def test_throttle_only_from_a_real_pacing_signal_and_without_ordinary_failure():
    provider_counters.begin_run()
    legacy.reset_ibkr_historical_failure_circuit()

    legacy._record_ibkr_historical_result(
        "AAPL", succeeded=False, errors=[{"code": 162, "message": "pacing violation"}]
    )
    legacy._record_ibkr_historical_result("MSFT", succeeded=False, timed_out=True)
    legacy._record_ibkr_historical_result("NVDA", succeeded=True)

    counts = provider_counters.snapshot()
    # Throttle comes ONLY from the pacing-class signal; the circuit recorder
    # itself never double-counts failures (the fetch boundary owns those).
    assert counts == {"daily_bars.throttle.ibkr": 1}
    legacy.reset_ibkr_historical_failure_circuit()


# ---------------------------------------------------------------------------
# runner integration: flush on success AND on failure
# ---------------------------------------------------------------------------
def _run_master_with_stub(monkeypatch, impl):
    import master_avwap_lib.runner as runner

    monkeypatch.setattr(runner, "_run_master_impl", impl)
    from diagnostics.run_manifest import load_recent_manifests
    from project_paths import get_diagnostics_dir

    manifest_dir = get_diagnostics_dir() / "run_manifests"
    before = {m.get("run_id") for m in load_recent_manifests(manifest_dir, limit=50)}
    try:
        runner.run_master()
        outcome = "ok"
    except RuntimeError:
        outcome = "failed"
    manifests = load_recent_manifests(manifest_dir, limit=50)
    new = [m for m in manifests if m.get("run_id") not in before]
    assert new, "run_master must write a manifest either way"
    return outcome, new[0]


def test_runner_flushes_provider_telemetry_on_a_successful_scan(monkeypatch):
    def impl(**kwargs):
        provider_counters.record("daily_bars", "lookup")
        provider_counters.record("daily_bars", "cache_hit")
        return {}

    outcome, manifest = _run_master_with_stub(monkeypatch, impl)
    assert outcome == "ok"
    counters = manifest["counters"]
    assert counters["provider.schema_version"] == provider_counters.SCHEMA_VERSION
    assert counters["provider.daily_bars.lookup"] == 1
    assert counters["provider.daily_bars.cache_hit"] == 1
    assert manifest["outputs"]["provider_families_instrumented"]


def test_runner_flushes_provider_telemetry_on_a_failed_scan(monkeypatch):
    def impl(**kwargs):
        provider_counters.record("daily_bars", "attempt", "ibkr")
        provider_counters.record("daily_bars", "failure", "ibkr")
        raise RuntimeError("scan exploded")

    outcome, manifest = _run_master_with_stub(monkeypatch, impl)
    assert outcome == "failed"
    assert manifest["status"] == "failed"
    counters = manifest["counters"]
    # A failed scan's provider counts are exactly what diagnoses it.
    assert counters["provider.schema_version"] == provider_counters.SCHEMA_VERSION
    assert counters["provider.daily_bars.attempt.ibkr"] == 1
    assert counters["provider.daily_bars.failure.ibkr"] == 1
