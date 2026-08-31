"""P1.4 — benchmark trends over the diagnostics that already exist (packet W7).

Phase 1's exit gate asks for "representative benchmark/golden fixtures and
trends for timings, provider calls, failures, coverage, and scan-stage latency".
Every one of those is already MEASURED and written to disk: the run manifests
carry per-phase seconds and the provider counter tree, and `job_ledger.jsonl`
carries every overnight job's outcome. What was missing was a reader that turns
them into a trend, so "the scan got slower" stops being a feeling.

**Zero new measurement.** Nothing here instruments a hot path, times anything,
or runs during a scan. It opens files that were written hours ago.

Three honesty rules, each tested below, and they are the same three every
evidence surface in this repo keeps:

* **n on every figure**, because a median over two runs is not a trend;
* **a comparison needs both halves** - a recent window with no baseline behind
  it reports the recent number and says the baseline is absent, rather than
  computing a change against nothing;
* **absent is not zero.** A manifest that never recorded a phase is missing that
  phase, not a phase that took no time.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from conftest import load_fixture_contract  # noqa: E402
from diagnostics import observability_trends as trends  # noqa: E402

NOW = datetime(2026, 8, 25, 3, 0, tzinfo=timezone.utc)


def _manifest(run_id, *, seconds=400.0, status="ok", phases=None, counters=None,
              started="2026-08-24T20:05:17+00:00"):
    return {
        "schema": "run_manifest_v1",
        "run_id": run_id,
        "job_type": "master_scan",
        "started_at": started,
        "ended_at": started,
        "status": status,
        "error": "",
        "total_seconds": seconds,
        "phases": phases if phases is not None else [
            {"label": "prep+fetch+priority", "seconds": 220.0},
            {"label": "output writes", "seconds": 91.0},
        ],
        "counters": counters if counters is not None else {
            "symbols_processed": 1122,
            "provider.daily_bars.lookup": 1176,
            "provider.daily_bars.cache_hit": 1171,
            "provider.daily_bars.attempt.yahoo": 5,
            "provider.daily_bars.failure.yahoo": 2,
        },
    }


# ---------------------------------------------------------------------------
# n, and the refusal to compute a trend from nothing
# ---------------------------------------------------------------------------


def test_every_figure_carries_its_n_over_the_recent_window():
    """And the window is the RECENT one, not everything on disk.

    Six manifests with a window of three means three runs measured and three
    held back as the baseline. A report that quietly averaged all six would be
    comparing a window against itself.
    """
    report = trends.build_trends(
        manifests=[_manifest(f"r{index}") for index in range(6)], now=NOW, window=3,
    )
    assert report["window"] == {
        "runs_per_window": 3, "recent_n": 3, "baseline_n": 3,
        "basis": report["window"]["basis"],
    }
    for stage in report["stage_latency"]:
        assert stage["n"] > 0
    assert report["runs"]["n"] == 3
    for family in report["providers"]:
        assert family["runs"] == 3


def test_a_missing_baseline_is_named_rather_than_compared_against_nothing():
    """Two runs are not a before and an after."""
    report = trends.build_trends(manifests=[_manifest("only")], now=NOW, window=3)
    stage = report["stage_latency"][0]
    assert stage["baseline_n"] == 0
    assert stage["change_pct"] is None
    assert "no baseline" in stage["change_basis"].lower()


def test_a_change_is_computed_only_when_both_windows_have_rows():
    recent = [
        _manifest(f"new{index}", phases=[{"label": "prep+fetch+priority", "seconds": 300.0}])
        for index in range(3)
    ]
    older = [
        _manifest(f"old{index}", phases=[{"label": "prep+fetch+priority", "seconds": 200.0}])
        for index in range(3)
    ]
    # Oldest first, the way the loader returns them.
    report = trends.build_trends(manifests=older + recent, now=NOW, window=3)
    stage = [row for row in report["stage_latency"] if row["label"] == "prep+fetch+priority"][0]

    assert stage["n"] == 3 and stage["baseline_n"] == 3
    assert stage["median"] == pytest.approx(300.0)
    assert stage["baseline_median"] == pytest.approx(200.0)
    assert stage["change_pct"] == pytest.approx(50.0)
    assert stage["direction"] == "slower"


def test_a_phase_absent_from_a_manifest_is_missing_not_zero():
    """A phase that never ran did not take no time; it was not measured."""
    with_phase = _manifest("a", phases=[{"label": "studies+enrichment", "seconds": 80.0}])
    without = _manifest("b", phases=[{"label": "output writes", "seconds": 10.0}])
    report = trends.build_trends(manifests=[with_phase, without], now=NOW, window=2)

    studies = [row for row in report["stage_latency"] if row["label"] == "studies+enrichment"][0]
    assert studies["n"] == 1, "the manifest that never recorded it contributes nothing"
    assert studies["median"] == pytest.approx(80.0)
    assert studies["runs_missing_phase"] == 1


def test_an_empty_input_reports_absence_rather_than_a_clean_run():
    report = trends.build_trends(manifests=[], now=NOW)
    assert report["runs"]["n"] == 0
    assert report["stage_latency"] == []
    assert "nothing" in report["summary"].lower() or "no run" in report["summary"].lower()


# ---------------------------------------------------------------------------
# providers, failures, coverage
# ---------------------------------------------------------------------------


def test_provider_counters_are_folded_per_family_with_a_failure_rate():
    report = trends.build_trends(manifests=[_manifest("a"), _manifest("b")], now=NOW)
    daily = [row for row in report["providers"] if row["family"] == "daily_bars"][0]

    assert daily["lookups"] == 2352 and daily["cache_hits"] == 2342
    assert daily["attempts"] == 10 and daily["failures"] == 4
    assert daily["failure_rate"] == pytest.approx(0.4)
    # Rounded to four places, like every other rate in this repo: a report that
    # prints seventeen significant figures invites a reader to believe them.
    assert daily["cache_hit_rate"] == pytest.approx(round(2342 / 2352, 4))
    assert daily["by_source"]["yahoo"]["failures"] == 4


def test_a_family_with_no_attempts_reports_a_blank_rate_never_a_zero():
    """Zero failures out of zero attempts is not a 0% failure rate."""
    manifest = _manifest("a", counters={"provider.earnings_calendar.lookup": 342,
                                        "provider.earnings_calendar.cache_hit": 342})
    report = trends.build_trends(manifests=[manifest], now=NOW)
    family = [row for row in report["providers"] if row["family"] == "earnings_calendar"][0]

    assert family["attempts"] == 0
    assert family["failure_rate"] is None
    assert "no attempt" in family["rate_basis"].lower()


def test_failed_runs_are_counted_and_their_errors_named():
    report = trends.build_trends(
        manifests=[
            _manifest("ok1"),
            {**_manifest("bad"), "status": "failed", "error": "IB refused the connection"},
        ],
        now=NOW,
    )
    assert report["runs"]["failed"] == 1
    assert "IB refused" in " ".join(report["runs"]["recent_errors"])


def test_job_ledger_outcomes_are_folded_per_job():
    report = trends.build_trends(
        manifests=[_manifest("a")],
        job_rows=[
            {"job": "ai_summary", "status": "ok", "session_date": "2026-08-21"},
            {"job": "ai_summary", "status": "failed", "session_date": "2026-08-20"},
            {"job": "ticker_briefs", "status": "ok", "session_date": "2026-08-21"},
        ],
        now=NOW,
    )
    summary = [row for row in report["jobs"] if row["job"] == "ai_summary"][0]
    assert summary["n"] == 2 and summary["ok"] == 1 and summary["failed"] == 1
    assert summary["failure_rate"] == pytest.approx(0.5)


def test_coverage_uses_the_counter_the_scan_already_wrote():
    report = trends.build_trends(
        manifests=[_manifest("a"), _manifest("b", counters={"symbols_processed": 900})],
        now=NOW,
    )
    assert report["coverage"]["symbols_processed"]["n"] == 2
    assert report["coverage"]["symbols_processed"]["median"] == pytest.approx(1011.0)


# ---------------------------------------------------------------------------
# The discipline every evidence surface here keeps
# ---------------------------------------------------------------------------


def test_the_report_is_labelled_discovery_and_names_its_schema():
    report = trends.build_trends(manifests=[_manifest("a")], now=NOW)
    assert report["schema"] == trends.TRENDS_SCHEMA
    assert report["evidence_label"] == "discovery"
    assert report["generated_at"].endswith("+00:00")


def test_nothing_in_this_module_measures_anything_live():
    """Zero new measurement on hot paths: it reads files and nothing else."""
    import ast

    tree = ast.parse(Path(trends.__file__).read_text(encoding="utf-8"))
    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for banned in ("perf_counter", "monotonic", "sleep"):
        assert banned not in called
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for name in imported:
        assert not name.startswith(
            ("bounce_bot", "autopilot_core", "master_avwap", "technical_integrity",
             "price_alert", "d1_level_feed")
        ), f"the trend reader reached into live decision code: {name}"


# ---------------------------------------------------------------------------
# The golden fixture
# ---------------------------------------------------------------------------


def test_the_trend_report_matches_its_golden_fixture():
    """A representative shape, frozen, so a change to the arithmetic is visible.

    Regenerate with `tests/observability_trends_fixture.py --note "why"`; the
    generator refuses to write a changed expectation without one.
    """
    golden = load_fixture_contract(trends.FIXTURE_NAME)
    report = trends.build_trends(
        manifests=golden["manifests"],
        job_rows=golden["job_rows"],
        window=golden.configuration["window"],
        now=datetime.fromisoformat(golden.as_of),
    )
    for section in ("stage_latency", "providers", "jobs", "runs", "coverage"):
        golden.assert_matches(report[section], golden[section], context=section)


def test_the_fixture_declares_the_inputs_it_was_built_from():
    golden = load_fixture_contract(trends.FIXTURE_NAME)
    assert golden.raw_input_keys == ("manifests", "job_rows")
    assert golden.raw_input_digest() == golden["raw_input_sha256"]
    assert golden.tolerance == 0.0
