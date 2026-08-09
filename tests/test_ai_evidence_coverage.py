"""Evidence packaging: what the model sees, and what it is told is absent.

The overnight brief reads whatever the desk left on disk, unattended, and the
old packaging could not tell four different situations apart. Every one of
these was observed or reachable (checkpoint review 2026-08-08 second review):

* a header-only CSV and a 40 KB tracker were both "available";
* the 80,000-char budget was first-come, so a large early source could zero
  every later one -- including the two the trader actually asked the review to
  read -- and the zeroed sources arrived with ``content: null`` and status
  "available", indistinguishable from genuinely empty ones and, when the
  budget hit exactly zero, without even the ``[package budget reached]`` marker;
* an ``auto_report`` from 2026-07-30 was packaged for a later session with
  nothing anywhere saying it was from a different day;
* nothing stopped a model from citing a source that carried no content.

The rule these tests defend: a source with real bytes on disk is never
presented as empty, and nothing the model cannot see is left unexplained.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SESSION = "2026-08-07"


def _touch(path: Path, text: str, *, session: str = SESSION) -> Path:
    """Write a file and stamp its mtime onto ``session`` at midday."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    when = datetime.fromisoformat(f"{session}T12:00:00").timestamp()
    os.utime(path, (when, when))
    return path


def _daily(tmp_path, **overrides) -> dict[str, Path]:
    """The three daily_report sources, each written unless overridden."""
    paths = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        if source_id in overrides:
            paths[source_id] = overrides[source_id]
            continue
        paths[source_id] = _touch(
            tmp_path / (source_id.replace(".", "_") + ".txt"),
            f"Real content from {source_id}.\n" * 5,
        )
    return paths


#: Every setup_trackers source, so a fixture can oversubscribe the budget on
#: purpose. Each source is independently capped at MAX_SOURCE_CHARS, so
#: starving one takes more sources than daily_report's three can provide.
TRACKER_SOURCE_IDS = (
    "setups.current_tracker",
    "setups.current_tiers",
    "setups.type_stats",
    "setups.recent_type_stats",
    "setups.short_horizon",
    "setups.playbooks",
    "setups.scan_factors",
    "setups.tier_performance",
    "setups.bounce_learning",
)


def _oversized_trackers(tmp_path, *, chars: int = 20_000) -> dict[str, Path]:
    return {
        source_id: _touch(
            tmp_path / (source_id.replace(".", "_") + ".txt"), "q" * chars
        )
        for source_id in TRACKER_SOURCE_IDS
    }


def _by_id(evidence) -> dict:
    return {row["source_id"]: row for row in evidence["sources"]}


def _coverage_by_id(evidence) -> dict:
    return {row["source_id"]: row for row in evidence["coverage"]["excluded"]}


# ---------------------------------------------------------------------------
# semantic statuses and content-aware emptiness
# ---------------------------------------------------------------------------
def test_a_header_only_csv_is_empty_not_available(tmp_path):
    from ai_summary import build_evidence_package

    csv_path = _touch(tmp_path / "types.csv", "setup_type,wins,losses\n")
    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.type_stats": csv_path},
        session_date=SESSION,
    )

    assert "setups.type_stats" not in _by_id(evidence)
    entry = _coverage_by_id(evidence)["setups.type_stats"]
    assert entry["status"] == "empty"
    assert "no data rows" in entry["reason"]


def test_whitespace_only_text_is_empty(tmp_path):
    from ai_summary import build_evidence_package

    blank = _touch(tmp_path / "report.txt", "   \n\n\t\n")
    evidence = build_evidence_package(
        ["daily_report"],
        source_overrides=_daily(tmp_path, **{"daily.auto_report": blank}),
        session_date=SESSION,
    )

    assert _coverage_by_id(evidence)["daily.auto_report"]["status"] == "empty"


def test_jsonl_with_no_valid_records_is_empty(tmp_path):
    from ai_summary import build_evidence_package

    junk = _touch(tmp_path / "learning.jsonl", "not json\n{oops\n\n")
    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.bounce_learning": junk},
        session_date=SESSION,
    )

    entry = _coverage_by_id(evidence)["setups.bounce_learning"]
    assert entry["status"] == "empty"
    assert "no valid records" in entry["reason"]


def test_json_with_no_records_in_its_containers_is_empty(tmp_path):
    from ai_summary import build_evidence_package

    fresh = _touch(
        tmp_path / "tracker.json",
        json.dumps({"schema_version": 2, "setups": {}, "stats": [], "daily_watchlists": {}}),
    )
    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tracker": fresh},
        session_date=SESSION,
    )

    assert _coverage_by_id(evidence)["setups.current_tracker"]["status"] == "empty"


def test_json_with_records_is_available(tmp_path):
    from ai_summary import build_evidence_package

    real = _touch(
        tmp_path / "tracker.json",
        json.dumps({"schema_version": 2, "setups": {"AAA-1": {"symbol": "AAA"}}}),
    )
    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tracker": real},
        session_date=SESSION,
    )

    source = _by_id(evidence)["setups.current_tracker"]
    assert source["status"] == "available"
    assert source["content"]["setups"]["AAA-1"]["symbol"] == "AAA"


def test_unparseable_json_is_invalid_not_empty(tmp_path):
    from ai_summary import build_evidence_package

    broken = _touch(tmp_path / "tracker.json", '{"setups": {"AAA": ')
    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tracker": broken},
        session_date=SESSION,
    )

    entry = _coverage_by_id(evidence)["setups.current_tracker"]
    assert entry["status"] == "invalid"
    assert "malformed JSON" in entry["reason"]


def test_a_file_that_does_not_exist_is_missing(tmp_path):
    from ai_summary import build_evidence_package

    evidence = build_evidence_package(
        ["daily_report"],
        source_overrides=_daily(tmp_path, **{"daily.master_events": tmp_path / "nope.txt"}),
        session_date=SESSION,
    )

    assert _coverage_by_id(evidence)["daily.master_events"]["status"] == "missing"


def test_a_journal_store_that_cannot_be_queried_is_unavailable():
    from ai_summary import build_evidence_package

    class _Broken:
        def list_trades(self):
            raise RuntimeError("journal database is locked")

        def list_opportunity_events(self, limit=0):  # pragma: no cover
            return []

    evidence = build_evidence_package(
        ["journal_review"], journal_store=_Broken(), session_date=SESSION
    )

    entry = _coverage_by_id(evidence)["journal.trades_and_reviews"]
    assert entry["status"] == "unavailable"
    assert "locked" in entry["reason"]


# ---------------------------------------------------------------------------
# the budget: fair, priority-aware, and never silently zeroing
# ---------------------------------------------------------------------------
def test_priority_scopes_are_funded_before_the_daily_narrative(tmp_path):
    """The defect: a large daily_report consumed the whole first-come budget.

    setup_trackers and journal_review are the two scopes the trader asked the
    nightly review to read; they cannot be starved by whichever artifact the
    packager happened to encode first.
    """
    from ai_summary import MAX_TOTAL_EVIDENCE_CHARS, build_evidence_package

    huge = "x" * (MAX_TOTAL_EVIDENCE_CHARS * 2)
    tracker = _touch(
        tmp_path / "tracker.json",
        json.dumps({"setups": {f"S{i}": {"symbol": f"SYM{i}"} for i in range(400)}}),
    )
    evidence = build_evidence_package(
        ["daily_report", "setup_trackers"],
        source_overrides={
            **_daily(tmp_path, **{"daily.auto_report": _touch(tmp_path / "big.txt", huge)}),
            "setups.current_tracker": tracker,
        },
        session_date=SESSION,
    )

    by_id = _by_id(evidence)
    assert "setups.current_tracker" in by_id, "the priority scope must still be funded"
    assert by_id["setups.current_tracker"]["content"], "and must carry real content"


def test_a_starved_source_is_declared_unfunded_never_empty(tmp_path):
    """The critical distinction: real bytes on disk are never called empty.

    The old budget set ``content`` to "" and left the status at "available", so
    a starved source and a genuinely empty one were the same object -- and when
    ``remaining`` hit exactly zero it did not even append the
    ``[package budget reached]`` marker.
    """
    from ai_summary import build_evidence_package

    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides=_oversized_trackers(tmp_path),
        session_date=SESSION,
    )

    excluded = _coverage_by_id(evidence)
    starved = [row for row in excluded.values() if row["status"] == "unfunded"]
    assert starved, "nine oversized sources must not all fit in the budget"
    for row in starved:
        assert row["status"] != "empty"
        assert "real content" in row["reason"]
        assert "budget" in row["reason"]
    # And nothing was left in the model package holding nothing -- the exact
    # shape the old first-come budget produced when remaining hit zero.
    for source in evidence["sources"]:
        assert source["content"] not in (None, "", [], {}), source["source_id"]


def test_a_partly_funded_tabular_source_keeps_its_most_recent_rows():
    """Tabular truncation drops the oldest rows and says so, in band.

    Most-recent is what a trading review reads from, and the banner is what
    stops a shortened artifact from looking like a complete one.
    """
    from ai_summary import _apply_evidence_budget, _source_record

    rows = [{"i": i, "pad": "p" * 100} for i in range(200)]
    source = _source_record(
        "setups.bounce_learning", "BounceBot learning state",
        status="available", content=rows,
    )
    _apply_evidence_budget({"setup_trackers": [source]}, total=6_000)

    assert source["status"] == "available"
    assert source["budget_truncated"] is True
    banner = source["content"][0]
    assert banner.startswith("[showing most recent ")
    assert "of 200 rows]" in banner
    kept = [row for row in source["content"][1:] if isinstance(row, dict)]
    assert kept, "truncation must keep rows, not blank the source"
    assert kept[-1]["i"] == 199, "the newest row survives"
    assert kept[0]["i"] > 0, "the oldest rows are the ones dropped"
    assert banner in source["notices"]


def test_a_partly_funded_text_source_keeps_its_head_and_says_so():
    from ai_summary import _apply_evidence_budget, _source_record

    source = _source_record(
        "daily.auto_report", "Auto/Away daily report",
        status="available", content="A" * 5_000,
    )
    _apply_evidence_budget({"daily_report": [source]}, total=2_000)

    assert source["status"] == "available"
    assert source["truncated"] is True
    assert "[showing the first 2000 of 5000 characters of this source]" in source["content"]


def test_a_source_that_fits_is_left_exactly_as_it_was():
    from ai_summary import _apply_evidence_budget, _source_record

    source = _source_record("daily.auto_report", "d", status="available", content="short")
    _apply_evidence_budget({"daily_report": [source]}, total=80_000)

    assert source["content"] == "short"
    assert source["truncated"] is False
    assert source["notices"] == []


def test_the_scope_allocator_gives_priority_scopes_the_larger_share():
    from ai_summary import _allocate_scope_budgets

    # Every scope wants far more than the whole budget, so nothing is returned
    # as surplus and the weights alone decide.
    allocation = _allocate_scope_budgets(
        {"setup_trackers": 10**7, "journal_review": 10**7,
         "daily_report": 10**7, "market_conditions": 10**7},
        total=1000,
    )

    assert sum(allocation.values()) <= 1000
    assert allocation["setup_trackers"] == allocation["journal_review"] == 300
    assert allocation["daily_report"] == allocation["market_conditions"] == 200
    # No scope is zeroed by another scope's appetite -- the first-come defect.
    assert all(value > 0 for value in allocation.values())


def test_a_scope_that_needs_less_hands_its_surplus_to_one_that_needs_more():
    from ai_summary import _allocate_scope_budgets

    allocation = _allocate_scope_budgets(
        {"setup_trackers": 900, "daily_report": 100}, total=1000
    )

    assert allocation["daily_report"] == 100, "capped at what it actually needs"
    assert allocation["setup_trackers"] == 900, "the surplus went where it was short"


def test_the_coverage_counts_add_up(tmp_path):
    from ai_summary import build_evidence_package

    overrides = _daily(tmp_path)
    _touch(overrides["daily.market_prep"], "  \n")  # empty
    overrides["daily.master_events"] = tmp_path / "gone.txt"  # missing

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )
    counts = evidence["coverage"]["counts"]

    assert counts["requested"] == 3
    assert counts["usable"] == 1
    assert counts["empty"] == 1
    assert counts["missing"] == 1
    assert len(evidence["sources"]) == counts["usable"]
    assert len(evidence["coverage"]["excluded"]) == 2


# ---------------------------------------------------------------------------
# session scoping and staleness
# ---------------------------------------------------------------------------
def test_a_source_from_another_session_is_flagged_stale_in_band_and_in_coverage(tmp_path):
    """The 2026-07-30 auto_report incident, made visible."""
    from ai_summary import build_evidence_package

    overrides = _daily(tmp_path)
    _touch(overrides["daily.auto_report"], "Report body.\n", session="2026-07-30")

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )

    source = _by_id(evidence)["daily.auto_report"]
    assert source["stale"] is True
    assert source["source_session"] == "2026-07-30"
    assert source["requested_session"] == SESSION
    assert any("STALE" in notice and "2026-07-30" in notice for notice in source["notices"])

    stale = {row["source_id"] for row in evidence["coverage"]["stale"]}
    assert stale == {"daily.auto_report"}
    assert evidence["coverage"]["counts"]["stale"] == 1


def test_same_session_sources_are_not_flagged(tmp_path):
    from ai_summary import build_evidence_package

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=_daily(tmp_path), session_date=SESSION
    )

    assert evidence["coverage"]["stale"] == []
    assert all(source["stale"] is False for source in evidence["sources"])


def test_the_brief_job_passes_its_session_into_packaging(monkeypatch, tmp_path):
    from ai_jobs import briefs

    seen = {}

    class _FakeSummary:
        @staticmethod
        def local_provider_enabled():
            return True

        @staticmethod
        def local_model(tier="medium"):
            return "gemma3:12b"

        @staticmethod
        def build_evidence_package(scopes, *, session_date=None, **kwargs):
            seen["scopes"] = list(scopes)
            seen["session_date"] = session_date
            return {"package_id": "abc", "sources": [], "coverage": {"counts": {"requested": 0, "usable": 0}}}

        @staticmethod
        def has_usable_sources(evidence):
            return False

        @staticmethod
        def degraded_result(evidence, *, reason, model=""):
            seen["degraded_reason"] = reason
            return {"summary": {}, "model": model}

        @staticmethod
        def export_ai_summary(result, evidence, *, output_dir):
            return {"markdown": output_dir / "x.md"}

    monkeypatch.setitem(sys.modules, "ai_summary", _FakeSummary)
    monkeypatch.setattr(briefs, "_summary_dir", lambda session: tmp_path)

    outcome = briefs.run_daily_summary(session_date=SESSION)

    assert seen["session_date"] == SESSION
    assert "setup_trackers" in seen["scopes"] and "journal_review" in seen["scopes"]
    assert outcome["status"] == "degraded_no_narrative"


# ---------------------------------------------------------------------------
# the model package, the validator, and the deterministic coverage merge
# ---------------------------------------------------------------------------
def _summary(ref: str) -> dict:
    from ai_summary import AI_SUMMARY_SECTIONS

    payload = {name: [] for name in AI_SUMMARY_SECTIONS}
    payload["executive_summary"] = "One measured finding."
    payload["what_is_working"] = [
        {"statement": "Swings led.", "evidence_refs": [ref], "confidence": "high"}
    ]
    return payload


def test_the_model_package_carries_only_usable_sources(tmp_path):
    from ai_summary import _model_visible_package, build_evidence_package

    overrides = _daily(tmp_path)
    _touch(overrides["daily.market_prep"], "\n")
    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )

    assert "daily.market_prep" not in _by_id(evidence)
    # And the coverage block is the code's, not the model's, so it is not sent.
    assert "coverage" not in _model_visible_package(evidence)


def test_the_prompt_explains_the_sources_that_are_absent(tmp_path):
    from ai_summary import COVERAGE_PROMPT_LINE, _user_prompt, build_evidence_package

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=_daily(tmp_path), session_date=SESSION
    )
    prompt = _user_prompt(evidence)

    assert COVERAGE_PROMPT_LINE in prompt
    assert "do not speculate" in COVERAGE_PROMPT_LINE


def test_citing_an_empty_source_is_rejected_and_the_source_is_named(tmp_path):
    """The regression case: real + empty + budget-starved in one package."""
    from ai_summary import build_evidence_package, validate_ai_summary

    overrides = _daily(tmp_path)
    _touch(overrides["daily.market_prep"], "   \n")                 # empty on disk
    _touch(overrides["daily.master_events"], "m" * 200_000)         # starved by budget
    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )

    statuses = {row["source_id"]: row["status"] for row in evidence["coverage"]["excluded"]}
    assert statuses["daily.market_prep"] == "empty"
    # The real source is present; the two non-usable ones are not.
    assert set(_by_id(evidence)) <= {"daily.auto_report", "daily.master_events"}

    # A citation of the empty source is rejected, and the rejection says why.
    with pytest.raises(ValueError) as excinfo:
        validate_ai_summary(_summary("daily.market_prep"), evidence)
    message = str(excinfo.value)
    assert "daily.market_prep" in message
    assert "empty" in message
    assert "not in this package" in message

    # A citation of a real source still validates.
    assert validate_ai_summary(_summary("daily.auto_report"), evidence)


def test_a_citation_of_an_unfunded_source_is_rejected_too(tmp_path):
    from ai_summary import build_evidence_package, validate_ai_summary

    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides=_oversized_trackers(tmp_path),
        session_date=SESSION,
    )

    unfunded = [
        row["source_id"]
        for row in evidence["coverage"]["excluded"]
        if row["status"] == "unfunded"
    ]
    assert unfunded, "the fixture must actually starve something"
    with pytest.raises(ValueError, match="unfunded"):
        validate_ai_summary(_summary(unfunded[0]), evidence)


def test_every_section_but_the_executive_summary_may_be_empty(tmp_path):
    from ai_summary import AI_SUMMARY_SECTIONS, build_evidence_package, validate_ai_summary

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=_daily(tmp_path), session_date=SESSION
    )
    thin = {name: [] for name in AI_SUMMARY_SECTIONS}
    thin["executive_summary"] = "A thin night with nothing to report is a valid answer."

    validated = validate_ai_summary(thin, evidence)
    assert all(validated[name] == [] for name in AI_SUMMARY_SECTIONS)

    thin["executive_summary"] = "   "
    with pytest.raises(ValueError, match="executive_summary"):
        validate_ai_summary(thin, evidence)


def test_coverage_rows_are_merged_by_code_with_exact_counts(tmp_path):
    from ai_summary import (
        COVERAGE_STATEMENT_PREFIX,
        build_evidence_package,
        merge_coverage_into_summary,
        validate_ai_summary,
    )

    overrides = _daily(tmp_path)
    _touch(overrides["daily.market_prep"], "\n")
    _touch(overrides["daily.auto_report"], "Report.\n", session="2026-07-30")
    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )
    validated = validate_ai_summary(_summary("daily.auto_report"), evidence)
    validated["data_quality"] = [
        {"statement": "A model observation.", "evidence_refs": [], "confidence": "low"}
    ]

    merged = merge_coverage_into_summary(validated, evidence)
    system_rows = [
        row["statement"] for row in merged["data_quality"]
        if row["statement"].startswith(COVERAGE_STATEMENT_PREFIX)
    ]

    assert merged["data_quality"][0]["statement"] == "A model observation."
    assert any("2 of 3 requested source(s) were usable" in row for row in system_rows)
    assert any("1 source(s) empty: daily.market_prep" in row for row in system_rows)
    assert any("different session" in row and "2026-07-30" in row for row in system_rows)
    # Merging twice must not duplicate the provenance rows.
    assert merge_coverage_into_summary(merged, evidence)["data_quality"] == merged["data_quality"]


# ---------------------------------------------------------------------------
# failure policy
# ---------------------------------------------------------------------------
def test_zero_usable_sources_never_calls_the_model(tmp_path, monkeypatch):
    from ai_jobs import briefs
    import ai_summary

    overrides = {name: tmp_path / f"{name}.txt" for name in
                 ("daily.auto_report", "daily.market_prep", "daily.master_events")}
    calls = []

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "gemma3:12b")
    monkeypatch.setattr(
        ai_summary,
        "request_ai_summary",
        lambda **kwargs: calls.append(kwargs),  # pragma: no cover - must not run
    )
    real_build = ai_summary.build_evidence_package
    monkeypatch.setattr(
        ai_summary,
        "build_evidence_package",
        lambda scopes, **kwargs: real_build(
            ["daily_report"], source_overrides=overrides, **kwargs
        ),
    )
    monkeypatch.setattr(briefs, "_summary_dir", lambda session: tmp_path / "out")

    outcome = briefs.run_daily_summary(session_date=SESSION)

    assert calls == [], "a model must not be asked to narrate nothing"
    assert outcome["status"] == "degraded_no_narrative"
    assert "no usable evidence" in outcome["reason"]
    published = (tmp_path / "out").glob("*.md")
    body = next(published).read_text(encoding="utf-8")
    assert "DEGRADED" in body


def test_a_second_validation_failure_publishes_a_degraded_document(tmp_path, monkeypatch):
    from ai_jobs import briefs
    import ai_summary

    attempts = []

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "gemma3:12b")
    real_build = ai_summary.build_evidence_package
    monkeypatch.setattr(
        ai_summary,
        "build_evidence_package",
        lambda scopes, **kwargs: real_build(
            ["daily_report"], source_overrides=_daily(tmp_path), **kwargs
        ),
    )

    def _always_rejects(**kwargs):
        attempts.append(kwargs.get("previous_error", ""))
        raise ValueError("what_is_working[0] cites unusable evidence: daily.ghost")

    monkeypatch.setattr(ai_summary, "request_ai_summary", _always_rejects)
    monkeypatch.setattr(briefs, "_summary_dir", lambda session: tmp_path / "out")

    outcome = briefs.run_daily_summary(session_date=SESSION)

    assert len(attempts) == 2, "exactly one retry"
    assert attempts[0] == "", "the first attempt carries no feedback"
    assert "daily.ghost" in attempts[1], "the retry is told exactly what was wrong"
    assert outcome["status"] == "degraded_no_narrative"
    assert "twice" in outcome["reason"]
    body = next((tmp_path / "out").glob("*.md")).read_text(encoding="utf-8")
    assert "DEGRADED" in body
    assert "no narrative" in body.lower()


def test_a_first_failure_that_the_retry_fixes_is_an_ok_run(tmp_path, monkeypatch):
    from ai_jobs import briefs
    import ai_summary

    attempts = []

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "gemma3:12b")
    real_build = ai_summary.build_evidence_package
    monkeypatch.setattr(
        ai_summary,
        "build_evidence_package",
        lambda scopes, **kwargs: real_build(
            ["daily_report"], source_overrides=_daily(tmp_path), **kwargs
        ),
    )

    def _flaky(**kwargs):
        attempts.append(kwargs.get("previous_error", ""))
        if len(attempts) == 1:
            raise ValueError("best_candidates[0] cites unusable evidence: daily.ghost")
        return {
            "provider": "local",
            "model": "gemma3:12b",
            "duration_seconds": 1.0,
            "summary": _summary("daily.auto_report"),
        }

    monkeypatch.setattr(ai_summary, "request_ai_summary", _flaky)
    monkeypatch.setattr(briefs, "_summary_dir", lambda session: tmp_path / "out")

    outcome = briefs.run_daily_summary(session_date=SESSION)

    assert outcome["status"] == "ok"
    assert "daily.ghost" in attempts[1]
    body = next((tmp_path / "out").glob("*.md")).read_text(encoding="utf-8")
    assert "DEGRADED" not in body
    # The system's provenance rows are in the published document.
    assert "[system] Evidence coverage:" in body


def test_the_degraded_status_is_retried_by_the_next_firing(tmp_path):
    from ai_jobs import ledger

    led = tmp_path / "ledger.jsonl"
    ledger.record(
        job="ai_summary",
        status=ledger.STATUS_DEGRADED,
        session_date=SESSION,
        reason="no usable evidence",
        path=led,
    )

    # Distinct from ok, and deliberately not "completed" -- the 30-minute
    # re-firing is exactly the retry.
    assert ledger.STATUS_DEGRADED == "degraded_no_narrative"
    assert ledger.completed_jobs(SESSION, path=led) == set()


def test_the_runner_records_a_degraded_job_as_degraded(tmp_path):
    from unittest import mock

    from ai_jobs import ledger, runner, store, window

    led = tmp_path / "ledger.jsonl"

    def job(*, session_date, now):
        return {"status": ledger.STATUS_DEGRADED, "reason": "no usable evidence"}

    with (
        mock.patch.object(store, "store_available", return_value=(True, "ready")),
        mock.patch.object(window, "launch_allowed", return_value=(True, "open")),
        mock.patch.object(window, "market_session_block", return_value=""),
    ):
        report = runner.run_slots(
            [runner.JobSlot(name="ai_summary", run=job)],
            now=datetime.fromisoformat("2026-08-08T02:00:00+00:00"),
            ledger_path=led,
        )

    assert report.degraded == 1
    assert report.ran == 0 and report.failed == 0
    assert "1 degraded" in report.summary()
    row = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "degraded_no_narrative"


def test_an_unknown_status_from_a_job_is_not_trusted_into_the_ledger(tmp_path):
    from unittest import mock

    from ai_jobs import runner, store, window

    led = tmp_path / "ledger.jsonl"

    with (
        mock.patch.object(store, "store_available", return_value=(True, "ready")),
        mock.patch.object(window, "launch_allowed", return_value=(True, "open")),
        mock.patch.object(window, "market_session_block", return_value=""),
    ):
        runner.run_slots(
            [runner.JobSlot(name="ai_summary", run=lambda **k: {"status": "wonderful"})],
            now=datetime.fromisoformat("2026-08-08T02:00:00+00:00"),
            ledger_path=led,
        )

    row = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "ok"


def test_the_degraded_document_states_coverage_without_a_model(tmp_path):
    from ai_summary import build_evidence_package, degraded_result, validate_ai_summary

    overrides = _daily(tmp_path)
    for name in overrides:
        _touch(overrides[name], "\n")
    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )

    result = degraded_result(evidence, reason="No usable evidence.")

    assert result["status"] == "degraded_no_narrative"
    assert result["model"] == ""
    # It is a valid document by the same validator every other summary passes.
    validate_ai_summary(result["summary"], evidence)
    assert "DEGRADED" in result["summary"]["executive_summary"]
    assert "0 of 3" in result["summary"]["executive_summary"]
    assert any(
        "3 source(s) empty" in row["statement"] for row in result["summary"]["data_quality"]
    )
    assert result["summary"]["risk_notes"], "an empty risk section would read as 'no risks'"


def test_a_source_read_error_does_not_take_the_package_down(tmp_path):
    from ai_summary import build_evidence_package

    directory = tmp_path / "a_directory.txt"
    directory.mkdir()
    when = datetime.fromisoformat(f"{SESSION}T12:00:00").timestamp()
    os.utime(directory, (when, when))

    evidence = build_evidence_package(
        ["daily_report"],
        source_overrides=_daily(tmp_path, **{"daily.auto_report": directory}),
        session_date=SESSION,
    )

    entry = _coverage_by_id(evidence)["daily.auto_report"]
    assert entry["status"] in {"unavailable", "invalid"}
    assert evidence["coverage"]["counts"]["usable"] == 2


def test_a_source_capped_at_read_time_reports_its_banner_in_coverage_too(tmp_path):
    """The banner has to reach notices, not only the content.

    A .json larger than MAX_SOURCE_CHARS is shortened when it is read, before
    the budget ever sees it. The banner went inline into the content but not
    into ``notices``, so the coverage block listed the source as truncated with
    nothing said about it -- observed on setups.current_tracker in the
    2026-08-08 controlled run.
    """
    from ai_summary import MAX_SOURCE_CHARS, build_evidence_package

    big = _touch(
        tmp_path / "tracker.json",
        json.dumps({"setups": {f"S{i}": {"pad": "p" * 80} for i in range(400)}}),
    )
    assert big.stat().st_size > MAX_SOURCE_CHARS

    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tracker": big},
        session_date=SESSION,
    )

    source = _by_id(evidence)["setups.current_tracker"]
    assert source["truncated"] is True
    assert source["status"] == "available"
    banner = f"[showing the first {MAX_SOURCE_CHARS} of "
    assert any(notice.startswith(banner) for notice in source["notices"])
    assert banner in source["content"], "and it stays inline where the model reads it"
    # The read-cap banner is not a rejection reason.
    assert source["status_reason"] == ""

    entry = next(
        row for row in evidence["coverage"]["truncated"]
        if row["source_id"] == "setups.current_tracker"
    )
    assert any(notice.startswith(banner) for notice in entry["notices"])
