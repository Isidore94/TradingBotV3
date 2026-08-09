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
        def list_import_runs(self, limit=25):
            return [{"status": "ok", "finished_at": f"{SESSION}T18:00:00-04:00"}]

        def list_trades(self, **kwargs):
            raise RuntimeError("journal database is locked")

        def list_opportunity_events(self, **kwargs):  # pragma: no cover
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
def test_a_stale_source_leaves_the_model_package_entirely(tmp_path):
    """The 2026-07-30 auto_report incident, now excluded rather than warned.

    It used to stay in the package with a STALE notice attached. The model
    narrated it as the session's own data anyway -- a warning the model may
    disregard is not a control (Sol 5.6 verification review, item 5). The
    daily brief reads current evidence or none.
    """
    from ai_summary import build_evidence_package

    overrides = _daily(tmp_path)
    _touch(overrides["daily.auto_report"], "Report body.\n", session="2026-07-30")

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )

    assert "daily.auto_report" not in _by_id(evidence), "stale evidence must not reach the model"
    excluded = _coverage_by_id(evidence)["daily.auto_report"]
    assert excluded["status"] == "stale"
    assert "2026-07-30" in excluded["reason"]
    assert excluded["reason"].endswith("current evidence or none")

    stale = {row["source_id"] for row in evidence["coverage"]["stale"]}
    assert stale == {"daily.auto_report"}
    assert evidence["coverage"]["counts"]["stale"] == 1
    entry = next(
        row for row in evidence["coverage"]["stale"] if row["source_id"] == "daily.auto_report"
    )
    assert entry["content_through"] == "2026-07-30"
    assert entry["content_through_basis"] == "mtime"


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
    from ai_summary import MODEL_SUMMARY_SECTIONS

    payload = {name: [] for name in MODEL_SUMMARY_SECTIONS}
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
    from ai_summary import MODEL_SUMMARY_SECTIONS, build_evidence_package, validate_ai_summary

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=_daily(tmp_path), session_date=SESSION
    )
    thin = {name: [] for name in MODEL_SUMMARY_SECTIONS}
    thin["executive_summary"] = "A thin night with nothing to report is a valid answer."

    validated = validate_ai_summary(thin, evidence)
    assert all(validated[name] == [] for name in MODEL_SUMMARY_SECTIONS)

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
    # daily.master_events is the only usable source left: market_prep is empty
    # and auto_report is stale.
    validated = validate_ai_summary(_summary("daily.master_events"), evidence)
    assert "data_quality" not in validated, "the model does not write this section"

    # A model observation smuggled in from an older document is *replaced*,
    # not appended to: a stale count sitting above the real one is worse than
    # no count at all.
    validated["data_quality"] = [
        {"statement": "A model observation.", "evidence_refs": [], "confidence": "low"}
    ]
    merged = merge_coverage_into_summary(validated, evidence)
    system_rows = [row["statement"] for row in merged["data_quality"]]

    assert all(row.startswith(COVERAGE_STATEMENT_PREFIX) for row in system_rows)
    assert "A model observation." not in system_rows
    assert any("1 of 3 requested source(s) were usable" in row for row in system_rows)
    assert any("1 source(s) empty: daily.market_prep" in row for row in system_rows)
    assert any("1 source(s) stale: daily.auto_report" in row for row in system_rows)
    assert any("2026-07-30" in row and "withheld from the model" in row for row in system_rows)
    # Merging twice is idempotent.
    assert merge_coverage_into_summary(merged, evidence)["data_quality"] == merged["data_quality"]


def test_the_model_is_told_not_to_write_the_machine_owned_section(tmp_path):
    from ai_summary import (
        MODEL_SUMMARY_SECTIONS,
        SYSTEM_SUMMARY_SECTIONS,
        AI_SUMMARY_JSON_SCHEMA,
        build_evidence_package,
        validate_ai_summary,
    )

    # The schema sent to every provider does not advertise the section, and
    # additionalProperties:False is what makes that enforceable.
    assert "data_quality" not in AI_SUMMARY_JSON_SCHEMA["properties"]
    assert AI_SUMMARY_JSON_SCHEMA["additionalProperties"] is False
    assert set(SYSTEM_SUMMARY_SECTIONS).isdisjoint(MODEL_SUMMARY_SECTIONS)

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=_daily(tmp_path), session_date=SESSION
    )
    overreaching = _summary("daily.auto_report")
    overreaching["data_quality"] = [
        {"statement": "All 3 sources looked fine.", "evidence_refs": [], "confidence": "high"}
    ]
    with pytest.raises(ValueError, match="written by the system"):
        validate_ai_summary(overreaching, evidence)


def test_a_published_document_must_have_system_authored_coverage(tmp_path):
    from ai_summary import build_evidence_package, validate_published_summary

    evidence = build_evidence_package(
        ["daily_report"], source_overrides=_daily(tmp_path), session_date=SESSION
    )
    document = _summary("daily.auto_report")
    document["data_quality"] = [
        {"statement": "Coverage was fine.", "evidence_refs": [], "confidence": "high"}
    ]
    with pytest.raises(ValueError, match="not system-authored"):
        validate_published_summary(document, evidence)

    document["data_quality"] = [
        {"statement": "[system] Evidence coverage: 3 of 3.", "evidence_refs": [], "confidence": "high"}
    ]
    assert validate_published_summary(document, evidence)["data_quality"]


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


def test_an_unknown_status_from_a_job_fails_closed(tmp_path):
    """"I do not know what happened" must never be filed as success.

    An unrecognised status used to coerce to ``ok``, so a typo, a status added
    by a later phase, or a half-written return value was recorded as a
    trustworthy completion -- and, because completed_jobs counts ok rows, was
    never retried (Sol 5.6 verification review, item 7).
    """
    from unittest import mock

    from ai_jobs import ledger, runner, store, window

    led = tmp_path / "ledger.jsonl"

    with (
        mock.patch.object(store, "store_available", return_value=(True, "ready")),
        mock.patch.object(window, "launch_allowed", return_value=(True, "open")),
        mock.patch.object(window, "market_session_block", return_value=""),
    ):
        report = runner.run_slots(
            [runner.JobSlot(name="ai_summary", run=lambda **k: {"status": "wonderful"})],
            now=datetime.fromisoformat("2026-08-08T02:00:00+00:00"),
            ledger_path=led,
        )

    row = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "failed"
    assert "wonderful" in row["reason"], "the unrecognised status is named, not swallowed"
    assert report.failed == 1 and report.ran == 0
    # And it is therefore retried by the next firing.
    assert ledger.completed_jobs(row["session_date"], path=led) == set()


def test_every_status_a_job_may_report_is_honoured(tmp_path):
    from unittest import mock

    from ai_jobs import ledger, runner, store, window

    for status in (ledger.STATUS_OK, ledger.STATUS_DEGRADED, ledger.STATUS_MANUAL):
        led = tmp_path / f"ledger_{status}.jsonl"
        with (
            mock.patch.object(store, "store_available", return_value=(True, "ready")),
            mock.patch.object(window, "launch_allowed", return_value=(True, "open")),
            mock.patch.object(window, "market_session_block", return_value=""),
        ):
            runner.run_slots(
                [runner.JobSlot(name="ai_summary", run=lambda **k: {"status": status})],
                now=datetime.fromisoformat("2026-08-08T02:00:00+00:00"),
                ledger_path=led,
            )
        row = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
        assert row["status"] == status


def test_the_degraded_document_states_coverage_without_a_model(tmp_path):
    from ai_summary import build_evidence_package, degraded_result, validate_published_summary

    overrides = _daily(tmp_path)
    for name in overrides:
        _touch(overrides[name], "\n")
    evidence = build_evidence_package(
        ["daily_report"], source_overrides=overrides, session_date=SESSION
    )

    result = degraded_result(evidence, reason="No usable evidence.")

    assert result["status"] == "degraded_no_narrative"
    assert result["model"] == ""
    # It is a valid document by the same validator every published summary passes.
    validate_published_summary(result["summary"], evidence)
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
        source_overrides={"setups.current_tiers": big},
        session_date=SESSION,
    )

    source = _by_id(evidence)["setups.current_tiers"]
    assert source["truncated"] is True
    assert source["status"] == "available"
    banner = f"[showing the first {MAX_SOURCE_CHARS} of "
    assert any(notice.startswith(banner) for notice in source["notices"])
    assert banner in source["content"], "and it stays inline where the model reads it"
    # The read-cap banner is not a rejection reason.
    assert source["status_reason"] == ""

    entry = next(
        row for row in evidence["coverage"]["truncated"]
        if row["source_id"] == "setups.current_tiers"
    )
    assert any(notice.startswith(banner) for notice in entry["notices"])


# ---------------------------------------------------------------------------
# journal honesty (Sol 5.6 verification review, item 3)
# ---------------------------------------------------------------------------
class _Journal:
    """A journal store stub with an import log, filtered like the real one."""

    def __init__(self, *, trades=(), events=(), imported_through=SESSION, status="ok"):
        self._trades = list(trades)
        self._events = list(events)
        self._imported_through = imported_through
        self._status = status

    def list_import_runs(self, limit=25):
        if not self._imported_through:
            return [{"status": "failed", "finished_at": "", "message": "broker refused"}]
        return [{"status": self._status, "finished_at": f"{self._imported_through}T18:30:00-04:00"}]

    def list_trades(self, *, trade_date=None, **kwargs):
        if trade_date is None:
            return list(self._trades)
        return [row for row in self._trades if row.get("trade_date") == trade_date]

    def list_opportunity_events(self, *, trade_date=None, limit=1000, **kwargs):
        if trade_date is None:
            return list(self._events)
        return [row for row in self._events if str(row.get("occurred_at", ""))[:10] == trade_date]


def _trade(trade_date, symbol="AAA"):
    return {
        "trade_id": f"{symbol}-{trade_date}",
        "trade_date": trade_date,
        "symbol": symbol,
        "direction": "LONG",
        "status": "CLOSED",
        "opened_at": f"{trade_date}T10:00:00-04:00",
        "closed_at": f"{trade_date}T14:00:00-04:00",
        "net_pnl": 120.0,
    }


def test_the_journal_is_filtered_to_the_target_session():
    """The proof run narrated a 2026-06-18 trade under an August heading.

    The package handed over the whole journal and trusted the model to pick
    the right rows out of it. It did not (Sol 5.6 verification review, item 3a).
    """
    from ai_summary import build_evidence_package

    store = _Journal(trades=[_trade(SESSION), _trade("2026-06-18", "OLD")])
    evidence = build_evidence_package(
        ["journal_review"], journal_store=store, session_date=SESSION
    )

    source = _by_id(evidence)["journal.trades_and_reviews"]
    symbols = {row["symbol"] for row in source["content"]["trades"]}
    assert symbols == {"AAA"}, "a trade from another session must not be in the package"
    assert source["content"]["session_date"] == SESSION


def test_a_session_with_no_journal_activity_is_an_honest_empty_source():
    from ai_summary import build_evidence_package

    store = _Journal(trades=[_trade("2026-06-18", "OLD")], imported_through=SESSION)
    evidence = build_evidence_package(
        ["journal_review"], journal_store=store, session_date=SESSION
    )

    entry = _coverage_by_id(evidence)["journal.trades_and_reviews"]
    assert entry["status"] == "empty", "no trades that session is empty, not stale"
    assert "no trades or lifecycle events" in entry["reason"]
    assert "current through" in entry["reason"]


def test_a_stalled_import_makes_the_journal_stale_and_hides_its_old_rows():
    """Imports stopped: the journal still answers, cheerfully, with old data."""
    from ai_summary import build_evidence_package

    store = _Journal(trades=[_trade("2026-06-18", "OLD")], imported_through="2026-06-18")
    evidence = build_evidence_package(
        ["journal_review"], journal_store=store, session_date=SESSION
    )

    assert "journal.trades_and_reviews" not in _by_id(evidence)
    entry = _coverage_by_id(evidence)["journal.trades_and_reviews"]
    assert entry["status"] == "stale"
    assert "2026-06-18" in entry["reason"]


def test_a_journal_with_no_successful_import_is_unavailable_not_empty():
    from ai_summary import build_evidence_package

    store = _Journal(trades=[_trade(SESSION)], imported_through="")
    evidence = build_evidence_package(
        ["journal_review"], journal_store=store, session_date=SESSION
    )

    entry = _coverage_by_id(evidence)["journal.trades_and_reviews"]
    assert entry["status"] == "unavailable"
    assert "no successful journal import" in entry["reason"]


def test_import_health_reaches_the_data_quality_section():
    from ai_summary import build_evidence_package, merge_coverage_into_summary

    store = _Journal(
        trades=[_trade(SESSION), _trade("2026-06-18", "OLD")], imported_through=SESSION
    )
    evidence = build_evidence_package(
        ["journal_review"], journal_store=store, session_date=SESSION
    )

    health = evidence["coverage"]["journal_import_health"]
    assert health["last_successful_import_date"] == SESSION
    assert health["newest_execution_date"] == SESSION
    assert health["lag_days"] == 0
    assert health["session_row_count"] == 1

    rows = merge_coverage_into_summary({}, evidence)["data_quality"]
    line = next(row["statement"] for row in rows if "Journal import health" in row["statement"])
    assert f"last successful import {SESSION}" in line
    assert "lag 0 day(s)" in line
    assert "1 row(s) for the reviewed session" in line


def test_a_stalled_import_reports_its_lag_in_days():
    from ai_summary import build_evidence_package, merge_coverage_into_summary

    store = _Journal(trades=[_trade("2026-06-18", "OLD")], imported_through="2026-06-18")
    evidence = build_evidence_package(
        ["journal_review"], journal_store=store, session_date=SESSION
    )

    assert evidence["coverage"]["journal_import_health"]["lag_days"] == 50
    rows = merge_coverage_into_summary({}, evidence)["data_quality"]
    assert any("lag 50 day(s)" in row["statement"] for row in rows)


# ---------------------------------------------------------------------------
# observed_at vs content_through (item 3b)
# ---------------------------------------------------------------------------
def test_a_freshly_rewritten_file_with_old_content_is_still_stale(tmp_path):
    """mtime cannot tell "rewritten" from "updated"."""
    from ai_summary import build_evidence_package

    # Written just now, but every record inside it is from an earlier session.
    ledger = tmp_path / "learning.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps({"as_of": "2026-06-18T12:00:00-04:00", "symbol": s}) for s in "ABC"
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.bounce_learning": ledger},
        session_date=SESSION,
    )

    entry = _coverage_by_id(evidence)["setups.bounce_learning"]
    assert entry["status"] == "stale"
    assert "2026-06-18" in entry["reason"]
    assert "content" in entry["reason"], "the basis must say the date was measured, not inferred"


def test_content_through_falls_back_to_mtime_and_says_so(tmp_path):
    from ai_summary import build_evidence_package

    plain = _touch(tmp_path / "report.txt", "A narrative report with no timestamps.\n")
    evidence = build_evidence_package(
        ["daily_report"],
        source_overrides=_daily(tmp_path, **{"daily.auto_report": plain}),
        session_date=SESSION,
    )

    source = _by_id(evidence)["daily.auto_report"]
    assert source["content_through"] == SESSION
    assert source["content_through_basis"] == "mtime"
    assert source["observed_at"] > source["as_of"], "observed_at is the read, not the data"


# ---------------------------------------------------------------------------
# bounded reads (item 8)
# ---------------------------------------------------------------------------
def test_a_huge_file_is_never_read_whole(tmp_path, monkeypatch):
    """setups.current_tracker was measured at 762 MB on the live desk."""
    import ai_summary

    big = tmp_path / "tracker.json"
    big.write_bytes(b'{"setups": {"A": "' + b"x" * (3 * ai_summary.MAX_SOURCE_BYTES) + b'"}}')
    when = datetime.fromisoformat(f"{SESSION}T12:00:00").timestamp()
    os.utime(big, (when, when))

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("read_text would allocate the whole file")

    monkeypatch.setattr(Path, "read_text", _forbidden)

    evidence = ai_summary.build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tiers": big},
        session_date=SESSION,
    )

    source = _by_id(evidence)["setups.current_tiers"]
    assert source["status"] == "available"
    assert len(json.dumps(source["content"])) < ai_summary.MAX_SOURCE_BYTES


def test_an_oversized_file_is_identified_without_hashing_all_of_it(tmp_path):
    import ai_summary

    big = tmp_path / "tracker.json"
    big.write_bytes(b'{"x": "' + b"y" * (ai_summary.MAX_HASHED_FILE_BYTES + 1024) + b'"}')
    when = datetime.fromisoformat(f"{SESSION}T12:00:00").timestamp()
    os.utime(big, (when, when))

    evidence = ai_summary.build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tiers": big},
        session_date=SESSION,
    )

    digest = _by_id(evidence)["setups.current_tiers"]["sha256"]
    assert digest.startswith("capped:"), "an oversized file must not claim a whole-file digest"

    # A small file keeps its real whole-file hash.
    small = _touch(tmp_path / "learning.json", json.dumps({"rows": [{"symbol": "AAA"}]}))
    evidence = ai_summary.build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.bounce_learning": small},
        session_date=SESSION,
    )
    assert not _by_id(evidence)["setups.bounce_learning"]["sha256"].startswith("capped:")


# ---------------------------------------------------------------------------
# the tracker extract and within-scope funding order (item 8)
# ---------------------------------------------------------------------------
def _huge_tracker(tmp_path):
    """A tracker whose file leads with old watchlists, like the real one."""
    import ai_summary

    path = tmp_path / "master_avwap_setup_tracker.json"
    watchlists = {
        f"2026-03-{day:02d}": {"symbols": ["OLD"] * 40} for day in range(1, 29)
    }
    padding = "z" * (3 * ai_summary.MAX_SOURCE_BYTES)
    path.write_text(
        json.dumps({"daily_watchlists": watchlists, "padding": padding}),
        encoding="utf-8",
    )
    when = datetime.fromisoformat(f"{SESSION}T12:00:00").timestamp()
    os.utime(path, (when, when))
    return path


def _snapshot(tmp_path, *, sessions=("2026-03-02", "2026-06-18", SESSION)):
    path = tmp_path / "master_avwap_tracker_scoring_snapshot.json"
    setups = {}
    for index, session in enumerate(sessions):
        for n in range(3):
            setups[f"S{index}-{n}"] = {
                "setup_id": f"S{index}-{n}",
                "symbol": f"SYM{index}{n}",
                "scan_date": session,
                "setup_status": "OPEN",
            }
    path.write_text(
        json.dumps({"source_record_count": len(setups), "setups": setups}),
        encoding="utf-8",
    )
    return path


def test_the_tracker_is_packaged_as_a_most_recent_extract_not_a_head_slice(tmp_path, monkeypatch):
    """A head slice of the real tracker was a list of March watchlists.

    The file leads with daily_watchlists, so its first 16,000 characters were
    whatever happened to serialise first -- not a sample of the tracker (Sol
    5.6 verification review, item 8).
    """
    import ai_summary

    monkeypatch.setattr(
        ai_summary, "_tracker_snapshot_path", lambda: _snapshot(tmp_path)
    )
    evidence = ai_summary.build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tracker": _huge_tracker(tmp_path)},
        session_date=SESSION,
    )

    source = _by_id(evidence)["setups.current_tracker"]
    assert source["status"] == "available"
    content = source["content"]
    assert "daily_watchlists" not in json.dumps(content), "no head slice of March watchlists"
    assert "most recent" in content["extract_note"]
    # Newest first, and the newest session is present.
    scan_dates = [row["scan_date"] for row in content["setups"]]
    assert scan_dates == sorted(scan_dates, reverse=True)
    assert scan_dates[0] == SESSION
    assert any(banner.startswith("[showing the ") for banner in source["notices"])


def test_a_tracker_with_no_snapshot_is_unavailable_rather_than_head_sliced(tmp_path, monkeypatch):
    import ai_summary

    monkeypatch.setattr(
        ai_summary, "_tracker_snapshot_path", lambda: tmp_path / "absent.json"
    )
    evidence = ai_summary.build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tracker": _huge_tracker(tmp_path)},
        session_date=SESSION,
    )

    entry = _coverage_by_id(evidence)["setups.current_tracker"]
    assert entry["status"] == "unavailable"
    assert "no bounded extract" in entry["reason"]
    assert "setups.current_tracker" not in _by_id(evidence)


def test_a_small_tracker_is_still_read_directly(tmp_path):
    # The extract path is a response to size, not a permanent detour.
    from ai_summary import build_evidence_package

    small = _touch(
        tmp_path / "tracker.json",
        json.dumps({"setups": {"AAA-1": {"symbol": "AAA", "scan_date": SESSION}}}),
    )
    evidence = build_evidence_package(
        ["setup_trackers"],
        source_overrides={"setups.current_tracker": small},
        session_date=SESSION,
    )

    source = _by_id(evidence)["setups.current_tracker"]
    assert source["content"]["setups"]["AAA-1"]["symbol"] == "AAA"
    assert "extract_note" not in source["content"]


def test_the_analytic_sub_sources_are_funded_before_the_raw_tracker(tmp_path, monkeypatch):
    """The tracker used to lead its scope and eat the whole budget.

    Everything derived from it -- type stats, horizons, tiers -- then arrived
    unfunded, which is the opposite of what the review is for.
    """
    import ai_summary

    monkeypatch.setattr(
        ai_summary, "_tracker_snapshot_path", lambda: _snapshot(tmp_path)
    )
    overrides = _oversized_trackers(tmp_path)
    overrides["setups.current_tracker"] = _huge_tracker(tmp_path)

    evidence = ai_summary.build_evidence_package(
        ["setup_trackers"], source_overrides=overrides, session_date=SESSION
    )

    funded = set(_by_id(evidence))
    excluded = _coverage_by_id(evidence)
    assert "setups.type_stats" in funded, "the distilled analysis must be funded first"
    assert "setups.recent_type_stats" in funded
    assert excluded.get("setups.current_tracker", {}).get("status") == "unfunded", (
        "the rawest, largest source is the one that gives way"
    )


def test_the_spec_order_puts_the_raw_tracker_last_in_its_scope():
    from ai_summary import _source_specs

    ids = [source_id for source_id, _label, _path in _source_specs()["setup_trackers"]]
    assert ids[-1] == "setups.current_tracker"
    assert ids.index("setups.type_stats") < ids.index("setups.current_tracker")
