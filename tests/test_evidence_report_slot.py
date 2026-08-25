"""R10.I - the evidence-report slot, built ahead of its window.

R10.I was specified to be built "after two weeks of R10.A collection". The
trader waived that **sequencing** on 2026-08-24 and explicitly did NOT waive the
gate it protects (decision record §4): until the window is met, every report
must state its n, label everything `discovery`, and say **in words** that the
window is not met.

So most of these tests are about what the report REFUSES to imply. A report
over a near-empty ledger is honest scaffolding; the one thing it must never be
is quiet or confident about it.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import evidence_report as er  # noqa: E402


# ==========================================================================
# the gate the override did NOT waive
# ==========================================================================
def test_a_report_over_an_empty_ledger_says_the_window_is_unmet():
    """The pinned proof. A report with nothing in it must not go quiet, and
    must not read as a finding either."""
    report = er.build_report(session_date="2026-08-24", ledger_rows=0, sessions_collected=0)

    assert report["window"]["window_met"] is False
    assert "COLLECTION WINDOW NOT MET" in report["window"]["statement"]
    assert "honest scaffolding, not a finding" in report["window"]["statement"]
    assert "Do not promote, demote, or change anything" in report["window"]["statement"]
    assert report["window"]["statement"] in report["summary"]


def test_the_window_statement_is_the_first_thing_rendered():
    """A reader who skims must not be able to miss it."""
    markdown = er.render_markdown(
        er.build_report(session_date="2026-08-24", ledger_rows=5, sessions_collected=1)
    )
    body = markdown.split("\n\n", 1)[1]

    assert body.startswith("**COLLECTION WINDOW NOT MET")


def test_every_report_carries_its_n():
    report = er.build_report(
        session_date="2026-08-24",
        ledger_rows=13394,
        sessions_collected=1,
        cohorts={"veto": [{"a": 1}], "like": [{"b": 2}, {"c": 3}]},
    )

    assert report["ledger_rows"] == 13394
    assert report["counts"]["cohorts"] == 3
    assert "n: 13394 outcome ledger row(s)" in report["summary"]
    assert "Collection: 1 of 10 session(s)" in report["summary"]


def test_a_nested_section_is_counted_rather_than_reported_as_zero():
    """An n of zero beside real data reads as "nothing measured", which is
    worse than no n at all - the first live run reported exactly that."""
    report = er.build_report(
        session_date="2026-08-24",
        cohorts={"veto": [{"a": 1}] * 16, "like": [{"b": 2}] * 28},
    )
    assert report["counts"]["cohorts"] == 44


def test_everything_is_labelled_discovery():
    report = er.build_report(session_date="2026-08-24", ledger_rows=99999, sessions_collected=99)

    assert report["evidence_label"] == "discovery"
    markdown = er.render_markdown(report)
    assert "discovery" in markdown


def test_a_met_window_still_does_not_promote_anything():
    """Meeting the collection window makes the data usable; it does not turn a
    post-hoc rollup into a confirmation."""
    report = er.build_report(session_date="2026-08-24", sessions_collected=20)

    assert report["window"]["window_met"] is True
    assert "still labelled `discovery`" in report["window"]["statement"]
    assert report["evidence_label"] == "discovery"


def test_a_caller_that_cannot_count_reads_as_unmet():
    """The conservative direction, and the only one that cannot turn
    scaffolding into a finding by accident."""
    assert er.collection_state(0)["window_met"] is False
    assert er.collection_state(-5)["sessions_collected"] == 0


def test_an_unreadable_source_makes_the_report_incomplete_not_empty():
    report = er.build_report(
        session_date="2026-08-24",
        unavailable={"veto cohort": "file not found"},
    )
    assert "INCOMPLETE rather than empty" in report["summary"]
    assert report["unavailable"] == {"veto cohort": "file not found"}


def test_the_schema_is_named_never_numbered():
    assert er.REPORT_SCHEMA == "evidence_report_v1"


# ==========================================================================
# the slot
# ==========================================================================
def test_the_slot_is_appended_last_and_never_reorders_the_others():
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]

    assert names[:3] == ["journal_import", "ai_summary", "ticker_briefs"]
    # Phase 2's `daily_digest` was appended after this slot on 2026-08-24, so
    # this is no longer last - which is the rule working, not breaking: later
    # phases append, and the ones already here never move.
    assert names.index("evidence_report") == 5
    # It reads what the cohorts produced, so it runs after them: a report ahead
    # of its inputs would describe last night's evidence.
    assert names.index("like_cohort_grading") < names.index("evidence_report")


def test_the_slot_reserves_the_five_minute_class_and_calls_no_model():
    from ai_jobs.runner import default_slots

    slot = [item for item in default_slots() if item.name == "evidence_report"][0]

    assert slot.reserve_minutes == 5.0
    assert slot.max_attempts == 3
    assert slot.enabled


def test_the_slot_needs_no_local_provider(monkeypatch, tmp_path):
    """Deterministic: it must not need, or touch, the model endpoint."""
    import ai_summary

    monkeypatch.setattr(
        ai_summary,
        "local_provider_enabled",
        lambda: pytest.fail("the evidence report must not consult the model provider"),
    )
    result = er.run_evidence_report(session_date="2026-08-24", report_dir=tmp_path)

    assert result["status"] == "ok"
    assert result["model"] == ""


def test_the_slot_publishes_both_halves_atomically(tmp_path):
    result = er.run_evidence_report(session_date="2026-08-24", report_dir=tmp_path)

    assert len(result["outputs"]) == 2
    payload = json.loads((tmp_path / "evidence_report.json").read_text(encoding="utf-8"))
    assert payload["schema"] == er.REPORT_SCHEMA
    assert (tmp_path / "evidence_report.md").read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_the_slot_says_scaffolding_in_its_ledger_reason(tmp_path):
    """The ledger row is what a later reader sees first."""
    result = er.run_evidence_report(session_date="2026-08-24", report_dir=tmp_path)
    assert "window NOT met - scaffolding only" in result["reason"]


def test_a_failed_publish_is_reported_as_failed(tmp_path):
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory", encoding="utf-8")

    result = er.run_evidence_report(session_date="2026-08-24", report_dir=blocked)

    assert result["status"] == "failed"
    assert result["outputs"] == []


# ==========================================================================
# the opt-in scope
# ==========================================================================
def test_the_market_journal_scope_is_registered_but_not_nightly():
    """Free-text journal entries reach an AI scope OPT-IN ONLY - a recorded
    trader decision, unchanged."""
    import ai_summary
    from ai_jobs import briefs

    assert "market_journal" in ai_summary.SCOPE_LABELS
    assert "market_journal" not in briefs.DEFAULT_SCOPES
    assert "market_journal" not in briefs.TICKER_BRIEF_SCOPES


def test_the_scope_funds_the_distilled_half_before_the_free_text():
    """The same rule every other scope keeps: the raw stream LAST, or it
    starves every analysis derived from it."""
    import ai_summary

    specs = ai_summary._source_specs()["market_journal"]
    assert [source_id for source_id, _label, _path in specs] == [
        "journal.evidence_report",
        "journal.day_context",
        "journal.entries",
    ]


def test_the_scope_resolves_all_three_sources():
    import ai_summary

    package = ai_summary.build_evidence_package(["market_journal"])
    coverage = package["coverage"]
    seen = set(coverage["usable_source_ids"]) | {
        row["source_id"] for row in coverage["excluded"]
    }
    assert seen == {"journal.evidence_report", "journal.day_context", "journal.entries"}


# ==========================================================================
# nothing in this chain reaches a live surface (R9.5 idiom)
# ==========================================================================
def test_the_report_module_reaches_no_live_surface():
    source = (SCRIPTS_DIR / "ai_jobs" / "evidence_report.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            reached.add(node.attr)
        elif isinstance(node, ast.Name):
            reached.add(node.id)
        elif isinstance(node, ast.Import):
            reached.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            reached.add((node.module or "").split(".")[0])

    for forbidden in (
        "review_policy",
        "focus_service",
        "FocusService",
        "focus_picks",
        "add_alert",
        "set_market_environment",
        "review_policy_json",
    ):
        assert forbidden not in reached, f"the evidence report must not reach {forbidden}"


def test_the_report_never_names_a_live_store_in_its_CODE():
    """Checked against string LITERALS, not the file text.

    The module's own docstring names `review_policy.json` in order to rule it
    out, and a test that cannot tell an explanation from a reference teaches
    you to delete the explanation - which is what happened when this was first
    written against the raw source.
    """
    tree = ast.parse((SCRIPTS_DIR / "ai_jobs" / "evidence_report.py").read_text(encoding="utf-8"))
    # Identified by POSITION, not by value: `ast.get_docstring` returns the
    # cleaned (dedented, stripped) text, which never equals the raw constant,
    # so comparing values silently excludes nothing.
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]
    joined = " ".join(literals).lower()
    for forbidden in ("longs.txt", "shorts.txt", "focus_longs", "review_policy"):
        assert forbidden not in joined, f"the evidence report must not name {forbidden}"
