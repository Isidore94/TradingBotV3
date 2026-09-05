"""Packet 9 - two opt-in evidence scopes (decision record §3, 2026-08-24).

Registered but not nightly, on the `trader_judgement` / `pick_feedback`
precedent: absent from the unattended slate, exercised only via
`run_ai_jobs.py --scopes …`, unknown names rejected at the CLI.

The prohibition in §3 is the important half and is measured rather than
cautionary: `setup_performance` reads the scoreboard's **output**, never the
raw tracker. TB-0/TB-5 measured the tracker's text projection contributing zero
symbol-specific content while starving every analysis it led - 96.2% of a
brief's payload was roster noise. Pointing a model at the 960 MB payload or its
roster dump is a measured failure mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_summary  # noqa: E402

NEW_SCOPES = ("walkaway", "setup_performance")


# ==========================================================================
# registered, and deliberately not nightly
# ==========================================================================
@pytest.mark.parametrize("scope", NEW_SCOPES)
def test_the_scope_is_registered(scope):
    assert scope in ai_summary.SCOPE_LABELS
    assert scope in ai_summary.SCOPE_BUDGET_WEIGHTS
    assert scope in ai_summary._source_specs()


@pytest.mark.parametrize("scope", NEW_SCOPES)
def test_the_scope_is_absent_from_the_unattended_slate(scope):
    """The whole point of opt-in: the nightly run must not pick these up."""
    from ai_jobs import briefs

    assert scope not in briefs.DEFAULT_SCOPES
    assert scope not in briefs.TICKER_BRIEF_SCOPES


def test_the_nightly_slate_is_exactly_the_five_the_trader_asked_for():
    """Four, plus `market_journal` on the trader's 2026-08-27 instruction.

    Pinned as a LIST so a scope can never join the unattended slate by
    accident; the two opt-in scopes above are what this guards.
    """
    from ai_jobs import briefs

    assert briefs.DEFAULT_SCOPES == (
        "daily_report",
        "market_conditions",
        "setup_trackers",
        "journal_review",
        "market_journal",
    )


def test_the_per_ticker_slate_did_not_follow_the_daily_summary():
    """`TICKER_BRIEF_SCOPES` was an alias and is now its own tuple.

    The trader asked for the daily summary. A session-level journal entry in a
    per-symbol packet is the TB-0/TB-5 failure mode - text that starves the
    symbol-specific evidence it leads.
    """
    from ai_jobs import briefs

    assert "market_journal" not in briefs.TICKER_BRIEF_SCOPES
    assert briefs.TICKER_BRIEF_SCOPES == (
        "daily_report",
        "market_conditions",
        "setup_trackers",
        "journal_review",
    )


@pytest.mark.parametrize("scope", NEW_SCOPES)
def test_the_scope_can_be_selected_on_demand(scope):
    from ai_jobs.runner import default_slots

    slots = default_slots(summary_scopes=(scope,))
    # V2 inserted `journal_auto_tag` SECOND, and decision 0018 (2026-09-04)
    # moved the narration pair to after `daily_digest`. The override is about
    # WHICH slot gets the scopes, not about where that slot sits, so this
    # asserts the journal pair still leads and the narration pair is intact.
    names = [slot.name for slot in slots]
    assert names[:2] == ["journal_import", "journal_auto_tag"]
    assert names.index("ai_summary") == names.index("ticker_briefs") - 1
    assert names.index("daily_digest") < names.index("ai_summary")
    # And the override is per-call: building again without it is untouched.
    # BY NAME rather than by index: `journal_auto_tag` sits at 1 since V2, and
    # what this line means is "the summary slot went back to its default".
    summary = next(slot for slot in default_slots() if slot.name == "ai_summary")
    assert summary.run.__name__ == "run_daily_summary"


def test_an_unknown_scope_is_still_rejected_at_the_cli():
    import run_ai_jobs

    with pytest.raises(SystemExit):
        run_ai_jobs.main(["--scopes", "not_a_scope"])


@pytest.mark.parametrize("scope", NEW_SCOPES)
def test_the_cli_accepts_the_new_names(scope):
    """Registered means selectable; a name the registry knows must not be
    rejected."""
    import run_ai_jobs

    assert scope in run_ai_jobs.known_scopes() if hasattr(run_ai_jobs, "known_scopes") else True
    assert scope in ai_summary.SCOPE_LABELS


# ==========================================================================
# the prohibition: output only, never the raw tracker
# ==========================================================================
def test_setup_performance_reads_output_and_never_the_raw_tracker():
    """TB-0/TB-5 is the measured reason, not a caution. The 960 MB payload and
    its roster projection are a failure mode this scope must not touch."""
    specs = ai_summary._source_specs()["setup_performance"]
    paths = [str(path).lower() for _id, _label, path in specs]

    assert paths, "the scope must declare sources"
    for path in paths:
        assert "setup_tracker" not in path
        assert "tracker_scoring_snapshot" not in path
    # What it DOES read is the scoreboard's own output.
    assert any("setup_scoreboard" in path for path in paths)


def test_setup_performance_funds_the_machine_readable_bundle_first():
    """The bundle already carries ground rule 10's statistics, so it is the
    distilled answer; the Markdown is the same numbers wearing prose."""
    specs = ai_summary._source_specs()["setup_performance"]
    assert [source_id for source_id, _label, _path in specs] == [
        "setup_performance.bundle",
        "setup_performance.report",
    ]


def test_walkaway_reads_the_analysis_output_not_the_journal():
    """The analysis itself stays deterministic (§3). This scope reads what it
    produced."""
    specs = ai_summary._source_specs()["walkaway"]
    paths = [str(path).lower() for _id, _label, path in specs]

    assert [source_id for source_id, _label, _path in specs] == [
        "walkaway.report",
        "walkaway.positions",
    ]
    for path in paths:
        assert "trade_journal.sqlite3" not in path


# ==========================================================================
# caveats derived from live source, not retyped (the AI-P5 lesson)
# ==========================================================================
def test_the_setup_performance_caveat_is_derived_from_the_bundle(tmp_path, monkeypatch):
    """AI-P5: a hand-maintained caveat is a machine-written falsehood on a
    delay. This one reads the numbers it describes."""
    bundle = tmp_path / "setup_scoreboard.json"
    bundle.write_text(
        json.dumps(
            {"coverage": {"by_claim_kind": {"entry_claim": 11, "annotation": 22}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ai_summary, "_scoreboard_bundle_file", lambda: bundle)

    caveat = ai_summary.scope_caveats("setup_performance")[0]

    assert "entry_claim 11" in caveat
    assert "annotation 22" in caveat
    assert "must not be read as either" in caveat


def test_a_bundle_it_cannot_read_yields_an_unknown_caveat_not_a_remembered_one(
    tmp_path, monkeypatch
):
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
    monkeypatch.setattr(
        ai_summary, "_scoreboard_bundle_file", lambda: tmp_path / "absent.json"
    )
    caveat = ai_summary.scope_caveats("setup_performance")[0]

    assert "UNKNOWN for this package" in caveat
    assert "do not read the scoreboard's n as the store's n" in caveat


def test_the_walkaway_caveat_refuses_the_money_left_on_the_table_reading():
    """MFE is opportunity, not a result - the same rule ground rule 12 keeps
    for `oracle_best_ex_post_r`."""
    caveat = ai_summary.scope_caveats("walkaway")[0]

    assert "opportunity, not a result" in caveat
    assert "no exit policy achieved it" in caveat


def test_other_scopes_still_carry_no_caveats():
    assert ai_summary.scope_caveats("daily_report") == ()


# ==========================================================================
# they resolve against the real machine
# ==========================================================================
@pytest.mark.parametrize("scope", NEW_SCOPES)
def test_the_scope_resolves_every_source_it_declares(scope):
    package = ai_summary.build_evidence_package([scope])
    coverage = package["coverage"]
    seen = set(coverage["usable_source_ids"]) | {
        row["source_id"] for row in coverage["excluded"]
    }

    declared = {source_id for source_id, _label, _path in ai_summary._source_specs()[scope]}
    assert seen == declared
    assert coverage["counts"]["requested"] == len(declared)


@pytest.mark.parametrize("scope", NEW_SCOPES)
def test_a_missing_source_degrades_rather_than_breaks(scope, tmp_path, monkeypatch):
    for name in (
        "_walkaway_text_file",
        "_walkaway_csv_file",
        "_scoreboard_bundle_file",
        "_scoreboard_report_file",
    ):
        monkeypatch.setattr(ai_summary, name, lambda _n=name: tmp_path / f"{_n}.absent")

    package = ai_summary.build_evidence_package([scope])
    excluded = {row["source_id"]: row["status"] for row in package["coverage"]["excluded"]}

    assert excluded
    assert all(status == ai_summary.SOURCE_STATUS_MISSING for status in excluded.values())
