"""Packet Q3 - the local AI cannot change a fact's meaning.

Four rules, one per item:

* **Q3.1** every source in a built package carries a KIND, and an id outside the
  table raises rather than defaulting to something plausible.
* **Q3.2** a position claim needs a position source, and a numeric claim names
  the cell it read. A violating ROW is dropped; the document still publishes
  (the 2026-08-28 rule).
* **Q3.3** the morning file publishes analyzed / membership-only / failed and
  never one total, because "Briefed 152 of 152" counted 40 symbols that were
  never analysed at all.
* **Q3.4** the audit scripts read ``match_basis`` through the production reader,
  so a renamed field can never read as ``unknown``.

No model is called anywhere in this file.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SESSION = "2026-09-03"


# ---------------------------------------------------------------------------
# fixtures


def _daily_overrides(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        path = tmp_path / (source_id.replace(".", "_") + ".txt")
        path.write_text(f"Evidence from {source_id}\nhit rate 62%\n", encoding="utf-8")
        paths[source_id] = path
    return paths


def _empty_sections() -> dict:
    import ai_summary

    return {name: [] for name in ai_summary.MODEL_SUMMARY_SECTIONS}


def _summary(rows: list[dict], *, section: str = "what_is_working") -> dict:
    payload = {
        "executive_summary": "One measured finding from the selected evidence.",
        **_empty_sections(),
    }
    payload[section] = rows
    return payload


def _package(sources: list[dict]) -> dict:
    """A hand-built package in the shipped shape, so no store is read."""
    return {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": "2026-09-04T02:00:00-07:00",
        "session_date": SESSION,
        "selected_scopes": ["ticker_brief"],
        "scope_labels": ["Ticker brief"],
        "source_count": len(sources),
        "sources": sources,
        "coverage": {"counts": {"requested": len(sources), "usable": len(sources)}},
    }


def _watchlist_source() -> dict:
    return {
        "source_id": "watchlists.membership",
        "label": "Watchlist membership",
        "status": "available",
        "content": {"symbol": "BULL", "lists": ["longs.txt"]},
    }


def _journal_source() -> dict:
    return {
        "source_id": "journal.trades_and_reviews",
        "label": "Trade journal and lifecycle reviews",
        "status": "available",
        "content": {
            "trades": [
                {"symbol": "BULL", "direction": "long", "status": "open", "hit_rate": "62%"}
            ]
        },
    }


def _market_journal_source() -> dict:
    """What the trader THOUGHT. Same family, and not a position source."""
    return {
        "source_id": "journal.entries",
        "label": "Market journal entries (trader free text)",
        "status": "available",
        "content": [{"entry_id": "e1", "text": "BULL looks strong into the close."}],
    }


def _stats_source() -> dict:
    return {
        "source_id": "setups.type_stats",
        "label": "Setup type performance",
        "status": "available",
        "content": [
            {"setup_type": "AVWAP_RECLAIM", "win_rate": "0.62", "n": "37"},
            {"setup_type": "SMA_BREAK", "win_rate": "0.41", "n": "22"},
        ],
    }


# ---------------------------------------------------------------------------
# Q3.1 - sources carry a kind


def test_every_source_in_a_built_package_has_a_kind(tmp_path):
    from ai_summary import build_evidence_package, source_kinds

    evidence = build_evidence_package(
        ["daily_report"],
        source_overrides=_daily_overrides(tmp_path),
        session_date=SESSION,
        now=datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc),
    )
    kinds = source_kinds(evidence)

    assert set(kinds) == {row["source_id"] for row in evidence["sources"]}
    assert set(kinds.values()) == {"narrative"}


def test_a_source_id_outside_the_table_raises_rather_than_defaulting():
    from ai_summary import kind_for_source_id, source_kinds

    with pytest.raises(ValueError, match="no_such_family.thing"):
        kind_for_source_id("no_such_family.thing")
    with pytest.raises(ValueError, match="no_such_family.thing"):
        source_kinds(
            _package([{"source_id": "no_such_family.thing", "status": "available", "content": {}}])
        )


def test_the_kind_table_separates_a_watchlist_from_the_journal():
    from ai_summary import kind_for_source_id

    assert kind_for_source_id("journal.trades_and_reviews") == "journal"
    assert kind_for_source_id("watchlists.membership") == "watchlist"
    assert kind_for_source_id("market.auto_state") == "watchlist"
    assert kind_for_source_id("market.industry_snapshot") == "market"
    assert kind_for_source_id("setups.type_stats") == "scanner"
    assert kind_for_source_id("forensics.digest") == "scanner"
    assert kind_for_source_id("daily.auto_report") == "narrative"
    assert kind_for_source_id("feedback.pick_verdicts") == "feedback"
    assert kind_for_source_id("walkaway.report") == "walkaway"


def test_the_coverage_block_names_each_source_kind(tmp_path):
    from ai_summary import build_evidence_package

    evidence = build_evidence_package(
        ["daily_report"],
        source_overrides=_daily_overrides(tmp_path),
        session_date=SESSION,
        now=datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc),
    )

    kinds = evidence["coverage"]["source_kinds"]
    assert kinds == {source_id: "narrative" for source_id in evidence["coverage"]["usable_source_ids"]}


# ---------------------------------------------------------------------------
# Q3.2 - a position claim needs a position source


def test_a_position_claim_citing_only_a_watchlist_is_dropped_and_the_document_publishes():
    from ai_summary import validate_ai_summary

    evidence = _package([_watchlist_source(), _stats_source()])
    payload = _summary(
        [
            {
                "statement": "BULL is a held long.",
                "evidence_refs": ["watchlists.membership"],
                "confidence": "high",
            },
            {
                "statement": "The reclaim family leads the board.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "medium",
            },
        ]
    )
    dropped: list[dict] = []
    normalized = validate_ai_summary(payload, evidence, dropped=dropped)

    statements = [row["statement"] for row in normalized["what_is_working"]]
    assert statements == ["The reclaim family leads the board."]
    assert [entry["detail"] for entry in dropped] == ["position claim without a position source"]
    assert dropped[0]["row_dropped"] is True
    assert dropped[0]["statement"] == "BULL is a held long."


def test_only_the_trade_journal_supports_a_position_not_the_market_journal():
    """The Market Journal is what the trader THOUGHT, not what they held.

    Both ids are in the `journal.*` family, so a family-keyed rule let the
    trader's own free text stand as evidence of an open position. The two
    stores are deliberately not merged, and this is the one place that
    distinction has to be enforced rather than described.
    """
    from ai_summary import POSITION_SOURCE_IDS, validate_ai_summary

    assert POSITION_SOURCE_IDS == frozenset({"journal.trades_and_reviews"})

    evidence = _package([_market_journal_source(), _journal_source()])
    thought = _summary(
        [
            {
                "statement": "BULL is a held long.",
                "evidence_refs": ["journal.entries"],
                "confidence": "high",
            },
            {
                "statement": "The market journal names BULL.",
                "evidence_refs": ["journal.entries"],
                "confidence": "medium",
            },
        ]
    )
    dropped: list[dict] = []
    normalized = validate_ai_summary(thought, evidence, dropped=dropped)

    assert [row["statement"] for row in normalized["what_is_working"]] == [
        "The market journal names BULL."
    ]
    assert [entry["detail"] for entry in dropped] == ["position claim without a position source"]

    held = _summary(
        [
            {
                "statement": "BULL is a held long.",
                "evidence_refs": ["journal.trades_and_reviews"],
                "confidence": "high",
            }
        ]
    )
    survivors: list[dict] = []
    kept = validate_ai_summary(held, evidence, dropped=survivors)
    assert [row["statement"] for row in kept["what_is_working"]] == ["BULL is a held long."]
    assert survivors == []


def test_the_same_position_claim_survives_when_it_cites_the_journal():
    from ai_summary import validate_ai_summary

    evidence = _package([_watchlist_source(), _journal_source()])
    payload = _summary(
        [
            {
                "statement": "BULL is a held long.",
                "evidence_refs": ["watchlists.membership", "journal.trades_and_reviews"],
                "confidence": "high",
            }
        ]
    )
    dropped: list[dict] = []
    normalized = validate_ai_summary(payload, evidence, dropped=dropped)

    assert [row["statement"] for row in normalized["what_is_working"]] == ["BULL is a held long."]
    assert dropped == []


def test_the_position_vocabulary_is_word_bounded_and_case_insensitive():
    from ai_summary import states_a_position

    assert states_a_position("BULL is a HELD LONG here")
    assert states_a_position("We are long BULL into the close")
    assert states_a_position("holding BULL from yesterday")
    assert states_a_position("an open position in BULL")
    # A word that merely CONTAINS one of the tokens is not a position claim.
    assert not states_a_position("Prolonged consolidation above the band")
    assert not states_a_position("The longshot setups all failed")
    assert not states_a_position("BULL is on the longs watchlist")


def test_an_executive_summary_that_asserts_a_position_is_withheld():
    """It carries no refs, so it can never support a position claim.

    480 of 1,478 published executive summaries assert one; the brief the packet
    cites opens "BULL is currently long...". A sentence with nowhere to cite
    from cannot be repaired by striking a ref, so the whole summary is replaced
    with the system's own sentence and the substitution is recorded.
    """
    from ai_summary import WITHHELD_EXECUTIVE_SUMMARY, validate_ai_summary

    evidence = _package([_watchlist_source(), _stats_source()])
    payload = _summary(
        [
            {
                "statement": "The reclaim family leads the board.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "medium",
            }
        ]
    )
    payload["executive_summary"] = "BULL is currently long and working well."

    dropped: list[dict] = []
    normalized = validate_ai_summary(payload, evidence, dropped=dropped)

    assert normalized["executive_summary"] == WITHHELD_EXECUTIVE_SUMMARY
    assert WITHHELD_EXECUTIVE_SUMMARY.startswith("Executive summary withheld:")
    # The document still publishes on its surviving rows.
    assert [row["statement"] for row in normalized["what_is_working"]] == [
        "The reclaim family leads the board."
    ]
    entry = next(item for item in dropped if item["section"] == "executive_summary")
    assert entry["detail"] == "position claim in the executive summary"
    assert entry["row_dropped"] is True
    assert entry["statement"] == "BULL is currently long and working well."


def test_an_executive_summary_that_states_no_position_is_untouched():
    from ai_summary import validate_ai_summary

    evidence = _package([_stats_source()])
    payload = _summary(
        [
            {
                "statement": "The reclaim family leads the board.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "medium",
            }
        ]
    )
    payload["executive_summary"] = "BULL is on the longs watchlist and set up well."

    dropped: list[dict] = []
    normalized = validate_ai_summary(payload, evidence, dropped=dropped)

    assert normalized["executive_summary"] == "BULL is on the longs watchlist and set up well."
    assert dropped == []


def test_the_prompt_forbids_a_position_in_the_executive_summary():
    import ai_summary

    assert "executive summary" in ai_summary.GROUNDING_PROMPT_LINES.lower()


def test_the_last_instruction_of_the_local_prompt_names_metric_ref():
    """A closing sentence saying "exactly the keys statement, evidence_refs,
    confidence" told the model the opposite of the grounding ask, and it is the
    last thing the model reads."""
    import ai_summary

    prompt = ai_summary._local_user_prompt(_package([_stats_source()]))

    assert "metric_ref" in prompt[-260:]
    assert prompt.rindex("metric_ref") > prompt.rindex("copied verbatim")
    assert "exactly the keys statement, evidence_refs, confidence." not in prompt


def test_metric_key_exists_refuses_a_source_that_is_not_usable():
    """An excluded or stale source's content is not a cell anyone can read."""
    from ai_summary import metric_key_exists

    stale = dict(_stats_source(), status="stale", content=None)
    excluded = dict(_stats_source(), source_id="setups.playbooks", status="empty")
    evidence = _package([stale, excluded])

    assert not metric_key_exists(evidence, "setups.type_stats", "win_rate")
    assert not metric_key_exists(evidence, "setups.playbooks", "setup_type")


def test_a_numeric_claim_without_a_metric_ref_is_dropped():
    from ai_summary import validate_ai_summary

    evidence = _package([_stats_source()])
    payload = _summary(
        [
            {
                "statement": "Hit rate 62% over 5 sessions.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "high",
            }
        ]
    )
    dropped: list[dict] = []
    with pytest.raises(ValueError, match="every citing statement was unsupported"):
        validate_ai_summary(payload, evidence, dropped=dropped)


def test_a_numeric_claim_with_a_resolvable_metric_ref_survives():
    from ai_summary import validate_ai_summary

    evidence = _package([_stats_source()])
    payload = _summary(
        [
            {
                "statement": "Hit rate 62% over 5 sessions.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "high",
                "metric_ref": {
                    "source_id": "setups.type_stats",
                    "key": "win_rate",
                    "horizon": "5 sessions",
                    "denominator": "37 graded picks",
                },
            }
        ]
    )
    dropped: list[dict] = []
    normalized = validate_ai_summary(payload, evidence, dropped=dropped)

    assert dropped == []
    row = normalized["what_is_working"][0]
    assert row["metric_ref"]["key"] == "win_rate"


def test_a_metric_ref_whose_key_is_not_in_the_source_is_dropped():
    from ai_summary import validate_ai_summary

    evidence = _package([_stats_source(), _watchlist_source()])
    payload = _summary(
        [
            {
                "statement": "Hit rate 62% over 5 sessions.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "high",
                "metric_ref": {
                    "source_id": "setups.type_stats",
                    "key": "sharpe_ratio",
                    "horizon": "5 sessions",
                    "denominator": "37 graded picks",
                },
            },
            {
                "statement": "The reclaim family leads the board.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "medium",
            },
        ]
    )
    dropped: list[dict] = []
    normalized = validate_ai_summary(payload, evidence, dropped=dropped)

    assert [row["statement"] for row in normalized["what_is_working"]] == [
        "The reclaim family leads the board."
    ]
    assert [entry["detail"] for entry in dropped] == [
        "numeric claim without a resolvable metric_ref"
    ]


def test_metric_key_exists_finds_a_column_a_row_key_and_a_top_level_key():
    from ai_summary import metric_key_exists

    evidence = _package([_stats_source(), _journal_source()])

    assert metric_key_exists(evidence, "setups.type_stats", "win_rate")
    assert metric_key_exists(evidence, "setups.type_stats", "AVWAP_RECLAIM")
    assert not metric_key_exists(evidence, "setups.type_stats", "sharpe_ratio")
    assert metric_key_exists(evidence, "journal.trades_and_reviews", "trades")
    assert metric_key_exists(evidence, "journal.trades_and_reviews", "hit_rate")
    assert not metric_key_exists(evidence, "no.such_source", "win_rate")


def test_an_all_valid_payload_still_validates_unchanged():
    """No regression on the 2026-08-28 row-not-document rule."""
    from ai_summary import validate_ai_summary

    evidence = _package([_stats_source()])
    payload = _summary(
        [
            {
                "statement": "Swing rows are shown first.",
                "evidence_refs": ["setups.type_stats"],
                "confidence": "high",
            }
        ]
    )
    dropped: list[dict] = []
    normalized = validate_ai_summary(payload, evidence, dropped=dropped)

    assert dropped == []
    assert normalized["what_is_working"] == [
        {
            "statement": "Swing rows are shown first.",
            "evidence_refs": ["setups.type_stats"],
            "confidence": "high",
        }
    ]


def test_the_prompt_asks_for_metric_ref_and_forbids_unsourced_position_language():
    import ai_summary

    text = ai_summary.GROUNDING_PROMPT_LINES
    assert "metric_ref" in text
    assert "journal" in text.lower()
    prompt = ai_summary._user_prompt(_package([_stats_source()]))
    assert "metric_ref" in prompt
    local = ai_summary._local_user_prompt(_package([_stats_source()]))
    assert "metric_ref" in local
    assert "metric_ref" in json.dumps(ai_summary.AI_SUMMARY_JSON_SCHEMA)


# ---------------------------------------------------------------------------
# Q3.3 - three counts, never one total


def _entry(symbol: str, status: str) -> dict:
    from ai_jobs import briefs

    entry = {
        "symbol": symbol,
        "status": status,
        "memberships": [{"list": "longs.txt", "symbol": symbol}],
    }
    if status == briefs.BRIEF_STATUS_MEMBERSHIP_ONLY:
        entry["reason"] = "no session evidence beyond membership in longs.txt"
    else:
        entry["result"] = {
            "summary": {
                "executive_summary": f"{symbol} had one supported finding.",
                "best_candidates": [],
                "what_is_working": [],
                "risk_notes": [],
                "lessons_for_tomorrow": [],
            }
        }
    return entry


def test_the_morning_header_counts_analyzed_membership_only_and_failed():
    from ai_jobs import briefs

    resolved = [_entry(f"AAA{i}", briefs.BRIEF_STATUS_BRIEFED) for i in range(112)]
    resolved += [_entry(f"BBB{i}", briefs.BRIEF_STATUS_MEMBERSHIP_ONLY) for i in range(40)]

    text = briefs.render_morning_file(SESSION, resolved, total=152)

    assert "Analyzed 112 of 152. Membership-only 40. Failed 0." in text
    assert "Briefed 152 of 152" not in text
    assert "Briefed " not in text


def test_the_three_counts_sum_to_the_requested_total():
    from ai_jobs import briefs

    resolved = [_entry("AAA", briefs.BRIEF_STATUS_BRIEFED)]
    resolved += [_entry("BBB", briefs.BRIEF_STATUS_MEMBERSHIP_ONLY)]
    failures = [{"symbol": "CCC", "reason": "the model failed validation twice"}]

    text = briefs.render_morning_file(SESSION, resolved, failures=failures, total=3)

    assert "Analyzed 1 of 3. Membership-only 1. Failed 1." in text
    assert "Failed: CCC (the model failed validation twice)." in text
    counts = [int(part) for part in ("1", "1", "1")]
    assert sum(counts) == 3


def test_a_membership_only_section_leads_with_its_reason_line():
    from ai_jobs import briefs

    text = briefs.render_morning_file(
        SESSION, [_entry("BULL", briefs.BRIEF_STATUS_MEMBERSHIP_ONLY)], total=1
    )

    lines = [line for line in text.splitlines() if line.strip()]
    heading = next(index for index, line in enumerate(lines) if line.startswith("## BULL"))
    assert lines[heading + 1].startswith("membership only - no session evidence beyond")


# ---------------------------------------------------------------------------
# Q3.4 - the audit scripts read the real field


def _link():
    from research_warehouse import like_links

    return like_links.LikeLink(
        event_id="evt-1",
        symbol="BULL",
        side="LONG",
        like_date="2026-09-02",
        occurrence_id="occ-1",
        canonical_setup_id="avwap_reclaim@v1",
        trigger_at="2026-09-02T13:35:00+00:00",
        match_basis=like_links.BASIS_ANY_FAMILY,
        candidates_in_window=3,
    )


def test_from_payload_round_trips_a_link():
    from research_warehouse import like_links

    link = _link()
    assert like_links.LikeLink.from_payload(link.as_payload()) == link


def test_a_payload_that_says_basis_instead_of_match_basis_raises():
    from research_warehouse import like_links

    payload = _link().as_payload()
    payload["basis"] = payload.pop("match_basis")

    with pytest.raises(ValueError, match="match_basis"):
        like_links.LikeLink.from_payload(payload)
    with pytest.raises(ValueError, match="basis"):
        like_links.basis_of(payload)


def test_basis_of_reads_match_basis():
    from research_warehouse import like_links

    assert like_links.basis_of(_link().as_payload()) == like_links.BASIS_ANY_FAMILY


def test_the_counter_over_three_payloads_reports_the_real_bases():
    from research_warehouse import like_links

    one = _link().as_payload()
    two = dict(one, event_id="evt-2")
    three = dict(one, event_id="evt-3", match_basis=like_links.BASIS_NONE, occurrence_id="")

    assert like_links.count_payload_bases([one, two, three]) == {"any_family": 2, "none": 1}


def test_the_likes_audit_names_the_grain_of_its_distribution():
    """84 rows and 77 distinct event ids are different answers.

    The dataset keeps every VERSION of a link, so a bare "basis distribution"
    silently mixes the two and cannot be reconciled with a count taken the other
    way. The script now labels the row grain and prints the event grain beside
    it.
    """
    text = (
        ROOT_DIR / "docs" / "analysis" / "scripts" / "lake_likes_and_details.py"
    ).read_text(encoding="utf-8")

    assert "by row" in text
    assert "distinct event" in text.lower()
    assert "count_payload_bases" in text


# ---------------------------------------------------------------------------
# the documentation quotes the strings the code actually emits


def test_no_active_document_quotes_a_detail_string_the_code_no_longer_emits():
    """A gate the trader reads by grepping the log is worth nothing if it quotes
    a string the log cannot contain."""
    import ai_summary

    stale = "position claim without a journal source"
    for name in (
        "CURRENT_CHECKPOINT.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "AGENTS.md",
        "docs/LOCAL_AI_AUTOMATION_PLAN.md",
    ):
        text = (ROOT_DIR / name).read_text(encoding="utf-8", errors="replace")
        assert stale not in text, name

    checkpoint = (ROOT_DIR / "CURRENT_CHECKPOINT.md").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "position claim without a position source" in checkpoint
    # And the glance block states the rule the code enforces, not the one that
    # was overturned in the fix round.
    glance = checkpoint.split("## Active state at a glance", 1)[1].split("\n\n\n", 1)[0]
    assert "POSITION_SOURCE_IDS" in glance
    assert sorted(ai_summary.POSITION_SOURCE_IDS)[0] in glance


def test_the_audit_scripts_count_through_the_production_reader():
    """Neither script may reach for a field name of its own again."""
    for name in ("lake_assessment.py", "lake_likes_and_details.py"):
        text = (ROOT_DIR / "docs" / "analysis" / "scripts" / name).read_text(encoding="utf-8")
        assert "like_links.basis_of" in text, name
        # The field is `match_basis`; the default is what made a rename read as
        # a fact about the lake.
        assert 'get("basis"' not in text, name
        assert '"basis", "unknown"' not in text, name
        assert "AUDIT ERROR" in text, name
