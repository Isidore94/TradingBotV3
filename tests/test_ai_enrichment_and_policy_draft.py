"""LOCAL-AI Phase 3 and Phase 4 machinery — BUILT, RUNS GATED (packet W6).

Authorized 2026-08-24 (`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`
§2), ahead of their phase gates, on the R10.I scaffolding pattern: each refuses
to run — or labels its output non-authoritative — until its recorded gate passes.

**Phase 3, journal enrichment.** Summarize and tag new journal rows from the
`SETUPS_MAJOR.md` / `SETUPS_TEST.md` vocabulary and link them to the day's
alert/review evidence. Gated on Phase 2's ten clean digest sessions, because an
enrichment pass over a layer whose own facts have not been audited would be
building on unverified ground. **Advisory fields ONLY**: R7's invariant I7 says
tags, notes, reviews, planned stop/risk and tax status are the trader's, and no
machine path writes them.

**Phase 4, review-policy curation.** Writes `review_policy_draft.json` and
nothing else, ever. Its gate is the two-week side-by-side the plan has required
since it was written, and a draft is how that window accumulates — so this one
RUNS and labels, rather than refusing. `review_policy.json` stays the trader's
to save.

The load-bearing test in this file is the AST pair at the bottom: neither module
may write the live policy file or any trader-owned journal field. That is
checked by walking the code, not by trusting these paragraphs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import enrichment, policy_draft  # noqa: E402

NOW = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)


def _digest_sessions(root: Path, count: int) -> None:
    """`count` CONSECUTIVE clean session packs (Q4.1 counts a run, not a set).

    No AI store is configured inside a test, so every pack would otherwise
    record "ai job ledger: No AI store configured" - a real failure record that
    makes the pack INCOMPLETE and breaks the run for a reason that has nothing
    to do with what these tests are about.
    """
    from unittest import mock

    from ai_jobs import digest

    days = [f"2026-08-{day:02d}" for day in (10, 11, 12, 13, 14, 17, 18, 19, 20, 21)]
    with mock.patch.object(digest, "_read_job_rows", return_value=[]):
        for day in days[:count]:
            digest.run_daily_digest(
                session_date=day, now=NOW, root=root, narrate=False,
                finals=[{
                    "symbol": "AAPL", "direction": "long", "trade_date": day,
                    "env_key": "bullish_weak|midday", "close_r": 1.0, "mfe_r": 2.0, "mae_r": -0.5,
                }],
            )


# ---------------------------------------------------------------------------
# Phase 3 - the gate
# ---------------------------------------------------------------------------


def test_enrichment_refuses_to_run_below_the_digest_gate(tmp_path, monkeypatch):
    """No model is called, and nothing is written, until Phase 2 has run its
    ten sessions. Building on unaudited facts is the failure this prevents."""
    called = []
    monkeypatch.setattr(enrichment, "_enrich_one", lambda **kwargs: called.append(kwargs))

    _digest_sessions(tmp_path / "digests", 3)
    result = enrichment.run_journal_enrichment(
        session_date="2026-09-04", now=NOW, digest_root=tmp_path / "digests",
    )

    assert called == []
    assert result["status"] == "ok"
    assert enrichment.GATE_NOT_MET_PREFIX in result["reason"]
    assert "3 of 10" in result["reason"]


def test_the_gate_reads_phase_twos_own_counter(tmp_path):
    """One definition of "ten clean digest sessions", not a second copy."""
    from ai_jobs import digest

    _digest_sessions(tmp_path / "digests", 10)
    state = enrichment.gate_state(tmp_path / "digests")
    assert state["sessions_collected"] == 10
    assert state["window_met"] is True
    assert state == digest.digest_gate_state(tmp_path / "digests")


def test_an_absent_digest_store_is_unmet_never_a_free_pass(tmp_path):
    state = enrichment.gate_state(tmp_path / "nothing-here")
    assert state["window_met"] is False and state["sessions_collected"] == 0


# ---------------------------------------------------------------------------
# Phase 3 - the vocabulary and the advisory boundary
# ---------------------------------------------------------------------------


def test_the_tag_vocabulary_comes_from_the_setup_documents():
    vocabulary = enrichment.setup_vocabulary()
    assert vocabulary, "the setup documents must yield at least one family"
    assert all(isinstance(name, str) and name.strip() for name in vocabulary)


def test_a_tag_outside_the_vocabulary_is_dropped_and_counted():
    """The model proposes; the vocabulary decides. An invented family name would
    become a bucket nobody can compare against anything."""
    kept, dropped = enrichment.filter_tags(
        ["avwap_reclaim", "definitely_not_a_real_setup"], vocabulary=("avwap_reclaim",),
    )
    assert kept == ["avwap_reclaim"]
    assert dropped == ["definitely_not_a_real_setup"]


def test_enrichment_writes_only_the_advisory_table(tmp_path, monkeypatch):
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "j.sqlite3")
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO trades(trade_id, broker, account_number, account_label, symbol, "
            "security_type, currency, direction, status, opened_at, closed_at, trade_date, "
            "updated_at) VALUES('t1','QUESTRADE','1','TFSA','AAPL','STK','USD','LONG',"
            "'CLOSED','2026-09-04T09:31:00-07:00','2026-09-04T12:00:00-07:00','2026-09-04','x')"
        )
        conn.execute(
            "INSERT INTO trade_annotations(trade_id, setup_tags, notes, updated_at) "
            "VALUES('t1','my own tag','my own note','x')"
        )

    store.save_ai_enrichment(
        trade_id="t1", session_date="2026-09-04", summary="a summary",
        tags=["avwap_reclaim"], evidence=[{"source_id": "review.alert_review_events"}],
        model="gemma3:12b", now="2026-09-05T02:00:00+00:00",
    )

    rows = store.list_ai_enrichment("t1")
    assert len(rows) == 1 and rows[0]["summary"] == "a summary"
    with store.connection() as conn:
        annotation = dict(
            conn.execute("SELECT * FROM trade_annotations WHERE trade_id='t1'").fetchone()
        )
    assert annotation["setup_tags"] == "my own tag", "I7: trader tags are never overwritten"
    assert annotation["notes"] == "my own note"


def test_enrichment_is_append_only_and_supersedes(tmp_path):
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "j.sqlite3")
    for index in (1, 2):
        store.save_ai_enrichment(
            trade_id="t1", session_date="2026-09-04", summary=f"pass {index}",
            tags=[], evidence=[], model="gemma3:12b",
            now=f"2026-09-0{4 + index}T02:00:00+00:00",
        )
    rows = store.list_ai_enrichment("t1")
    assert [row["summary"] for row in rows] == ["pass 1", "pass 2"], (
        "a re-run adds a row; it never rewrites what an earlier night believed"
    )


# ---------------------------------------------------------------------------
# Phase 4 - drafts only, and the window accumulates
# ---------------------------------------------------------------------------


def test_the_draft_writer_targets_the_draft_file_and_never_the_live_one(tmp_path, monkeypatch):
    import project_paths
    from ai_jobs import policy_draft as module

    live = tmp_path / "review_policy.json"
    draft = tmp_path / "review_policy_draft.json"
    live.write_text('{"schema": "review_policy_v1", "rules": []}', encoding="utf-8")
    monkeypatch.setattr(project_paths, "REVIEW_POLICY_FILE", live)
    monkeypatch.setattr(project_paths, "REVIEW_POLICY_DRAFT_FILE", draft)
    import review_policy

    monkeypatch.setattr(review_policy, "REVIEW_POLICY_FILE", live)
    monkeypatch.setattr(review_policy, "REVIEW_POLICY_DRAFT_FILE", draft)
    before = live.read_bytes()

    result = module.run_review_policy_draft(
        session_date="2026-09-04", now=NOW, root=tmp_path / "store",
        state={"blind_spots": [], "leaks": []}, narrate=False,
    )

    assert result["status"] == "ok"
    assert draft.is_file()
    assert live.read_bytes() == before, "the live policy is never touched"


def test_the_draft_carries_no_suppression_field(tmp_path, monkeypatch):
    """The format deliberately has none. Do not add one."""
    import project_paths
    import review_policy

    draft = tmp_path / "review_policy_draft.json"
    monkeypatch.setattr(project_paths, "REVIEW_POLICY_DRAFT_FILE", draft)
    monkeypatch.setattr(review_policy, "REVIEW_POLICY_DRAFT_FILE", draft)

    policy_draft.run_review_policy_draft(
        session_date="2026-09-04", now=NOW, root=tmp_path / "store",
        state={
            "blind_spots": [{"dimension": "setup", "segment": "avwap", "reason": "unseen"}],
            "leaks": [],
        },
        narrate=False,
    )
    payload = json.loads(draft.read_text(encoding="utf-8"))

    def _keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key).lower()
                yield from _keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from _keys(item)

    # FIELD NAMES, not prose: the draft's own `notes` says in words that it
    # carries no suppression field, and a scan that could not tell the two apart
    # would forbid the file from stating its own contract.
    names = set(_keys(payload))
    for banned in ("suppress", "suppressed", "mute", "muted", "hide", "hidden", "silence"):
        assert banned not in names
    assert not any(banned in name for name in names
                   for banned in ("suppress", "mute", "hide", "silence"))


def test_a_draft_is_labelled_non_authoritative_until_the_window_passes(tmp_path, monkeypatch):
    import project_paths
    import review_policy

    draft = tmp_path / "review_policy_draft.json"
    monkeypatch.setattr(project_paths, "REVIEW_POLICY_DRAFT_FILE", draft)
    monkeypatch.setattr(review_policy, "REVIEW_POLICY_DRAFT_FILE", draft)

    result = policy_draft.run_review_policy_draft(
        session_date="2026-09-04", now=NOW, root=tmp_path / "store",
        state={"blind_spots": [], "leaks": []}, narrate=False,
    )
    payload = json.loads(draft.read_text(encoding="utf-8"))
    assert policy_draft.GATE_NOT_MET_PREFIX in payload["notes"]
    assert policy_draft.GATE_NOT_MET_PREFIX in result["reason"]


def test_the_side_by_side_window_accumulates_one_day_at_a_time(tmp_path, monkeypatch):
    """The gate IS the drafts: two weeks of them is what the trader compares.

    So this writer runs while its gate is unmet - refusing would make the gate
    unreachable - and every draft says it is not authoritative yet.
    """
    import project_paths
    import review_policy

    draft = tmp_path / "review_policy_draft.json"
    monkeypatch.setattr(project_paths, "REVIEW_POLICY_DRAFT_FILE", draft)
    monkeypatch.setattr(review_policy, "REVIEW_POLICY_DRAFT_FILE", draft)
    root = tmp_path / "store"

    assert policy_draft.side_by_side_days(root) == 0
    for day in range(1, 11):
        policy_draft.run_review_policy_draft(
            session_date=f"2026-09-{day:02d}", now=NOW, root=root,
            state={"blind_spots": [], "leaks": []}, narrate=False,
        )
    assert policy_draft.side_by_side_days(root) == 10
    assert policy_draft.gate_state(root)["window_met"] is True
    assert policy_draft.REQUIRED_SIDE_BY_SIDE_DAYS == 10


def test_every_draft_is_kept_so_the_comparison_has_something_to_compare(tmp_path, monkeypatch):
    import project_paths
    import review_policy

    draft = tmp_path / "review_policy_draft.json"
    monkeypatch.setattr(project_paths, "REVIEW_POLICY_DRAFT_FILE", draft)
    monkeypatch.setattr(review_policy, "REVIEW_POLICY_DRAFT_FILE", draft)
    root = tmp_path / "store"

    for day in ("2026-09-01", "2026-09-02"):
        policy_draft.run_review_policy_draft(
            session_date=day, now=NOW, root=root,
            state={"blind_spots": [], "leaks": []}, narrate=False,
        )
    kept = sorted(path.name for path in (root / "policy_drafts").rglob("*.json"))
    assert len(kept) == 2, f"one archived draft per session, got {kept}"


def test_priority_deltas_stay_inside_the_clamp(tmp_path, monkeypatch):
    import project_paths
    import review_policy

    draft = tmp_path / "review_policy_draft.json"
    monkeypatch.setattr(project_paths, "REVIEW_POLICY_DRAFT_FILE", draft)
    monkeypatch.setattr(review_policy, "REVIEW_POLICY_DRAFT_FILE", draft)

    policy_draft.run_review_policy_draft(
        session_date="2026-09-04", now=NOW, root=tmp_path / "store",
        state={
            "blind_spots": [{"dimension": "setup", "segment": "a", "reason": "r"}],
            "leaks": [{"dimension": "setup", "segment": "b", "reason": "r"}],
        },
        narrate=False,
    )
    payload = json.loads(draft.read_text(encoding="utf-8"))
    for rule in payload["rules"]:
        assert abs(int(rule["priority_delta"])) <= review_policy.MAX_PRIORITY_DELTA


# ---------------------------------------------------------------------------
# The boundary, walked rather than asserted in prose
# ---------------------------------------------------------------------------


def _path_tokens(module) -> list[str]:
    """String constants that could BE a path: a filename has no spaces.

    Prose is excluded on purpose - both modules NAME the files they must never
    write, in the sentences forbidding themselves from writing them.
    """
    import ast

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip()
        and not any(character.isspace() for character in node.value)
    ]


def test_neither_module_names_the_live_policy_file_as_a_path():
    for module in (enrichment, policy_draft):
        tokens = _path_tokens(module)
        assert not any(
            token == "review_policy.json" or token.endswith("/review_policy.json")
            for token in tokens
        ), f"{module.__name__} names the live policy file as a path"
        assert not any("REVIEW_POLICY_FILE" == token for token in tokens)


def test_the_draft_writer_only_ever_resolves_the_draft_constant():
    import ast

    tree = ast.parse(Path(policy_draft.__file__).read_text(encoding="utf-8"))
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert "REVIEW_POLICY_DRAFT_FILE" in names
    assert "REVIEW_POLICY_FILE" not in names
    assert "save_review_policy" in names or "save_review_policy" in Path(
        policy_draft.__file__
    ).read_text(encoding="utf-8")


def test_neither_module_writes_a_trader_owned_journal_field():
    """I7. `setup_tags`, `notes`, `planned_*` and `tax_status` are the trader's."""
    import ast

    trader_owned = ("setup_tags", "notes", "planned_entry", "planned_stop",
                    "planned_risk", "tax_status")
    for module in (enrichment, policy_draft):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        writes = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and ("UPDATE " in node.value.upper() or "INSERT " in node.value.upper())
        ]
        for statement in writes:
            for field in trader_owned:
                assert field not in statement, (
                    f"{module.__name__} writes the trader-owned field {field}"
                )
        # And it never calls the trader-facing save at all.
        called = {
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "save_annotation" not in called
        assert "save_risk_fields" not in called


def test_both_slots_are_appended_and_never_reorder_the_slate():
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    # P5 inserted the two cohort slots and P6 inserted
    # `preference_trade_outcomes` after them, all BEFORE `evidence_report` -
    # which is where they have to be, because the report READS what they
    # produce and a report ahead of its inputs would describe last night's
    # evidence. `preference_trade_outcomes` sits after the cohorts for the same
    # reason, one level down: it reads their outcome files.
    #
    # Each insertion moves the later slots' INDEX without reordering any
    # existing PAIR, so the assertion is pairwise. That is the real invariant
    # and it does not need editing the next time a slot is added - which is the
    # third time in three packets that it would have.
    #
    # DECISION 0018 (2026-09-04) moved the narration pair to after
    # `daily_digest`, so it is no longer in this list; the deterministic
    # stage's own pairwise order is unchanged, which is what this asserts.
    ORDERED = [
        "journal_import",
        "veto_cohort_grading", "like_cohort_grading",
        "pass_cohort_grading", "rejection_cohort_grading",
        "preference_trade_outcomes", "evidence_report",
    ]
    positions = [names.index(item) for item in ORDERED]
    assert positions == sorted(positions), "a later phase appends inside its stage"
    assert names.index("journal_enrichment") > names.index("daily_digest")
    assert names.index("review_policy_draft") > names.index("daily_digest")
    # Both gated slots still sit after the narration pair, in stage 3.
    assert names.index("journal_enrichment") > names.index("ticker_briefs")
    assert names.index("review_policy_draft") > names.index("ticker_briefs")


def test_neither_module_reaches_into_live_decision_code():
    import ast

    for module in (enrichment, policy_draft):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
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
            ), f"{module.__name__} reached into live decision code: {name}"
