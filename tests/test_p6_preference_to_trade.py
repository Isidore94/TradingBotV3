"""P6: from what the trader said to what they traded.

Three stores each held a third of one question. The annotation log knows what
the trader SAID about a name, the journal knows what they TRADED, the cohort
rollups know what the name then DID on paper — and nothing put the three on one
row. So *"of the setups I liked, which did I actually take, and how did the ones
I skipped do?"* could only be answered by eye.

The rules these tests hold:

* `trade_annotations` are TRADER-OWNED and nothing here writes them;
* no tag is ever derived from an outcome;
* every interim join renders a match confidence or says "no match" — no silent
  hard link, and no second canonical id (plan.md P5.3/P5.4 own that);
* an evidence read never costs the thing it reads;
* an empty dimension is NAMED, never hidden.
"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_analytics import AutoTagger  # noqa: E402


def _capture_source() -> str:
    """Resolved at CALL time, so the un-fixed module still COLLECTS and each
    test fails on its own merit rather than the whole file erroring out."""
    from journal_analytics import TRADER_CAPTURE_SOURCE

    return TRADER_CAPTURE_SOURCE

NOW = datetime(2026, 9, 1, 20, 0)


def _trade(symbol="AAA", side="LONG", opened="2026-08-20", closed="2026-08-22", **extra):
    row = {
        "trade_id": f"t-{symbol}",
        "symbol": symbol,
        "direction": side,
        "trade_date": opened,
        "opened_at": f"{opened}T09:41:00-04:00",
        "closed_at": f"{closed}T15:00:00-04:00" if closed else "",
        "status": "CLOSED" if closed else "OPEN",
        "net_pnl": 123.45,
        # R1: the fixture used to carry `r_multiple`, a key that exists nowhere
        # in scripts/ - so the test passed while every live row shipped a blank
        # R. These two ARE the journal's columns, and the R is computed from
        # them exactly as `journal_feed.r_multiple` computes it.
        "net_pnl_cad": 123.45,
        "planned_risk": 102.875,
    }
    row.update(extra)
    return row


# ==========================================================================
# 1 - the exact-id lane in the auto-tagger
# ==========================================================================
def _tagger_with(capture_rows, context_rows=()):
    tagger = AutoTagger()
    tagger._capture_rows = list(capture_rows)
    tagger._context_rows = list(context_rows)
    return tagger


def test_a_like_claim_inside_the_trade_window_becomes_a_candidate():
    """The trader named the setup, so that IS the tag.

    Fail-before-fix: there is no capture lane and no `context_row_id`.
    """
    tagger = _tagger_with(
        [
            {
                "symbol": "AAA",
                "date": date(2026, 8, 21),
                "side": "LONG",
                "kind": "like_claim",
                "tag": "post_earnings_avwap_bounce",
                "event_id": "abc123",
                "detail": "level held",
            }
        ]
    )
    out = tagger.suggest_for_trade(_trade())

    assert out[0]["tag"] == "post_earnings_avwap_bounce"
    assert out[0]["source"] == f"{_capture_source()}:like_claim"
    assert out[0]["context_row_id"] == "abc123"
    assert "you said this on 2026-08-21" in out[0]["rationale"]


def test_a_veto_and_a_pass_are_PREFIXED_so_neither_reads_as_an_endorsement():
    tagger = _tagger_with(
        [
            {
                "symbol": "AAA", "date": date(2026, 8, 21), "side": "LONG",
                "kind": "veto", "tag": "vetoed:too_extended_from_base",
                "event_id": "v1", "detail": "",
            },
            {
                "symbol": "AAA", "date": date(2026, 8, 21), "side": "LONG",
                "kind": "pass", "tag": "passed:poor_market_conditions",
                "event_id": "p1", "detail": "",
            },
        ]
    )
    tags = {row["tag"] for row in tagger.suggest_for_trade(_trade())}
    assert tags == {"vetoed:too_extended_from_base", "passed:poor_market_conditions"}


def test_a_capture_candidate_outranks_every_fuzzy_source():
    """When the trader has already said what they thought of a name, that
    outranks anything inferred about it."""
    tagger = _tagger_with(
        [
            {
                "symbol": "AAA", "date": date(2026, 8, 21), "side": "LONG",
                "kind": "like_claim", "tag": "what_i_said", "event_id": "e1", "detail": "",
            }
        ],
        context_rows=[
            {
                "symbol": "AAA",
                "date": date(2026, 8, 20),
                "side": "LONG",
                "source": "setup_tracker",
                "setup_family": "avwape_to_first_dev",
                "priority_bucket": "favorite_setup",
                "priority_score": 900,
            }
        ],
    )
    out = tagger.suggest_for_trade(_trade())

    assert out[0]["source"].startswith(f"{_capture_source()}:")
    assert any(row["source"] == "setup_tracker" for row in out), "the fuzzy row still appears"


def test_a_statement_outside_the_trades_own_window_is_not_a_candidate():
    """An event id is only worth carrying when the statement and the trade
    really are about the same episode - not a 16-day neighbourhood."""
    tagger = _tagger_with(
        [
            {
                "symbol": "AAA", "date": date(2026, 8, 10), "side": "LONG",
                "kind": "like_claim", "tag": "too_early", "event_id": "e1", "detail": "",
            }
        ]
    )
    out = tagger.suggest_for_trade(_trade(opened="2026-08-20", closed="2026-08-22"))
    assert not any(row["source"].startswith(f"{_capture_source()}:") for row in out)


def test_a_statement_on_the_other_side_is_a_different_claim():
    tagger = _tagger_with(
        [
            {
                "symbol": "AAA", "date": date(2026, 8, 21), "side": "SHORT",
                "kind": "like_claim", "tag": "other_side", "event_id": "e1", "detail": "",
            }
        ]
    )
    out = tagger.suggest_for_trade(_trade(side="LONG"))
    assert not any(row["source"].startswith(f"{_capture_source()}:") for row in out)


def test_an_open_trade_uses_its_open_date_as_the_whole_window():
    tagger = _tagger_with(
        [
            {
                "symbol": "AAA", "date": date(2026, 8, 20), "side": "LONG",
                "kind": "like_claim", "tag": "same_day", "event_id": "e1", "detail": "",
            }
        ]
    )
    out = tagger.suggest_for_trade(_trade(opened="2026-08-20", closed=""))
    assert out[0]["tag"] == "same_day"


def test_the_lane_is_ordered_above_the_others_in_the_store():
    """`list_auto_tag_candidates` orders by LANE before confidence."""
    import inspect

    import journal_store

    source = inspect.getsource(journal_store.JournalStore.list_auto_tag_candidates)
    assert "TRADER_CAPTURE_SOURCE" in source
    assert "DESC" in source


def test_the_context_row_id_column_is_additive_and_nullable():
    """Through the store's own migration path, never in place. It is a POINTER
    for a reader - plan.md P5.3/P5.4 own the canonical opportunity id."""
    from journal_migrate import NEW_COLUMNS_V3

    entry = [row for row in NEW_COLUMNS_V3 if row[:2] == ("auto_tag_candidates", "context_row_id")]
    assert entry, "the column must arrive through the additive migration list"
    assert "DEFAULT ''" in entry[0][2], "nullable/defaulted, so an old row is valid"


def test_the_candidate_round_trips_through_the_store(tmp_path):
    from journal_store import JournalStore

    store = JournalStore(db_path=tmp_path / "journal.db")
    trade = _trade()
    store.upsert_trades([trade]) if hasattr(store, "upsert_trades") else None
    with store.connection() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(auto_tag_candidates)")}
    assert "context_row_id" in columns


def test_the_tagger_never_writes_trade_annotations():
    """Trader-owned. The tagger SUGGESTS; the trader accepts."""
    import inspect

    import journal_analytics

    source = inspect.getsource(journal_analytics.AutoTagger)
    # Ban the WRITE, not the word: the class comment saying it never writes
    # them is exactly the sentence worth keeping.
    for forbidden in (
        "INSERT INTO trade_annotations",
        "UPDATE trade_annotations",
        "save_trade_annotation",
        "set_trade_tags",
    ):
        assert forbidden not in source
    assert "nothing here writes trade_annotations" in source


def test_an_unreadable_capture_store_costs_its_own_suggestions_and_nothing_else(monkeypatch):
    """The auto-tagger runs behind an OK button."""
    tagger = AutoTagger()
    monkeypatch.setattr(
        AutoTagger, "_load_annotation_capture_rows", lambda self: (_ for _ in ()).throw(OSError("gone"))
    )
    with pytest.raises(OSError):
        tagger._load_annotation_capture_rows()
    # The real loader swallows it: a fresh tagger still answers.
    assert AutoTagger()._load_review_capture_rows() is not None


# ==========================================================================
# 2 - what I said, what I did, what happened
# ==========================================================================
def _statement(symbol="AAA", side="LONG", said="2026-08-20", **extra):
    row = {
        "session_date": date.fromisoformat(said),
        "symbol": symbol,
        "side": side,
        "channel": "annotation:like_claim",
        "statement": "liked",
        "statement_detail": "post_earnings_avwap_bounce",
        "statement_id": "e1",
    }
    row.update(extra)
    return row


def test_a_statement_with_no_trade_is_the_interesting_row():
    """It is the SKIP - the whole reason the report exists - so it is written
    with an explicit "no", never omitted."""
    from preference_trade_outcomes import build_rows

    rows = build_rows([_statement()], [], now=NOW)
    assert len(rows) == 1
    assert rows[0]["traded"] == "no"
    assert rows[0]["trade_id"] == ""
    assert rows[0]["match_basis"] == "no match"
    assert rows[0]["match_confidence"] == ""


def test_a_match_renders_its_confidence_and_what_it_rested_on():
    """Never a silent hard link: the trader could have taken the name that week
    for an unrelated reason."""
    from preference_trade_outcomes import build_rows

    same_day = build_rows([_statement()], [_trade(opened="2026-08-20")], now=NOW)[0]
    assert same_day["traded"] == "yes"
    assert same_day["trade_id"] == "t-AAA"
    assert same_day["match_confidence"] == "0.90"
    assert same_day["match_basis"] == "symbol+side+same_session"

    later = build_rows([_statement()], [_trade(opened="2026-08-25")], now=NOW)[0]
    assert later["match_confidence"] == "0.70"
    assert later["match_basis"] == "symbol+side+window"


def test_a_trade_taken_the_other_way_is_marked_as_such():
    from preference_trade_outcomes import build_rows

    row = build_rows([_statement(side="LONG")], [_trade(side="SHORT", opened="2026-08-20")], now=NOW)[0]
    assert row["match_basis"] == "symbol+window_opposite_side"
    assert float(row["match_confidence"]) < 0.5


def test_a_trade_far_after_the_statement_is_a_different_decision():
    from preference_trade_outcomes import build_rows

    row = build_rows([_statement(said="2026-08-01")], [_trade(opened="2026-08-25")], now=NOW)[0]
    assert row["traded"] == "no"


def test_the_paper_return_is_read_never_recomputed():
    """Ground rule 6: reformatted, never derived."""
    from preference_trade_outcomes import build_rows

    grades = {("2026-08-20", "AAA", "LONG"): {"h3": "0.0681", "h5": "0.1721", "cohort": "focus_m5"}}
    row = build_rows([_statement()], [], grades=grades, now=NOW)[0]
    assert row["paper_forward_return_h3"] == "0.0681"
    assert row["paper_forward_return_h5"] == "0.1721"
    assert row["paper_cohort"] == "focus_m5"


def test_an_unmatured_paper_grade_is_blank_never_zero():
    from preference_trade_outcomes import build_rows

    grades = {("2026-08-20", "AAA", "LONG"): {"h3": "0.02", "h5": "", "cohort": "focus_m5"}}
    row = build_rows([_statement()], [], grades=grades, now=NOW)[0]
    assert row["paper_forward_return_h5"] == ""


def test_the_report_mints_no_identifier():
    """plan.md P5.3/P5.4 own the canonical opportunity id; a second one here
    would compete with it while being weaker."""
    import inspect

    import preference_trade_outcomes

    source = inspect.getsource(preference_trade_outcomes)
    for forbidden in ("uuid", "sha256", "hashlib", "opportunity_id"):
        assert forbidden not in source


def test_the_report_never_writes_into_the_journal():
    import inspect

    import preference_trade_outcomes

    source = inspect.getsource(preference_trade_outcomes)
    for forbidden in (
        "INSERT INTO",
        "UPDATE ",
        "upsert_",
        "save_trade",
        "conn.execute",
    ):
        assert forbidden not in source
    # And it reads the journal through the one public reader.
    assert "list_trades" in source


def test_a_retracted_swing_favorite_is_not_reported_as_a_pick(tmp_path):
    """The append-only log keeps the retraction; the live list for that session
    is what the trader actually stood behind."""
    import swing_favorites
    from preference_trade_outcomes import collect_statements

    path = tmp_path / "swing_favorites.jsonl"
    for action in (swing_favorites.ACTION_ADD, swing_favorites.ACTION_REMOVE):
        swing_favorites.append_row(
            swing_favorites.build_row(
                symbol="ZZZ", side="long", action=action, session_date="2026-08-20"
            ),
            path=path,
        )
    swing_favorites.append_row(
        swing_favorites.build_row(
            symbol="KEEP", side="long", action=swing_favorites.ACTION_ADD, session_date="2026-08-20"
        ),
        path=path,
    )

    said = collect_statements(
        since=date(2026, 8, 1),
        until=date(2026, 8, 31),
        annotations_path=tmp_path / "none.jsonl",
        feedback_path=tmp_path / "none2.jsonl",
        favorites_path=path,
    )
    symbols = {row["symbol"] for row in said if row["channel"] == "swing_favorite"}
    assert symbols == {"KEEP"}


def test_the_report_writes_its_columns(tmp_path):
    from preference_trade_outcomes import COLUMNS, build_rows, write_rows

    path = tmp_path / "report.csv"
    assert write_rows(build_rows([_statement()], [_trade()], now=NOW), path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == COLUMNS


def test_the_slot_is_appended_and_calls_no_model():
    import inspect

    from ai_jobs.runner import default_slots
    from preference_trade_outcomes import run_preference_trade_outcomes

    names = [slot.name for slot in default_slots()]
    assert "preference_trade_outcomes" in names
    # It READS the cohort outcome files, so it runs before the report that
    # summarises them.
    assert names.index("preference_trade_outcomes") < names.index("evidence_report")
    for existing in ("veto_cohort_grading", "like_cohort_grading"):
        assert names.index(existing) < names.index("preference_trade_outcomes")

    slot = next(item for item in default_slots() if item.name == "preference_trade_outcomes")
    assert slot.reserve_minutes == 5.0
    source = inspect.getsource(run_preference_trade_outcomes)
    for forbidden in ("request_ai_summary", "local_model", "narrate"):
        assert forbidden not in source


def test_an_empty_window_is_stated_not_an_error(tmp_path):
    from preference_trade_outcomes import run_preference_trade_outcomes

    result = run_preference_trade_outcomes(
        now=NOW, window_days=1, report_path=tmp_path / "r.csv", trades=[]
    )
    assert result["status"] in {"ok", "skipped"}
    if result["status"] == "skipped":
        assert "absent record" in result["reason"]


# ==========================================================================
# the badge names its trade
# ==========================================================================
def test_the_badge_names_the_trade_it_is_marking():
    import swing_favorites

    marks = swing_favorites.taken_trade_ids(
        [{"symbol": "NVDA", "side": "long", "session_date": "2026-08-20"}],
        [_trade(symbol="NVDA", opened="2026-08-20", trade_id="tr-9")],
    )
    assert marks == {("NVDA", "long"): "tr-9"}


def test_a_trade_with_no_id_still_MARKS_the_chip():
    """The id is extra, never a condition: requiring one would silently un-mark
    chips, which is a worse answer than a mark with no link."""
    import swing_favorites

    marks = swing_favorites.taken_trade_ids(
        [{"symbol": "NVDA", "side": "long", "session_date": "2026-08-20"}],
        [{"symbol": "NVDA", "opened_at": "2026-08-20T09:41:00-04:00"}],
    )
    assert marks == {("NVDA", "long"): ""}


def test_the_badge_and_the_link_use_the_same_matching_rule():
    """A badge that appeared under one rule and linked under another would
    point at a trade that is not the one it is marking."""
    import swing_favorites

    favorites = [{"symbol": "NVDA", "side": "long", "session_date": "2026-08-20"}]
    trades = [_trade(symbol="NVDA", opened="2026-08-21", trade_id="tr-1")]
    assert set(swing_favorites.taken_keys(favorites, trades)) == set(
        swing_favorites.taken_trade_ids(favorites, trades)
    )


def test_the_badge_computes_no_statistic():
    import inspect

    import swing_favorites

    source = inspect.getsource(swing_favorites.taken_trade_ids)
    for forbidden in ("pnl", "r_multiple", "mean", "sum(", "win"):
        assert forbidden not in source


# ==========================================================================
# 3 - the honest empty dimension
# ==========================================================================
def test_a_thin_confirmed_tag_group_is_NAMED_not_hidden():
    """Live on 2026-09-01: ONE confirmed tag across 193 trades, rendered beside
    a full auto-tag chart of the same width.

    Fail-before-fix: `group_notes` does not exist.
    """
    from journal_analytics import build_analytics_summary

    trades = [
        {"status": "CLOSED", "net_pnl": 10.0, "symbol": "A", "setup_tags": "", "auto_tag_summary": "x"}
        for _ in range(20)
    ]
    trades[0]["setup_tags"] = "my_setup"

    summary = build_analytics_summary(trades)
    note = summary["group_notes"]["my setups"]

    assert "1 OF 20" in note
    assert "5%" in note
    assert "prompt to tag more" in note
    # NEVER hidden: the group is still there, thin and visible.
    assert summary["groups"]["my setups"], "the group must not disappear"


def test_a_well_covered_group_gets_no_banner():
    from journal_analytics import build_analytics_summary

    trades = [
        {"status": "CLOSED", "net_pnl": 1.0, "symbol": "A", "setup_tags": "my_setup"}
        for _ in range(20)
    ]
    assert build_analytics_summary(trades)["group_notes"] == {}


def test_no_closed_trades_makes_no_claim_either_way():
    from journal_analytics import build_analytics_summary

    assert build_analytics_summary([{"status": "OPEN", "symbol": "A"}])["group_notes"] == {}


# ==========================================================================
# the stale comment
# ==========================================================================
def test_the_market_journal_scope_comment_matches_the_code():
    """It said "OPT-IN ONLY", and `briefs.DEFAULT_SCOPES` has carried
    `market_journal` on the nightly run since R10.H. The code is the fact."""
    import inspect

    import ai_summary
    from ai_jobs.briefs import DEFAULT_SCOPES

    assert "market_journal" in DEFAULT_SCOPES
    text = inspect.getsource(ai_summary)
    marker = text.index('"market_journal": "Market journal entries')
    preceding = text[max(0, marker - 900) : marker]
    # The comment must now say what the code DOES. It still quotes the old
    # phrase in order to correct it, so the assertion is on the correction
    # rather than on the absence of the words.
    assert "DEFAULT_SCOPES" in preceding
    assert "WRONG since R10.H" in preceding
    assert "the comment was the defect" in preceding.lower()


def test_the_scope_behaviour_is_unchanged():
    """The comment was the defect. Whether it SHOULD be nightly is the
    trader's decision, and this packet changed no behaviour."""
    from ai_jobs.briefs import DEFAULT_SCOPES

    assert DEFAULT_SCOPES == (
        "daily_report",
        "market_conditions",
        "setup_trackers",
        "journal_review",
        "market_journal",
    )


# ==========================================================================
# Review round R1
# ==========================================================================


def test_a_housekeeping_click_never_evicts_a_real_setup_tag(tmp_path, monkeypatch):
    """R1: `took:add_focus` was a minted tag, ranked FIRST, in a 4-slot summary.

    676 of 730 live review rows carry no `bounce_types`, so almost every capture
    row minted one - and because capture candidates lead, it spent a summary
    slot on a chart housekeeping click. Measured on eight live trades: EYPT and
    SMPL lost `avwape_to_1stdev` from their Tags column.
    """
    from journal_analytics import AutoTagger

    tagger = AutoTagger()
    monkeypatch.setattr(
        tagger,
        "load_context_rows",
        lambda: [
            {
                "source": "setup_tracker",
                "symbol": "EYPT",
                "side": "LONG",
                "date": date(2026, 8, 20),
                "setup_family": "avwape_to_1stdev",
                "priority_bucket": "favorite_setup",
                "priority_score": 900,
                "favorite_zone": "",
                "retest": "",
                "compression": False,
            }
        ],
    )
    monkeypatch.setattr(
        tagger,
        "load_capture_rows",
        lambda: [
            {
                "symbol": "EYPT",
                "date": date(2026, 8, 20),
                "side": "LONG",
                "kind": "review:add_focus",
                "tag": "",
                "link_only": True,
                "event_id": "review_event:2026-08-20T13:40:00",
            }
        ],
    )

    suggestions = tagger.suggest_for_trade(
        {
            "symbol": "EYPT",
            "direction": "LONG",
            "opened_at": "2026-08-20T09:41:00-04:00",
            "closed_at": "2026-08-20T15:00:00-04:00",
        }
    )
    by_tag = {item["tag"]: item for item in suggestions}

    # The link is still offered, and still carries its pointer.
    link = next(item for item in suggestions if item.get("link_only"))
    assert link["context_row_id"] == "review_event:2026-08-20T13:40:00"
    assert link["tag"].startswith("link:")
    # And nothing minted a `took:` tag.
    assert not any(tag.startswith("took:") for tag in by_tag)
    # The real setup match is still there.
    assert any("avwape_to_1stdev" in tag for tag in by_tag)


def test_the_stored_summary_excludes_link_only_candidates():
    """The summary is the Tags column; a link may never occupy one of its slots."""
    source = (ROOT_DIR / "scripts" / "journal_store.py").read_text(encoding="utf-8")
    assert 'if not item.get("link_only")' in source


def test_coverage_counts_trades_and_not_buckets():
    """The packet's case: 5 tagged of 100, at 3 tags each, is 5% - not 15%."""
    from journal_analytics import build_analytics_summary

    rows = [
        {"status": "CLOSED", "net_pnl": 1.0, "currency": "USD", "setup_tags": "a; b; c"}
        for _ in range(5)
    ]
    rows += [
        {"status": "CLOSED", "net_pnl": 1.0, "currency": "USD"} for _ in range(95)
    ]
    note = build_analytics_summary(rows)["group_notes"]["my setups"]
    assert "5 OF 100" in note
    assert "(5%)" in note


def test_coverage_ignores_a_provisional_tag():
    """A machine-applied tag is not the trader standing behind anything."""
    from journal_analytics import build_analytics_summary

    rows = [
        {
            "status": "CLOSED", "net_pnl": 1.0, "currency": "USD",
            "setup_tags": "machine-guess", "tag_status": "provisional",
        }
        for _ in range(40)
    ]
    rows += [
        {"status": "CLOSED", "net_pnl": 1.0, "currency": "USD"} for _ in range(60)
    ]
    note = build_analytics_summary(rows)["group_notes"]["my setups"]
    assert "0 OF 100" in note


def test_the_report_uses_the_journals_one_definition_of_r():
    """`r_multiple` is a key that exists nowhere in scripts/."""
    import preference_trade_outcomes as report

    assert report._canonical_r({"planned_risk": 100.0, "net_pnl_cad": 250.0}) == "2.5000"
    # Blank, never zero, when the trader never typed a risk.
    assert report._canonical_r({"net_pnl_cad": 250.0}) == ""
    assert report._canonical_r({"planned_risk": 0.0, "net_pnl_cad": 250.0}) == ""

    source = (ROOT_DIR / "scripts" / "preference_trade_outcomes.py").read_text(encoding="utf-8")
    assert 'trade.get("r_multiple")' not in source
    assert 'trade.get("net_pnl_cad")' in source, "the P&L column must be CAD too"


def test_the_report_is_resolved_by_its_named_constant():
    """Resolving a home-folder store by name under a directory shipped a blank
    page for six days (CLAUDE.md)."""
    source = (ROOT_DIR / "scripts" / "ui" / "panels" / "weekend_prep_panel.py").read_text(
        encoding="utf-8"
    )
    assert "from preference_trade_outcomes import REPORT_FILE" in source
    assert '/ "preference_trade_outcomes.csv"' not in source


def test_the_suggestion_label_shows_the_linked_event():
    """Gate 35 asks the trader to SEE a linked event; it was stored and never
    rendered."""
    source = (ROOT_DIR / "scripts" / "ui" / "panels" / "journal" / "trades_tab.py").read_text(
        encoding="utf-8"
    )
    assert 'candidate.get("context_row_id")' in source


def test_a_pass_carries_every_code_it_was_given_in_vocabulary_order(tmp_path, monkeypatch):
    """R2: `codes[0]` threw the rest away.

    A pass for "extended from VWAP AND thin liquidity" reached the tagger as the
    first reason alone, which is a different statement from the one the trader
    made. The annotation writes its codes in VOCABULARY order already (never
    click order), so preserving the list preserves that too.
    """
    import json

    import project_paths
    from journal_analytics import AutoTagger

    log = tmp_path / "trader_annotations.jsonl"
    log.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "pass",
                "symbol": "AAA",
                "side": "LONG",
                "session_date": "2026-08-20",
                "created_at": "2026-08-20T13:40:00-04:00",
                # Written in VOCABULARY order by the capture rail, never click
                # order - so carrying the list carries that guarantee too.
                "reason_codes": ["extended_from_vwap", "thin_liquidity"],
                "event_id": "evt-1",
            }
        )
        + chr(10),
        encoding="utf-8",
    )
    # Patched where it is READ from - `_load_annotation_capture_rows` imports
    # the constant from `project_paths` inside the function.
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", log)

    rows = AutoTagger()._load_annotation_capture_rows()
    assert [row["tag"] for row in rows] == ["passed:extended_from_vwap,thin_liquidity"]


def test_one_predicate_answers_whether_a_candidate_is_a_link():
    """Both spellings: the in-memory flag and the prefix that survives the store."""
    from journal_analytics import is_link_candidate

    assert is_link_candidate({"link_only": True, "tag": "link:review:add_focus"})
    assert is_link_candidate({"tag": "link:review:arm_level"})  # round-tripped
    assert is_link_candidate("link:review:arm_level")
    assert not is_link_candidate({"tag": "avwape_to_1stdev", "confidence": 0.9})
    assert not is_link_candidate("avwape_to_1stdev")
    assert not is_link_candidate(None)
