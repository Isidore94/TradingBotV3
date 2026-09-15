"""WS-5B: the "what I said against what I did" report is SYMMETRIC.

WISHLIST item 5, block B. Today `preference_trade_outcomes` asks the stores for
likes, favorites and passes and never asks for a rejection at all
(`collect_statements` -> `load_annotations(..., event_types=(EVENT_LIKE_CLAIM,
EVENT_PASS))` and `_feedback_statements`' `verdict == "like"` filter), so the
report can say *"you liked it and did not take it"* and can never say *"you
vetoed it and took it anyway"* - which is the half of the record that costs
money.

The rules these tests hold, and the reason each one is here:

* **Every explicit verdict is a statement, and `unfavorite` is not a verdict**
  (CLAUDE.md P5: "`unfavorite` is never graded"). Taking a name out of Focus is
  housekeeping; turning it into a negative judgement would teach the loop a
  lesson the trader never gave it.
* **No two verdicts are combined** (P5). A veto, a pass, a dislike, a not-today
  and a click-away are five verdicts, they ride in five channels, and the
  endorse family and the reject family are never pooled into one number.
* **A like keeps the mode it was made in** (P9): `quick` / `claimed`, and a row
  written before P9 has no `like_mode` at all - absence reads `claimed`.
* **Money is counted once per trade** (ST5.2 / `trade_level_summary`), even when
  three statements matched it.
* **A miss has three honest shapes** and they are not the same answer: the
  10-SESSION window is still open, the window closed with no trade, or the
  journal could not be read. `scripts/ai_summary.preference_to_trade_section`
  already DERIVES those three from `match_basis` + `session_date`; the new
  `match_state` column must AGREE with it, never contradict it.
* **The first 19 columns are a published contract.** WS-AI1's
  `preference_to_trade` section reads `match_basis`, `trade_id` and
  `session_date` out of this CSV, so the golden below pins every one of those
  columns byte-identical to what the CURRENT code writes. Only `schema` may
  move (the packet bumps the version) and the new columns go at the END.

Names this packet fixes, so the builder and the consumers agree:

* channels ``annotation:veto``, ``annotation:pass`` (already),
  ``pick_feedback:dislike``, ``pick_feedback:not_today``,
  ``review_event:m5_click_away``;
* `collect_statements(events_path=...)` for the review-event store, matching the
  three path kwargs it already takes;
* new columns ``like_mode``, ``verdict_family``, ``match_state`` at the end;
* `trade_level_summary(...)["n_statements_by_family"]`;
* the Weekend Prep widgets ``preference_rejection_table`` /
  ``preference_rejection_note``, beside the existing ``preference_table``
  (``rejection_table`` is already taken by the P5 cohort).
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


#: The session every statement in these fixtures is ABOUT. A real trading day,
#: and `statement_window_end` closes its 10-session window on 2026-09-03.
SAID = "2026-08-20"
#: A session whose 10-session window is still OPEN at :data:`NOW` (it closes
#: 2026-09-22). Counted on the exchange calendar, never in calendar days.
SAID_RECENT = "2026-09-08"
NOW = datetime(2026, 9, 10, 20, 0)
CREATED = "2026-08-20T10:15:00-04:00"


# ---------------------------------------------------------------------------
# fixtures: the stores, written the way the desk writes them
# ---------------------------------------------------------------------------
def _trade(symbol="AAA", side="LONG", opened=SAID, trade_id=""):
    """One journal trade, in `JournalStore.list_trades`' own column names."""
    return {
        "trade_id": trade_id or f"t-{symbol}",
        "symbol": symbol,
        "direction": side,
        "trade_date": opened,
        "opened_at": f"{opened}T09:41:00-04:00",
        "closed_at": f"{opened}T15:00:00-04:00",
        "status": "CLOSED",
        "net_pnl": 123.45,
        "net_pnl_cad": 123.45,
        # 123.45 / 102.875 is exactly 1.2, so the golden's R is not a rounding
        # artefact of the fixture.
        "planned_risk": 102.875,
    }


def _append_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _feedback_row(symbol: str, verdict: str, *, side="LONG", origin="d1",
                  session=SAID, reason="", category="swing") -> dict:
    """A `pick_feedback.jsonl` row with an EXPLICIT `trade_date`.

    `record_pick_feedback` stamps `trade_date` from the clock, so a test that
    used it could only ever write today's session.
    """
    return {
        "ts": f"{session}T16:05:00-04:00",
        "trade_date": session,
        "symbol": symbol,
        "side": side,
        "verdict": verdict,
        "category": category,
        "origin": origin,
        "reason": reason,
        "context": "",
    }


def _click_away_row(symbol: str, *, side="LONG", session=SAID) -> dict:
    """The M5 click-away, as `alert_center_panel` writes it.

    A click away from an M5 alert IS a pass (trader, 2026-09-01); it has no verb
    of its own, only an `action: "skip"` whose detail reason is
    `pick_feedback.M5_CLICK_AWAY_REASON`. Written by hand for the same reason as
    the feedback row: `record_review_event` stamps its own `trade_date`.
    """
    from pick_feedback import M5_CLICK_AWAY_REASON

    return {
        "schema": "alert_review_event_v1",
        "review_record_id": f"rr-{symbol}",
        "ts": f"{session}T11:02:00-04:00",
        "trade_date": session,
        "installation_id": "test-installation",
        "machine": "test",
        "pid": 0,
        "action": "skip",
        "symbol": symbol,
        "side": side,
        "detail": {"reason": M5_CLICK_AWAY_REASON},
    }


def _write_all_channels(tmp_path: Path) -> dict[str, Path]:
    """Every store, holding one row of every explicit verdict the desk can make.

    AAA claimed like / BBB quick like / CCC pre-P9 like with NO `like_mode` key /
    DDD coded veto / EEE day-trade pass / FFF pick_feedback like /
    GGG dislike / HHH not-today / III unfavorite (never a verdict) /
    JJJ swing favorite / KKK favorite added then retracted /
    LLL M5 click-away.
    """
    import swing_favorites
    from ui.annotations.store import (
        EVENT_LIKE_CLAIM,
        EVENT_PASS,
        EVENT_VETO,
        record_annotation,
    )

    annotations = tmp_path / "trader_annotations.jsonl"
    feedback = tmp_path / "pick_feedback.jsonl"
    favorites = tmp_path / "swing_favorites.jsonl"
    events = tmp_path / "alert_review_events.jsonl"

    record_annotation(
        EVENT_LIKE_CLAIM,
        path=annotations,
        symbol="AAA",
        side="LONG",
        session_date=SAID,
        created_at=datetime.fromisoformat(CREATED),
        like_mode="claimed",
        claimed_setup_id="avwap_reclaim",
        event_id="e-claimed",
    )
    record_annotation(
        EVENT_LIKE_CLAIM,
        path=annotations,
        symbol="BBB",
        side="SHORT",
        session_date=SAID,
        created_at=datetime.fromisoformat(CREATED),
        like_mode="quick",
        event_id="e-quick",
    )
    # A row written BEFORE P9: the key is not empty, it is ABSENT.
    _append_json(
        annotations,
        {
            "schema_version": 1,
            "event_id": "e-legacy",
            "event_type": "like_claim",
            "symbol": "CCC",
            "side": "LONG",
            "session_date": SAID,
            "created_at": CREATED,
            "source": "chart_review",
            "claimed_setup_id": "avwap_reclaim",
        },
    )
    record_annotation(
        EVENT_VETO,
        path=annotations,
        symbol="DDD",
        side="LONG",
        session_date=SAID,
        created_at=datetime.fromisoformat(CREATED),
        reason_code="incoming_trendline",
        event_id="e-veto",
    )
    record_annotation(
        EVENT_PASS,
        path=annotations,
        symbol="EEE",
        side="LONG",
        session_date=SAID,
        created_at=datetime.fromisoformat(CREATED),
        reason_codes=["low_rvol"],
        event_id="e-pass",
    )

    for row in (
        _feedback_row("FFF", "like"),
        _feedback_row("GGG", "dislike", reason="chased it last time"),
        _feedback_row("HHH", "not_today", category="m5"),
        _feedback_row("III", "unfavorite"),
    ):
        _append_json(feedback, row)

    swing_favorites.append_row(
        swing_favorites.build_row(
            symbol="JJJ", side="long", action=swing_favorites.ACTION_ADD,
            session_date=SAID, origin="vetted",
        ),
        path=favorites,
    )
    for action in (swing_favorites.ACTION_ADD, swing_favorites.ACTION_REMOVE):
        swing_favorites.append_row(
            swing_favorites.build_row(
                symbol="KKK", side="long", action=action, session_date=SAID,
                origin="vetted",
            ),
            path=favorites,
        )

    _append_json(events, _click_away_row("LLL"))

    return {
        "annotations_path": annotations,
        "feedback_path": feedback,
        "favorites_path": favorites,
        "events_path": events,
    }


def _collect(
    paths: dict[str, Path],
    *,
    with_events: bool = True,
    since=date(2026, 8, 1),
    until=date(2026, 9, 30),
):
    """`collect_statements` over the fixture stores.

    ``with_events`` is False for the tests that are not ABOUT the click-away, so
    the missing `events_path` kwarg cannot mask the behaviour they name.
    """
    from preference_trade_outcomes import collect_statements

    kwargs = dict(paths)
    if not with_events:
        kwargs.pop("events_path", None)
    return collect_statements(since=since, until=until, **kwargs)


# ===========================================================================
# 1 - every explicit verdict is a statement
# ===========================================================================
def test_every_explicit_verdict_channel_becomes_a_statement_and_unfavorite_does_not(tmp_path):
    """The reject half of the record exists.

    Fail-before-fix: `collect_statements` asks the annotation log only for
    `like_claim` and `pass`, keeps only `verdict == "like"` from pick_feedback,
    and has never read the review-event store at all - so the veto, the dislike,
    the not-today and the click-away are absent by omission, not by decision.

    `unfavorite` stays absent BY decision: it is housekeeping, not a judgement.
    """
    statements = _collect(_write_all_channels(tmp_path))

    by_symbol = {row["symbol"]: row for row in statements}
    assert by_symbol.keys() == {
        "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "JJJ", "LLL"
    }
    assert {row["channel"] for row in statements} == {
        "annotation:like_claim",
        "annotation:veto",
        "annotation:pass",
        "pick_feedback:like",
        "pick_feedback:dislike",
        "pick_feedback:not_today",
        "swing_favorite",
        "review_event:m5_click_away",
    }
    # No two verdicts are combined: each reject channel keeps its own name.
    assert by_symbol["DDD"]["channel"] == "annotation:veto"
    assert by_symbol["GGG"]["channel"] == "pick_feedback:dislike"
    assert by_symbol["HHH"]["channel"] == "pick_feedback:not_today"
    assert by_symbol["LLL"]["channel"] == "review_event:m5_click_away"
    # Families, so the report can never pool an endorsement with a refusal.
    families = {row["symbol"]: row["verdict_family"] for row in statements}
    assert families == {
        "AAA": "endorse", "BBB": "endorse", "CCC": "endorse",
        "FFF": "endorse", "JJJ": "endorse",
        "DDD": "reject", "EEE": "reject", "GGG": "reject",
        "HHH": "reject", "LLL": "reject",
    }


def test_a_coded_veto_carries_its_reason_and_the_version_that_coded_it(tmp_path):
    """Cohort identity on write is `(vocab_version, reason_code)`, so a reject
    statement that dropped the version could not be joined back to its cohort.

    The version is READ from the vocabulary, never typed here (CLAUDE.md: never
    assert a literal `vocab_version` in a test).

    Fail-before-fix: there is no veto statement to carry anything.
    """
    from ui.annotations.vocabulary import load_veto_vocabulary

    statements = _collect(_write_all_channels(tmp_path), with_events=False)
    veto = next(row for row in statements if row["symbol"] == "DDD")

    version = load_veto_vocabulary().vocab_version
    assert "incoming_trendline" in veto["statement_detail"]
    assert str(version) in veto["statement_detail"]
    assert veto["statement_id"] == "e-veto"
    assert veto["statement"] and veto["statement"] != "liked"


def test_a_like_carries_the_mode_it_was_made_in_and_a_legacy_like_reads_claimed(tmp_path):
    """P9: **a LIKE has two modes and only one names a setup.** A quick like
    graded beside a claimed one would credit the claim vocabulary for a keypress
    that never named a setup.

    A row written before P9 has the key ABSENT, and absence reads `claimed`
    (`ui.annotations.store.like_mode_of`).

    Fail-before-fix: `like_mode` is never read and there is no column for it.
    """
    from preference_trade_outcomes import build_rows

    statements = _collect(_write_all_channels(tmp_path), with_events=False)
    rows = {row["symbol"]: row for row in build_rows(statements, [], now=NOW)}

    assert rows["AAA"]["like_mode"] == "claimed"
    assert rows["BBB"]["like_mode"] == "quick"
    assert rows["CCC"]["like_mode"] == "claimed"
    # A non-like says nothing about a mode it never had.
    assert rows["DDD"]["like_mode"] == ""
    assert rows["JJJ"]["like_mode"] == ""


def test_the_schema_bumps_and_the_first_nineteen_columns_keep_their_names_and_order():
    """The new columns go at the END and the published 19 do not move.

    WS-AI1's `preference_to_trade` section reads this file positionally by name
    out of a `csv.DictReader`; a column inserted in the middle is invisible to
    it, but a RENAMED or dropped one is not.

    Fail-before-fix: `COLUMNS` has 19 entries and `SCHEMA` is still v1.
    """
    from preference_trade_outcomes import COLUMNS, SCHEMA

    assert COLUMNS[:19] == [
        "schema",
        "generated_at",
        "session_date",
        "symbol",
        "side",
        "channel",
        "statement",
        "statement_detail",
        "statement_id",
        "traded",
        "trade_id",
        "trade_opened_at",
        "match_confidence",
        "match_basis",
        "journal_r",
        "journal_net_pnl",
        "paper_forward_return_h3",
        "paper_forward_return_h5",
        "paper_cohort",
    ]
    assert COLUMNS[19:] == ["like_mode", "verdict_family", "match_state"]
    assert SCHEMA.startswith("preference_trade_outcomes_v")
    assert SCHEMA != "preference_trade_outcomes_v1"


# ===========================================================================
# the golden - what the CURRENT code writes for a like row
# ===========================================================================
#: Pinned from the code as it stands on `claude/wishlist-sweep-2026-09-12`, per
#: like channel. `schema` is excluded because this packet bumps it; every other
#: published column is byte-identical, because WS-AI1 reads them.
GOLDEN_LIKE_ROWS = {
    ("annotation:like_claim", "AAA"): {
        "generated_at": "2026-09-10T20:00:00",
        "session_date": "2026-08-20",
        "symbol": "AAA",
        "side": "LONG",
        "channel": "annotation:like_claim",
        "statement": "liked",
        "statement_detail": "avwap_reclaim",
        "statement_id": "e-claimed",
        "traded": "yes",
        "trade_id": "t-AAA",
        "trade_opened_at": "2026-08-20T09:41:00-04:00",
        "match_confidence": "0.90",
        "match_basis": "symbol+side+same_session",
        "journal_r": "1.2000",
        "journal_net_pnl": "123.4500",
        "paper_forward_return_h3": "0.0681",
        "paper_forward_return_h5": "0.1721",
        "paper_cohort": "focus_m5",
    },
    ("annotation:like_claim", "BBB"): {
        "generated_at": "2026-09-10T20:00:00",
        "session_date": "2026-08-20",
        "symbol": "BBB",
        "side": "SHORT",
        "channel": "annotation:like_claim",
        "statement": "liked",
        "statement_detail": "",
        "statement_id": "e-quick",
        "traded": "no",
        "trade_id": "",
        "trade_opened_at": "",
        "match_confidence": "",
        "match_basis": "no match",
        "journal_r": "",
        "journal_net_pnl": "",
        "paper_forward_return_h3": "",
        "paper_forward_return_h5": "",
        "paper_cohort": "",
    },
    ("annotation:like_claim", "CCC"): {
        "generated_at": "2026-09-10T20:00:00",
        "session_date": "2026-08-20",
        "symbol": "CCC",
        "side": "LONG",
        "channel": "annotation:like_claim",
        "statement": "liked",
        "statement_detail": "avwap_reclaim",
        "statement_id": "e-legacy",
        "traded": "no",
        "trade_id": "",
        "trade_opened_at": "",
        "match_confidence": "",
        "match_basis": "no match",
        "journal_r": "",
        "journal_net_pnl": "",
        "paper_forward_return_h3": "",
        "paper_forward_return_h5": "",
        "paper_cohort": "",
    },
    ("pick_feedback:like", "FFF"): {
        "generated_at": "2026-09-10T20:00:00",
        "session_date": "2026-08-20",
        "symbol": "FFF",
        "side": "LONG",
        "channel": "pick_feedback:like",
        "statement": "liked",
        "statement_detail": "d1",
        "statement_id": "",
        "traded": "no",
        "trade_id": "",
        "trade_opened_at": "",
        "match_confidence": "",
        "match_basis": "no match",
        "journal_r": "",
        "journal_net_pnl": "",
        "paper_forward_return_h3": "",
        "paper_forward_return_h5": "",
        "paper_cohort": "",
    },
    ("swing_favorite", "JJJ"): {
        "generated_at": "2026-09-10T20:00:00",
        "session_date": "2026-08-20",
        "symbol": "JJJ",
        "side": "LONG",
        "channel": "swing_favorite",
        "statement": "picked",
        "statement_detail": "vetted",
        "statement_id": "",
        "traded": "no",
        "trade_id": "",
        "trade_opened_at": "",
        "match_confidence": "",
        "match_basis": "no match",
        "journal_r": "",
        "journal_net_pnl": "",
        "paper_forward_return_h3": "",
        "paper_forward_return_h5": "",
        "paper_cohort": "",
    },
}


def test_the_like_rows_keep_the_first_nineteen_columns_byte_identical(tmp_path):
    """A CHARACTERIZATION test, and it passes before the fix ON PURPOSE.

    WS-AI1's `preference_to_trade` section is already reading `match_basis`,
    `trade_id` and `session_date` out of this CSV and deriving its coverage
    buckets from them. Adding the reject half must not move one byte of what a
    like row writes today - so the golden is pinned from the CURRENT code and
    the builder's job is to leave it alone.

    Driven through the real path: the stores on disk -> `collect_statements` ->
    `build_rows` -> `write_rows` -> the CSV a reader opens.
    """
    from preference_trade_outcomes import build_rows, collect_statements, write_rows

    paths = _write_all_channels(tmp_path)
    # The three path kwargs the function takes TODAY, so this golden is readable
    # on the unfixed code: no like was ever written to the review-event store.
    statements = collect_statements(
        since=date(2026, 8, 1),
        until=date(2026, 9, 30),
        annotations_path=paths["annotations_path"],
        feedback_path=paths["feedback_path"],
        favorites_path=paths["favorites_path"],
    )
    grades = {
        ("2026-08-20", "AAA", "LONG"): {
            "h3": "0.0681", "h5": "0.1721", "cohort": "focus_m5",
        }
    }
    report = tmp_path / "preference_trade_outcomes.csv"
    assert write_rows(
        build_rows(statements, [_trade()], grades=grades, now=NOW), report
    )

    with report.open("r", newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))

    like_channels = {"annotation:like_claim", "pick_feedback:like", "swing_favorite"}
    seen = {}
    for row in written:
        if row["channel"] not in like_channels:
            continue
        key = (row["channel"], row["symbol"])
        seen[key] = {
            column: row[column]
            for column in GOLDEN_LIKE_ROWS[key]
        }
    assert seen == GOLDEN_LIKE_ROWS
    # Only the version string may move among the published 19.
    assert all(row["schema"] for row in written)


# ===========================================================================
# 2 - money once per trade, and the families never pool
# ===========================================================================
def test_two_statements_about_one_trade_sum_the_money_once(tmp_path):
    """ST5.2's rule, restated as a regression pin: three things said about one
    name are three statements and ONE trade's P&L.

    This one PASSES before the fix (`trade_level_summary` already keys money by
    `trade_id`); it is here so the reject half cannot re-introduce the bug by
    counting a veto's matched trade a second time.
    """
    from preference_trade_outcomes import build_rows, trade_level_summary

    statements = [
        {
            "session_date": date.fromisoformat(SAID), "symbol": "AAA", "side": "LONG",
            "channel": "annotation:like_claim", "statement": "liked",
            "statement_detail": "", "statement_id": "e-1",
        },
        {
            "session_date": date.fromisoformat(SAID), "symbol": "AAA", "side": "LONG",
            "channel": "swing_favorite", "statement": "picked",
            "statement_detail": "vetted", "statement_id": "",
        },
    ]
    rows = build_rows(statements, [_trade()], now=NOW)
    summary = trade_level_summary(rows)

    assert summary["n_statements"] == 2
    assert summary["n_statements_matched"] == 2
    assert summary["n_trades_matched"] == 1
    assert summary["net_pnl"] == pytest.approx(123.45)
    assert summary["duplicate_statement_rows"] == 1


def test_the_summary_counts_statements_by_family_and_the_trade_once(tmp_path):
    """The gate's number: `n_statements` PER FAMILY beside `n_trades_matched`.

    One endorsement and one refusal about the same name, both matching the one
    trade the trader actually took. The families are two answers - "I said take
    it" and "I said leave it" - and a single `n_statements` cannot carry both.
    The money is still the trade's, once.

    Fail-before-fix: `trade_level_summary` has no family breakdown at all.
    """
    from preference_trade_outcomes import build_rows, trade_level_summary

    said = date.fromisoformat(SAID)
    statements = [
        {
            "session_date": said, "symbol": "AAA", "side": "LONG",
            "channel": "annotation:like_claim", "statement": "liked",
            "statement_detail": "", "statement_id": "e-1",
            "verdict_family": "endorse", "like_mode": "quick",
        },
        {
            "session_date": said, "symbol": "AAA", "side": "LONG",
            "channel": "annotation:veto", "statement": "vetoed",
            "statement_detail": "incoming_trendline", "statement_id": "e-2",
            "verdict_family": "reject", "like_mode": "",
        },
        {
            "session_date": said, "symbol": "ZZZ", "side": "LONG",
            "channel": "pick_feedback:dislike", "statement": "disliked",
            "statement_detail": "", "statement_id": "",
            "verdict_family": "reject", "like_mode": "",
        },
    ]
    rows = build_rows(statements, [_trade()], now=NOW)
    summary = trade_level_summary(rows)

    assert summary["n_statements_by_family"] == {"endorse": 1, "reject": 2}
    assert summary["n_trades_matched"] == 1
    assert summary["net_pnl"] == pytest.approx(123.45)


def test_the_slot_reports_both_families_and_never_pools_them(tmp_path, monkeypatch):
    """The nightly result carries what the gate reads: statements per family and
    the distinct trade count.

    Fail-before-fix: the slot returns one `rows` count for everything, because
    everything it can see is an endorsement.
    """
    import project_paths
    from preference_trade_outcomes import run_preference_trade_outcomes

    paths = _write_all_channels(tmp_path)
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", paths["annotations_path"])
    monkeypatch.setattr(project_paths, "PICK_FEEDBACK_FILE", paths["feedback_path"])
    monkeypatch.setattr(project_paths, "ALERT_REVIEW_EVENTS_FILE", paths["events_path"])

    result = run_preference_trade_outcomes(
        now=NOW,
        report_path=tmp_path / "report.csv",
        trades=[_trade(symbol="DDD")],
    )

    assert result["status"] == "ok"
    families = result["n_statements_by_family"]
    assert families["endorse"] >= 4
    assert families["reject"] >= 3
    assert result["n_trades_matched"] == 1
    # Two families, never added into one headline.
    assert families["endorse"] + families["reject"] == result["rows"]


# ===========================================================================
# 3 - ambiguity is kept, never resolved into a claim
# ===========================================================================
def test_a_sideless_match_stays_ambiguous_and_is_never_asserted(tmp_path):
    """A symbol/date coincidence with an unknown side is a COINCIDENCE.

    The report already says so with `symbol+window_side_unknown` at 0.50, and
    that string is a published value WS-AI1 reads - so it does not move. What is
    new is that the row also says, in its own column, that a trade WAS found:
    ambiguity is labelled, not hidden and not upgraded.

    Fail-before-fix: there is no `match_state` column.
    """
    from preference_trade_outcomes import build_rows

    statement = {
        "session_date": date.fromisoformat(SAID), "symbol": "AAA", "side": "",
        "channel": "annotation:like_claim", "statement": "liked",
        "statement_detail": "", "statement_id": "e-1",
    }
    row = build_rows([statement], [_trade()], now=NOW)[0]

    assert row["match_basis"] == "symbol+window_side_unknown"
    assert row["match_confidence"] == "0.50"
    assert row["match_state"] == "matched"


def test_an_open_window_is_not_a_miss_and_a_closed_one_is(tmp_path):
    """Three different answers, never one blank.

    * the 10-SESSION window has not closed -> `window_open`: not yet an answer;
    * it closed with no trade -> `no_match_after_window`: the real "said it,
      did not do it";
    * a trade was found -> `matched`.

    Counted on the exchange calendar: `SAID_RECENT`'s window closes 2026-09-22,
    two weeks after the statement, which no calendar-day arithmetic reaches.

    Fail-before-fix: there is no `match_state` column, so the three collapse
    into `traded == "no"`.
    """
    from preference_trade_outcomes import build_rows

    def _statement(symbol, said):
        return {
            "session_date": date.fromisoformat(said), "symbol": symbol,
            "side": "LONG", "channel": "annotation:like_claim",
            "statement": "liked", "statement_detail": "", "statement_id": f"e-{symbol}",
        }

    rows = {
        row["symbol"]: row
        for row in build_rows(
            [
                _statement("AAA", SAID),
                _statement("BBB", SAID),
                _statement("CCC", SAID_RECENT),
            ],
            [_trade(symbol="AAA")],
            now=NOW,
        )
    }

    assert rows["AAA"]["match_state"] == "matched"
    assert rows["BBB"]["match_state"] == "no_match_after_window"
    assert rows["CCC"]["match_state"] == "window_open"


def test_a_calendar_refusal_narrows_the_window_it_never_widens_it(monkeypatch):
    """Uncertainty must not invent a match.

    A calendar outside its validated range is uncertainty, and the fallback is
    the strictly NARROWER calendar-day arithmetic. Passes before the fix; it is
    pinned here because `match_state` now depends on the same window, and a
    widened fallback would turn a real miss into "still open".
    """
    import market_calendar
    from preference_trade_outcomes import TRADE_WINDOW_SESSIONS, statement_window_end

    said = date.fromisoformat(SAID)
    true_end = statement_window_end(said)

    def _refuse(*_args, **_kwargs):
        raise market_calendar.SessionCalendarError("outside the validated range")

    monkeypatch.setattr(market_calendar, "is_session", _refuse)
    fallback = statement_window_end(said)

    assert fallback == date(2026, 8, 30)
    assert fallback < true_end
    assert (fallback - said).days == TRADE_WINDOW_SESSIONS


def test_a_missing_journal_keeps_the_statements_and_names_the_gap(tmp_path, monkeypatch):
    """The trader still SAID it. An unreadable journal is a gap in one half of
    the row, not a reason to publish nothing.

    `match_basis` stays EMPTY on these rows, because that is exactly how
    `ai_summary.preference_to_trade_section` already recognises the
    `journal_unavailable` bucket - the new column has to agree with the reader
    that shipped first.

    Fail-before-fix: the slot returns `status="skipped"` and writes no file at
    all when `JournalStore` raises.
    """
    import journal_store
    import project_paths
    from preference_trade_outcomes import run_preference_trade_outcomes

    paths = _write_all_channels(tmp_path)
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", paths["annotations_path"])
    monkeypatch.setattr(project_paths, "PICK_FEEDBACK_FILE", paths["feedback_path"])
    monkeypatch.setattr(project_paths, "ALERT_REVIEW_EVENTS_FILE", paths["events_path"])

    def _boom(*_args, **_kwargs):
        raise RuntimeError("journal database is locked")

    monkeypatch.setattr(journal_store, "JournalStore", _boom)

    report = tmp_path / "report.csv"
    result = run_preference_trade_outcomes(now=NOW, report_path=report)

    assert result["status"] != "skipped"
    assert report.is_file()
    with report.open("r", newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written
    assert {row["match_state"] for row in written} == {"journal_unavailable"}
    assert {row["match_basis"] for row in written} == {""}
    assert "journal" in result["reason"].lower()


def test_the_match_state_column_agrees_with_the_ai_sections_derived_buckets(tmp_path):
    """Two readers, one answer.

    WS-AI1 derives `window_open` / `no_match_after_window` / `journal_unavailable`
    from `match_basis` and `session_date`. This packet writes the same three as a
    COLUMN. If they ever disagree the desk has two truths about the same row, so
    the agreement is asserted rather than assumed.

    Fail-before-fix: there is no `match_state` column to compare.
    """
    from ai_summary import preference_to_trade_section
    from preference_trade_outcomes import build_rows, write_rows

    def _statement(symbol, said):
        return {
            "session_date": date.fromisoformat(said), "symbol": symbol,
            "side": "LONG", "channel": "annotation:like_claim",
            "statement": "liked", "statement_detail": "", "statement_id": f"e-{symbol}",
        }

    report = tmp_path / "report.csv"
    rows = build_rows(
        [
            _statement("AAA", SAID),
            _statement("BBB", SAID),
            _statement("CCC", SAID_RECENT),
        ],
        [_trade(symbol="AAA")],
        now=NOW,
    )
    assert write_rows(rows, report)

    section = preference_to_trade_section(report, now=NOW.astimezone())
    with report.open("r", newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    states = [row["match_state"] for row in written]

    coverage = section["coverage"]
    assert section["n_trades_matched"] == states.count("matched")
    assert coverage["window_open"] == states.count("window_open")
    assert coverage["no_match_after_window"] == states.count("no_match_after_window")
    assert coverage["journal_unavailable"] == states.count("journal_unavailable")


def test_a_duplicate_statement_is_written_once(tmp_path):
    """The annotation log heals torn tails rather than claiming atomicity, so a
    row can reach the file twice. One thing the trader said once is ONE
    statement; two rows would double its family count and its cohort weight.

    Fail-before-fix: there is no veto statement at all, so there is nothing to
    de-duplicate.
    """
    from ui.annotations.store import EVENT_VETO, record_annotation

    annotations = tmp_path / "trader_annotations.jsonl"
    for _ in range(2):
        record_annotation(
            EVENT_VETO,
            path=annotations,
            symbol="DDD",
            side="LONG",
            session_date=SAID,
            created_at=datetime.fromisoformat(CREATED),
            reason_code="incoming_trendline",
            event_id="e-dup",
        )
    assert len(annotations.read_text(encoding="utf-8").strip().splitlines()) == 2

    statements = _collect(
        {
            "annotations_path": annotations,
            "feedback_path": tmp_path / "none-feedback.jsonl",
            "favorites_path": tmp_path / "none-favorites.jsonl",
        },
        with_events=False,
    )

    vetoes = [row for row in statements if row["symbol"] == "DDD"]
    assert len(vetoes) == 1
    assert vetoes[0]["statement_id"] == "e-dup"


def test_a_retraction_removes_the_pick_and_an_unfavorite_is_never_a_verdict(tmp_path):
    """What the trader STANDS BEHIND, not everything they ever clicked.

    A swing favorite that was added and then retracted is not a pick they made;
    an `unfavorite` is housekeeping and never a negative judgement (P5). A
    dislike, beside them, IS a verdict and must be there - which is what makes
    this test red today.
    """
    statements = _collect(_write_all_channels(tmp_path), with_events=False)
    symbols = {row["symbol"] for row in statements}

    assert "GGG" in symbols  # the dislike is a verdict
    assert "KKK" not in symbols  # retracted favorite
    assert "III" not in symbols  # unfavorite is never graded
    assert not any(
        "unfavorite" in str(row.get("channel") or "")
        or "unfavorite" in str(row.get("statement") or "")
        for row in statements
    )


# ===========================================================================
# 4 - the Weekend Prep consumer
# ===========================================================================
def _report_csv(path: Path) -> None:
    """A 5B-schema report covering both families, in the week of 2026-09-07."""
    from preference_trade_outcomes import COLUMNS

    rows = [
        {
            "schema": "preference_trade_outcomes_v2", "generated_at": "2026-09-12T20:00:00",
            "session_date": "2026-09-08", "symbol": "LIKEA", "side": "LONG",
            "channel": "annotation:like_claim", "statement": "liked",
            "statement_detail": "avwap_reclaim", "statement_id": "e-1",
            "traded": "yes", "trade_id": "t-1", "trade_opened_at": "2026-09-08T09:41:00-04:00",
            "match_confidence": "0.90", "match_basis": "symbol+side+same_session",
            "journal_r": "1.2000", "journal_net_pnl": "123.4500",
            "paper_forward_return_h3": "", "paper_forward_return_h5": "", "paper_cohort": "",
            "like_mode": "claimed", "verdict_family": "endorse", "match_state": "matched",
        },
        {
            "schema": "preference_trade_outcomes_v2", "generated_at": "2026-09-12T20:00:00",
            "session_date": "2026-09-09", "symbol": "LIKEB", "side": "SHORT",
            "channel": "annotation:like_claim", "statement": "liked",
            "statement_detail": "", "statement_id": "e-2",
            "traded": "no", "trade_id": "", "trade_opened_at": "",
            "match_confidence": "", "match_basis": "no match",
            "journal_r": "", "journal_net_pnl": "",
            "paper_forward_return_h3": "", "paper_forward_return_h5": "0.0400",
            "paper_cohort": "like_quick",
            "like_mode": "quick", "verdict_family": "endorse", "match_state": "window_open",
        },
        {
            "schema": "preference_trade_outcomes_v2", "generated_at": "2026-09-12T20:00:00",
            "session_date": "2026-09-09", "symbol": "VETOC", "side": "LONG",
            "channel": "annotation:veto", "statement": "vetoed",
            "statement_detail": "incoming_trendline (v3)", "statement_id": "e-3",
            "traded": "yes", "trade_id": "t-2", "trade_opened_at": "2026-09-10T10:02:00-04:00",
            "match_confidence": "0.70", "match_basis": "symbol+side+window",
            "journal_r": "-0.8000", "journal_net_pnl": "-82.3000",
            "paper_forward_return_h3": "", "paper_forward_return_h5": "", "paper_cohort": "",
            "like_mode": "", "verdict_family": "reject", "match_state": "matched",
        },
        {
            "schema": "preference_trade_outcomes_v2", "generated_at": "2026-09-12T20:00:00",
            "session_date": "2026-09-10", "symbol": "DISD", "side": "LONG",
            "channel": "pick_feedback:dislike", "statement": "disliked",
            "statement_detail": "chased it last time", "statement_id": "",
            "traded": "no", "trade_id": "", "trade_opened_at": "",
            "match_confidence": "", "match_basis": "no match",
            "journal_r": "", "journal_net_pnl": "",
            "paper_forward_return_h3": "", "paper_forward_return_h5": "", "paper_cohort": "",
            "like_mode": "", "verdict_family": "reject", "match_state": "window_open",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def test_the_weekend_prep_reader_carries_the_family_and_the_match_state(tmp_path, monkeypatch):
    """The page cannot separate what the reader threw away.

    Fail-before-fix: `_read_preference_trade_rows` builds a fixed eight-key dict
    with no family and no match state in it.
    """
    import preference_trade_outcomes
    from ui.panels import weekend_prep_panel as panel_module

    report = tmp_path / "preference_trade_outcomes.csv"
    _report_csv(report)
    monkeypatch.setattr(preference_trade_outcomes, "REPORT_FILE", report)

    rows = panel_module._read_preference_trade_rows((date(2026, 9, 7), date(2026, 9, 11)))

    assert len(rows) == 4
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["LIKEA"]["verdict_family"] == "endorse"
    assert by_symbol["VETOC"]["verdict_family"] == "reject"
    assert by_symbol["VETOC"]["match_state"] == "matched"
    assert by_symbol["DISD"]["match_state"] == "window_open"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_the_weekend_prep_page_shows_rejections_in_their_own_table(tmp_path, monkeypatch, qapp):
    """"Rejections that were traded anyway / not traded", beside the likes -
    never pooled into one table.

    A veto the trader took anyway and a like they skipped are two different
    lessons; one table sorted by date would read as one population. The
    `match_state` is on screen because "no trade yet" and "no trade, window
    closed" are the difference between a pending row and a broken promise.

    Fail-before-fix: the page has one `preference_table` and renders every
    statement into it.
    """
    from PySide6.QtWidgets import QTableWidget
    import preference_trade_outcomes
    from ui.panels import weekend_prep_panel as panel_module

    report = tmp_path / "preference_trade_outcomes.csv"
    _report_csv(report)
    monkeypatch.setattr(preference_trade_outcomes, "REPORT_FILE", report)

    panel = panel_module.WeekendPrepPanel()
    try:
        page = panel.focus_review
        rows = panel_module._read_preference_trade_rows(
            (date(2026, 9, 7), date(2026, 9, 11))
        )
        page._render_preference_trades(rows)
        qapp.processEvents()

        likes: QTableWidget = page.preference_table
        rejects: QTableWidget = page.preference_rejection_table

        def _symbols(table: QTableWidget) -> set[str]:
            found = set()
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    item = table.item(row, column)
                    text = item.text() if item is not None else ""
                    if text in {"LIKEA", "LIKEB", "VETOC", "DISD"}:
                        found.add(text)
            return found

        def _cells(table: QTableWidget) -> set[str]:
            return {
                (table.item(row, column).text() if table.item(row, column) else "")
                for row in range(table.rowCount())
                for column in range(table.columnCount())
            }

        assert likes.rowCount() == 2
        assert _symbols(likes) == {"LIKEA", "LIKEB"}
        assert rejects.rowCount() == 2
        assert _symbols(rejects) == {"VETOC", "DISD"}
        # The match state is VISIBLE on the rejection table, not just in the CSV.
        assert {"matched", "window_open"} <= _cells(rejects)
        # Nothing pooled: the likes note may not claim the four statements.
        assert "4" not in page.preference_note.text()
        assert page.preference_rejection_note.text().strip()
    finally:
        panel.shutdown()
