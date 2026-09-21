r"""TJ-9E - who READS an exit note, and what may never reach it.

Lead decision 9: *"(a) Day Review's trades section shows the exit note and, when
confirmed, its three fields beside the trade; (b) `day_report_card.process_line`
gains 'exits explained K of N' with `n`, never a rate under its floor; the
registry names that consumer."* A field nobody reads is a field the desk did not
build, and `mentor_questions` already refuses one by name.

And the structural half: nothing this packet writes is read by a detector, a
score, an alert, a watchlist, Focus, the review queue or `review_policy.json`.
Traced by CONTACT and IMPORTERS, the way
`tests/test_tj7_reported_never_acted_on.py` does it after the lead's 2026-09-20
amendment - never a two-word grep over prose, which was unsatisfiable by its own
packet.

RED FOR: `ai_jobs.exit_note_fields`, `trade_mentor_trade_check.exit_notes_for_session`
and the `exits explained` clause do not exist on this branch (verified 2026-09-21
at `05988440`). The three scans over shipped modules are STATED GREEN GUARDS -
they pass today and exist so the builder cannot make them stop.
"""

from __future__ import annotations

import ast
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for _extra in (SCRIPTS_DIR, ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj9e_support as fx  # noqa: E402

#: Everything that DECIDES something - TJ-7's list, unchanged, because the fence
#: is the same fence.
DECIDERS = (
    "bounce_bot.py",
    "bounce_bot_lib/legacy.py",
    "master_avwap.py",
    "master_avwap_lib/legacy.py",
    "focus_adoption_gate.py",
    "regime_pause_focus.py",
    "armed_alert_expiry.py",
    "candidate_registry.py",
    "daytrade_watchlist_reset.py",
    "market_state.py",
    "greatness_monitor.py",
    "review_learning.py",
    "ui/panels/alert_center_panel.py",
)

#: The words an exit note is stored and read under. A decider that names ANY of
#: them has read the trader's exit note.
EXIT_WORDS = (
    "exit_note_fields",
    "exit_reasons",
    "EXIT_NOTE_RAW",
    "exit_notes_for_session",
    "save_exit_note",
    "confirm_exit_fields",
)

#: Anything that could put a RESULT in front of an exit note.
OUTCOME_MODULES = {
    "setup_scoreboard", "outcome_semantics", "walkaway_day", "expected_r",
    "swing_headline", "evidence_stats", "market_read_grades", "real_miss",
    "evidence_contrast", "working_lately", "review_learning", "pick_feedback",
    "journal_walkaway", "setup_points",
}

NIGHT = datetime.fromisoformat("2026-09-14T02:00:00-04:00")


def _sources(names) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        path = SCRIPTS_DIR / name
        assert path.is_file(), f"{name} is not where this test thinks it is"
        out[name] = path.read_text(encoding="utf-8", errors="replace")
    return out


def _indicator_files() -> list[str]:
    return [
        str(path.relative_to(SCRIPTS_DIR)).replace("\\", "/")
        for path in sorted((SCRIPTS_DIR / "indicators").glob("*.py"))
    ]


# ---------------------------------------------------------------------------
# reader (a) - Day Review's trades section, inside the ONE payload
# ---------------------------------------------------------------------------
def test_the_day_reviews_trades_section_carries_the_note_and_the_confirmed_fields(
    tmp_path, monkeypatch
):
    """TJ-1: ONE payload, ONE worker. The note travels WITH the trade row.

    Hand-counted: 6 trades on the reviewed session, 2 of them with a raw note
    and exactly 1 of those confirmed. So the payload holds 6 trade rows, 2
    carrying `exit_note` and 1 carrying `exit_fields`; the other four carry the
    keys PRESENT and EMPTY, never absent - a reader that has to tell "no note"
    from "this build did not look" is reading two absences as one.
    """
    import exit_reasons
    import trade_mentor_trade_check as check
    import trader_state_tags
    from ai_jobs import exit_note_fields
    from ui.services import journal_feed
    from ui.services.day_review_service import DayReviewService

    store, ids = fx.ready_store(tmp_path)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)

    for symbol in (fx.SWING, fx.DAY_TRADE):
        check.save_exit_note(
            store, ids[symbol], fx.EXIT_NOTE, exit_session=fx.REVIEWED,
            now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
        )
    root = tmp_path / "packs"
    out = exit_note_fields.run_exit_note_fields(
        session_date=fx.REVIEWED, now=NIGHT, root=root, store=store,
        request=fx.fake_request(
            fx.good_reply(exit_reasons.codes()[0], trader_state_tags.codes()[:1])
        ),
    )
    assert out["status"] == "ok", out
    check.confirm_exit_fields(
        store,
        ids[fx.SWING],
        exit_note_fields.draft_for(ids[fx.SWING], fx.REVIEWED, root=root),
        now=datetime.fromisoformat("2026-09-14T09:30:00-04:00"),
    )

    # ONE session-wide read of the notes, not one per trade: six trades on a Qt
    # worker must not become six walks of an append-only table.
    reads: list[str] = []
    real = check.exit_notes_for_session
    monkeypatch.setattr(
        check,
        "exit_notes_for_session",
        lambda *args, **kwargs: (reads.append("read"), real(*args, **kwargs))[1],
    )

    service = DayReviewService()
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    payload = service.read_day(fx.REVIEWED, now=datetime(2026, 9, 14, 9, 0))

    rows = {str(row.get("symbol") or ""): row for row in payload["trades"]}
    assert len(rows) == fx.ROWS_EXPECTED_AFTER_THE_FIX, sorted(rows)
    assert len(reads) <= 1, f"{len(reads)} note reads for one payload"

    for symbol, row in rows.items():
        assert "exit_note" in row and "exit_fields" in row, (symbol, sorted(row))
    assert rows[fx.SWING]["exit_note"] == fx.EXIT_NOTE
    assert rows[fx.DAY_TRADE]["exit_note"] == fx.EXIT_NOTE
    assert rows[fx.ENTRY_ONLY]["exit_note"] == ""
    assert rows[fx.SWING]["exit_fields"]["fields"]["why"]["code"] in exit_reasons.codes()
    assert rows[fx.DAY_TRADE]["exit_fields"] == {}, "a draft was shown as confirmed"


# ---------------------------------------------------------------------------
# reader (b) - the report card's process line
# ---------------------------------------------------------------------------
def test_the_process_line_says_exits_explained_k_of_n_with_its_n(tmp_path):
    """Hand-counted: 5 of the six trades exited in the reviewed session and 2
    carry a note, so the line reads "exits explained 2 of 5" and the extras
    carry the two integers.

    The denominator is trades WITH AN EXIT, not all trades: a position nobody
    closed has no exit to explain, and counting it would make the number say
    the trader is worse at explaining than they are.
    """
    import day_report_card
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    for symbol in (fx.SWING, fx.DAY_TRADE):
        check.save_exit_note(
            store, ids[symbol], fx.EXIT_NOTE, exit_session=fx.REVIEWED,
            now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
        )
    trades = store.list_trades(trade_date=fx.REVIEWED)
    notes = check.exit_notes_for_session(store, fx.REVIEWED)
    assert len(notes) == 2, sorted(notes)

    line = day_report_card.process_line(trades, exit_notes=notes)

    assert line["exits_explained"] == 2, line
    assert line["exits_n"] == fx.EXIT_BOXES_EXPECTED, line
    assert "exits explained 2 of 5" in line["text"].lower(), line["text"]


def test_an_unread_exit_lane_is_unmeasured_and_never_a_zero(tmp_path):
    """A caller that did not open the notes and a session with none look
    identical from inside the line, and the difference is the whole meaning of
    the number (`process_line`'s own `lanes_read` rule).

    Hand-counted: `exit_notes=None` -> `exits_explained` is None, the word
    `unmeasured` is in the text, and "0 of 5" is NOT.
    """
    import day_report_card

    store, _ids = fx.ready_store(tmp_path)
    trades = store.list_trades(trade_date=fx.REVIEWED)

    line = day_report_card.process_line(trades, exit_notes=None)

    assert line["exits_explained"] is None, line
    assert "0 of" not in line["text"], line["text"]
    assert "unmeasured" in line["text"].lower(), line["text"]


def test_no_rate_is_printed_under_the_reporting_floor(tmp_path):
    """Hand-counted: 5 exits is far under `evidence_stats.MIN_REPORTABLE_N` (30,
    read from the module and never typed), so the line states the two COUNTS and
    prints no percentage at all. Nothing named under the floor - ground rule 10.
    """
    import day_report_card
    import evidence_stats
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    check.save_exit_note(
        store, ids[fx.SWING], fx.EXIT_NOTE, exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:05:00-04:00"),
    )
    trades = store.list_trades(trade_date=fx.REVIEWED)
    notes = check.exit_notes_for_session(store, fx.REVIEWED)

    line = day_report_card.process_line(trades, exit_notes=notes)

    assert line["exits_n"] < evidence_stats.MIN_REPORTABLE_N
    assert "%" not in line["text"], line["text"]
    assert "exits explained 1 of 5" in line["text"].lower(), line["text"]


def test_the_registry_names_the_reader_and_its_probe_passes_for_real():
    """`mentor_questions` refuses a kind that names no reader, and its probe is
    an AST walk of the consumer's own source - not a grep, not a call.

    So the `exit_draft_review` kind must name a consumer that really reads its
    `answer_key`, and `consumer_report()` must say so. A kind whose reader is
    not built yet ships DORMANT instead; this packet builds the reader, so it
    does not.
    """
    import mentor_questions

    kinds = {kind.kind: kind for kind in mentor_questions.REGISTRY}
    assert "exit_draft_review" in kinds, sorted(kinds)
    assert not str(kinds["exit_draft_review"].dormant_until or ""), (
        "this packet builds the reader, so the kind does not ship dormant"
    )

    unresolved = [
        row for row in mentor_questions.consumer_report()
        if not row.get("resolved") or not row.get("reads_key")
    ]
    assert unresolved == [], unresolved


# ---------------------------------------------------------------------------
# the fence
# ---------------------------------------------------------------------------
def test_no_detector_score_alert_watchlist_focus_or_review_module_reads_an_exit_note():
    """STATED GREEN GUARD, and it must stay green."""
    offenders: list[str] = []
    for name, source in _sources(list(DECIDERS) + _indicator_files()).items():
        for word in EXIT_WORDS:
            if word in source:
                offenders.append(f"{name}: {word}")
    assert not offenders, offenders


def test_nothing_that_touches_the_review_policy_handles_an_exit_note():
    """CONTACT, not words. A module that imports, loads, drafts or saves
    `review_policy.json` may not also handle an exit note.

    The narrowed scan must still SEE the policy's own modules, or it guards
    nothing - which is the defect the lead's 2026-09-20 amendment fixed in
    TJ-7's version of this test.
    """
    policy_contact = (
        "import review_policy",
        "from review_policy",
        "REVIEW_POLICY_FILE",
        "save_review_policy",
        "load_review_policy",
        "draft_policy_from_state",
    )
    hits: list[str] = []
    touching: list[str] = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="replace")
        if not any(token in source for token in policy_contact):
            continue
        touching.append(str(path.relative_to(SCRIPTS_DIR)))
        if any(word in source for word in EXIT_WORDS):
            hits.append(str(path.relative_to(SCRIPTS_DIR)))
    assert touching, "no module touches the review policy - the contact tokens went stale"
    assert not hits, hits


def test_the_exit_reason_vocabulary_cannot_reach_an_outcome():
    """No R statistic, verdict or outcome may select, rank or pre-fill an exit
    reason, so the module that owns the vocabulary cannot import one.

    The same shape as `tests/test_tj7_reported_never_acted_on.py:173-185`, which
    is why `scripts/exit_reasons.py` is a picklist loader and nothing else.
    """
    source = (SCRIPTS_DIR / "exit_reasons.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & OUTCOME_MODULES, sorted(imported & OUTCOME_MODULES)


def test_the_night_slot_cannot_reach_an_outcome_either():
    """The slot is blind to the result by CONSTRUCTION, not by discipline.

    `run_exit_note_fields` reads a note, two picklists and a store's own event
    table. A module that cannot import a grader cannot leak one into a prompt,
    however the payload is built later.
    """
    source = (SCRIPTS_DIR / "ai_jobs" / "exit_note_fields.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & OUTCOME_MODULES, sorted(imported & OUTCOME_MODULES)


# ---------------------------------------------------------------------------
# entries are untouched
# ---------------------------------------------------------------------------
def test_the_night_slot_never_imports_the_day_time_entry_draft():
    """*"trade entrys are good the way they are"*.

    `scripts/trade_mentor_ai.py` is the day-time, on-demand draft the card runs
    when the trader submits raw text for the FOUR entry fields. The night slot
    is a different job on a different clock over a different vocabulary, and it
    must not reach into it - a shared helper here is how the two grow one
    behaviour between them.
    """
    source = (SCRIPTS_DIR / "ai_jobs" / "exit_note_fields.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("trade_mentor_ai" in alias.name for alias in node.names), node.names
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert "trade_mentor_ai" not in node.module, node.module


def test_the_day_time_entry_draft_keeps_its_contract():
    """STATED GREEN GUARD. The entry draft's two public functions and their four
    answer states are exactly what they are today.

    Hand-counted: `validate_draft` keeps ONE answer whose `source_span` is a
    literal substring of the raw text, and RAISES on one whose is not - the rule
    `scripts/trade_mentor_ai.py:88` states, unchanged by this packet. Its
    signature is pinned too, because the temptation here is to widen it for the
    exit and end up with one function serving two clocks.
    """
    import inspect

    import trade_mentor_ai
    import trade_mentor_trade_check as check

    signature = inspect.signature(trade_mentor_ai.validate_draft)
    assert list(signature.parameters) == ["payload", "raw_text", "missing_fields"], signature

    raw = "I was long into the 50 day and my stop was the morning low."
    good = {
        "answers": [
            {
                "field": "stop",
                "state": check.ANSWER_NOT_SUPPLIED,
                "text": "the morning low",
                "source_span": "the morning low",
            }
        ]
    }
    kept = trade_mentor_ai.validate_draft(good, raw_text=raw, missing_fields=("stop",))
    assert [row["field"] for row in kept["answers"]] == ["stop"], kept

    bad = {
        "answers": [
            {
                "field": "stop",
                "state": check.ANSWER_NOT_SUPPLIED,
                "text": "the 200 day",
                "source_span": "a phrase the trader never wrote",
            }
        ]
    }
    with pytest.raises(ValueError):
        trade_mentor_ai.validate_draft(bad, raw_text=raw, missing_fields=("stop",))
