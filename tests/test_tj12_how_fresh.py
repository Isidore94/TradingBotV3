"""TJ-12's sixth line - How fresh, and the failed night named the next morning.

`plan.md` §12.4 TJ-12, AMENDED 2026-09-19: *"a sixth, smaller line **How fresh**
states what the card rests on - story written when, fills current to which date,
grades through which session, and any slot that failed last night by name (the
job ledger's last row per slot). A failed night is said on the page the next
morning."* Decision 0021 answer 27.

Packet item 3: *"The job ledger is read on the worker, tail-only, and a missing
AI store says `night status unknown`."*

Measured on the desk, 2026-09-20: the live ledger is
`\\\\MINI-PC\\Trading Bot Data\\ai_store\\logs\\ai_job_ledger.jsonl`, 1,256,082
bytes. `ai_jobs.ledger.recent_rows` is the bounded reader; its default path is
`ledger_path()`, which CREATES the store directory, so a reader that wants to
say "unknown" about a missing store has to hand it a path.

THE CONTRACT
------------
    day_report_card.LEDGER_TAIL_ROWS -> int > 0

    day_report_card.how_fresh(freshness) -> dict
        The `how_fresh` line, on its own, so the week re-cut and the page share
        one implementation. `freshness` is `{session, story_written_at,
        fills_current_to, reads_graded_through, ledger_path}`; `ledger_path`
        None or missing means the desk has no AI store.

        {"key": "how_fresh", "text", "n", "measured", "target",
         "failed_slots": (name, ...), "night_status": "read" | "unknown"}
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tj12_support as fx  # noqa: E402


def _ledger(tmp_path: Path, **kwargs) -> Path:
    return fx.write_ledger_file(
        tmp_path / "ai_store" / "logs" / "ai_job_ledger.jsonl",
        fx.ledger_rows(**kwargs),
    )


def test_how_fresh_names_the_story_time_the_fills_date_and_the_graded_session(tmp_path):
    import day_report_card

    path = _ledger(tmp_path)
    line = day_report_card.how_fresh(fx.freshness_facts(ledger_path=path))

    assert line["key"] == "how_fresh"
    assert "2026-09-18" in line["text"], line["text"]
    assert "2026-09-17" in line["text"], "the fills date the desk actually has"
    text = line["text"].lower()
    assert "story" in text and "fills" in text and "read" in text


def test_a_slot_whose_last_row_is_not_ok_is_named(tmp_path):
    """The job ledger's LAST row per slot, and the slot is named."""
    import day_report_card

    path = _ledger(tmp_path)
    line = day_report_card.how_fresh(fx.freshness_facts(ledger_path=path))

    assert tuple(line["failed_slots"]) == ("day_review_narration",)
    assert "day_review_narration" in line["text"], line["text"]


def test_a_slot_that_failed_then_recovered_is_not_named(tmp_path):
    """`market_story_narration` failed at 22:10 and succeeded at 22:20."""
    import day_report_card

    path = _ledger(tmp_path)
    line = day_report_card.how_fresh(fx.freshness_facts(ledger_path=path))

    assert "market_story_narration" not in line["text"], line["text"]


def test_a_missing_ai_store_says_night_status_unknown_and_creates_nothing(tmp_path):
    """Unknown is not "fine". `ledger_path()` would have made the folder."""
    import day_report_card

    absent = tmp_path / "no_ai_store" / "logs" / "ai_job_ledger.jsonl"
    line = day_report_card.how_fresh(fx.freshness_facts(ledger_path=absent))

    assert line["night_status"] == "unknown"
    assert "night status unknown" in line["text"].lower(), line["text"]
    assert not absent.parent.exists(), "a read must not create the store"


def test_no_path_at_all_is_unknown_and_never_a_clean_night(tmp_path):
    import day_report_card

    line = day_report_card.how_fresh(fx.freshness_facts(ledger_path=None))
    assert line["night_status"] == "unknown"
    assert not line["failed_slots"]
    assert "night status unknown" in line["text"].lower()


def test_the_ledger_is_read_through_the_owners_bounded_tail(tmp_path, monkeypatch):
    """Tail-only, by NAME, with an explicit path - the live file is 1.2 MB."""
    import ai_jobs.ledger as ledger
    import day_report_card

    path = _ledger(tmp_path)
    rows = fx.ledger_rows()
    seen: list[dict] = []

    def _recent(limit=50, *, path=None):
        seen.append({"limit": limit, "path": path})
        return list(rows)

    monkeypatch.setattr(ledger, "recent_rows", _recent)
    monkeypatch.setattr(
        ledger, "_read_rows",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("the whole file was read")),
    )
    line = day_report_card.how_fresh(fx.freshness_facts(ledger_path=path))

    assert seen, "`how_fresh` did not go through ai_jobs.ledger.recent_rows"
    assert seen[0]["path"] is not None, "a default path would create the AI store"
    assert 0 < int(seen[0]["limit"]) == day_report_card.LEDGER_TAIL_ROWS
    assert tuple(line["failed_slots"]) == ("day_review_narration",)


def test_the_tail_is_a_named_constant_and_is_bounded():
    import day_report_card

    assert isinstance(day_report_card.LEDGER_TAIL_ROWS, int)
    assert 0 < day_report_card.LEDGER_TAIL_ROWS <= 5_000


def test_how_fresh_counts_the_slots_it_looked_at(tmp_path):
    """`n` is the slots the tail held for this session; `measured` those read."""
    import day_report_card

    path = _ledger(tmp_path)
    line = day_report_card.how_fresh(fx.freshness_facts(ledger_path=path))
    assert line["n"] == 2
    assert line["measured"] == 2

    blank = day_report_card.how_fresh(fx.freshness_facts(ledger_path=None))
    assert blank["n"] == 0 and blank["measured"] == 0


def test_a_missing_fills_date_is_named_never_dated(tmp_path):
    """`trade_mentor_trade_check.fills_current_to` answers None on no ledger.

    An absence is not a date (TJ-9), so the line says the desk has no verified
    coverage rather than printing today's date or an empty string.
    """
    import day_report_card

    path = _ledger(tmp_path)
    line = day_report_card.how_fresh(
        fx.freshness_facts(ledger_path=path, fills_current_to="")
    )
    lowered = line["text"].lower()
    assert "no verified" in lowered or "nothing yet" in lowered, line["text"]


def test_the_card_carries_the_same_how_fresh_line(tmp_path):
    """One implementation: the sixth line on the card IS `how_fresh`'s answer."""
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    path = _ledger(tmp_path)
    inputs = fx.day_inputs(tmp_path, ledger_path=path)
    built = day_report_card.build(inputs)

    assert built.lines[-1]["key"] == "how_fresh"
    assert built.lines[-1] == day_report_card.how_fresh(inputs["freshness"])
