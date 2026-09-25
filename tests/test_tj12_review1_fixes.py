"""TJ-12 review round 1 - the two blockers and the three cheap fixes.

Reviewer, 2026-09-20, reproduced on a read-only COPY of the live ledger
(`\\\\MINI-PC\\Trading Bot Data\\ai_store\\logs\\ai_job_ledger.jsonl`, 1,256,082
bytes, 483 rows) and on copies of the live journal and annotation stores.

BLOCKER 1 - a ledger ``status`` is a vocabulary `ai_jobs/ledger.py` OWNS, and
the card invented its own reading of it. `status != "ok"` named 22 slots broken
for 2026-09-18; **20 of them ran `ok` and were `skipped` by the next
half-hourly pass** (`daily_digest` ok 22:02:10, skipped 03:30:41;
`weekly_synthesis` ok 02:52:23, skipped three minutes later). Only
`journal_import` (failed x3) and `ai_summary` (degraded) were really not ok.
The one line whose job is to say when the night failed becomes the line the
trader learns to ignore.

BLOCKER 2 - `trade_origin.planned_state` cannot tell an UNREAD lane from
"nothing was said", and the desk ships `focus_adds` and `armed` EMPTY. Measured
on the live journal: **30 of 33 trades since 2026-08-20 read `unplanned`**, and
since the wake the Mentor asks about each one. Missing data read as
confirmation (plan.md sec 5). The lane readers are TJ-12F's; the honest wording
is this packet's.

The lead's decisions, 2026-09-20:

* import the ledger's OWN constants, never re-spell them; `skipped`,
  `manual_test` and `correction` are not failures; `failed` and `degraded` are
  NAMED SEPARATELY; the line says how many slots finished ok;
* the Process line never says a bare `unplanned` while a lane is unread - it
  counts ``planned`` and ``no claim or like before the fill`` and NAMES the
  unread lanes, carrying `lanes_read` / `lanes_unread` so TJ-5 and the pack can
  say the same; with every lane declared read, the plain wording comes back;
* the question says the same thing and gains ONE option, `a_focus_pick`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj12_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
SESSION = fx.SESSION


# ---------------------------------------------------------------------------
# helpers - every status comes from the OWNER, never spelled here
# ---------------------------------------------------------------------------
def _row(job: str, status: str, minute: int, *, session: str = SESSION) -> dict:
    import ai_jobs.ledger as ledger

    moment = datetime(2026, 9, 18, 22, 0, tzinfo=PACIFIC) + timedelta(minutes=minute)
    return {
        "schema": ledger.LEDGER_SCHEMA,
        "job": job,
        "status": status,
        "session_date": session,
        "model": "",
        "started_at": moment.isoformat(timespec="seconds"),
        "finished_at": (moment + timedelta(minutes=1)).isoformat(timespec="seconds"),
        "duration_seconds": 60.0,
        "reason": "",
        "outputs": [],
        "tokens": {},
        "error": "",
    }


def _fresh(tmp_path: Path, rows) -> dict:
    path = fx.write_ledger_file(tmp_path / "ai_store" / "logs" / "ai_job_ledger.jsonl", rows)
    return fx.freshness_facts(ledger_path=path)


def _line(card, key):
    for line in card.lines:
        if line["key"] == key:
            return line
    raise AssertionError(f"no {key!r} line: {[row['key'] for row in card.lines]}")


# ---------------------------------------------------------------------------
# BLOCKER 1 - the status vocabulary belongs to `ai_jobs/ledger.py`
# ---------------------------------------------------------------------------
def test_a_slot_that_ran_and_was_then_skipped_is_not_a_failure(tmp_path):
    """The live shape: `daily_digest` ok at 22:02, `skipped` by the 03:30 pass.

    A skip is a normal outcome - the runner writes one whenever the window or
    the already-done check says there is nothing to do.
    """
    import ai_jobs.ledger as ledger
    import day_report_card

    line = day_report_card.how_fresh(
        _fresh(
            tmp_path,
            [
                _row("daily_digest", ledger.STATUS_OK, 2),
                _row("daily_digest", ledger.STATUS_SKIPPED, 330),
            ],
        )
    )

    assert line["failed_slots"] == ()
    assert "daily_digest" not in line["text"], line["text"]
    assert line["night_status"] == "read"


def test_a_slot_that_only_ever_failed_is_named_as_failed(tmp_path):
    import ai_jobs.ledger as ledger
    import day_report_card

    line = day_report_card.how_fresh(
        _fresh(
            tmp_path,
            [_row("journal_import", ledger.STATUS_FAILED, minute) for minute in (10, 40, 70)],
        )
    )

    assert line["failed_slots"] == ("journal_import",)
    assert line["slots_failed"] == ("journal_import",)
    assert line["slots_degraded"] == ()
    assert "failed: journal_import" in line["text"], line["text"]


def test_a_slot_that_failed_then_recovered_is_not_named(tmp_path):
    import ai_jobs.ledger as ledger
    import day_report_card

    line = day_report_card.how_fresh(
        _fresh(
            tmp_path,
            [
                _row("market_story_narration", ledger.STATUS_FAILED, 10),
                _row("market_story_narration", ledger.STATUS_OK, 20),
            ],
        )
    )

    assert line["failed_slots"] == ()
    assert "market_story_narration" not in line["text"], line["text"]


def test_a_degraded_slot_is_named_separately_from_a_failed_one(tmp_path):
    """`degraded_no_narrative` published a real document with no narrative.

    Pooling it with `failed` would tell the trader nothing ran when something
    did - and the owner keeps them apart for exactly that reason.
    """
    import ai_jobs.ledger as ledger
    import day_report_card

    line = day_report_card.how_fresh(
        _fresh(
            tmp_path,
            [
                _row("ai_summary", ledger.STATUS_DEGRADED, 10),
                _row("journal_import", ledger.STATUS_FAILED, 20),
            ],
        )
    )

    assert line["slots_failed"] == ("journal_import",)
    assert line["slots_degraded"] == ("ai_summary",)
    assert "failed: journal_import" in line["text"], line["text"]
    assert "degraded: ai_summary" in line["text"], line["text"]


def test_a_manual_run_alone_is_never_a_failure(tmp_path):
    import ai_jobs.ledger as ledger
    import day_report_card

    line = day_report_card.how_fresh(
        _fresh(tmp_path, [_row("ticker_briefs", ledger.STATUS_MANUAL, 10)])
    )

    assert line["failed_slots"] == ()
    assert "ticker_briefs" not in line["text"], line["text"]


def test_the_live_shaped_night_names_exactly_the_two_slots_that_broke(tmp_path):
    """The reviewer's own reproduction, as a fixture: twenty slots that ran and
    were then skipped, one that failed three times, one that degraded."""
    import ai_jobs.ledger as ledger
    import day_report_card

    rows = []
    for index in range(20):
        name = f"slot_{index:02d}"
        rows.append(_row(name, ledger.STATUS_OK, index))
        rows.append(_row(name, ledger.STATUS_SKIPPED, 300 + index))
    rows.extend(_row("journal_import", ledger.STATUS_FAILED, m) for m in (30, 60, 90))
    rows.append(_row("ai_summary", ledger.STATUS_DEGRADED, 100))

    line = day_report_card.how_fresh(_fresh(tmp_path, rows))

    assert line["n"] == 22
    assert line["failed_slots"] == ("ai_summary", "journal_import")
    assert line["slots_ok"] == 20
    for index in range(20):
        assert f"slot_{index:02d}" not in line["text"], line["text"]


def test_the_card_never_spells_a_status_of_its_own():
    """The vocabulary is the owner's. A second copy of it in this module is a
    copy that drifts the day the runner adds a status."""
    import ai_jobs.ledger as ledger

    source = (SCRIPTS_DIR / "day_report_card.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for spelled in (ledger.STATUS_DEGRADED, ledger.STATUS_MANUAL, ledger.STATUS_SKIPPED):
        assert f'"{spelled}"' not in body, f"{spelled} is spelled in day_report_card"


# ---------------------------------------------------------------------------
# BLOCKER 2 - an unread lane is never read as "nothing was said"
# ---------------------------------------------------------------------------
def test_the_process_line_names_the_lanes_nobody_reads(tmp_path):
    """30 of the trader's last 33 trades read `unplanned` for this reason."""
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    line = _line(day_report_card.build(fx.day_inputs(tmp_path)), "process")

    assert tuple(line["lanes_read"]) == tuple(day_report_card.DESK_ORIGIN_LANES_READ)
    assert tuple(line["lanes_unread"]) == ("focus_adds", "armed")
    lowered = line["text"].lower()
    assert "focus" in lowered and "armed" in lowered, line["text"]
    assert "no claim or like before the fill" in lowered, line["text"]


def test_a_bare_unplanned_is_never_printed_while_a_lane_is_unread(tmp_path):
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    line = _line(day_report_card.build(fx.day_inputs(tmp_path)), "process")

    assert "unplanned" not in line["text"].lower(), line["text"]
    # The COUNT keeps its name: `trade_origin`'s answer is unchanged and TJ-9's
    # own readers still see exactly what they always saw.
    assert line["unplanned"] == 1


def test_with_every_lane_read_the_plain_wording_comes_back(tmp_path):
    """So TJ-12F only has to fill the lanes - not re-word the card."""
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)
    inputs["origin_lanes_read"] = day_report_card.ORIGIN_LANES
    line = _line(day_report_card.build(inputs), "process")

    assert tuple(line["lanes_unread"]) == ()
    assert "unplanned" in line["text"].lower(), line["text"]
    assert "not read yet" not in line["text"].lower(), line["text"]


def test_the_two_lane_builders_declare_the_same_lanes(monkeypatch):
    """ONE constant, not two literals: the desk and the worker cannot disagree
    about which lanes were opened."""
    pytest.importorskip("PySide6", reason="the desk window needs PySide6")
    import claimed_picks
    import day_report_card
    from ui.app import MainWindow

    monkeypatch.setattr(MainWindow, "_mentor_annotation_lane", staticmethod(lambda _d: []))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *_a, **_k: [])

    lanes = MainWindow._mentor_origin_lanes((SESSION,))
    read = {name for name in day_report_card.DESK_ORIGIN_LANES_READ}

    assert read == {name for name in lanes if name in read}
    assert set(day_report_card.ORIGIN_LANES) == set(lanes)
    assert set(day_report_card.ORIGIN_LANES) - read == {"focus_adds", "armed"}


def test_the_question_says_why_it_is_asking_and_offers_a_focus_pick():
    """(b): the prompt carries the same caveat the card's line does, and the
    trader can answer what the desk cannot yet read."""
    import mentor_questions

    kind = mentor_questions.kind_named("trade_origin")
    assert "a_focus_pick" in kind.options
    # The four the tester pinned as a SUBSET are all still there.
    assert {"planned_off_the_desk", "an_alert", "impulse", "other"} <= set(kind.options)

    subjects = kind.trigger(
        {
            "session": SESSION,
            "trades": [
                {
                    "trade_id": "T-1",
                    "symbol": "AAPL",
                    "direction": "LONG",
                    "status": "CLOSED",
                    "opened_at": f"{SESSION}T07:31:00-04:00",
                }
            ],
        }
    )

    assert len(subjects) == 1
    prompt = subjects[0].prompt.lower()
    assert "no claim or like" in prompt, subjects[0].prompt
    assert "focus" in prompt and "armed" in prompt, subjects[0].prompt
    assert "a_focus_pick" in subjects[0].options


def test_process_reads_a_focus_pick_answer_like_any_other(tmp_path):
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)
    inputs["mentor_answers"] = [{"subject_id": "T3", "trade_origin": "a_focus_pick"}]
    line = _line(day_report_card.build(inputs), "process")

    assert line["told_us"] == 1
    assert line["origin_answers"] == {"a_focus_pick": 1}
    assert "told the desk" in line["text"].lower(), line["text"]


# ---------------------------------------------------------------------------
# A1 - one owner raising costs ONE line
# ---------------------------------------------------------------------------
def test_an_owner_that_raises_costs_its_own_line_and_no_other(tmp_path, monkeypatch, caplog):
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)

    def _boom(*_args, **_kwargs):
        raise ValueError("the congruence store is torn")

    monkeypatch.setattr(day_report_card, "congruence_line", _boom)
    with caplog.at_level("DEBUG"):
        card = day_report_card.build(inputs)

    assert tuple(row["key"] for row in card.lines) == tuple(day_report_card.LINE_KEYS)
    broken = _line(card, "congruence")
    assert "could not be read" in broken["text"].lower(), broken["text"]
    assert broken["measured_ok"] is False
    assert broken["n"] == 0 and broken["measured"] == 0
    assert "%" not in broken["text"]
    # The other five still say what they measured.
    assert _line(card, "did_well")["n"] == 4
    assert _line(card, "process")["planned"] == 2


def test_a_broken_line_still_reaches_the_page_through_the_worker(monkeypatch):
    """The whole-card guard used to swallow six lines and leave `error` empty."""
    pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
    import day_report_card

    import test_tj12_day_review_page as page_tests

    def _boom(*_args, **_kwargs):
        raise ValueError("the congruence store is torn")

    monkeypatch.setattr(day_report_card, "congruence_line", _boom)
    service = page_tests._wire(monkeypatch)
    payload = service.read_day(SESSION, now=page_tests.NOW)

    lines = payload["report_card"]["lines"]
    assert len(lines) == 6
    assert [row["key"] for row in lines] == list(day_report_card.LINE_KEYS)
    broken = next(row for row in lines if row["key"] == "congruence")
    assert broken["measured_ok"] is False


# ---------------------------------------------------------------------------
# A2 - the number and the table it points at agree
# ---------------------------------------------------------------------------
def test_did_wells_count_is_the_table_its_click_opens(tmp_path):
    """Gate #157: the numbers on the card match the tables under them."""
    import day_report_card
    import walkaway_day

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)
    day = inputs["walkaway"]
    claimed = (
        fx.walkaway_row("ZZZ", what_you_did="claimed D1 pick", real_miss_verdict="run"),
        fx.walkaway_row("YYY", what_you_did="claimed D1 pick", real_miss_verdict="run"),
    )
    inputs["walkaway"] = walkaway_day.WalkawayDay(
        liked_not_traded=day.liked_not_traded,
        rejected=day.rejected,
        claimed_d1=claimed,
        skill=day.skill,
        sentences=day.sentences,
        money=day.money,
    )
    line = _line(day_report_card.build(inputs), "process")  # keeps the card honest
    assert line["n"] == 4

    did_well = _line(day_report_card.build(inputs), "did_well")
    target = day_report_card.LINE_TARGETS["did_well"]
    assert target == "liked_not_traded"
    assert did_well["n"] == len(day.liked_not_traded), "the count must be the table's"
    assert did_well["claimed"] == 2
    assert "2 claimed D1 pick" in did_well["text"], did_well["text"]


# ---------------------------------------------------------------------------
# A4 - a week never pools the same session twice
# ---------------------------------------------------------------------------
def test_a_week_pools_each_session_once(tmp_path):
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    day = fx.day_inputs(tmp_path)
    once = day_report_card.week([day])
    twice = day_report_card.week([day, dict(day)])

    assert tuple(twice.sessions) == (fx.SESSION,)
    for first, second in zip(once.lines, twice.lines, strict=False):
        assert first["n"] == second["n"], first["key"]
        assert first["measured"] == second["measured"], first["key"]
