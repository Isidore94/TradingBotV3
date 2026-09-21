r"""TJ-5 review round 1 - the five cheap advisories, as tests. Offscreen.

Reviewer GO at `f7fac9cb` with no blockers; these are its advisories 1, 2, 4, 5
and 6, each turned into the thing that would have caught it. The tester's own
files are UNTOUCHED - advisory 6 is a test the reviewer asked the builder to
ADD beside `test_the_week_story_makes_at_most_one_model_call_per_run`, not an
edit to it.

**NO MODEL IS EVER CALLED HERE.** The one test that reaches
`run_week_review_narration` hands it a `request` callable of its own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as fx  # noqa: E402


def _lines(card):
    return {line["key"]: line for line in card.lines}


# ---------------------------------------------------------------------------
# advisory 1 - a reserve equal to the timeout reserves nothing for the load
# ---------------------------------------------------------------------------
def test_the_unmeasured_reserve_is_longer_than_one_call_may_take():
    """A call that runs to its own timeout must not consume the whole reserve.

    `DEFAULT_RESERVE_MINUTES` was 30.0 beside a `TIMEOUT_SECONDS` of 1800 - the
    same thirty minutes - so a slot that reserved it and then spent the timeout
    had nothing left for the model load, the schema validation, the grounding
    checks and the write, and ended past the window it reserved against.

    Asserted as a RELATION rather than as two literals, so retuning either
    constant cannot quietly reintroduce it.
    """
    import ai_jobs.week_review_narration as week

    assert week.RESERVE_MARGIN_MINUTES > 0
    assert week.DEFAULT_RESERVE_MINUTES > week.TIMEOUT_SECONDS / 60.0
    assert week.DEFAULT_RESERVE_MINUTES == pytest.approx(
        week.TIMEOUT_SECONDS / 60.0 + week.RESERVE_MARGIN_MINUTES
    )


def test_with_no_probe_row_the_slot_reserves_that_default(tmp_path, monkeypatch):
    """`reserve_minutes()` is the declared default until a probe measures one."""
    import ai_jobs.week_review_narration as week
    from ai_jobs import model_probe

    monkeypatch.setattr(
        model_probe, "reserve_minutes_from_probe", lambda **_kwargs: None
    )
    assert week.reserve_minutes() == week.DEFAULT_RESERVE_MINUTES

    # A MEASURED probe still wins, whatever the default says.
    monkeypatch.setattr(
        model_probe, "reserve_minutes_from_probe", lambda **_kwargs: 12.5
    )
    assert week.reserve_minutes() == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# advisory 2 - a day the desk could not read is not a quiet day
# ---------------------------------------------------------------------------
def _unreadable_card(session: str, monkeypatch):
    """A REAL card whose `did_well` line went through `_unreadable_line`.

    Built by making the line's own owner raise, so the marker comes from
    `day_report_card._guarded` exactly as it does on the desk - not from a dict
    typed here to look like one.
    """
    import day_report_card

    def _boom(*_args, **_kwargs):
        raise RuntimeError("the walk-away tables were unreadable")

    monkeypatch.setattr(day_report_card, "did_well_line", _boom)
    try:
        return day_report_card.build(fx.day_inputs_for(session))
    finally:
        monkeypatch.undo()


def test_an_unreadable_day_is_named_on_the_pooled_line(monkeypatch):
    """Hand-counted over `tj5_support.CARD_COUNTS`.

    2026-09-15 `did_well` is (2, 2, 1) and 2026-09-18 is (3, 3, 2), so the two
    readable days pool to n 5, measured 5, runs 3. 2026-09-14's `did_well` line
    could not be built at all: it adds NOTHING to any count - it never measured
    anything - and it is NAMED, because a cell that dropped the marker is
    indistinguishable from a day on which the trader simply did nothing.
    """
    import day_report_card

    bad = _unreadable_card("2026-09-14", monkeypatch)
    good = [fx.card_for("2026-09-15"), fx.card_for("2026-09-18")]

    assert _lines(bad)["did_well"]["measured_ok"] is False

    week = day_report_card.week_from_cards([bad, *good])
    line = _lines(week)["did_well"]
    assert (line["n"], line["measured"], line["runs"]) == (5, 5, 3)
    assert line["unreadable_sessions"] == ("2026-09-14",)
    assert "1 day(s) could not be read" in line["text"]
    assert "2026-09-14" in line["text"]

    # The counts are UNCHANGED by the marker: an unreadable day adds nothing.
    without = day_report_card.week_from_cards(good)
    other = _lines(without)["did_well"]
    assert (other["n"], other["measured"], other["runs"]) == (5, 5, 3)


def test_a_week_with_nothing_unreadable_says_nothing_about_it():
    """The key is PRESENT and EMPTY, and the text is untouched.

    A reader has one shape either way; an absent key would be a third state,
    and a sentence about zero unreadable days is noise on every normal week.
    """
    import day_report_card

    week = day_report_card.week_from_cards(
        [fx.card_for(session) for session in fx.PACKED_SESSIONS]
    )
    for line in week.lines:
        assert line["unreadable_sessions"] == ()
        assert "could not be read" not in line["text"], line["key"]


def test_the_same_marker_reaches_the_walkaway_week_path(monkeypatch):
    """`week(sessions)` and `week_from_cards` share ONE pooling, so both carry it."""
    import day_report_card

    def _boom(*_args, **_kwargs):
        raise RuntimeError("the walk-away tables were unreadable")

    monkeypatch.setattr(day_report_card, "missed_line", _boom)
    week = day_report_card.week(
        [fx.day_inputs_for(session) for session in ("2026-09-15", "2026-09-18")]
    )
    line = _lines(week)["missed"]
    assert line["unreadable_sessions"] == ("2026-09-15", "2026-09-18")
    assert "2 day(s) could not be read" in line["text"]
    assert (line["n"], line["measured"]) == (0, 0)


def test_an_empty_week_still_carries_the_key():
    import day_report_card

    for line in day_report_card.week([]).lines:
        assert line["unreadable_sessions"] == ()


# ---------------------------------------------------------------------------
# advisory 6 - EXACTLY one model call on the narrated path
# ---------------------------------------------------------------------------
def test_the_narrated_week_makes_exactly_one_model_call(tmp_path, monkeypatch):
    """`plan.md` TJ-5 change 2: *"One call per week."*

    The tester's neighbour test asserts `<= 1`, which a build that called
    NOTHING would also pass. This pins the other side: a week with three
    narrated days publishes a story, and it costs exactly one call - not two,
    and not zero.
    """
    import project_paths

    import ai_jobs.week_review_narration as week

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    fx.write_week(root, narrated=fx.PACKED_SESSIONS)

    inputs = week.build_week_inputs(fx.FRIDAY, root=root)
    tally = inputs["were_you_right"]
    reply = {
        "headline": "A week you were right more than wrong.",
        "what_happened": "Three of five sessions have facts.",
        "were_you_right": {
            "right": tally["right"],
            "wrong": tally["wrong"],
            "unresolved": tally["unresolved"],
            "examples": [],
        },
        "chased": [],
        "tendencies": [],
        "process_pattern": "You waited for the second test.",
        "next_week_watch": [],
        "sources": list(inputs["allowed_source_ids"])[:1],
    }
    calls: list[dict] = []

    def _request(**kwargs):
        calls.append(dict(kwargs))
        return {"summary": reply, "model": "gemma3:27b"}

    outcome = week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=root,
        request=_request,
    )
    assert outcome["status"] == "ok", outcome
    assert len(calls) == 1, f"the week story made {len(calls)} model calls"
    assert week.read_week_narration(fx.WEEK_ID, root=root)["narration"]["headline"]


# ---------------------------------------------------------------------------
# advisories 4 and 5 - what the page SAYS about a day that does not exist,
# and about a week nothing was packed in
# ---------------------------------------------------------------------------
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_a_day_that_does_not_exist_says_only_that():
    """A holiday week has four sessions, and the fifth card is not a MISSED day.

    It read "no session this exchange week: not packed - the desk has no facts
    for this day" - a double negative about a day nobody was ever going to
    measure.
    """
    from ui.panels.weekend_prep_panel import _WeekDayCard

    card = _WeekDayCard()
    try:
        card.show_row({})
        assert card.session == ""
        assert card.heading.text() == "no session this exchange week"
        assert card.headline.text() == ""
        assert "not packed" not in card.headline.text()
        assert card.tally.text() == ""
        # A REAL day with no pack still says what it is.
        card.show_row({"session": "2026-09-14", "has_facts": False})
        assert "not packed" in card.headline.text()
        assert "0" not in card.tally.text()
    finally:
        card.deleteLater()


def test_a_week_with_nothing_packed_prints_no_walkaway_totals():
    """Five zeros with their qualifier UNDER them read as a week of doing nothing.

    With no packed session there is nothing to total, so the sentence is the
    whole answer; with packed sessions the qualifier LEADS, because it is what
    makes the numbers under it honest.
    """
    from ui.panels.weekend_prep_panel import WeekReviewPage
    from ui.services.weekend_prep_service import WeekendPrepService, empty_week_payload

    service = WeekendPrepService(now=fx.SATURDAY_AFTERNOON.replace(tzinfo=None))
    page = WeekReviewPage(service)
    try:
        empty = empty_week_payload(fx.WEEK_ID)
        empty["walkaway"] = {
            "counts": {"liked_not_traded": 0, "rejected": 0},
            "sessions": 0,
            "n": 0,
        }
        text = "\n".join(page._summary_lines(empty, []))
        assert "no session packed yet" in text
        assert "liked not traded: 0" not in text

        packed = dict(empty)
        packed["walkaway"] = {
            "counts": {"liked_not_traded": 9, "rejected": 13},
            "sessions": 3,
            "n": 22,
        }
        lines = page._summary_lines(packed, [])
        heading = next(line for line in lines if line.startswith("WALK-AWAY"))
        assert "3 packed session(s)" in heading and "n 22" in heading
        assert lines.index(heading) < lines.index("  liked not traded: 9")
    finally:
        page.shutdown()
        page.deleteLater()
        service.shutdown()


def test_the_strip_cell_says_when_a_day_could_not_be_read():
    """The marker has to reach the surface the trader actually looks at."""
    from ui.panels.weekend_prep_panel import week_strip_cell

    clean = {"n": 9, "measured": 8, "rate": None, "unreadable_sessions": ()}
    assert "could not be read" not in week_strip_cell(clean)

    hurt = dict(clean, unreadable_sessions=("2026-09-14", "2026-09-15"))
    assert "2 day(s) could not be read" in week_strip_cell(hurt)
