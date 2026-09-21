r"""TJ-6 - the ideas travel in the ONE payload each page already reads. RED.

`plan.md` §12.4 TJ-6 change 3: the card is on Day Review AND Week Review - the
night's ideas with Keep / Dismiss, and on Week Review the kept ones with their
before and after.

Both pages have ONE payload built on ONE worker (TJ-1 and TJ-5's rule), so the
ideas are READ THERE and never on the Qt thread:

* `ui/services/weekend_prep_service.py` already RESERVES the key -
  ``WEEK_PAYLOAD_KEYS`` line 498-500 on this branch says *"Reserved for TJ-6's
  kept ideas. The key exists so the page has one shape before that packet lands;
  nothing writes it here."* This file is what makes something write it.
* `ui/services/day_review_service.py` ``PAYLOAD_KEYS`` (line 45) does NOT carry
  an ideas key yet - verified on this branch - so TJ-6 adds one, and
  `empty_payload` carries it too, because that shape IS the first paint.

VERIFIED ON THIS BRANCH (1b9d77e0): `read_week_review` returns ``ideas: []``
always and `day_review_service.PAYLOAD_KEYS` has no ideas key, so the first test
fails on an empty list where a row is owed and the day tests fail on the missing
key.

**NO MODEL IS EVER CALLED HERE**, and no Qt object is built: this is the worker
side of both pages.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj6_support as fx  # noqa: E402

TEXT = "Wait for the second test before sizing up."
NOW = datetime(2026, 9, 19, 8, 0)


@pytest.fixture
def night(tmp_path, monkeypatch):
    return fx.install_stores(monkeypatch, tmp_path)


def _keep_a_process_idea(night, monkeypatch, *, reading=None):
    from ai_jobs import improvement_ideas

    name = improvement_ideas.MEASURABLES[0].name
    row = fx.stored_idea_row(TEXT, session=fx.SESSION, measurable=name)
    fx.write_ideas(night["ideas"], [row])
    body = dict(reading or fx.BASELINE)
    monkeypatch.setattr(
        improvement_ideas,
        "measure",
        lambda measurable, **kwargs: {
            "measurable": measurable,
            "value": body["value"],
            "n": body["n"],
            "measured": True,
            "window_sessions": fx.lately_sessions(),
        },
    )
    improvement_ideas.keep_idea(row["idea_id"], end_session=fx.SESSION)
    return row


# ---------------------------------------------------------------------------
# Week Review
# ---------------------------------------------------------------------------
def test_a_week_with_no_ideas_carries_an_empty_list_and_says_nothing_else(night):
    """GREEN GUARD - this one PASSES on this branch, deliberately.

    The state the trader is in today: no ideas store at all. TJ-5 already
    reserved the key and leaves it empty, and that is the right answer for an
    empty store. The guard is here so the packet cannot fill the key with a
    zero, a placeholder or a sentence when there is nothing to show.

    Hand-counted: zero ideas -> zero rows, and the key is still there, because
    the page has ONE render and no special cases.
    """
    from ui.services import weekend_prep_service as prep

    payload = prep.read_week_review(friday=fx.SESSION, root=night["root"])
    assert "ideas" in prep.WEEK_PAYLOAD_KEYS
    assert list(payload["ideas"]) == []


def test_the_weeks_kept_ideas_arrive_with_their_before_and_after(night, monkeypatch):
    """Hand-counted: one kept `process` idea -> one row, before 0.30 (n 34),
    after 0.48 (n 41), and the date the trader kept it."""
    from ai_jobs import improvement_ideas
    from ui.services import weekend_prep_service as prep

    row = _keep_a_process_idea(night, monkeypatch)
    monkeypatch.setattr(
        improvement_ideas,
        "measure",
        lambda measurable, **kwargs: {
            "measurable": measurable,
            "value": fx.AFTER["value"],
            "n": fx.AFTER["n"],
            "measured": True,
            "window_sessions": fx.lately_sessions(),
        },
    )

    payload = prep.read_week_review(friday=fx.SESSION, root=night["root"])
    ideas = list(payload["ideas"])
    assert len(ideas) == 1, ideas
    item = ideas[0]
    assert item["idea_id"] == row["idea_id"]
    assert item["text"] == TEXT
    assert item["before"]["value"] == fx.BASELINE["value"]
    assert item["before"]["n"] == fx.BASELINE["n"]
    assert item["after"]["value"] == fx.AFTER["value"]
    assert item["after"]["n"] == fx.AFTER["n"]
    assert item["kept_at"]


def test_the_week_page_reads_its_ideas_exactly_once(night, monkeypatch):
    """ONE payload, ONE worker, ONE read. A page that asked twice would open the
    store twice per tab click - and the second read is the one that lands on the
    Qt thread when somebody "just needs the count"."""
    from ai_jobs import improvement_ideas
    from ui.services import weekend_prep_service as prep

    _keep_a_process_idea(night, monkeypatch)
    calls: list[dict] = []
    real = improvement_ideas.checked_ideas

    def _counted(**kwargs):
        calls.append(dict(kwargs))
        return real(**kwargs)

    monkeypatch.setattr(improvement_ideas, "checked_ideas", _counted)
    prep.read_week_review(friday=fx.SESSION, root=night["root"])
    assert len(calls) == 1, f"the week page read the ideas {len(calls)} times"


def test_an_unreadable_ideas_store_costs_the_ideas_and_not_the_week(night, monkeypatch):
    """One unreadable store costs ONE section. The five day cards, the story and
    the strip are a different question (`weekend_prep_service._week_callouts`'s
    own rule)."""
    from ai_jobs import improvement_ideas
    from ui.services import weekend_prep_service as prep

    def _boom(**_kwargs):
        raise RuntimeError("the ideas store could not be read")

    monkeypatch.setattr(improvement_ideas, "checked_ideas", _boom)
    payload = prep.read_week_review(friday=fx.SESSION, root=night["root"])
    assert sorted(payload) == sorted(prep.WEEK_PAYLOAD_KEYS)
    assert list(payload["ideas"]) == []
    assert list(payload["sessions"]), "the rest of the week was lost with the ideas"


# ---------------------------------------------------------------------------
# Day Review
# ---------------------------------------------------------------------------
def test_the_day_payload_declares_an_ideas_key_and_a_first_paint_for_it():
    """`empty_payload` IS the first paint, a failed read and a quiet session."""
    from ui.services import day_review_service as day

    assert "ideas" in day.PAYLOAD_KEYS
    assert day.empty_payload("2026-09-18")["ideas"] == []


def test_the_day_read_carries_that_sessions_ideas(night, monkeypatch):
    """Hand-counted: the store holds two ideas for 2026-09-18 and one for the
    session before it, so the day's card gets exactly two.

    Read on the WORKER with everything else, never fetched by the card.
    """
    from ai_jobs import improvement_ideas
    from ui.services.day_review_service import DayReviewService

    fx.write_ideas(
        night["ideas"],
        [
            fx.stored_idea_row("One.", session=fx.SESSION),
            fx.stored_idea_row("Two.", session=fx.SESSION),
            fx.stored_idea_row("Three.", session=fx.sessions_back(fx.SESSION, 1)),
        ],
    )

    class _Journal:
        def entries_about(self, _session):
            return []

        def daily_story(self, _session):
            return None

        def theses_for(self, _session):
            return []

    service = DayReviewService(journal_service=_Journal())
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])

    payload = service.read_day(fx.SESSION, now=NOW)
    texts = sorted(str(row["text"]) for row in payload["ideas"])
    assert texts == ["One.", "Two."], payload.get("error")
    assert all(row["status"] == "" for row in payload["ideas"]), "nothing is decided yet"

    # And the same rows come back out of the store's own reader, so the page and
    # the store cannot disagree about what tonight offered.
    direct = improvement_ideas.ideas_for_session(fx.SESSION)
    assert sorted(str(row["text"]) for row in direct) == ["One.", "Two."]


def test_a_dismissed_idea_is_not_offered_to_the_day_card_again(night, monkeypatch):
    """*"a dismissed idea never returns"* - including to the card it was
    dismissed on, across a restart.

    Hand-counted: 2 stored for the session, 1 dismissed -> 1 offered.
    """
    from ui.services.day_review_service import DayReviewService

    first = fx.stored_idea_row("One.", session=fx.SESSION)
    second = fx.stored_idea_row("Two.", session=fx.SESSION)
    fx.write_ideas(night["ideas"], [first, second])
    fx.write_state(night["state"], {second["idea_id"]: fx.dismissed_record(second["idea_id"])})

    class _Journal:
        def entries_about(self, _session):
            return []

        def daily_story(self, _session):
            return None

        def theses_for(self, _session):
            return []

    service = DayReviewService(journal_service=_Journal())
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])

    payload = service.read_day(fx.SESSION, now=NOW)
    assert [str(row["text"]) for row in payload["ideas"]] == ["One."]
