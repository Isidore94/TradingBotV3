r"""TJ-5 - WHERE the week story runs, and on which model. RED.

Packet `.claude/packets/TJ-5-6-7-13B.md` "TJ-5"; `plan.md` §12.4 TJ-5 change 2
with its **AMENDED 2026-09-19** block and §12.4 TJ-13 items 5, 6, 7 and 9;
decision 0021 answers 19 and 20; decision 0018 and its 2026-09-20 addendum.

**NO MODEL IS EVER CALLED HERE.** Every slate is built with `dataclasses.replace`
spies, the provider seam is handed a `post` of this module's own, and the two
tests that reach `ai_summary` monkeypatch it to something that records and
returns.

The contract these tests pin (the builder may ADD, never remove)
----------------------------------------------------------------

``scripts/ai_jobs/runner.py``:

* ``default_slots()`` gains ONE slot, ``week_review_narration``, inside decision
  0018's stage 2, DIRECTLY after ``observation_tags`` and DIRECTLY before
  ``ticker_briefs``. That is the position `plan.md` §12.4 TJ-13 item 9 lists
  for it ("Stage 2 `day_review_narration` (TJ-4), `observation_tags` (TJ-16),
  `week_review_narration` (TJ-5, Saturday night)"), and it keeps the slot ahead
  of the two hours `ticker_briefs` reserves.
* It declares ``uses_model=True`` (it loads a 27B), ``model_free_kwargs is None``
  (there is no week story without a model; the deterministic week/month strip is
  the PAGE's, not this slot's) and a real ``max_attempts``.
* It runs on the SATURDAY slate only. `runner.WEEKEND_ONLY_SLOTS` is the seam
  that already takes `ai_summary` off the weeknight slate; whatever the builder
  uses, `slots_for("weeknight")` must not offer it.

``scripts/ai_jobs/week_review_narration.py``:

    run_week_review_narration(*, session_date="", now=None, root=None,
                              request=None, ledger_path=None, **_ignored)
        -> {"status", "model", "reason", "outputs", "model_attribution"}

The row the slot writes carries the attribution under
`ai_jobs.provider.LEDGER_FIELD`, because the live ledger's ``model`` column is
one string and a night the large model was never attempted is otherwise
indistinguishable from a night it was and fell back (TJ-13B).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as fx  # noqa: E402

SLOT = "week_review_narration"
EASTERN = ZoneInfo("America/New_York")

#: The live window, in the trader's own settings (01:00-09:00 ET = 22:00-06:00
#: Pacific). Named rather than read so a desk setting cannot move a test.
LIVE_START = "01:00"
LIVE_END = "09:00"


def _settings(**values):
    from ai_jobs import store

    return mock.patch.object(
        store._paths(),
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


def _no_session_block(monkeypatch):
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")


def _frozen_clock(monkeypatch, moment):
    from ai_jobs import window

    real = window.market_now
    monkeypatch.setattr(window, "market_now", lambda now=None: real(moment))


def _rows(path):
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _slate_with_spies(calls):
    from ai_jobs import runner

    def _record(name):
        def _run(**kwargs):
            calls.append((name, dict(kwargs)))
            return {"reason": f"{name} ran"}

        return _run

    return [replace(slot, run=_record(slot.name)) for slot in runner.default_slots()]


@pytest.fixture
def day_review_root(tmp_path, monkeypatch):
    """A scratch `DAY_REVIEW_DIR` holding the three packs the live store has."""
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    fx.write_week(root, narrated=fx.PACKED_SESSIONS)
    return root


# ---------------------------------------------------------------------------
# where the slot sits
# ---------------------------------------------------------------------------
def test_the_week_story_sits_in_stage_two_between_the_tags_and_the_briefs():
    """One new slot, in the position `plan.md` TJ-13 item 9 lists for it.

    Ahead of `ticker_briefs` because the briefs reserve 120 minutes and the 27B
    needs the front of the night; after `observation_tags` because
    `measured_report`/`ai_summary` must stay adjacent and two other files pin
    that pair.
    """
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    assert SLOT in names, f"{SLOT} is not a registered slot: {names}"
    assert names[names.index("observation_tags") + 1] == SLOT
    assert names[names.index(SLOT) + 1] == "ticker_briefs"
    # It is in stage 2, which means AFTER the slot that closes stage 1.
    assert names.index(SLOT) > names.index(runner._STAGE_ONE_LAST_SLOT)


def test_the_week_story_slot_says_it_loads_a_model_and_has_no_deterministic_half():
    """`--force` may not buy it the daytime clock, and there is no half of the
    week STORY that runs without a model - the deterministic week and month
    strip is TJ-5 change 3, which lives on the page and calls nothing."""
    from ai_jobs import runner

    slot = next(slot for slot in runner.default_slots() if slot.name == SLOT)
    assert slot.uses_model is True
    assert slot.model_free_kwargs is None
    assert slot.enabled is True
    # plan.md §12.3: every slot sets `max_attempts`, never 0.
    assert slot.max_attempts and slot.max_attempts <= 3
    assert slot.reserve_minutes > 0


def test_the_week_story_is_never_offered_on_a_weeknight():
    """It is a weekly job on the 27B. A weeknight slate that held it would put
    the heaviest model load the desk owns on a night that has a session behind
    it and another in front."""
    from ai_jobs import runner

    # Named first, so this cannot pass by the slot not existing at all.
    assert SLOT in [slot.name for slot in runner.default_slots()]
    weeknight = [slot.name for slot in runner.slots_for(runner.NIGHT_WEEKNIGHT)]
    assert SLOT not in weeknight, weeknight


def test_the_week_story_is_on_the_saturday_slate():
    from ai_jobs import runner

    saturday = [slot.name for slot in runner.slots_for(runner.NIGHT_SATURDAY)]
    assert SLOT in saturday, saturday


def test_sunday_offers_the_week_story_only_when_saturday_left_it_unfinished(tmp_path):
    """Sunday is the BACKLOG night, not a second weekly slate.

    Ledger written by hand: one `failed` attempt on Friday's session date, which
    is what both weekend nights key to. A slot that never ran is not owed.
    """
    from ai_jobs import ledger, runner

    empty = tmp_path / "empty.jsonl"
    fx.write_ledger(empty, [])
    quiet = [
        slot.name
        for slot in runner.slots_for(
            runner.NIGHT_SUNDAY, session_date=fx.FRIDAY, ledger_path=empty
        )
    ]
    assert SLOT not in quiet, "a slot nobody attempted is not owed"

    tried = tmp_path / "tried.jsonl"
    fx.write_ledger(
        tried,
        [fx.ledger_row(SLOT, fx.FRIDAY, ledger.STATUS_FAILED, reason="the 27B timed out")],
    )
    owed = [
        slot.name
        for slot in runner.slots_for(
            runner.NIGHT_SUNDAY, session_date=fx.FRIDAY, ledger_path=tried
        )
    ]
    assert SLOT in owed, owed


# ---------------------------------------------------------------------------
# the night-only rule, from inside
# ---------------------------------------------------------------------------
def test_a_forced_daytime_week_story_records_skipped_and_calls_nothing(tmp_path, monkeypatch):
    """Saturday 14:00 ET with `--force`: the one moment TJ-13A item 1 is about.

    The trader is at the desk all Saturday afternoon. A 14 GB model load in
    front of them is exactly what the rule forbids, and `--force` buys the
    attempt caps and the already-done check, never the clock.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    calls: list[tuple[str, dict]] = []
    led = tmp_path / "ledger.jsonl"

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            runner.run_slots(
                _slate_with_spies(calls),
                now=fx.SATURDAY_AFTERNOON,
                force=True,
                only=SLOT,
                ledger_path=led,
            )

    assert [name for name, _kwargs in calls] == [], "a model slot ran by day"
    rows = [row for row in _rows(led) if row.get("job") == SLOT]
    assert rows, "the refusal was not recorded"
    assert rows[-1]["status"] == ledger.STATUS_SKIPPED
    assert "window" in rows[-1]["reason"].lower()


def test_the_week_story_makes_at_most_one_model_call_per_run(day_review_root, monkeypatch):
    """`plan.md` TJ-5 change 2: "One call per week."

    TJ-4's round-2 blocker was a slot that swept extra sessions and was still
    loading a model past the window's end. The week story's answer to that rule
    is that there is nothing to sweep: one week, one call.
    """
    import ai_jobs.week_review_narration as week

    calls: list[dict] = []

    def _request(**kwargs):
        calls.append(dict(kwargs))
        return {"summary": _reply(day_review_root), "model": "gemma3:27b"}

    week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=day_review_root,
        request=_request,
    )
    assert len(calls) <= 1, f"the week story made {len(calls)} model calls"


def _reply(root):
    """A narration that PASSES every check, built from the inputs themselves."""
    import ai_jobs.week_review_narration as week

    inputs = week.build_week_inputs(fx.FRIDAY, root=root)
    tally = inputs["were_you_right"]
    allowed = list(inputs["allowed_source_ids"])
    return {
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
        "sources": allowed[:1],
    }


# ---------------------------------------------------------------------------
# which model, and what the record says afterwards
# ---------------------------------------------------------------------------
def test_with_no_probe_row_the_week_story_still_runs_on_the_medium_model(
    day_review_root, tmp_path, monkeypatch
):
    """Lead decision 1, 2026-09-19: the trader wants a week story every Saturday.

    The 27B has never run on this desk (TJ-13B's measurement is owed), so
    `provider.week_review_plan` answers `may_run_large: False`. That must cost
    the LARGE MODEL, never the story - and the reason must reach the record.
    """
    from ai_jobs import provider

    import ai_jobs.week_review_narration as week

    empty = tmp_path / "ledger.jsonl"
    fx.write_ledger(empty, [])
    plan = provider.week_review_plan(ledger_path=empty)
    assert plan["may_run_large"] is False
    medium = plan["model"]
    assert medium, "the plan named no model to fall back to"

    asked: list[str] = []

    def _request(**kwargs):
        asked.append(str(kwargs.get("model") or ""))
        return {"summary": _reply(day_review_root), "model": medium}

    outcome = week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=day_review_root,
        request=_request,
        ledger_path=empty,
    )
    attribution = outcome.get("model_attribution") or {}
    assert attribution, "the run recorded no model attribution"
    assert attribution.get("model_answered") == medium
    assert "measured" in str(attribution.get("fallback_reason") or plan["reason"]).lower()


def test_the_openai_setting_makes_the_week_story_a_failed_row_never_a_crash(
    day_review_root, tmp_path
):
    """`plan.md` TJ-13 item 7: "`request_with_fallback` RAISES for it - TJ-5's
    slot must catch that and record a FAILED row."

    A ValueError out of the seam is a sentence in the ledger, not a lost night.
    """
    from ai_jobs import ledger, provider

    import ai_jobs.week_review_narration as week

    empty = tmp_path / "ledger.jsonl"
    fx.write_ledger(empty, [])

    with _settings(**{provider.WEEK_REVIEW_PROVIDER_SETTING: provider.OPENAI}):
        outcome = week.run_week_review_narration(
            session_date=fx.FRIDAY,
            now=fx.SATURDAY_NIGHT,
            root=day_review_root,
            ledger_path=empty,
        )

    assert outcome["status"] == ledger.STATUS_FAILED, outcome
    assert "openai" in outcome["reason"].lower()
    assert outcome["outputs"] == [], "a refused provider published something"
