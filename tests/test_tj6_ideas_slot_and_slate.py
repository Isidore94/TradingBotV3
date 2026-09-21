r"""TJ-6 - WHERE the ideas slot runs, and when it refuses. RED.

`plan.md` §12.4 TJ-6 change 1: *"Slot `improvement_ideas`
(`scripts/ai_jobs/improvement_ideas.py`), Stage 3, appended LAST, local medium
model, `reserve_minutes=10`, `max_attempts=2`"*; §12.4 TJ-13 item 9 lists it as
the one Stage 3 slot this program adds; decision 0018 and its 2026-09-20
addendum own the stage order.

**NO MODEL IS EVER CALLED HERE.** Every slate is built with `dataclasses.replace`
spies and every run is handed a `request=` of this module's own.

VERIFIED ON THIS BRANCH (1b9d77e0): `runner.default_slots()` ends at
`setup_research` and no slot named `improvement_ideas` exists, so every test in
this file fails on the missing slot or the missing module.

THE PINS THE BUILDER MUST UPDATE (each would otherwise fail on the next run)
---------------------------------------------------------------------------
Appending one slot at the END of stage 3 touches exactly four other files,
measured on this branch:

* `tests/test_ai_jobs_runner.py` - ``EXPECTED_SLOT_ORDER`` (line 376 is
  ``"setup_research",``): append ``"improvement_ideas",`` after it.
* `tests/test_veto_cohort_grading.py` - the byte-pinned slate ending at line 664
  ``"setup_research",``: append the same name.
* `tests/test_tj13b_local_large_provider.py` line 421 - the ``expected``
  weeknight slate: append the same name.
* `tests/test_tj13b_probe_guards.py` line 619 - the e8c04f88 weeknight guard:
  add ``"improvement_ideas"`` to its ``set_aside`` tuple (line 640-643), NOT to
  ``pinned_at_e8c04f88``; that guard is about the 2026-08 set and a new slot is
  pinned where it sits and then set aside, exactly as TJ-4, TJ-10 and TJ-16 did.

`tests/test_ws_rp_shared_report.py` imports ``EXPECTED_SLOT_ORDER`` from the
runner test file, so it needs no edit; `tests/test_ws_10d_market_story.py`,
`tests/test_tj15_miss_contrast_slot.py`, `tests/test_tj16_prediction_contrast_slot.py`
and `tests/test_opt_in_evidence_scopes.py` assert only RELATIVE positions above
stage 3 and need none either. `tests/test_tj5_week_slot_and_slate.py` pins
``observation_tags`` -> ``week_review_narration`` -> ``ticker_briefs``, which a
stage 3 append does not touch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj6_support as fx  # noqa: E402

SLOT = "improvement_ideas"

#: The live window, in the trader's own settings (01:00-09:00 ET).
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


# ---------------------------------------------------------------------------
# where it sits
# ---------------------------------------------------------------------------
def test_the_ideas_slot_is_the_last_slot_of_stage_three():
    """*"Stage 3, appended LAST"*. Nothing in the night runs after it.

    It reads what every other slot wrote - the packs, the narrations, the week's
    totals - and feeds nothing, so last is the only place it can be.
    """
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    assert SLOT in names, f"{SLOT} is not a registered slot: {names[-3:]}"
    assert names[-1] == SLOT, names[-3:]
    assert names.index(SLOT) > names.index("setup_research")
    # Stage 3 means AFTER the slot that closes stage 1 - which is also what
    # keeps it off the Sunday deterministic slate.
    assert names.index(SLOT) > names.index(runner._STAGE_ONE_LAST_SLOT)


def test_the_runner_order_pin_names_the_ideas_slot_too():
    """The one file the whole house reads for slot order must agree.

    `tests/test_ai_jobs_runner.py` holds ``EXPECTED_SLOT_ORDER`` and
    `tests/test_ws_rp_shared_report.py` imports it rather than re-typing it. A
    new slot that is not in that tuple turns the runner file red on the next
    full suite, which is the 2026-09-20 integration lesson TJ-15 learned.
    """
    from ai_jobs import runner

    spec = importlib.util.spec_from_file_location(
        "_tj6_runner_pin", ROOT_DIR / "tests" / "test_ai_jobs_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = tuple(module.EXPECTED_SLOT_ORDER)

    assert expected[-1] == SLOT, expected[-3:]
    assert tuple(slot.name for slot in runner.default_slots()) == expected


def test_the_ideas_slot_declares_the_numbers_the_packet_gives_it():
    """`reserve_minutes=10`, `max_attempts=2`, and a model it declares honestly.

    It loads the local MEDIUM model, so `uses_model` is True and a forced
    daytime run may not buy it the clock. There is no half of an IDEA that runs
    without a model, so no `model_free_kwargs`.
    """
    from ai_jobs import runner

    slot = next(slot for slot in runner.default_slots() if slot.name == SLOT)
    assert slot.reserve_minutes == 10.0
    assert slot.max_attempts == 2
    assert slot.uses_model is True
    assert slot.model_free_kwargs is None
    assert slot.enabled is True
    assert str(slot.description).strip()


def test_the_ideas_slot_runs_on_a_weeknight_and_on_saturday():
    """Up to three ideas A NIGHT (TJ-6 change 1) - not once a week.

    `WEEKEND_ONLY_SLOTS` is the seam that takes a slot off the weeknight slate;
    this one must not be in it.
    """
    from ai_jobs import runner

    assert SLOT not in runner.WEEKEND_ONLY_SLOTS
    assert SLOT in [slot.name for slot in runner.slots_for(runner.NIGHT_WEEKNIGHT)]
    assert SLOT in [slot.name for slot in runner.slots_for(runner.NIGHT_SATURDAY)]


def test_sunday_offers_the_ideas_slot_only_when_the_weekend_left_it_unfinished(tmp_path):
    """Sunday is the BACKLOG night, not a second slate.

    Ledger written by hand: one `failed` attempt on Friday's session date, which
    is what both weekend nights key to.
    """
    import tj5_support as tj5
    from ai_jobs import ledger, runner

    empty = tmp_path / "empty.jsonl"
    tj5.write_ledger(empty, [])
    quiet = [
        slot.name
        for slot in runner.slots_for(
            runner.NIGHT_SUNDAY, session_date=fx.SESSION, ledger_path=empty
        )
    ]
    assert SLOT not in quiet, "a slot nobody attempted is not owed"

    tried = tmp_path / "tried.jsonl"
    tj5.write_ledger(
        tried,
        [tj5.ledger_row(SLOT, fx.SESSION, ledger.STATUS_FAILED, reason="no model answered")],
    )
    owed = [
        slot.name
        for slot in runner.slots_for(
            runner.NIGHT_SUNDAY, session_date=fx.SESSION, ledger_path=tried
        )
    ]
    assert SLOT in owed, owed


# ---------------------------------------------------------------------------
# the night-only rule, from inside
# ---------------------------------------------------------------------------
def test_a_forced_daytime_ideas_run_records_skipped_and_calls_nothing(tmp_path, monkeypatch):
    """Friday 14:00 ET with `--force`: TJ-13A item 1's whole point.

    `--force` buys the attempt caps and the already-done check, never the clock,
    for a slot that starts local inference.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    calls: list[tuple[str, dict]] = []
    led = tmp_path / "ledger.jsonl"

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            runner.run_slots(
                _slate_with_spies(calls),
                now=fx.FRIDAY_AFTERNOON,
                force=True,
                only=SLOT,
                ledger_path=led,
            )

    assert [name for name, _kwargs in calls] == [], "a model slot ran by day"
    rows = [row for row in _rows(led) if row.get("job") == SLOT]
    assert rows, "the refusal was not recorded"
    assert rows[-1]["status"] == ledger.STATUS_SKIPPED
    assert "window" in rows[-1]["reason"].lower()


def test_the_ideas_slot_makes_at_most_one_model_call_per_night(tmp_path, monkeypatch):
    """One night, one ask. There is nothing to sweep.

    TJ-4's round-2 blocker was a slot still loading a model past the window's
    end because it swept extra sessions.
    """
    import project_paths
    from ai_jobs import improvement_ideas

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    monkeypatch.setattr(
        project_paths, "AI_IDEAS_FILE", tmp_path / "ai_ideas.jsonl", raising=False
    )
    monkeypatch.setattr(
        project_paths, "AI_IDEAS_STATE_FILE", tmp_path / "ai_ideas_state.json", raising=False
    )
    fx.write_week(root, narrated=fx.PACKED_SESSIONS)

    calls: list[dict] = []
    improvement_ideas.run_improvement_ideas(
        session_date=fx.SESSION,
        now=fx.WEEKNIGHT,
        root=root,
        request=fx.fake_request(fx.reply([]), calls=calls),
    )
    assert len(calls) <= 1, f"the ideas slot made {len(calls)} model calls"


def test_the_slot_never_raises_and_always_names_a_status(tmp_path, monkeypatch):
    """A crash in the last slot of the night is a lost night, never an exception.

    The provider raises here; the slot must answer with a recorded status and a
    sentence, the way every sibling slot does.
    """
    import project_paths
    from ai_jobs import improvement_ideas, ledger

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    monkeypatch.setattr(
        project_paths, "AI_IDEAS_FILE", tmp_path / "ai_ideas.jsonl", raising=False
    )
    monkeypatch.setattr(
        project_paths, "AI_IDEAS_STATE_FILE", tmp_path / "ai_ideas_state.json", raising=False
    )
    fx.write_week(root, narrated=fx.PACKED_SESSIONS)

    def _boom(**_kwargs):
        raise RuntimeError("the endpoint refused")

    outcome = improvement_ideas.run_improvement_ideas(
        session_date=fx.SESSION, now=fx.WEEKNIGHT, root=root, request=_boom
    )
    assert isinstance(outcome, dict)
    assert outcome.get("status") in ledger.RECOGNISED_JOB_STATUSES
    assert outcome.get("status") != ledger.STATUS_OK
    assert str(outcome.get("reason") or "").strip()
