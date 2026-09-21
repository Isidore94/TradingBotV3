r"""TJ-6 - ONE ask a night, through the REAL runner. RED before review 1's fix.

Review 1's blocker, reproduced by the reviewer on 67b9bf54: a night that called
the model and then answered `ledger.STATUS_SKIPPED` with no artifact re-asked on
every one of the scheduled task's passes. `skipped` is in neither
`ledger.CANONICAL_COMPLETION_STATUSES` nor `ledger.ATTEMPT_STATUSES`, so neither
`run_slots`' already-done check nor `max_attempts=2` ever bit, and the task
repeats every 30 minutes for eight hours (`register_ai_jobs_task.ps1`): up to
SIXTEEN local-model calls and sixteen ledger rows for a night that writes
nothing. The live home folder holds ZERO `pack.json`, so that is the first thing
that would happen after the merge.

The lead's rule, in two halves:

* **Nothing to cite -> no model call.** A window with no packs carries no id an
  idea could cite, so the slot answers `skipped` BEFORE any model load and says
  so. Repeating that every pass costs a ledger row and some file reads.
* **Asked once is done.** A run that DID ask and stored nothing ends `ok` - it
  is a finished night - so the runner's own already-done check stops every later
  pass, and the slot leaves an ASKED marker beside its store so its own
  unchanged-hash skip arms as well. A whole REJECTION stays `failed` and is
  capped by `max_attempts`.

**NO MODEL IS EVER CALLED HERE**: every run is handed this module's own
`request=`. **The real `ai_jobs_runner` lock is never touched** - each test
points `runner.RUNNER_LOCK_KEY` at a key of its own, so the nightly task holding
the real one (22:00 PDT onward) can neither block these tests nor be blocked by
them.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as tj5  # noqa: E402
import tj6_support as fx  # noqa: E402

SLOT = "improvement_ideas"

#: 02:00 ET on Friday 2026-09-18 - inside the trader's live 01:00-09:00 window,
#: on a session day, so `session_date_for` is Thursday 2026-09-17 and an
#: already-covered slot is skipped SILENTLY rather than with a no-session row.
NIGHT = datetime(2026, 9, 18, 2, 0, tzinfo=tj5.EASTERN)
SESSION = "2026-09-17"

LIVE_START = "01:00"
LIVE_END = "09:00"


def _settings(**values):
    from ai_jobs import store

    return mock.patch.object(
        store._paths(),
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


def _ledger_rows(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if row.get("job") == SLOT]


@pytest.fixture
def night(tmp_path, monkeypatch):
    """A scratch night: the stores, the window, the store check, and NO real lock."""
    import project_paths
    from ai_jobs import miss_contrast, runner, window

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "day_review"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    monkeypatch.setattr(
        project_paths, "AI_IDEAS_FILE", tmp_path / "ai_ideas.jsonl", raising=False
    )
    monkeypatch.setattr(
        project_paths, "AI_IDEAS_STATE_FILE", tmp_path / "ai_ideas_state.json", raising=False
    )
    # The contrast pack is the one input that does NOT live under the scratch
    # root, and a pack another test published would quietly give this night
    # something to cite. Pinned to "there is none", which is the live state.
    monkeypatch.setattr(miss_contrast, "read_latest", lambda *a, **k: None)
    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    # NEVER the real `ai_jobs_runner` lock: the nightly task holds that one.
    monkeypatch.setattr(runner, "RUNNER_LOCK_KEY", f"tj6-test-{uuid.uuid4().hex}")
    return {
        "root": root,
        "ideas": tmp_path / "ai_ideas.jsonl",
        "asked": tmp_path / "ai_ideas_asked.json",
        "ledger": tmp_path / "ledger.jsonl",
    }


def _passes(night, answer, *, count: int = 3) -> list[dict]:
    """`count` consecutive task firings at the SAME moment, through `run_slots`.

    The real runner, the real slot function, the real ledger - only the model is
    this module's own. Three passes, because the scheduled task fires sixteen
    times a night and the second one is where a re-ask shows up.
    """
    from ai_jobs import improvement_ideas, runner, store

    calls: list[dict] = []
    request = fx.fake_request(answer, calls=calls)
    slate = [
        replace(slot, run=partial(improvement_ideas.run_improvement_ideas, request=request))
        for slot in runner.default_slots()
        if slot.name == SLOT
    ]
    assert len(slate) == 1, "the ideas slot is not registered"
    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            for _firing in range(count):
                runner.run_slots(slate, now=NIGHT, only=SLOT, ledger_path=night["ledger"])
    return calls


def _one_idea(night, *, measurable: str = "", evidence=None):
    from ai_jobs import improvement_ideas

    inputs = improvement_ideas.build_ideas_inputs(SESSION, root=night["root"])
    allowed = list(inputs["allowed_source_ids"])
    return fx.reply(
        [
            fx.idea_payload(
                "Wait for the second test before sizing up.",
                measurable=measurable or str(improvement_ideas.MEASURABLES[0].name),
                evidence=allowed[:1] if evidence is None else evidence,
            )
        ]
    )


# ---------------------------------------------------------------------------
# half (a) - nothing to cite, no model load
# ---------------------------------------------------------------------------
def test_a_window_with_nothing_to_cite_never_loads_the_model(night):
    """The live state: zero packs. Hand-counted: 3 firings -> 0 model calls.

    Every idea must cite an id the night carries, so a night with nothing to
    cite can only ever produce a whole rejection. The honest answer is the count
    of sessions with facts, said before anything is loaded.
    """
    from ai_jobs import ledger

    calls = _passes(night, fx.reply([]))
    assert calls == [], f"{len(calls)} model call(s) for a night with nothing to cite"
    rows = _ledger_rows(night["ledger"])
    assert len(rows) == 3, [row.get("status") for row in rows]
    assert {row["status"] for row in rows} == {ledger.STATUS_SKIPPED}
    assert "nothing to cite" in rows[0]["reason"]
    assert not night["ideas"].exists()
    assert not night["asked"].exists(), "a night that asked nothing left an asked marker"


# ---------------------------------------------------------------------------
# half (b) - asked once is done
# ---------------------------------------------------------------------------
def test_a_night_that_asked_and_stored_nothing_is_done_for_the_night(night):
    """THE BLOCKER. Hand-counted: 3 firings -> exactly ONE model call.

    The one idea offered names a measurable the desk does not compute, so it is
    dropped and nothing is stored. That is a finished night: `ok`, with the
    counts, and the asked marker beside the store.
    """
    from ai_jobs import improvement_ideas, ledger

    tj5.write_week(night["root"], narrated=tj5.PACKED_SESSIONS)
    calls = _passes(night, _one_idea(night, measurable="not_a_measurable_the_desk_has"))

    assert len(calls) == 1, f"the night asked the model {len(calls)} times"
    rows = _ledger_rows(night["ledger"])
    assert len(rows) == 1, [(row["status"], row["reason"]) for row in rows]
    assert rows[0]["status"] == ledger.STATUS_OK
    assert "asked once" in rows[0]["reason"]
    # `ledger.record` FLATTENS a slot's `extra` into the row (setdefault per
    # key), so the counts are top-level fields here.
    assert rows[0]["drop_reasons"] == {improvement_ideas.DROP_UNKNOWN_MEASURABLE: 1}
    assert rows[0]["offered"] == 1 and rows[0]["dropped"] == 1
    assert improvement_ideas.read_ideas() == ()
    marker = improvement_ideas.read_asked_marker()
    assert marker["session_date"] == SESSION
    assert marker["stored"] == 0
    assert marker["inputs_hash"]


def test_the_slot_itself_short_circuits_on_its_own_marker(night):
    """Belt and braces INSIDE the slot, with no runner and no ledger in sight.

    Hand-counted: two direct calls -> ONE model call. The runner's already-done
    check is the first guard; this is the second, for a caller that has no
    ledger (a redo, a manual run, a later phase).
    """
    from ai_jobs import improvement_ideas

    tj5.write_week(night["root"], narrated=tj5.PACKED_SESSIONS)
    answer = _one_idea(night, measurable="not_a_measurable_the_desk_has")
    calls: list[dict] = []
    for _attempt in range(2):
        improvement_ideas.run_improvement_ideas(
            session_date=SESSION,
            now=fx.WEEKNIGHT,
            root=night["root"],
            request=fx.fake_request(answer, calls=calls),
        )
    assert len(calls) == 1, f"{len(calls)} model calls for one unchanged night"


def test_a_rejected_night_is_an_attempt_and_is_capped(night):
    """A rejection is NOT a finished night - it may try again, twice.

    Hand-counted: 3 firings, `max_attempts=2` -> exactly TWO model calls, two
    `failed` rows, and the third firing stopped by the cap. The counts of what
    the rejected answer held travel with the row.
    """
    from ai_jobs import ledger

    tj5.write_week(night["root"], narrated=tj5.PACKED_SESSIONS)
    calls = _passes(night, _one_idea(night, evidence=["2020-01-02/report_card:did_well"]))

    assert len(calls) == 2, f"the night asked the model {len(calls)} times"
    rows = _ledger_rows(night["ledger"])
    failed = [row for row in rows if row["status"] == ledger.STATUS_FAILED]
    assert len(failed) == 2, [(row["status"], row["reason"]) for row in rows]
    assert failed[0]["offered"] == 1
    assert failed[0]["rejected"] is True
    assert not night["asked"].exists(), "a rejected night armed the skip"


def test_a_normal_night_asks_once_and_the_later_passes_do_nothing(night):
    """Hand-counted: 3 firings -> ONE model call, ONE `ok` row, ONE stored idea."""
    from ai_jobs import improvement_ideas, ledger

    tj5.write_week(night["root"], narrated=tj5.PACKED_SESSIONS)
    calls = _passes(night, _one_idea(night))

    assert len(calls) == 1, f"the night asked the model {len(calls)} times"
    rows = _ledger_rows(night["ledger"])
    assert len(rows) == 1 and rows[0]["status"] == ledger.STATUS_OK
    stored = improvement_ideas.read_ideas()
    assert len(stored) == 1
    assert len(night["ideas"].read_text(encoding="utf-8").splitlines()) == 1
