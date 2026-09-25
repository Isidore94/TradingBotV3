r"""TJ-4 item 2 - the overnight day story NARRATES measured verdicts. RED.

Packet `.claude/packets/TJ-4.md` "The one rule that changed" and item 2;
`plan.md` §12.4 "TJ-4" change 2 and its **AMENDED 2026-09-19** block; §12.3;
decision 0021 answer 14; decision 0018 and both its 2026-09-19 amendments.

**NO MODEL IS EVER CALLED HERE.** Every test hands the module a `request`
callable of its own, or monkeypatches `ai_summary.request_ai_summary` to one
that raises if it is reached. The nightly task holds
`local_writer_lock("ai_jobs_runner")` for hours, so the runner tests replace
that mutex, pass a frozen `now`, NAME the night kind instead of asking
`night_kind()`, and point the AI store at `tmp_path`.

The contract these tests pin (the builder may ADD keys, never remove one)
------------------------------------------------------------------------

``scripts/ai_jobs/day_review_narration.py``::

    PROMPT_VERSION = SCHEMA = "day_review_narration_v1"
    NARRATION_JSON_SCHEMA: dict           # closed: additionalProperties False

    run_day_review_narration(*, session_date="", now=None, root=None,
                             request=None, **_ignored)
        -> {"status", "model", "reason", "outputs": [path, ...]}

    narration_path(session, *, root=None)   # <root>/narration/<date>.json
    read_narration(session, *, root=None) -> dict | None

``root`` defaults to ``day_review_pack.default_root()`` - `DAY_REVIEW_DIR`, read
at CALL time. The written file mirrors `market_story_narration`'s shape:
``{"schema", "session_date", "generated_at", "inputs_hash", "prompt_version",
"model", "narration": {...}}``.

The model's object is ``headline``, ``what_happened``, ``what_you_thought``,
``were_you_right`` [{``claim``, ``source_id``, ``verdict``, ``evidence_id``}],
``chased_against_news`` {``verdict``, ``evidence_id``}, ``process``, ``sources``.

* ``source_id`` names a `trader_said` item and MUST be one whose kind is
  ``prediction`` - an ``observation`` is quoted as what the trader saw, never
  graded as a call.
* ``evidence_id`` names a `reads` item, and the stated ``verdict`` must EQUAL
  that row's measured verdict.
* Any breach rejects the output WHOLE and leaves the last verified file
  byte-identical.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj4_support as fx  # noqa: E402

SESSION = fx.SESSION
SLOT = "day_review_narration"

#: The stage-3 names as `runner.default_slots()` spells them today (measured on
#: this branch, 2026-09-20).
STAGE_THREE = ("journal_enrichment", "review_policy_draft", "setup_research")
#: The slot that CLOSES decision 0018's stage 1 (`runner._STAGE_ONE_LAST_SLOT`).
STAGE_ONE_LAST = "measured_report"


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture
def day_review_root(tmp_path, monkeypatch):
    """A scratch `DAY_REVIEW_DIR` with this session's pack already written."""
    import day_review_pack
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    day_review_pack.write_pack(fx.build(), root=root)
    return root


@pytest.fixture
def ai_store(tmp_path, monkeypatch):
    root = tmp_path / "ai_store"
    root.mkdir()
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(root))
    return root


@pytest.fixture
def unlocked(monkeypatch):
    import local_writer_lock as lock_mod

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)


def _pack(root):
    import day_review_pack

    pack = day_review_pack.read_pack(SESSION, root=root)
    assert pack is not None, "fixture drift: the pack was not written"
    return pack


def _ids(root):
    """`(prediction source_id, observation source_id, read source_id, verdict)`.

    Read OFF the pack rather than typed, because the id format is the builder's
    choice and the rule under test is about what an id POINTS AT.
    """
    pack = _pack(root)
    clicked = fx.observing_and_predicting_entry()
    said = [
        item for item in pack["trader_said"]
        if str(item.get("entry_id") or "") == clicked["entry_id"]
    ]
    call = next(item for item in said if item["kind"] == "prediction")
    seen = next(item for item in said if item["kind"] == "observation")
    read = next(
        row for row in pack["reads"]
        if str(row.get("entry_id") or "") == clicked["entry_id"]
    )
    return call["source_id"], seen["source_id"], read["source_id"], read["verdict"]


def _reply(root, **overrides):
    """A well-formed narration that agrees with the measured row."""
    call, _seen, read, verdict = _ids(root)
    narration = {
        "headline": "You called the afternoon up and it went up.",
        "what_happened": "SPY closed +2.00% and held over its session VWAP all day.",
        "what_you_thought": "At 07:02 you read breadth as better and called it up.",
        "were_you_right": [
            {
                "claim": "Rest of day: up",
                "source_id": call,
                "verdict": verdict,
                "evidence_id": read,
            }
        ],
        "chased_against_news": {"verdict": "unknown", "evidence_id": read},
        "process": "One call, one note, two trades.",
        "sources": [call, read],
    }
    narration.update(overrides)
    return {"model": "local-test-medium", "summary": narration}


def _run(root, reply, *, session=SESSION, now=None):
    from ai_jobs.day_review_narration import run_day_review_narration

    calls: list[dict] = []

    def request(**kwargs):
        calls.append(kwargs)
        return reply(**kwargs) if callable(reply) else reply

    outcome = run_day_review_narration(
        session_date=session,
        now=now or fx.OVERNIGHT,
        root=root,
        request=request,
    )
    return outcome, calls


def _never_called(**_kwargs):  # pragma: no cover - must never run
    raise AssertionError("no model may be called on this path")


# ---------------------------------------------------------------------------
# registration, stage and slate (decision 0018 + TJ-13A)
# ---------------------------------------------------------------------------


def _slot():
    from ai_jobs import runner

    by_name = {slot.name: slot for slot in runner.default_slots()}
    assert SLOT in by_name, sorted(by_name)
    return by_name[SLOT]


def test_the_day_story_is_a_registered_model_slot_with_a_retry_budget():
    """plan.md §12.3 and TJ-4 change 2: `reserve_minutes=10`, `max_attempts=3`.

    `uses_model` is declared HONESTLY: this slot loads the medium local model,
    so `--force` may not buy it the clock by day (TJ-13A item 1).
    """
    slot = _slot()

    assert slot.enabled is True
    assert slot.uses_model is True, "the day story loads a local model"
    assert slot.reserve_minutes == 10.0
    assert slot.max_attempts == 3, "plan.md §12.3: set max_attempts, never 0"
    assert slot.description.strip()


def test_the_day_story_runs_inside_stage_two_and_ahead_of_the_ticker_briefs():
    """Packet TJ-4: "the day story ... runs BEFORE `ticker_briefs`".

    Gate #158 reads the ledger for a day story finished before 23:30 Pacific,
    and `ticker_briefs` reserves 120 minutes in front of it. Decision 0018 still
    holds: this is INSIDE stage 2 and never across a boundary, so every
    deterministic slot - `measured_report` last among them - is already done.
    """
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    here = names.index(SLOT)

    assert here > names.index(STAGE_ONE_LAST), "stage 1 finishes first"
    assert here < names.index("ticker_briefs"), names
    assert here < names.index("market_story_narration"), names
    for later in STAGE_THREE:
        assert here < names.index(later), later


def test_the_expected_slot_order_pin_was_extended_and_still_equals_the_slate():
    """`tests/test_ai_jobs_runner.py::EXPECTED_SLOT_ORDER` is the ONE list.

    WS-RP already asserts `default_slots()` equals it exactly; this says the
    same from TJ-4's side, so a slot registered but not pinned fails here too.
    """
    import importlib.util

    from ai_jobs import runner

    spec = importlib.util.spec_from_file_location(
        "_tj4_runner_pin", ROOT_DIR / "tests" / "test_ai_jobs_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = tuple(module.EXPECTED_SLOT_ORDER)

    assert SLOT in expected, expected
    assert expected.index(SLOT) < expected.index("ticker_briefs"), expected
    assert tuple(slot.name for slot in runner.default_slots()) == expected


def test_the_day_story_is_on_the_weeknight_and_the_saturday_slate(tmp_path):
    """TJ-13A: the night kinds. A weeknight is where the day story earns its
    keep - and Saturday still runs the whole slate.

    The kind is NAMED here rather than read from `night_kind()`: a slate built
    from the wall clock is a different test every night.
    """
    from ai_jobs import runner

    for kind in ("weeknight", "saturday"):
        names = [slot.name for slot in runner.slots_for(kind)]
        assert SLOT in names, (kind, names)
        if kind == "weeknight":
            # the briefs are Saturday-only since 2026-09-24 (trader, WISHLIST P1-3 3b)
            assert "ticker_briefs" not in names, names
            continue
        assert names.index(SLOT) < names.index("ticker_briefs"), (kind, names)

    sunday = [
        slot.name
        for slot in runner.slots_for(
            "sunday", session_date=SESSION, ledger_path=tmp_path / "never.jsonl"
        )
    ]
    assert SLOT not in sunday, (
        "Sunday is stage 1 plus the weekend's backlog; a model slot joins it "
        "only when the ledger says it was attempted and unfinished"
    )


def test_a_forced_daytime_run_records_skipped_and_calls_no_model(
    tmp_path, ai_store, unlocked, day_review_root, monkeypatch
):
    """TJ-13A item 1: `--force` never buys the clock for a model slot.

    "I always want the bot to run overnight never during the day" (trader,
    2026-09-19). A 14 GB load in front of the trader's own market prep is the
    thing the rule is about, and the Redo button's queue-for-tonight (change 4)
    is the other half of the same rule.
    """
    import ai_summary
    from ai_jobs import runner, window

    monkeypatch.setattr(ai_summary, "request_ai_summary", _never_called)
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(
        window, "launch_allowed", lambda *_a, **_k: (False, "outside the night window")
    )

    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots([_slot()], now=fx.OVERNIGHT, force=True, ledger_path=led)

    assert len(report.results) == 1, report.results
    row = report.results[0]
    assert row["job"] == SLOT
    assert row["status"] == "skipped", row
    assert "night" in str(row["reason"]).lower() or "window" in str(row["reason"]).lower()


def test_the_real_runner_writes_the_story_at_night_through_the_local_seam(
    tmp_path, ai_store, unlocked, day_review_root, monkeypatch
):
    """The whole path: `run_slots` -> the slot -> `ai_summary.request_ai_summary`.

    The provider seam is FAKED, never reached. `local_model("medium")` is what
    `market_story_narration` asks for and this slot asks for the same tier.
    """
    import ai_summary
    from ai_jobs import runner, window
    from ai_jobs.day_review_narration import narration_path

    seen: list[dict] = []

    def request(**kwargs):
        seen.append(kwargs)
        return _reply(day_review_root)

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "request_ai_summary", request)
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (True, "window open"))

    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots([_slot()], now=fx.OVERNIGHT, ledger_path=led)

    row = report.results[0]
    assert row["status"] == "ok", row
    assert len(seen) == 1, seen
    assert seen[0]["provider"] == "local"
    assert seen[0]["model"] == ai_summary.local_model("medium")
    written = narration_path(SESSION, root=day_review_root)
    assert written.exists(), row
    assert [Path(p) for p in (row.get("outputs") or ())] == [written]


# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------


def test_the_schema_is_closed_and_avoids_the_grammar_defects_length():
    """Packet: "Avoid a `maxLength` of exactly 2,000 on any schema field"
    (gate #144's grammar defect). And a closed schema, like the pattern's.
    """
    from ai_jobs.day_review_narration import NARRATION_JSON_SCHEMA

    assert NARRATION_JSON_SCHEMA["additionalProperties"] is False
    properties = NARRATION_JSON_SCHEMA["properties"]
    for name in (
        "headline", "what_happened", "what_you_thought", "were_you_right",
        "chased_against_news", "process", "sources",
    ):
        assert name in properties, (name, sorted(properties))
    assert properties["headline"]["maxLength"] == 160
    assert properties["what_happened"]["maxLength"] == 1200
    assert properties["what_you_thought"]["maxLength"] == 600
    assert properties["process"]["maxLength"] == 400

    def _lengths(node):
        if isinstance(node, dict):
            if "maxLength" in node:
                yield node["maxLength"]
            for value in node.values():
                yield from _lengths(value)
        elif isinstance(node, list):
            for value in node:
                yield from _lengths(value)

    assert 2000 not in set(_lengths(NARRATION_JSON_SCHEMA))


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


def test_a_grounded_story_is_written_with_the_packs_hash_and_the_model_named(
    day_review_root,
):
    from ai_jobs.day_review_narration import (
        PROMPT_VERSION,
        SCHEMA,
        narration_path,
        read_narration,
    )

    outcome, calls = _run(day_review_root, _reply(day_review_root))

    assert outcome["status"] == "ok", outcome
    path = narration_path(SESSION, root=day_review_root)
    assert path == day_review_root / "narration" / f"{SESSION}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema"] == SCHEMA
    assert saved["prompt_version"] == PROMPT_VERSION
    assert saved["session_date"] == SESSION
    assert saved["model"] == "local-test-medium"
    assert saved["inputs_hash"] == _pack(day_review_root)["inputs_hash"]
    assert saved["narration"]["headline"].startswith("You called")
    assert read_narration(SESSION, root=day_review_root)["inputs_hash"] == (
        saved["inputs_hash"]
    )

    evidence = calls[0]["evidence"]
    import day_review_pack

    assert sorted(evidence["allowed_source_ids"]) == sorted(
        str(value) for value in day_review_pack.allowed_source_ids(_pack(day_review_root))
    )


def test_the_previous_days_narration_is_read_only_context(day_review_root):
    """plan TJ-4 change 2: "the pack ... plus the previous day's narration
    read-only". Read-only means exactly that: it is still byte-identical after.
    """
    from ai_jobs.day_review_narration import narration_path

    import market_calendar

    yesterday = market_calendar.previous_session(date.fromisoformat(SESSION)).isoformat()
    prior = narration_path(yesterday, root=day_review_root)
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text('{"narration":{"headline":"yesterday"}}\n', encoding="utf-8")
    before = prior.read_bytes()

    _outcome, _calls = _run(day_review_root, _reply(day_review_root))

    assert prior.read_bytes() == before


# ---------------------------------------------------------------------------
# "the model narrates verdicts, it never makes them"
# ---------------------------------------------------------------------------


def _prior_file(root) -> tuple[Path, bytes]:
    from ai_jobs.day_review_narration import narration_path

    path = narration_path(SESSION, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"the story from the night before"}\n', encoding="utf-8")
    return path, path.read_bytes()


def test_a_verdict_that_disagrees_with_the_measured_row_is_rejected_whole(
    day_review_root,
):
    """THE rule of this packet (packet TJ-4, "The one rule that changed").

    The read row says `right`. A story that calls the same claim `wrong` is not
    a narration of a measured row; it is an opinion, and the whole output goes.
    """
    path, before = _prior_file(day_review_root)
    call, _seen, read, verdict = _ids(day_review_root)
    assert verdict == "right", "fixture drift: the tape must grade this `right`"

    outcome, _calls = _run(
        day_review_root,
        _reply(
            day_review_root,
            were_you_right=[{
                "claim": "Rest of day: up",
                "source_id": call,
                "verdict": "wrong",
                "evidence_id": read,
            }],
        ),
    )

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before, "the last verified story was overwritten"


def test_a_claim_no_read_row_carries_is_rejected_whole(day_review_root):
    """"grades a claim no read row carries" - packet TJ-4."""
    path, before = _prior_file(day_review_root)
    call, _seen, _read, _verdict = _ids(day_review_root)

    outcome, _calls = _run(
        day_review_root,
        _reply(
            day_review_root,
            were_you_right=[{
                "claim": "You said oil would roll over",
                "source_id": call,
                "verdict": "right",
                "evidence_id": "read:nobody-measured-this",
            }],
        ),
    )

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_an_observation_graded_as_a_call_is_rejected_whole(day_review_root):
    """"Only a `prediction` may be called a call; an `observation` is quoted as
    what the trader saw" - packet TJ-4.

    The observation and the prediction come off the SAME Mentor card, so this is
    the mistake a model actually makes.
    """
    path, before = _prior_file(day_review_root)
    _call, seen, read, verdict = _ids(day_review_root)

    outcome, _calls = _run(
        day_review_root,
        _reply(
            day_review_root,
            were_you_right=[{
                "claim": "You said breadth was better",
                "source_id": seen,
                "verdict": verdict,
                "evidence_id": read,
            }],
        ),
    )

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_an_unknown_source_id_is_rejected_whole(day_review_root):
    """The grounding rule `market_story_narration` already enforces."""
    path, before = _prior_file(day_review_root)
    call, _seen, read, _verdict = _ids(day_review_root)

    outcome, _calls = _run(
        day_review_root,
        _reply(day_review_root, sources=[call, read, "journal:mj-invented"]),
    )

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_a_story_citing_nothing_at_all_is_rejected_whole(day_review_root):
    path, before = _prior_file(day_review_root)

    outcome, _calls = _run(day_review_root, _reply(day_review_root, sources=[]))

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_a_chased_verdict_without_a_forecast_may_only_be_unknown(
    tmp_path, monkeypatch
):
    """plan TJ-4 change 2: "A forecast absent -> `chased_against_news.verdict =
    unknown`", and "the desk does not measure oil or the 10-year, so a condition
    the desk cannot see is `unknown`, never assumed".
    """
    import day_review_pack
    import project_paths

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    day_review_pack.write_pack(fx.build(with_forecast=False), root=root)
    path, before = _prior_file(root)

    call, _seen, read, verdict = _ids(root)
    outcome, _calls = _run(
        root,
        {
            "model": "local-test-medium",
            "summary": {
                "headline": "You chased into a bad tape.",
                "what_happened": "SPY closed +2.00%.",
                "what_you_thought": "You called it up.",
                "were_you_right": [{
                    "claim": "Rest of day: up", "source_id": call,
                    "verdict": verdict, "evidence_id": read,
                }],
                "chased_against_news": {"verdict": "yes", "evidence_id": read},
                "process": "One call.",
                "sources": [call, read],
            },
        },
    )

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_a_reply_that_breaks_the_closed_schema_is_rejected_whole(day_review_root):
    """A field over its `maxLength` is not "mostly fine"; it is rejected."""
    path, before = _prior_file(day_review_root)

    outcome, _calls = _run(day_review_root, _reply(day_review_root, headline="x" * 200))

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_a_model_that_raises_keeps_the_last_verified_story(day_review_root):
    """The pattern's own failure rule, and the reason it is the pattern."""
    path, before = _prior_file(day_review_root)

    def _boom(**_kwargs):
        raise RuntimeError("the local endpoint refused the connection")

    outcome, _calls = _run(day_review_root, _boom)

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# the hash skip, and Redo
# ---------------------------------------------------------------------------


def test_an_unchanged_pack_hash_skips_the_call_entirely(day_review_root):
    """plan TJ-4 change 2: "An unchanged hash skips the call"."""
    first, calls = _run(day_review_root, _reply(day_review_root))
    assert first["status"] == "ok", first
    assert len(calls) == 1

    from ai_jobs.day_review_narration import run_day_review_narration

    again = run_day_review_narration(
        session_date=SESSION,
        now=fx.OVERNIGHT + timedelta(days=1),
        root=day_review_root,
        request=_never_called,
    )
    assert again["status"] == "ok", again
    assert "unchanged" in str(again["reason"]).lower()


def test_a_redo_marker_makes_the_night_narrate_again_and_is_then_cleared(
    day_review_root,
):
    """plan TJ-4 change 4: a daytime Redo "writes a `redo_requested` marker the
    nightly slot HONOURS".

    Without this the queue-for-tonight is a lie: the hash has not moved, so the
    night would skip the very session the trader asked to be redone.
    """
    import day_review_pack

    first, _calls = _run(day_review_root, _reply(day_review_root))
    assert first["status"] == "ok", first

    marker = day_review_pack.request_redo(SESSION, root=day_review_root)
    assert marker.exists()
    assert day_review_pack.redo_requested(SESSION, root=day_review_root) is True

    second, calls = _run(day_review_root, _reply(day_review_root))

    assert second["status"] == "ok", second
    assert len(calls) == 1, "the redo did not reach the model"
    assert day_review_pack.redo_requested(SESSION, root=day_review_root) is False


def test_a_session_with_no_pack_is_skipped_and_never_a_failed_night(tmp_path, monkeypatch):
    """An evidence job is never allowed to cost the night (miss_contrast's rule).

    A session the post-close tick never reached has no pack; that is a `skipped`
    row with a reason, not a failure and not an invented story.
    """
    import project_paths
    from ai_jobs.day_review_narration import run_day_review_narration

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)

    outcome = run_day_review_narration(
        session_date=SESSION, now=fx.OVERNIGHT, root=root, request=_never_called
    )

    assert outcome["status"] in ("skipped", "degraded_no_narrative"), outcome
    assert str(outcome["reason"]).strip()
    assert list((root / "narration").glob("*.json")) == []
