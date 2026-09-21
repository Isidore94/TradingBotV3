r"""TJ-5 - the week story narrates MEASURED rows and rejects the rest. RED.

Packet `.claude/packets/TJ-5-6-7-13B.md` "TJ-5"; `plan.md` §12.4 TJ-5 change 2
with its AMENDED 2026-09-19 block; decision 0021 answers 14 and 20.

**NO MODEL IS EVER CALLED HERE.** Every test hands
`run_week_review_narration` a `request` callable of its own.

The rule this packet inherits from TJ-4, restated because it is what these
tests are for: **a JSON schema's `maxItems` / `additionalProperties` is a
grammar hint to the provider, never a guard.** Three packets shipped that bug
on 2026-09-20. So every bound the week slot declares is derived FROM ITS OWN
INPUT and RE-CHECKED by the slot after the reply comes back.

The contract these tests pin (the builder may ADD keys, never remove one)
------------------------------------------------------------------------

``scripts/ai_jobs/week_review_narration.py``::

    PROMPT_VERSION = SCHEMA = "week_review_narration_v1"
    WEEK_NARRATION_JSON_SCHEMA: dict      # closed: additionalProperties False
    MIN_NARRATED_DAYS = 3                 # plan.md: "fewer than three narrated days"
    TENDENCY_LIMIT = 3                    # "at most three tendencies"
    WEEK_SOURCE_SEPARATOR = "/"
    EVIDENCE_KEYS: tuple[str, ...]        # the CLOSED set of evidence sections

    week_id(session_date) -> "YYYY-Www"
    week_source_id(session, source_id) -> "<session>/<source_id>"
    narration_path(week, *, root=None)    # <root>/week/<W>.json
    read_week_narration(week, *, root=None) -> dict | None

    build_week_inputs(session_date, *, root=None, ledger_path=None) -> dict
        week_id, sessions (5), sessions_with_facts, sessions_missing,
        narrated_sessions, days, allowed_source_ids, were_you_right
        {right, wrong, unresolved, n}, tendencies [{text, n, source_id}],
        misses, rollup, walkaway_totals, inputs_hash

    check_week_narration(narration, inputs) -> None    # raises WeekNarrationRejected
    run_week_review_narration(*, session_date="", now=None, root=None,
                              request=None, ledger_path=None, force=False,
                              **_ignored)
        -> {"status", "model", "reason", "outputs", "model_attribution"}

``force`` is the week's redo: it re-spends the unchanged-hash skip and nothing
else - it never buys the night window, which is the runner's gate and TJ-13A
item 1's rule.

The model's object is ``headline``, ``what_happened`` (<=1500),
``were_you_right`` {``right``, ``wrong``, ``unresolved``, ``examples`` (<=3,
each citing an id)}, ``chased`` [cited], ``tendencies`` [{``text``,
``source_id``, ``n``}], ``process_pattern`` (<=600), ``next_week_watch``,
``sources`` - `plan.md` TJ-5 change 2 plus the packet's tendencies clause.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as fx  # noqa: E402


@pytest.fixture
def week_root(tmp_path, monkeypatch):
    """A scratch `DAY_REVIEW_DIR` with THREE of the week's five packs.

    Three, because that is what the live store holds: 2026-09-16, -17 and -18
    have session folders and none of the five has a `pack.json` at all
    (measured read-only, 2026-09-20). A week page whose first honest sentence is
    not "3 of 5 sessions have facts" is wrong before it prints a number.
    """
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    fx.write_week(root, narrated=fx.PACKED_SESSIONS)
    return root


@pytest.fixture
def contrast_packs(tmp_path, monkeypatch):
    """TJ-15's and TJ-16's packs where their own `read_latest` will find them."""
    import ai_jobs.miss_contrast as miss
    import ai_jobs.prediction_contrast as pred

    folder = tmp_path / "digests"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{pred.PACK_PREFIX}-{fx.FRIDAY}.json").write_text(
        json.dumps(fx.prediction_contrast_pack()), encoding="utf-8"
    )
    (folder / f"{miss.PACK_PREFIX}-{fx.FRIDAY}.json").write_text(
        json.dumps(fx.miss_contrast_pack()), encoding="utf-8"
    )
    from ai_jobs import store

    monkeypatch.setattr(store, "digests_dir", lambda create=True: folder)
    return folder


def _inputs(root):
    import ai_jobs.week_review_narration as week

    return week.build_week_inputs(fx.FRIDAY, root=root)


def _good(inputs, **overrides):
    """A reply that passes every check, built out of the inputs themselves."""
    tally = inputs["were_you_right"]
    body = {
        "headline": "You were right more than wrong, on three of five days.",
        "what_happened": "Three sessions carry facts; two were never packed.",
        "were_you_right": {
            "right": tally["right"],
            "wrong": tally["wrong"],
            "unresolved": tally["unresolved"],
            "examples": [],
        },
        "chased": [],
        "tendencies": [],
        "process_pattern": "You waited for the second test three times.",
        "next_week_watch": [],
        "sources": list(inputs["allowed_source_ids"])[:1],
    }
    body.update(overrides)
    return body


def _request_for(reply, calls=None):
    def _request(**kwargs):
        if calls is not None:
            calls.append(dict(kwargs))
        return {"summary": reply, "model": "gemma3:27b"}

    return _request


# ---------------------------------------------------------------------------
# what the week is, and what it is allowed to see
# ---------------------------------------------------------------------------
def test_the_week_is_five_sessions_and_the_missing_two_are_named(week_root):
    """`evidence_stats.WEEK_SESSIONS` is 5, and a session with no pack is NAMED.

    Hand-counted: the fixture writes 2026-09-16, -17 and -18; 2026-09-14 and
    2026-09-15 have no pack on disk at all. A missing day is never padded into
    a quiet one.
    """
    import evidence_stats

    inputs = _inputs(week_root)
    assert len(inputs["sessions"]) == evidence_stats.WEEK_SESSIONS == 5
    assert tuple(inputs["sessions"]) == fx.WEEK
    assert tuple(inputs["sessions_with_facts"]) == fx.PACKED_SESSIONS
    assert tuple(inputs["sessions_missing"]) == ("2026-09-14", "2026-09-15")


def test_two_packs_cannot_collide_on_one_source_id(week_root):
    """Each pack mints its ids with its OWN minter, so `said:...:prediction`
    is the same string in all five. The week's ids are session-qualified, or a
    citation in the week story names two different rows."""
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    allowed = list(inputs["allowed_source_ids"])
    assert len(allowed) == len(set(allowed)), "the week's ids are not unique"
    for session in fx.PACKED_SESSIONS:
        assert any(item.startswith(f"{session}{week.WEEK_SOURCE_SEPARATOR}") for item in allowed)
    assert week.week_source_id("2026-09-18", "said:x:prediction") in {
        f"2026-09-18{week.WEEK_SOURCE_SEPARATOR}said:x:prediction"
    }


def test_the_week_story_never_sees_bars_and_never_sees_the_lake(week_root, contrast_packs):
    """`plan.md` TJ-5 change 2: "inputs are the five packs, five narrations and
    the weekly rollup - never bars, never the lake" (plus the two contrast packs
    the packet adds). The evidence key set is CLOSED and declared."""
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    calls: list[dict] = []
    week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=week_root,
        request=_request_for(_good(inputs), calls),
    )
    assert calls, "no evidence was built"
    evidence = calls[0].get("evidence") or {}
    extra = set(evidence) - set(week.EVIDENCE_KEYS)
    assert not extra, f"the week story was sent sections nobody declared: {sorted(extra)}"
    forbidden = {"bars", "m5_bars", "daily_bars", "spy_bars", "research", "lake"}
    assert not (set(week.EVIDENCE_KEYS) & forbidden)


# ---------------------------------------------------------------------------
# the bounds are derived from the input and RE-CHECKED
# ---------------------------------------------------------------------------
def test_at_most_three_tendencies_are_offered_and_none_under_its_floor(
    week_root, contrast_packs
):
    """The packet: "it may narrate at most three tendencies, each citing its
    cell and `n`".

    Hand-counted: `prediction_contrast_pack` holds FIVE reportable by-hour cells
    with n = 40, 39, 38, 37, 36 and one thin cell (n = MIN_REPORTABLE_N - 5,
    `reportable` False). The three offered are the three largest `n`, in `n`
    order - a SIZE rule, never a rate - and the thin one is never offered.
    """
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    offered = list(inputs["tendencies"])
    assert len(offered) == week.TENDENCY_LIMIT == 3
    assert [item["n"] for item in offered] == [40, 39, 38]
    assert all(item.get("source_id") for item in offered)


def test_a_fourth_tendency_is_rejected_even_though_the_schema_said_three(
    week_root, contrast_packs
):
    """A schema is a grammar hint, not a guard. The slot re-checks its own bound.

    The reply below carries FOUR tendencies. Nothing about the JSON schema the
    provider was handed can stop that from arriving; the verifier must.
    """
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    offered = list(inputs["tendencies"])
    four = [
        {"text": item["text"], "source_id": item["source_id"], "n": item["n"]}
        for item in offered
    ]
    four.append({**four[0], "text": four[0]["text"] + " (again)"})
    with pytest.raises(week.WeekNarrationRejected):
        week.check_week_narration(_good(inputs, tendencies=four), inputs)


def test_a_tendency_citing_a_cell_the_input_does_not_hold_is_rejected(
    week_root, contrast_packs
):
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    invented = [{"text": "Mornings are kind to you.", "source_id": "tendency:invented", "n": 44}]
    with pytest.raises(week.WeekNarrationRejected):
        week.check_week_narration(_good(inputs, tendencies=invented), inputs)


def test_a_tendency_that_restates_its_n_wrongly_is_rejected(week_root, contrast_packs):
    """The model quotes a number it did not compute. If it changes it, the page
    would print a count nothing measured."""
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    first = dict(inputs["tendencies"][0])
    bent = [{"text": first["text"], "source_id": first["source_id"], "n": first["n"] + 7}]
    with pytest.raises(week.WeekNarrationRejected):
        week.check_week_narration(_good(inputs, tendencies=bent), inputs)


# ---------------------------------------------------------------------------
# a verdict must EQUAL the measured row
# ---------------------------------------------------------------------------
def test_the_were_you_right_tally_must_equal_the_measured_one(week_root):
    """Hand-counted over the three packs that exist (`tj5_support`'s table):

        2026-09-16   1 right   1 wrong   0 other
        2026-09-17   0 right   2 wrong   0 other
        2026-09-18   1 right   0 wrong   1 other  (unmeasured:no_tape)
        --------------------------------------------------
                     2 right   3 wrong   1 unresolved   n = 6

    A week story that rounded the unresolved read into "wrong" would be a
    verdict the desk never measured.
    """
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    counted = fx.tally_over(fx.PACKED_SESSIONS)
    assert counted == {"right": 2, "wrong": 3, "unresolved": 1, "n": 6}
    assert {
        key: inputs["were_you_right"][key] for key in ("right", "wrong", "unresolved", "n")
    } == counted

    week.check_week_narration(_good(inputs), inputs)  # the measured tally passes

    bent = {"right": 2, "wrong": 4, "unresolved": 0, "examples": []}
    with pytest.raises(week.WeekNarrationRejected):
        week.check_week_narration(_good(inputs, were_you_right=bent), inputs)


def test_a_citation_the_week_does_not_carry_is_rejected(week_root):
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    with pytest.raises(week.WeekNarrationRejected):
        week.check_week_narration(_good(inputs, sources=["2026-09-18/said:nope"]), inputs)
    with pytest.raises(week.WeekNarrationRejected):
        week.check_week_narration(_good(inputs, sources=[]), inputs)


def test_a_week_story_with_no_headline_is_rejected(week_root):
    """A story with an empty headline is written and then read as "no story yet"
    OVER a story. An answer that says nothing is not an answer (TJ-4, round 2)."""
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    with pytest.raises(week.WeekNarrationRejected):
        week.check_week_narration(_good(inputs, headline="   "), inputs)


# ---------------------------------------------------------------------------
# publishing
# ---------------------------------------------------------------------------
def test_the_week_story_is_one_file_per_week_under_the_day_review_root(week_root):
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    outcome = week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=week_root,
        request=_request_for(_good(inputs)),
    )
    path = week.narration_path(fx.WEEK_ID, root=week_root)
    assert path == week_root / "week" / f"{fx.WEEK_ID}.json"
    assert path.exists(), outcome
    assert [str(path)] == list(outcome["outputs"])
    # temp-and-rename: nothing half-written is left beside it.
    assert not list((week_root / "week").glob("*.tmp"))
    stored = week.read_week_narration(fx.WEEK_ID, root=week_root)
    assert stored["schema"] == week.SCHEMA
    assert stored["prompt_version"] == week.PROMPT_VERSION
    assert stored["week_id"] == fx.WEEK_ID


def test_a_rejected_week_story_leaves_the_prior_file_byte_identical(week_root):
    """The last verified week is the fallback. A rejection costs the NIGHT, and
    never the file the trader read last Saturday."""
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=week_root,
        request=_request_for(_good(inputs)),
    )
    path = week.narration_path(fx.WEEK_ID, root=week_root)
    before = path.read_bytes()

    bad = _good(inputs, were_you_right={"right": 99, "wrong": 0, "unresolved": 0, "examples": []})
    outcome = week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=week_root,
        request=_request_for(bad),
        force=True,
    )
    assert path.read_bytes() == before
    assert outcome["status"] != "ok", outcome


def test_an_unchanged_week_is_not_paid_for_twice(week_root):
    """The same five packs, the same prompt version: the verified file stands
    and no model is loaded. `inputs_hash` is over the SECTIONS, never the clock."""
    import ai_jobs.week_review_narration as week

    inputs = _inputs(week_root)
    week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=week_root,
        request=_request_for(_good(inputs)),
    )
    calls: list[dict] = []
    outcome = week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=week_root,
        request=_request_for(_good(inputs), calls),
    )
    assert calls == [], "the week story was narrated twice for one unchanged week"
    assert outcome["status"] == "ok", outcome


# ---------------------------------------------------------------------------
# the honest state: fewer than three narrated days
# ---------------------------------------------------------------------------
def test_fewer_than_three_narrated_days_gets_the_scaffold_and_no_model_call(
    tmp_path, monkeypatch
):
    """`plan.md` TJ-5 change 2: "Fewer than three narrated days -> a
    deterministic scaffold and 'narrated K of 5'."

    Hand-counted: two packs on disk, two TJ-4 narrations, so K is 2 and the
    floor is `MIN_NARRATED_DAYS` (3). The 27B is not loaded to say so.
    """
    import project_paths

    import ai_jobs.week_review_narration as week

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    two = fx.PACKED_SESSIONS[:2]
    fx.write_week(root, sessions=two, narrated=two)

    calls: list[dict] = []
    outcome = week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=root,
        request=_request_for({}, calls),
    )
    assert calls == [], "a week with two narrated days still loaded a model"
    assert "narrated 2 of 5" in outcome["reason"], outcome["reason"]

    stored = week.read_week_narration(fx.WEEK_ID, root=root)
    assert stored is not None, "the deterministic scaffold was not written"
    assert stored.get("scaffold") is True
    assert "narrated 2 of 5" in json.dumps(stored)
    assert not stored.get("narration"), "a scaffold is not a story"


def test_a_week_with_no_packs_at_all_says_so_and_invents_nothing(tmp_path, monkeypatch):
    """The live state on 2026-09-20: three session folders, ZERO `pack.json`.

    Zero of five is a sentence, not five zero days and not a crash.
    """
    import project_paths

    import ai_jobs.week_review_narration as week

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    (root / "sessions" / fx.FRIDAY).mkdir(parents=True)

    inputs = week.build_week_inputs(fx.FRIDAY, root=root)
    assert tuple(inputs["sessions_with_facts"]) == ()
    assert tuple(inputs["sessions_missing"]) == fx.WEEK
    assert inputs["were_you_right"] == {"right": 0, "wrong": 0, "unresolved": 0, "n": 0}

    calls: list[dict] = []
    outcome = week.run_week_review_narration(
        session_date=fx.FRIDAY,
        now=fx.SATURDAY_NIGHT,
        root=root,
        request=_request_for({}, calls),
    )
    assert calls == []
    assert "narrated 0 of 5" in outcome["reason"], outcome["reason"]
