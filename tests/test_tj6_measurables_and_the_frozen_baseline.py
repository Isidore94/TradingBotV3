r"""TJ-6 - advice is CHECKED: one named measurable, frozen before and after. RED.

`plan.md` §12.4 TJ-6 **AMENDED 2026-09-19 (trader)**: *"A `process` idea must
name ONE measurable the desk already computes (a veto reason's count and
real-miss rate, a report-card line, a TJ-14 question's answer mix) or it is
dropped ... Keeping it freezes a baseline - that measurable over the
`LATELY_SESSIONS` before the keep - in `AI_IDEAS_STATE_FILE`. Week Review then
prints, for every kept idea, before and after with both `n`, deterministically,
until the trader retires it; under the floor it says "too few to call". The
model never grades its own advice."* Decision 0021 answer 23 is the one-line
version.

WHAT THE DESK REALLY COMPUTES (recon, this branch, 2026-09-20)
-------------------------------------------------------------
* **a veto reason's count and real-miss rate** - `ai_jobs/miss_contrast.py:376`
  `build_pack(...)`'s ``groups``: one row per veto ``reason_code`` carrying
  ``n``, ``measured``, ``rate`` (misses / measured, ``None`` when measured is 0)
  and ``reportable``. It judges D1 decisions only and takes a
  ``window_sessions`` that already defaults to `evidence_stats.LATELY_SESSIONS`.
* **a report-card line** - `day_report_card.py:154` `_line(...)`: every line is
  ``{"key", "text", "n", "measured", "target", ...}``, so ``n`` and ``measured``
  are already there; `day_report_card.week_from_cards` (line 1100) pools them.
* **a TJ-14 question's answer mix** - PREMISE REFUTED. The only counter that
  exists is `day_report_card.process_line`'s ``origin_answers`` (line 521-528,
  returned at 600), which is ONE question kind (`trade_origin`), ONE session,
  and is DROPPED by `_pool_cards` (lines 1226-1256), so there is no multi-
  session answer mix to name. This file therefore requires the two that exist
  and does not require the third.

**NO MODEL IS EVER CALLED HERE.**

VERIFIED ON THIS BRANCH (1b9d77e0): `ai_jobs.improvement_ideas` does not exist,
so every test below fails on the import.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj6_support as fx  # noqa: E402

TEXT = "Wait for the second test before sizing up."


@pytest.fixture
def night(tmp_path, monkeypatch):
    return fx.install_stores(monkeypatch, tmp_path)


def _resolve(dotted: str):
    """Import the longest importable prefix of `dotted` and walk the rest.

    The same walk `mentor_questions._resolve` does for a question's consumer -
    written out here rather than imported, so a registry that resolved its own
    names with a shim would not be checking itself.
    """
    parts = [part for part in str(dotted).split(".") if part]
    module = None
    index = 0
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
        except Exception:  # noqa: BLE001 - keep shortening until one imports
            continue
        index = stop
        break
    if module is None:
        raise AssertionError(f"no prefix of {dotted!r} imports")
    target = module
    for part in parts[index:]:
        target = getattr(target, part)
    return target


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
def test_every_measurable_names_a_reader_that_really_exists():
    """A measurable whose reader does not resolve is a promise the desk cannot
    keep - the same defect `mentor_questions.consumer_report` exists to catch
    for a question's answer."""
    from ai_jobs import improvement_ideas

    registry = tuple(improvement_ideas.MEASURABLES)
    assert registry, "the desk offers no measurable at all"
    for item in registry:
        assert str(item.name).strip()
        assert str(item.label).strip(), f"{item.name} has no sentence for the trader"
        resolved = _resolve(item.reader)
        assert callable(resolved), f"{item.name} -> {item.reader} is not callable"
    names = [item.name for item in registry]
    assert len(set(names)) == len(names), names


def test_the_two_measurables_the_packet_names_and_the_desk_has_are_registered():
    """The veto reason's count and real-miss rate, and a report-card line.

    Both were measured on this branch (this file's docstring); the third example
    in the amendment - a Mentor question's answer mix over a window - does not
    exist as a callable today and is not required here.
    """
    from ai_jobs import improvement_ideas

    readers = [str(item.reader) for item in improvement_ideas.MEASURABLES]
    assert any(reader.startswith("day_report_card.") for reader in readers), readers
    assert any("miss_contrast" in reader for reader in readers), readers


def test_an_unknown_measurable_raises_rather_than_defaulting():
    """`mentor_questions.kind_named`'s rule: an unknown name RAISES, never a
    default. A default here would let an unmeasurable idea through the drop."""
    from ai_jobs import improvement_ideas

    first = improvement_ideas.MEASURABLES[0]
    assert improvement_ideas.measurable_named(first.name) is first
    with pytest.raises(KeyError):
        improvement_ideas.measurable_named("not_a_measurable_the_desk_has")


# ---------------------------------------------------------------------------
# reading one
# ---------------------------------------------------------------------------
def test_a_reading_carries_its_own_n_and_its_window(night):
    """Every number the trader is shown carries `n` and says its window.

    `LATELY_SESSIONS` is read out of `evidence_stats` (20 today), never typed.
    """
    from ai_jobs import improvement_ideas

    name = improvement_ideas.MEASURABLES[0].name
    reading = improvement_ideas.measure(name, end_session=fx.SESSION)
    assert set(reading) >= {"measurable", "value", "n", "measured", "window_sessions"}
    assert reading["measurable"] == name
    assert reading["window_sessions"] == fx.lately_sessions()
    assert isinstance(reading["n"], int)
    assert isinstance(reading["measured"], bool)


def test_a_reader_that_cannot_answer_is_unmeasured_and_never_zero(night, monkeypatch):
    """`plan.md` sec 5: missing data is uncertainty, never confirmation.

    A store that will not open must not become "your real-miss rate is 0%".
    """
    from ai_jobs import improvement_ideas

    item = improvement_ideas.MEASURABLES[0]
    module_name, _, attribute = str(item.reader).rpartition(".")
    module = importlib.import_module(module_name)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("the pack could not be read")

    monkeypatch.setattr(module, attribute, _boom)
    reading = improvement_ideas.measure(item.name, end_session=fx.SESSION)
    assert reading["measured"] is False
    assert reading["value"] is None, "an unreadable store became a number"
    assert reading["n"] == 0


# ---------------------------------------------------------------------------
# the keep freezes a baseline
# ---------------------------------------------------------------------------
def _kept_process_idea(night, monkeypatch, *, reading=None):
    """Store one `process` idea, then KEEP it with `measure` pinned to a hand
    number. Returns the idea id."""
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
    return row["idea_id"]


def test_keeping_a_process_idea_freezes_the_measurable_as_it_was(night, monkeypatch):
    """Hand-counted: the baseline is value 0.30 with n 34, and it is stored
    beside the keep with the window it was read over."""
    from ai_jobs import improvement_ideas

    idea_id = _kept_process_idea(night, monkeypatch)
    state = improvement_ideas.read_state()
    record = state[idea_id]
    assert record["status"] == improvement_ideas.STATUS_KEPT
    assert str(record["at"]).strip()
    baseline = record["baseline"]
    assert baseline["value"] == fx.BASELINE["value"]
    assert baseline["n"] == fx.BASELINE["n"]
    assert baseline["measured"] is True
    assert baseline["window_sessions"] == fx.lately_sessions()
    assert baseline["measurable"] == improvement_ideas.MEASURABLES[0].name


def test_the_frozen_baseline_is_never_re_read_or_overwritten(night, monkeypatch):
    """Before and after, both with `n`, and BEFORE is the number at the keep.

    Hand-counted: before 0.30 (n 34), after 0.48 (n 41). A baseline recomputed
    at read time would show 0.48 on both sides and every kept idea would look
    like it changed nothing.
    """
    from ai_jobs import improvement_ideas

    idea_id = _kept_process_idea(night, monkeypatch)
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

    rows = {row["idea_id"]: row for row in improvement_ideas.checked_ideas(end_session=fx.SESSION)}
    row = rows[idea_id]
    assert row["before"]["value"] == fx.BASELINE["value"]
    assert row["before"]["n"] == fx.BASELINE["n"]
    assert row["after"]["value"] == fx.AFTER["value"]
    assert row["after"]["n"] == fx.AFTER["n"]
    assert row["kept_at"]
    # And the state file still holds the ORIGINAL numbers.
    assert improvement_ideas.read_state()[idea_id]["baseline"]["n"] == fx.BASELINE["n"]


def test_under_the_floor_a_checked_idea_says_too_few_to_call(night, monkeypatch):
    """`evidence_stats.MIN_REPORTABLE_N` is 30, so n 29 is under the floor.

    Hand-counted: before n 34 (reportable), after n 29 (not) -> the row is
    "too few to call" and names no winner.
    """
    from ai_jobs import improvement_ideas

    idea_id = _kept_process_idea(night, monkeypatch)
    thin = fx.thin_reading()
    assert thin["n"] == fx.min_reportable_n() - 1
    monkeypatch.setattr(
        improvement_ideas,
        "measure",
        lambda measurable, **kwargs: {
            "measurable": measurable,
            "value": thin["value"],
            "n": thin["n"],
            "measured": True,
            "window_sessions": fx.lately_sessions(),
        },
    )

    rows = {row["idea_id"]: row for row in improvement_ideas.checked_ideas(end_session=fx.SESSION)}
    assert rows[idea_id]["verdict"] == "too few to call"


def test_a_kept_program_idea_is_listed_for_wishlist_and_never_graded(night, monkeypatch):
    """*"Kept `program` ideas are listed 'For WISHLIST - copy'"*.

    A `program` idea names no measurable, so it has no before and no after - and
    a blank pair must not read as a measured zero.
    """
    from ai_jobs import improvement_ideas

    row = fx.stored_idea_row(
        "Put the walk-away table beside the trades on Day Review.",
        session=fx.SESSION,
        kind="program",
        measurable="",
    )
    fx.write_ideas(night["ideas"], [row])
    improvement_ideas.keep_idea(row["idea_id"], end_session=fx.SESSION)

    state = improvement_ideas.read_state()
    assert state[row["idea_id"]]["status"] == improvement_ideas.STATUS_KEPT
    assert "baseline" not in state[row["idea_id"]], "a program idea has no measurable"

    checked = {item["idea_id"]: item for item in improvement_ideas.checked_ideas(end_session=fx.SESSION)}
    item = checked[row["idea_id"]]
    assert item["kind"] == "program"
    assert item["for_wishlist"] is True
    assert item["before"] == {} and item["after"] == {}
    assert item["verdict"] == "not checked here"


def test_checking_an_idea_calls_no_model_at_all(night, monkeypatch):
    """*"The model never grades its own advice."*

    The provider seam is replaced with something that raises: a `checked_ideas`
    that reached a model would fail here instead of reading two numbers.
    """
    from ai_jobs import improvement_ideas, provider

    idea_id = _kept_process_idea(night, monkeypatch)

    def _boom(*_args, **_kwargs):
        raise AssertionError("checking an idea asked a model")

    monkeypatch.setattr(provider, "request_with_fallback", _boom)
    rows = {row["idea_id"]: row for row in improvement_ideas.checked_ideas(end_session=fx.SESSION)}
    assert idea_id in rows
