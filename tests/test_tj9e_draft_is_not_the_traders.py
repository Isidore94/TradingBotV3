r"""TJ-9E - the draft is the machine's reading until the trader clicks.

Lead decision 7: *"Confirm writes a `confirmed` row (the trader's click);
Correct opens the three values for edit ... Until then the draft is
`provisional`: shown as the AI's reading, counted nowhere as the trader's. It
costs the card ONE of the budget of three only when a draft is waiting; it never
greys Save."* And lead decision 8: *"The exit's feelings never touch the session
mood"*.

This is the rule 33 live trades carry a `provisional` tag and exactly ONE a
confirmed one already taught this desk (`scripts/trade_mentor_trade_check.py:18-24`):
a machine guess that retires a question silently is how 215 trades reached one
confirmed label.

RED FOR: `ai_jobs.exit_note_fields`, `trade_mentor_trade_check.confirm_exit_fields`
/ `correct_exit_fields` / `exit_fields`, the `exit_draft_review` registry kind and
the card's `exit_confirm_button` / `exit_correct_button` do not exist on this
branch (verified 2026-09-21 at `05988440`). Each test below fails at
`ModuleNotFoundError` / `AttributeError` until the builder writes them.
"""

from __future__ import annotations

import ast
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for _extra in (SCRIPTS_DIR, ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj9e_support as fx  # noqa: E402

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

MONDAY_MORNING = datetime.fromisoformat("2026-09-14T09:05:00-04:00")

#: The card is the ONLY place a confirmed exit row may be written from, and the
#: module that defines the writer is allowed to name it.
ALLOWED_CONFIRM_CALLERS = {
    "trade_mentor_trade_check.py",
    "ui/widgets/trade_mentor_card.py",
}


def _card(tmp_path):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    return TradeMentorCard(drafts_path=Path(tmp_path) / "drafts.json")


def _drafted(tmp_path, night_root: Path):
    """One trade, one raw note, and ONE draft published by a fake night.

    The draft is written through the slot's real publish path with this file's
    own reply, so the shape under test is the shape the night really stores.
    """
    import exit_reasons
    import trader_state_tags
    from ai_jobs import exit_note_fields

    store, trade_id = fx.swing_with_money(tmp_path)
    reply = fx.good_reply(exit_reasons.codes()[0], trader_state_tags.codes()[:1])
    out = exit_note_fields.run_exit_note_fields(
        session_date=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T02:00:00-04:00"),
        root=night_root,
        store=store,
        request=fx.fake_request(reply),
    )
    assert out["status"] == "ok", out
    return store, trade_id


# ---------------------------------------------------------------------------
# provisional everywhere until the click
# ---------------------------------------------------------------------------
def test_a_drafted_note_reads_provisional_everywhere_until_the_trader_clicks(tmp_path):
    """Hand-counted: 1 draft, status `provisional`, and 0 confirmed rows.

    The word is read from the journal's OWN tag vocabulary
    (`journal_store.TAG_STATUS_PROVISIONAL`), never spelled here: one word for
    "a machine wrote this" across the whole desk.
    """
    import journal_store
    import trade_mentor_trade_check as check
    from ai_jobs import exit_note_fields

    root = tmp_path / "packs"
    store, trade_id = _drafted(tmp_path, root)

    stored = exit_note_fields.read_latest(fx.REVIEWED, root=root)
    assert len(stored["drafts"]) == 1, stored["drafts"]
    assert stored["drafts"][0]["status"] == journal_store.TAG_STATUS_PROVISIONAL

    draft = exit_note_fields.draft_for(trade_id, fx.REVIEWED, root=root)
    assert draft["status"] == journal_store.TAG_STATUS_PROVISIONAL
    assert check.exit_fields(store, trade_id) == {}, "a draft was counted as the trader's"


def test_confirm_is_the_only_thing_that_makes_a_draft_the_traders(tmp_path):
    """The trader's one click. Hand-counted: 1 confirmed row afterwards, holding
    the same three values, and the draft file UNCHANGED - a confirm is an
    append, never a rewrite of what the machine said."""
    import journal_store
    import trade_mentor_trade_check as check
    from ai_jobs import exit_note_fields

    root = tmp_path / "packs"
    store, trade_id = _drafted(tmp_path, root)
    draft = exit_note_fields.draft_for(trade_id, fx.REVIEWED, root=root)
    before = {path.name: path.read_bytes() for path in sorted(root.glob("*.json"))}

    result = check.confirm_exit_fields(store, trade_id, draft, now=MONDAY_MORNING)
    assert result["ok"] is True, result

    saved = check.exit_fields(store, trade_id)
    assert saved["status"] == journal_store.TAG_STATUS_CONFIRMED
    assert saved["fields"]["why"]["code"] == draft["fields"]["why"]["code"]
    assert {path.name: path.read_bytes() for path in sorted(root.glob("*.json"))} == before


def test_correct_writes_the_traders_own_values_and_refuses_an_unknown_code(tmp_path):
    """Correct opens the three values for edit; the codes still come from the two
    closed vocabularies, because a corrected row is still a row a later reader
    has to interpret. Hand-counted: 1 confirmed row with the SECOND why code."""
    import exit_reasons
    import trade_mentor_trade_check as check

    root = tmp_path / "packs"
    store, trade_id = _drafted(tmp_path, root)
    chosen = exit_reasons.codes()[1]

    result = check.correct_exit_fields(
        store, trade_id, why=chosen, felt=(), watching=("my own words",), now=MONDAY_MORNING
    )
    assert result["ok"] is True, result

    saved = check.exit_fields(store, trade_id)
    assert saved["fields"]["why"]["code"] == chosen
    assert saved["fields"]["watching"] == ["my own words"]

    with pytest.raises(ValueError):
        check.correct_exit_fields(
            store, trade_id, why="a_code_nobody_shipped", felt=(), watching=(),
            now=MONDAY_MORNING,
        )


def test_only_the_card_writes_a_confirmed_exit_row():
    """STRUCTURAL, by CONTACT rather than by words (TJ-7's amended guard).

    No nightly slot, no importer, no grader and no service may call the two
    writers. `trade_annotations` is trader-owned and so is this: a confirm is
    the trader's ACT, and a machine that could call it would make the word
    `confirmed` mean nothing.
    """
    offenders: list[str] = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        name = str(path.relative_to(SCRIPTS_DIR)).replace("\\", "/")
        if name in ALLOWED_CONFIRM_CALLERS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover - not our file to fix
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            called_name = (
                called.attr if isinstance(called, ast.Attribute)
                else called.id if isinstance(called, ast.Name)
                else ""
            )
            if called_name in ("confirm_exit_fields", "correct_exit_fields"):
                offenders.append(f"{name}: {called_name}(...)")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# the card - the budget, the gate, and ONE write in flight
# ---------------------------------------------------------------------------
def test_the_draft_line_costs_a_budget_slot_only_when_a_draft_waits():
    """The registry's own rules: a kind names its reader or ships DORMANT, and
    the budget of three is the budget of three.

    Hand-counted: the new kind is budgeted (unlike the FORCED `trade_label`),
    its trigger offers nothing when no draft waits, and `BUDGET` is still 3.
    """
    import mentor_questions

    kinds = {kind.kind: kind for kind in mentor_questions.REGISTRY}
    assert "exit_draft_review" in kinds, sorted(kinds)
    kind = kinds["exit_draft_review"]

    assert mentor_questions.BUDGET == 3
    assert kind.budgeted is True, "the draft line is not a forced row"
    assert str(kind.consumer or "").strip(), "the kind names no reader"
    assert str(kind.answer_key or "").strip()

    # The probe really resolves the reader and really finds the key in it.
    report = mentor_questions.consumer_report()
    row = [item for item in report if str(item.get("kind")) == "exit_draft_review"]
    assert len(row) == 1, report
    assert row[0].get("resolved") is True, row[0]
    assert row[0].get("reads_key") is True, row[0]


def test_the_draft_line_never_greys_save(tmp_path):
    """A waiting draft is something to LOOK at, not a field to fill.

    Hand-counted: the swing's exit box is the only gate. With a draft waiting
    and the box filled, Save is green - a draft nobody clicked must never be
    able to hold the morning hostage.
    """
    import trade_mentor_trade_check as check

    root = tmp_path / "packs"
    store, trade_id = _drafted(tmp_path, root)
    task = check.build_task(store, fx.SESSION_TODAY)

    card = _card(tmp_path)
    card.set_trade_check(task, store=store, drafts_root=root)
    assert card.exit_draft_line(trade_id).strip(), "no draft line was shown"

    card.exit_note_box(trade_id).setPlainText(fx.EXIT_NOTE)
    assert card.save_answers_button.isEnabled() is True, (
        "a waiting draft greyed Save"
    )


def test_one_write_is_in_flight_and_every_ending_re_enables_the_buttons(tmp_path, monkeypatch):
    """Nothing expensive on the Qt thread, and no double write.

    The writer is blocked on an Event, so the test WAITS rather than sleeping.
    Hand-counted: two clicks while one write is in flight -> exactly ONE call,
    both buttons disabled for the whole of it, and both settled once it ends.
    """
    import trade_mentor_trade_check as check

    root = tmp_path / "packs"
    store, trade_id = _drafted(tmp_path, root)
    task = check.build_task(store, fx.SESSION_TODAY)

    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    real_confirm = check.confirm_exit_fields

    def _slow_confirm(*args, **kwargs):
        calls.append("confirm")
        started.set()
        assert release.wait(20.0), "the writer was never released"
        return real_confirm(*args, **kwargs)

    monkeypatch.setattr(check, "confirm_exit_fields", _slow_confirm)

    card = _card(tmp_path)
    card.set_trade_check(task, store=store, drafts_root=root)
    confirm = card.exit_confirm_button(trade_id)
    correct = card.exit_correct_button(trade_id)

    confirm.click()
    assert started.wait(20.0), "the confirm never reached a worker"
    assert confirm.isEnabled() is False, "Confirm stayed clickable mid-write"
    assert correct.isEnabled() is False, "Correct stayed clickable mid-write"

    confirm.click()  # the second click must find nothing to start
    assert len(calls) == 1, f"{len(calls)} writes in flight"

    release.set()
    for _spin in range(200):
        _app.processEvents()
        if not card.exit_write_in_flight(trade_id):
            break
    assert card.exit_write_in_flight(trade_id) is False, "the card never settled"
    assert check.exit_fields(store, trade_id)["fields"]["why"]["code"]


def test_a_write_that_raises_still_re_enables_the_buttons_and_says_so(tmp_path, monkeypatch):
    """*"buttons disabled until EVERY ending answers, including a raise"*.

    A worker whose only failure path is the happy one leaves a card the trader
    cannot use again without a restart. Hand-counted: 1 call, an exception, and
    both buttons clickable afterwards.
    """
    import trade_mentor_trade_check as check

    root = tmp_path / "packs"
    store, trade_id = _drafted(tmp_path, root)
    task = check.build_task(store, fx.SESSION_TODAY)

    def _boom(*_args, **_kwargs):
        raise OSError("the journal database is locked")

    monkeypatch.setattr(check, "confirm_exit_fields", _boom)

    card = _card(tmp_path)
    card.set_trade_check(task, store=store, drafts_root=root)
    card.exit_confirm_button(trade_id).click()

    for _spin in range(200):
        _app.processEvents()
        if not card.exit_write_in_flight(trade_id):
            break
    assert card.exit_write_in_flight(trade_id) is False
    assert card.exit_confirm_button(trade_id).isEnabled() is True
    assert card.exit_correct_button(trade_id).isEnabled() is True
    said = card.status_label.text().lower()
    assert "not" in said and "saved" in said, card.status_label.text()


# ---------------------------------------------------------------------------
# TJ-7's fence - two grains, two rows, no copying
# ---------------------------------------------------------------------------
def test_a_confirmed_exit_row_carries_no_mood_key_at_all(tmp_path):
    """Different grain, different row (lead decision 8).

    Hand-counted: 1 confirmed row; its feelings live under `fields.felt` and the
    row carries no `mood` and no `state_tags` key at any level. A feeling about
    ONE exit is not how the trader felt about the day, and a row that carried
    both would let a reader average them.
    """
    import trade_mentor_trade_check as check
    from ai_jobs import exit_note_fields

    root = tmp_path / "packs"
    store, trade_id = _drafted(tmp_path, root)
    draft = exit_note_fields.draft_for(trade_id, fx.REVIEWED, root=root)
    assert draft["fields"]["felt"], "the fixture draft carries no felt code"

    check.confirm_exit_fields(store, trade_id, draft, now=MONDAY_MORNING)
    saved = check.exit_fields(store, trade_id)

    def _keys(payload):
        if isinstance(payload, dict):
            for key, value in payload.items():
                yield str(key)
                yield from _keys(value)
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                yield from _keys(item)

    names = set(_keys(saved))
    assert "mood" not in names and "state_tags" not in names, sorted(names)
    assert saved["fields"]["felt"], saved


def test_neither_the_exit_writer_nor_the_night_slot_touches_the_mood_seam():
    """STRUCTURAL, by CONTACT (TJ-7's amended guard, `tests/test_tj7_reported_never_acted_on.py:114-141`).

    TJ-7's own invariant already says only the trader's surfaces hand a mood to
    the journal writer. This states the other half for this packet: the two
    modules that handle an exit's feelings never call a mood writer and the
    night slot does not import the journal that holds one.
    """
    import ast as _ast

    for name in ("trade_mentor_trade_check.py", "ai_jobs/exit_note_fields.py"):
        path = SCRIPTS_DIR / name
        assert path.is_file(), f"{name} is not where this test thinks it is"
        tree = _ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            called = node.func
            called_name = (
                called.attr if isinstance(called, _ast.Attribute)
                else called.id if isinstance(called, _ast.Name)
                else ""
            )
            if called_name not in ("build_entry", "write_entry", "build_mood"):
                continue
            for keyword in node.keywords:
                assert keyword.arg not in ("mood", "state_tags", "process"), (
                    f"{name}: {called_name}({keyword.arg}=...)"
                )

    slot = _ast.parse(
        (SCRIPTS_DIR / "ai_jobs" / "exit_note_fields.py").read_text(encoding="utf-8")
    )
    imported: set[str] = set()
    for node in _ast.walk(slot):
        if isinstance(node, _ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "market_journal" not in imported, sorted(imported)
