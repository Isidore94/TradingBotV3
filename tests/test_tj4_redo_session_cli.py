r"""TJ-4 change 4 - `--session` reaches ONE slot, narrowly. Builder's tests.

The tester pinned only the PARSER (`test_tj4_day_review_page.py::
test_the_cli_accepts_the_session_the_redo_names`) and left the rest to the
lead, whose decision (packet correction 2, 2026-09-20) is:

* `--session YYYY-MM-DD` is accepted ONLY together with
  `--slot day_review_narration`; any other use is a parser error, exit 2, and
  nothing runs;
* it is handed to that ONE slot as its `session_date` and reaches no other job;
* `runner.session_date_for`, `runner.night_kind` and every other slot's
  already-done check are untouched, and the night-only rule still holds.

**NO MODEL IS CALLED HERE.** The slot's `run` is replaced by a spy through
`dataclasses.replace`, which is the house pattern, and the runner's machine
lock and launch window are both replaced.
"""

from __future__ import annotations

import dataclasses
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

SLOT = "day_review_narration"
#: A real regular-close session (the TJ-10/TJ-4 fixture day, a Friday).
SESSION = "2026-09-18"
#: 02:00 Eastern the following Monday - inside every shipped night window.
OVERNIGHT = datetime(2026, 9, 21, 2, 0, tzinfo=ZoneInfo("America/New_York"))


@pytest.fixture
def unlocked(monkeypatch):
    import local_writer_lock as lock_mod

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)


@pytest.fixture
def ai_store(tmp_path, monkeypatch):
    root = tmp_path / "ai_store"
    root.mkdir()
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(root))
    return root


@pytest.fixture
def night(monkeypatch):
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (True, "window open"))


def _spy_slot(name: str, seen: list[dict]):
    from ai_jobs import runner

    def _run(**kwargs):
        seen.append(dict(kwargs))
        return {"status": "ok", "model": "", "reason": "spy", "outputs": []}

    return runner.JobSlot(
        name=name, run=_run, reserve_minutes=1.0, max_attempts=3, uses_model=False
    )


# ---------------------------------------------------------------------------
# the parser refuses everything but the one command
# ---------------------------------------------------------------------------


def test_a_session_without_the_day_story_slot_is_refused_and_runs_nothing(monkeypatch):
    """A flag that silently did nothing for every other slot would read as a
    general override of the night's own session date, which it is not."""
    import run_ai_jobs
    from ai_jobs import runner

    def _never(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("nothing may run when the parser refused")

    monkeypatch.setattr(runner, "run_slots", _never)

    with pytest.raises(SystemExit) as refused:
        run_ai_jobs.main(["--session", SESSION])
    assert refused.value.code == 2


def test_a_session_named_beside_another_slot_is_refused(monkeypatch):
    import run_ai_jobs
    from ai_jobs import runner

    monkeypatch.setattr(
        runner, "run_slots", lambda *a, **k: pytest.fail("nothing may run")
    )

    with pytest.raises(SystemExit) as refused:
        run_ai_jobs.main(["--slot", "ai_summary", "--session", SESSION])
    assert refused.value.code == 2


@pytest.mark.parametrize("asked", ["yesterday", "2026-13-01", "2026-09-19"])
def test_a_malformed_or_non_session_date_is_refused(monkeypatch, asked):
    """`2026-09-19` is a SATURDAY: a day the exchange never opened has no pack
    and no story, so the command is refused rather than run to find nothing."""
    import run_ai_jobs
    from ai_jobs import runner

    monkeypatch.setattr(
        runner, "run_slots", lambda *a, **k: pytest.fail("nothing may run")
    )

    with pytest.raises(SystemExit) as refused:
        run_ai_jobs.main(["--slot", SLOT, "--session", asked])
    assert refused.value.code == 2


# ---------------------------------------------------------------------------
# and hands the one slot the one day
# ---------------------------------------------------------------------------


def test_the_asked_session_reaches_the_slot_that_was_named(
    tmp_path, ai_store, unlocked, night
):
    """The whole point of the flag: the Redo button names a day, and the slot
    narrates THAT day rather than the one the clock happens to be on."""
    from ai_jobs import runner

    seen: list[dict] = []
    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots(
        [_spy_slot(SLOT, seen)],
        now=OVERNIGHT,
        only=SLOT,
        ledger_path=led,
        session_override=SESSION,
    )

    assert [call["session_date"] for call in seen] == [SESSION]
    # The night's own identity is untouched: the REPORT still names the session
    # the clock names, and only the row for this slot follows the ask.
    assert report.session_date == runner.session_date_for(OVERNIGHT)
    assert report.results[0]["session_date"] == SESSION


def test_an_override_with_no_named_slot_reaches_nobody(
    tmp_path, ai_store, unlocked, night
):
    """It is the ONE narrow door, and `only` is the doorway. Without a named
    slot the night's own session date is what every slot gets."""
    from ai_jobs import runner

    seen: list[dict] = []
    led = tmp_path / "ledger.jsonl"
    runner.run_slots(
        [_spy_slot("first_slot", seen), _spy_slot("second_slot", seen)],
        now=OVERNIGHT,
        ledger_path=led,
        session_override=SESSION,
    )

    tonight = runner.session_date_for(OVERNIGHT)
    assert [call["session_date"] for call in seen] == [tonight, tonight]


def test_the_day_story_slot_keeps_its_own_run_when_nothing_is_overridden(
    tmp_path, ai_store, unlocked, night
):
    """The default path is byte-identical: no override, no change."""
    from ai_jobs import runner

    seen: list[dict] = []
    led = tmp_path / "ledger.jsonl"
    runner.run_slots([_spy_slot(SLOT, seen)], now=OVERNIGHT, only=SLOT, ledger_path=led)

    assert [call["session_date"] for call in seen] == [runner.session_date_for(OVERNIGHT)]


def test_the_registered_slot_is_replaceable_the_way_this_file_assumes():
    """A guard on the house pattern: the spy above stands in for the real slot
    only while `JobSlot` is a frozen dataclass with one callable."""
    from ai_jobs import runner

    slot = next(s for s in runner.default_slots() if s.name == SLOT)
    swapped = dataclasses.replace(slot, run=lambda **_k: {"status": "ok"})
    assert swapped.name == SLOT
    assert swapped.uses_model is True
