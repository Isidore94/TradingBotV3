"""TJ-10 item R - the nightly slot that closes a matured read.

The packet left the registration to integration; the reviewer moved it here,
because the slot belongs with the fix that makes it safe to run (blocker 1: a
night on a desk with no daily store used to retire a correct `pending` row).

What these pin: the slot exists, it is DETERMINISTIC (`uses_model=False`, no
model, no push), it sits INSIDE the deterministic stage - after the cohort
graders and before `miss_contrast` / `market_story_rollups` / `measured_report`,
which is what keeps it on the Sunday slate - and an unreachable store is a
recorded REASON on an `ok` row rather than an exception into the runner.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
AFTER_THE_HORIZON = datetime(2026, 9, 26, 2, 0, tzinfo=PACIFIC)
FULL_CLOSES = {
    fx.SESSION: 100.0,
    fx.NEXT_SESSIONS[0]: 104.0,
    fx.NEXT_SESSIONS[1]: 105.0,
    fx.NEXT_SESSIONS[2]: 106.0,
    fx.NEXT_SESSIONS[3]: 107.0,
    fx.NEXT_SESSIONS[4]: 109.0,
}


def test_the_slot_is_registered_inside_the_deterministic_stage():
    from ai_jobs import runner

    names = tuple(slot.name for slot in runner.default_slots())

    assert "read_grades_mature" in names, names
    # Inside stage 1: `_deterministic_stage` walks up to and INCLUDING
    # `_STAGE_ONE_LAST_SLOT`, so a slot after that name leaves the Sunday slate.
    assert names.index("read_grades_mature") <= names.index(runner._STAGE_ONE_LAST_SLOT)
    assert names.index("read_grades_mature") > names.index("rejection_cohort_grading")
    for closer in ("miss_contrast", "market_story_rollups", "measured_report"):
        assert names.index("read_grades_mature") < names.index(closer), closer


def test_the_slot_is_deterministic_and_capped():
    from ai_jobs import runner

    slot = {item.name: item for item in runner.default_slots()}["read_grades_mature"]

    assert slot.enabled
    assert slot.uses_model is False
    assert slot.max_attempts == 3
    assert 0 < slot.reserve_minutes <= 5.0


def test_the_slot_is_on_the_sunday_slate():
    from ai_jobs import runner

    names = [slot.name for slot in runner._deterministic_stage(runner.default_slots())]

    assert "read_grades_mature" in names


def test_the_slot_closes_a_matured_read_and_says_so(monkeypatch, tmp_path):
    import chart_snapshot
    import d1_environment_store
    import market_read_grades as grader
    from ai_jobs import read_grades_mature

    entry = fx.old_entry(
        "empty", text="D1 SPY is uptrending into next week", timeframe="D1"
    )
    row = dict(grader.read_rows([entry], session=fx.SESSION)[0])
    row["direction"] = "up"
    pending = grader.grade_read(
        row,
        daily_bars=fx.daily_bars({fx.SESSION: 100.0}),
        atr=fx.atr_for(9.0, 3.0),
        now=datetime(2026, 9, 21, 17, 0, tzinfo=PACIFIC),
    )
    assert pending["verdict"].startswith(grader.PENDING_PREFIX)
    grader.append_grades(fx.SESSION, [pending], root=tmp_path)

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])
    monkeypatch.setattr(
        d1_environment_store, "_cached_daily_bars",
        lambda _symbol: fx.daily_bars(
            {**FULL_CLOSES, **{f"2026-07-{day:02d}": 90.0 + day for day in range(1, 20)}}
        ),
    )

    outcome = read_grades_mature.run_read_grades_mature(
        session_date=fx.SESSION, now=AFTER_THE_HORIZON, root=tmp_path
    )

    assert outcome["status"] == "ok"
    assert outcome["model"] == ""
    assert "closed 1 matured read" in outcome["reason"]
    stored = grader.read_grades(fx.SESSION, root=tmp_path)
    assert len(stored) == 2
    assert stored[1]["supersedes"] == pending["grade_id"]


def test_an_unreachable_store_is_a_reason_and_never_an_exception(monkeypatch, tmp_path):
    import market_read_grades as grader
    from ai_jobs import read_grades_mature

    def _boom(*_args, **_kwargs):
        raise OSError("the day_review folder is unreadable")

    monkeypatch.setattr(grader, "regrade_matured", _boom)

    outcome = read_grades_mature.run_read_grades_mature(
        session_date=fx.SESSION, now=AFTER_THE_HORIZON, root=tmp_path
    )

    assert outcome["status"] == "ok"
    assert "unreadable" in outcome["reason"]
    assert outcome["outputs"] == []


def test_an_empty_ledger_is_a_fact_about_the_night(tmp_path):
    from ai_jobs import read_grades_mature

    outcome = read_grades_mature.run_read_grades_mature(
        session_date=fx.SESSION, now=AFTER_THE_HORIZON, root=tmp_path
    )

    assert outcome["status"] == "ok"
    assert outcome["reason"] == read_grades_mature.NOTHING_OPEN


def test_the_slot_calls_no_model_and_reaches_no_notifier(monkeypatch, tmp_path):
    import push_notify
    from ai_jobs import read_grades_mature

    def _never(*_args, **_kwargs):
        raise AssertionError("the nightly re-grade reached a model or the phone")

    monkeypatch.setattr(push_notify, "send_push", _never)
    monkeypatch.setattr(push_notify, "build_push_request", _never)
    import ai_summary

    monkeypatch.setattr(ai_summary, "summarize", _never, raising=False)

    read_grades_mature.run_read_grades_mature(
        session_date=fx.SESSION, now=AFTER_THE_HORIZON, root=tmp_path
    )
