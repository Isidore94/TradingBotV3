"""Trade Mentor prefetch queue, Inbox and commands: interactive first, quiet by default."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import commands, prefetch  # noqa: E402
from mentor_app.inbox import Inbox  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")


# --------------------------------------------------------------------------- prefetch
def test_an_interactive_job_is_served_before_queued_background_jobs():
    queue = prefetch.PrefetchQueue()
    queue.submit("embed", lambda: None, priority=prefetch.PRIORITY_EMBED, needs_model=True)
    queue.submit("refresh", lambda: None, priority=prefetch.PRIORITY_REFRESH, needs_model=True)
    queue.submit("ask", lambda: None, priority=prefetch.PRIORITY_INTERACTIVE, needs_model=True)
    order = []
    while queue.run_one():
        order.append(queue.ran[-1])
    assert order == ["ask", "refresh", "embed"]


def test_background_model_jobs_wait_while_a_chat_turn_is_in_flight():
    queue = prefetch.PrefetchQueue()
    queue.submit("assess", lambda: None, priority=prefetch.PRIORITY_REFRESH, needs_model=True)
    queue.submit("context", lambda: None, priority=prefetch.PRIORITY_REFRESH, needs_model=False)
    queue.begin_interactive()
    assert queue.run_one() and queue.ran == ["context"], "a deterministic rebuild may still run"
    assert not queue.run_one(), "no model job may start behind a chat turn"
    queue.end_interactive()
    assert queue.run_one() and queue.ran[-1] == "assess"


def test_one_slot_mode_asks_a_running_model_job_to_yield_to_a_chat_turn():
    queue = prefetch.PrefetchQueue()
    queue.begin_interactive()
    assert not queue.should_yield(), "two slots: the chat turn has its own slot"
    queue.set_single_slot(True)
    assert queue.should_yield(), "one slot (night-started serve): a running job stops at its next step"
    queue.end_interactive()
    assert not queue.should_yield()


def test_model_jobs_pause_in_the_night_window_and_deterministic_ones_run():
    reason = {"text": "night"}
    queue = prefetch.PrefetchQueue(blocked=lambda: reason["text"])
    queue.submit("embed", lambda: None, priority=prefetch.PRIORITY_EMBED, needs_model=True)
    queue.submit("context", lambda: 1, needs_model=False)
    assert queue.run_one() and queue.ran == ["context"]
    assert not queue.run_one() and queue.pending() == ["embed"]
    reason["text"] = ""
    assert queue.run_one() and queue.ran[-1] == "embed"


def test_a_job_may_not_ask_for_more_than_600_output_tokens():
    queue = prefetch.PrefetchQueue()
    assert queue.submit("ok", lambda: None, max_tokens=600)
    with pytest.raises(ValueError):
        queue.submit("big", lambda: None, max_tokens=601)


def test_the_same_key_is_queued_once_and_a_failure_does_not_stop_the_queue():
    queue = prefetch.PrefetchQueue()
    assert queue.submit("a", lambda: 1 / 0, key="context")
    assert not queue.submit("a again", lambda: None, key="context")
    errors: list = []
    queue.submit("b", lambda: 2, on_done=errors.append)
    assert queue.run_one() and queue.run_one()
    assert errors == [2]


def test_the_consumer_thread_runs_jobs_one_at_a_time():
    import threading

    queue = prefetch.PrefetchQueue()
    done = threading.Event()
    queue.submit("x", lambda: None, on_done=lambda _: done.set())
    queue.start()
    try:
        assert done.wait(5)
    finally:
        queue.stop()


# --------------------------------------------------------------------------- inbox
def _clock(start):
    state = {"now": start}
    return state, (lambda: state["now"])


def test_the_daily_cap_refuses_the_seventh_item():
    state, now = _clock(datetime(2026, 9, 30, 9, 0, tzinfo=PT))
    inbox = Inbox(per_day_cap=6, now=now)
    added = [inbox.add("question", f"q{i}") for i in range(7)]
    assert all(added[:6]) and added[6] is None
    assert "cap" in inbox.last_refusal and inbox.badge() == 6
    state["now"] += timedelta(days=1)
    assert inbox.add("question", "tomorrow") is not None


@pytest.mark.parametrize(("hh", "mm", "quiet"), [(6, 29, False), (6, 30, True), (6, 59, True), (7, 0, False)])
def test_quiet_hours_around_the_open(hh, mm, quiet):
    _, now = _clock(datetime(2026, 9, 30, hh, mm, tzinfo=PT))
    inbox = Inbox(now=now)
    assert (inbox.add("card", "x") is None) is quiet


def test_quiet_command_mutes_and_items_expire():
    state, now = _clock(datetime(2026, 9, 30, 9, 0, tzinfo=PT))
    inbox = Inbox(now=now)
    inbox.add("card", "short-lived", ttl=timedelta(minutes=30))
    inbox.mute(timedelta(hours=2))
    assert inbox.add("card", "muted") is None and "muted" in inbox.last_refusal
    state["now"] += timedelta(hours=2)
    assert inbox.items() == [] and inbox.add("card", "back") is not None


def test_the_app_never_pops_beeps_or_pushes():
    banned = re.compile(r"send_push|push_notify|QSystemTrayIcon|showMessage|\.beep\(|QSound|QSoundEffect|winsound")
    for path in sorted((SCRIPTS_DIR / "mentor_app").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not banned.search(text), f"{path.name} may not pop, beep or push"


def test_the_app_never_touches_order_code():
    banned = re.compile(r"place_order|placeOrder|ib_insync|bounce_bot_lib|m5_signal_engines|master_avwap_lib\.legacy")
    for folder in ("mentor_app", "mentor_packs"):
        for path in sorted((SCRIPTS_DIR / folder).glob("*.py")):
            assert not banned.search(path.read_text(encoding="utf-8")), path.name


# --------------------------------------------------------------------------- commands
@pytest.mark.parametrize(
    ("text", "minutes"), [("2h", 120), ("30m", 30), ("1h30m", 90), ("45", 45), ("0", None), ("soon", None)]
)
def test_parse_duration(text, minutes):
    got = commands.parse_duration(text)
    assert (got.total_seconds() / 60 if got else None) == minutes


def test_commands_route_and_stub():
    assert commands.handle("hello") is None
    assert commands.handle("/help").action == "help"
    quiet = commands.handle("/quiet 2h")
    assert quiet.action == "quiet" and quiet.arg == timedelta(hours=2)
    assert commands.handle("/remember I stop after two losses").arg == "I stop after two losses"
    assert commands.handle("/remember").action == "error"
    assert commands.handle("/tape").action == "tape"
    # P2 built /pick: it routes to the window with the symbol (was a Phase 2 stub).
    assert commands.handle("/pick nvda").action == "pick" and commands.handle("/pick nvda").arg == ("NVDA", "")
    # P3 built /vetoes: it routes to the window with the optional date (was a Phase 3 stub).
    assert commands.handle("/vetoes").action == "vetoes" and commands.handle("/vetoes").arg == ""
    assert commands.handle("/vetoes 2026-09-29").arg == "2026-09-29"
    assert commands.handle("/vetoes yesterday").action == "error"
    assert commands.handle("/scorecard").action == "scorecard"
    assert commands.handle("/nope").action == "error"


def test_the_clock_used_by_the_inbox_is_tz_aware():
    _, now = _clock(datetime(2026, 9, 30, 9, 0, tzinfo=PT))
    item = Inbox(now=now).add("card", "x", ttl=timedelta(hours=1))
    assert item.created_utc.tzinfo is timezone.utc and item.expires_utc.tzinfo is timezone.utc
