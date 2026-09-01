"""A Focus pick that never says anything eventually fades - Phase 0.12 A3.

Trader, 2026-09-01. A Focus list only means "the names I am watching" while
something takes names off it. A pick that has fired no alert and printed no
pullback event for ten trading days is not being watched; it is furniture.

The rules this file pins, each because it is a way the fade could do harm:

* the clock is SESSIONS, and it starts at ADD time;
* activity RESETS it - a fired Focus alert, an armed-watch hit, or the
  trader's own "keep in Focus" on the review chart;
* it applies to swing AND M5 picks, the trader's own included. That is an
  explicit authorization to remove a name the trader typed, so it is scoped
  to Focus and goes through the store's own removal path - a hand-maintained
  watchlist line is still never touched (plan.md sec 5);
* it is REVERSIBLE and nothing is lost: a faded pick lands in a faded list
  with an append-only row behind it, and restoring gives it a fresh clock;
* uncertainty never fades - a date the calendar cannot reason about stays.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _make_store(tmp_path):
    from focus_picks import FocusPickStore

    return FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "focus_pick_membership.json",
    )


def _symbols(path):
    from watchlist_utils import read_watchlist_symbols

    return read_watchlist_symbols(path)


def test_an_added_pick_is_stamped_with_the_day_it_arrived(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    assert store.pick_clock("NVDA", "long", "swing") == date(2026, 8, 3)


def test_a_pick_present_with_no_stamp_gets_today_never_an_older_guess(tmp_path):
    """First load after the upgrade. Guessing backwards would fade the
    trader's whole list on the first slow tick."""
    (tmp_path / "focus_longs.txt").write_text("NVDA\n", encoding="utf-8")
    store = _make_store(tmp_path)
    assert store.pick_clock("NVDA", "long", "m5") == date.today()


def test_a_quiet_pick_fades_after_ten_trading_days(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    # Nine sessions later: still live.
    assert store.fade_stale_picks(today=date(2026, 8, 14)) == []
    assert store.focus_symbols("long", "swing") == ["NVDA"]
    # The tenth session: it fades.
    faded = store.fade_stale_picks(today=date(2026, 8, 17))
    assert [row["symbol"] for row in faded] == ["NVDA"]
    assert store.focus_symbols("long", "swing") == []
    assert [row["symbol"] for row in store.faded_picks()] == ["NVDA"]


def test_activity_resets_the_clock(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    store.note_focus_activity("NVDA", reason="focus_d1_flag", today=date(2026, 8, 10))
    assert store.pick_clock("NVDA", "long", "swing") == date(2026, 8, 10)
    assert store.fade_stale_picks(today=date(2026, 8, 17)) == []
    assert store.fade_stale_picks(today=date(2026, 8, 24)) != []


def test_fading_uninjects_only_what_focus_injected(tmp_path):
    """The CandidateRegistry invariant survives: a name the trader maintains
    in the broad list themselves is left exactly where it is."""
    (tmp_path / "longs.txt").write_text("AMD\n", encoding="utf-8")
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "m5", today=date(2026, 8, 3))
    store.add("AMD", "long", "m5", today=date(2026, 8, 3))
    store.fade_stale_picks(today=date(2026, 8, 17))
    assert _symbols(tmp_path / "longs.txt") == ["AMD"]


def test_an_undateable_stamp_never_fades(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    store._pick_clocks["NVDA|long|swing"]["clock_from"] = "not-a-date"
    assert store.fade_stale_picks(today=date(2026, 8, 24)) == []
    assert store.focus_symbols("long", "swing") == ["NVDA"]


def test_restoring_a_faded_pick_gives_it_a_fresh_clock(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    store.fade_stale_picks(today=date(2026, 8, 17))
    assert store.restore_faded("NVDA", "long", "swing", today=date(2026, 8, 17)) is True
    assert store.focus_symbols("long", "swing") == ["NVDA"]
    assert store.pick_clock("NVDA", "long", "swing") == date(2026, 8, 17)
    assert store.faded_picks() == []
    # Restoring is not a fade-proof: it is a fresh ten sessions, no more.
    assert store.fade_stale_picks(today=date(2026, 8, 31)) != []


def test_discarding_a_faded_pick_clears_the_list_and_leaves_the_evidence(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    store.fade_stale_picks(today=date(2026, 8, 17))
    assert store.discard_faded("NVDA", "long", "swing") is True
    assert store.faded_picks() == []
    assert store.focus_symbols("long", "swing") == []
    rows = [json.loads(line) for line in _fade_rows(tmp_path)]
    assert [row["event"] for row in rows] == ["focus_pick_faded", "focus_pick_discarded"]


def test_the_faded_list_survives_a_restart(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    store.fade_stale_picks(today=date(2026, 8, 17))
    reopened = _make_store(tmp_path)
    assert [row["symbol"] for row in reopened.faded_picks()] == ["NVDA"]


def test_a_fade_row_names_what_it_removed_and_when(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing", today=date(2026, 8, 3))
    store.fade_stale_picks(today=date(2026, 8, 17))
    row = json.loads(_fade_rows(tmp_path)[0])
    assert row["symbol"] == "NVDA"
    assert row["side"] == "long"
    assert row["category"] == "swing"
    assert row["clock_from"] == "2026-08-03"
    assert row["faded_on"] == "2026-08-17"
    assert row["trading_days"] == 10
    assert row["schema"] == "focus_fade_event_v1"


def _fade_rows(tmp_path) -> list[str]:
    path = tmp_path / "focus_fade_events.jsonl"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
