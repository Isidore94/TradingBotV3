"""The combined M5 Focus adoption gate (plan.md Phase 0.5, packet R2 Part A).

Trader rule 2026-08-14: an auto M5 Focus pick must be **above the previous
day's high AND above session VWAP** on the M5 for longs, and below both for
shorts. The same test runs at candidate build, on every staging refresh (a
pick that falls back through either level is evicted), and again at adoption.

This file opens with the golden characterization fixture required by
plan.md sec 5 before any detector/routing change: it freezes what the
candidate filter selected BEFORE the VWAP half existed, so the gate's effect
on selection is a reviewable diff rather than an assertion nobody can check.
"""

import sys
from pathlib import Path

import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "auto_pick_focus_gate_v1"


def _filtered(fixture):
    import autopilot_core as core

    result = core.filter_candidates_by_prev_day_extremes(
        fixture["candidates"], fixture["profiles"], fixture["daily_context"]
    )
    return {
        side: sorted(row["symbol"] for row in result[side])
        for side in ("longs", "shorts")
    }


def test_candidate_filter_golden_fixture():
    """Loading re-verifies raw_input_sha256 over the fixture's own inputs, so
    editing a profile without re-freezing the expectations fails here."""
    fixture = load_fixture_contract(FIXTURE_NAME)
    assert fixture.schema == "auto_pick_focus_gate_v1"
    fixture.assert_matches(_filtered(fixture), fixture["expected"], "candidate filter")


def test_the_gate_removed_exactly_the_documented_rows():
    """Name the difference the fixture's re-freeze recorded.

    The baseline kept 5 longs and 2 shorts. Three of those survivors were the
    cases the VWAP half exists for, and two more are the completed-bar rule
    biting. Spelling them out here means a future edit that quietly widens the
    gate cannot hide inside a re-frozen expectation.
    """
    fixture = load_fixture_contract(FIXTURE_NAME)
    kept = _filtered(fixture)
    assert kept["longs"] == ["LONG_PDH_ABOVE_VWAP"]
    assert kept["shorts"] == ["SHORT_PDL_BELOW_VWAP"]


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_a_long_needs_both_halves():
    from focus_adoption_gate import passes_focus_adoption_gate

    # Above yesterday's high AND above VWAP.
    passes, reason = passes_focus_adoption_gate("long", 103.0, 100.5, 98.0, 101.0)
    assert passes and "above" in reason

    # Above the high, under VWAP.
    passes, reason = passes_focus_adoption_gate("long", 103.0, 100.5, 98.0, 104.0)
    assert not passes and reason == "not above session VWAP"

    # Above VWAP, inside yesterday's range.
    passes, reason = passes_focus_adoption_gate("long", 100.0, 100.5, 98.0, 99.0)
    assert not passes and reason == "not above yesterday's high"

    # Neither.
    passes, reason = passes_focus_adoption_gate("long", 99.0, 100.5, 98.0, 99.5)
    assert not passes and reason == "not above yesterday's high and not above session VWAP"


def test_a_short_mirrors_it():
    from focus_adoption_gate import passes_focus_adoption_gate

    passes, _ = passes_focus_adoption_gate("short", 95.0, 100.5, 98.0, 97.0)
    assert passes

    passes, reason = passes_focus_adoption_gate("short", 95.0, 100.5, 98.0, 94.0)
    assert not passes and reason == "not below session VWAP"

    passes, reason = passes_focus_adoption_gate("short", 99.0, 100.5, 98.0, 100.0)
    assert not passes and reason == "not below yesterday's low"


def test_unknown_always_fails_and_says_which_half():
    """plan.md sec 5: missing data is uncertainty, never confirmation."""
    from focus_adoption_gate import UNKNOWN, focus_adoption_gate_state

    state, reason = focus_adoption_gate_state("long", 103.0, 100.5, 98.0, None)
    assert state == UNKNOWN and reason == "cannot verify session VWAP"

    state, reason = focus_adoption_gate_state("long", 103.0, None, None, 101.0)
    assert state == UNKNOWN and "break of yesterday's high" in reason

    state, reason = focus_adoption_gate_state("long", None, 100.5, 98.0, 101.0)
    assert state == UNKNOWN

    state, reason = focus_adoption_gate_state("long", None, None, None, None)
    assert state == UNKNOWN and "no completed price" in reason


def test_a_non_finite_price_or_vwap_is_unknown_not_a_pass():
    """NaN compares False against everything, so an unguarded gate would read
    a NaN VWAP as 'not above' - or worse, a NaN price as a pass on the other
    side. Both must be UNKNOWN."""
    from focus_adoption_gate import UNKNOWN, focus_adoption_gate_state

    for bad in (float("nan"), float("inf"), "not a number"):
        state, _ = focus_adoption_gate_state("long", 103.0, 100.5, 98.0, bad)
        assert state == UNKNOWN, f"vwap={bad!r}"
        state, _ = focus_adoption_gate_state("long", bad, 100.5, 98.0, 101.0)
        assert state == UNKNOWN, f"price={bad!r}"


def test_the_gate_reads_the_completed_bar_not_the_forming_one():
    """A break the forming bar made and then closed back inside is the noise
    this gate removes, so the candidate verdict reads `last_complete`."""
    import autopilot_core as core

    ctx = {"prev_high": 100.5, "prev_low": 98.0}
    forming_only = {"last": 103.0, "last_complete": 100.2, "completed_session_vwap": 99.0}
    passes, reason = core.candidate_focus_gate_verdict("long", forming_only, ctx)
    assert not passes and reason == "not above yesterday's high"

    completed = {"last": 100.2, "last_complete": 103.0, "completed_session_vwap": 99.0}
    passes, _ = core.candidate_focus_gate_verdict("long", completed, ctx)
    assert passes, "a break the completed bar holds still qualifies"


def test_the_filter_fails_open_only_when_there_is_no_daily_store_at_all():
    """One unknown symbol fails alone; a missing store never empties the list."""
    import autopilot_core as core

    candidates = {"longs": [{"symbol": "AAA"}, {"symbol": "BBB"}], "shorts": []}
    profiles = {
        "AAA": {"last_complete": 103.0, "completed_session_vwap": 101.0},
        "BBB": {"last_complete": 103.0, "completed_session_vwap": None},
    }
    context = {
        "AAA": {"prev_high": 100.5, "prev_low": 98.0},
        "BBB": {"prev_high": 100.5, "prev_low": 98.0},
    }
    kept = core.filter_candidates_by_prev_day_extremes(candidates, profiles, context)
    assert [row["symbol"] for row in kept["longs"]] == ["AAA"]

    for empty in (None, {}):
        untouched = core.filter_candidates_by_prev_day_extremes(candidates, profiles, empty)
        assert [row["symbol"] for row in untouched["longs"]] == ["AAA", "BBB"]


def test_the_refusals_are_logged_with_their_reasons():
    """An evicted or refused pick's disappearance has to be explainable."""
    import autopilot_core as core

    lines: list[str] = []
    core.filter_candidates_by_prev_day_extremes(
        {"longs": [{"symbol": "AAA"}], "shorts": []},
        {"AAA": {"last_complete": 103.0, "completed_session_vwap": 104.0}},
        {"AAA": {"prev_high": 100.5, "prev_low": 98.0}},
        log=lines.append,
    )
    assert any("AAA (not above session VWAP)" in line for line in lines)


def test_session_vwap_rides_on_the_completed_bars_of_this_session():
    """The profile's VWAP must come from the same completed bars as its close,
    and restart on the session boundary rather than blending yesterday."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import autopilot_core as core

    tz = ZoneInfo("America/New_York")
    yesterday = datetime(2026, 7, 1, 9, 30, tzinfo=tz)
    today = datetime(2026, 7, 2, 9, 30, tzinfo=tz)
    rows = []
    # Yesterday traded far away from today's prices; if it leaked into the
    # accumulator the VWAP below would be nowhere near 100.
    for index in range(6):
        rows.append({
            "dt": yesterday + timedelta(minutes=5 * index),
            "open": 50.0, "high": 50.5, "low": 49.5, "close": 50.0, "volume": 1000.0,
        })
    for index in range(6):
        rows.append({
            "dt": today + timedelta(minutes=5 * index),
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0,
        })

    metrics = core._intraday_extreme_metrics(rows, now=today + timedelta(minutes=30))
    assert metrics["completed_bar_count"] == 6
    assert metrics["completed_session_vwap"] == pytest.approx(100.0, abs=0.01)


def test_a_profile_without_bars_carries_no_vwap():
    import autopilot_core as core

    metrics = core._intraday_extreme_metrics([], now=None)
    assert metrics["completed_session_vwap"] is None


# ---------------------------------------------------------------------------
# Staging: evict what has fallen back, stamp what still qualifies
# ---------------------------------------------------------------------------


def _staged(tmp_path, profiles, context, candidates, *, now, log=None):
    import autopilot_core as core

    return core.stage_auto_populate_candidates(
        candidates,
        "neutral_chop",
        profiles=profiles,
        daily_context=context,
        pending_path=tmp_path / "pending.json",
        membership_path=tmp_path / "membership.json",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        now=now,
        log=log,
    )


def _pending(tmp_path, now):
    """The queue is day-scoped, so it must be read on the same clock it was
    written on - otherwise every fixture reads as yesterday's leftovers."""
    import autopilot_core as core

    return core.load_auto_populate_pending_picks(tmp_path / "pending.json", now=now)


PASSING = {"last_complete": 103.0, "completed_session_vwap": 101.0}
FAILING = {"last_complete": 103.0, "completed_session_vwap": 104.0}
CTX = {"AAA": {"prev_high": 100.5, "prev_low": 98.0}}
ROWS = {"longs": [{"symbol": "AAA", "reason": "fixture", "score": 2.0}], "shorts": []}


def test_a_staged_pick_carries_a_fresh_gate_verdict(tmp_path):
    from datetime import datetime

    moment = datetime(2026, 7, 2, 9, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)
    entry = _pending(tmp_path, moment)["pending"]["long"]["AAA"]
    assert entry["gate_state"] == "open"
    assert entry["gate_checked_at"] == moment.isoformat(timespec="seconds")


def test_a_queued_pick_that_falls_back_through_vwap_is_evicted(tmp_path):
    from datetime import datetime, timedelta

    moment = datetime(2026, 7, 2, 9, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)
    assert "AAA" in _pending(tmp_path, moment)["pending"]["long"]

    lines: list[str] = []
    # Half an hour later it is back under VWAP: out of the queue, with a reason.
    summary = _staged(
        tmp_path, {"AAA": FAILING}, CTX, {"longs": [], "shorts": []},
        now=moment + timedelta(minutes=30), log=lines.append,
    )
    assert "AAA" not in _pending(tmp_path, moment + timedelta(minutes=30))["pending"]["long"]
    assert summary["evicted"]["long"] == ["AAA (not above session VWAP)"]
    assert any("evicted" in line and "AAA" in line for line in lines)


def test_an_evicted_pick_may_re_propose_when_it_qualifies_again(tmp_path):
    """Trader decision 2026-08-15: the queue says what qualifies NOW.

    A name that pulled back mid-morning and broke out cleanly in the afternoon
    is the setup, not the noise.
    """
    from datetime import datetime, timedelta

    moment = datetime(2026, 7, 2, 9, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)
    _staged(tmp_path, {"AAA": FAILING}, CTX, {"longs": [], "shorts": []}, now=moment + timedelta(minutes=30))
    assert "AAA" not in _pending(tmp_path, moment + timedelta(minutes=30))["pending"]["long"]

    later = moment + timedelta(hours=4)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=later)
    entry = _pending(tmp_path, later)["pending"]["long"]["AAA"]
    assert entry["gate_checked_at"] == later.isoformat(timespec="seconds")


def test_a_surviving_pick_is_restamped_rather_than_left_to_age(tmp_path):
    from datetime import datetime, timedelta

    moment = datetime(2026, 7, 2, 9, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)
    later = moment + timedelta(minutes=30)
    _staged(tmp_path, {"AAA": PASSING}, CTX, {"longs": [], "shorts": []}, now=later)
    entry = _pending(tmp_path, later)["pending"]["long"]["AAA"]
    assert entry["gate_checked_at"] == later.isoformat(timespec="seconds")


def test_missing_evidence_never_empties_the_queue(tmp_path):
    """An unmeasurable queue is not an empty one - the verdict ages instead."""
    from datetime import datetime, timedelta

    moment = datetime(2026, 7, 2, 9, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)
    for profiles, context in (({}, CTX), ({"AAA": PASSING}, {}), (None, None)):
        _staged(
            tmp_path, profiles, context, {"longs": [], "shorts": []},
            now=moment + timedelta(minutes=30),
        )
        assert "AAA" in _pending(tmp_path, moment)["pending"]["long"]
    # ... and the stamp did not advance, so adoption still ages it out.
    entry = _pending(tmp_path, moment)["pending"]["long"]["AAA"]
    assert entry["gate_checked_at"] == moment.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Adoption reads the stored verdict
# ---------------------------------------------------------------------------


def test_adoption_accepts_only_a_fresh_passing_verdict():
    from datetime import datetime, timedelta

    from autopilot_core import FOCUS_GATE_VERDICT_MAX_AGE_MINUTES, pending_pick_gate_ok

    now = datetime(2026, 7, 2, 11, 2)          # latest completed M5 bar ended 11:00
    fresh = {
        "gate_state": "open",
        "gate_reason": "ok",
        "gate_checked_at": (now - timedelta(minutes=1)).isoformat(),
        "gate_bar_end": datetime(2026, 7, 2, 11, 0).isoformat(),
    }
    assert pending_pick_gate_ok(fresh, now)[0]

    stale = dict(
        fresh,
        gate_checked_at=(now - timedelta(minutes=FOCUS_GATE_VERDICT_MAX_AGE_MINUTES + 1)).isoformat(),
    )
    ok, reason = pending_pick_gate_ok(stale, now)
    assert not ok and "old" in reason

    failing = dict(fresh, gate_state="closed", gate_reason="not above session VWAP")
    ok, reason = pending_pick_gate_ok(failing, now)
    assert not ok and reason == "not above session VWAP"


def test_adoption_refuses_a_verdict_measured_too_many_bars_ago():
    """The real staleness guard (R2.1).

    45 minutes of wall clock is up to nine M5 bars - long enough for a breakout
    to complete, fail and reverse - so the binding limit is which BAR was
    measured, not when the clock was read.
    """
    from datetime import datetime

    from autopilot_core import FOCUS_GATE_MAX_BAR_LAG, pending_pick_gate_ok

    now = datetime(2026, 7, 2, 11, 2)          # latest completed bar ended 11:00

    def verdict(bar_end):
        return {
            "gate_state": "open",
            "gate_reason": "ok",
            # Wall clock deliberately current, so only the bar can fail it.
            "gate_checked_at": now.isoformat(),
            "gate_bar_end": bar_end.isoformat(),
        }

    assert pending_pick_gate_ok(verdict(datetime(2026, 7, 2, 11, 0)), now)[0], "current bar"
    assert pending_pick_gate_ok(verdict(datetime(2026, 7, 2, 10, 50)), now)[0], "2 bars is the limit"

    ok, reason = pending_pick_gate_ok(verdict(datetime(2026, 7, 2, 10, 45)), now)
    assert not ok and "M5 bars ago" in reason

    # The nine-bar case the review named: fresh clock, ancient tape.
    ok, reason = pending_pick_gate_ok(verdict(datetime(2026, 7, 2, 10, 15)), now)
    assert not ok and "9 M5 bars ago" in reason

    assert FOCUS_GATE_MAX_BAR_LAG == 2


def test_the_accepted_two_bar_exposure_still_has_the_backstop_it_names():
    """R2.2 item 3: the tolerance is accepted BECAUSE something removes the
    residue.

    Two bars of lag can adopt a name that crossed back through VWAP in the
    unseen bars. That is a trader-accepted exposure, bounded by BounceBot's
    triple-VWAP invalidation removing the pick within roughly four completed
    bars. If either constant moves, the documented bound in
    `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` §11.2 and in
    FOCUS_GATE_MAX_BAR_LAG's own comment is no longer the truth, and this test
    is what says so.
    """
    from autopilot_core import FOCUS_GATE_MAX_BAR_LAG
    from bounce_bot_lib.legacy import VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES

    assert FOCUS_GATE_MAX_BAR_LAG == 2
    assert VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES == 4


def test_adoption_refuses_a_verdict_with_no_measured_bar_at_all():
    from datetime import datetime

    from autopilot_core import pending_pick_gate_ok

    now = datetime(2026, 7, 2, 11, 2)
    ok, reason = pending_pick_gate_ok(
        {"gate_state": "open", "gate_checked_at": now.isoformat()}, now
    )
    assert not ok and "no measured bar" in reason

    ok, reason = pending_pick_gate_ok(
        {"gate_state": "open", "gate_checked_at": now.isoformat(), "gate_bar_end": "nonsense"},
        now,
    )
    assert not ok and "unreadable measured-bar" in reason


def test_the_latest_completed_bar_is_the_floor_not_the_current_one():
    from datetime import datetime

    from autopilot_core import latest_completed_m5_end

    assert latest_completed_m5_end(datetime(2026, 7, 2, 11, 7, 30)) == datetime(2026, 7, 2, 11, 5)
    assert latest_completed_m5_end(datetime(2026, 7, 2, 11, 5)) == datetime(2026, 7, 2, 11, 5)
    assert latest_completed_m5_end(datetime(2026, 7, 2, 11, 4, 59)) == datetime(2026, 7, 2, 11, 0)


def test_adoption_refuses_a_pick_nothing_has_measured():
    """Missing is refused for the same reason UNKNOWN fails everywhere else."""
    from datetime import datetime

    from autopilot_core import pending_pick_gate_ok

    now = datetime(2026, 7, 2, 11, 0)
    assert not pending_pick_gate_ok(None, now)[0]
    assert not pending_pick_gate_ok({}, now)[0]
    assert not pending_pick_gate_ok({"gate_state": "open"}, now)[0]
    assert not pending_pick_gate_ok({"gate_state": "open", "gate_checked_at": "nonsense"}, now)[0]
    # A stamp from the future is not fresh, it is a broken clock.
    assert not pending_pick_gate_ok(
        {"gate_state": "open", "gate_checked_at": datetime(2026, 7, 2, 23, 0).isoformat()}, now
    )[0]


def test_a_verdict_measured_before_the_return_to_the_desk_is_refused():
    """The flip barrier (R2.2 item 1).

    Wall-clock age and bar lag both say this verdict is fine; it is refused
    purely because it was measured before the trader came back, which is the
    one thing the drain must not depend on the other bounds to catch.
    """
    from datetime import datetime, timedelta

    from autopilot_core import pending_pick_gate_ok

    now = datetime(2026, 7, 2, 11, 2)
    flip = datetime(2026, 7, 2, 11, 1)
    verdict = {
        "gate_state": "open",
        "gate_reason": "ok",
        "gate_bar_end": datetime(2026, 7, 2, 11, 0).isoformat(),
    }

    before = dict(verdict, gate_checked_at=(flip - timedelta(seconds=1)).isoformat())
    ok, reason = pending_pick_gate_ok(before, now, not_before=flip)
    assert not ok and "predates the return to the desk" in reason
    assert pending_pick_gate_ok(before, now)[0], "and it passes every other bound"

    # Inclusive at the flip's own second: `gate_checked_at` is stamped to the
    # second, so a re-measurement finishing inside that second is after it.
    same = dict(verdict, gate_checked_at=flip.isoformat(timespec="seconds"))
    assert pending_pick_gate_ok(same, now, not_before=flip)[0]
    after = dict(verdict, gate_checked_at=(flip + timedelta(seconds=1)).isoformat())
    assert pending_pick_gate_ok(after, now, not_before=flip)[0]


def test_the_verdict_window_tolerates_exactly_one_missed_refresh():
    from autopilot_core import (
        AUTO_POPULATE_REFRESH_MINUTES,
        FOCUS_GATE_VERDICT_MAX_AGE_MINUTES,
    )

    assert FOCUS_GATE_VERDICT_MAX_AGE_MINUTES == int(AUTO_POPULATE_REFRESH_MINUTES * 1.5)


# ---------------------------------------------------------------------------
# The flip re-measures before it drains (R2.1)
# ---------------------------------------------------------------------------


def test_reverify_re_measures_only_the_queued_symbols(tmp_path):
    """Not a discovery pass: it stages nothing and touches no watchlist, so it
    cannot collide with BounceBot's ownership of the periodic refresh."""
    from datetime import datetime

    import autopilot_core as core

    moment = datetime(2026, 7, 2, 11, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)

    asked: list[list[str]] = []

    def downloader(symbols, **_kwargs):
        asked.append(list(symbols))
        raise RuntimeError("stop here - the symbol list is what matters")

    with pytest.raises(RuntimeError):
        core.reverify_pending_picks(
            pending_path=tmp_path / "pending.json", downloader=downloader, now=moment
        )
    assert asked == [["AAA"]], "only the queued pick, not the universe"


def test_reverify_evicts_a_pick_that_has_gone_bad(tmp_path, monkeypatch):
    from datetime import datetime

    import autopilot_core as core

    moment = datetime(2026, 7, 2, 11, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)
    assert "AAA" in _pending(tmp_path, moment)["pending"]["long"]

    monkeypatch.setattr(core, "fetch_intraday_profiles", lambda *_a, **_k: {"AAA": FAILING})
    monkeypatch.setattr(core, "load_daily_context", lambda *_a, **_k: CTX)
    result = core.reverify_pending_picks(pending_path=tmp_path / "pending.json", now=moment)

    assert result["evicted"]["long"] == ["AAA (not above session VWAP)"]
    assert "AAA" not in _pending(tmp_path, moment)["pending"]["long"]


def test_reverify_raises_rather_than_half_measuring(tmp_path, monkeypatch):
    """A failure must leave the queue exactly as it was, so the caller can
    leave the picks staged instead of adopting on partial evidence."""
    from datetime import datetime

    import autopilot_core as core

    moment = datetime(2026, 7, 2, 11, 0)
    _staged(tmp_path, {"AAA": PASSING}, CTX, ROWS, now=moment)
    before = _pending(tmp_path, moment)["pending"]["long"]["AAA"]

    monkeypatch.setattr(core, "fetch_intraday_profiles", lambda *_a, **_k: {})
    with pytest.raises(RuntimeError):
        core.reverify_pending_picks(pending_path=tmp_path / "pending.json", now=moment)

    assert _pending(tmp_path, moment)["pending"]["long"]["AAA"] == before


def test_an_empty_queue_needs_no_fetch(tmp_path):
    from datetime import datetime

    import autopilot_core as core

    def explode(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("nothing queued, so nothing to fetch")

    result = core.reverify_pending_picks(
        pending_path=tmp_path / "pending.json",
        downloader=explode,
        now=datetime(2026, 7, 2, 11, 0),
    )
    assert result["symbols"] == 0


def test_a_staged_pick_records_which_bar_it_was_measured_on(tmp_path):
    """Without this the wall clock is the only freshness signal, and a stalled
    feed looks current."""
    from datetime import datetime

    moment = datetime(2026, 7, 2, 11, 0)
    profile = dict(PASSING, as_of="2026-07-02T10:55:00")
    _staged(tmp_path, {"AAA": profile}, CTX, ROWS, now=moment)
    entry = _pending(tmp_path, moment)["pending"]["long"]["AAA"]
    assert entry["gate_bar_end"] == "2026-07-02T10:55:00"
