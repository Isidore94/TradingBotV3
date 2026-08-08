"""Setup-tracker staleness catch-up (docs/DURABILITY_CATCHUP_PLAN.md sec 2.1).

`should_update_setup_tracker_now` gates the tracker/watchlist refresh to the
final market hour onward, and it checks wall-clock only. Miss the after-close
window once (the desk never ran on 2026-08-03) and every scan the next day is
blocked from refreshing until 15:00, so the desk trades all session on setups
computed two sessions ago.

The override re-runs the *existing*
`backfill_setup_tracker_from_recent_sessions` over the sessions that were
missed, capped at the last completed session so today's forming D1 bar can
never be evaluated (plan.md sec 5: completed bars only, a forming bar is
preview).

The load-bearing test here is the characterization test: a catch-up run the
next morning -- later wall clock, and a violent forming bar already present in
the D1 frames -- produces a tracker byte-identical to the after-close run it is
standing in for. Timing changes; scoring never does. Its sensitivity control
runs the same harness *without* the completed-session cap and proves the
comparison can tell the two vintages apart.
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402
from master_avwap_lib import runner  # noqa: E402


SESSION_D = date(2026, 8, 6)  # the last completed session
SESSION_NEXT = date(2026, 8, 7)  # "today" during the catch-up run: still forming


def _tracker_payload(updated_at: str | None) -> dict:
    return {"schema_version": 2, "updated_at": updated_at, "setups": {}}


class CatchupPlanTests(unittest.TestCase):
    """Staleness detection: which sessions count, and how many to replay."""

    def _plan(self, updated_at, sessions, now, **kwargs):
        with mock.patch.object(m, "get_recent_market_session_dates", lambda n=1: sessions[:n]):
            return m.compute_setup_tracker_catchup_plan(
                now=now,
                tracker_payload=_tracker_payload(updated_at),
                **kwargs,
            )

    def test_tracker_written_after_yesterdays_close_is_not_stale(self):
        plan = self._plan(
            "2026-08-06T15:30:12",
            [SESSION_NEXT, SESSION_D, date(2026, 8, 5)],
            datetime(2026, 8, 7, 10, 15),
        )
        self.assertFalse(plan["stale"])
        self.assertEqual(plan["last_completed_session"], SESSION_D)
        self.assertEqual(plan["lookback_sessions"], 0)

    def test_one_missed_close_is_stale_for_exactly_that_session(self):
        plan = self._plan(
            "2026-08-05T15:31:00",
            [SESSION_NEXT, SESSION_D, date(2026, 8, 5), date(2026, 8, 4)],
            datetime(2026, 8, 7, 10, 15),
        )
        self.assertTrue(plan["stale"])
        self.assertEqual(plan["lookback_sessions"], 1)
        self.assertEqual(plan["last_completed_session"], SESSION_D)
        self.assertEqual(plan["last_update_session"], date(2026, 8, 5))

    def test_multiple_missed_sessions_replay_all_of_them(self):
        plan = self._plan(
            "2026-08-03T15:45:00",
            [SESSION_NEXT, SESSION_D, date(2026, 8, 5), date(2026, 8, 4), date(2026, 8, 3)],
            datetime(2026, 8, 7, 9, 45),
        )
        self.assertTrue(plan["stale"])
        self.assertEqual(plan["lookback_sessions"], 3)

    def test_long_outage_is_capped_so_recovery_stays_bounded(self):
        sessions = [SESSION_NEXT] + [date(2026, 8, 6) - pd.Timedelta(days=i).to_pytimedelta() for i in range(20)]
        plan = self._plan(
            "2026-06-01T15:30:00",
            sessions,
            datetime(2026, 8, 7, 9, 45),
            max_lookback_sessions=3,
        )
        self.assertTrue(plan["stale"])
        self.assertEqual(plan["lookback_sessions"], 3)

    def test_todays_forming_session_never_counts_as_completed(self):
        # Tracker last written after the close on 2026-08-06; "today" is
        # 2026-08-07 and its bar is still forming, so there is nothing to
        # catch up on even though a newer session date exists.
        plan = self._plan(
            "2026-08-06T16:05:00",
            [SESSION_NEXT, SESSION_D],
            datetime(2026, 8, 7, 11, 0),
        )
        self.assertFalse(plan["stale"])
        self.assertEqual(plan["last_completed_session"], SESSION_D)

    def test_unstamped_tracker_does_not_trigger_an_automatic_rebuild(self):
        # Missing data is uncertainty, never confirmation: an unstamped tracker
        # gets a logged reason, not an unattended IB-spending rebuild.
        plan = self._plan(None, [SESSION_NEXT, SESSION_D], datetime(2026, 8, 7, 10, 0))
        self.assertFalse(plan["stale"])
        self.assertIn("updated_at", plan["reason"])

    def test_unknown_session_calendar_does_not_trigger_a_rebuild(self):
        plan = self._plan("2026-07-01T15:30:00", [], datetime(2026, 8, 7, 10, 0))
        self.assertFalse(plan["stale"])
        self.assertIsNone(plan["last_completed_session"])

    def test_window_gate_itself_is_still_wall_clock_only(self):
        # The override lives beside the gate, never inside it: an intraday scan
        # must not gain the right to rewrite the tracker from forming bars.
        self.assertFalse(m.should_update_setup_tracker_now(now=datetime(2026, 8, 7, 10, 0)))
        self.assertTrue(m.should_update_setup_tracker_now(now=datetime(2026, 8, 7, 15, 30)))


def _daily_frame(last: date, periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp(last), periods=periods)
    rows = []
    for i, dt in enumerate(dates):
        base = 100.0 + (i % 37) * 0.8 - (i % 11) * 0.4
        rows.append(
            {
                "datetime": dt,
                "open": base,
                "high": base + 1.5,
                "low": base - 1.5,
                "close": base + 0.35,
                "volume": 1_000_000 + (i % 17) * 5_000,
            }
        )
    return pd.DataFrame(rows)


def _with_forming_bar(frame: pd.DataFrame, session: date) -> pd.DataFrame:
    """The same history plus one still-forming bar, gapped hard enough that any
    leakage into the completed-session vintage is impossible to miss."""
    last_close = float(frame.iloc[-1]["close"])
    forming = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(session),
                "open": last_close,
                "high": last_close + 25.0,
                "low": last_close - 25.0,
                "close": last_close + 20.0,
                "volume": 9_000_000,
            }
        ]
    )
    return pd.concat([frame, forming], ignore_index=True)


#: Wall-clock stamps the writer takes from ``datetime.now()``. They record when
#: the file was written, not what the data says, so they are normalised before
#: the byte comparison -- and nothing else is.
_CLOCK_STAMP_FIELDS = ("updated_at",)


def _normalize_write_clock(payload_text: str) -> str:
    payload = json.loads(payload_text)
    for field in _CLOCK_STAMP_FIELDS:
        if field in payload:
            payload[field] = "<write-clock>"
    for entry in (payload.get("daily_watchlists") or {}).values():
        if isinstance(entry, dict):
            for field in _CLOCK_STAMP_FIELDS:
                if field in entry:
                    entry[field] = "<write-clock>"
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class CatchupCharacterizationTests(unittest.TestCase):
    """Catch-up output == after-close output for the same data vintage."""

    def setUp(self):
        self.frame_completed = _daily_frame(SESSION_D)
        self.frame_with_forming = _with_forming_bar(self.frame_completed, SESSION_NEXT)
        self.longs = Path(m.DATA_DIR) / "catchup_test_longs.txt"
        self.shorts = Path(m.DATA_DIR) / "catchup_test_shorts.txt"
        self.longs.parent.mkdir(parents=True, exist_ok=True)
        self.longs.write_text("AAA\n", encoding="utf-8")
        self.shorts.write_text("", encoding="utf-8")

    def tearDown(self):
        # The tracker lives in the suite-wide temp home; leaving this test's
        # synthetic records behind would leak into any later tracker test.
        for path in (m.SETUP_TRACKER_FILE, m.SETUP_TRACKER_FILE.with_suffix(".json.bak")):
            path.unlink(missing_ok=True)
        self.longs.unlink(missing_ok=True)
        self.shorts.unlink(missing_ok=True)

    def _seed_tracker(self) -> None:
        """One open setup, so the run recomputes real records instead of
        comparing two empty files."""
        frame = self.frame_completed
        entry_row = frame.iloc[-25]
        setup = {
            "setup_id": "AAA-CATCHUP-LONG",
            "symbol": "AAA",
            "side": "LONG",
            "priority_bucket": "favorite_setup",
            "setup_family": "avwap_retest_followthrough",
            "entry_price": float(entry_row["close"]),
            "entry_trade_date": entry_row["datetime"].date().isoformat(),
            "scan_date": entry_row["datetime"].date().isoformat(),
            "anchor_date": frame.iloc[-40]["datetime"].date().isoformat(),
            "setup_status": "OPEN",
            "scenarios": {
                "s1": {
                    "tradeable": True,
                    "shares": 100,
                    "stop_reference_label": "LOWER_1",
                    "stop_reference_level": float(entry_row["close"]) - 3.0,
                    "status": "OPEN",
                }
            },
        }
        payload = {
            "schema_version": 2,
            "updated_at": "2026-08-05T15:30:00",
            "setups": {setup["setup_id"]: setup},
            "control_setups": {},
            "study_setups": {},
            "stats": [],
            "setup_type_stats": [],
            "attribute_registry": {},
            "daily_watchlists": {},
        }
        m.SETUP_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        m.SETUP_TRACKER_FILE.write_text(json.dumps(payload), encoding="utf-8")

    def _run_backfill(self, *, sessions, bars, end_date) -> str:
        self._seed_tracker()
        with (
            mock.patch.object(m, "connect_daily_data_client", return_value=None),
            mock.patch.object(m, "disconnect_daily_data_client", lambda *a, **k: None),
            mock.patch.object(m, "get_recent_market_session_dates", lambda n=1: sessions[:n]),
            mock.patch.object(m, "fetch_daily_bars", lambda ib, sym, days, **k: bars.copy()),
            mock.patch.object(m, "load_scan_earnings_context", lambda syms: ({}, {})),
            mock.patch.object(
                m, "append_master_avwap_d1_watchlist_symbols", lambda lo, sh: (lo, sh, 0)
            ),
            mock.patch.object(m, "run_priority_scoring_tuner", lambda **k: ""),
            mock.patch.object(m, "calibrate_expected_r_prior_anchors", lambda **k: None),
        ):
            m.backfill_setup_tracker_from_recent_sessions(
                lookback_sessions=1,
                longs_path=self.longs,
                shorts_path=self.shorts,
                end_date=end_date,
            )
        return m.SETUP_TRACKER_FILE.read_text(encoding="utf-8")

    def test_catchup_tracker_is_byte_identical_to_the_after_close_tracker(self):
        after_close = self._run_backfill(
            # Run after the close on SESSION_D: only completed bars exist.
            sessions=[SESSION_D, date(2026, 8, 5), date(2026, 8, 4)],
            bars=self.frame_completed,
            end_date=SESSION_D,
        )
        catch_up = self._run_backfill(
            # Run the next morning: later wall clock, and the frames now carry
            # SESSION_NEXT's forming bar.
            sessions=[SESSION_NEXT, SESSION_D, date(2026, 8, 5), date(2026, 8, 4)],
            bars=self.frame_with_forming,
            end_date=SESSION_D,
        )
        # Guard against a vacuous pass: the comparison must be over a tracker
        # that actually carries recomputed records, not two empty files.
        recomputed = json.loads(after_close).get("setups") or {}
        self.assertIn("AAA-CATCHUP-LONG", recomputed)
        self.assertTrue(
            (recomputed["AAA-CATCHUP-LONG"].get("scenarios") or {}).get("s1", {}).get("events"),
            "seeded setup was not recomputed, so the byte comparison proves nothing",
        )

        self.assertEqual(
            _normalize_write_clock(catch_up),
            _normalize_write_clock(after_close),
            "catch-up refresh diverged from the after-close refresh it stands in for",
        )

        # Sensitivity control: drop the completed-session cap and the same
        # harness evaluates the forming session instead, which must differ.
        uncapped = self._run_backfill(
            sessions=[SESSION_NEXT, SESSION_D, date(2026, 8, 5)],
            bars=self.frame_with_forming,
            end_date=None,
        )
        self.assertNotEqual(
            _normalize_write_clock(uncapped),
            _normalize_write_clock(after_close),
            "comparison is vacuous: it cannot tell two data vintages apart",
        )


class RunnerCatchupInvocationTests(unittest.TestCase):
    """When the scan auto-invokes the catch-up, and when it must not."""

    def setUp(self):
        self.calls: list[dict] = []

        def _record(**kwargs):
            self.calls.append(kwargs)
            return {"dates": ["2026-08-06"], "watchlists": {}}

        self.backfill = _record

    def _invoke(self, *, plan, update_setup_tracker=None, enabled=True, backfill=None, now=None):
        stale_plan = {
            "stale": False,
            "reason": "tracker already reflects the last completed session",
            "last_update_session": None,
            "last_completed_session": None,
            "lookback_sessions": 0,
        }
        stale_plan.update(plan)
        with (
            mock.patch.object(m, "get_local_setting", lambda key, default=None: enabled),
            mock.patch.object(runner, "compute_setup_tracker_catchup_plan", lambda **k: stale_plan),
            mock.patch.object(
                runner,
                "backfill_setup_tracker_from_recent_sessions",
                backfill or self.backfill,
            ),
        ):
            return runner._maybe_run_setup_tracker_catchup(
                update_setup_tracker=update_setup_tracker,
                use_shared_watchlists=True,
                now=now or datetime(2026, 8, 7, 10, 30),
            )

    def test_stale_tracker_outside_the_window_runs_the_existing_backfill(self):
        outcome = self._invoke(
            plan={
                "stale": True,
                "lookback_sessions": 2,
                "last_completed_session": SESSION_D,
                "reason": "tracker last updated for 2026-08-04",
            }
        )
        self.assertTrue(outcome["ran"])
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["lookback_sessions"], 2)
        # The cap is what keeps the recovery on completed bars only.
        self.assertEqual(call["end_date"], SESSION_D)
        self.assertTrue(call["use_shared_watchlists"])

    def test_current_tracker_is_left_alone(self):
        outcome = self._invoke(plan={"stale": False})
        self.assertFalse(outcome["ran"])
        self.assertEqual(self.calls, [])

    def test_run_that_already_refreshes_the_tracker_skips_the_catchup(self):
        # An after-close run writes the tracker itself; a second rebuild would
        # burn IB budget for nothing.
        outcome = self._invoke(
            plan={"stale": True, "lookback_sessions": 1, "last_completed_session": SESSION_D},
            update_setup_tracker=True,
        )
        self.assertFalse(outcome["ran"])
        self.assertEqual(self.calls, [])

    def test_inside_the_after_close_window_skips_the_catchup(self):
        outcome = self._invoke(
            plan={"stale": True, "lookback_sessions": 1, "last_completed_session": SESSION_D},
            now=datetime(2026, 8, 7, 15, 30),
        )
        self.assertFalse(outcome["ran"])
        self.assertEqual(self.calls, [])

    def test_setting_off_disables_the_catchup_entirely(self):
        outcome = self._invoke(
            plan={"stale": True, "lookback_sessions": 1, "last_completed_session": SESSION_D},
            enabled=False,
        )
        self.assertFalse(outcome["ran"])
        self.assertEqual(self.calls, [])
        self.assertIn("disabled", outcome["reason"])

    def test_failed_catchup_never_fails_the_scan(self):
        def _boom(**kwargs):
            raise RuntimeError("IBKR refused client 1004")

        outcome = self._invoke(
            plan={"stale": True, "lookback_sessions": 1, "last_completed_session": SESSION_D},
            backfill=_boom,
        )
        self.assertFalse(outcome["ran"])
        self.assertIn("IBKR refused client 1004", outcome["reason"])


if __name__ == "__main__":
    unittest.main()
