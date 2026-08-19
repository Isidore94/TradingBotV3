"""The 2026-08-19 adoption-gate crash: naive clock vs aware measured bar.

On the first DESK morning every adoption attempt raised

    TypeError: can't subtract offset-naive and offset-aware datetimes

inside ``pending_pick_gate_ok``'s bar-lag check, the Alert Center caught it,
logged "Focus gate check unavailable; refusing adoption" and refused all 121
staged picks every 30 seconds. Zero adoptions all session.

The two stamps come from different writers and had drifted apart:

- ``gate_bar_end`` is the profile's ``as_of``, which
  ``_intraday_extreme_metrics`` writes **always aware** — the source bar's own
  offset when it has one, market-local otherwise;
- ``gate_checked_at`` and the caller's ``now`` are plain ``datetime.now()``,
  **naive**.

So the age check (naive − naive) passed and the bar check (naive − aware)
raised, which is exactly the line the traceback named.

These tests fail on the unfixed code. They cover both directions, because a
future writer flipping the other stamp aware would be the same bug mirrored,
and they check the ARITHMETIC as well as the absence of a crash: normalizing by
stripping the offset instead of attaching one would stop the crash and start
refusing every pick as three hours stale.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autopilot_core as core  # noqa: E402
import focus_adoption_gate  # noqa: E402

NY = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")


def verdict(*, checked, bar_end, reason="passed the Focus gate"):
    def _text(value):
        return value.isoformat(timespec="seconds") if isinstance(value, datetime) else str(value)

    return {
        "gate_state": focus_adoption_gate.OPEN,
        "gate_reason": reason,
        "gate_checked_at": _text(checked),
        "gate_bar_end": _text(bar_end),
    }


class TestTheLiveDefect:
    def test_an_aware_measured_bar_and_a_naive_clock_adopt(self):
        """The exact shape of 2026-08-19: aware `as_of`, naive `datetime.now()`."""
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 7, 0),
            bar_end=datetime(2026, 8, 19, 8, 5, 0, tzinfo=PT),
        )
        ok, reason = core.pending_pick_gate_ok(entry, now=datetime(2026, 8, 19, 8, 7, 30))
        assert ok is True, reason

    def test_the_reverse_mix_also_adopts(self):
        """A naive stored bar against an aware clock is the same bug mirrored."""
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 7, 0, tzinfo=PT),
            bar_end=datetime(2026, 8, 19, 8, 5, 0),
        )
        ok, reason = core.pending_pick_gate_ok(
            entry, now=datetime(2026, 8, 19, 8, 7, 30, tzinfo=PT)
        )
        assert ok is True, reason

    def test_an_aware_check_stamp_against_a_naive_clock_adopts(self):
        """The age comparison is the other subtraction in the same function."""
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 7, 0, tzinfo=PT),
            bar_end=datetime(2026, 8, 19, 8, 5, 0),
        )
        ok, reason = core.pending_pick_gate_ok(entry, now=datetime(2026, 8, 19, 8, 7, 30))
        assert ok is True, reason

    def test_an_aware_flip_barrier_against_a_naive_check_stamp(self):
        """`not_before` is the third datetime the caller supplies."""
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 7, 0),
            bar_end=datetime(2026, 8, 19, 8, 5, 0),
        )
        ok, reason = core.pending_pick_gate_ok(
            entry,
            now=datetime(2026, 8, 19, 8, 7, 30),
            not_before=datetime(2026, 8, 19, 8, 6, 0, tzinfo=PT),
        )
        assert ok is True, reason


class TestTheArithmeticSurvivesTheFix:
    def test_the_same_instant_in_two_zones_is_zero_bars_of_lag(self):
        """Attaching an offset, not stripping one.

        08:05 PT and 11:05 ET are the same bar. A fix that dropped the offset
        would make this read as 36 bars of lag and refuse every pick - the
        crash would be gone and the outage would not.
        """
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 7, 0),
            bar_end=datetime(2026, 8, 19, 11, 5, 0, tzinfo=NY),
        )
        ok, reason = core.pending_pick_gate_ok(entry, now=datetime(2026, 8, 19, 8, 7, 30))
        assert ok is True, reason

    def test_a_genuinely_stale_bar_is_still_refused(self):
        """Fail-closed semantics are unchanged; only the arithmetic is fixed."""
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 7, 0),
            bar_end=datetime(2026, 8, 19, 7, 30, 0, tzinfo=PT),
        )
        ok, reason = core.pending_pick_gate_ok(entry, now=datetime(2026, 8, 19, 8, 7, 30))
        assert ok is False
        assert "M5 bars ago" in reason

    def test_a_bar_from_the_future_is_still_refused(self):
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 7, 0),
            bar_end=datetime(2026, 8, 19, 8, 25, 0, tzinfo=PT),
        )
        ok, reason = core.pending_pick_gate_ok(entry, now=datetime(2026, 8, 19, 8, 7, 30))
        assert ok is False
        assert "ahead of the tape" in reason

    def test_a_verdict_from_before_the_flip_is_still_refused(self):
        entry = verdict(
            checked=datetime(2026, 8, 19, 8, 0, 0),
            bar_end=datetime(2026, 8, 19, 8, 5, 0, tzinfo=PT),
        )
        ok, reason = core.pending_pick_gate_ok(
            entry,
            now=datetime(2026, 8, 19, 8, 7, 30),
            not_before=datetime(2026, 8, 19, 8, 6, 0, tzinfo=PT),
        )
        assert ok is False
        assert "predates the return to the desk" in reason

    def test_a_stale_wall_clock_verdict_is_still_refused(self):
        entry = verdict(
            checked=datetime(2026, 8, 19, 7, 0, 0),
            bar_end=datetime(2026, 8, 19, 8, 5, 0, tzinfo=PT),
        )
        ok, reason = core.pending_pick_gate_ok(entry, now=datetime(2026, 8, 19, 8, 7, 30))
        assert ok is False
        assert "min old" in reason


class TestTheProductionShape:
    def test_a_verdict_stamped_the_way_production_stamps_it_adopts(self):
        """End to end through the real writers, not a hand-built dict.

        `_intraday_extreme_metrics` is what produces `as_of`, and it is the
        reason the stored bar is aware at all - so the regression test uses it
        rather than asserting a string this test wrote itself.
        """
        start = datetime(2026, 8, 19, 6, 30, tzinfo=PT)
        rows = [
            {
                "dt": start + timedelta(minutes=5 * index),
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
            }
            for index in range(12)
        ]
        now_naive = datetime(2026, 8, 19, 7, 32)
        profile = core._intraday_extreme_metrics(rows, now=now_naive)
        assert profile["as_of"], "the profile must stamp the bar it measured"

        entry = {
            "gate_state": focus_adoption_gate.OPEN,
            "gate_reason": "passed the Focus gate at candidate build",
            # Exactly how the staging path stamps it: naive wall clock.
            "gate_checked_at": now_naive.isoformat(timespec="seconds"),
            "gate_bar_end": str(profile["as_of"]),
        }
        ok, reason = core.pending_pick_gate_ok(entry, now=now_naive)
        assert ok is True, reason


class TestTheAlertCenterPath:
    def test_the_panel_wrapper_adopts_instead_of_refusing(self):
        """The wrapper caught the TypeError and called it "unavailable"."""
        pytest.importorskip("PySide6")
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel

        entry = verdict(
            checked=datetime.now().replace(microsecond=0),
            bar_end=core.latest_completed_m5_end(datetime.now()).replace(tzinfo=PT),
        )
        panel = AlertCenterPanel()
        ok, reason = panel._pending_pick_gate_ok(entry)
        assert ok is True, reason
        assert reason != "gate check unavailable"


class TestTheLogFloodIsBounded:
    """121 tracebacks every 30 seconds rotated the log and nearly buried the
    evidence. The fault stays loud; the volume does not."""

    def _panel(self, monkeypatch, tmp_path, picks):
        pytest.importorskip("PySide6")
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.alert_center_panel import AlertCenterPanel

        panel = AlertCenterPanel()
        monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: "DESK")
        panel._auto_mode_cached = None
        panel._auto_pick_pending_path = tmp_path / "pending.json"
        payload = {
            "date": "2026-08-19",
            "pending": {"long": {symbol: {"reason": "PDH break"} for symbol in picks}},
        }
        monkeypatch.setattr(
            "autopilot_core.load_auto_populate_pending_picks", lambda *_a, **_k: payload
        )

        def _boom(*_args, **_kwargs):
            raise TypeError("can't subtract offset-naive and offset-aware datetimes")

        monkeypatch.setattr("autopilot_core.pending_pick_gate_ok", _boom)
        return panel

    def test_one_traceback_and_one_summary_per_cycle(self, monkeypatch, tmp_path, caplog):
        import logging

        picks = [f"SYM{index:03d}" for index in range(121)]
        panel = self._panel(monkeypatch, tmp_path, picks)

        with caplog.at_level(logging.WARNING):
            panel._poll_auto_pick_pending()

        tracebacks = [record for record in caplog.records if record.exc_info]
        assert len(tracebacks) == 1, "one traceback per cycle, not one per pick"
        summaries = [
            record
            for record in caplog.records
            if "Focus gate check unavailable for" in record.getMessage()
        ]
        assert len(summaries) == 1
        # The summary has to carry the scale and the cause, or it replaces a
        # flood with a shrug.
        assert "121" in summaries[0].getMessage()
        assert "offset-naive" in summaries[0].getMessage()

    def test_every_pick_is_still_refused(self, monkeypatch, tmp_path):
        """Fail-closed is unchanged: bounding the logging must not adopt anything."""
        adopted: list[str] = []
        panel = self._panel(monkeypatch, tmp_path, ["AAA", "BBB", "CCC"])
        panel._adopt_auto_pick_into_focus = (  # type: ignore[method-assign]
            lambda *a, **k: adopted.append(a[0]) or True
        )
        panel._poll_auto_pick_pending()
        assert adopted == []
        # And nothing was marked seen, so the next cycle can still adopt them
        # once the fault clears.
        assert not panel._auto_picks_enqueued

    def test_the_next_cycle_logs_its_own_traceback(self, monkeypatch, tmp_path, caplog):
        """Per CYCLE, not once per process: a fault that is still there on the
        next poll must still say so."""
        import logging

        panel = self._panel(monkeypatch, tmp_path, ["AAA", "BBB"])
        with caplog.at_level(logging.WARNING):
            panel._poll_auto_pick_pending()
            first = len([record for record in caplog.records if record.exc_info])
            panel._poll_auto_pick_pending()
            second = len([record for record in caplog.records if record.exc_info])
        assert first == 1
        assert second == 2


class TestTheSameClassOfBugNextDoor:
    """`minutes_since_open` had the identical subtraction one function away.

    Every caller passes a naive clock today, so this is a hardening rather than
    a fix - but the scheduler is not where the desk should discover that a
    provider started handing back aware stamps.
    """

    def test_naive_and_aware_clocks_agree(self):
        naive = core.minutes_since_open(datetime(2026, 8, 19, 7, 30))
        aware = core.minutes_since_open(datetime(2026, 8, 19, 7, 30, tzinfo=PT))
        assert naive == aware

    def test_an_aware_clock_no_longer_raises(self):
        assert core.minutes_since_open(datetime(2026, 8, 19, 7, 30, tzinfo=NY)) is not None
