"""R10.A - the outcome ledger dual-write canary.

Every outcome row the BounceBot writes to `intraday_bounce_outcomes.csv` is
mirrored to the append-only ledger. During the canary **the CSV stays the
authority**: the mirror runs after the CSV write, cannot change it, and cannot
fail it. The point of a canary is that the two stores can be compared before
anything is asked to believe the new one.

Three properties are load-bearing and each has a test:

* **Fail-open, always.** A ledger that raises, a directory that cannot be
  created, a missing module - the CSV row still stands and the scan continues.
* **Bounded.** A defect in the mirror costs disk space once, not indefinitely,
  and reaching the cap says so once rather than every row.
* **No header widening.** New fields (family, canary marker, source store) exist
  in the ledger row only; `BOUNCE_OUTCOME_COLUMNS` is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evidence_ledger as el  # noqa: E402


class _Mirror:
    """A minimal host for the canary methods.

    Instantiating the real BounceBot means IB, timers and a Qt-adjacent world;
    the canary is a handful of methods with no dependence on any of it, so it is
    exercised directly. The wiring - that the one CSV writer calls it - is
    asserted separately below, on the source.
    """


def _host(tmp_path, *, enabled=True, ledger=None, cap=3):
    from bounce_bot_lib.legacy import BounceBot

    host = _Mirror.__new__(_Mirror)
    host.pending_bounce_outcomes = {}
    host._outcome_ledger_rows = 0
    host._outcome_ledger_obj = ledger if ledger is not None else el.intraday_outcome_ledger(tmp_path)
    host._outcome_ledger_failed = False
    host._outcome_ledger_capped = False
    host.LEDGER_CANARY_ROW_CAP = cap
    host.LEDGER_CANARY_SETTING = BounceBot.LEDGER_CANARY_SETTING
    host._ledger_canary_enabled = (lambda: enabled)
    host._outcome_ledger = BounceBot._outcome_ledger.__get__(host, _Mirror)
    host._mirror_outcome_row_to_ledger = BounceBot._mirror_outcome_row_to_ledger.__get__(host, _Mirror)
    return host


def _row(event_id="AAPL_long_20260821_06_30_00_h1_ema10_bounce", **kwargs):
    row = {
        "event_id": event_id,
        "event_type": "registered",
        "symbol": "AAPL",
        "trade_date": "2026-08-21",
        "close_r": "",
        "status": "open",
    }
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# it mirrors
# ---------------------------------------------------------------------------
def test_a_csv_row_is_mirrored_to_the_ledger(tmp_path):
    host = _host(tmp_path)
    host._mirror_outcome_row_to_ledger(_row(), {"event_id": "x"})
    rows = list(host._outcome_ledger_obj.read())
    assert len(rows) == 1
    assert rows[0]["event_id"].startswith("AAPL_long")
    assert rows[0]["schema"] == "intraday_outcome_event_v1"
    assert rows[0]["canary"] == "dual_write"
    assert rows[0]["source_store"] == "intraday_bounce_outcomes.csv"


def test_the_family_is_recorded_once_instead_of_re_derived_forever(tmp_path):
    """The CSV has no family column; every rollup has had to dig it out of the id."""
    host = _host(tmp_path)
    host._mirror_outcome_row_to_ledger(_row(), None)
    assert list(host._outcome_ledger_obj.read())[0]["family"] == "h1_ema10_bounce"


def test_an_id_with_no_family_records_an_empty_one_rather_than_guessing(tmp_path):
    host = _host(tmp_path)
    host._mirror_outcome_row_to_ledger(_row(event_id="odd"), None)
    assert list(host._outcome_ledger_obj.read())[0]["family"] == ""


def test_the_row_records_whether_the_setup_is_still_pending(tmp_path):
    host = _host(tmp_path)
    host.pending_bounce_outcomes = {"still-open": {}}
    host._mirror_outcome_row_to_ledger(_row(), {"event_id": "still-open"})
    host._mirror_outcome_row_to_ledger(_row(), {"event_id": "finished"})
    rows = list(host._outcome_ledger_obj.read())
    assert rows[0]["pending_after"] is True
    assert rows[1]["pending_after"] is False


def test_the_caller_s_row_is_not_mutated(tmp_path):
    host = _host(tmp_path)
    row = _row()
    before = dict(row)
    host._mirror_outcome_row_to_ledger(row, None)
    assert row == before, "the CSV row object must come back exactly as it went in"


# ---------------------------------------------------------------------------
# it fails open
# ---------------------------------------------------------------------------
def test_a_ledger_that_raises_never_reaches_the_caller(tmp_path):
    class Angry:
        def append(self, event, **kwargs):
            raise RuntimeError("disk full")

    host = _host(tmp_path, ledger=Angry())
    host._mirror_outcome_row_to_ledger(_row(), None)  # must not raise
    assert host._outcome_ledger_rows == 0


def test_the_canary_can_be_switched_off(tmp_path):
    host = _host(tmp_path, enabled=False)
    host._outcome_ledger_obj = None
    host._mirror_outcome_row_to_ledger(_row(), None)
    assert not list(el.intraday_outcome_ledger(tmp_path).read())


def test_an_unreadable_setting_leaves_the_canary_running():
    """A canary that switches itself off on an unrelated failure proves nothing."""
    from unittest import mock

    from bounce_bot_lib.legacy import BounceBot

    host = object.__new__(BounceBot)
    with mock.patch("project_paths.get_local_setting", side_effect=RuntimeError):
        assert BounceBot._ledger_canary_enabled(host) is True


def test_off_is_the_only_value_that_stops_it():
    from unittest import mock

    from bounce_bot_lib.legacy import BounceBot

    host = object.__new__(BounceBot)
    for value, expected in (("off", False), ("OFF", False), ("", True), ("on", True), ("nonsense", True)):
        with mock.patch("project_paths.get_local_setting", return_value=value):
            assert BounceBot._ledger_canary_enabled(host) is expected, value


# ---------------------------------------------------------------------------
# it is bounded
# ---------------------------------------------------------------------------
def test_the_cap_stops_the_mirror_and_says_so_once(tmp_path, caplog):
    import logging

    host = _host(tmp_path, cap=2)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            host._mirror_outcome_row_to_ledger(_row(), None)
    rows = list(host._outcome_ledger_obj.read())
    mirrored = [row for row in rows if row.get("canary") == "dual_write"]
    assert len(mirrored) == 2
    capped = [r for r in caplog.records if "row cap" in r.getMessage()]
    assert len(capped) == 1, "the cap announces itself once, not once per row"
    # ...and it says so IN the store, so a reader sees where it stops rather
    # than inferring it from a gap.
    events = [row for row in rows if row.get("event_type") == "canary_capped"]
    assert len(events) == 1 and events[0]["cap"] == 2


def test_the_cap_resets_with_the_session_day(tmp_path):
    """A process-lifetime cap is a silent switch-off on an always-on desk.

    3.6k-6.1k outcome rows a day against a 50,000 cap means the mirror stopped
    after 8-14 days, with one log line and nothing in the ledger.
    """
    host = _host(tmp_path, cap=2)
    for _ in range(4):
        host._mirror_outcome_row_to_ledger(_row(), None)
    assert host._outcome_ledger_rows == 2
    host._outcome_ledger_day = "1999-01-01"      # the clock rolls over
    host._mirror_outcome_row_to_ledger(_row(), None)
    assert host._outcome_ledger_rows == 1, "a new session day starts a new count"


def test_the_real_cap_is_a_session_sized_number():
    from bounce_bot_lib.legacy import BounceBot

    assert BounceBot.LEDGER_CANARY_ROW_CAP >= 10_000


# ---------------------------------------------------------------------------
# it does not touch the CSV
# ---------------------------------------------------------------------------
def test_the_csv_header_is_not_widened():
    """New fields live in the ledger row only (R10.A: no header widening)."""
    from bounce_bot_lib.legacy import BOUNCE_OUTCOME_COLUMNS

    for field in ("family", "canary", "source_store", "pending_after"):
        assert field not in BOUNCE_OUTCOME_COLUMNS


def test_the_mirror_runs_after_the_csv_write_and_cannot_change_it():
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._append_bounce_outcome_row)
    csv_at = source.index("_append_learning_row(INTRADAY_BOUNCE_OUTCOMES_CSV")
    mirror_at = source.index("_mirror_outcome_row_to_ledger(row, state)")
    assert csv_at < mirror_at, "the CSV is written first and stays the authority"


def test_there_is_exactly_one_place_that_mirrors():
    """One owner, one transaction: a second call site is a second writer."""
    import inspect

    from bounce_bot_lib import legacy

    source = inspect.getsource(legacy)
    assert source.count("self._mirror_outcome_row_to_ledger(") == 1
