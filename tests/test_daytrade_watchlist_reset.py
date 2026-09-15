"""The day-trade watchlists are wiped after the close (trader 2026-09-15).

`longs.txt` / `shorts.txt` hold names for ONE session. The rule is stateless:
a list that holds names and was last written at or before the last completed
session's close is due; a name typed after the close is tomorrow's and stays.
The wipe writes the file first and records `remove` rows with the source
`session_reset` after, so the WS-5D stream reconciles to an empty list and
invents no `observed_external` removals. The swing lists are never touched.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import daytrade_watchlist_reset as reset  # noqa: E402
import watchlist_intent_events as wie  # noqa: E402

ET = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

# Monday 2026-09-14 and Tuesday 2026-09-15 are ordinary sessions; Saturday
# 2026-09-12 is not; Monday 2026-09-07 was Labor Day.
MON = datetime(2026, 9, 14, tzinfo=ET)
TUE = datetime(2026, 9, 15, tzinfo=ET)


def _at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute)


# --------------------------------------------------------------------------- helpers
@pytest.fixture()
def events_path(tmp_path, monkeypatch):
    target = tmp_path / "watchlist_intent_events.jsonl"
    monkeypatch.setattr(wie, "EVENTS_FILE", target)
    return target


def _lists(tmp_path: Path, *, longs="AAPL\nMSFT\n", shorts="TSLA\n", written: datetime | None = None):
    paths = {"longs": tmp_path / "longs.txt", "shorts": tmp_path / "shorts.txt"}
    paths["longs"].write_text(longs, encoding="utf-8")
    paths["shorts"].write_text(shorts, encoding="utf-8")
    if written is not None:
        stamp = written.timestamp()
        for path in paths.values():
            os.utime(path, (stamp, stamp))
    return paths


def _on() -> object:
    return lambda key, default=None: default


# --------------------------------------------------------------------------- the rule
def test_a_list_written_during_the_session_is_due_after_that_close():
    now = _at(MON, 16, 5)  # five minutes after Monday's close
    assert reset.reset_due(now, written_at=_at(MON, 10, 30)) == MON.date()


def test_a_list_written_during_the_session_is_not_due_before_the_close():
    now = _at(MON, 15, 55)
    # Friday 09-11 is the last completed session; a Monday-morning write is after its close.
    assert reset.reset_due(now, written_at=_at(MON, 10, 30)) is None


def test_a_name_typed_after_the_close_survives_to_the_next_close():
    evening_write = _at(MON, 18, 0)
    assert reset.reset_due(_at(MON, 21, 0), written_at=evening_write) is None
    assert reset.reset_due(_at(TUE, 9, 0), written_at=evening_write) is None
    assert reset.reset_due(_at(TUE, 16, 1), written_at=evening_write) == TUE.date()


def test_a_saturday_start_owes_fridays_wipe():
    saturday = datetime(2026, 9, 12, 9, 0, tzinfo=ET)
    friday_write = datetime(2026, 9, 11, 11, 0, tzinfo=ET)
    assert reset.reset_due(saturday, written_at=friday_write) == datetime(2026, 9, 11).date()


def test_a_holiday_is_not_a_session():
    labor_day = datetime(2026, 9, 7, 12, 0, tzinfo=ET)
    friday_write = datetime(2026, 9, 4, 11, 0, tzinfo=ET)
    assert reset.reset_due(labor_day, written_at=friday_write) == datetime(2026, 9, 4).date()


def test_a_write_exactly_at_the_close_is_still_that_sessions():
    assert reset.reset_due(_at(MON, 16, 30), written_at=_at(MON, 16, 0)) == MON.date()


def test_no_file_means_nothing_is_due():
    assert reset.reset_due(_at(MON, 16, 30), written_at=None) is None


def test_a_naive_desk_clock_is_read_as_local_time():
    # 13:05 Pacific on Monday is five minutes after the close; the desk's
    # `datetime.now()` is naive local, so the rule must read it that way.
    local_now = _at(MON, 16, 5).astimezone().replace(tzinfo=None)
    assert reset.reset_due(local_now, written_at=_at(MON, 10, 0)) == MON.date()


def test_written_at_is_aware():
    stamp = reset.written_at(Path(__file__))
    assert stamp is not None and stamp.tzinfo is not None
    assert reset.written_at(Path(__file__).with_name("does-not-exist.txt")) is None


# --------------------------------------------------------------------------- the wipe
def test_the_wipe_empties_both_lists_and_records_a_remove_per_name(tmp_path, events_path):
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    results = reset.apply_reset(_at(MON, 16, 5), paths=paths, get_setting=_on())

    assert [r.wiped for r in results] == [True, True]
    assert paths["longs"].read_text(encoding="utf-8") == ""
    assert paths["shorts"].read_text(encoding="utf-8") == ""
    rows = wie.read_events()
    assert [(r["list"], r["symbol"], r["action"], r["source"]) for r in rows] == [
        ("longs", "AAPL", "remove", "session_reset"),
        ("longs", "MSFT", "remove", "session_reset"),
        ("shorts", "TSLA", "remove", "session_reset"),
    ]
    assert all(r["writer"] == "daytrade_watchlist_reset" for r in rows)
    assert all("2026-09-14 close" in r["reason"] for r in rows)
    assert [r.recorded for r in results] == [2, 1]


def test_the_stream_reconciles_to_the_empty_list_after_a_wipe(tmp_path, events_path):
    """The Watchlist tab's next load must not invent `observed_external` removes."""
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    wie.record_baseline(list_name="longs", symbols=[], writer="t")
    wie.record_changes(list_name="longs", added=["AAPL", "MSFT"], source="trader_edit", writer="t")
    assert wie.reconstruct_membership("longs") == ["AAPL", "MSFT"]
    reset.apply_reset(_at(MON, 16, 5), paths=paths, get_setting=_on())

    assert wie.reconstruct_membership("longs") == []
    before = len(wie.read_events())
    wie.observe_list(list_name="longs", symbols=[], writer="t")
    assert len(wie.read_events()) == before, "an observation that matches the stream appends nothing"


def test_a_second_pass_after_the_wipe_does_nothing(tmp_path, events_path):
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    reset.apply_reset(_at(MON, 16, 5), paths=paths, get_setting=_on())
    again = reset.apply_reset(_at(MON, 16, 6), paths=paths, get_setting=_on())
    assert [r.reason for r in again] == ["already empty", "already empty"]
    assert len(wie.read_events()) == 3


def test_a_list_written_after_the_close_is_left_alone(tmp_path, events_path):
    paths = _lists(tmp_path, written=_at(MON, 18, 0))
    results = reset.apply_reset(_at(MON, 21, 0), paths=paths, get_setting=_on())
    assert [r.wiped for r in results] == [False, False]
    assert paths["longs"].read_text(encoding="utf-8") == "AAPL\nMSFT\n"
    assert wie.read_events() == []


def test_the_switch_off_wipes_nothing(tmp_path, events_path):
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    results = reset.apply_reset(
        _at(MON, 16, 5), paths=paths, get_setting=lambda key, default=None: False
    )
    assert [r.reason for r in results] == ["switched off", "switched off"]
    assert paths["longs"].read_text(encoding="utf-8") == "AAPL\nMSFT\n"


def test_the_switch_defaults_on(tmp_path, events_path, monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "get_local_setting", lambda key, default=None: default)
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    results = reset.apply_reset(_at(MON, 16, 5), paths=paths)
    assert [r.wiped for r in results] == [True, True]


def test_a_refused_write_records_nothing(tmp_path, events_path, monkeypatch):
    """A `remove` for a name still on the file would describe a wipe that did not happen."""
    import autopilot_core

    monkeypatch.setattr(autopilot_core, "write_watchlist_file", lambda path, symbols: False)
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    results = reset.apply_reset(_at(MON, 16, 5), paths=paths, get_setting=_on())
    assert [r.reason for r in results] == ["write refused", "write refused"]
    assert paths["longs"].read_text(encoding="utf-8") == "AAPL\nMSFT\n"
    assert wie.read_events() == []


def test_a_failed_append_never_costs_the_wipe(tmp_path, events_path, monkeypatch):
    def _boom(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(wie, "record_changes", _boom)
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    results = reset.apply_reset(_at(MON, 16, 5), paths=paths, get_setting=_on())
    assert [r.wiped for r in results] == [True, True]
    assert [r.recorded for r in results] == [0, 0]
    assert paths["longs"].read_text(encoding="utf-8") == ""
    assert "2 of 2" not in reset.describe(results) and "0 of 2 intent rows recorded" in reset.describe(results)


def test_a_dry_run_writes_nowhere(tmp_path, events_path):
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    results = reset.apply_reset(_at(MON, 16, 5), paths=paths, get_setting=_on(), dry_run=True)
    assert [(r.reason, r.removed) for r in results] == [("dry run", ("AAPL", "MSFT")), ("dry run", ("TSLA",))]
    assert paths["longs"].read_text(encoding="utf-8") == "AAPL\nMSFT\n"
    assert wie.read_events() == []


def test_one_lists_failure_never_costs_the_other(tmp_path, events_path, monkeypatch):
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    real = reset.read_watchlist_symbols

    def _read(path, **kwargs):
        if path.name == "longs.txt":
            raise RuntimeError("unreadable")
        return real(path, **kwargs)

    monkeypatch.setattr(reset, "read_watchlist_symbols", _read)
    results = reset.apply_reset(_at(MON, 16, 5), paths=paths, get_setting=_on())
    assert [r.reason for r in results] == ["failed", "wiped"]
    assert paths["shorts"].read_text(encoding="utf-8") == ""


def test_the_swing_lists_are_not_in_scope():
    assert reset.RESET_LISTS == ("longs", "shorts")
    assert set(reset.default_paths()) == {"longs", "shorts"}
    assert "session_reset" in wie.SOURCES and wie.SOURCE_SESSION_RESET == reset.INTENT_SOURCE


def test_the_cli_is_a_dry_run_by_default(tmp_path, events_path, monkeypatch, capsys):
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    monkeypatch.setattr(reset, "default_paths", lambda: paths)
    monkeypatch.setattr(reset, "datetime", type("D", (datetime,), {"now": staticmethod(lambda tz=None: _at(MON, 16, 5))}))
    assert reset.main([]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "AAPL, MSFT" in out
    assert paths["longs"].read_text(encoding="utf-8") == "AAPL\nMSFT\n"
    assert reset.main(["--apply"]) == 0
    assert paths["longs"].read_text(encoding="utf-8") == ""


# --------------------------------------------------------------------------- the desk seam
def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _bare_service(monkeypatch):
    _qapp()
    from ui.services import autopilot_service as mod

    service = mod.AutopilotService.__new__(mod.AutopilotService)
    service._enabled = True
    service._profile = "desk"
    service._state = {"autopilot_written": {"longs": ["AAPL"], "shorts": []}}
    service._alerts_date = None
    service._alerts_today = {}
    service._d1_events_pending = []
    logged: list[str] = []
    monkeypatch.setattr(mod.AutopilotService, "_log", lambda self, message: logged.append(message))
    saved: list[int] = []
    monkeypatch.setattr(mod.AutopilotService, "_save_state", lambda self: saved.append(1))
    return mod, service, logged, saved


def test_the_tick_resets_before_the_weekend_short_circuit(monkeypatch):
    """A desk started on Saturday still owes Friday's wipe."""
    mod, service, _logged, _saved = _bare_service(monkeypatch)
    calls: list[str] = []
    service._roll_day_state = lambda: calls.append("roll")  # type: ignore[method-assign]
    service._maybe_reset_daytrade_watchlists = lambda now: calls.append("reset")  # type: ignore[method-assign]
    service._apply_scan_window = lambda now: calls.append("window")  # type: ignore[method-assign]
    service._apply_quiet_hours = lambda now: calls.append("quiet")  # type: ignore[method-assign]
    service._maybe_auto_arm = lambda now: calls.append("arm")  # type: ignore[method-assign]
    saturday = datetime(2026, 9, 12, 9, 0)
    monkeypatch.setattr(
        "ui.services.autopilot_service.datetime",
        type("D", (datetime,), {"now": staticmethod(lambda tz=None: saturday)}),
    )
    service._weekend_logged_date = saturday.date().isoformat()

    service._tick()

    assert calls[:2] == ["roll", "reset"], calls
    assert "arm" not in calls, "the weekend short-circuit still holds after the reset"


def test_a_wipe_forgets_what_auto_pilot_wrote_and_says_so(monkeypatch, tmp_path, events_path):
    mod, service, logged, saved = _bare_service(monkeypatch)
    paths = _lists(tmp_path, written=_at(MON, 10, 0))
    monkeypatch.setattr(reset, "default_paths", lambda: paths)
    monkeypatch.setattr(reset, "_enabled", lambda get_setting: True)

    service._maybe_reset_daytrade_watchlists(_at(MON, 16, 5))

    assert service._state["autopilot_written"] == {"longs": [], "shorts": []}
    assert saved == [1]
    assert len(logged) == 1 and "longs.txt: wiped 2 name(s) after the 2026-09-14 close" in logged[0]
    assert "shorts.txt: wiped 1 name(s)" in logged[0]


def test_nothing_due_means_no_log_no_save_no_state_change(monkeypatch, tmp_path, events_path):
    mod, service, logged, saved = _bare_service(monkeypatch)
    paths = _lists(tmp_path, written=_at(MON, 18, 0))
    monkeypatch.setattr(reset, "default_paths", lambda: paths)
    monkeypatch.setattr(reset, "_enabled", lambda get_setting: True)

    service._maybe_reset_daytrade_watchlists(_at(MON, 21, 0))

    assert service._state["autopilot_written"] == {"longs": ["AAPL"], "shorts": []}
    assert saved == [] and logged == []


def test_a_reset_failure_never_fails_the_tick(monkeypatch):
    mod, service, logged, saved = _bare_service(monkeypatch)

    def _boom(now):
        raise RuntimeError("no calendar")

    monkeypatch.setattr(reset, "apply_reset", _boom)
    service._maybe_reset_daytrade_watchlists(datetime(2026, 9, 14, 13, 5))
    assert saved == [] and logged == []


def test_the_utc_stamp_and_the_et_close_compare_correctly():
    # 20:05 UTC on 2026-09-14 is 16:05 ET, after the close; 19:55 UTC is before.
    after = datetime(2026, 9, 14, 20, 5, tzinfo=timezone.utc)
    before = datetime(2026, 9, 14, 19, 55, tzinfo=timezone.utc)
    assert reset.reset_due(after + timedelta(minutes=1), written_at=before) == MON.date()
    assert reset.reset_due(after + timedelta(minutes=1), written_at=after) is None
