"""P8 P8: Movers phone pushes (AWAY/EVENING) and the desk sound (DESK).

The decider (`movers_notify.MoversNotifier`): new names only, top 3 per list,
one notice per list per bar, 6 per 10 minutes, mode gating, the hide switches.
The service: pushes on the worker, emits DESK notices, logs each name.
The Alert Center: a DESK notice beeps through the existing sound path.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import movers_notify as mn  # noqa: E402

NY = ZoneInfo("America/New_York")
LA = ZoneInfo("America/Los_Angeles")
T0 = datetime(2026, 9, 25, 10, 35, 20, tzinfo=NY)


def _row(symbol, pop=None, dip=None, rvol=2.0):
    return {"symbol": symbol, "pop_score": pop, "dip_score": dip, "rvol": rvol}


def _board(*, bar_minute=30, pop_long=(), pop_short=(), dip_long=(), rip_short=(),
           state=None, stale=False):
    as_of = datetime(2026, 9, 25, 10, bar_minute, tzinfo=NY).isoformat(timespec="seconds")
    return {
        "as_of": as_of,
        "as_of_stale": stale,
        "state": state or {"state": "up_day"},
        "pop": {"long": [_row(s, pop=p) for s, p in pop_long],
                "short": [_row(s, pop=p) for s, p in pop_short]},
        "dip": {"long": [_row(s, dip=d) for s, d in dip_long], "short": []},
        "rip": {"long": [], "short": [_row(s, dip=d) for s, d in rip_short]},
    }


PULLBACK = {"state": "up_day", "pullback": True, "start_dt": "2026-09-25T10:00:00-04:00"}
RALLY = {"state": "up_day", "rally": True, "start_dt": "2026-09-25T10:10:00-04:00"}


# ------------------------------------------------------------------ decider
def test_first_tick_announces_top_three_new_pop_names_biggest_move_first():
    notifier = mn.MoversNotifier()
    board = _board(pop_long=[("NVDA", 1.2), ("AMD", 0.9), ("TSLA", 0.6)],
                   pop_short=[("INTC", -1.0), ("F", -0.5)])
    notices = notifier.decide(board, mode="AWAY", now=T0, local_tz=NY)
    assert [n.list_key for n in notices] == ["pop"]
    notice = notices[0]
    assert notice.channel == mn.CHANNEL_PUSH
    assert [r["symbol"] for r in notice.rows] == ["NVDA", "INTC", "AMD"]
    assert notice.line == ("Pop: NVDA +1.2 ATR rvol 2.0 · INTC -1.0 ATR rvol 2.0 · "
                           "AMD +0.9 ATR rvol 2.0 · 10:35")


def test_unchanged_list_sends_nothing_on_the_next_bar():
    notifier = mn.MoversNotifier()
    notifier.decide(_board(pop_long=[("NVDA", 1.2)]), mode="AWAY", now=T0)
    again = notifier.decide(_board(bar_minute=35, pop_long=[("NVDA", 1.4)]), mode="AWAY",
                            now=T0 + timedelta(minutes=5))
    assert again == []


def test_only_names_new_since_the_last_tick_are_announced():
    notifier = mn.MoversNotifier()
    notifier.decide(_board(pop_long=[("NVDA", 1.2), ("AMD", 1.0)]), mode="AWAY", now=T0)
    notices = notifier.decide(
        _board(bar_minute=35, pop_long=[("NVDA", 1.3), ("MU", 1.1), ("AMD", 1.0)]),
        mode="EVENING", now=T0 + timedelta(minutes=5))
    assert [r["symbol"] for r in notices[0].rows] == ["MU"]
    assert notices[0].mode == "EVENING"


def test_one_notice_per_list_per_bar():
    notifier = mn.MoversNotifier()
    notifier.decide(_board(pop_long=[("NVDA", 1.2)]), mode="AWAY", now=T0)
    same_bar = notifier.decide(_board(pop_long=[("NVDA", 1.2), ("MU", 1.0)]), mode="AWAY",
                               now=T0 + timedelta(seconds=30))
    assert same_bar == []


def test_state_change_counts_every_name_as_new_capped_at_three():
    notifier = mn.MoversNotifier()
    names = [("AAA", 2.0), ("BBB", 1.5), ("CCC", 1.2), ("DDD", 1.0)]
    first = notifier.decide(_board(dip_long=names, state=PULLBACK), mode="AWAY", now=T0)
    assert [r["symbol"] for r in first[0].rows] == ["AAA", "BBB", "CCC"]
    assert first[0].label == "Dip-strong" and first[0].state == "pullback"
    # A new pullback (another start bar): the same names are new again.
    later = dict(PULLBACK, start_dt="2026-09-25T10:25:00-04:00")
    again = notifier.decide(_board(bar_minute=35, dip_long=names, state=later), mode="AWAY",
                            now=T0 + timedelta(minutes=5))
    assert [r["symbol"] for r in again[0].rows] == ["AAA", "BBB", "CCC"]


def test_rip_weak_list_is_announced_in_a_rally():
    notifier = mn.MoversNotifier()
    notices = notifier.decide(_board(rip_short=[("XYZ", -1.1)], state=RALLY), mode="AWAY",
                              now=T0)
    assert [(n.label, n.state) for n in notices] == [("Rip-weak", "rally")]
    assert notices[0].rows[0]["side"] == "short"


def test_no_more_than_six_notices_in_ten_minutes():
    notifier = mn.MoversNotifier()
    sent = 0
    for i in range(4):
        board = _board(bar_minute=30 + i, state=dict(PULLBACK, start_dt=f"s{i}"),
                       pop_long=[(f"P{i}", 1.0)], dip_long=[(f"D{i}", 1.0)])
        sent += len(notifier.decide(board, mode="AWAY", now=T0 + timedelta(minutes=i)))
    assert sent == mn.MAX_NOTICES_PER_WINDOW == 6
    later = _board(bar_minute=45, pop_long=[("LATE", 1.0)])
    assert notifier.decide(later, mode="AWAY", now=T0 + timedelta(minutes=11))


@pytest.mark.parametrize("mode", ["OFF", "", None, "SOMETHING"])
def test_off_or_unknown_mode_sends_nothing_but_the_list_memory_advances(mode):
    notifier = mn.MoversNotifier()
    assert notifier.decide(_board(pop_long=[("NVDA", 1.2)]), mode=mode, now=T0) == []
    # Switching to AWAY on the next bar does not dump the old list as new.
    later = notifier.decide(_board(bar_minute=35, pop_long=[("NVDA", 1.2)]), mode="AWAY",
                            now=T0 + timedelta(minutes=5))
    assert later == []


def test_desk_mode_is_the_desk_channel():
    notices = mn.MoversNotifier().decide(_board(pop_long=[("NVDA", 1.2)]), mode="desk", now=T0)
    assert notices[0].channel == mn.CHANNEL_DESK


def test_stale_spy_sends_nothing():
    board = _board(pop_long=[("NVDA", 1.2)], stale=True)
    assert mn.MoversNotifier().decide(board, mode="AWAY", now=T0) == []


def test_hidden_names_never_take_a_slot():
    notifier = mn.MoversNotifier()
    board = _board(pop_long=[("XOM", 2.0), ("NVDA", 1.5), ("AMD", 1.2), ("MU", 1.1),
                             ("TSLA", 1.0)])
    notices = notifier.decide(board, mode="AWAY", now=T0, hidden_keys={"NVDA|long"},
                              is_sector_hidden=lambda s: s == "XOM")
    assert [r["symbol"] for r in notices[0].rows] == ["AMD", "MU", "TSLA"]


def test_notice_records_one_row_per_name():
    notice = mn.MoversNotifier().decide(_board(pop_long=[("NVDA", 1.2), ("AMD", 1.0)]),
                                        mode="AWAY", now=T0)[0]
    rows = mn.notice_records(notice, pushed_at=T0, result="delivered")
    assert [(r["kind"], r["symbol"], r["list"], r["mode"], r["state"]) for r in rows] == [
        ("notice", "NVDA", "pop", "AWAY", "up_day"), ("notice", "AMD", "pop", "AWAY", "up_day")]
    assert rows[0]["pushed_at"] == T0.isoformat(timespec="seconds")


# ------------------------------------------------------------------ service
@pytest.fixture
def service_factory(tmp_path, monkeypatch):
    from ui.services import movers_service as svc

    monkeypatch.setattr(svc, "_market_local_tz", lambda: NY)

    def make(mode, *, sender=None, hidden=("", set())):
        pushes = []

        def send(title, message):
            pushes.append((title, message))
            return {"ok": True, "kind": "delivered"}

        service = svc.MoversService(
            clock=lambda: T0, autostart=False,
            outcomes_path=tmp_path / "movers_dip_outcomes.jsonl",
            mode_provider=lambda: mode, push_sender=sender or send,
            hidden_provider=lambda: hidden, sector_hidden=lambda _s: False,
        )
        return service, pushes

    return svc, make


def _publish(svc, service, board, monkeypatch, now=T0):
    monkeypatch.setattr(svc.movers_scan, "build_movers_board", lambda *a, **k: dict(board))
    service._publish({}, [], now, {"long": [], "short": []}, NY, final=True)


def _log(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_service_pushes_in_away_and_logs_the_names(service_factory, monkeypatch, tmp_path):
    svc, make = service_factory
    service, pushes = make("AWAY")
    desk = []
    service.moversNotice.connect(desk.append)
    board = _board(pop_long=[("NVDA", 1.2)], dip_long=[("AAA", 1.0)], state=PULLBACK)
    _publish(svc, service, board, monkeypatch)
    assert [title for title, _ in pushes] == ["Movers Pop", "Movers Dip-strong"]
    assert desk == []
    pop_rows = [r for r in _log(tmp_path / mn_pop_log()) if r["kind"] == "notice"]
    dip_rows = [r for r in _log(tmp_path / "movers_dip_outcomes.jsonl") if r["kind"] == "notice"]
    assert [(r["symbol"], r["list"], r["mode"], r["result"]) for r in pop_rows] == [
        ("NVDA", "pop", "AWAY", "delivered")]
    assert [(r["symbol"], r["state"]) for r in dip_rows] == [("AAA", "pullback")]
    # Same list next bar: nothing more.
    later = dict(board, as_of=datetime(2026, 9, 25, 10, 35, tzinfo=NY).isoformat())
    _publish(svc, service, later, monkeypatch, now=T0 + timedelta(minutes=5))
    assert len(pushes) == 2


def mn_pop_log():
    import movers_outcomes

    return movers_outcomes.POP_LOG_NAME


def test_service_desk_mode_emits_a_notice_and_never_pushes(service_factory, monkeypatch):
    svc, make = service_factory
    service, pushes = make("DESK")
    desk = []
    service.moversNotice.connect(desk.append)
    _publish(svc, service, _board(pop_long=[("NVDA", 1.2)]), monkeypatch)
    assert pushes == []
    assert [n["line"].split(" ")[0:2] for n in desk] == [["Pop:", "NVDA"]]


def test_service_off_mode_sends_nothing(service_factory, monkeypatch):
    svc, make = service_factory
    service, pushes = make("OFF")
    desk = []
    service.moversNotice.connect(desk.append)
    _publish(svc, service, _board(pop_long=[("NVDA", 1.2)]), monkeypatch)
    assert pushes == [] and desk == []


def test_service_honours_todays_hide_but_not_yesterdays(service_factory, monkeypatch):
    svc, make = service_factory
    service, pushes = make("AWAY", hidden=("2026-09-25", {"NVDA|long"}))
    _publish(svc, service, _board(pop_long=[("NVDA", 1.2), ("AMD", 1.0)]), monkeypatch)
    assert "NVDA" not in pushes[0][1] and "AMD" in pushes[0][1]
    stale_service, stale_pushes = make("AWAY", hidden=("2026-09-24", {"NVDA|long"}))
    _publish(svc, stale_service, _board(pop_long=[("NVDA", 1.2)]), monkeypatch)
    assert "NVDA" in stale_pushes[0][1]


def test_failed_log_write_never_blocks_the_push(service_factory, monkeypatch):
    svc, make = service_factory
    service, pushes = make("EVENING")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(svc.movers_outcomes, "append_records", boom)
    _publish(svc, service, _board(pop_long=[("NVDA", 1.2)]), monkeypatch)
    assert len(pushes) == 1


def test_failed_push_is_logged_and_the_board_still_publishes(service_factory, monkeypatch,
                                                             tmp_path):
    svc, make = service_factory

    def broken(_title, _message):
        raise RuntimeError("network down")

    service, _ = make("AWAY", sender=broken)
    boards = []
    service.moversChanged.connect(boards.append)
    _publish(svc, service, _board(pop_long=[("NVDA", 1.2)]), monkeypatch)
    assert boards
    rows = [r for r in _log(tmp_path / mn_pop_log()) if r["kind"] == "notice"]
    assert rows and rows[0]["result"].startswith("error")


# ------------------------------------------------------------------ desk sound
@pytest.fixture
def panel(tmp_path):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel

    return AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")


def test_desk_notice_beeps_and_shows_the_line(panel, monkeypatch):
    from ui.panels import alert_center_panel as panel_mod

    beeps, status = [], []
    monkeypatch.setattr(panel_mod.QApplication, "beep", lambda: beeps.append("beep"))
    monkeypatch.setattr(panel, "_alerts_may_sound", lambda: True)
    panel.statusChanged.connect(status.append)
    panel.announce_movers({"line": "Pop: NVDA +1.2 ATR rvol 2.0 · 10:35"})
    assert beeps == ["beep"]
    assert status == ["Pop: NVDA +1.2 ATR rvol 2.0 · 10:35"]
    assert panel.movers_board.status_label.text() == status[0]


def test_desk_notice_respects_the_sound_checkbox(panel, monkeypatch):
    from ui.panels import alert_center_panel as panel_mod

    beeps = []
    monkeypatch.setattr(panel_mod.QApplication, "beep", lambda: beeps.append("beep"))
    panel.sound_input.setChecked(False)
    panel.announce_movers({"line": "Pop: NVDA"})
    assert beeps == []


def test_attach_wires_the_service_notice_to_the_sound_path(panel, monkeypatch):
    from ui.services import movers_service as svc

    heard = []
    monkeypatch.setattr(panel, "announce_movers", heard.append)
    service = svc.MoversService(autostart=False, mode_provider=lambda: "DESK")
    panel.attach_movers_service(service)
    service.moversNotice.emit({"line": "Pop: NVDA"})
    assert heard == [{"line": "Pop: NVDA"}]
