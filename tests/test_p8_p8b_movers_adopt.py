"""P8 P8 item 2: new Movers names feed the M5 Focus watch (AUTO lane).

Trader, 2026-09-25: "My only thing with the movers is longs need to be above
previous day high and over vwap. Shorts need to be below previous day [low] and
below vwap. Outside of that I'm ok with those being put into the M5 list."
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
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
T0 = datetime(2026, 9, 25, 10, 35, 20, tzinfo=NY)
BAR = datetime(2026, 9, 25, 10, 30, tzinfo=NY).isoformat(timespec="seconds")


def _row(symbol, *, pop, last, vwap):
    return {"symbol": symbol, "pop_score": pop, "rvol": 2.0, "last": last, "session_vwap": vwap}


def _board(long_rows=(), short_rows=()):
    return {"as_of": BAR, "as_of_stale": False, "state": {"state": "up_day"},
            "pop": {"long": list(long_rows), "short": list(short_rows)},
            "dip": {"long": [], "short": []}, "rip": {"long": [], "short": []}}


def _candidates(board, levels, mode="DESK"):
    notices = mn.MoversNotifier().decide(board, mode=mode, now=T0, local_tz=NY)
    return mn.adoption_candidates(notices, levels)


LEVELS = {"AAA": {"prev_high": 100.0, "prev_low": 90.0}}


# ------------------------------------------------------------------ level gate
@pytest.mark.parametrize("last, vwap, passes", [
    (101.0, 100.5, True),    # above prev high, above VWAP
    (101.0, 102.0, False),   # above prev high, below VWAP
    (99.0, 98.0, False),     # below prev high, above VWAP
    (99.0, 99.5, False),     # below both
])
def test_long_needs_prev_high_and_vwap(last, vwap, passes):
    cands = _candidates(_board([_row("AAA", pop=1.0, last=last, vwap=vwap)]), LEVELS)
    assert [(c["symbol"], c["side"], c["passes"]) for c in cands] == [("AAA", "long", passes)]
    assert cands[0]["level_gate"] == {"prev_high": 100.0, "prev_low": 90.0, "vwap": vwap,
                                      "last": last}


@pytest.mark.parametrize("last, vwap, passes", [
    (89.0, 89.5, True),      # below prev low, below VWAP
    (89.0, 88.0, False),     # below prev low, above VWAP
    (91.0, 92.0, False),     # above prev low, below VWAP
    (91.0, 90.5, False),     # above both
])
def test_short_needs_prev_low_and_vwap(last, vwap, passes):
    cands = _candidates(_board(short_rows=[_row("AAA", pop=-1.0, last=last, vwap=vwap)]), LEVELS)
    assert [(c["side"], c["passes"]) for c in cands] == [("short", passes)]


def test_unknown_previous_day_is_no_adoption():
    cands = _candidates(_board([_row("ZZZ", pop=1.0, last=101.0, vwap=100.0)]), {})
    assert cands[0]["passes"] is False and cands[0]["gate"] == mn.focus_adoption_gate.UNKNOWN


def test_unknown_vwap_is_no_adoption():
    cands = _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=None)]), LEVELS)
    assert cands[0]["passes"] is False


@pytest.mark.parametrize("mode", ["DESK", "AWAY", "EVENING"])
def test_desk_away_and_evening_all_feed_the_watch(mode):
    cands = _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]), LEVELS, mode)
    assert cands and cands[0]["mode"] == mode


def test_off_mode_feeds_nothing():
    assert _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]), LEVELS,
                       "OFF") == []


def test_adopt_record_shape():
    cands = _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]), LEVELS, "AWAY")
    rows = mn.adopt_records([dict(cands[0], result="adopted")], at=T0.isoformat())
    assert rows == [{
        "kind": "adopt", "session": "2026-09-25", "symbol": "AAA", "side": "long",
        "list": "pop", "state": "up_day", "mode": "AWAY", "bar": BAR,
        "level_gate": {"prev_high": 100.0, "prev_low": 90.0, "vwap": 100.5, "last": 101.0},
        "gate": mn.focus_adoption_gate.OPEN, "gate_reason": cands[0]["gate_reason"], "result": "adopted",
        "adopted_at": T0.isoformat()}]


# ------------------------------------------------------------------ service
def _service(tmp_path, mode, levels, monkeypatch, board):
    from ui.services import movers_service as svc

    monkeypatch.setattr(svc, "_market_local_tz", lambda: NY)
    monkeypatch.setattr(svc.movers_scan, "build_movers_board", lambda *a, **k: dict(board))
    asked = []

    def provider(symbols, session):
        asked.append((list(symbols), session))
        if isinstance(levels, Exception):
            raise levels
        return levels

    service = svc.MoversService(
        clock=lambda: T0, autostart=False, outcomes_path=tmp_path / "movers_dip_outcomes.jsonl",
        mode_provider=lambda: mode, push_sender=lambda *_a: {"ok": True, "kind": "delivered"},
        hidden_provider=lambda: ("", set()), sector_hidden=lambda _s: False,
        daily_levels_provider=provider,
    )
    offered = []
    service.moversAdopt.connect(offered.append)
    service._publish({}, [], T0, {"long": [], "short": []}, NY, final=True)
    return offered, asked


def test_service_offers_gated_names_with_the_durable_levels(tmp_path, monkeypatch):
    board = _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)])
    offered, asked = _service(tmp_path, "AWAY", LEVELS, monkeypatch, board)
    assert asked == [(["AAA"], T0.date())]
    cand = offered[0]["candidates"][0]
    assert (cand["symbol"], cand["passes"], cand["mode"]) == ("AAA", True, "AWAY")
    assert offered[0]["log_paths"]["dip"].endswith("movers_dip_outcomes.jsonl")


def test_service_off_mode_offers_nothing(tmp_path, monkeypatch):
    board = _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)])
    offered, asked = _service(tmp_path, "OFF", LEVELS, monkeypatch, board)
    assert offered == [] and asked == []


def test_service_failed_level_read_adopts_nothing(tmp_path, monkeypatch):
    board = _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)])
    offered, _ = _service(tmp_path, "DESK", OSError("no cache"), monkeypatch, board)
    assert [c["passes"] for c in offered[0]["candidates"]] == [False]


def test_gate_patched_to_refuse_blocks_adoption(tmp_path, monkeypatch):
    monkeypatch.setattr(mn.focus_adoption_gate, "focus_adoption_gate_state",
                        lambda *a, **k: (mn.focus_adoption_gate.CLOSED, "refused by the gate"))
    board = _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)])
    offered, _ = _service(tmp_path, "DESK", LEVELS, monkeypatch, board)
    cand = offered[0]["candidates"][0]
    assert cand["passes"] is False and cand["gate_reason"] == "refused by the gate"


# ------------------------------------------------------------------ Alert Center
class FakeStore:
    def __init__(self, existing=(), declined=()):
        self.lists = {"long": list(existing), "short": []}
        self.declined = set(declined)
        self.markers = []
        self.removed = []

    def add_many(self, symbols, side, category="m5"):
        added = [s for s in symbols if s not in self.lists[side]]
        self.lists[side].extend(added)
        return added

    def mark_auto_adopted(self, symbol, side, category="m5", *, staged_at="", reason=""):
        self.markers.append((symbol, side, staged_at, reason))

    def declined_today(self, symbol, side, category="m5"):
        return symbol in self.declined

    def remove(self, *a, **k):  # never called
        self.removed.append(a)


class FakeFocus:
    def __init__(self, store):
        self.store = store

    def add(self, *a, **k):  # the trader-like path must never be used
        raise AssertionError("FocusService.add forges a trader like")


@pytest.fixture
def panel(tmp_path):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel

    return AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")


def _payload(tmp_path, cands):
    return {"candidates": cands, "at": T0.isoformat(),
            "log_paths": {"pop": str(tmp_path / "pop.jsonl"), "dip": str(tmp_path / "dip.jsonl")}}


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def test_panel_adopts_only_gated_names_marks_them_and_logs_rows(panel, tmp_path):
    store = FakeStore(existing=["OWN"])
    panel.focus_service = FakeFocus(store)
    board = _board([_row("AAA", pop=2.0, last=101.0, vwap=100.5),
                    _row("BBB", pop=1.5, last=99.0, vwap=98.0),
                    _row("OWN", pop=1.2, last=101.0, vwap=100.5)])
    levels = {"AAA": LEVELS["AAA"], "BBB": LEVELS["AAA"], "OWN": LEVELS["AAA"]}
    cands = _candidates(board, levels, "AWAY")
    panel.adopt_movers_picks(_payload(tmp_path, cands))
    panel._movers_adopt_log_thread.join(5)
    assert store.lists["long"] == ["OWN", "AAA"]
    assert [m[0] for m in store.markers] == ["AAA"]  # the trader's OWN is never re-marked
    assert store.removed == []
    rows = _rows(tmp_path / "pop.jsonl")
    assert {r["symbol"]: r["result"] for r in rows} == {
        "AAA": "adopted", "BBB": rows[1]["result"], "OWN": "already in M5 Focus"}
    assert rows[1]["result"].startswith("gated:")
    assert all(r["kind"] == "adopt" and r["mode"] == "AWAY" for r in rows)


def test_panel_honours_not_today_and_taken_off_today(panel, tmp_path):
    store = FakeStore(declined=["BBB"])
    panel.focus_service = FakeFocus(store)
    panel._ignored_symbols.add("AAA")
    board = _board([_row("AAA", pop=2.0, last=101.0, vwap=100.5),
                    _row("BBB", pop=1.5, last=101.0, vwap=100.5)])
    cands = _candidates(board, {"AAA": LEVELS["AAA"], "BBB": LEVELS["AAA"]})
    panel.adopt_movers_picks(_payload(tmp_path, cands))
    panel._movers_adopt_log_thread.join(5)
    assert store.lists["long"] == []
    results = {r["symbol"]: r["result"] for r in _rows(tmp_path / "pop.jsonl")}
    assert results == {"AAA": "refused: you said not today",
                       "BBB": "refused: you took it off today"}


def test_attach_wires_the_adopt_signal(panel, monkeypatch):
    from ui.services import movers_service as svc

    heard = []
    monkeypatch.setattr(panel, "adopt_movers_picks", heard.append)
    service = svc.MoversService(autostart=False, mode_provider=lambda: "DESK")
    panel.attach_movers_service(service)
    service.moversAdopt.emit({"candidates": []})
    assert heard == [{"candidates": []}]
