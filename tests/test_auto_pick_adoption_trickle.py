"""F4: the DESK auto-adoption drain trickles instead of flooding.

2026-08-31, 07:41:58-07:42:11: the Alert Center drain adopted **45 staged picks
into M5 Focus inside 13 seconds**, one at a time, each add writing the focus
file, writing the auto-pick marker sidecar, and firing `focusChanged` at five
listeners that each rebuilt everything they own. The desk was Not Responding
either side of it and the trader killed it twice.

The trader authorized the pacing change on 2026-08-31 ("cap the auto-adopt
batch and slow the redraws"). This file pins what that change is allowed to be:

* **pacing, never policy.** Nothing decides differently about a pick. The
  freshness gate, the flip barrier, the ownership markers and AWAY/EVENING's
  refusal are all untouched;
* **no pick is ever dropped.** What does not fit in this cycle stays staged and
  is adopted by a later one - a cap that withheld a pick would be a suppression
  field, which this chain deliberately does not have.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

#: The burst that froze the desk.
STAGED = [f"SYM{index:02d}" for index in range(45)]


def _panel(monkeypatch, tmp_path, mode="DESK"):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: mode)
    panel._auto_mode_cached = None
    panel._auto_pick_pending_path = tmp_path / "pending.json"
    return panel


def _payload():
    """45 staged longs, every one carrying a passing, current verdict."""
    import autopilot_core

    now = datetime.now().isoformat(timespec="seconds")
    bar_end = autopilot_core.latest_completed_m5_end().isoformat()
    return {
        "date": "2026-08-31",
        "pending": {
            "long": {
                symbol: {
                    "reason": "PDH break",
                    "gate_state": "open",
                    "gate_checked_at": now,
                    "gate_bar_end": bar_end,
                }
                for symbol in STAGED
            },
            "short": {},
        },
    }


def _wire(panel, monkeypatch, adopted):
    payload = _payload()
    monkeypatch.setattr(
        "autopilot_core.load_auto_populate_pending_picks", lambda *_a, **_k: payload
    )
    panel._adopt_auto_pick_into_focus = (  # type: ignore[method-assign]
        lambda symbol, *a, **k: (adopted.append(symbol), True)[1]
    )
    panel._start_pending_reverify = lambda: None  # type: ignore[method-assign]
    return payload


def test_one_cycle_adopts_at_most_the_batch_cap(monkeypatch, tmp_path):
    from ui.panels.alert_center_panel import AUTO_ADOPT_BATCH_LIMIT

    panel = _panel(monkeypatch, tmp_path)
    adopted: list[str] = []
    _wire(panel, monkeypatch, adopted)

    panel._poll_auto_pick_pending()
    assert len(adopted) == AUTO_ADOPT_BATCH_LIMIT
    assert adopted == STAGED[:AUTO_ADOPT_BATCH_LIMIT], "in the order they were staged"


def test_every_staged_pick_is_adopted_by_a_later_cycle(monkeypatch, tmp_path):
    """The cap spreads adoption; it never withholds a pick."""
    from ui.panels.alert_center_panel import AUTO_ADOPT_BATCH_LIMIT

    panel = _panel(monkeypatch, tmp_path)
    adopted: list[str] = []
    _wire(panel, monkeypatch, adopted)

    cycles = 0
    while len(adopted) < len(STAGED) and cycles < 20:
        panel._poll_auto_pick_pending()
        cycles += 1
    assert adopted == STAGED, "no pick is dropped, and none is adopted twice"
    assert cycles == -(-len(STAGED) // AUTO_ADOPT_BATCH_LIMIT)


def test_a_cycle_under_the_cap_still_drains_in_one_go(monkeypatch, tmp_path):
    """The ordinary day - a handful of picks - is unchanged."""
    panel = _panel(monkeypatch, tmp_path)
    adopted: list[str] = []
    payload = _wire(panel, monkeypatch, adopted)
    entries = payload["pending"]["long"]
    payload["pending"]["long"] = {
        symbol: entries[symbol] for symbol in STAGED[:3]
    }

    panel._poll_auto_pick_pending()
    assert adopted == STAGED[:3]


def test_a_refused_pick_does_not_spend_a_slot(monkeypatch, tmp_path):
    """The cap counts adoptions, not iterations.

    A day where the gate refuses most of the queue must still adopt a full
    batch of the ones that qualify, rather than stopping after N refusals.
    """
    from ui.panels.alert_center_panel import AUTO_ADOPT_BATCH_LIMIT

    panel = _panel(monkeypatch, tmp_path)
    adopted: list[str] = []
    payload = _wire(panel, monkeypatch, adopted)
    # Every other pick fails the gate outright.
    for index, symbol in enumerate(STAGED):
        if index % 2:
            payload["pending"]["long"][symbol]["gate_state"] = "closed"
            payload["pending"]["long"][symbol]["gate_reason"] = "not above session VWAP"

    panel._poll_auto_pick_pending()
    assert len(adopted) == AUTO_ADOPT_BATCH_LIMIT
    assert all(STAGED.index(symbol) % 2 == 0 for symbol in adopted)


def test_the_cap_never_marks_an_unadopted_pick_seen(monkeypatch, tmp_path):
    """A pick left for the next cycle must still be pending, not resolved."""
    from ui.panels.alert_center_panel import AUTO_ADOPT_BATCH_LIMIT

    panel = _panel(monkeypatch, tmp_path)
    adopted: list[str] = []
    _wire(panel, monkeypatch, adopted)

    panel._poll_auto_pick_pending()
    seen = {key[2] for key in panel._auto_picks_enqueued}
    assert seen == set(STAGED[:AUTO_ADOPT_BATCH_LIMIT])
    assert not seen & set(STAGED[AUTO_ADOPT_BATCH_LIMIT:]), (
        "a deferred pick must not be marked seen, or the next cycle skips it"
    )


@pytest.mark.parametrize("mode", ["AWAY", "EVENING"])
def test_the_cap_changes_nothing_about_who_may_adopt(monkeypatch, tmp_path, mode):
    panel = _panel(monkeypatch, tmp_path, mode)
    adopted: list[str] = []
    _wire(panel, monkeypatch, adopted)

    panel._poll_auto_pick_pending()
    assert adopted == [], f"{mode} still refuses adoption outright"
    assert not panel._auto_picks_enqueued
