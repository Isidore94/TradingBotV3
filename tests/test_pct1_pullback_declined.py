"""PCT-1 - a hand-armed Pullback alert and its disarm. ADDED by the builder;
rewritten 2026-09-17 when Pullback alerts became manual-only (trader direction).

The lead's ruling 3 (2026-09-15) named the behaviour and asked for this test:

    ``disarm_chart_watch_for`` deletes the row today; for an AUTO-armed
    pullback watch a hand disarm instead keeps the row with ``declined=True``
    (persisted in the same store, hidden from the Armed board, never
    evaluated, never pushed) so the sweep does not re-arm it; a hand-armed
    watch is deleted on disarm as today.

The tester covered the "not armed again" half. This covers the half a trader
would notice first: a watch they turned off is not still sitting on the board
saying they are waiting on it - and the one they armed by hand is still simply
deleted.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_pct1_pullback_desk import (  # noqa: E402
    WATCH_KIND,
    _claim,
    _install_stub_caches,
    _panel,
)


def _board_symbols(panel) -> list[str]:
    panel._refresh_armed_list()
    return [row[0] for row in panel.armed_list._rows]


def test_a_claim_arms_nothing_and_a_hand_armed_watch_is_deleted_on_disarm(
    monkeypatch, tmp_path
):
    """Trader direction 2026-09-17: Pullback alerts are MANUAL only.

    The declined-row behaviour this file used to prove belonged to the AUTO
    sweep, which no longer exists: a claim arms nothing, so there is no
    automatic row to decline. What the trader notices now is the other half of
    ruling 3 - a watch they armed by hand is on the board, and a hand disarm
    simply deletes it (no `declined` row is kept, nothing re-arms it).
    """
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    _claim(tmp_path, "NVDA", "LONG")
    panel._poll_pullback_watches(now=datetime.now())
    assert _board_symbols(panel) == []

    assert panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND) is True
    assert _board_symbols(panel) == ["NVDA"]

    assert panel.disarm_chart_watch_for("NVDA", WATCH_KIND) is True
    assert _board_symbols(panel) == []
    assert [w for w in panel._chart_watches if w.kind == WATCH_KIND] == []

    panel._poll_pullback_watches(now=datetime.now() + timedelta(minutes=1))
    assert _board_symbols(panel) == []
