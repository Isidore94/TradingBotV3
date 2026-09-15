"""PCT-1 - a DECLINED auto-armed Pullback alert. ADDED by the builder.

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


def test_the_armed_board_hides_a_declined_row_and_the_store_remembers_it(
    monkeypatch, tmp_path
):
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    _claim(tmp_path, "NVDA", "LONG")
    panel._poll_pullback_watches(now=datetime.now())
    assert _board_symbols(panel) == ["NVDA"]

    assert panel.disarm_chart_watch_for("NVDA", WATCH_KIND) is True

    # Gone from the board the trader reads...
    assert _board_symbols(panel) == []
    # ...but REMEMBERED, so the 60-second sweep does not simply put it back.
    kept = [watch for watch in panel._chart_watches if watch.kind == WATCH_KIND]
    assert len(kept) == 1
    assert kept[0].symbol == "NVDA"
    assert bool(kept[0].declined) is True

    panel._poll_pullback_watches(now=datetime.now() + timedelta(minutes=1))
    assert _board_symbols(panel) == []

    # And it survives a desk restart in the same store.
    from chart_watch import load_chart_watches

    back = load_chart_watches(
        tmp_path / "chart_watches.json", market_date="2999-01-01"
    )
    assert [(watch.symbol, bool(watch.declined)) for watch in back] == [
        ("NVDA", True)
    ]


def test_a_hand_armed_watch_is_still_simply_deleted_on_disarm(monkeypatch, tmp_path):
    """Nothing about the trader's own click changed: no `auto:` source, no
    memory, no row left behind."""
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    assert panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND) is True
    assert _board_symbols(panel) == ["AAPL"]

    assert panel.disarm_chart_watch_for("AAPL", WATCH_KIND) is True

    assert [watch for watch in panel._chart_watches if watch.kind == WATCH_KIND] == []
    assert _board_symbols(panel) == []
