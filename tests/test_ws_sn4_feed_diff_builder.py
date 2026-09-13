"""WS-SN4, the builder's own additions to the tester's file.

`tests/test_ws_sn4_feed_diff.py` is the packet's test and is not touched here
(one fixture line aside, recorded in the build commit). This file adds four
things that file does not cover:

* the sequence's rebuild count in the form that can actually hold - the
  tester's own `assert rebuilds == []` counts the `panel._rebuild_feed()` the
  test itself makes three lines earlier, so it is unsatisfiable for any
  implementation and is left red on purpose;
* the cost proxy that does not depend on a clock: a veto and a coalesced Focus
  change must CONSTRUCT no row widget;
* the measurement the packet asks for, printed under `-s`;
* the two behaviours the diff had to change inside the same seam - the
  open-burst digest survives a veto, and a name that escalated leaves one row,
  not two.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from test_ws_sn4_feed_diff import (  # noqa: E402 - path set above
    _badge_text,
    _fill,
    _identities,
    _m5_alert,
    _panel,
    _rows,
    _spy_rebuild,
    _visible,
)


def _count_row_constructions(monkeypatch) -> list:
    """Every `_ClickableItem` built from here on. The cost that matters."""
    from ui.panels.alert_center_panel import _ClickableItem

    built: list = []
    original = _ClickableItem.__init__

    def counted(self, *args, **kwargs):
        built.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_ClickableItem, "__init__", counted)
    return built


def test_the_whole_sequence_reaches_that_state_without_one_rebuild(
    tmp_path, monkeypatch
):
    """The tester's headline assertion, in the form that can hold.

    Same sequence, same comparison; the count is `[1]` because the test makes
    one `_rebuild_feed` call of its own and the spy is on the class.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)

    rows = _rows(panel.feed_layout)
    vetoed = rows[5].alert.symbol
    liked = rows[60].alert.symbol
    repeated_index = int(rows[120].alert.symbol[-3:])

    rebuilds = _spy_rebuild(monkeypatch, panel)

    panel._ignore_alert_symbol(vetoed)
    panel.focus_service.add(liked, "long", "m5", origin="alert_center")
    panel.flush_pending_focus_refresh()
    panel.add_alert(_m5_alert(901))
    panel.add_alert(_m5_alert(repeated_index))

    assert rebuilds == [], "the sequence itself must cause no rebuild"

    live = _visible(panel)
    panel._rebuild_feed()
    assert _visible(panel) == live
    assert rebuilds == [1], "only the rebuild this test asked for"


def test_a_veto_and_a_focus_change_construct_no_row_widget(tmp_path, monkeypatch):
    """The cost proxy, with no clock in it.

    350 row widget trees is what the 4.0-4.1 s veto and the 24.2 s Focus change
    were made of on 2026-09-08. A veto destroys one and builds none; a Focus
    change builds none at all.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)
    liked = _rows(panel.feed_layout)[3].alert.symbol
    vetoed = _rows(panel.feed_layout)[9].alert.symbol

    built = _count_row_constructions(monkeypatch)
    panel._ignore_alert_symbol(vetoed)
    assert built == [], "a veto builds no row"

    panel.focus_service.add(liked, "long", "m5", origin="alert_center")
    panel.flush_pending_focus_refresh()
    assert built == [], "a Focus change builds no row"

    # ...and the rebuild it replaced still builds every one of them.
    panel._rebuild_feed()
    assert len(built) == len(_rows(panel.feed_layout)) + len(
        _rows(panel.d1_feed_layout)
    )


def test_the_measurement_the_packet_asks_for(tmp_path, monkeypatch, capsys):
    """Print the offscreen cost of one veto and one coalesced Focus flush.

    Run with `-s` to read it. The assertion is deliberately loose - a wall
    clock in a test suite is not a benchmark - and exists only so the
    measurement cannot silently stop being taken.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)
    vetoed = _rows(panel.feed_layout)[7].alert.symbol
    liked = _rows(panel.feed_layout)[11].alert.symbol

    start = time.perf_counter()
    panel._ignore_alert_symbol(vetoed)
    veto_ms = (time.perf_counter() - start) * 1000.0

    panel.focus_service.add(liked, "long", "m5", origin="alert_center")
    start = time.perf_counter()
    panel.flush_pending_focus_refresh()
    focus_ms = (time.perf_counter() - start) * 1000.0

    with capsys.disabled():
        print(
            f"\nSN4 offscreen, 250 M5 + 100 D1 rows: "
            f"veto {veto_ms:.1f} ms, coalesced focus flush {focus_ms:.1f} ms"
        )
    assert veto_ms < 1000.0
    assert focus_ms < 1000.0


def test_the_open_burst_digest_survives_a_veto(tmp_path, monkeypatch):
    """A veto inside the open burst must not explode the digest into rows.

    The rebuild destroyed the digest row and then drew one row per digested
    alert - the exact pile-up the digest exists to prevent. The row is now a
    function of the digested-key registry, so it is redrawn in place and the
    vetoed name leaves it.
    """
    import alert_repetition

    monkeypatch.setattr(
        alert_repetition.RepetitionLedger, "_in_digest_window", lambda self, now: True
    )
    panel = _panel(tmp_path, monkeypatch)
    for index in range(5):
        panel.add_alert(_m5_alert(index))

    assert _rows(panel.feed_layout) == [], "the burst is one row, not five"
    assert panel._digest_row is not None
    assert "5 name(s)" in panel._digest_row.text()

    victim = _m5_alert(2).symbol
    panel._ignore_alert_symbol(victim)

    assert _rows(panel.feed_layout) == []
    assert panel._digest_row is not None
    assert "4 name(s)" in panel._digest_row.text()
    assert victim not in panel._digest_row.text()


def test_a_name_that_escalated_leaves_one_row_not_two(tmp_path, monkeypatch):
    """One live row per symbol + side, after the refresh as well as before.

    An escalation re-floats the name with a NEW row while the old one is still
    in the layout, so the feed briefly carries the name twice. The target is
    one row per (symbol, side) at the oldest entry's position, carrying the
    ledger's count - which is what the rebuild used to get wrong in the other
    direction (it drew a row per ENTRY, badge-less).
    """
    panel = _panel(tmp_path, monkeypatch)
    for index in range(4):
        panel.add_alert(_m5_alert(index))
    symbol = _m5_alert(0).symbol
    panel.add_alert(_m5_alert(0, tier="S"))

    panel._sync_feed()

    rows = [item for item in _rows(panel.feed_layout) if item.alert.symbol == symbol]
    assert len(rows) == 1
    assert _badge_text(rows[0]) == "×2"
    # The oldest entry's position: it arrived first, so it is the bottom row.
    assert _rows(panel.feed_layout)[-1] is rows[0]
    # And a full rebuild agrees, row for row.
    before = _visible(panel)
    identities = _identities(panel.feed_layout)
    panel._rebuild_feed()
    assert _visible(panel) == before
    assert _identities(panel.feed_layout) != identities, "a rebuild rebuilds"
