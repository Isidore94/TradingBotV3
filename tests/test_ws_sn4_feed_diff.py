"""WS-SN4 - the alert feed diffs itself instead of rebuilding (WISHLIST item 4).

Measured live on 2026-09-08: `alert_center_panel.py:5640 _ignore_alert_symbol`
calls `:1980 _rebuild_feed`, which destroys and reconstructs up to
MAX_FEED_ITEMS (250) M5 plus MAX_D1_FEED_ITEMS (100) D1 row widgets on the Qt
thread - 4.0-4.1 s per veto at 13:16 and 24.2 s on one coalesced
`focusChanged` at 13:01:22.

Required result (WISHLIST item 4, "4. SN4 - diff the feed"): a veto removes
THAT row; a Focus change restyles the star on the rows it touches; a new alert
inserts one row; the digest row and the repetition fold update in place.
`_rebuild_feed` remains for the tier-mode switch and the day roll only. **The
visible rows after any sequence of veto / focus / new-alert / repeat are
identical to a full rebuild** - the test below compares the two.

These tests drive the real `AlertCenterPanel` offscreen with the full live
shape of the feed: 250 M5 rows (exactly MAX_FEED_ITEMS) and 100 D1 rows
(exactly MAX_D1_FEED_ITEMS), built the way `tests/test_qt_alert_center.py`
builds them.

What is stubbed and why: the chart review queue (`_enqueue_review_alert`,
`_advance_review_queue`) and the M5 routing predicate. The queue is a different
surface with its own tests; leaving it live would have 350 fixture alerts ask
`ChartDataService` for bars. Everything the packet names - `add_alert`,
`_insert_item_into`, `_fold_into_existing_row`, `_ignore_alert_symbol`,
`_rebuild_feed`, the `focusChanged` coalescer, `_on_prefs_changed` - runs for
real.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


#: The live feed's own ceilings. Asserted against the module's constants in
#: `test_the_fixture_is_the_live_feed_shape` so a change to either is loud.
M5_ROWS = 250
D1_ROWS = 100


# ---------------------------------------------------------------- fixtures


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _m5_alert(index: int, *, tier: str = "B", trigger: str = ""):
    """One ordinary intraday row, the shape `test_qt_alert_center.py` uses."""
    from ui.models.bounce import BounceAlert

    symbol = f"MFIVE{index:03d}"
    side = "LONG" if index % 2 == 0 else "SHORT"
    return BounceAlert(
        time_text=f"09:{30 + index % 30:02d}:{index % 60:02d}",
        symbol=symbol,
        side=side,
        trigger=trigger or f"[{tier}-TIER] Bounce confirmed",
        timeframe="5m",
        raw_text=f"[{tier}-TIER] {symbol}: Bounce confirmed ({side.lower()})",
    )


def _d1_alert(index: int):
    """One ready D1 row (`MASTER_AVWAP_D1_ZONE` is in `_D1_READY_PREFIXES`)."""
    from ui.models.bounce import BounceAlert

    symbol = f"DONE{index:03d}"
    side = "LONG" if index % 2 == 0 else "SHORT"
    return BounceAlert(
        time_text=f"06:{index % 60:02d}:00",
        symbol=symbol,
        side=side,
        trigger="D1 band zone bounce",
        timeframe="D1",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) band zone",
        is_d1=True,
    )


def _panel(tmp_path, monkeypatch):
    """A real panel with a real FocusService, loaded with the live row counts."""
    from test_qt_focus_panel import _service
    from ui.panels.alert_center_panel import AlertCenterPanel

    _app()

    # Routing off (same reason as `test_qt_alert_center.py`'s autouse fixture)
    # and the review queue stubbed: this file is about the FEED's widgets.
    monkeypatch.setattr(
        AlertCenterPanel, "_is_m5_review_alert", staticmethod(lambda alert: False)
    )
    monkeypatch.setattr(
        AlertCenterPanel, "_enqueue_review_alert", lambda self, alert: None
    )
    monkeypatch.setattr(AlertCenterPanel, "_advance_review_queue", lambda self: None)
    # The open-burst digest is disabled by a None session open (the documented
    # fail-open branch), so the fixture cannot become a digest row at 06:32 PT
    # and a stack of rows at 07:00.
    monkeypatch.setattr(
        AlertCenterPanel,
        "_current_session_bounds",
        staticmethod(lambda: ("2026-09-08", None)),
    )

    panel = AlertCenterPanel(
        _service(tmp_path),
        ignored_symbols_path=tmp_path / "alert_center_ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "review_events.jsonl",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )
    panel.sound_input.setChecked(False)
    panel.min_tier_input.setCurrentIndex(max(0, panel.min_tier_input.findData("all")))
    return panel


def _fill(panel):
    """250 M5 + 100 D1 rows through the real `add_alert`."""
    for index in range(M5_ROWS):
        panel.add_alert(_m5_alert(index))
    for index in range(D1_ROWS):
        panel.add_alert(_d1_alert(index))


def _rows(layout):
    """The feed's row widgets in layout order (top first)."""
    from ui.panels.alert_center_panel import _ClickableItem

    items = []
    for position in range(layout.count()):
        widget = layout.itemAt(position).widget()
        if isinstance(widget, _ClickableItem):
            items.append(widget)
    return items


def _star_state(item) -> str:
    star = item.feed_item.favorite_button
    if star is None:
        return "no-star"
    return str(star.property("focusOn"))


def _badge_text(item) -> str:
    badge = item.repeat_badge
    return badge.text() if badge.isVisible() else ""


def _visible(panel):
    """(symbol, side, star state, repeat badge) per row, top first.

    Exactly the tuple the packet names: symbol, side, order, star state,
    repeat badge text. Order is the list order.
    """
    return [
        (
            item.alert.symbol,
            item.alert.side,
            _star_state(item),
            _badge_text(item),
        )
        for item in _rows(panel.feed_layout)
    ]


def _identities(layout):
    return [id(item) for item in _rows(layout)]


def _spy_rebuild(monkeypatch, panel):
    """Count real `_rebuild_feed` calls without changing what it does.

    Patched on the CLASS because the focus coalescer is late-bound
    (`lambda: self._rebuild_feed()`), so the seam a spy sees is the one that
    runs.
    """
    calls: list[int] = []
    original = type(panel)._rebuild_feed

    def counted(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(type(panel), "_rebuild_feed", counted)
    return calls


def _spy_destroyed(monkeypatch):
    """Every feed row widget whose destruction is scheduled from Python."""
    from ui.panels.alert_center_panel import _ClickableItem

    destroyed: list[object] = []
    original = _ClickableItem.deleteLater

    def counted(self):
        destroyed.append(self)
        return original(self)

    monkeypatch.setattr(_ClickableItem, "deleteLater", counted, raising=False)
    return destroyed


# ------------------------------------------------------------------ tests


def test_the_fixture_is_the_live_feed_shape(tmp_path, monkeypatch):
    """The harness loads the real ceilings, not a toy feed.

    If this fails every other test in the file is measuring the wrong thing.
    """
    from ui.panels import alert_center_panel as module

    assert module.MAX_FEED_ITEMS == M5_ROWS
    assert module.MAX_D1_FEED_ITEMS == D1_ROWS

    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)
    assert len(_rows(panel.feed_layout)) == M5_ROWS
    assert len(_rows(panel.d1_feed_layout)) == D1_ROWS


def test_a_veto_destroys_one_row_widget_and_leaves_every_other_row_alive(
    tmp_path, monkeypatch
):
    """A veto is a removal of ONE row, not a repaint of 350.

    Today `_ignore_alert_symbol` calls `_rebuild_feed`, which destroys all 250
    M5 and all 100 D1 widgets and builds 349 new ones. The rows the trader did
    not veto must survive as the same widgets, the layout object must be the
    same object, and exactly one `AlertFeedItem` wrapper may be destroyed.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)

    before = _rows(panel.feed_layout)
    before_ids = [id(item) for item in before]
    before_d1_ids = _identities(panel.d1_feed_layout)
    feed_layout = panel.feed_layout
    victim = before[10]
    symbol = victim.alert.symbol

    destroyed = _spy_destroyed(monkeypatch)
    panel._ignore_alert_symbol(symbol)

    # The layout itself is never replaced.
    assert panel.feed_layout is feed_layout
    # Exactly one row widget destroyed - the vetoed one.
    assert [id(item) for item in destroyed] == [id(victim)]
    # The survivors are the SAME widgets, in the same order, minus that row.
    assert _identities(panel.feed_layout) == [
        widget_id for widget_id in before_ids if widget_id != id(victim)
    ]
    # The veto named no D1 symbol, so the D1 feed is untouched.
    assert _identities(panel.d1_feed_layout) == before_d1_ids


def test_a_veto_keeps_the_repeat_badge_on_the_rows_it_did_not_remove(
    tmp_path, monkeypatch
):
    """A fold is display state the rebuild throws away; a diff keeps it.

    `_insert_item_into` only stamps a badge when it is handed a `repeat`, so a
    rebuild re-creates the row of a name that has alerted twice with no ×2 on
    it. The trader vetoed a different symbol.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)

    repeated = _rows(panel.feed_layout)[40].alert
    # Same symbol, same side, same tier, not proven -> ACTION_FOLD.
    panel.add_alert(_m5_alert(int(repeated.symbol[-3:])))
    folded = [
        item
        for item in _rows(panel.feed_layout)
        if item.alert.symbol == repeated.symbol
    ]
    assert len(folded) == 1, "the fixture's repeat did not fold into one row"
    assert _badge_text(folded[0]) == "×2"

    other = _rows(panel.feed_layout)[3].alert.symbol
    assert other != repeated.symbol
    panel._ignore_alert_symbol(other)

    survivors = [
        item
        for item in _rows(panel.feed_layout)
        if item.alert.symbol == repeated.symbol
    ]
    assert len(survivors) == 1
    assert _badge_text(survivors[0]) == "×2"


def test_a_focus_change_lights_the_star_on_its_own_rows_without_rebuilding(
    tmp_path, monkeypatch
):
    """Liking a name restyles that name's star; the other 249 rows do not move.

    Today one `focusChanged` coalesces into a full `_rebuild_feed` - 24.2 s
    measured on 2026-09-08. The star is a dynamic property (`focusOn`) read
    from `theme.qss`, so it can be restyled in place with
    `style().unpolish/polish`.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)

    before = _rows(panel.feed_layout)
    before_ids = [id(item) for item in before]
    liked = before[7]
    symbol = liked.alert.symbol
    assert _star_state(liked) == "false"

    panel.focus_service.add(symbol, "long", "m5", origin="alert_center")
    panel.flush_pending_focus_refresh()

    after = _rows(panel.feed_layout)
    # Nothing was destroyed and nothing was rebuilt: same widgets, same order.
    assert [id(item) for item in after] == before_ids
    # The liked name's own row now shows the lit star.
    assert _star_state(liked) == "true"
    assert "★" in liked.feed_item.favorite_button.text()
    # Every other row keeps its unlit star.
    assert [
        _star_state(item) for item in after if item.alert.symbol != symbol
    ] == ["false"] * (len(after) - 1)


def test_a_new_alert_inserts_one_row_at_the_top_and_trims_only_the_oldest(
    tmp_path, monkeypatch
):
    """A full feed takes a new row at the top and drops exactly the bottom one.

    The insert side of the diff. Guard: the builder may not turn an insert into
    a rebuild while making the veto cheap.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)

    before_ids = _identities(panel.feed_layout)
    destroyed = _spy_destroyed(monkeypatch)

    fresh = _m5_alert(900)
    panel.add_alert(fresh)

    after = _rows(panel.feed_layout)
    assert len(after) == M5_ROWS
    assert after[0].alert.symbol == fresh.symbol
    # One in at the top, one out at the bottom, every other widget untouched.
    assert [id(item) for item in destroyed] == [before_ids[-1]]
    assert [id(item) for item in after[1:]] == before_ids[:-1]


def test_the_feed_after_veto_focus_new_alert_and_repeat_equals_a_full_rebuild(
    tmp_path, monkeypatch
):
    """The packet's headline: the diffed feed IS the rebuilt feed.

    The sequence is the packet's - veto, focus toggle, new alert, repeat - and
    the comparison is the panel's own `_rebuild_feed` run afterwards on the
    same state. Symbol, side, order, star state and repeat badge must agree.

    Two things are wrong today and both show up here. The veto and the focus
    change each run a full rebuild (so the sequence costs 350 widget trees
    twice), and `_rebuild_feed` is not fold-aware: it re-inserts BOTH entries a
    repeated name left in `self._alerts`, at the newest one's position, with no
    ×N badge - so the rebuilt feed does not match the feed the trader was
    looking at a moment earlier.
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

    live = _visible(panel)

    panel._rebuild_feed()
    rebuilt = _visible(panel)

    assert live == rebuilt
    # ...and the sequence reached that state without one rebuild.
    assert rebuilds == []


def test_a_burst_of_three_focus_changes_is_one_reaction_and_not_a_rebuild(
    tmp_path, monkeypatch
):
    """The coalescer seam survives the change (one reaction per burst)...

    ...and the reaction it runs is no longer a rebuild. `SignalCoalescer` is a
    leading-edge 200 ms window at the LISTENER: three adds inside one event-loop
    slot are one call, and `flush_pending_focus_refresh` is the seam the tests
    drive.
    """
    from ui import timer_utils

    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)

    fired: list[int] = []
    original_fire = timer_utils.SignalCoalescer._fire

    def counted(self):
        pending = self.is_pending()
        result = original_fire(self)
        if pending:
            fired.append(1)
        return result

    monkeypatch.setattr(timer_utils.SignalCoalescer, "_fire", counted)
    rebuilds = _spy_rebuild(monkeypatch, panel)

    before_ids = _identities(panel.feed_layout)
    for offset in (11, 12, 13):
        symbol = _rows(panel.feed_layout)[offset].alert.symbol
        panel.focus_service.add(symbol, "long", "m5", origin="alert_center")
    panel.flush_pending_focus_refresh()

    assert fired == [1]
    assert rebuilds == []
    assert _identities(panel.feed_layout) == before_ids
    assert sum(1 for item in _rows(panel.feed_layout) if _star_state(item) == "true") == 3


def test_the_tier_switch_and_the_feed_clear_still_rebuild_every_row(
    tmp_path, monkeypatch
):
    """`_rebuild_feed` stays, and these two callers keep using it.

    The tier-mode switch changes which alerts pass the gate for every row at
    once, and the day's Clear empties both feeds: both are a whole-feed
    decision, not a diff. Guard against a builder deleting the rebuild.
    """
    panel = _panel(tmp_path, monkeypatch)
    _fill(panel)

    rebuilds = _spy_rebuild(monkeypatch, panel)
    before_ids = _identities(panel.feed_layout)

    # Tier switch: the real combo box, the real `_on_prefs_changed`.
    panel.min_tier_input.setCurrentIndex(max(0, panel.min_tier_input.findData("A")))
    assert len(rebuilds) == 1
    after_ids = _identities(panel.feed_layout)
    assert not set(after_ids) & set(before_ids)
    # A B-tier feed under an A-tier gate shows nothing.
    assert after_ids == []

    panel.min_tier_input.setCurrentIndex(max(0, panel.min_tier_input.findData("all")))
    assert len(rebuilds) == 2
    assert len(_rows(panel.feed_layout)) == M5_ROWS

    panel.clear_feed()
    assert len(rebuilds) == 3
    assert _rows(panel.feed_layout) == []
    assert _rows(panel.d1_feed_layout) == []
