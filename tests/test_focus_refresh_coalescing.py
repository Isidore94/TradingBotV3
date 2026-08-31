"""The 2026-08-31 refresh storm: one Focus add must not repaint the desk.

The morning of 2026-08-31 the desk was Not Responding for ~500 s across a
16-minute session, with single stalls up to 44.3 s. At 07:41:58-07:42:11 the
Alert Center drain adopted **45 staged picks into M5 Focus one at a time**, and
the 15.2 s stall charged to `focus_picks_panel.py:441` landed at the end of that
burst.

`FocusPickStore.add()` notifies on every add, so 45 adds emitted `focusChanged`
45 times, and every listener treated each one as "rebuild everything":

* the Focus board rebuilt all four side editors AND forced a snapshot write;
* the Alert Center destroyed and rebuilt its whole alert feed - up to 350
  widget trees, each with its own stylesheet;
* the setups table repainted its entire viewport through `SetupTableDelegate`;
* the strength board re-rendered its HTML document;
* the price-alert board cleared and refilled its symbol combo.

None of that is wrong per add; all of it is wrong 45 times in 13 seconds. These
tests pin the fix: **the store still emits per mutation** (other listeners rely
on it), and every listener collapses a burst into ONE reaction.
"""

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

pytestmark = pytest.mark.qt


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _spin(predicate, timeout_ms: int = 2000) -> bool:
    """Run the real event loop until `predicate` holds or the budget is spent."""
    app = _app()
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    return FocusService(store)


#: The size of the burst that froze the desk, straight from
#: `focus_auto_picks.json` for 2026-08-31.
BURST = [f"SYM{index:02d}" for index in range(45)]


def _chips(editor):
    return [
        editor.chip_flow.itemAt(index).widget()
        for index in range(editor.chip_flow.count())
    ]


def _chip_symbols(editor):
    return [chip.symbol for chip in _chips(editor)]


# ---------------------------------------------------------------------------
# The coalescer itself
# ---------------------------------------------------------------------------


def test_a_burst_inside_one_event_loop_slot_fires_once():
    _app()
    from ui.timer_utils import SignalCoalescer

    calls: list[int] = []
    coalescer = SignalCoalescer(lambda: calls.append(1), 200)
    for _ in range(45):
        coalescer.request()
    assert calls == [], "nothing may run while the burst is still arriving"
    coalescer.flush()
    assert calls == [1], "45 requests, one reaction"


def test_flushing_with_nothing_owed_does_nothing():
    _app()
    from ui.timer_utils import SignalCoalescer

    calls: list[int] = []
    coalescer = SignalCoalescer(lambda: calls.append(1), 200)
    coalescer.flush()
    assert calls == []


def test_the_window_is_not_restarted_by_later_requests():
    """A trickle must fire on a fixed cadence, never be starved by its own tail.

    A plain debounce restarts on every signal, so a stream arriving faster than
    the window never fires at all. This one opens the window on the FIRST
    request and closes it on schedule.
    """
    _app()
    from ui.timer_utils import SignalCoalescer

    coalescer = SignalCoalescer(lambda: None, 200)
    coalescer.request()
    remaining = coalescer.remaining_ms()
    for _ in range(10):
        coalescer.request()
    assert coalescer.remaining_ms() <= remaining


def test_the_coalescer_really_fires_on_its_own():
    _app()
    from ui.timer_utils import SignalCoalescer

    calls: list[int] = []
    coalescer = SignalCoalescer(lambda: calls.append(1), 20)
    for _ in range(45):
        coalescer.request()
    assert _spin(lambda: calls == [1]), f"expected exactly one call, got {calls}"


def test_a_failing_reaction_never_leaves_the_window_stuck_open():
    """Evidence must never cost the thing it records - nor a repaint the next one."""
    _app()
    from ui.timer_utils import SignalCoalescer

    calls: list[int] = []

    def explode() -> None:
        calls.append(1)
        raise RuntimeError("boom")

    coalescer = SignalCoalescer(explode, 200)
    coalescer.request()
    coalescer.flush()
    coalescer.request()
    coalescer.flush()
    assert calls == [1, 1], "a raised reaction must not disarm the coalescer"


# ---------------------------------------------------------------------------
# F1 - the Focus board
# ---------------------------------------------------------------------------


def test_a_burst_of_adds_refreshes_the_focus_board_once(tmp_path, monkeypatch):
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    refreshes: list[int] = []
    snapshots: list[bool] = []
    monkeypatch.setattr(panel, "_refresh_all", lambda: refreshes.append(1))
    monkeypatch.setattr(
        panel,
        "snapshot_today",
        lambda **kwargs: snapshots.append(bool(kwargs.get("force"))),
    )

    for symbol in BURST:
        panel.service.store.add(symbol, "long", "m5")

    assert refreshes == [], "the burst must not repaint the board 45 times"
    panel.flush_pending_refresh()
    assert refreshes == [1], "one refresh for the whole burst"
    assert snapshots == [True], "one forced snapshot, after the last add"


def test_the_store_still_emits_once_per_mutation(tmp_path):
    """The coalescing lives in the panel; the signal contract is untouched."""
    _app()
    service = _service(tmp_path)
    emissions: list[int] = []
    service.focusChanged.connect(lambda: emissions.append(1))
    for symbol in BURST:
        service.store.add(symbol, "long", "m5")
    assert len(emissions) == len(BURST)


def test_the_coalesced_snapshot_still_runs_after_the_last_add(tmp_path):
    """Merge semantics unchanged: the snapshot sees the whole burst, not part."""
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    seen: list[int] = []
    panel.snapshot_today = lambda **_k: seen.append(  # type: ignore[method-assign]
        len(panel.service.focus_symbols("long", "m5"))
    )
    for symbol in BURST:
        panel.service.store.add(symbol, "long", "m5")
    panel.flush_pending_refresh()
    assert seen == [len(BURST)]


def test_the_board_catches_up_without_being_flushed_by_hand(tmp_path):
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    for symbol in BURST:
        panel.service.store.add(symbol, "long", "m5")
    assert _spin(lambda: panel.m5_long_editor.chip_flow.count() == len(BURST))


# ---------------------------------------------------------------------------
# F2 - the side editor is a real diff
# ---------------------------------------------------------------------------


def _layout_spy(editor, monkeypatch):
    """Count every genuine layout mutation on the chip flow."""
    ops: list[str] = []
    flow = editor.chip_flow
    real_add = flow.addWidget
    real_take = flow.takeAt

    monkeypatch.setattr(
        flow, "addWidget", lambda widget: (ops.append("add"), real_add(widget))[1]
    )
    monkeypatch.setattr(
        flow, "takeAt", lambda index: (ops.append("take"), real_take(index))[1]
    )
    return ops


def test_an_unchanged_symbol_list_performs_no_layout_work(tmp_path, monkeypatch):
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    editor = panel.m5_long_editor
    panel.service.add_many(BURST, "long", "m5")
    panel.flush_pending_refresh()
    assert editor.chip_flow.count() == len(BURST)

    ops = _layout_spy(editor, monkeypatch)
    editor.refresh()
    assert ops == [], "an unchanged board must not touch the layout at all"


def test_an_unchanged_refresh_still_hands_every_chip_its_state(tmp_path):
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    editor = panel.m5_long_editor
    panel.service.add_many(["NVDA", "AMD"], "long", "m5")
    panel.flush_pending_refresh()

    updated: list[str] = []
    for chip in _chips(editor):
        real = chip.update_state
        chip.update_state = lambda state, _c=chip, _r=real: (
            updated.append(_c.symbol),
            _r(state),
        )[1]
    editor.refresh()
    assert updated == ["NVDA", "AMD"]


def test_an_arrival_only_adds_the_new_chip(tmp_path, monkeypatch):
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    editor = panel.m5_long_editor
    panel.service.add_many(["NVDA", "AMD"], "long", "m5")
    panel.flush_pending_refresh()

    panel.service.store.add("TSLA", "long", "m5")
    ops = _layout_spy(editor, monkeypatch)
    editor.refresh()
    assert ops.count("add") == 1, "only the arrival is inserted"
    assert ops.count("take") == 0, "nothing that stayed is removed"
    assert _chip_symbols(editor) == ["NVDA", "AMD", "TSLA"]


def test_a_departure_removes_only_that_chip(tmp_path, monkeypatch):
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    editor = panel.m5_long_editor
    panel.service.add_many(["NVDA", "AMD", "TSLA"], "long", "m5")
    panel.flush_pending_refresh()

    panel.service.remove("AMD", "long", "m5")
    ops = _layout_spy(editor, monkeypatch)
    editor.refresh()
    assert ops.count("add") == 0, "the chips that stayed are not re-inserted"
    assert _chip_symbols(editor) == ["NVDA", "TSLA"]


def test_a_reorder_is_still_honoured(tmp_path):
    """The trader reads the board as a list, so the service's order wins."""
    _app()
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    editor = panel.m5_long_editor
    panel.service.add_many(["NVDA", "AMD", "TSLA"], "long", "m5")
    panel.flush_pending_refresh()

    panel.service.remove("NVDA", "long", "m5")
    panel.service.store.add("NVDA", "long", "m5")
    editor.refresh()
    assert _chip_symbols(editor) == ["AMD", "TSLA", "NVDA"]


# ---------------------------------------------------------------------------
# F3 - a bounce alert is one chip, not four editors
# ---------------------------------------------------------------------------


def test_a_bounce_alert_does_not_rebuild_unrelated_editors(tmp_path, monkeypatch):
    _app()
    from ui.models.bounce import BounceAlert
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.service.add_many(["NVDA", "AMD"], "long", "m5")
    panel.service.add_many(["TSLA"], "short", "m5")
    panel.service.add_many(["MSFT"], "long", "swing")
    panel.flush_pending_refresh()

    refreshed: list[str] = []
    for editor in panel.editors:
        real = editor.refresh
        editor.refresh = lambda _e=editor, _r=real: (  # type: ignore[method-assign]
            refreshed.append(f"{_e.category}/{_e.side}"),
            _r(),
        )[1]
    reads: list[int] = []
    monkeypatch.setattr(panel, "refresh_reviewed_today", lambda: reads.append(1))

    panel.record_bounce_alert(
        BounceAlert.from_callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")
    )

    assert refreshed == [], "one chip's badge must not rebuild four editors"
    assert reads == [], "and must not re-read the feedback file"


def test_a_bounce_alert_still_lights_the_matching_chip(tmp_path):
    _app()
    from ui.models.bounce import BounceAlert
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.service.add_many(["NVDA", "AMD"], "long", "m5")
    panel.flush_pending_refresh()

    panel.record_bounce_alert(
        BounceAlert.from_callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")
    )
    chips = {chip.symbol: chip for chip in _chips(panel.m5_long_editor)}
    assert chips["NVDA"].live_flag.text() == "BOUNCE"
    assert chips["AMD"].live_flag.text() == ""


def test_a_bounce_alert_for_a_name_on_two_boards_lights_both(tmp_path):
    _app()
    from ui.models.bounce import BounceAlert
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.service.add_many(["NVDA"], "long", "m5")
    panel.service.add_many(["NVDA"], "long", "swing")
    panel.flush_pending_refresh()

    panel.record_bounce_alert(
        BounceAlert.from_callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")
    )
    for editor in (panel.m5_long_editor, panel.swing_long_editor):
        chip = editor.chip_flow.itemAt(0).widget()
        assert chip.symbol == "NVDA" and chip.live_flag.text() == "BOUNCE"


def test_a_bounce_state_recorded_before_a_chip_exists_still_reaches_it(tmp_path):
    """The state is the record; the chip is a view of it.

    `record_bounce_alert` no longer rebuilds the board, so a name that joins
    Focus AFTER its alert has to pick the badge up when its chip is built.
    """
    _app()
    from ui.models.bounce import BounceAlert
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.service.add_many(["NVDA"], "long", "m5")
    panel.flush_pending_refresh()
    panel.record_bounce_alert(
        BounceAlert.from_callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")
    )

    panel.service.store.add("NVDA", "long", "swing")
    panel.flush_pending_refresh()
    chip = panel.swing_long_editor.chip_flow.itemAt(0).widget()
    assert chip.live_flag.text() == "BOUNCE"


# ---------------------------------------------------------------------------
# F5 - the other listeners of the same signal
# ---------------------------------------------------------------------------


def test_a_burst_repaints_the_setups_table_once(tmp_path, monkeypatch):
    _app()
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    service = _service(tmp_path)
    panel = MasterAvwapPanel(focus_service=service)
    repaints: list[int] = []
    monkeypatch.setattr(panel.table.viewport(), "update", lambda: repaints.append(1))

    for symbol in BURST:
        service.store.add(symbol, "long", "m5")
    assert repaints == [], "45 focus adds must not repaint the table 45 times"
    panel.flush_pending_refresh()
    assert repaints == [1]


def test_a_burst_re_renders_the_strength_board_once(tmp_path, monkeypatch):
    _app()
    from ui.widgets.focus_strength_board import FocusStrengthBoard

    service = _service(tmp_path)
    board = FocusStrengthBoard()
    board.set_focus_service(service)
    renders: list[int] = []
    monkeypatch.setattr(board, "_render", lambda: renders.append(1))

    for symbol in BURST:
        service.store.add(symbol, "long", "m5")
    assert renders == []
    board.flush_pending_refresh()
    assert renders == [1]


def test_a_burst_refills_the_price_alert_symbols_once(tmp_path, monkeypatch):
    _app()
    from ui.services.price_alert_service import PriceAlertService
    from ui.widgets.price_alert_board import PriceAlertBoard

    service = _service(tmp_path)
    board = PriceAlertBoard(PriceAlertService(), service)
    refills: list[int] = []
    monkeypatch.setattr(board, "_refresh_symbol_choices", lambda: refills.append(1))

    for symbol in BURST:
        service.store.add(symbol, "long", "m5")
    assert refills == []
    board.flush_pending_refresh()
    assert refills == [1]


def test_a_burst_rebuilds_the_alert_feed_once(tmp_path, monkeypatch):
    """Approved by the trader 2026-08-31 (ask-first rule, alert_center_panel.py).

    The trigger is coalesced and nothing else: which alerts pass the feed gate,
    their order, the fold and the digest are all decided inside `_rebuild_feed`
    and are untouched.
    """
    _app()
    from ui.panels.alert_center_panel import AlertCenterPanel

    service = _service(tmp_path)
    panel = AlertCenterPanel(focus_service=service)
    rebuilds: list[int] = []
    monkeypatch.setattr(panel, "_rebuild_feed", lambda: rebuilds.append(1))

    for symbol in BURST:
        service.store.add(symbol, "long", "m5")
    assert rebuilds == [], "45 focus adds must not rebuild 350 feed rows 45 times"
    panel.flush_pending_focus_refresh()
    assert rebuilds == [1]
