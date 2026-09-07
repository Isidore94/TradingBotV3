"""Packet G7 - the speed pass: first loads on first show, the Setup Tracker's
refresh off the Qt thread, and a bench that stops taxing what it measures.

Written by the TESTER before any fix exists, on `claude/g7-speed-pass` off
`origin/claude/g5-research-results` at ``0a1478e0`` (2026-09-07). Every test
here is RED on that commit except the ones whose docstring says
GREEN BY DESIGN - those pin the half of an item that must NOT move (the settle
result, single-flight, the rendered cells).

What each block drives, and why it can fail
-------------------------------------------
**G7.0** drives `ui.desk_bench.settle` itself over a synthetic 1,000-widget
tree. The cost is measured, not asserted from the source: the reference is a
local copy of today's full walk, timed in the same process on the same tree, and
the new per-poll cost has to be a tenth of it. The walk COUNT is read off a root
widget that counts its own `findChildren` calls, so "computed once per op" is
observed rather than described. The worker tests are the other half of the item
- the result of a settle may not change - and one of them starts its worker
300 ms INTO the settle, which is the case a cached candidate set can silently
lose.

**G7.1** patches each panel module's OWN readers (never `project_paths`: these
panels bind their file constants at import time) and counts calls. Construction
must read nothing; the first `show()` must read exactly once; a second show must
not read again. The Research-tab test is the packet's actual claim - one child's
load per first paint, not nine - and it drives the real `QTabWidget`.

**G7.2** counts `fit_columns` and `set_rows` on the panel's own thirteen tables
and models, and records `threading.get_ident()` inside a monkeypatched
`_load_csv_rows_cached`. The unchanged-file case asserts ZERO of each, so a
"cache" that skips only the parse still fails. The golden re-renders the G2b
fixture THROUGH the new asynchronous seam and holds it against
`tests/fixtures/g2b_tracker_render_golden.json`, which was pinned from
``7e018c99`` before either packet existed.

**G7.3** counts `CandleChart` children of the real `MarketJournalPanel`.

No live store is opened: `conftest.py` points `TRADINGBOTV3_DATA_DIR` at a
temporary directory and every path these panels read is patched on top of it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

#: The 4K desk viewport every G-lane packet measures at.
DESK_WIDTH = 3456
DESK_HEIGHT = 2160


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _drain(app, predicate, timeout: float = 10.0) -> bool:
    """Spin the Qt loop until `predicate()` or the deadline. Every wait carries
    a deadline (CLAUDE.md, the Qt-thread block)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


# ===========================================================================
# G7.0 - the bench stops taxing what it measures
# ===========================================================================


class _CountingRoot(QWidget):
    """A page root that counts how often anything walks its tree.

    `findChildren` is the walk `_widget_workers_running` makes twice per poll
    today. Counting it here is a black-box measure of "the candidate set is
    computed once per op", and it survives whatever the builder renames.
    """

    def __init__(self) -> None:
        super().__init__()
        self.walks = 0

    def findChildren(self, *args, **kwargs):  # noqa: N802 - Qt's own spelling
        self.walks += 1
        return super().findChildren(*args, **kwargs)


class _CountingApp:
    """Stands in for `QApplication` inside `settle`, timing its own drains.

    `settle` calls exactly one thing on the app it is handed. Wrapping it is how
    the per-poll cost is separated from the `processEvents()` it brackets: the
    poll cost is what is left of the settle after the drains.
    """

    def __init__(self, real) -> None:
        self._real = real
        self.polls = 0
        self.process_ms = 0.0

    def processEvents(self, *args, **kwargs):  # noqa: N802 - Qt's own spelling
        self.polls += 1
        started = time.perf_counter()
        self._real.processEvents(*args, **kwargs)
        self.process_ms += (time.perf_counter() - started) * 1000.0


def _synthetic_page(widgets: int = 1000) -> _CountingRoot:
    """A page the size of Research: one root, `widgets` children, a handful of
    worker attributes among them.

    The workers are FEW on purpose. A page holds a thousand widgets and three or
    four reads; a tree with a thread on every widget would be a different shape
    and would make the fix look worse than it is.
    """
    root = _CountingRoot()
    group = root
    for index in range(widgets):
        if index % 100 == 0:
            group = QWidget(root)
        child = QWidget(group)
        if index in (17, 511, 903):
            # Finished reads, still held on their attributes - the Day-trade
            # Tracker keeps `_decisions_thread` and `_held_run_thread` this way.
            child._worker = threading.Thread(target=lambda: None)
    return root


def _todays_full_walk(root) -> bool:
    """A local copy of `_widget_workers_running` as it stands on the base.

    The reference cost is measured against THIS, not against the function the
    builder is about to change, so the comparison cannot be gamed by renaming or
    by the fix leaking into its own baseline.
    """
    for worker in root.findChildren(QThread):
        try:
            if worker.isRunning():
                return True
        except RuntimeError:
            continue
    for widget in [root, *root.findChildren(QWidget)]:
        try:
            attributes = list(vars(widget).values())
        except TypeError:
            continue
        for value in attributes:
            if isinstance(value, threading.Thread) and value.is_alive():
                return True
    return False


def _reference_walk_ms(root, passes: int = 9) -> float:
    """The median cost of one of today's full walks over this tree."""
    samples = []
    for _ in range(passes):
        started = time.perf_counter()
        _todays_full_walk(root)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _settle(app, root, *, deadline_s: float):
    """`desk_bench.settle`, tolerant of the builder adding a fourth value."""
    from ui import desk_bench

    result = desk_bench.settle(app, root, deadline_s=deadline_s)
    if isinstance(result, tuple):
        return float(result[0]), float(result[1]), bool(result[2])
    return (
        float(result.settle_ms),
        float(result.longest_iteration_ms),
        bool(result.settled),
    )


def test_the_per_poll_settle_cost_on_a_thousand_widgets_drops_by_an_order_of_magnitude(app):
    """G7.0. The bench holds the GIL for a large share of the wait it times.

    Measured on a QUIET tree, where `settle` takes no 2 ms yield at all, so the
    whole of the wait is `processEvents()` plus the poll. Subtract the drains it
    timed for itself and what is left, divided by the number of polls, IS the
    per-poll cost.
    """
    root = _synthetic_page()
    try:
        reference_ms = _reference_walk_ms(root)
        counting = _CountingApp(app)
        settle_ms, _longest, settled = _settle(counting, root, deadline_s=5.0)

        assert settled is True, "a quiet page settles"
        assert counting.polls > 0
        per_poll_ms = (settle_ms - counting.process_ms) / counting.polls

        assert per_poll_ms * 10.0 <= reference_ms, (
            f"the settle poll costs {per_poll_ms:.3f} ms on a 1,000-widget tree "
            f"against a {reference_ms:.3f} ms full walk - the bench is still "
            f"paying tree-walk prices on every one of its {counting.polls} polls"
        )
    finally:
        root.deleteLater()
        app.processEvents()


def test_the_settle_poll_stops_walking_the_whole_widget_tree_every_iteration(app):
    """G7.0. The candidate set is computed once per op, re-walked at most every
    250 ms - so a settle of a few hundred milliseconds costs a handful of walks,
    not one (two, in fact) per poll."""
    root = _synthetic_page()
    worker = threading.Thread(target=lambda: time.sleep(0.4), name="g7-fixture-worker")
    root._fixture_worker = worker
    worker.start()
    try:
        root.walks = 0
        settle_ms, _longest, settled = _settle(app, root, deadline_s=5.0)

        assert settled is True
        # A re-walk at most every 250 ms over the measured settle, plus the one
        # that builds the set, and each re-walk may make two `findChildren`
        # calls (the QThread pass and the widget pass).
        allowed = 2 * (int(settle_ms // 250) + 2)
        assert root.walks <= allowed, (
            f"the settle walked the tree {root.walks} times in {settle_ms:.0f} ms; "
            f"at most {allowed} walks are allowed for that wait"
        )
    finally:
        worker.join(timeout=5.0)
        root.deleteLater()
        app.processEvents()


def test_a_plain_thread_worker_still_holds_the_settle_open(app):
    """G7.0, GREEN BY DESIGN - the result of a settle may not change.

    The Day-trade Tracker's reads are plain `threading.Thread`s held on
    attributes. A settle that stopped seeing them would report a page settled
    while it was still filling in.
    """
    root = _synthetic_page(widgets=50)
    worker = threading.Thread(target=lambda: time.sleep(0.35), name="g7-plain-worker")
    root._fixture_worker = worker
    worker.start()
    try:
        started = time.perf_counter()
        settle_ms, _longest, settled = _settle(app, root, deadline_s=5.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        assert settled is True
        assert not worker.is_alive(), "the settle returned while the read was in flight"
        assert elapsed_ms >= 350.0, f"the settle returned after only {elapsed_ms:.0f} ms"
        assert settle_ms >= 350.0
    finally:
        worker.join(timeout=5.0)
        root.deleteLater()
        app.processEvents()


def test_a_worker_that_starts_during_the_settle_is_still_waited_for(app):
    """G7.0, GREEN BY DESIGN - and the case a cached candidate set loses.

    The packet allows the set to be re-walked at most every 250 ms; a worker
    started 300 ms in must still be caught, or the bench reports a settle that
    ended before the page's second read began.
    """
    root = _synthetic_page(widgets=50)
    late: dict[str, threading.Thread] = {}

    def _late_start() -> None:
        time.sleep(0.3)
        second = threading.Thread(target=lambda: time.sleep(0.4), name="g7-late-worker")
        late["second"] = second
        root._late_worker = second
        second.start()

    starter = threading.Thread(target=_late_start, name="g7-late-starter")
    root._fixture_worker = starter
    starter.start()
    try:
        _settle_ms, _longest, settled = _settle(app, root, deadline_s=8.0)

        assert settled is True
        assert "second" in late, "the late worker never started; the test proved nothing"
        assert not late["second"].is_alive(), (
            "the settle declared the page quiet while a worker started during the "
            "wait was still reading"
        )
    finally:
        starter.join(timeout=5.0)
        if "second" in late:
            late["second"].join(timeout=5.0)
        root.deleteLater()
        app.processEvents()


def test_a_qthread_worker_still_holds_the_settle_open(app):
    """G7.0, GREEN BY DESIGN. The other worker idiom: a `ReadWorker` QThread
    parented to the widget."""

    class _Sleeper(QThread):
        def run(self) -> None:
            self.msleep(350)

    root = _synthetic_page(widgets=50)
    worker = _Sleeper(root)
    worker.start()
    try:
        _settle_ms, _longest, settled = _settle(app, root, deadline_s=5.0)

        assert settled is True
        assert not worker.isRunning(), "the settle returned while the QThread was running"
    finally:
        worker.wait(5000)
        root.deleteLater()
        app.processEvents()


def test_the_bench_json_carries_a_poll_cost_for_every_op(app):
    """G7.0. The overhead has to be VISIBLE in the artifact, not merely smaller -
    a bench that taxes what it measures should say by how much."""
    from ui import desk_bench

    root = _synthetic_page(widgets=50)
    try:
        reading = desk_bench.time_op(
            app, root, "synthetic.op", "1x1", lambda: None, deadline_s=1.0
        )
        rows = desk_bench._aggregate([reading])
    finally:
        root.deleteLater()
        app.processEvents()

    assert len(rows) == 1
    row = rows[0]
    assert "poll_cost_ms" in row, (
        f"the bench's own overhead is invisible in its JSON; the op row carries "
        f"{sorted(row)}"
    )
    stats = row["poll_cost_ms"]
    assert set(stats) >= {"n", "p50", "p95", "max"}, stats
    assert stats["n"] == 1
    assert stats["p50"] is not None and float(stats["p50"]) >= 0.0


# ===========================================================================
# G7.1 - first load on first show, the Market Journal way
# ===========================================================================


class _Calls:
    """Every call to one reader, with the thread it was made on."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.threads: list[int] = []
        self.arguments: list[str] = []

    def _record(self, args) -> None:
        self.threads.append(threading.get_ident())
        self.arguments.append(str(args[0]) if args else "")

    def returning(self, factory):
        def _call(*args, **_kwargs):
            self._record(args)
            return factory()

        return _call

    def wrapping(self, real):
        def _call(*args, **kwargs):
            self._record(args)
            return real(*args, **kwargs)

        return _call

    def __len__(self) -> int:
        return len(self.threads)

    def excluding(self, needle) -> int:
        """How many calls named something other than `needle`.

        The attribute leaderboard is read by the panel's own worker, which is
        already off the Qt thread and which the packet leaves alone; a count
        that included it would race that thread's completion.
        """
        text = str(needle)
        return sum(1 for argument in self.arguments if argument != text)

    @property
    def on_the_gui_thread(self) -> list[int]:
        gui = threading.main_thread().ident
        return [ident for ident in self.threads if ident == gui]

    def clear(self) -> None:
        self.threads.clear()
        self.arguments.clear()


def _assert_silent(*readers: _Calls) -> None:
    noisy = {reader.name: len(reader) for reader in readers if len(reader)}
    assert not noisy, (
        f"the constructor read the store before the page was ever shown: {noisy} "
        f"- the desk builds every left-nav panel at startup and most are never opened"
    )


def test_the_day_trade_tracker_reads_nothing_until_it_is_shown(app, monkeypatch):
    """G7.1. `reload_from_disk()` at the end of `__init__` is a CSV parse plus a
    JSON read plus every dimension model rebuilt and fitted, paid by a page the
    trader may never open."""
    from ui.panels import daytrade_tracker_panel as module

    performance = _Calls("_load_performance_rows")
    learning = _Calls("load_bounce_learning_state")
    monkeypatch.setattr(module, "_load_performance_rows", performance.returning(list))
    monkeypatch.setattr(module, "load_bounce_learning_state", learning.returning(dict))
    monkeypatch.setattr(
        module,
        "load_held_run_report",
        lambda *_a, **_k: {"summaries": {}, "window": {}, "outcome_coverage": {}},
    )

    panel = module.DaytradeTrackerPanel()
    try:
        app.processEvents()
        _assert_silent(performance, learning)

        panel.resize(DESK_WIDTH, DESK_HEIGHT)
        panel.show()
        assert _drain(app, lambda: len(performance) >= 1), "the first show read nothing"
        assert len(performance) == 1, f"{len(performance)} performance reads for one show"
        assert len(learning) == 1, f"{len(learning)} learning-state reads for one show"

        panel.hide()
        panel.show()
        app.processEvents()
        assert len(performance) == 1, "a page switch is not a re-read"
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


def test_the_setup_tracker_reads_nothing_until_it_is_shown(app, monkeypatch):
    """G7.1. `self.refresh()` on the last line of `__init__` - the 1.3 s op in
    the G0 baseline - runs whether or not the tab is ever selected."""
    from ui.panels import setup_tracker_panel as module

    module.clear_setup_tracker_csv_cache()
    exports = _Calls("_load_csv_rows_cached")
    human = _Calls("load_human_focus_performance_rows")
    monkeypatch.setattr(module, "_load_csv_rows_cached", exports.returning(list))
    monkeypatch.setattr(module, "load_human_focus_performance_rows", human.returning(list))

    panel = module.SetupTrackerPanel()
    try:
        app.processEvents()
        _assert_silent(exports, human)

        panel.resize(DESK_WIDTH, DESK_HEIGHT)
        panel.show()
        assert _drain(app, lambda: len(human) >= 1), "the first show read nothing"
        # The attribute leaderboard's own worker is excluded: it is already off
        # the Qt thread, the packet leaves it there, and counting it would race
        # that thread rather than measure this page.
        attributes = module.MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE
        assert len(human) == 1, f"{len(human)} human-focus reads for one show"
        assert exports.excluding(attributes) >= 12, (
            f"only {exports.excluding(attributes)} export reads for one show"
        )

        _drain(app, lambda: False, timeout=0.3)
        seen = exports.excluding(attributes)
        panel.hide()
        panel.show()
        app.processEvents()
        assert len(human) == 1, "a page switch is not a re-read"
        assert exports.excluding(attributes) == seen, "a page switch is not a re-read"
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
        module.clear_setup_tracker_csv_cache()


def test_the_market_prep_page_reads_nothing_until_it_is_shown(app, monkeypatch):
    """G7.1. Two local reads plus the human-focus picks, at construction."""
    from ui.panels import master_market_prep_panel as module

    payload = _Calls("load_market_prep_payload")
    report = _Calls("load_market_prep_report")
    picks = _Calls("load_human_focus_daily_picks")
    monkeypatch.setattr(module, "load_market_prep_payload", payload.returning(dict))
    monkeypatch.setattr(module, "load_market_prep_report", report.returning(str))
    monkeypatch.setattr(module, "load_human_focus_daily_picks", picks.returning(list))

    panel = module.MasterMarketPrepPanel()
    try:
        app.processEvents()
        _assert_silent(payload, report, picks)

        panel.resize(DESK_WIDTH, DESK_HEIGHT)
        panel.show()
        assert _drain(app, lambda: len(payload) >= 1), "the first show read nothing"
        assert len(payload) == 1 and len(report) == 1 and len(picks) == 1

        panel.hide()
        panel.show()
        app.processEvents()
        assert len(payload) == 1, "a page switch is not a re-read"
    finally:
        panel.deleteLater()
        app.processEvents()


def test_the_price_alerts_page_reads_nothing_until_it_is_shown(app, monkeypatch):
    """G7.1, the table load only. The service's own timer is a separate
    question and the packet leaves it alone, so this test hands the panel a
    service with its engine off and watches the STORE."""
    import price_alerts
    from ui.panels import price_alerts_panel as module
    from ui.services.price_alert_service import PriceAlertService

    entries = _Calls("price_alerts.load_price_alerts")
    monkeypatch.setattr(price_alerts, "load_price_alerts", entries.returning(list))

    service = PriceAlertService(engine_enabled=False)
    panel = module.PriceAlertsPanel(service)
    try:
        app.processEvents()
        _assert_silent(entries)

        panel.resize(DESK_WIDTH, DESK_HEIGHT)
        panel.show()
        assert _drain(app, lambda: len(entries) >= 1), "the first show read nothing"
        assert len(entries) == 1, f"{len(entries)} store reads for one show"

        panel.hide()
        panel.show()
        app.processEvents()
        assert len(entries) == 1, "a page switch is not a re-read"
    finally:
        panel.deleteLater()
        service.deleteLater()
        app.processEvents()


def test_the_setup_playbook_reads_no_export_on_the_gui_thread(app, monkeypatch):
    """G7.1. `setCurrentRow(0)` in `__init__` renders the overview, and the
    overview's banner does two UNCACHED CSV reads inside `render_best_now_html`
    - on the Qt thread, at startup, for a tab nobody opened."""
    from ui.panels import setup_docs_panel as module
    from ui.panels import setup_tracker_panel as tracker_module

    tracker_module.clear_setup_tracker_csv_cache()
    raw = _Calls("_load_csv_rows")
    cached = _Calls("_load_csv_rows_cached")
    monkeypatch.setattr(tracker_module, "_load_csv_rows", raw.returning(list))
    monkeypatch.setattr(tracker_module, "_load_csv_rows_cached", cached.returning(list))
    monkeypatch.setattr(module, "family_record_sentences", lambda *_a, **_k: {})

    panel = module.SetupDocsPanel()
    try:
        app.processEvents()
        assert raw.on_the_gui_thread == [] and cached.on_the_gui_thread == [], (
            f"the playbook read its exports on the Qt thread: "
            f"{len(raw.on_the_gui_thread)} uncached + {len(cached.on_the_gui_thread)} "
            f"cached reads"
        )

        panel.resize(DESK_WIDTH, DESK_HEIGHT)
        panel.show()
        _drain(app, lambda: False, timeout=0.4)
        assert raw.on_the_gui_thread == [] and cached.on_the_gui_thread == [], (
            "the first show put the export read back on the Qt thread"
        )
        # The page still renders: a deferred read is not a dropped one.
        assert panel.doc_view.toPlainText().strip(), "the overview rendered nothing"
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
        tracker_module.clear_setup_tracker_csv_cache()


def test_showing_the_research_tab_loads_only_the_child_whose_tab_is_open(app, monkeypatch):
    """G7.1, the packet's actual claim: Research's first paint costs ONE child's
    load, not nine. A `QTabWidget` child gets its `showEvent` when its tab is
    selected, so the eight tabs behind the open one stay silent until they are
    picked."""
    import price_alerts
    from ui.panels import daytrade_tracker_panel as daytrade
    from ui.panels import master_market_prep_panel as prep
    from ui.panels import setup_tracker_panel as tracker
    from ui.panels.research_panel import ResearchPanel

    tracker.clear_setup_tracker_csv_cache()
    readers = {
        "day trade": _Calls("_load_performance_rows"),
        "market prep": _Calls("load_market_prep_payload"),
        "setup tracker": _Calls("load_human_focus_performance_rows"),
        "price alerts": _Calls("price_alerts.load_price_alerts"),
    }
    monkeypatch.setattr(
        daytrade, "_load_performance_rows", readers["day trade"].returning(list)
    )
    monkeypatch.setattr(
        daytrade,
        "load_held_run_report",
        lambda *_a, **_k: {"summaries": {}, "window": {}, "outcome_coverage": {}},
    )
    monkeypatch.setattr(prep, "load_market_prep_payload", readers["market prep"].returning(dict))
    monkeypatch.setattr(
        tracker, "load_human_focus_performance_rows", readers["setup tracker"].returning(list)
    )
    monkeypatch.setattr(
        tracker, "_load_csv_rows_cached", _Calls("exports").returning(list)
    )
    monkeypatch.setattr(price_alerts, "load_price_alerts", readers["price alerts"].returning(list))

    panel = ResearchPanel(None)
    try:
        panel.resize(DESK_WIDTH, DESK_HEIGHT)
        panel.show()
        _drain(app, lambda: False, timeout=0.5)

        loud = {name: len(reader) for name, reader in readers.items() if len(reader)}
        assert not loud, (
            f"the Research tab's first paint loaded children whose tab is not open: "
            f"{loud}"
        )

        titles = [panel.tabs.tabText(index) for index in range(panel.tabs.count())]
        assert "Setup Tracker" in titles, titles
        panel.tabs.setCurrentIndex(titles.index("Setup Tracker"))
        assert _drain(app, lambda: len(readers["setup tracker"]) >= 1), (
            "selecting the Setup Tracker tab did not load it"
        )
        still_quiet = {
            name: len(reader)
            for name, reader in readers.items()
            if name != "setup tracker" and len(reader)
        }
        assert not still_quiet, f"selecting one tab loaded another: {still_quiet}"
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
        tracker.clear_setup_tracker_csv_cache()


# ===========================================================================
# G7.2 - the Setup Tracker refresh leaves the Qt thread
# ===========================================================================

#: The thirteen tables `refresh()` re-fits and the thirteen models it resets.
#: `attribute_*` is deliberately absent: it is filled by the worker slot the
#: packet leaves alone.
TRACKER_TABLES = (
    "current_table",
    "human_pick_table",
    "setup_type_table",
    "recent_type_table",
    "short_term_table",
    "playbook_table",
    "scan_factor_table",
    "tier_performance_table",
    "catch_rate_table",
    "band_variant_table",
    "control_discovery_table",
    "study_discovery_table",
    "exit_framework_table",
)
TRACKER_MODELS = tuple(name.replace("_table", "_model") for name in TRACKER_TABLES)


@pytest.fixture
def tracker_module(app):
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    yield setup_tracker_panel
    setup_tracker_panel.clear_setup_tracker_csv_cache()


def _await_refresh(app, panel, *, timeout: float = 20.0) -> None:
    """Ask for a refresh the way the Refresh button does, and wait for the rows.

    G7.2 moves the twelve export reads onto a worker, so `refresh()` stops being
    the moment the tables hold rows. The panel announces the render it made on
    the Qt thread; a panel with no such signal has no seam for a test to await,
    and that is the red state, not a missing feature of the test.
    """
    signal = getattr(panel, "refreshFinished", None)
    assert signal is not None and hasattr(signal, "connect"), (
        "SetupTrackerPanel has no `refreshFinished` signal to await, so its "
        "refresh cannot have left the Qt thread"
    )
    landed: list[bool] = []
    signal.connect(lambda *_a: landed.append(True))
    panel.refresh()
    assert _drain(app, lambda: bool(landed), timeout), "the refresh never finished"


def _counted(panel, names: tuple[str, ...], method: str, counts: dict[str, int]):
    """Count one method per named attribute, in place, on the instances the
    panel's own `refresh()` reaches."""
    for name in names:
        target = getattr(panel, name)
        real = getattr(target, method)

        def _call(*args, _real=real, _name=name, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _real(*args, **kwargs)

        setattr(target, method, _call)


def _populated_tracker(app, tracker_module, tmp_path, monkeypatch):
    """The G2b fixture exports, written to `tmp_path`, behind a real panel."""
    from tests.test_g2b_named_columns import _point_panel_at

    _point_panel_at(tracker_module, tmp_path, monkeypatch, populated=True)
    panel = tracker_module.SetupTrackerPanel()
    panel.resize(DESK_WIDTH, DESK_HEIGHT)
    panel.show()
    app.processEvents()
    return panel


def test_the_tracker_refresh_reads_its_exports_off_the_gui_thread(
    app, tracker_module, tmp_path, monkeypatch
):
    """G7.2. Twelve cached CSV reads plus the human-focus read, 1.3 s p95 at the
    desk's own size, all of it between the trader and the next repaint."""
    reads = _Calls("_load_csv_rows_cached")
    human = _Calls("load_human_focus_performance_rows")
    monkeypatch.setattr(
        tracker_module, "_load_csv_rows_cached", reads.wrapping(tracker_module._load_csv_rows_cached)
    )

    panel = _populated_tracker(app, tracker_module, tmp_path, monkeypatch)
    monkeypatch.setattr(
        tracker_module,
        "load_human_focus_performance_rows",
        human.wrapping(tracker_module.load_human_focus_performance_rows),
    )
    try:
        reads.clear()
        _await_refresh(app, panel)

        assert len(reads) >= 12, f"only {len(reads)} export reads in a refresh"
        assert reads.on_the_gui_thread == [], (
            f"{len(reads.on_the_gui_thread)} of {len(reads)} export reads ran on "
            f"the Qt thread"
        )
        assert len(human) >= 1
        assert human.on_the_gui_thread == [], "the human-focus read ran on the Qt thread"
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


def test_a_second_refresh_over_unchanged_files_refits_no_table_and_resets_no_model(
    app, tracker_module, tmp_path, monkeypatch
):
    """G7.2. The mtime cache skips only the PARSE today: the thirteen models are
    reset and the thirteen tables re-fitted on every spinbox step, over files a
    scan rewrites a few times a day."""
    panel = _populated_tracker(app, tracker_module, tmp_path, monkeypatch)
    try:
        fits: dict[str, int] = {}
        sets: dict[str, int] = {}
        _counted(panel, TRACKER_TABLES, "fit_columns", fits)
        _counted(panel, TRACKER_MODELS, "set_rows", sets)

        _await_refresh(app, panel)
        assert sum(fits.values()) >= 13, f"the first render fitted {fits}"
        assert sum(sets.values()) >= 13, f"the first render set {sets}"

        fits.clear()
        sets.clear()
        _await_refresh(app, panel)

        assert fits == {}, f"nothing changed and these tables were re-fitted: {fits}"
        assert sets == {}, f"nothing changed and these models were reset: {sets}"
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


def test_a_changed_export_refits_exactly_its_own_table(
    app, tracker_module, tmp_path, monkeypatch
):
    """G7.2. The other half: a memo that never invalidates is not a memo."""
    from tests.test_g2b_named_columns import _tracker_fixture_rows, _write_csv

    panel = _populated_tracker(app, tracker_module, tmp_path, monkeypatch)
    try:
        fits: dict[str, int] = {}
        sets: dict[str, int] = {}
        _counted(panel, TRACKER_TABLES, "fit_columns", fits)
        _counted(panel, TRACKER_MODELS, "set_rows", sets)
        _await_refresh(app, panel)
        fits.clear()
        sets.clear()

        # One export changes, and its signature changes with it: a rewrite
        # inside one test can land on the same `(mtime_ns, size)` stamp, so the
        # stamp is moved explicitly rather than hoped for.
        path = Path(str(tracker_module.BAND_VARIANT_STATS_FILE))
        rows = [dict(row) for row in _tracker_fixture_rows()["band_variant"]]
        rows[0]["setup_family"] = "a_new_family_name_from_tonights_scan"
        _write_csv(path, rows)
        stamp = time.time() + 5.0
        os.utime(path, (stamp, stamp))

        _await_refresh(app, panel)

        assert fits == {"band_variant_table": 1}, (
            f"one export changed; these tables were re-fitted: {fits}"
        )
        assert sets == {"band_variant_model": 1}, (
            f"one export changed; these models were reset: {sets}"
        )
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


def test_the_refresh_never_runs_two_reads_at_once_and_shutdown_joins_the_worker(
    app, tracker_module, tmp_path, monkeypatch
):
    """G7.2. Single-flight, and no read outlives the panel it was going to
    update. The reader is slowed so "the call returned before the read did" is a
    measurement rather than a hope."""
    in_flight = {"now": 0, "peak": 0}
    lock = threading.Lock()
    real_reader = tracker_module._load_csv_rows_cached

    def _slow(path):
        with lock:
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        try:
            time.sleep(0.05)
            return real_reader(path)
        finally:
            with lock:
                in_flight["now"] -= 1

    reads = _Calls("_load_csv_rows_cached")
    monkeypatch.setattr(tracker_module, "_load_csv_rows_cached", reads.wrapping(_slow))

    panel = _populated_tracker(app, tracker_module, tmp_path, monkeypatch)
    try:
        reads.clear()
        started = time.perf_counter()
        panel.refresh()
        panel.refresh()  # the second request coalesces; it never doubles the read
        sync_ms = (time.perf_counter() - started) * 1000.0

        assert sync_ms < 200.0, (
            f"refresh() held the Qt thread for {sync_ms:.0f} ms; twelve reads at "
            f"50 ms each did not happen anywhere else"
        )

        panel.shutdown()
        assert len(reads) >= 12, (
            f"shutdown returned with only {len(reads)} of the twelve reads done - "
            f"it did not join the worker"
        )
        assert in_flight["peak"] <= 1, (
            f"{in_flight['peak']} reads were in flight at once; the refresh is not "
            f"single-flight"
        )
        assert not _todays_full_walk(panel), "a worker outlived the shutdown"
    finally:
        panel.deleteLater()
        app.processEvents()


def test_the_worker_refresh_renders_the_g2b_golden_unchanged(
    app, tracker_module, tmp_path, monkeypatch
):
    """G7.2. Moving a read to a worker may not move a cell.

    The golden is `tests/fixtures/g2b_tracker_render_golden.json`, rendered from
    ``7e018c99`` before G2b or G7 existed. This test renders the same fixture
    through the ASYNCHRONOUS seam - the panel's own finished signal is what it
    waits on - and holds it against that file.
    """
    from tests.conftest import load_fixture_contract
    from tests.test_g2b_named_columns import (
        TRACKER_RENDER_GOLDEN,
        _render_grid,
        _tracker_fixture_rows,
    )

    contract = load_fixture_contract(TRACKER_RENDER_GOLDEN)
    assert contract["fixture_rows"] == _tracker_fixture_rows(), (
        "the G2b fixture rows moved, so this golden pins nothing"
    )

    panel = _populated_tracker(app, tracker_module, tmp_path, monkeypatch)
    try:
        _await_refresh(app, panel)
        panel._on_attributes_loaded(
            {
                "rows": tracker_module._rank_attribute_leaderboard(
                    _tracker_fixture_rows()["attribute"]
                ),
                "message": "",
            }
        )
        app.processEvents()
        rendered = _render_grid(panel)
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()

    contract.assert_matches(rendered, contract["rendered"], "setup tracker cells")


# ===========================================================================
# G7.3 - Market Journal charts on demand
# ===========================================================================


def test_the_market_journal_builds_no_candle_chart_until_a_capture_is_rendered(
    app, tmp_path, monkeypatch
):
    """G7.3. Four `CandleChart`s built synchronously in `__init__` are the only
    expensive-constructor candidate for the page's 299 ms, and the trader sees a
    chart only once they click an entry that has one."""
    from datetime import datetime, timedelta

    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)

    import market_journal_capture as mjc
    from ui.panels.market_journal_panel import MarketJournalPanel
    from ui.services.market_journal_service import MarketJournalService
    from ui.widgets.candle_chart import CandleChart

    panel = MarketJournalPanel(service=MarketJournalService())
    try:
        app.processEvents()
        assert panel.findChildren(CandleChart) == [], (
            f"{len(panel.findChildren(CandleChart))} charts were built for a page "
            f"with no entry selected"
        )

        bars = [
            {
                "dt": datetime(2026, 8, 27, 9, 30) + timedelta(minutes=5 * index),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.2,
                "volume": 1_000,
            }
            for index in range(20)
        ]
        capture = mjc.build_capture(entry_id="mj-1", symbol="DT", m5_bars=bars, d1_bars=bars)
        entry = {
            "entry_id": "mj-1",
            "session_date": "2026-08-27",
            "created_at": "2026-08-27T13:36:35+00:00",
            "timeframe": "M5",
            "symbols": ["DT"],
            "origin": "desk_tab",
            "text": "the entry the capture belongs to",
            "written_after_the_session": False,
        }
        panel._render(
            {
                "entries": [entry],
                "sessions": ["2026-08-27"],
                "timeline": {"shifts": [], "agreement": {"rate": None, "note": "none"}},
                "context": {"measured": False, "reason": "not measured"},
                "digests": {"mj-1": {"digest": capture["digest"]}},
            }
        )
        app.processEvents()
        # TRIGGER ONLY (builder, 2026-09-07). Selecting the entry above starts
        # the panel's own `_CaptureWorker`, and this fixture's capture is built
        # in memory rather than stored, so that worker lands an EMPTY payload
        # for `mj-1` - which clears the charts. Whether it lands before or after
        # the direct call below is a coin flip on machine load (green alone,
        # red under `-k "market_journal or journal_capture or g3"`). In the real
        # desk `_render_capture` is only ever called by that worker, once per
        # selection, so two landings for one entry is a test artifact and not a
        # behaviour to design around. Letting the empty one land FIRST is the
        # deterministic order; not one assertion below moved.
        _drain(
            app,
            lambda: panel._capture_worker is not None
            and not panel._capture_worker.isRunning()
            and "could not be read" in panel.charts_note.text(),
        )
        panel._render_capture("mj-1", capture)
        app.processEvents()

        assert len(panel.findChildren(CandleChart)) == 4, (
            f"the first capture built {len(panel.findChildren(CandleChart))} of the "
            f"four panes"
        )
        assert panel.charts["symbol_m5"]._bars, "the pane was built but never drawn"
        assert "20 bars" in panel.chart_titles["symbol_m5"].text()
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


# ===========================================================================
# G7 fix round (reviewer GO with advisories, 2026-09-07): four small items
# ===========================================================================


def test_a_refresh_asked_for_in_the_workers_teardown_window_is_not_orphaned(
    app, tracker_module, tmp_path, monkeypatch
):
    """G7 fix round item 1.

    `_read_until_nothing_is_pending` releases `_refresh_lock` with
    `_refresh_pending` False the moment its loop decides to stop - but the
    `ReadWorker` QThread has not actually finished at that instant, so
    `refresh()` calls made in that window still see `worker.isRunning() ==
    True`, set `_refresh_pending = True`, and return. Nobody was ever going to
    check that flag again: the loop already left. The un-fixed panel shows
    rows ranked at the PREVIOUS `min_closed` forever, until something else
    happens to call `refresh()`.

    `ReadWorker.run()` is monkeypatched (a test-only change - the file itself
    is untouched) to pause, with a `threading.Event`, exactly between the
    reader returning and the thread's own `finished` firing: the real
    teardown window, held open on purpose rather than hoped for.
    """
    import ui.read_worker as read_worker_module

    entered_window = threading.Event()

    def _slow_run(self) -> None:
        try:
            result = self._work()
        except Exception as exc:  # noqa: BLE001 - mirrors the real run()
            self.failed.emit(str(exc))
            return
        entered_window.set()
        time.sleep(0.3)
        self.finished_with.emit(result)

    monkeypatch.setattr(read_worker_module.ReadWorker, "run", _slow_run)

    seen_min_closed: list[int] = []
    real_read = tracker_module._read_tracker_exports

    def _counting_read(min_closed):
        payload = real_read(min_closed)
        seen_min_closed.append(int(payload.get("min_closed") or 0))
        return payload

    monkeypatch.setattr(tracker_module, "_read_tracker_exports", _counting_read)

    panel = _populated_tracker(app, tracker_module, tmp_path, monkeypatch)
    try:
        assert entered_window.wait(timeout=5.0), (
            "the constructor's own refresh never reached the widened window"
        )
        assert panel._read_worker is not None and panel._read_worker.isRunning(), (
            "the worker had already finished; the window was not held open"
        )

        landed: list[bool] = []
        panel.refreshFinished.connect(lambda: landed.append(True))
        panel.min_closed_input.blockSignals(True)
        panel.min_closed_input.setValue(11)
        panel.min_closed_input.blockSignals(False)
        panel.refresh()

        with panel._refresh_lock:
            assert panel._refresh_pending is True, (
                "the refresh() call landed outside the teardown window - "
                "widen it rather than change the assertion"
            )

        assert _drain(app, lambda: len(landed) >= 2, timeout=10.0), (
            "the refresh asked for during the worker's teardown never got its "
            "own pass - the page is stuck showing the previous min_closed"
        )
        assert len(seen_min_closed) >= 2, (
            f"only {len(seen_min_closed)} export reads; the pending refresh was dropped"
        )
        assert seen_min_closed[-1] == 11, (
            f"the second pass read min_closed={seen_min_closed[-1]}, not 11"
        )
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


def test_five_refreshes_leave_at_most_one_qthread_child(
    app, tracker_module, tmp_path, monkeypatch
):
    """G7 fix round item 2. Every `refresh()` built a new `ReadWorker` parented
    to the panel and none was ever deleted - five refreshes, five leaked
    `QThread` objects. `finished` is connected to `deleteLater` so a worker
    that has already handed back its rows is collected once Qt's event loop
    gets a turn."""
    panel = _populated_tracker(app, tracker_module, tmp_path, monkeypatch)
    try:
        for _ in range(5):
            _await_refresh(app, panel)

        for _ in range(10):
            app.processEvents()

        remaining = panel.findChildren(QThread)
        assert len(remaining) <= 1, (
            f"{len(remaining)} QThread children survived five refreshes - "
            f"finished workers were never deleted"
        )
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()


def test_price_alerts_save_before_first_load_writes_nothing(app, monkeypatch):
    """G7 fix round item 3. `_table_entries()` reads the `QTableWidget`, which
    is EMPTY before the first `showEvent` load (G7.1's own idiom) - so a save
    that fires before the panel was ever shown (an armed save timer, a stray
    `cellChanged` signal) would write an empty table over the real store."""
    import price_alerts
    from ui.panels import price_alerts_panel as module
    from ui.services.price_alert_service import PriceAlertService

    saves = _Calls("price_alerts.save_price_alerts")
    monkeypatch.setattr(price_alerts, "save_price_alerts", saves.returning(lambda: True))

    service = PriceAlertService(engine_enabled=True)
    panel = module.PriceAlertsPanel(service)
    try:
        app.processEvents()
        # Never shown: `_loaded_once` is still False, and `self.table` has
        # zero rows.
        panel._save_table()
        assert len(saves) == 0, (
            f"a save before the first load wrote {len(saves)} time(s) over the "
            f"real store"
        )
    finally:
        panel.deleteLater()
        service.deleteLater()
        app.processEvents()
