"""Packet G5 - Research > Results, the PANEL (`scripts/ui/panels/research_results_panel.py`).

Written by the TESTER on `claude/g5-research-results`, off `origin/main`
`7e018c99`, and proven RED there before any of it existed. The builder makes
these pass; it may ADD tests and may not weaken, skip or delete one.

Trader, 2026-09-06 (decision 0016 answer 7, AMENDED): Research gains a
**Results** landing page the trader may read. Decision 3 of that day: it opens
on **Bot setups x Swing x Recent 20 sessions**, last choice remembered.

Offscreen. **No live store is opened**: `read_persisted_snapshot` and
`journal_feed.load_trades` are monkeypatched at their own module seams, and
`local_settings` is the pytest temp `LOCALAPPDATA` `conftest.py` already points
at - asserted, not assumed, in `_settings_file`.

===========================================================================
THE API THIS FILE PINS
===========================================================================

`scripts/ui/panels/research_results_panel.py`::

    RESULTS_SELECTION_KEY = "research_results_selection"
    DEFAULT_SELECTION = ("bot", "swing", "recent")

    class ResearchResultsPanel(QFrame):
        population_buttons: dict[str, QAbstractButton]   # "bot" / "mine"
        horizon_buttons:    dict[str, QAbstractButton]   # "swing" / "day"
        window_buttons:     dict[str, QAbstractButton]   # "recent"/"all"/"custom"
        shortlist:          DataTable
        explanation_view:   ResearchExplanationView
        def selection(self) -> tuple[str, str, str]
        def freshness_text(self) -> str
        def set_working_lately_snapshot(self, payload: dict) -> None
        def refresh(self) -> None
        def shutdown(self) -> None

`scripts/ui/panels/research_panel.py`::

    ResearchPanel.results_panel                # the Results tab, FIRST
    ResearchPanel.set_working_lately_snapshot  # forwards to the results page

Visibility is asserted with `isHidden()` - the widget's own explicit hide flag,
which is what `setVisible` moves - because a child of a never-`show()`n parent
reports `isVisible() == False` whatever the pane itself was told (the G4 rule).
"""

from __future__ import annotations

import json
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
pytest.importorskip("PySide6", reason="the Results page is a Qt panel")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# The two module-scoped fixtures, imported UNDER THEIR OWN NAMES: pytest
# registers a fixture under the attribute name it finds in this module, so an
# alias would rename them and every test below would error on a missing fixture.
from test_g5_research_results import journal_trades, snapshot_payload  # noqa: E402,F401

_app = QApplication.instance() or QApplication([])

#: The thread that owns the QApplication. Nothing may read a file on it.
GUI_THREAD_IDENT = threading.get_ident()

#: The order the eight existing tabs are in today (`research_panel.py`), which
#: G5 may not disturb. The packet's prose says "the other seven"; there are
#: EIGHT, and all eight keep their order behind Results.
EXISTING_TABS = (
    "market_prep_panel",
    "setup_tracker_panel",
    "setup_docs_panel",
    "move_forensics_panel",
    "daytrade_tracker_panel",
    "ticker_lookup_panel",
    "price_alerts_panel",
    "warehouse_readout_panel",
)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _settle(widget, timeout_ms: int = 15000) -> None:
    """Wait out every reader thread the widget owns, then drain the loop."""
    for _ in range(3):
        for thread in widget.findChildren(QThread):
            thread.wait(timeout_ms)
        for _ in range(20):
            _app.processEvents()
        time.sleep(0.01)
    _app.processEvents()


def _settings_file() -> Path:
    """The pytest temp `local_settings.json` - never the trader's own."""
    import project_paths

    path = Path(project_paths.LOCAL_SETTINGS_FILE)
    assert "pytest-localappdata" in str(path), (
        f"refusing to touch a real local_settings.json at {path}"
    )
    return path


def _forget_the_saved_selection() -> None:
    import project_paths
    from ui.panels import research_results_panel as module

    path = _settings_file()
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            payload = {}
    payload.pop(module.RESULTS_SELECTION_KEY, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    project_paths.invalidate_local_settings_cache()


@pytest.fixture
def reads(monkeypatch, snapshot_payload, journal_trades):
    """Every disk read the page makes, replaced by a fixture that records its thread."""
    from ui.services import journal_feed
    from ui.services import working_lately_service

    state = {"snapshot_idents": [], "trade_idents": []}

    def fake_snapshot(*_args, **_kwargs):
        state["snapshot_idents"].append(threading.get_ident())
        return json.loads(json.dumps(snapshot_payload))

    def fake_trades(*_args, **_kwargs):
        state["trade_idents"].append(threading.get_ident())
        return list(journal_trades)

    monkeypatch.setattr(working_lately_service, "read_persisted_snapshot", fake_snapshot)
    monkeypatch.setattr(journal_feed, "load_trades", fake_trades)
    # The panel may have pulled either name into its own namespace at import.
    from ui.panels import research_results_panel as module

    for name, replacement in (
        ("read_persisted_snapshot", fake_snapshot),
        ("load_trades", fake_trades),
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, replacement)
    return state


@pytest.fixture
def page(reads):
    """A built Results page, its first read settled."""
    from ui.panels.research_results_panel import ResearchResultsPanel

    panel = ResearchResultsPanel()
    _settle(panel)
    try:
        yield panel
    finally:
        panel.shutdown()
        _settle(panel)
        panel.deleteLater()
        _app.processEvents()
        # The page WRITES its selection - real production behaviour, and this
        # is one of the two files that make it happen. A remembered "My
        # trades" makes every later `ResearchPanel` construction open the
        # journal on its worker, which creates the journal database, which is
        # the state `test_qt_journal_panel::test_migration_failure_stays_visible
        # _instead_of_claiming_no_accounts` exists to find the absence of. The
        # cure used to be an autouse fixture in `conftest.py` standing over
        # 1,800 tests; the leak belongs to the tests that leak.
        _forget_the_saved_selection()


def _click(button) -> None:
    button.setChecked(True)
    button.click()
    _app.processEvents()


# ===========================================================================
# 6
# ===========================================================================


def test_research_opens_on_results_and_the_existing_tabs_keep_their_order():
    """Item G5.3.6. The landing page is the landing page.

    Deliberately WITHOUT the `reads` fixture: `ResearchPanel` exists today, so
    this must fail on its own assertion - "there is no Results tab" - rather
    than on a fixture that cannot import the module yet. Construction under
    `conftest.py`'s temp `LOCALAPPDATA` / `TRADINGBOTV3_DATA_DIR` reaches no
    live store, which is what `test_st6_service_and_surfaces.py` already does.

    Decision 0016 as amended: Research gains a Results page *the trader may
    read*. A page that is eighth in the strip and not the current one is not a
    landing page, and the pointer label under the header still tells the trader
    the whole tab is somebody else's.
    """
    from ui.panels.research_panel import ResearchPanel

    panel = ResearchPanel(None)
    try:
        _settle(panel)
        tabs = panel.tabs

        assert hasattr(panel, "results_panel"), "ResearchPanel has no Results page"
        assert tabs.widget(0) is panel.results_panel, (
            f"tab 0 is {tabs.tabText(0)!r}, not Results"
        )
        assert "Results" in tabs.tabText(0)
        assert tabs.currentIndex() == 0, (
            f"Research does not OPEN on Results; it opens on {tabs.tabText(tabs.currentIndex())!r}"
        )

        assert tabs.count() == len(EXISTING_TABS) + 1, (
            f"expected Results plus the {len(EXISTING_TABS)} existing tabs, got {tabs.count()}"
        )
        for offset, attribute in enumerate(EXISTING_TABS, start=1):
            assert tabs.widget(offset) is getattr(panel, attribute), (
                f"tab {offset} is {tabs.tabText(offset)!r}, not {attribute}"
            )

        # The click-through ST6 built still lands on the Setup Tracker.
        panel.show_setup_tracker()
        assert tabs.currentWidget() is panel.setup_tracker_panel

        # The pointer label is reworded to the amendment: Results is readable.
        from PySide6.QtWidgets import QLabel

        pointer = " ".join(
            label.text()
            for label in panel.findChildren(QLabel)
            if "BUILDER" in label.text() or "Results" in label.text()
        )
        assert "Results" in pointer, (
            "the pointer label still describes Research as the builder's surface "
            f"only: {pointer!r}"
        )

        # And the page forwards a pushed snapshot to the Results tab.
        assert hasattr(panel, "set_working_lately_snapshot"), (
            "ResearchPanel has no set_working_lately_snapshot seam for app.py to connect"
        )
    finally:
        panel.shutdown()
        _settle(panel)
        for thread in panel.findChildren(QThread):
            assert not thread.isRunning(), (
                "ResearchPanel.shutdown left a reader thread running - the Results "
                "page is not on its by-hand shutdown list"
            )
        panel.deleteLater()
        _app.processEvents()


# ===========================================================================
# 7
# ===========================================================================


def test_the_default_selection_is_bot_swing_recent_and_a_change_survives_a_rebuild(
    reads, snapshot_payload
):
    """Item G5.3.7. Decision 3 of 2026-09-06, and "last choice remembered"."""
    import project_paths
    from ui.panels import research_results_panel as module
    from ui.panels.research_results_panel import ResearchResultsPanel

    _forget_the_saved_selection()
    assert project_paths.get_local_setting(module.RESULTS_SELECTION_KEY) is None

    first = ResearchResultsPanel()
    try:
        _settle(first)
        assert first.selection() == ("bot", "swing", "recent"), (
            f"Results must open on Bot setups x Swing x Recent, got {first.selection()}"
        )
        assert module.DEFAULT_SELECTION == ("bot", "swing", "recent")

        # The freshness line names the reading the page is showing.
        first.set_working_lately_snapshot(snapshot_payload)
        _settle(first)
        freshness = first.freshness_text()
        assert snapshot_payload["snapshot_id"][:8] in freshness, (
            f"the freshness line does not name the snapshot: {freshness!r}"
        )
        assert snapshot_payload["as_of"] in freshness

        # Change every control through its own real button.
        _click(first.population_buttons["mine"])
        _click(first.horizon_buttons["day"])
        _click(first.window_buttons["all"])
        _settle(first)
        assert first.selection() == ("mine", "day", "all")
    finally:
        first.shutdown()
        _settle(first)
        first.deleteLater()
        _app.processEvents()

    project_paths.invalidate_local_settings_cache()
    saved = project_paths.get_local_setting(module.RESULTS_SELECTION_KEY)
    assert saved, f"the selection was never written to {module.RESULTS_SELECTION_KEY!r}"

    second = ResearchResultsPanel()
    try:
        _settle(second)
        assert second.selection() == ("mine", "day", "all"), (
            f"the selection did not survive a rebuild, got {second.selection()}"
        )
    finally:
        second.shutdown()
        _settle(second)
        second.deleteLater()
        _app.processEvents()
        _forget_the_saved_selection()


# ===========================================================================
# 8
# ===========================================================================


def test_changing_a_control_hides_the_detail_pane_and_a_refresh_re_shows_the_same_row(
    page, snapshot_payload
):
    """Item G5.3.8. G4's rule, one page later.

    A detail pane that outlives the context that opened it is read as the
    numbers beside it. Changing a control changes the population, so the pane
    comes down; a plain refresh does not, so the same row comes back.
    """
    page.set_working_lately_snapshot(snapshot_payload)
    _settle(page)

    view = page.explanation_view
    assert view.isHidden(), "the detail pane is on screen before any row was clicked"

    model = page.shortlist.model()
    assert model is not None and model.rowCount() > 0, (
        "the shortlist is empty, so there is no row to select"
    )
    page.shortlist.clicked.emit(model.index(0, 0))
    _app.processEvents()

    assert not view.isHidden(), "clicking a shortlist row showed nothing"
    identity = view.shown_identity
    assert identity is not None, "the pane does not know what it is showing"

    # A plain refresh: the same row is still there, so it comes back.
    page.refresh()
    _settle(page)
    assert not view.isHidden(), "a refresh took down a row that is still on the page"
    assert view.shown_identity == identity, (
        f"the pane came back showing something else: {view.shown_identity!r} != {identity!r}"
    )

    # A CONTROL change is a different population. The pane goes.
    _click(page.horizon_buttons["day"])
    _settle(page)
    assert view.isHidden(), (
        "the detail pane survived a control change - it now describes a row that "
        "is not in the table beside it"
    )
    assert view.shown_identity is None, (
        f"a hidden pane still claims to be showing {view.shown_identity!r}"
    )

    # And the same is true of the other two controls.
    page.shortlist.clicked.emit(page.shortlist.model().index(0, 0))
    _app.processEvents()
    assert not view.isHidden()
    _click(page.window_buttons["all"])
    _settle(page)
    assert view.isHidden(), "the window control did not take the pane down"

    page.shortlist.clicked.emit(page.shortlist.model().index(0, 0))
    _app.processEvents()
    assert not view.isHidden()
    _click(page.population_buttons["mine"])
    _settle(page)
    assert view.isHidden(), "the population control did not take the pane down"


# ===========================================================================
# 9
# ===========================================================================


def test_the_snapshot_and_the_journal_are_never_read_on_the_qt_thread(page, reads):
    """Item G5.3.9. "Nothing expensive belongs on the Qt thread."

    The snapshot is a JSON file under `%LOCALAPPDATA%` and the journal is a
    SQLite query over the trader's whole trade history. Both are disk, and this
    page is the Research tab's landing page, so both happen on the way in.
    """
    assert reads["snapshot_idents"], (
        "the page never read the persisted snapshot at all - a page with no "
        "pushed payload must read one on the worker"
    )
    assert GUI_THREAD_IDENT not in reads["snapshot_idents"], (
        "read_persisted_snapshot ran on the GUI thread"
    )

    _click(page.population_buttons["mine"])
    _settle(page)

    assert reads["trade_idents"], "My trades never loaded the journal"
    assert GUI_THREAD_IDENT not in reads["trade_idents"], (
        "journal_feed.load_trades ran on the GUI thread"
    )

    # A refresh is the same promise.
    before = len(reads["snapshot_idents"]) + len(reads["trade_idents"])
    page.refresh()
    _settle(page)
    after = len(reads["snapshot_idents"]) + len(reads["trade_idents"])
    assert after > before, "refresh() read nothing"
    assert GUI_THREAD_IDENT not in reads["snapshot_idents"]
    assert GUI_THREAD_IDENT not in reads["trade_idents"]
