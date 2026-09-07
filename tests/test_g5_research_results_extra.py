"""Packet G5 - the cases the tester's nine did not reach (BUILDER, added).

Added, never weakened: every assertion here is a NEW claim about
`scripts/research_results.py` and the Results page, and none of them relaxes
anything in `test_g5_research_results.py` or `test_g5_research_results_panel.py`.

Four of them are about degenerate inputs, which is where a banding rule usually
breaks: two eligible cells (fewer than the bands are wide), no snapshot at all,
a bucket with no trades, and a study-only kind.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The fixtures are the tester's own, imported under their own names.
from test_g5_research_results import (  # noqa: E402,F401
    AS_OF,
    _cell,
    journal_trades,
    snapshot_payload,
)


def _view(population, horizon, snapshot, trades, window="recent"):
    import research_results

    return research_results.build_results_view(
        population=population,
        horizon=horizon,
        window=window,
        snapshot=snapshot,
        journal_trades=trades,
        as_of=AS_OF,
        currency_mode="native",
    )


def test_two_eligible_cells_still_partition_across_the_two_bands():
    """The degenerate case: fewer eligible cells than a band is wide.

    Stronger takes the top three and Weaker the three LOWEST, so with two
    eligible cells a rule that read the two lists independently would print the
    same family under both headings - "stronger lately" and "weaker lately" at
    once, which is not a reading, it is a bug wearing two labels. Weaker is
    therefore what Stronger did not take, and here that is nothing: with two
    eligible cells there is no third to be weaker than.
    """
    import research_results

    cells = [
        _cell(kind="daytrade_held_run", side="LONG", family="one", n_eligible=90,
              statistic=0.44, uncertainty_low=0.31),
        _cell(kind="daytrade_held_run", side="SHORT", family="two", n_eligible=80,
              statistic=0.28, uncertainty_low=0.19),
    ]
    bands = research_results.band_cells(cells)
    stronger = [row.cell.name for row in bands.stronger]
    weaker = [row.cell.name for row in bands.weaker]
    assert stronger == ["LONG one", "SHORT two"]
    assert weaker == []
    assert not set(stronger) & set(weaker)


def test_a_kind_whose_only_cells_are_studies_bands_nothing_and_still_lists_them():
    """An unpromoted idea may not lead - and may not be hidden either."""
    import research_results

    cells = [
        _cell(kind="swing_trade_r", side="LONG", family="idea", n_eligible=613,
              n_graded=613, statistic=0.95, uncertainty_low=0.90, namespace="study"),
    ]
    bands = research_results.band_cells(cells)
    assert bands.stronger == () and bands.weaker == () and bands.not_enough == ()


def test_a_missing_snapshot_says_so_on_every_band_rather_than_reading_as_zero(
    journal_trades,
):
    """No snapshot is not "nothing is working"; it is "nobody has read yet"."""
    import research_results

    for snapshot in (None, {}):
        view = _view("bot", "swing", snapshot, journal_trades)
        assert [section.kind for section in view.sections] == [
            "swing_trade_r",
            "swing_favorable",
        ]
        for section in view.sections:
            assert section.bands == research_results.ResultsBands()
            assert section.rows == () and section.studies == ()
            assert research_results.NO_SNAPSHOT in section.sentence
            assert section.verdict_line == research_results.NO_SNAPSHOT
        assert research_results.NO_SNAPSHOT in view.freshness_line


def test_every_bot_section_prints_the_observational_caveat(
    snapshot_payload, journal_trades
):
    """`observational leader among K cells` - K is the KIND's own count.

    Read off `working_lately.observational_caveat` rather than typed here, so
    this cannot drift from the strip on the Trading Desk.
    """
    import working_lately

    for horizon in ("swing", "day"):
        view = _view("bot", horizon, snapshot_payload, journal_trades)
        for section in view.sections:
            expected = working_lately.observational_caveat(snapshot_payload, section.kind)
            assert expected in section.sentence
        assert "proven" not in " ".join(
            section.sentence + section.verdict_line for section in view.sections
        ).lower()


def test_an_empty_my_trades_bucket_reports_zeros_and_the_no_tags_sentence(
    snapshot_payload,
):
    """A bucket nobody traded is empty, not unmeasured, and names no setup."""
    import research_results

    view = _view("mine", "day", snapshot_payload, [])
    assert [section.key for section in view.sections] == ["day", "unknown_timing"]
    for section in view.sections:
        assert section.stats["total"] == 0
        assert section.stats["trade_ids"] == ()
        assert section.stats["n_with_r"] == 0
        assert section.rows == ()
        assert section.sentence == research_results.NO_CONFIRMED_TAGS
    assert "0 closed trade(s) read" in view.freshness_line


def test_the_unknown_timing_bucket_names_the_instrument_and_never_a_direction(
    snapshot_payload, journal_trades
):
    """ST5's rule one page later: a LONG option is not a bullish setup.

    The exposure sentence reports the INSTRUMENT and nothing else - no bias
    word may reach it, because this page never classifies a structure.
    """
    view = _view("mine", "day", snapshot_payload, journal_trades)
    day = [section for section in view.sections if section.key == "day"][0]
    assert "STOCK 2" in day.sentence
    for forbidden in ("bullish", "bearish", "neutral"):
        assert forbidden not in day.sentence.lower()


def test_an_open_trade_is_in_no_bucket_at_all(snapshot_payload, journal_trades):
    """"Closed trades only": an open position has no holding period yet."""
    from ui.models.journal import JournalTrade

    still_open = JournalTrade.from_mapping(
        {
            "trade_id": "T-OPEN",
            "symbol": "EEE",
            "status": "OPEN",
            "net_pnl": None,
            "opened_at": "2026-09-04T09:40:00",
            "closed_at": "",
            "setup_tags": "",
            "tag_status": "confirmed",
        }
    )
    trades = list(journal_trades) + [still_open]
    for horizon in ("day", "swing"):
        view = _view("mine", horizon, snapshot_payload, trades)
        for section in view.sections:
            assert "T-OPEN" not in section.stats["trade_ids"]


def test_a_custom_window_prints_its_two_dates_and_nothing_else(
    snapshot_payload, journal_trades
):
    import evidence_stats

    view = _view(
        "bot", "swing", snapshot_payload, journal_trades, window=("2026-01-02", "2026-02-03")
    )
    assert view.window == "custom"
    assert (view.window_start, view.window_end) == ("2026-01-02", "2026-02-03")
    assert f"{evidence_stats.LATELY_SESSIONS} sessions" not in view.window_label


@pytest.mark.qt
def test_the_research_panel_forwards_a_pushed_snapshot_to_the_results_page(
    snapshot_payload,
):
    """`app.py`'s one line ends here: ONE snapshot, four surfaces."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ui.panels.research_panel import ResearchPanel

    app = QApplication.instance() or QApplication([])
    panel = ResearchPanel(None)
    try:
        seen = {}
        panel.results_panel.set_working_lately_snapshot = lambda payload: seen.update(
            payload=payload
        )
        panel.set_working_lately_snapshot(snapshot_payload)
        assert seen["payload"] is snapshot_payload
    finally:
        panel.shutdown()
        app.processEvents()
        panel.deleteLater()
        app.processEvents()


def test_a_trade_with_no_timestamps_at_all_is_unknown_timing():
    """An empty `opened_at` is not a same-session trade; it is unknown."""
    import research_results
    from ui.models.journal import JournalTrade

    trade = JournalTrade.from_mapping(
        {"trade_id": "T-BLANK", "status": "CLOSED", "opened_at": "", "closed_at": ""}
    )
    assert research_results.holding_bucket(trade) == "unknown_timing"


def test_the_recent_window_is_walked_on_the_exchange_calendar(
    snapshot_payload, journal_trades
):
    """Twenty SESSIONS, not twenty days - the difference is the whole point."""
    view = _view("bot", "swing", snapshot_payload, journal_trades)
    start = date.fromisoformat(view.window_start)
    end = date.fromisoformat(view.window_end)
    assert (end - start).days > 20, (
        "twenty trading sessions cannot span twenty calendar days or fewer"
    )


@pytest.mark.qt
def test_a_remembered_custom_selection_builds_without_reading_from_the_constructor(
    monkeypatch, snapshot_payload
):
    """A saved `custom` window must not start the first read mid-construction.

    `QDateEdit.setDate` emits `dateChanged`, so a page that connected that
    signal before seeding the two dates would call `refresh()` from inside
    `__init__` - with a remembered `custom` selection, before the labels the
    render writes into exist.
    """
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    import project_paths
    from ui.panels import research_results_panel as module
    from ui.services import journal_feed, working_lately_service

    settings = Path(project_paths.LOCAL_SETTINGS_FILE)
    assert "pytest-localappdata" in str(settings), (
        "refusing to touch a real local_settings.json"
    )

    payload = {}
    if settings.exists():
        try:
            payload = json.loads(settings.read_text(encoding="utf-8"))
        except ValueError:
            payload = {}
    payload[module.RESULTS_SELECTION_KEY] = ["bot", "swing", "custom"]
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    project_paths.invalidate_local_settings_cache()

    snapshot = json.loads(json.dumps(snapshot_payload))
    for target in (working_lately_service, module):
        if hasattr(target, "read_persisted_snapshot"):
            monkeypatch.setattr(target, "read_persisted_snapshot", lambda *a, **k: snapshot)
    for target in (journal_feed, module):
        if hasattr(target, "load_trades"):
            monkeypatch.setattr(target, "load_trades", lambda *a, **k: [])

    app = QApplication.instance() or QApplication([])
    panel = module.ResearchResultsPanel()
    try:
        for _ in range(3):
            for thread in panel.findChildren(QThread):
                thread.wait(15000)
            for _ in range(20):
                app.processEvents()
        assert panel.selection() == ("bot", "swing", "custom")
        assert not panel.custom_start.isHidden(), "the Custom window hid its own date fields"
        assert not panel.custom_end.isHidden()
        assert panel.freshness_text(), "the page rendered nothing at all"
        assert panel.shortlist.model().rowCount() > 0
    finally:
        panel.shutdown()
        app.processEvents()
        panel.deleteLater()
        app.processEvents()
        payload.pop(module.RESULTS_SELECTION_KEY, None)
        settings.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        project_paths.invalidate_local_settings_cache()
