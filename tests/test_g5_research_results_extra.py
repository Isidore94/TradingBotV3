"""Packet G5 - the cases the tester's nine did not reach (BUILDER, added).

Added, never weakened: every assertion here is a NEW claim about
`scripts/research_results.py` and the Results page, and none of them relaxes
anything in `test_g5_research_results.py` or `test_g5_research_results_panel.py`.

Four of them are about degenerate inputs, which is where a banding rule usually
breaks: two eligible cells (fewer than the bands are wide), no snapshot at all,
a bucket with no trades, and a study-only kind.
"""

from __future__ import annotations

import contextlib
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
    monkeypatch, snapshot_payload
):
    """`app.py`'s one line ends here: ONE snapshot, four surfaces.

    Both of the Results page's reads are stubbed and its selection is asked for
    rather than inherited. Without that this test opens the REAL journal store
    on its worker whenever a previous test left the page remembering "My
    trades" - which is the page behaving correctly and the test being
    unhermetic, and it made `test_qt_journal_panel`'s migration test fail two
    files later.
    """
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import project_paths
    from ui.panels import research_results_panel as results_module
    from ui.panels.research_panel import ResearchPanel
    from ui.services import journal_feed, working_lately_service

    settings = Path(project_paths.LOCAL_SETTINGS_FILE)
    assert "pytest-localappdata" in str(settings), (
        "refusing to touch a real local_settings.json"
    )
    saved = {}
    if settings.exists():
        try:
            saved = json.loads(settings.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps(
            {**saved, results_module.RESULTS_SELECTION_KEY: ["bot", "swing", "recent"]},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    project_paths.invalidate_local_settings_cache()

    for target in (working_lately_service, results_module):
        if hasattr(target, "read_persisted_snapshot"):
            monkeypatch.setattr(target, "read_persisted_snapshot", lambda *a, **k: {})
    for target in (journal_feed, results_module):
        if hasattr(target, "load_trades"):
            monkeypatch.setattr(target, "load_trades", lambda *a, **k: [])

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
        settings.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")
        project_paths.invalidate_local_settings_cache()


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
def test_the_shortlist_labels_a_study_row_and_puts_it_under_every_live_one(
    monkeypatch, snapshot_payload
):
    """"Listed under their own label" has to be visible IN THE TABLE.

    A `study` cell with the best bound on the page sitting in the shortlist
    with no label is an unpromoted idea reading as a result. It is last, and
    the row says which population it belongs to.
    """
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    import project_paths
    from ui.models.tracker_table_model import ROW_ROLE
    from ui.panels import research_results_panel as module
    from ui.services import journal_feed, working_lately_service

    # The page REMEMBERS its selection, so a previous test's choice would decide
    # which population this one looks at. Asked for explicitly, restored after.
    settings = Path(project_paths.LOCAL_SETTINGS_FILE)
    assert "pytest-localappdata" in str(settings), (
        "refusing to touch a real local_settings.json"
    )
    saved = {}
    if settings.exists():
        try:
            saved = json.loads(settings.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps({**saved, module.RESULTS_SELECTION_KEY: ["bot", "swing", "recent"]}, indent=1)
        + "\n",
        encoding="utf-8",
    )
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
        assert panel.selection() == ("bot", "swing", "recent")
        panel.set_working_lately_snapshot(snapshot)
        for _ in range(3):
            for thread in panel.findChildren(QThread):
                thread.wait(15000)
            for _ in range(20):
                app.processEvents()

        columns = [key for key, _label in module.BOT_COLUMNS]
        assert "namespace" in columns, (
            "the shortlist has no column naming the population a row belongs to"
        )
        assert "kind" in columns, (
            "the shortlist holds two sections and no column says which MEASURE a "
            "row is - a 0.72 closed-R rate and a 58.0 favorable percent would "
            "share one Statistic column with nothing to tell them apart"
        )
        model = panel.shortlist.model()
        rows = [model.index(i, 0).data(ROW_ROLE) for i in range(model.rowCount())]
        pairs = [
            (str(row.get("kind") or ""), str(row.get("namespace") or "")) for row in rows
        ]
        assert {"swing_trade_r", "swing_favorable"} == {kind for kind, _ in pairs}
        assert "study" in {name for _kind, name in pairs}, (
            "the study cells never reached the shortlist"
        )
        # Within each measure the live rows come first and the studies last.
        for kind in ("swing_trade_r", "swing_favorable"):
            names = [name for row_kind, name in pairs if row_kind == kind]
            assert "study" in names and "live" in names, names
            first_study = names.index("study")
            assert all(name == "study" for name in names[first_study:]), (
                f"a live {kind} row sits below a study one: {names}"
            )
        # And the two measures are not interleaved: each is one contiguous block.
        kinds = [kind for kind, _name in pairs]
        assert kinds == sorted(kinds, key=lambda k: kinds.index(k)), kinds
        assert len([i for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]) == 1, (
            f"the two swing measures are interleaved in one table: {kinds}"
        )
    finally:
        panel.shutdown()
        app.processEvents()
        panel.deleteLater()
        app.processEvents()
        settings.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")
        project_paths.invalidate_local_settings_cache()


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


# ===========================================================================
# THE FIX ROUND (2026-09-07) - three blockers and eight advisories
#
# Every test below this line was written RED against the reviewed tip
# `85d11227` and is a NEW claim; nothing above it was weakened to make one
# pass.
# ===========================================================================


@pytest.fixture(autouse=True)
def _forget_the_results_selection():
    """Advisory 7. The leak belongs to the tests that leak, not to the suite.

    The Results page remembers its selection in `local_settings.json` - real
    production behaviour - and a remembered "My trades" makes every later
    `ResearchPanel` construction open the journal on its worker, which is what
    made `test_qt_journal_panel`'s migration test fail two files away. The cure
    used to be an autouse fixture in `tests/conftest.py`, standing over 1,800
    tests that never touch this key. It lives here and in the panel file's
    `page` fixture instead: the two places that write it.
    """
    yield
    try:
        import json as _json

        import project_paths as _project_paths

        path = Path(_project_paths.LOCAL_SETTINGS_FILE)
        if "pytest-localappdata" not in str(path) or not path.exists():
            return
        payload = _json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "research_results_selection" not in payload:
            return
        payload.pop("research_results_selection", None)
        path.write_text(_json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        _project_paths.invalidate_local_settings_cache()
    except Exception:  # noqa: BLE001 - a cleanup never fails a test
        pass


def test_the_suite_wide_results_selection_fixture_is_gone_from_conftest():
    """Advisory 7, fenced. A cleanup for two files may not stand over 1,800."""
    source = (ROOT_DIR / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_restore_the_results_selection" not in source, (
        "the suite-wide autouse fixture is back in conftest.py; the two files "
        "that write the key clean up after themselves"
    )


def _section(view, key):
    matches = [section for section in view.sections if section.key == key]
    assert matches, f"no section {key!r} in {[s.key for s in view.sections]}"
    return matches[0]


def _closed_row(trade_id, *, closed_at, opened_at, currency="USD", net_pnl=100.0):
    """One CLOSED `JournalTrade`, shaped the way `load_trades` shapes one."""
    from ui.models.journal import JournalTrade

    return JournalTrade.from_mapping(
        {
            "trade_id": trade_id,
            "trade_date": str(closed_at)[:10],
            "symbol": "ZZZ",
            "direction": "LONG",
            "status": "CLOSED",
            "quantity_closed": 100,
            "net_pnl": net_pnl,
            "commission": 1.0,
            "fees": 0.0,
            "currency": currency,
            "account_label": "MAIN",
            "broker": "IBKR",
            "setup_tags": "",
            "display_tags": "",
            "auto_tag_summary": "",
            "tag_status": "confirmed",
            "notes": "",
            "opened_at": opened_at,
            "closed_at": closed_at,
            "planned_risk": None,
            "instrument_kind": "STOCK",
        }
    )


def _panel_module():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from ui.panels import research_results_panel as module

    return module


@contextlib.contextmanager
def _results_panel(monkeypatch, snapshot, selection, trades=()):
    """A built Results page with both reads stubbed and its selection asked for.

    The page REMEMBERS its selection, so a test that inherited one would look
    at whichever population ran last.
    """
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import project_paths
    from ui.panels import research_results_panel as module
    from ui.services import journal_feed, working_lately_service

    settings = Path(project_paths.LOCAL_SETTINGS_FILE)
    assert "pytest-localappdata" in str(settings), (
        "refusing to touch a real local_settings.json"
    )
    saved = {}
    if settings.exists():
        try:
            saved = json.loads(settings.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps({**saved, module.RESULTS_SELECTION_KEY: list(selection)}, indent=1) + "\n",
        encoding="utf-8",
    )
    project_paths.invalidate_local_settings_cache()

    payload = json.loads(json.dumps(dict(snapshot)))
    for target in (working_lately_service, module):
        if hasattr(target, "read_persisted_snapshot"):
            monkeypatch.setattr(target, "read_persisted_snapshot", lambda *a, **k: payload)
    for target in (journal_feed, module):
        if hasattr(target, "load_trades"):
            monkeypatch.setattr(target, "load_trades", lambda *a, **k: list(trades))

    app = QApplication.instance() or QApplication([])
    panel = module.ResearchResultsPanel()
    try:
        _drain(panel)
        yield panel
    finally:
        panel.shutdown()
        app.processEvents()
        panel.deleteLater()
        app.processEvents()
        saved.pop(module.RESULTS_SELECTION_KEY, None)
        settings.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")
        project_paths.invalidate_local_settings_cache()


def _drain(panel) -> None:
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    for _ in range(3):
        for thread in panel.findChildren(QThread):
            thread.wait(15000)
        for _ in range(20):
            app.processEvents()


# ---------------------------------------------------------------------------
# BLOCKER 1 - the window control is inert
# ---------------------------------------------------------------------------


def test_a_trade_closed_years_before_the_window_is_not_counted_inside_it(
    snapshot_payload, journal_trades
):
    """BLOCKER 1. The reviewer's own reproduction: a 2019 trade under 2026.

    `_window_of` labelled the window and nothing filtered by it, so every
    selection showed the same trades under a different heading - which is worse
    than no control at all, because the heading is a claim.
    """
    ancient = _closed_row(
        "T-2019",
        opened_at="2019-03-04T09:45:00",
        closed_at="2019-03-04T14:10:00",
        net_pnl=999.0,
    )
    trades = list(journal_trades) + [ancient]

    custom = _view(
        "mine", "day", snapshot_payload, trades, window=("2026-09-01", "2026-09-04")
    )
    day = _section(custom, "day")
    assert "T-2019" not in day.stats["trade_ids"], (
        "a trade closed in 2019 was counted under the window 2026-09-01 to "
        "2026-09-04"
    )
    assert day.stats["total"] == 2 and day.stats["net_pnl"] == pytest.approx(80.0)
    assert day.stats["n_outside_window"] == 1

    recent = _view("mine", "day", snapshot_payload, trades)
    assert "T-2019" not in _section(recent, "day").stats["trade_ids"]

    # "All history" is the selection that means every session, and there the
    # 2019 trade is evidence rather than noise.
    everything = _view("mine", "day", snapshot_payload, trades, window="all")
    day_all = _section(everything, "day")
    assert "T-2019" in day_all.stats["trade_ids"]
    assert day_all.stats["n_outside_window"] == 0


def test_the_custom_window_is_inclusive_at_both_ends(snapshot_payload):
    """A window states two dates; a trade closed ON one of them is inside it."""
    trades = [
        _closed_row("T-FIRST", opened_at="2026-08-03T09:45:00", closed_at="2026-08-03T15:00:00"),
        _closed_row("T-LAST", opened_at="2026-09-04T09:45:00", closed_at="2026-09-04T15:00:00"),
        _closed_row("T-BEFORE", opened_at="2026-08-02T09:45:00", closed_at="2026-08-02T15:00:00"),
        _closed_row("T-AFTER", opened_at="2026-09-05T09:45:00", closed_at="2026-09-05T15:00:00"),
    ]
    view = _view(
        "mine", "day", snapshot_payload, trades, window=("2026-08-03", "2026-09-04")
    )
    ids = set(_section(view, "day").stats["trade_ids"])
    assert ids == {"T-FIRST", "T-LAST"}, ids


def test_the_bot_page_says_the_snapshot_owns_its_window(snapshot_payload, journal_trades):
    """BLOCKER 1, the other half. Never a date range the numbers do not have.

    A bot cell was measured over the window its own aggregator walked
    (`EvidenceCell.window_sessions`, ending at the snapshot's `as_of`).
    Printing "Custom window 2026-01-02 to 2026-02-03" over numbers that never
    saw those dates is a false claim about the evidence.
    """
    view = _view(
        "bot", "swing", snapshot_payload, journal_trades, window=("2026-01-02", "2026-02-03")
    )
    assert view.window_applies is False, (
        "the bot page claims the chosen window was applied to the snapshot"
    )
    assert "20 sessions" in view.window_sentence
    assert snapshot_payload["as_of"] in view.window_sentence
    for date_text in ("2026-01-02", "2026-02-03"):
        assert date_text not in view.freshness_line, (
            f"the bot freshness line prints {date_text}, a date its numbers never "
            f"saw: {view.freshness_line!r}"
        )

    mine = _view("mine", "day", snapshot_payload, journal_trades, window="all")
    assert mine.window_applies is True
    assert mine.window_sentence == mine.window_label


@pytest.mark.qt
def test_the_window_buttons_are_disabled_on_the_bot_page_and_say_why(
    monkeypatch, snapshot_payload
):
    """BLOCKER 1 on the screen: a control that changes nothing is not offered."""
    module = _panel_module()
    with _results_panel(monkeypatch, snapshot_payload, ("bot", "swing", "recent")) as panel:
        for key, button in panel.window_buttons.items():
            assert not button.isEnabled(), (
                f"the {key!r} window button is live on the Bot page, where it "
                "changes nothing"
            )
            assert module.WINDOW_ON_BOT_TOOLTIP in button.toolTip(), (
                f"the disabled {key!r} button does not say why"
            )

        panel.population_buttons["mine"].setChecked(True)
        panel.population_buttons["mine"].click()
        _drain(panel)
        for key, button in panel.window_buttons.items():
            assert button.isEnabled(), f"My trades left the {key!r} window button dead"


# ---------------------------------------------------------------------------
# BLOCKER 2 - the freshness line printed `an unnamed file @ None`
# ---------------------------------------------------------------------------


def _live_shaped_sources(payload):
    """The real writer's shape: `path` "" and `mtime` null, `rows` present."""
    import copy

    out = copy.deepcopy(dict(payload))
    out["sources"] = {
        "swing_trade_r": {"path": "", "mtime": None, "rows": 10, "rows_by_session": {}},
        "swing_favorable": {"path": "", "mtime": None, "rows": 3, "rows_by_session": {}},
        "daytrade_held_run": {"path": "", "mtime": None, "rows": 4, "rows_by_session": {}},
    }
    return out


def test_the_freshness_line_prints_row_counts_and_never_the_literal_none(
    snapshot_payload, journal_trades
):
    """BLOCKER 2. `working_lately` writes `path: ""` and `mtime: null`.

    Every real snapshot on this machine reads that way, so the line said
    `an unnamed file @ None` three times over. It DOES carry `rows`, which is
    the fact the trader can use.
    """
    payload = _live_shaped_sources(snapshot_payload)
    view = _view("bot", "swing", payload, journal_trades)
    line = view.freshness_line
    assert "None" not in line, f"the literal None reached the screen: {line!r}"
    assert "an unnamed file" not in line, f"still naming a file that is not there: {line!r}"
    assert "swing_trade_r <- 10 row" in line, line
    assert "daytrade_held_run <- 4 row" in line, line


def test_a_snapshot_with_no_sources_at_all_says_so(snapshot_payload, journal_trades):
    """BLOCKER 2's other half - and still never the literal None."""
    import copy

    import research_results

    payload = copy.deepcopy(dict(snapshot_payload))
    payload["sources"] = {}
    view = _view("bot", "swing", payload, journal_trades)
    assert research_results.NO_SOURCES in view.freshness_line
    assert "None" not in view.freshness_line


def test_a_source_that_kept_its_path_and_mtime_still_prints_them(
    snapshot_payload, journal_trades
):
    """The fixture's own shape: a path and an mtime are printed when present."""
    view = _view("bot", "swing", snapshot_payload, journal_trades)
    for name, source in snapshot_payload["sources"].items():
        assert str(source["path"]) in view.freshness_line, name
        assert str(source["mtime"]) in view.freshness_line, name
        assert f"{name} <- {source['rows']} row" in view.freshness_line, name


# ---------------------------------------------------------------------------
# BLOCKER 3 - a card said "27 shown" over six lines
# ---------------------------------------------------------------------------


def _many_ineligible_cells(count=7):
    """`count` below-floor cells of ONE kind - a band with no cap on it."""
    return [
        _cell(
            kind="swing_trade_r",
            side="LONG",
            family=f"thin{index}",
            n_eligible=index + 1,
            n_graded=index + 1,
            statistic=0.5,
            uncertainty_low=0.4,
            meets_floor=False,
        )
        for index in range(count)
    ]


@pytest.mark.qt
def test_a_band_card_counts_the_lines_it_actually_rendered():
    """BLOCKER 3. "27 shown" over six printed lines is a false count.

    The card prints at most three rows per section by design; the count above
    them said how many the BAND held. `N of M shown` states both, so the
    trader knows the table below is where the rest of them are.
    """
    import research_results

    module = _panel_module()
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    bands = research_results.band_cells(_many_ineligible_cells(7))
    assert len(bands.not_enough) == 7

    card = module._BandCard("Not enough evidence")
    card.set_sections([("Swing - closed R", bands.not_enough)], empty_text="nothing")
    assert card.count_label.text() == "3 of 7 shown", card.count_label.text()
    assert card.body_label.text().count("•") == 3, card.body_label.text()

    # Two sections: the count is the sum of both, and so is the rendered count.
    card.set_sections(
        [("A", bands.not_enough), ("B", bands.not_enough[:2])], empty_text="nothing"
    )
    assert card.count_label.text() == "5 of 9 shown", card.count_label.text()
    assert card.body_label.text().count("•") == 5

    card.set_sections([("A", ())], empty_text="nothing")
    assert card.count_label.text() == "nothing to show"


@pytest.mark.qt
def test_an_empty_section_leaves_no_dangling_heading_on_a_card():
    """Advisory 5. A heading with nothing under it reads as a failed read."""
    import research_results

    module = _panel_module()
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    bands = research_results.band_cells(_many_ineligible_cells(4))
    card = module._BandCard("Not enough evidence")
    card.set_sections(
        [("Swing - closed R", bands.not_enough), ("Swing - favorable", ())],
        empty_text="nothing",
    )
    assert "Swing - favorable:" not in card.body_label.text(), card.body_label.text()
    assert "Swing - closed R:" in card.body_label.text()


# ---------------------------------------------------------------------------
# Advisories 1-6
# ---------------------------------------------------------------------------


def test_the_window_button_label_reads_the_declared_session_count():
    """Advisory 1. "Recent 20 sessions" typed by hand is a second definition."""
    import evidence_stats

    module = _panel_module()
    label = dict(module.WINDOWS)["recent"]
    assert label == f"Recent {evidence_stats.LATELY_SESSIONS} sessions", label
    # A source-level guard, because the label agrees with the constant today by
    # coincidence: the button must READ `evidence_stats.LATELY_SESSIONS`, so a
    # future change to "lately" moves this button with it.
    source = (ROOT_DIR / "scripts" / "ui" / "panels" / "research_results_panel.py").read_text(
        encoding="utf-8"
    )
    assert "evidence_stats.LATELY_SESSIONS" in source, (
        "the window button types its own session count instead of reading the "
        "one definition of 'lately'"
    )
    assert '"Recent 20 sessions"' not in source


def test_a_bucket_names_the_currency_its_money_is_in(snapshot_payload, journal_trades):
    """Advisory 2. A number with no currency beside it is not money."""
    view = _view("mine", "day", snapshot_payload, journal_trades)
    day = _section(view, "day")
    assert day.stats["currency"] == "USD"
    assert "USD" in day.verdict_line, day.verdict_line


def test_a_refused_total_prints_the_reason_and_the_currencies(snapshot_payload):
    """Advisory 2. `resolve_pnl_key` refuses with a sentence; print it.

    "net unmeasured" says the measurement failed. It did not: the trades are
    measured, in two currencies nobody converted - a different fact, and the
    one the trader can act on.
    """
    trades = [
        _closed_row(
            "T-USD",
            opened_at="2026-09-02T09:45:00",
            closed_at="2026-09-02T15:00:00",
            currency="USD",
        ),
        _closed_row(
            "T-CAD",
            opened_at="2026-09-02T09:46:00",
            closed_at="2026-09-02T15:01:00",
            currency="CAD",
        ),
    ]
    view = _view("mine", "day", snapshot_payload, trades)
    day = _section(view, "day")
    assert day.stats["net_pnl"] is None
    assert day.stats["pnl_note"], "resolve_pnl_key refused without saying why"
    assert day.stats["pnl_note"] in day.verdict_line, day.verdict_line
    assert "CAD" in day.verdict_line and "USD" in day.verdict_line
    assert "net unmeasured" not in day.verdict_line, day.verdict_line


def test_the_untagged_bucket_never_appears_under_the_confirmed_tag_header(
    snapshot_payload, journal_trades
):
    """Advisory 3. "untagged" is not a setup the trader named.

    Coverage is the sentence's business (`confirmed / total`); a row called
    `untagged` in a table headed "Confirmed tag" is a category error.
    """
    view = _view("mine", "day", snapshot_payload, journal_trades)
    day = _section(view, "day")
    labels = [str(row.values.get("label") or "") for row in day.rows]
    assert "untagged" not in labels, labels
    assert labels == ["avwap-reclaim"], labels
    # The coverage is still stated, just not as a row.
    assert "1" in day.sentence and "provisional" in day.sentence


@pytest.mark.qt
def test_a_study_row_is_muted_the_way_an_ineligible_one_is(monkeypatch, snapshot_payload):
    """Advisory 4. Only the Population column told a study from a result.

    A study row drawn in the live rows' own weight reads as a result at a
    glance, and a glance is what a shortlist is for.
    """
    from ui.models.tracker_table_model import ROW_ROLE

    with _results_panel(monkeypatch, snapshot_payload, ("bot", "swing", "recent")) as panel:
        model = panel.shortlist.model()
        rows = [model.index(i, 0).data(ROW_ROLE) for i in range(model.rowCount())]
        studies = [row for row in rows if str(row.get("namespace")) == "study"]
        assert studies, "no study row reached the shortlist"
        for row in studies:
            assert row.get("_muted_row") is True, (
                "a study row is drawn in the live rows' own weight"
            )


@pytest.mark.qt
def test_the_section_text_is_one_short_verdict_line_with_the_rest_in_a_tooltip(
    monkeypatch, snapshot_payload
):
    """Advisory 6. A 1,922-character label ran the whole width of the desk.

    One line per kind - the machine's own state and its own reason - and the
    full leader line, the policy line and the population sentence in the
    tooltip, where a dozen clauses belong.
    """
    with _results_panel(monkeypatch, snapshot_payload, ("bot", "swing", "recent")) as panel:
        text = panel.section_label.text()
        tooltip = panel.section_label.toolTip()
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) == 2, lines
        assert len(text) < 900, f"the section text is {len(text)} characters long"
        assert "namespace live" not in text, (
            "the leader's whole policy line is still printed on the page"
        )
        assert "namespace live" in tooltip, "the tooltip lost the full leader line"
        assert "no_clear_leader" in text, text


@pytest.mark.qt
def test_every_running_text_label_is_capped_at_a_readable_measure(
    monkeypatch, snapshot_payload
):
    """Advisory 6. G3's rule, one page later: 45-100 characters of its font.

    The pane keeps its width; the TEXT is capped and sits at the left of it.
    """
    from PySide6.QtCore import Qt

    from ui import theme

    module = _panel_module()
    assert module.READER_MEASURE_CHARS == 100

    with _results_panel(monkeypatch, snapshot_payload, ("bot", "swing", "recent")) as panel:
        for name in ("freshness_label", "section_label", "status_label"):
            label = getattr(panel, name)
            cap = label.maximumWidth()
            assert 0 < cap <= theme.px(module.READER_MEASURE_MAX_PX), (
                f"{name} is uncapped at {cap} px"
            )
            assert label.alignment() & Qt.AlignmentFlag.AlignLeft, name
            assert label.wordWrap(), name


def test_the_page_and_the_market_journal_reader_share_one_measure():
    """Advisory 6. Two constants that must agree, asserted rather than hoped."""
    module = _panel_module()
    from ui.panels import market_journal_panel

    assert module.READER_MEASURE_CHARS == market_journal_panel.READER_MEASURE_CHARS
    assert module.READER_MEASURE_MAX_PX == market_journal_panel.READER_MEASURE_MAX_PX
