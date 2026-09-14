"""Packet D1C-B, item 2 - Research > Setup Tracker gains a `My claims` tab.

The trader's surface for *"let me compare FAV, HC and My liked trades, and
compare the setup types I claimed"*. Three blocks, built the way every other
population tab on this page is built (`_make_explained_tab`, win rate first
with its `n` and the ONE Wilson bound, sorted by the BOUND), and read on the
panel's existing worker - **the CSV/JSONL loads never run on the Qt thread**
(CLAUDE.md: nothing expensive belongs on the Qt thread).

Tester's ruling on the packet's "three blocks in one table (or three small
tables - say which)": **two tables and a status sentence.**

* ``panel.claim_population_table`` / ``claim_population_model`` /
  ``claim_population_rows`` over ``CLAIM_POPULATION_COLUMNS``
* ``panel.claim_setup_table`` / ``claim_setup_model`` / ``claim_setup_rows``
  over ``CLAIM_SETUP_COLUMNS``
* the leader line and the quick-like footnote as label text on the tab.

The packet's "the reader returns both so the tab shows two column groups" is
rendered as the LATELY group in the packet's own column names plus an `_all`
group beside it, so one row carries both clocks' windows and neither is hidden.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tests import d1c_claim_grading_fixtures as fx  # noqa: E402

TAB_TITLE = "My claims"


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _sessions(count: int) -> list[str]:
    """The last `count` exchange sessions ending today, oldest first.

    The panel's own clock is today, so the fixture's dates have to be real
    sessions inside `evidence_stats.lately_window(today)` - never a count of
    calendar days, which shortens exactly when the market was closed.
    """
    import market_calendar

    cursor = date.today()
    if not market_calendar.is_session(cursor):
        cursor = market_calendar.previous_session(cursor)
    out = [cursor.isoformat()]
    for _ in range(count - 1):
        cursor = market_calendar.previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _panel_fixture(root: Path) -> dict[str, Path]:
    """The canonical fixture, re-dated onto real sessions inside `lately`.

    lately: liked n=3 wins=2, FAV n=4 wins=3, Near n=2 wins=1, HC 0 rows,
    pending 1, unmeasured 1, duplicates 1, quick likes 1, also_fav 1.
    all:    liked n=4, FAV n=5.
    """
    s = _sessions(6)  # s[0] oldest .. s[5] the last session
    picks = [
        fx.like_pick(s[2], "AAA", "LONG", fx.BREAKOUT),
        fx.like_pick(s[2], "BBB", "LONG", fx.BREAKOUT),
        fx.like_pick(s[3], "CCC", "LONG", fx.BREAKOUT),
        fx.like_pick(s[3], "DDD", "SHORT", fx.ONE_STDEV),
        fx.like_pick(s[4], "EEE", "LONG", fx.ONE_STDEV),
        fx.like_pick(s[4], "FFF", "LONG", "like_unclaimed"),
        fx.like_pick(s[4], "GGG", "LONG", fx.BREAKOUT, like_mode="quick"),
        fx.like_pick(s[2], "AAA", "LONG", fx.BREAKOUT),
        fx.like_pick(fx.LONG_AGO, "OLD", "LONG", fx.BREAKOUT),
    ]
    matured = "1,3,5,10"
    outcomes = [
        fx.like_outcome(s[2], "AAA", "LONG", fx.BREAKOUT, h5_return="0.052631", matured_horizons=matured),
        fx.like_outcome(s[2], "BBB", "LONG", fx.BREAKOUT, h5_return="-0.028885", matured_horizons=matured),
        fx.like_outcome(s[3], "CCC", "LONG", fx.BREAKOUT, h5_return="0.012000", matured_horizons=matured),
        fx.like_outcome(s[3], "DDD", "SHORT", fx.ONE_STDEV, h5_return="", matured_horizons="1,3"),
        fx.like_outcome(s[4], "FFF", "LONG", "like_unclaimed", h5_return="0.09", matured_horizons=matured),
        fx.like_outcome(s[4], "GGG", "LONG", fx.BREAKOUT, h5_return="0.08", matured_horizons=matured),
        fx.like_outcome(fx.LONG_AGO, "OLD", "LONG", fx.BREAKOUT, h5_return="0.031", matured_horizons=matured),
    ]
    tiers = [
        fx.tier_row(s[2], "AAA", "LONG", "favorite_setup", win="True"),
        fx.tier_row(s[0], "FV1", "LONG", "favorite_setup", win="True"),
        fx.tier_row(s[1], "FV2", "SHORT", "favorite_setup", win="True"),
        fx.tier_row(s[1], "FV3", "LONG", "favorite_setup", win="False"),
        fx.tier_row(fx.LONG_AGO, "FV4", "LONG", "favorite_setup", win="True"),
        fx.tier_row(s[0], "FV5", "LONG", "favorite_setup", win="True", stale_horizon="True"),
        fx.tier_row(s[0], "FV6", "LONG", "favorite_setup", win="True", horizon_sessions="1"),
        fx.tier_row(s[3], "CCC", "LONG", "near_favorite_zone", win="True"),
        fx.tier_row(s[0], "NR1", "LONG", "near_favorite_zone", win="False"),
    ]
    claim_rows = [fx.claim_row("AAA", "LONG", "avwap_breakout", s[2])]
    return fx.write_store(
        root, picks=picks, outcomes=outcomes, tiers=tiers, claim_rows=claim_rows
    )


@pytest.fixture
def panel_module(monkeypatch, tmp_path):
    if _qt_app() is None:
        pytest.skip("PySide6 is not installed")
    import claimed_pick_evidence
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    paths = _panel_fixture(tmp_path)
    named = {
        "LIKE_COHORT_PICKS_FILE": paths["picks_path"],
        "LIKE_COHORT_OUTCOMES_FILE": paths["outcomes_path"],
        "MASTER_AVWAP_TIER_OUTCOMES_FILE": paths["tier_path"],
        "CLAIMED_PICKS_FILE": paths["claims_path"],
    }
    # Resolved at CALL time on the reader module, the way `tracker_export_files`
    # already is, so a test can point the tab at a temporary home folder.
    for name, value in named.items():
        monkeypatch.setattr(claimed_pick_evidence, name, value, raising=False)
        if hasattr(setup_tracker_panel, name):
            monkeypatch.setattr(setup_tracker_panel, name, value)
    return setup_tracker_panel


def _load(panel) -> None:
    from tests.conftest import refresh_setup_tracker

    refresh_setup_tracker(panel)


def _tab_titles(panel) -> list[str]:
    return [panel.tabs.tabText(index) for index in range(panel.tabs.count())]


def _tab_widget(panel):
    for index in range(panel.tabs.count()):
        if panel.tabs.tabText(index) == TAB_TITLE:
            return panel.tabs.widget(index)
    raise AssertionError(f"no {TAB_TITLE!r} tab: {_tab_titles(panel)}")


def _tab_texts(panel) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [label.text() for label in _tab_widget(panel).findChildren(QLabel)]


def _keys(columns) -> list[str]:
    return [key for key, _header in columns]


def _by_population(panel) -> dict[str, dict]:
    return {str(row["population"]): row for row in panel.claim_population_rows}


# ---------------------------------------------------------------------------
# The tab exists, and it is built the way this page builds a population tab
# ---------------------------------------------------------------------------


def test_the_my_claims_tab_is_on_the_setup_tracker_page(panel_module):
    panel = panel_module.SetupTrackerPanel()
    try:
        assert TAB_TITLE in _tab_titles(panel)
    finally:
        panel.deleteLater()


def test_win_rate_leads_the_statistics_in_both_blocks(panel_module):
    """Decision 0016: win rate is the swing headline, with `n` and the ONE
    Wilson bound beside it - never a bound-first or an n-first table."""
    population = _keys(panel_module.CLAIM_POPULATION_COLUMNS)
    setup = _keys(panel_module.CLAIM_SETUP_COLUMNS)

    for keys in (population, setup):
        assert "win_rate" in keys
        for later in ("bound", "n", "wins", "pending", "unmeasured"):
            assert keys.index("win_rate") < keys.index(later), (keys, later)
    assert "also_fav" in population
    assert "note" in population
    assert "population" in population
    assert "source" in setup
    # Two column groups: the lately window in the packet's names, `all` beside it.
    assert "n_all" in population and "win_rate_all" in population


# ---------------------------------------------------------------------------
# The numbers the trader reads
# ---------------------------------------------------------------------------


def test_the_populations_block_carries_four_rows_with_their_real_counts(panel_module):
    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        rows = _by_population(panel)
        assert set(rows) == {"My liked trades", "FAV", "HC", "Near"}

        liked = rows["My liked trades"]
        assert int(liked["n"]) == 3
        assert int(liked["wins"]) == 2
        assert float(liked["win_rate"]) == pytest.approx(2 / 3)
        assert float(liked["bound"]) == pytest.approx(0.20765960080204768)
        assert int(liked["pending"]) == 1
        assert int(liked["unmeasured"]) == 1
        assert int(liked["also_fav"]) == 1
        assert int(liked["n_all"]) == 4

        fav = rows["FAV"]
        assert int(fav["n"]) == 4
        assert int(fav["wins"]) == 3
        assert float(fav["win_rate"]) == pytest.approx(0.75)
        assert int(fav["n_all"]) == 5

        near = rows["Near"]
        assert int(near["n"]) == 2
        assert int(near["wins"]) == 1
    finally:
        panel.deleteLater()


def test_the_hc_cell_says_unmeasured_in_words_and_shows_no_percent(panel_module):
    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        hc = _by_population(panel)["HC"]
        assert hc["note"] == (
            "unmeasured: the tracker records favorite_setup / near_favorite_zone "
            "only (0 HC rows)"
        )
        assert str(hc["win_rate"] or "") == ""
        assert str(hc["bound"] or "") == ""
        assert "%" not in str(hc["note"])
    finally:
        panel.deleteLater()


def test_no_rendered_cell_is_the_sum_of_the_two_clocks(panel_module):
    """3 liked + 4 FAV = 7. A union of two clocks is not a sample."""
    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        values = []
        for row in panel.claim_population_rows:
            for key in ("n", "wins", "pending", "unmeasured", "also_fav", "n_all"):
                text = str(row.get(key, "")).strip()
                if text.lstrip("-").isdigit():
                    values.append(int(text))
        assert 7 not in values, "a cell pooled My liked trades with FAV"
        assert 9 not in values, "a cell pooled the two `all` windows"
    finally:
        panel.deleteLater()


def test_the_by_setup_block_is_sorted_by_the_bound(panel_module):
    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        sources = [str(row["source"]) for row in panel.claim_setup_rows]
        assert sources == [fx.BREAKOUT, fx.ONE_STDEV], (
            "sorted by the raw rate or by name, not by the Wilson lower bound"
        )
        breakout = panel.claim_setup_rows[0]
        assert int(breakout["n"]) == 3
        assert int(breakout["wins"]) == 2
    finally:
        panel.deleteLater()


def test_the_tab_prints_the_leader_line_and_the_quick_like_footnote(panel_module):
    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        texts = " || ".join(_tab_texts(panel))
        assert "no setup has n >= 30 yet (best n was 3)" in texts
        assert "quick likes excluded: 1" in texts
    finally:
        panel.deleteLater()


def test_both_models_render_their_rows(panel_module):
    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        assert panel.claim_population_model.rowCount() == 4
        assert panel.claim_setup_model.rowCount() == 2
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# The explanation - what a row is, and what this tab is NOT
# ---------------------------------------------------------------------------


def test_the_explanation_names_both_clocks_the_overlap_rule_and_the_journal(panel_module):
    panel = panel_module.SetupTrackerPanel()
    try:
        prose = " ".join(_tab_texts(panel)).lower()
        assert "scan row" in prose, "the tracker's clock is not named"
        assert "session" in prose, "the like cohort's clock is not named"
        assert "overlap" in prose
        assert "journal" in prose
        assert "rank" in prose or "score" in prose
        assert "alert" in prose
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Nothing expensive on the Qt thread
# ---------------------------------------------------------------------------


def test_the_stores_are_read_on_the_worker_and_never_on_the_qt_thread(
    panel_module, monkeypatch
):
    """`load_inputs` is a four-file read. It runs where the other thirteen do."""
    import claimed_pick_evidence

    qt_thread = threading.current_thread().ident
    calls: list[int | None] = []
    original = claimed_pick_evidence.load_inputs

    def _spy(*args, **kwargs):
        calls.append(threading.current_thread().ident)
        return original(*args, **kwargs)

    monkeypatch.setattr(claimed_pick_evidence, "load_inputs", _spy)
    if hasattr(panel_module, "load_inputs"):
        monkeypatch.setattr(panel_module, "load_inputs", _spy)

    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        assert calls, "the tab never read the claim stores"
        assert qt_thread not in calls, "the claim stores were read on the Qt thread"
    finally:
        panel.deleteLater()


def test_the_tab_builds_with_the_journal_modules_unimportable(panel_module, monkeypatch):
    """Opportunity results and journal trade results are different surfaces."""
    import importlib.abc

    forbidden = {"journal_store", "preference_trade_outcomes"}

    class _Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                raise ImportError(f"{fullname} is a different surface")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in forbidden:
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Refuse(), *sys.meta_path])

    panel = panel_module.SetupTrackerPanel()
    _load(panel)
    try:
        assert int(_by_population(panel)["My liked trades"]["n"]) == 3
    finally:
        panel.deleteLater()
