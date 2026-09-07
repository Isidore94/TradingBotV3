"""Packet G4b.3 - the Setup Tracker's detail pane clears when its context changes.

The same defect packet G4 fixed on the Day-trade Tracker, on the panel that was
being rewritten at the time. `SetupDetailView.show_setup` / `show_research_row`
set HTML and `setVisible(True)`; nothing on `setup_tracker_panel.py` ever takes
the pane back down. `self.tabs` (fourteen tabs) has no `currentChanged` handler,
and `refresh()` replaces every model's rows with `set_rows` without touching the
pane. So the last row the trader clicked stays on screen through every tab
switch and every re-read, beside a table that no longer contains it - and on
this page the pane carries STOP AND TARGET PRICES, so a stale one is a price
plan read against the wrong symbol.

Every test drives the REAL path:

* the table's own `clicked` signal into the connection the panel made in
  `__init__` (`current_table.clicked` -> `_on_pick_clicked`, and the eight
  `explained_tables` lambdas -> `_on_research_row_clicked`);
* `QTabWidget.setCurrentIndex` into the real `currentChanged`;
* `panel.refresh()`, the real slot the Refresh button and the min-closed
  spinbox coalescer call, reading the real CSV-cache path
  (`_load_csv_rows_cached`) off fixture files in `tmp_path`.

The fixtures are CSVs and the module's own file CONSTANTS are patched - never an
alias of them, and never the reader function (the 2026-09-05 incident: a scratch
export patched an alias of the leaderboard path and overwrote the live CSVs).
Every export this page reads is redirected, so the panel touches nothing under
`C:\\TradingBotData`; every value is a STRING because a real row arrives through
`csv.DictReader`, and a column a row has nothing for is PRESENT AND EMPTY rather
than absent.

Visibility is asserted with `isHidden()` rather than `isVisible()`: the panel is
never `show()`n in a test, so a child of a hidden parent reports
`isVisible() == False` whatever the pane itself was told. `isHidden()` is the
widget's own explicit hide flag and is what `setVisible` moves.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6", reason="the Setup Tracker is a Qt panel")


# ---------------------------------------------------------------------------
# fixture exports - one CSV per file constant the panel reads
# ---------------------------------------------------------------------------


def _current_pick(*, tier, symbol, side, priority_score, setup_family, last_close, factors):
    """One line of `master_avwap_tier_list.csv`."""
    return {
        "tier": tier,
        "symbol": symbol,
        "side": side,
        "priority_score": priority_score,
        "setup_family": setup_family,
        "favorite_zone": "band1_to_band2",
        "current_band_zone": "band1_to_band2" if side == "LONG" else "above_band2",
        "trend_20d": "up" if side == "LONG" else "down",
        "scan_factor_match_count": str(len(factors.split(",")) if factors else 0),
        # PRESENT AND EMPTY on the row that matched nothing, which is what the
        # export writes - not an absent key.
        "scan_factor_matches": factors,
        "last_close": last_close,
    }


def current_picks(*, aapl_tier: str = "A") -> list[dict]:
    """Ranked by `_rank_current_picks` to NVDA (S), AAPL (A), MSFT (B)."""
    return [
        _current_pick(
            tier=aapl_tier,
            symbol="AAPL",
            side="LONG",
            priority_score="82.50",
            setup_family="avwape_to_1stdev",
            last_close="191.25",
            factors="rvol,gap,trend",
        ),
        _current_pick(
            tier="B",
            symbol="MSFT",
            side="SHORT",
            priority_score="61.00",
            setup_family="htf_ema15_rejection",
            last_close="402.10",
            factors="",
        ),
        _current_pick(
            tier="S",
            symbol="NVDA",
            side="LONG",
            priority_score="95.00",
            setup_family="avwap_band_bounce",
            last_close="121.40",
            factors="rvol,gap,trend,news",
        ),
    ]


def _setup_type(
    *,
    side,
    setup_family,
    closed,
    wins,
    losses,
    avg_closed_r,
    score_delta,
    bucket="favorite_setup",
):
    """One line of `master_avwap_setup_type_stats.csv`.

    `closed_setups` is at or above the panel's default `min_closed` of 5 so the
    row survives `_rank_setup_types`' own filter; `n_wins`/`n_losses` are the
    integer counts `_counted_win_rate_rows` reads (never a rate).
    """
    return {
        "side": side,
        "priority_bucket": bucket,
        "setup_family": setup_family,
        "favorite_zone": "band1_to_band2",
        "retest_label": "retest_confirmed",
        "closed_setups": str(closed),
        "open_setups": "4",
        "avg_closed_r": avg_closed_r,
        "avg_closed_r_edge": "0.12",
        "target_hit_rate": "0.55",
        "stop_rate": "0.30",
        "score_delta": score_delta,
        "ranking_score": "1.00",
        "sample_setups": "AAPL, MSFT",
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "0",
        "tracked_setups": str(closed + 4),
        "n_expired_unmeasured": "0",
        "tracker_saved_at": "2026-09-05T21:04:11-07:00",
        "tracker_saved_by": "scan",
        "latest_measured_session": "2026-09-04",
    }


def setup_types(*, headline_r: str = "0.40", drop_headline: bool = False) -> list[dict]:
    """The clicked row is `avwape_to_1stdev` LONG.

    `SetupDetailView.shown_identity` for it is
    `("setup_type", "LONG", "avwape_to_1stdev", "", "")`; the other two rows
    carry DIFFERENT families, so nothing collides on that tuple.
    """
    rows = [
        _setup_type(
            side="LONG",
            setup_family="sma_breakout",
            closed=20,
            wins=9,
            losses=11,
            avg_closed_r="0.10",
            score_delta="3",
        ),
        _setup_type(
            side="SHORT",
            setup_family="top_pattern",
            closed=12,
            wins=5,
            losses=7,
            avg_closed_r="-0.20",
            score_delta="1",
            bucket="general",
        ),
    ]
    if not drop_headline:
        rows.insert(
            0,
            _setup_type(
                side="LONG",
                setup_family="avwape_to_1stdev",
                closed=30,
                wins=18,
                losses=12,
                avg_closed_r=headline_r,
                score_delta="12",
            ),
        )
    return rows


RECENT_TYPES = [
    {
        "status": "RISING",
        "namespace": "live",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "setup_family": "avwape_to_1stdev",
        "win_rate_closed": "0.58",
        "closed_setups": "18",
        "tracked_setups": "22",
        "avg_closed_r": "0.35",
        "avg_closed_r_edge": "0.08",
        "target_hit_rate": "0.50",
        "stop_rate": "0.33",
        "representative_closed_r": "0.30",
        "sample_setups": "AAPL, NVDA",
        "n_wins": "10",
        "n_losses": "8",
        "n_flats": "0",
        "latest_measured_session": "2026-09-04",
    },
    {
        # An older row that never earned a status: the column is PRESENT AND
        # EMPTY, which is what the export writes.
        "status": "",
        "namespace": "live",
        "side": "SHORT",
        "priority_bucket": "general",
        "setup_family": "top_pattern",
        "win_rate_closed": "0.40",
        "closed_setups": "10",
        "tracked_setups": "14",
        "avg_closed_r": "-0.10",
        "avg_closed_r_edge": "-0.04",
        "target_hit_rate": "0.30",
        "stop_rate": "0.50",
        "representative_closed_r": "-0.12",
        "sample_setups": "TSLA",
        "n_wins": "4",
        "n_losses": "6",
        "n_flats": "0",
        "latest_measured_session": "2026-09-04",
    },
]

SHORT_TERM = [
    {
        "side": "LONG",
        "setup_family": "avwape_to_1stdev",
        "samples_2d": "40",
        "win_rate_2d": "0.55",
        "avg_r_1d": "0.20",
        "avg_r_2d": "0.45",
        "median_r_2d": "0.30",
        "avg_mfe_r_2d": "1.10",
        "avg_mae_r_2d": "-0.50",
        "recent_samples_2d": "12",
        "recent_avg_r_2d": "0.40",
        "short_term_score": "1.25",
        "sample_setups": "AAPL",
    },
    {
        "side": "SHORT",
        "setup_family": "top_pattern",
        "samples_2d": "8",
        "win_rate_2d": "0.38",
        "avg_r_1d": "-0.05",
        "avg_r_2d": "0.05",
        "median_r_2d": "0.00",
        "avg_mfe_r_2d": "0.60",
        "avg_mae_r_2d": "-0.70",
        "recent_samples_2d": "3",
        "recent_avg_r_2d": "-0.10",
        "short_term_score": "0.20",
        "sample_setups": "TSLA",
    },
]

PLAYBOOKS = [
    {
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "setup_family": "avwape_to_1stdev",
        "favorite_zone": "band1_to_band2",
        "stop_reference_label": "AVWAPE",
        "profit_take_summary": "50% at band1, runner to band2",
        "closed_setups": "26",
        "open_setups": "3",
        "robust_closed_r": "0.42",
        "robust_closed_r_edge": "0.14",
        "win_rate_closed": "0.60",
        "target_hit_rate": "0.55",
        "ranking_score": "2.10",
        "sample_setups": "AAPL, NVDA",
    },
    {
        "side": "SHORT",
        "priority_bucket": "general",
        "setup_family": "top_pattern",
        "favorite_zone": "below_band1",
        "stop_reference_label": "LOWER_1",
        "profit_take_summary": "full at band2",
        "closed_setups": "9",
        "open_setups": "1",
        "robust_closed_r": "-0.05",
        "robust_closed_r_edge": "-0.02",
        "win_rate_closed": "0.44",
        "target_hit_rate": "0.33",
        "ranking_score": "0.40",
        "sample_setups": "TSLA",
    },
]

SCAN_FACTORS = [
    {
        "horizon_sessions": "5",
        "side": "LONG",
        "factor_label": "relative_volume",
        "value_label": ">= 2.0",
        "observation_count": "120",
        "symbol_count": "44",
        "win_rate": "0.58",
        "avg_side_return_pct": "1.90",
        "side_return_edge_pct": "0.70",
        "success_score": "1.40",
        "sample_observations": "AAPL, NVDA",
    },
    {
        "horizon_sessions": "5",
        "side": "SHORT",
        "factor_label": "gap_pct",
        "value_label": "<= -3.0",
        "observation_count": "35",
        "symbol_count": "18",
        "win_rate": "0.46",
        "avg_side_return_pct": "0.40",
        "side_return_edge_pct": "-0.10",
        "success_score": "0.20",
        "sample_observations": "TSLA",
    },
]

TIER_PERFORMANCE = [
    {
        "horizon_sessions": "5",
        "tier": "S",
        "side": "LONG",
        "observation_count": "60",
        "symbol_count": "31",
        "win_rate": "0.62",
        "avg_side_return_pct": "2.30",
        "side_return_edge_pct": "1.10",
        "positive_scan_factor_match_rate": "0.71",
        "sample_observations": "NVDA",
    },
    {
        "horizon_sessions": "5",
        "tier": "A",
        "side": "LONG",
        "observation_count": "140",
        "symbol_count": "77",
        "win_rate": "0.54",
        "avg_side_return_pct": "1.20",
        "side_return_edge_pct": "0.40",
        "positive_scan_factor_match_rate": "0.62",
        "sample_observations": "AAPL",
    },
]

CATCH_RATES = [
    {
        "horizon_sessions": "5",
        "side": "LONG",
        "factor_opportunity_count": "310",
        "factor_winner_count": "150",
        "caught_winner_count": "45",
        "caught_winner_rate": "0.30",
        "missed_winner_count": "105",
        "sample_caught_winners": "AAPL, NVDA",
        "sample_missed_winners": "AMD, META",
    }
]


#: The last scan's anchor levels, in the shape `load_symbol_levels` returns.
#: Patched in so the detail pane's background level read opens no file and the
#: rendered plan is the same on every machine.
SYMBOL_LEVELS = {
    "AAPL": {
        "vwap": 185.00,
        "bands": {
            "UPPER_1": 190.00,
            "UPPER_2": 196.00,
            "UPPER_3": 202.00,
            "LOWER_1": 180.00,
            "LOWER_2": 174.00,
            "LOWER_3": 168.00,
        },
        "anchor_date": "2026-08-01",
        "atr20": 4.10,
        "last_close": 191.25,
        "side": "LONG",
    },
    "NVDA": {
        "vwap": 118.00,
        "bands": {
            "UPPER_1": 122.00,
            "UPPER_2": 127.00,
            "UPPER_3": 132.00,
            "LOWER_1": 114.00,
            "LOWER_2": 109.00,
            "LOWER_3": 104.00,
        },
        "anchor_date": "2026-08-14",
        "atr20": 3.20,
        "last_close": 121.40,
        "side": "LONG",
    },
    "MSFT": {
        "vwap": 410.00,
        "bands": {
            "UPPER_1": 418.00,
            "UPPER_2": 426.00,
            "UPPER_3": 434.00,
            "LOWER_1": 402.00,
            "LOWER_2": 394.00,
            "LOWER_3": 386.00,
        },
        "anchor_date": "2026-08-08",
        "atr20": 8.40,
        "last_close": 402.10,
        "side": "SHORT",
    },
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


#: Every file constant `refresh()` (and the attribute worker) reads, mapped to
#: the fixture written for it.
WRITTEN_EXPORTS = {
    "MASTER_AVWAP_TIER_LIST_FILE": "tier_list.csv",
    "SETUP_TYPE_STATS_FILE": "setup_type_stats.csv",
    "RECENT_SETUP_TYPE_STATS_FILE": "setup_type_recent_stats.csv",
    "SHORT_HORIZON_FILE": "short_horizon.csv",
    "SETUP_PLAYBOOKS_FILE": "playbooks.csv",
    "MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE": "scan_factors.csv",
    "MASTER_AVWAP_TIER_PERFORMANCE_FILE": "tier_performance.csv",
    "MASTER_AVWAP_TIER_CATCH_RATE_FILE": "catch_rates.csv",
}
#: Redirected and deliberately NOT written: the four shadow exports and the
#: 19.7 MB attribute leaderboard. An absent export is the real state of a fresh
#: machine, each of those tabs renders its own empty-state sentence, and the
#: golden pins the empty table rather than pretending the file is there.
ABSENT_EXPORTS = {
    "BAND_VARIANT_STATS_FILE": "band_variant.csv",
    "CONTROL_DISCOVERY_STATS_FILE": "control_discovery.csv",
    "STUDY_DISCOVERY_STATS_FILE": "study_discovery.csv",
    "EXIT_FRAMEWORK_STATS_FILE": "exit_framework.csv",
    "MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE": "attribute_leaderboard.csv",
    "MASTER_AVWAP_SETUP_TRACKER_FILE": "tracker.json",
}


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class _Tracker:
    """A built `SetupTrackerPanel` plus the fixture exports it reads back."""

    def __init__(self, panel, app, module, tmp_path: Path) -> None:
        self.panel = panel
        self.app = app
        self.module = module
        self.tmp_path = tmp_path

    # -- driving the real seams -----------------------------------------
    @property
    def view(self):
        return self.panel.detail_view

    @property
    def text(self) -> str:
        return self.view.toPlainText()

    def tab_index(self, title: str) -> int:
        for index in range(self.panel.tabs.count()):
            if self.panel.tabs.tabText(index) == title:
                return index
        raise AssertionError(f"no tab titled {title!r}")

    def switch_to(self, title: str) -> None:
        """`QTabWidget.setCurrentIndex` - into the real `currentChanged`."""
        self.panel.tabs.setCurrentIndex(self.tab_index(title))
        self.app.processEvents()

    def index_of(self, table, **match):
        """The proxy index of the row whose keys all match, found by scanning.

        Never a position: every table on this page is sorted by its own key and
        a fixture that moved would silently click a different row.
        """
        from ui.models.tracker_table_model import ROW_ROLE

        proxy = table.model()
        for position in range(proxy.rowCount()):
            index = proxy.index(position, 0)
            row = index.data(ROW_ROLE)
            if isinstance(row, dict) and all(
                str(row.get(key)) == str(value) for key, value in match.items()
            ):
                return index
        raise AssertionError(f"no row matching {match} in {proxy.rowCount()} rows")

    def click(self, table, **match) -> None:
        """The table's own `clicked` signal - the connection `__init__` made."""
        table.clicked.emit(self.index_of(table, **match))
        self.app.processEvents()
        self.settle(lambda: not self.view._levels_loading)

    def rewrite(self, constant: str, rows: list[dict]) -> None:
        _write_csv(self.tmp_path / WRITTEN_EXPORTS[constant], rows)

    def refresh(self) -> None:
        """The real slot the Refresh button and the spinbox coalescer call.

        The export cache is keyed on `(mtime_ns, size)`; a rewrite inside the
        same test can land on the same stamp, so the panel's own public
        forget-everything helper runs first. Nothing else about the read path
        is bypassed.
        """
        from tests.conftest import refresh_setup_tracker

        self.module.clear_setup_tracker_csv_cache()
        # G7 moves the twelve export reads onto a worker, so the trigger waits
        # for the render the panel announces. Before that packet lands this is
        # `panel.refresh()` and nothing else.
        refresh_setup_tracker(self.panel)
        self.app.processEvents()
        self.settle(self._attribute_read_finished)

    def _attribute_read_finished(self) -> bool:
        thread = getattr(self.panel, "_attributes_thread", None)
        return thread is None or not thread.is_alive()

    def settle(self, predicate, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                self.app.processEvents()
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return bool(predicate())


def render_all_tables(panel) -> dict[str, list[list[str]]]:
    """Every tab's table AS RENDERED: header labels, then each visible row.

    Read through the sort proxy at `Qt.DisplayRole`, which is the cell the
    trader sees - not the underlying row dict.
    """
    from PySide6.QtCore import Qt

    out: dict[str, list[list[str]]] = {}
    for name, table in _all_tables(panel).items():
        proxy = table.model()
        header = [
            str(
                proxy.headerData(
                    column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
                )
            )
            for column in range(proxy.columnCount())
        ]
        rows = [
            [
                str(proxy.index(position, column).data(Qt.ItemDataRole.DisplayRole) or "")
                for column in range(proxy.columnCount())
            ]
            for position in range(proxy.rowCount())
        ]
        out[name] = [header, *rows]
    return out


def _all_tables(panel) -> dict:
    return {
        "Current Picks": panel.current_table,
        "Short-Term 1-2d": panel.short_term_table,
        "Human Picks": panel.human_pick_table,
        "Setup Types": panel.setup_type_table,
        "Last 30 Days": panel.recent_type_table,
        "Playbooks": panel.playbook_table,
        "Scan Factors": panel.scan_factor_table,
        "Tier Performance": panel.tier_performance_table,
        "Catch Rate": panel.catch_rate_table,
        "Band Variant": panel.band_variant_table,
        "Controls": panel.control_discovery_table,
        "Studies": panel.study_discovery_table,
        "Exit frameworks": panel.exit_framework_table,
        "Attributes": panel.attribute_table,
    }


@pytest.fixture
def tracker(qapp, monkeypatch, tmp_path):
    """A real panel whose every export is a fixture CSV in `tmp_path`.

    The module's own file CONSTANTS are replaced - the names `refresh()` and
    `_attributes_worker` look up - so no alias can leave a live path in play.
    """
    from ui.panels import setup_tracker_panel as panel_module
    from ui.widgets import setup_detail_view as detail_module

    panel_module.clear_setup_tracker_csv_cache()
    for constant, filename in {**WRITTEN_EXPORTS, **ABSENT_EXPORTS}.items():
        assert hasattr(panel_module, constant), constant
        monkeypatch.setattr(panel_module, constant, tmp_path / filename)
    # The Human Picks tab joins a store this packet does not exercise; an empty
    # read is the honest fixture and keeps the golden off any shared temp dir.
    monkeypatch.setattr(
        panel_module, "load_human_focus_performance_rows", lambda *a, **k: []
    )
    # The detail pane's level read runs on a background thread against the last
    # scan's ai_state. Pinned so the pane's plan is deterministic and no file is
    # opened.
    monkeypatch.setattr(
        detail_module, "load_symbol_levels", lambda *a, **k: dict(SYMBOL_LEVELS)
    )

    _write_csv(tmp_path / WRITTEN_EXPORTS["MASTER_AVWAP_TIER_LIST_FILE"], current_picks())
    _write_csv(tmp_path / WRITTEN_EXPORTS["SETUP_TYPE_STATS_FILE"], setup_types())
    _write_csv(tmp_path / WRITTEN_EXPORTS["RECENT_SETUP_TYPE_STATS_FILE"], RECENT_TYPES)
    _write_csv(tmp_path / WRITTEN_EXPORTS["SHORT_HORIZON_FILE"], SHORT_TERM)
    _write_csv(tmp_path / WRITTEN_EXPORTS["SETUP_PLAYBOOKS_FILE"], PLAYBOOKS)
    _write_csv(
        tmp_path / WRITTEN_EXPORTS["MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE"],
        SCAN_FACTORS,
    )
    _write_csv(
        tmp_path / WRITTEN_EXPORTS["MASTER_AVWAP_TIER_PERFORMANCE_FILE"], TIER_PERFORMANCE
    )
    _write_csv(tmp_path / WRITTEN_EXPORTS["MASTER_AVWAP_TIER_CATCH_RATE_FILE"], CATCH_RATES)

    made = panel_module.SetupTrackerPanel()
    harness = _Tracker(made, qapp, panel_module, tmp_path)
    # G7.1: the constructor reads nothing and the first SHOW is what loads the
    # page, so the read this fixture waited on is asked for explicitly. The
    # harness's own refresh IS that trigger and settles on the same read.
    harness.refresh()
    assert harness.settle(harness._attribute_read_finished), (
        "the first read never finished"
    )
    try:
        yield harness
    finally:
        made.shutdown()
        made.deleteLater()
        qapp.processEvents()
        panel_module.clear_setup_tracker_csv_cache()


# ---------------------------------------------------------------------------
# 1. a tab change is a context change
# ---------------------------------------------------------------------------


def test_switching_tabs_takes_down_the_setup_detail_pane(tracker):
    """G4b.1. The trader clicks AAPL on Current Picks and moves to Playbooks.

    What stays on screen is a stop price, a 1R and two targets for AAPL, beside
    a table of stop/exit COMBINATIONS that contains no symbol at all. The pane
    has to go down, and it has to forget what it was showing - a pane that is
    hidden but still claims an identity is a pane the next refresh can bring
    back.
    """
    tracker.click(tracker.panel.current_table, symbol="AAPL")

    assert not tracker.view.isHidden(), "the click should have opened the pane"
    assert tracker.view.shown_identity == (
        "setup",
        "LONG",
        "avwape_to_1stdev",
        "AAPL",
        "",
    ), tracker.view.shown_identity
    assert "AAPL" in tracker.text

    tracker.switch_to("Playbooks")

    assert tracker.view.isHidden(), (
        "the Playbooks tab is showing and the pane still prices AAPL"
    )
    assert tracker.view.shown_identity is None, (
        "a hidden pane still claims to be showing a setup"
    )
    assert tracker.text.strip() == "", "a cleared pane kept its text to come back with"


# ---------------------------------------------------------------------------
# 2. a refresh re-shows the open row from the NEW numbers, or takes it down
# ---------------------------------------------------------------------------


def test_a_refresh_repaints_the_open_setup_type_row_with_the_new_mean_r(tracker):
    """G4b.2, the found branch. The nightly re-grade moved this family's mean
    closed R from +0.40 to +1.25.

    The row is still in the table the trader is looking at, so the pane stays
    open - and it must read +1.25R. Leaving +0.40R beside a table that now says
    +1.25 is the same defect wearing a number instead of a name, and re-showing
    the CACHED row dict reproduces it exactly, which is why the assertion is the
    NUMBER and not merely "the pane is still open".
    """
    tracker.switch_to("Setup Types")
    tracker.click(tracker.panel.setup_type_table, setup_family="avwape_to_1stdev")

    assert tracker.view.shown_identity == (
        "setup_type",
        "LONG",
        "avwape_to_1stdev",
        "",
        "",
    ), tracker.view.shown_identity
    assert "+0.40R" in tracker.text, tracker.text[:400]

    tracker.rewrite("SETUP_TYPE_STATS_FILE", setup_types(headline_r="1.25"))
    tracker.refresh()

    assert not tracker.view.isHidden(), (
        "the row is still in the current table, so the pane stays open"
    )
    assert "+1.25R" in tracker.text, "the pane is showing the pre-refresh number"
    assert "+0.40R" not in tracker.text


def test_a_refresh_that_drops_the_open_setup_type_row_takes_the_pane_down(tracker):
    """G4b.2, the not-found branch.

    The family is gone from the export; its explanation must go too. The table
    is NOT empty afterwards - two families survive - so a pane that stays up is
    describing a row the trader can no longer see, not merely lagging an empty
    table.
    """
    tracker.switch_to("Setup Types")
    tracker.click(tracker.panel.setup_type_table, setup_family="avwape_to_1stdev")
    assert not tracker.view.isHidden()

    tracker.rewrite("SETUP_TYPE_STATS_FILE", setup_types(drop_headline=True))
    tracker.refresh()

    assert tracker.panel.setup_type_model.rowCount() == 2, (
        "the fixture must leave the table populated, or this proves nothing"
    )
    with pytest.raises(AssertionError):
        tracker.index_of(tracker.panel.setup_type_table, setup_family="avwape_to_1stdev")

    assert tracker.view.isHidden(), (
        "the pane is explaining a family the refreshed table no longer holds"
    )
    assert tracker.view.shown_identity is None
    assert tracker.text.strip() == ""


# ---------------------------------------------------------------------------
# 3. a refresh re-shows a current pick from the NEW row dict
# ---------------------------------------------------------------------------


def test_a_refresh_repaints_the_open_current_pick_from_the_new_row(tracker):
    """G4b.2 on the Current Picks tab, where the pane carries PRICES.

    The scan re-tiered AAPL from A to S. Tier is deliberately NOT part of
    `shown_identity` - the identity is what makes two rows the same row, and a
    re-tier does not make AAPL LONG a different setup - so the pane must stay
    open AND repaint from the new row dict. Asserting the tier is what separates
    "re-shown from the new row" from "left alone because it still matches".
    """
    tracker.click(tracker.panel.current_table, symbol="AAPL")
    assert "A AAPL LONG" in tracker.text, tracker.text[:400]

    tracker.rewrite("MASTER_AVWAP_TIER_LIST_FILE", current_picks(aapl_tier="S"))
    tracker.refresh()

    assert not tracker.view.isHidden(), "AAPL is still in the table; the pane stays open"
    assert tracker.view.shown_identity == (
        "setup",
        "LONG",
        "avwape_to_1stdev",
        "AAPL",
        "",
    )
    assert "S AAPL LONG" in tracker.text, "the pane is showing the pre-refresh tier"
    assert "A AAPL LONG" not in tracker.text


# ---------------------------------------------------------------------------
# 4. a refresh never RESURRECTS a pane the trader is not looking at
# ---------------------------------------------------------------------------


def test_a_refresh_leaves_a_hidden_pane_hidden_even_when_its_row_is_present(tracker):
    """G4b.2's guard rail, reached the way the trader reaches it.

    Click a Setup Types row, move to Playbooks, come back to Setup Types. The
    pane is down and the identity is forgotten - and the row it used to show is
    STILL IN THE CURRENT TABLE, unchanged. A refresh that scans for a match and
    re-shows it without first asking whether the pane is visible would pop an
    explanation open under a trader who closed it, so the discriminating fact
    here is the PRESENCE of the row, not its absence.
    """
    tracker.switch_to("Setup Types")
    tracker.click(tracker.panel.setup_type_table, setup_family="avwape_to_1stdev")
    assert not tracker.view.isHidden()

    tracker.switch_to("Playbooks")
    tracker.switch_to("Setup Types")
    assert tracker.view.isHidden(), "the pane survived two tab changes"
    assert tracker.view.shown_identity is None

    tracker.refresh()

    assert (
        tracker.index_of(
            tracker.panel.setup_type_table, setup_family="avwape_to_1stdev"
        )
        is not None
    ), "the row must still be there, or this proves nothing"
    assert tracker.view.isHidden(), "a refresh reopened a pane the trader had closed"
    assert tracker.view.shown_identity is None
    assert tracker.text.strip() == ""


# ---------------------------------------------------------------------------
# 5. the golden: no number, no sort, no cell moves
# ---------------------------------------------------------------------------

#: Every tab's table as rendered on the fixture above, captured from the code at
#: `7e018c99` - BEFORE this packet's change. Header row first, then each visible
#: row through the sort proxy at `Qt.DisplayRole`.
GOLDEN_TABLES: dict[str, list[list[str]]] = {
    'Current Picks': [
        [
            'Tier',
            'Symbol',
            'Side',
            'Score',
            'Setup Family',
            'Favorite Zone',
            'Current Zone',
            '20D Trend',
            'Factor Hits',
            'Positive Factors',
        ],
        [
            'S',
            'NVDA',
            'LONG',
            '95.00',
            'avwap_band_bounce',
            'band1_to_band2',
            'band1_to_band2',
            'up',
            '4',
            'rvol,gap,trend,news',
        ],
        [
            'A',
            'AAPL',
            'LONG',
            '82.50',
            'avwape_to_1stdev',
            'band1_to_band2',
            'band1_to_band2',
            'up',
            '3',
            'rvol,gap,trend',
        ],
        [
            'B',
            'MSFT',
            'SHORT',
            '61.00',
            'htf_ema15_rejection',
            'band1_to_band2',
            'above_band2',
            'down',
            '0',
            '',
        ],
    ],
    'Short-Term 1-2d': [
        [
            'Side',
            'Setup Family',
            'Samples',
            'Win @2d',
            'R @1d',
            'R @2d',
            'Med R @2d',
            'MFE 2d',
            'MAE 2d',
            'N 30d',
            'R @2d 30d',
            'Rank',
            'Recent Samples',
        ],
        [
            'LONG',
            'avwape_to_1stdev',
            '40',
            '55.0%',
            '+0.20',
            '+0.45',
            '+0.30',
            '+1.10',
            '-0.50',
            '12',
            '+0.40',
            '+1.25',
            'AAPL',
        ],
        [
            'SHORT',
            'top_pattern',
            '8',
            '38.0%',
            '-0.05',
            '+0.05',
            '+0.00',
            '+0.60',
            '-0.70',
            '3',
            '-0.10',
            '+0.20',
            'TSLA',
        ],
    ],
    'Human Picks': [
        [
            'Cohort',
            'Side',
            'Horizon',
            'Human N',
            'Human Win',
            'Human Avg %',
            'Human PF',
            'Bot S/A N',
            'Bot S/A Win',
            'Bot S/A Avg %',
            'Delta %',
        ],
    ],
    'Setup Types': [
        [
            'Side',
            'Bucket',
            'Setup Family',
            'Zone',
            'Retest',
            'Win %',
            'Closed',
            'Open',
            'Closed R',
            'R Edge',
            'Target Hit',
            'Stop',
            'Score Delta',
            'Recent Samples',
        ],
        [
            'LONG',
            'favorite_setup',
            'avwape_to_1stdev',
            'band1_to_band2',
            'retest_confirmed',
            '60% (>=42%, n=30)',
            '30',
            '4',
            '+0.40',
            '+0.12',
            '55.0%',
            '30.0%',
            '+12.00',
            'AAPL, MSFT',
        ],
        [
            'LONG',
            'favorite_setup',
            'sma_breakout',
            'band1_to_band2',
            'retest_confirmed',
            '45% (>=26%, n=20)',
            '20',
            '4',
            '+0.10',
            '+0.12',
            '55.0%',
            '30.0%',
            '+3.00',
            'AAPL, MSFT',
        ],
        [
            'SHORT',
            'general',
            'top_pattern',
            'band1_to_band2',
            'retest_confirmed',
            '42% (>=19%, n=12)',
            '12',
            '4',
            '-0.20',
            '+0.12',
            '55.0%',
            '30.0%',
            '+1.00',
            'AAPL, MSFT',
        ],
    ],
    'Last 30 Days': [
        [
            'Status',
            'Source',
            'Side',
            'Bucket',
            'Setup Family',
            'Win % (unweighted)',
            'Win % (recency-weighted)',
            'Closed 30d',
            'Tracked 30d',
            'Closed R',
            'R Edge',
            'Target Hit',
            'Stop',
            'Repr R',
            'Recent Samples',
        ],
        [
            'RISING',
            'live',
            'LONG',
            'favorite_setup',
            'avwape_to_1stdev',
            '56% (>=34%, n=18)',
            '58.0%',
            '18',
            '22',
            '+0.35',
            '+0.08',
            '50.0%',
            '33.0%',
            '+0.30',
            'AAPL, NVDA',
        ],
        [
            '',
            'live',
            'SHORT',
            'general',
            'top_pattern',
            '40% (>=17%, n=10)',
            '40.0%',
            '10',
            '14',
            '-0.10',
            '-0.04',
            '30.0%',
            '50.0%',
            '-0.12',
            'TSLA',
        ],
    ],
    'Playbooks': [
        [
            'Side',
            'Bucket',
            'Setup Family',
            'Zone',
            'Stop',
            'Exit Plan',
            'Closed',
            'Open',
            'Robust R',
            'R Edge',
            'Win Rate',
            'Target Hit',
            'Rank',
            'Recent Samples',
        ],
        [
            'LONG',
            'favorite_setup',
            'avwape_to_1stdev',
            'band1_to_band2',
            'AVWAPE',
            '50% at band1, runner to band2',
            '26',
            '3',
            '+0.42',
            '+0.14',
            '60.0%',
            '55.0%',
            '2.10',
            'AAPL, NVDA',
        ],
        [
            'SHORT',
            'general',
            'top_pattern',
            'below_band1',
            'LOWER_1',
            'full at band2',
            '9',
            '1',
            '-0.05',
            '-0.02',
            '44.0%',
            '33.0%',
            '0.40',
            'TSLA',
        ],
    ],
    'Scan Factors': [
        [
            'Horizon',
            'Side',
            'Factor',
            'Value',
            'Obs',
            'Symbols',
            'Win',
            'Avg Side %',
            'Edge %',
            'Success',
            'Samples',
        ],
        [
            '5',
            'LONG',
            'relative_volume',
            '>= 2.0',
            '120',
            '44',
            '58.0%',
            '+1.90%',
            '+0.70%',
            '+1.40',
            'AAPL, NVDA',
        ],
        [
            '5',
            'SHORT',
            'gap_pct',
            '<= -3.0',
            '35',
            '18',
            '46.0%',
            '+0.40%',
            '-0.10%',
            '+0.20',
            'TSLA',
        ],
    ],
    'Tier Performance': [
        [
            'Horizon',
            'Tier',
            'Side',
            'Obs',
            'Symbols',
            'Win',
            'Avg Side %',
            'Edge %',
            'Factor Hit Rate',
            'Samples',
        ],
        [
            '5',
            'S',
            'LONG',
            '60',
            '31',
            '62.0%',
            '+2.30%',
            '+1.10%',
            '71.0%',
            'NVDA',
        ],
        [
            '5',
            'A',
            'LONG',
            '140',
            '77',
            '54.0%',
            '+1.20%',
            '+0.40%',
            '62.0%',
            'AAPL',
        ],
    ],
    'Catch Rate': [
        [
            'Horizon',
            'Side',
            'Factor Opps',
            'Factor Winners',
            'Caught Winners',
            'Caught Winners',
            'Missed Winners',
            'Caught Samples',
            'Missed Samples',
        ],
        [
            '5',
            'LONG',
            '310',
            '150',
            '45',
            '30.0%',
            '105',
            'AAPL, NVDA',
            'AMD, META',
        ],
    ],
    'Band Variant': [
        [
            'Family',
            'Side',
            'Bucket',
            'n',
            'n Variant',
            'n Unmeasured',
            'Champ R',
            'Variant R',
            'Champ Stop%',
            'Variant Stop%',
            'Champ Target%',
            'Variant Target%',
            'Champ Stop ATR',
            'Variant Stop ATR',
            'Exit Template',
        ],
    ],
    'Controls': [
        [
            'Window',
            'Kind',
            'Cohort',
            'Side',
            'Family',
            'Win %',
            'Win % (low)',
            'n',
            'Wins',
            'Losses',
            'Avg R',
            'Expired',
            'Flag',
        ],
    ],
    'Studies': [
        [
            'Window',
            'Kind',
            'Cohort',
            'Side',
            'Family',
            'Win %',
            'Win % (low)',
            'n',
            'Wins',
            'Losses',
            'Avg R',
            'Expired',
            'Flag',
        ],
    ],
    'Exit frameworks': [
        [
            'Framework',
            'Experimental',
            'Exit Template',
            'Side',
            'Bucket',
            'Win %',
            'Win % (low)',
            'n',
            'n Closed',
            'Avg R',
            'Stop%',
            'Target%',
            'Expired',
            'Filtered',
        ],
    ],
    'Attributes': [
        [
            'Attribute',
            'Value',
            'Side',
            'Bucket',
            'n',
            'n Closed',
            'Floor',
            'Closed R',
            'Closed R Edge',
            'Target% Edge',
            'Stop% Edge',
            'Examples',
        ],
    ],
}


def test_no_tab_table_renders_a_different_cell_after_the_pane_learns_to_clear(tracker):
    """GREEN BY DESIGN, and the only test in this file that is.

    G4b is a layout-lane packet: it may take a pane down and put it back up,
    and it may not move a number, a sort, a column or a read. This golden is
    pinned from the BASE commit's own render, so it is green before the fix and
    must stay green after it. Its job starts the moment the builder touches
    `refresh()` - a re-show inserted before the `set_rows` calls, a row dict
    mutated in place while scanning for the match, or a `_rank_*` call moved,
    turns it red.

    The second half repeats the render after a full click / tab-switch /
    refresh cycle: showing and clearing the pane must leave all fourteen tables
    identical too.
    """
    before = render_all_tables(tracker.panel)
    assert before == GOLDEN_TABLES, "a tab's rendered table moved off the pinned golden"

    tracker.click(tracker.panel.current_table, symbol="AAPL")
    tracker.switch_to("Setup Types")
    tracker.click(tracker.panel.setup_type_table, setup_family="avwape_to_1stdev")
    tracker.switch_to("Playbooks")
    tracker.refresh()

    after = render_all_tables(tracker.panel)
    assert after == GOLDEN_TABLES, "the pane's own lifecycle moved a table cell"


# ---------------------------------------------------------------------------
# 6. BUILDER-ADDED: the widened identity, which the five above cannot see
# ---------------------------------------------------------------------------


def _zoned_setup_types(*, band1_r: str, below_r: str) -> list[dict]:
    """Two rows of ONE (side, family) that differ only by zone.

    `SetupDetailView.shown_identity` for both is
    `("setup_type", "LONG", "avwape_to_1stdev", "", "")` - the Setup Types
    export has no `dimension` and no `symbol`, so the pane's own identity
    cannot tell them apart. The `band1_to_band2` row wins more, so it sorts
    FIRST by the Wilson bound and is what an identity-only rescan would find.
    """
    band1 = _setup_type(
        side="LONG",
        setup_family="avwape_to_1stdev",
        closed=30,
        wins=18,
        losses=12,
        avg_closed_r=band1_r,
        score_delta="12",
    )
    below = _setup_type(
        side="LONG",
        setup_family="avwape_to_1stdev",
        closed=30,
        wins=6,
        losses=24,
        avg_closed_r=below_r,
        score_delta="2",
    )
    below["favorite_zone"] = "below_band1"
    return [band1, below, *setup_types(drop_headline=True)]


def test_a_refresh_repaints_the_zone_the_trader_clicked_not_its_twin(tracker):
    """G4b.2's widened identity (builder, honouring the tester's finding).

    Two Setup Types rows share the pane's whole identity and differ only in
    `favorite_zone`. The trader clicked the LOSING one; a re-show keyed on the
    identity alone would find the winning twin first and quietly swap a -0.75R
    explanation for a +0.40R one under an unchanged heading - a worse defect
    than the stale pane this packet exists to fix, because nothing on screen
    says the row changed.
    """
    tracker.rewrite(
        "SETUP_TYPE_STATS_FILE", _zoned_setup_types(band1_r="0.40", below_r="-0.75")
    )
    tracker.refresh()

    tracker.switch_to("Setup Types")
    tracker.click(
        tracker.panel.setup_type_table,
        setup_family="avwape_to_1stdev",
        favorite_zone="below_band1",
    )
    assert tracker.view.shown_identity == (
        "setup_type",
        "LONG",
        "avwape_to_1stdev",
        "",
        "",
    ), "the two rows must share the pane's identity, or this proves nothing"
    assert "-0.75R" in tracker.text, tracker.text[:400]

    tracker.rewrite(
        "SETUP_TYPE_STATS_FILE", _zoned_setup_types(band1_r="0.40", below_r="-1.60")
    )
    tracker.refresh()

    assert not tracker.view.isHidden(), "the clicked row is still in the table"
    assert "-1.60R" in tracker.text, "the pane is showing the pre-refresh number"
    assert "+0.40R" not in tracker.text, "the pane swapped to the better-scoring twin"
