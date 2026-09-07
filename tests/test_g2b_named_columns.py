"""Packet G2b - the Setup Tracker's tabs, the Desk Setups table and the AWAY
tables NAME the column that takes the slack, instead of measuring for it.

Written by the TESTER before any fix exists, on `claude/g2b-tracker-desk-away-columns`
off `main` at ``7e018c99`` (2026-09-07). Every test here is red on that commit
except the two GOLDENS, whose docstrings say so.

Why these tests exist
---------------------
`ui.widgets.data_table.apply_width_rule` has two modes. A caller that NAMES its
text column gets that column stretched; a caller that names nothing gets the
widest MEASURED text column, ties to the lowest index. `DataTable.set_width_rule`
exists to carry the names - and on ``7e018c99`` it has **zero production
callers**, so every table on these three surfaces runs the measured path.

Measured is content-dependent, and content moves. The defects the GUI review
recorded are all the same defect:

* Setup Tracker | Human Picks - `cohort` is the only text column on a row of
  ten measurements, so it wins the slack and pushes the numbers off the right;
* Setup Tracker | Controls with no export yet - an EMPTY table measures its
  HEADERS, and "Win % (low)" is the longest one, so the empty tab stretches a
  numeric column and squeezes `Family`;
* Setup Tracker | Playbooks / Catch Rate - `sample_setups` and
  `sample_caught_winners` are long free text, so they eat the width the "Exit
  Plan" and "Missed Samples" columns were built to show.

So the fix is to name them. These tests drive the REAL panels offscreen at
3456 wide, through the real `refresh()` / `set_column_profile("full")` /
`_render()`, with fixture CSVs in `tmp_path` and the module's OWN file
constants patched (never `project_paths`' - the panel binds
`SETUP_TYPE_STATS_FILE` and friends at import time, so patching the
`project_paths` attribute they were derived from changes nothing).

How a fixture is built so the test CAN fail
-------------------------------------------
Asserting "the named column stretches" only proves the naming if the MEASURED
answer would be a different column. So every populated fixture carries a DECOY:
one other text column whose value is longer than the column the packet names.
On ``7e018c99`` the decoy wins and the assertion fails; after the fix the name
wins whatever the decoy measures. The decoys are the real defect, not
scaffolding - `sample_caught_winners` eating Catch Rate's width, and `cohort`
eating Controls', are what the review saw on the live desk.

`tier_performance` is the one tab with no plausible decoy (its only free-text
column IS `sample_observations`), so its red comes from the EMPTY render, where
the widest header - "Factor Hit Rate" - takes the slack.

No live store is opened: `conftest.py` already points `TRADINGBOTV3_DATA_DIR`
at a temp directory, and every path this panel reads is patched on top of that.

State on ``7e018c99``
---------------------
RED: the two `..._stretches_the_column_its_packet_names` cases,
`..._human_picks_budgets_the_cohort_column...`,
`..._playbooks_catch_rate_and_attributes...`,
`..._the_full_setups_table_stretches_the_tags...`,
`..._the_away_recap_elides_its_long_identifiers...`.
GREEN BY DESIGN: `..._no_cell_text_and_no_row_order_changes_on_any_tab`,
`..._the_compact_profile_still_pins_every_width_it_pinned_before`.
GREEN ALREADY, kept as a regression guard with the reason in its docstring:
`..._the_away_focus_table_names_its_symbol_column`.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QHeaderView  # noqa: E402

from ui.widgets.data_table import MAX_COLUMN_WIDTH, MiddleElideDelegate  # noqa: E402


#: The 4K desk viewport the packet names. Every width assertion here is made at
#: this size, because "the column is too narrow" is only a claim about a width
#: once the window it sits in is stated.
DESK_WIDTH = 3456
DESK_HEIGHT = 2160

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures"
TRACKER_RENDER_GOLDEN = GOLDEN_DIR / "g2b_tracker_render_golden.json"


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


# ===========================================================================
# Setup Tracker fixtures
# ===========================================================================


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


#: The decoy. Longer than any value in the column the packet names, and shaped
#: like the real thing - the scanner really does emit family names this long.
LONG_FAMILY = "avwap_second_deviation_reclaim_with_volume_expansion_and_earnings_anchor"
LONG_SAMPLES = (
    "NVDA 2026-08-11 +2.14R; AMD 2026-08-12 -1.00R; SMCI 2026-08-13 +0.42R; "
    "MU 2026-08-14 +1.08R; AVGO 2026-08-15 -0.31R"
)
SHORT_SAMPLES = "NVDA; AMD"


def _tracker_fixture_rows() -> dict[str, list[dict]]:
    """One populated row set per export. Every dict key is a column the panel
    reads; the values are the shapes the live exports carry."""
    return {
        "tier_list": [
            {
                "tier": "S",
                "symbol": "NVDA",
                "side": "LONG",
                "priority_score": "88.4",
                # DECOY: the natural text column today.
                "setup_family": LONG_FAMILY,
                "favorite_zone": "above_2nd_dev",
                "current_band_zone": "above_1st_dev",
                "trend_20d": "up",
                "scan_factor_match_count": "4",
                "scan_factor_matches": "rvol; gap",
            },
            {
                "tier": "A",
                "symbol": "AMD",
                "side": "SHORT",
                "priority_score": "71.2",
                "setup_family": "avwap_breakdown",
                "favorite_zone": "below_1st_dev",
                "current_band_zone": "below_1st_dev",
                "trend_20d": "down",
                "scan_factor_match_count": "2",
                "scan_factor_matches": "atr",
            },
        ],
        "setup_type": [
            {
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": LONG_FAMILY,
                "favorite_zone": "above_2nd_dev",
                "retest_label": "retested",
                "n_wins": "24",
                "n_losses": "16",
                "closed_setups": "40",
                "open_setups": "6",
                "avg_closed_r": "0.51",
                "avg_closed_r_edge": "0.12",
                "target_hit_rate": "0.44",
                "stop_rate": "0.28",
                "score_delta": "1.4",
                "ranking_score": "12.0",
                "sample_setups": SHORT_SAMPLES,
            },
        ],
        "recent_type": [
            {
                "status": "NEW",
                "namespace": "live",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": LONG_FAMILY,
                "n_wins": "12",
                "n_losses": "8",
                "win_rate_closed": "0.58",
                "closed_setups": "20",
                "tracked_setups": "31",
                "avg_closed_r": "0.44",
                "avg_closed_r_edge": "0.10",
                "target_hit_rate": "0.40",
                "stop_rate": "0.30",
                "representative_closed_r": "0.44",
                "sample_setups": SHORT_SAMPLES,
            },
        ],
        "short_horizon": [
            {
                "side": "LONG",
                "setup_family": LONG_FAMILY,
                "samples_2d": "18",
                "win_rate_2d": "0.55",
                "avg_r_1d": "0.21",
                "avg_r_2d": "0.34",
                "median_r_2d": "0.28",
                "avg_mfe_r_2d": "0.91",
                "avg_mae_r_2d": "-0.44",
                "recent_samples_2d": "9",
                "recent_avg_r_2d": "0.30",
                "short_term_score": "3.4",
                "sample_setups": SHORT_SAMPLES,
            },
        ],
        "playbooks": [
            {
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_family": "avwap_reclaim",
                "favorite_zone": "above_2nd_dev",
                "stop_reference_label": "below 1st dev",
                # The column the packet names. Deliberately SHORTER than the
                # samples beside it - that is the live defect.
                "profit_take_summary": "2R then trail the 1st dev",
                "closed_setups": "40",
                "open_setups": "5",
                "robust_closed_r": "0.62",
                "robust_closed_r_edge": "0.15",
                "win_rate_closed": "0.57",
                "target_hit_rate": "0.48",
                "ranking_score": "9.1",
                "experimental": "",
                # DECOY.
                "sample_setups": LONG_SAMPLES,
            },
        ],
        "scan_factor": [
            {
                "horizon_sessions": "5",
                "side": "LONG",
                # DECOY.
                "factor_label": "relative_volume_vs_20d_median_at_the_scan_bar_close",
                "value_label": "high",
                "observation_count": "44",
                "symbol_count": "31",
                "win_rate": "0.58",
                "avg_side_return_pct": "1.42",
                "side_return_edge_pct": "0.51",
                "success_score": "6.2",
                "impact_score": "2.1",
                "sample_observations": "NVDA; AMD",
            },
        ],
        "tier_performance": [
            {
                "horizon_sessions": "5",
                "tier": "S",
                "side": "LONG",
                "observation_count": "120",
                "symbol_count": "74",
                "win_rate": "0.61",
                "avg_side_return_pct": "1.88",
                "side_return_edge_pct": "0.62",
                "positive_scan_factor_match_rate": "0.71",
                "sample_observations": "NVDA 2026-08-11; AMD 2026-08-12",
            },
            {
                "horizon_sessions": "5",
                "tier": "A",
                "side": "SHORT",
                "observation_count": "80",
                "symbol_count": "52",
                "win_rate": "0.49",
                "avg_side_return_pct": "0.41",
                "side_return_edge_pct": "0.08",
                "positive_scan_factor_match_rate": "0.55",
                "sample_observations": "SMCI 2026-08-13",
            },
        ],
        "catch_rate": [
            {
                "horizon_sessions": "5",
                "side": "LONG",
                "factor_opportunity_count": "318",
                "factor_winner_count": "96",
                "caught_winner_count": "29",
                "caught_winner_rate": "0.30",
                "missed_winner_count": "67",
                # DECOY - and the live defect verbatim: the caught examples ate
                # the width and the missed ones clipped.
                "sample_caught_winners": LONG_SAMPLES,
                "sample_missed_winners": "TSLA; PLTR",
            },
        ],
        "band_variant": [
            {
                # DECOY.
                "setup_family": LONG_FAMILY,
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "n": "412",
                "n_variant": "398",
                "n_variant_unmeasured": "14",
                "avg_total_r_champion": "0.51",
                "avg_total_r_variant": "0.58",
                "stop_out_rate_champion": "0.28",
                "stop_out_rate_variant": "0.24",
                "target_hit_rate_champion": "0.44",
                "target_hit_rate_variant": "0.47",
                "mean_stop_distance_atr_champion": "1.10",
                "mean_stop_distance_atr_variant": "1.22",
                "exit_template_id": "baseline_2r_trail",
            },
        ],
        "discovery": [
            {
                "window": "all",
                "row_kind": "cohort",
                # DECOY - and the packet's own example: `cohort` must elide, so
                # that `Win % (low)` never takes the slack.
                "cohort": "control_rejected_by_gate_second_dev_reclaim_long_favorite",
                "side": "LONG",
                "setup_family": "avwap_reclaim",
                "win_rate": "0.42",
                "win_rate_lb": "0.33",
                "n": "112",
                "wins": "47",
                "losses": "65",
                "avg_closed_r": "-0.08",
                "n_expired_unmeasured": "9",
                "flag": "",
            },
        ],
        "exit_framework": [
            {
                "framework_family": "baseline",
                "experimental": "false",
                # DECOY.
                "exit_template_id": "baseline_2r_then_trail_the_first_deviation_band",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "win_rate": "0.55",
                "win_rate_lb": "0.46",
                "n": "140",
                "n_closed": "128",
                "avg_closed_r": "0.44",
                "stop_out_rate": "0.29",
                "target_hit_rate": "0.46",
                "n_expired_unmeasured": "12",
                "n_filtered_by_experiment": "0",
            },
            {
                "framework_family": "comparison_apr2026",
                "experimental": "true",
                "exit_template_id": "comparison_apr2026_atr_trail",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "win_rate": "0.51",
                "win_rate_lb": "0.42",
                "n": "119",
                "n_closed": "110",
                "avg_closed_r": "0.39",
                "stop_out_rate": "0.31",
                "target_hit_rate": "0.41",
                "n_expired_unmeasured": "9",
                "n_filtered_by_experiment": "21",
            },
        ],
        "attribute": [
            {
                "attribute_label": "Relative volume",
                # DECOY - and the column the packet asks to elide.
                "value_label": "above the 80th percentile of the trailing 20 sessions",
                "side": "LONG",
                "priority_bucket": "favorite_setup",
                "setup_count": "240",
                "closed_tradeable_setup_count": "212",
                "meets_n_floor": "true",
                "avg_closed_r": "0.62",
                "avg_closed_r_edge": "0.18",
                "target_hit_rate_edge": "0.05",
                "stop_rate_edge": "-0.03",
                "sample_setups": SHORT_SAMPLES,
            },
        ],
        # Input to `build_human_focus_comparison_rows`, not a panel CSV. The
        # cohort key is an origin sub-cohort, so the rendered `Cohort` cell is
        # long - the identity column that eats Human Picks' width today.
        "human_focus": [
            {
                "cohort": "human_focus_swing_second_dev_breakout",
                "side": "LONG",
                "horizon_sessions": "5",
                "sample_count": "44",
                "win_rate": "0.57",
                "avg_side_return": "0.0182",
                "profit_factor": "1.41",
            },
            {
                "cohort": "human_focus_m5_alerts_intraday_bounce",
                "side": "SHORT",
                "horizon_sessions": "5",
                "sample_count": "18",
                "win_rate": "0.44",
                "avg_side_return": "-0.0031",
                "profit_factor": "0.88",
            },
        ],
    }


def _point_panel_at(panel_module, tmp_path: Path, monkeypatch, *, populated: bool):
    """Patch the panel module's OWN file constants, then fill them or not.

    Never `project_paths.<NAME>`: `SETUP_TYPE_STATS_FILE` and its siblings are
    derived from `MASTER_AVWAP_SETUP_STATS_FILE` at IMPORT time, so patching the
    source attribute would leave the panel reading the same path it always did.
    """
    rows = _tracker_fixture_rows()
    targets = {
        "SETUP_TYPE_STATS_FILE": ("master_avwap_setup_type_stats.csv", rows["setup_type"]),
        "RECENT_SETUP_TYPE_STATS_FILE": (
            "master_avwap_setup_type_recent_stats.csv",
            rows["recent_type"],
        ),
        "SETUP_PLAYBOOKS_FILE": ("master_avwap_setup_playbooks.csv", rows["playbooks"]),
        "SHORT_HORIZON_FILE": ("master_avwap_setup_short_horizon.csv", rows["short_horizon"]),
        "BAND_VARIANT_STATS_FILE": ("master_avwap_band_variant_stats.csv", rows["band_variant"]),
        "CONTROL_DISCOVERY_STATS_FILE": (
            "master_avwap_control_discovery.csv",
            rows["discovery"],
        ),
        "STUDY_DISCOVERY_STATS_FILE": ("master_avwap_study_discovery.csv", rows["discovery"]),
        "EXIT_FRAMEWORK_STATS_FILE": (
            "master_avwap_exit_framework_stats.csv",
            rows["exit_framework"],
        ),
        "MASTER_AVWAP_TIER_LIST_FILE": ("master_avwap_tier_list.csv", rows["tier_list"]),
        "MASTER_AVWAP_TIER_PERFORMANCE_FILE": (
            "master_avwap_tier_performance.csv",
            rows["tier_performance"],
        ),
        "MASTER_AVWAP_TIER_CATCH_RATE_FILE": (
            "master_avwap_tier_catch_rate.csv",
            rows["catch_rate"],
        ),
        "MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE": (
            "master_avwap_scan_factor_leaderboard.csv",
            rows["scan_factor"],
        ),
        "MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE": (
            "master_avwap_setup_attribute_leaderboard.csv",
            rows["attribute"],
        ),
    }
    for name, (filename, csv_rows) in targets.items():
        assert hasattr(panel_module, name), f"{name} is not bound in the panel module"
        path = tmp_path / filename
        if populated:
            _write_csv(path, csv_rows)
        monkeypatch.setattr(panel_module, name, path)

    human_rows = rows["human_focus"] if populated else []
    monkeypatch.setattr(
        panel_module, "load_human_focus_performance_rows", lambda *_a, **_k: list(human_rows)
    )
    panel_module.clear_setup_tracker_csv_cache()


def _tracker_panel(app, panel_module, tmp_path, monkeypatch, *, populated: bool):
    _point_panel_at(panel_module, tmp_path, monkeypatch, populated=populated)
    panel = panel_module.SetupTrackerPanel()
    panel.resize(DESK_WIDTH, DESK_HEIGHT)
    panel.show()
    app.processEvents()
    # The Refresh button's own path, with the desk's geometry already applied -
    # the rule measures what is on screen.
    panel.refresh()
    # The attribute leaderboard arrives through the worker's slot; calling the
    # slot directly is the same code the signal reaches, without racing a thread.
    panel._on_attributes_loaded(
        {
            "rows": panel_module._rank_attribute_leaderboard(
                _tracker_fixture_rows()["attribute"] if populated else []
            ),
            "message": "",
        }
    )
    app.processEvents()
    return panel


@pytest.fixture
def panel_module(app):
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    yield setup_tracker_panel
    setup_tracker_panel.clear_setup_tracker_csv_cache()


#: (table attribute, column-tuple attribute, the key that must stretch, the keys
#: that must carry the middle elision). Straight from packet G2b's item list.
TRACKER_TEXT_COLUMNS = (
    ("current_table", "CURRENT_PICK_COLUMNS", "scan_factor_matches", ()),
    ("setup_type_table", "SETUP_TYPE_COLUMNS", "sample_setups", ()),
    ("recent_type_table", "RECENT_TYPE_COLUMNS", "sample_setups", ()),
    ("short_term_table", "SHORT_TERM_COLUMNS", "sample_setups", ()),
    ("playbook_table", "PLAYBOOK_COLUMNS", "profit_take_summary", ("sample_setups",)),
    ("scan_factor_table", "SCAN_FACTOR_COLUMNS", "sample_observations", ()),
    ("tier_performance_table", "TIER_PERFORMANCE_COLUMNS", "sample_observations", ()),
    (
        "catch_rate_table",
        "CATCH_RATE_COLUMNS",
        "sample_missed_winners",
        ("sample_caught_winners",),
    ),
    ("band_variant_table", "BAND_VARIANT_COLUMNS", "setup_family", ("exit_template_id",)),  # lead 2026-09-07: the 10-char template id left ~2,000 px dead
    ("control_discovery_table", "DISCOVERY_COLUMNS", "setup_family", ("cohort",)),
    ("study_discovery_table", "DISCOVERY_COLUMNS", "setup_family", ("cohort",)),
    ("exit_framework_table", "EXIT_FRAMEWORK_COLUMNS", "framework_family", ("exit_template_id",)),
    ("attribute_table", "ATTRIBUTE_LEADERBOARD_COLUMNS", "sample_setups", ("value_label",)),
)


def _index_of(panel_module, columns_attr: str, key: str) -> int:
    columns = getattr(panel_module, columns_attr)
    for index, (column_key, _label) in enumerate(columns):
        if column_key == key:
            return index
    raise AssertionError(f"{key!r} is not a column of {columns_attr}")


def _stretching(table) -> list[int]:
    header = table.horizontalHeader()
    return [
        column
        for column in range(table.model().columnCount())
        if header.sectionResizeMode(column) == QHeaderView.ResizeMode.Stretch
    ]


# ===========================================================================
# 1 - every tab names its text column, populated and empty
# ===========================================================================


@pytest.mark.parametrize("populated", [True, False], ids=["populated", "empty"])
def test_every_setup_tracker_tab_stretches_the_column_its_packet_names(
    app, panel_module, tmp_path, monkeypatch, populated
):
    """The assignment is by NAME, not by content.

    Each populated fixture carries a decoy text column longer than the named
    one, so the measured path picks the decoy; each empty render measures
    HEADERS, where "Win % (low)" and "Factor Hit Rate" beat "Family" and
    "Samples". Both are the live complaint.
    """
    panel = _tracker_panel(app, panel_module, tmp_path, monkeypatch, populated=populated)
    try:
        failures = []
        for table_attr, columns_attr, text_key, _elide in TRACKER_TEXT_COLUMNS:
            table = getattr(panel, table_attr)
            wanted = _index_of(panel_module, columns_attr, text_key)
            header = table.horizontalHeader()
            columns = table.model().columnCount()
            stretching = _stretching(table)
            if stretching != [wanted]:
                names = [getattr(panel_module, columns_attr)[c][0] for c in stretching]
                failures.append(
                    f"{table_attr}: {text_key!r} (column {wanted}) must be the one "
                    f"stretched column; stretched instead: {names or 'nothing'}"
                )
            for column in range(columns):
                if column == wanted:
                    continue
                width = header.sectionSize(column)
                if width > MAX_COLUMN_WIDTH:
                    key = getattr(panel_module, columns_attr)[column][0]
                    failures.append(
                        f"{table_attr}: {key!r} is {width}px, over the "
                        f"{MAX_COLUMN_WIDTH}px ceiling"
                    )
        assert not failures, "\n".join(failures)
    finally:
        panel.deleteLater()


# ===========================================================================
# 2 - Human Picks: an identifier and ten measurements, and nothing stretches
# ===========================================================================


def test_human_picks_budgets_the_cohort_column_and_stretches_nothing(
    app, panel_module, tmp_path, monkeypatch
):
    """The GUI review's words: "a cohort column consumed most of the width and
    pushed measurements away". There is no free-text column on this tab, so the
    slack stays empty rather than being handed to an identifier."""
    panel = _tracker_panel(app, panel_module, tmp_path, monkeypatch, populated=True)
    try:
        table = panel.human_pick_table
        header = table.horizontalHeader()
        cohort = _index_of(panel_module, "HUMAN_PICK_COLUMNS", "cohort")
        assert table.model().rowCount() > 0, "the Human Picks fixture rendered no rows"

        assert _stretching(table) == [], (
            "no column on Human Picks may take the slack - it is an identifier "
            "and ten measurements"
        )
        assert header.stretchLastSection() is False, (
            "stretchLastSection would hand the slack to `Delta %`"
        )
        assert header.sectionSize(cohort) <= MAX_COLUMN_WIDTH, (
            f"cohort is {header.sectionSize(cohort)}px, over the "
            f"{MAX_COLUMN_WIDTH}px budget"
        )
        assert isinstance(table.itemDelegateForColumn(cohort), MiddleElideDelegate), (
            "cohort is an identifier: it elides in the MIDDLE so its tail survives"
        )
        # The full value stays reachable. `MiddleElideDelegate` serves it from
        # the DisplayRole when the model offers no ToolTipRole, which is this
        # model's case for `cohort`; so the cell text is the whole value and the
        # elision is a rendering, never a truncation of the data.
        model = table.model()
        index = model.index(0, cohort)
        full = str(model.data(index, Qt.ItemDataRole.DisplayRole) or "")
        assert full, "cohort cell is empty"
        tooltip = model.data(index, Qt.ItemDataRole.ToolTipRole)
        assert tooltip in (None, "", full), (
            "a cohort tooltip that is neither absent nor the full value would "
            f"leave the elided tail unrecoverable: {tooltip!r}"
        )
    finally:
        panel.deleteLater()


# ===========================================================================
# 3 - the three tabs the review named, and the worker path
# ===========================================================================


def test_playbooks_catch_rate_and_attributes_show_the_column_the_review_asked_for(
    app, panel_module, tmp_path, monkeypatch
):
    """Playbooks' Exit Plan, Catch Rate's Missed Samples, and the attribute
    leaderboard's Examples - the last one reached only through the worker
    slot's own `fit_columns()`, which is a second call site the fix has to
    cover."""
    panel = _tracker_panel(app, panel_module, tmp_path, monkeypatch, populated=True)
    try:
        playbook_exit = _index_of(panel_module, "PLAYBOOK_COLUMNS", "profit_take_summary")
        playbook_samples = _index_of(panel_module, "PLAYBOOK_COLUMNS", "sample_setups")
        assert panel.playbook_table.model().rowCount() > 0
        assert _stretching(panel.playbook_table) == [playbook_exit]
        assert isinstance(
            panel.playbook_table.itemDelegateForColumn(playbook_samples),
            MiddleElideDelegate,
        )

        missed = _index_of(panel_module, "CATCH_RATE_COLUMNS", "sample_missed_winners")
        caught = _index_of(panel_module, "CATCH_RATE_COLUMNS", "sample_caught_winners")
        assert panel.catch_rate_table.model().rowCount() > 0
        assert _stretching(panel.catch_rate_table) == [missed]
        assert isinstance(
            panel.catch_rate_table.itemDelegateForColumn(caught), MiddleElideDelegate
        )

        examples = _index_of(panel_module, "ATTRIBUTE_LEADERBOARD_COLUMNS", "sample_setups")
        value_label = _index_of(panel_module, "ATTRIBUTE_LEADERBOARD_COLUMNS", "value_label")
        assert panel.attribute_table.model().rowCount() > 0
        assert _stretching(panel.attribute_table) == [examples]
        assert isinstance(
            panel.attribute_table.itemDelegateForColumn(value_label), MiddleElideDelegate
        )
    finally:
        panel.deleteLater()


# ===========================================================================
# 4 - GOLDEN: this is a layout packet, so no cell and no order may move
# ===========================================================================


def _render_grid(panel) -> dict:
    """Every tab's rendered DisplayRole text, in the order the proxy shows it."""
    grid: dict[str, list[list[str]]] = {}
    for table_attr, _columns_attr, _text_key, _elide in TRACKER_TEXT_COLUMNS + (
        ("human_pick_table", "HUMAN_PICK_COLUMNS", "", ()),
    ):
        table = getattr(panel, table_attr)
        model = table.model()
        rows = []
        for row in range(model.rowCount()):
            rows.append(
                [
                    str(model.data(model.index(row, column), Qt.ItemDataRole.DisplayRole) or "")
                    for column in range(model.columnCount())
                ]
            )
        grid[table_attr] = rows
    return grid


def test_no_cell_text_and_no_row_order_changes_on_any_tab(
    app, panel_module, tmp_path, monkeypatch
):
    """GOLDEN - GREEN BY DESIGN on ``7e018c99``.

    G2b is a layout lane: widths, elision and positions only. This pins every
    tab's rendered cells and their order against
    ``tests/fixtures/g2b_tracker_render_golden.json``, which was generated FROM
    THE BASE COMMIT by ``_write_render_golden`` below. The builder must never
    regenerate it - a fixture written by the code it pins is a self-portrait.

    The fixture carries its Milestone 3 contract and its own INPUT, and the
    input is checked against `_tracker_fixture_rows()` first: a golden whose
    input drifted would otherwise pin nothing and still pass.
    """
    from tests.conftest import load_fixture_contract

    contract = load_fixture_contract(TRACKER_RENDER_GOLDEN)
    assert contract["fixture_rows"] == _tracker_fixture_rows(), (
        "the fixture rows in this test file no longer match the ones the golden "
        "was rendered from, so the golden pins nothing"
    )

    panel = _tracker_panel(app, panel_module, tmp_path, monkeypatch, populated=True)
    try:
        rendered = _render_grid(panel)
    finally:
        panel.deleteLater()

    contract.assert_matches(rendered, contract["rendered"], "setup tracker cells")


def _write_render_golden(app, panel_module, tmp_path, monkeypatch) -> None:
    """Generator, run ONCE by the tester on ``7e018c99``. Never by the builder."""
    import hashlib

    panel = _tracker_panel(app, panel_module, tmp_path, monkeypatch, populated=True)
    try:
        rows = _tracker_fixture_rows()
        payload = {
            "schema": "g2b_tracker_render/v1",
            "feature_version": "setup_tracker_panel_pre_g2b",
            "raw_input_keys": ["fixture_rows"],
            "raw_input_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "acquired_at": "2026-09-07T00:00:00-07:00",
            "universe_version": "synthetic-g2b-tracker-fixture",
            "provider_assumptions": (
                "none - the rows are written to CSVs in tmp_path and read back "
                "through the panel's own reader; no network, no live store"
            ),
            "as_of": "2026-09-07T00:00:00-07:00",
            "expected_keys": ["rendered"],
            "numeric_tolerance": 0.0,
            "intentional_difference": "",
            "source": {
                "repo_commit": "7e018c99",
                "generated_by": (
                    "tests/test_g2b_named_columns.py::_write_render_golden, run "
                    "by the tester before any G2b fix existed"
                ),
            },
            "fixture_rows": rows,
            "rendered": _render_grid(panel),
        }
        TRACKER_RENDER_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        TRACKER_RENDER_GOLDEN.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        panel.deleteLater()


# ===========================================================================
# Desk Setups (MasterAvwapPanel) - fixtures
# ===========================================================================


#: A key level long enough to need more than the 260px ceiling. The live scan
#: writes these: the level, the band it belongs to and the date it was anchored.
LONG_KEY_LEVEL = (
    "$412.50 2nd dev band, anchored 2026-05-12 post-earnings, retested 2026-08-27"
)
#: The tags the table exists to show. Shorter than the long key level above, so
#: the MEASURED rule cannot pick this column and the packet's name has to.
SETUP_TAGS = ["2nd dev", "post-earnings", "rvol"]


def _desk_rows():
    from ui.models.setup import SetupRow

    return [
        SetupRow(
            symbol="NVDA",
            side="LONG",
            score=88.4,
            bucket="favorite_setup",
            setup_tags=list(SETUP_TAGS),
            key_level="$412.50 2nd dev",
            supports=3,
            expected_r=1.42,
            sector="Technology",
            industry="Semiconductors",
            d1_vs_sector=1.8,
            d1_vs_industry=0.9,
            last_trade_date="2026-09-04",
            raw={"setup_family": "avwap_reclaim"},
        ),
        SetupRow(
            symbol="AMD",
            side="SHORT",
            score=71.2,
            bucket="favorite_setup",
            setup_tags=["breakdown"],
            # The row that needs the elision: longer than any budget this table
            # can give a non-stretching column.
            key_level=LONG_KEY_LEVEL,
            supports=1,
            expected_r=0.61,
            sector="Technology",
            industry="Semiconductors",
            d1_vs_sector=-1.1,
            d1_vs_industry=-0.4,
            last_trade_date="2026-09-04",
            raw={"setup_family": "avwap_breakdown"},
        ),
    ]


def _desk_panel(app, *, width: int, height: int, profile: str):
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    panel = MasterAvwapPanel()
    panel.resize(width, height)
    panel.show()
    app.processEvents()
    panel.set_rows(_desk_rows())
    # `set_rows` re-applies whatever profile is active, so the profile is
    # switched afterwards through the panel's own public verb - the same call
    # the desk makes when the setups page goes full width.
    panel.set_column_profile(profile)
    app.processEvents()
    return panel


def _desk_index(key: str) -> int:
    from ui.models.setup_table_model import SetupTableModel

    for index, (column_key, _label) in enumerate(SetupTableModel.COLUMNS):
        if column_key == key:
            return index
    raise AssertionError(f"{key!r} is not a SetupTableModel column")


# ===========================================================================
# 5 - Desk Setups, FULL profile
# ===========================================================================


def test_the_full_setups_table_stretches_the_tags_and_elides_the_key_level(app):
    """At 3456 wide the trader reads symbol, side, bucket and key level whole.

    `setup_tags` is the column with room to grow; `key_level` is an identifier
    whose long form must elide in the MIDDLE rather than take the slack. On
    ``7e018c99`` the measured rule hands the slack to `key_level` (the AMD row
    is 75 characters), so the tags stay pinned at their measured width and no
    column carries the elision.

    The `horizontalAdvance <= sectionSize` check is a FLOOR, not the whole
    requirement: `side` and `bucket` are painted as chips with padding around
    the text, so a column that only just clears this bound is still tight. It
    is what the packet asks for and what a clipped column fails.
    """
    panel = _desk_panel(app, width=DESK_WIDTH, height=DESK_HEIGHT, profile="full")
    try:
        table = panel.table
        header = table.horizontalHeader()
        model = table.model()
        assert model.rowCount() == 2, "the desk fixture rendered no rows"

        tags = _desk_index("setup_tags")
        key_level = _desk_index("key_level")

        stretching = [
            column
            for column in range(model.columnCount())
            if header.sectionResizeMode(column) == QHeaderView.ResizeMode.Stretch
        ]
        assert stretching == [tags], (
            "`Setup Tags` is the full profile's text column; stretched instead: "
            + str([panel.model.COLUMNS[c][0] for c in stretching] or "nothing")
        )

        metrics = table.fontMetrics()
        # Row 0 is the ordinary row: every one of these four reads whole.
        for key in ("symbol", "side", "bucket", "key_level"):
            column = _desk_index(key)
            text = str(model.data(model.index(0, column), Qt.ItemDataRole.DisplayRole) or "")
            assert text, f"{key} rendered nothing on row 0"
            assert metrics.horizontalAdvance(text) <= header.sectionSize(column), (
                f"{key} is clipped at the desk's full width: {text!r} needs "
                f"{metrics.horizontalAdvance(text)}px and has "
                f"{header.sectionSize(column)}px"
            )

        # Row 1's key level is past any budget, so it elides in the MIDDLE.
        long_text = str(
            model.data(model.index(1, key_level), Qt.ItemDataRole.DisplayRole) or ""
        )
        assert long_text == LONG_KEY_LEVEL
        assert metrics.horizontalAdvance(long_text) > header.sectionSize(key_level), (
            "the fixture no longer overflows its column, so it proves nothing "
            "about the elision"
        )
        assert isinstance(table.itemDelegateForColumn(key_level), MiddleElideDelegate), (
            "`Key Level / Entry` is an identifier: its tail is the anchor date "
            "and an end elision loses it"
        )

        assert header.sectionSize(0) == 36, "the star column moved off 36px"
        assert header.sectionSize(1) == 36, "the dislike column moved off 36px"
    finally:
        panel.deleteLater()


# ===========================================================================
# 6 - GOLDEN: the compact profile does not move
# ===========================================================================


#: Copied by hand from `master_avwap_panel.COMPACT_COLUMN_WIDTHS` at ``7e018c99``
#: - the OLD code. If the builder edits that dict, or lets `fit_columns` run
#: over the compact profile, this diverges.
COMPACT_WIDTHS_AT_BASE = {
    "favorite": 34,
    "dislike": 34,
    "symbol": 74,
    "side": 66,
    "bucket": 96,
    "key_level": 116,
    "expected_r": 58,
    "setup_tags": 120,
    "industry": 130,
    "d1_vs_sector": 84,
    "d1_vs_industry": 88,
    "family_win_rate": 132,
}
COMPACT_HIDDEN_AT_BASE = {"score", "supports", "sector", "last_trade_date"}


def test_the_compact_profile_still_pins_every_width_it_pinned_before(app):
    """GOLDEN - GREEN BY DESIGN on ``7e018c99``.

    G2b.3 is the FULL profile only. The compact profile is what keeps the desk's
    1640px setups pane free of a horizontal scrollbar, and `fit_columns`' 80px
    floor is exactly what would break it - so this pins the pinned widths, the
    hidden set, and the moved `Exp R` position, at the size the compact profile
    was measured for.

    `family_win_rate` is the last section and the compact branch ends with
    `setStretchLastSection(True)`, so it is checked as a MINIMUM: it never
    shrinks below its pinned 132px.
    """
    from ui.panels import master_avwap_panel

    assert master_avwap_panel.COMPACT_COLUMN_WIDTHS == COMPACT_WIDTHS_AT_BASE
    assert master_avwap_panel.COMPACT_HIDDEN_COLUMNS == COMPACT_HIDDEN_AT_BASE

    panel = _desk_panel(app, width=1640, height=980, profile="compact")
    try:
        table = panel.table
        header = table.horizontalHeader()
        last_visible = max(
            column
            for column in range(len(panel.model.COLUMNS))
            if not table.isColumnHidden(column)
        )
        for key, expected in COMPACT_WIDTHS_AT_BASE.items():
            column = _desk_index(key)
            if key in COMPACT_HIDDEN_AT_BASE:
                continue
            assert not table.isColumnHidden(column), f"{key} vanished from the compact profile"
            actual = header.sectionSize(column)
            if column == last_visible:
                assert actual >= expected, f"{key}: {actual}px is under its pinned {expected}px"
            else:
                assert actual == expected, f"{key}: {actual}px, pinned at {expected}px"
        for key in COMPACT_HIDDEN_AT_BASE:
            assert table.isColumnHidden(_desk_index(key)), f"{key} should be hidden when compact"
        # `Exp R` is moved into reading position beside the level, by visual
        # index, never by editing COLUMNS.
        assert header.visualIndex(_desk_index("expected_r")) == (
            header.visualIndex(_desk_index("key_level")) + 1
        )
    finally:
        panel.deleteLater()


# ===========================================================================
# 7 - AWAY Recap: the long identifiers elide, Focus names its own column
# ===========================================================================


LONG_SWING_LINE = (
    "1. FROG LONG - reclaimed the 2nd deviation band on 3.1x relative volume and "
    "held the previous day's high into the close"
)
LONG_TRIGGER = (
    "M5 bounce off the anchored VWAP with 2.4x relative volume and a higher low "
    "against the session VWAP"
)
LONG_CELL = "bounce_reclaim_long / held 22m x ran 1.8R"


def _away_recap() -> dict:
    return {
        "summary": "one day",
        "best_swings": [
            {"rank": 1, "symbol": "FROG", "side": "LONG", "text": LONG_SWING_LINE},
        ],
        "classified_alerts": [
            {
                "time_text": "09:31",
                "symbol": "OKTA",
                "side": "LONG",
                "tier": "A",
                "is_d1": False,
                "trigger": LONG_TRIGGER,
                "cell": "bounce_reclaim_long",
                "held_run_suffix": "/ held 22m x ran 1.8R",
            },
        ],
        "staged_picks": [{"symbol": "MRK", "side": "LONG"}],
        "focus_to_manage": [{"symbol": "GFS", "side": "SHORT"}],
    }


def test_the_away_recap_elides_its_long_identifiers_and_keeps_the_full_value(app):
    """`Line`, `Trigger` and `Cell / held x ran` are long free text that stretch
    today and elide at the END when the window is narrow, which loses the
    `held x ran` suffix - the part a reader of the recap is looking for. The
    middle elision keeps the tail, and the item's own tooltip keeps the whole
    string.

    `elide_columns` is passed nowhere in `away_recap_panel.py` on ``7e018c99``,
    so every one of these is red.
    """
    from ui.panels.away_recap_panel import AwayRecapPanel

    panel = AwayRecapPanel()
    panel.resize(DESK_WIDTH, DESK_HEIGHT)
    panel.show()
    app.processEvents()
    try:
        panel._render(_away_recap())

        assert isinstance(
            panel.swings.itemDelegateForColumn(3), MiddleElideDelegate
        ), "the ranked-swing `Line` column carries no middle elision"
        assert panel.swings.item(0, 3).toolTip() == LONG_SWING_LINE

        assert isinstance(
            panel.alerts.itemDelegateForColumn(5), MiddleElideDelegate
        ), "the alert `Trigger` column carries no middle elision"
        assert isinstance(
            panel.alerts.itemDelegateForColumn(6), MiddleElideDelegate
        ), "the alert `Cell / held x ran` column carries no middle elision"
        assert panel.alerts.item(0, 5).toolTip() == LONG_TRIGGER
        assert panel.alerts.item(0, 6).toolTip() == LONG_CELL
    finally:
        panel.deleteLater()


def test_the_away_focus_table_names_its_symbol_column(app):
    """G2b.4's fourth line. **PASSES ALREADY on ``7e018c99``** - and the tester
    could not make it fail honestly.

    `focus_table` passes NO `text_columns`, so its slack goes wherever the
    measurement lands. It has exactly two columns, `Symbol` and `Side`, and
    both hold short enum-ish values, so the measured answer is column 0 for
    every row set the recap can produce: `Symbol` beats `Side` as a header and
    `LONG`/`SHORT` never overtake a ticker plus that header. Forcing a red here
    would mean inventing a `Side` value the recap cannot emit.

    So the test is a REGRESSION guard, not a fail-first proof: naming column 0
    (G2b.4) has to keep this answer, and must not make it drift.
    """
    from ui.panels.away_recap_panel import AwayRecapPanel

    panel = AwayRecapPanel()
    panel.resize(DESK_WIDTH, DESK_HEIGHT)
    panel.show()
    app.processEvents()
    try:
        panel._render(_away_recap())
        header = panel.focus_table.horizontalHeader()
        assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
        assert header.stretchLastSection() is False
    finally:
        panel.deleteLater()
