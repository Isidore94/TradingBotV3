"""Packet M5.2 / M5.3 - the Controls, Studies and Exit frameworks tabs.

Three pure CSV readers, like every other section on this page. What they add is
a POPULATION SENTENCE above each table, because these three populations are the
easiest on the desk to misread:

* a **control** row is a setup the scan REJECTED. A reader who takes it for a
  pick has read the gate's holdout as a recommendation.
* a **study** row is an idea that has never been promoted and touches no score.
* an **exit framework** row includes `comparison_apr2026`, an EXPERIMENTAL exit
  template simulated on the same setups as the baseline. It is a what-if, and
  the row says so in its own cells.

Win rate leads and the sort is the **Wilson lower bound**, never the raw rate -
`CLAUDE.md`'s headline rule. A raw 100% on two rejected setups would otherwise
sit above a 60% on ninety and read as the strongest thing on the page.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


DISCOVERY_FIELDS = (
    "window",
    "window_sessions",
    "window_start",
    "window_end",
    "row_kind",
    "cohort",
    "side",
    "setup_family",
    "n",
    "wins",
    "losses",
    "win_rate",
    "win_rate_lb",
    "meets_n_floor",
    "avg_closed_r",
    "n_expired_unmeasured",
    "flag",
)


def _discovery_row(family, *, n, wins, win_rate, win_rate_lb, avg_r, kind="family"):
    return {
        "window": "all",
        "window_sessions": "",
        "window_start": "",
        "window_end": "",
        "row_kind": kind,
        "cohort": "",
        "side": "LONG",
        "setup_family": family,
        "n": str(n),
        "wins": str(wins),
        "losses": str(n - wins),
        "win_rate": str(win_rate),
        "win_rate_lb": str(win_rate_lb),
        "meets_n_floor": "False",
        "avg_closed_r": str(avg_r),
        "n_expired_unmeasured": "0",
        "flag": "",
    }


#: A thin cell with a perfect raw rate, and a fat cell with a lower raw rate.
#: The lower bound is the honest ordering and putting the thin one on top is the
#: exact defect this sort exists to prevent.
THIN = _discovery_row("thin_but_perfect", n=2, wins=2, win_rate=1.0, win_rate_lb=0.342, avg_r=1.9)
FAT = _discovery_row("fat_and_good", n=90, wins=54, win_rate=0.6, win_rate_lb=0.496, avg_r=0.4)
DISCOVERY_ROWS = [THIN, FAT]

FRAMEWORK_FIELDS = (
    "framework_family",
    "exit_template_id",
    "exit_template_label",
    "side",
    "priority_bucket",
    "experimental",
    "framework_version",
    "n",
    "n_closed",
    "wins",
    "losses",
    "win_rate",
    "win_rate_lb",
    "meets_n_floor",
    "avg_closed_r",
    "stop_out_rate",
    "target_hit_rate",
    "n_expired_unmeasured",
)

FRAMEWORK_ROWS = [
    {
        "framework_family": "baseline",
        "exit_template_id": "full_band2",
        "exit_template_label": "Full at band2",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "experimental": "False",
        "framework_version": "baseline",
        "n": "40",
        "n_closed": "30",
        "wins": "18",
        "losses": "12",
        "win_rate": "0.6",
        "win_rate_lb": "0.423",
        "meets_n_floor": "True",
        "avg_closed_r": "0.35",
        "stop_out_rate": "0.4",
        "target_hit_rate": "0.6",
        "n_expired_unmeasured": "0",
    },
    {
        "framework_family": "comparison_apr2026",
        "exit_template_id": "exp_full_band2_hard_stop_125r",
        "exit_template_label": "EXP Full at band2 + 1.25R hard stop",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "experimental": "True",
        "framework_version": "2026-04-14",
        "n": "40",
        "n_closed": "30",
        "wins": "15",
        "losses": "15",
        "win_rate": "0.5",
        "win_rate_lb": "0.331",
        "meets_n_floor": "True",
        "avg_closed_r": "0.12",
        "stop_out_rate": "0.5",
        "target_hit_rate": "0.5",
        "n_expired_unmeasured": "0",
    },
]


def _write(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


CONTROL_CSV = "control_discovery.csv"
STUDY_CSV = "study_discovery.csv"
FRAMEWORK_CSV = "exit_framework.csv"


@pytest.fixture
def panel_module(monkeypatch, tmp_path):
    if _qt_app() is None:
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    monkeypatch.setattr(
        setup_tracker_panel, "CONTROL_DISCOVERY_STATS_FILE", tmp_path / CONTROL_CSV
    )
    monkeypatch.setattr(
        setup_tracker_panel, "STUDY_DISCOVERY_STATS_FILE", tmp_path / STUDY_CSV
    )
    monkeypatch.setattr(
        setup_tracker_panel, "EXIT_FRAMEWORK_STATS_FILE", tmp_path / FRAMEWORK_CSV
    )
    return setup_tracker_panel


def _tab_titles(panel) -> list[str]:
    return [panel.tabs.tabText(index) for index in range(panel.tabs.count())]


# ---------------------------------------------------------------------------
# The tabs exist and render
# ---------------------------------------------------------------------------


def test_the_three_tabs_are_on_the_page(panel_module, tmp_path):
    _write(tmp_path / CONTROL_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    _write(tmp_path / STUDY_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    _write(tmp_path / FRAMEWORK_CSV, FRAMEWORK_FIELDS, FRAMEWORK_ROWS)
    panel = panel_module.SetupTrackerPanel()
    try:
        titles = _tab_titles(panel)
        assert "Controls" in titles
        assert "Studies" in titles
        assert "Exit frameworks" in titles
    finally:
        panel.deleteLater()


def test_the_tabs_render_their_exports(panel_module, tmp_path):
    _write(tmp_path / CONTROL_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    _write(tmp_path / STUDY_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    _write(tmp_path / FRAMEWORK_CSV, FRAMEWORK_FIELDS, FRAMEWORK_ROWS)
    panel = panel_module.SetupTrackerPanel()
    try:
        assert panel.control_discovery_model.rowCount() == 2
        assert panel.study_discovery_model.rowCount() == 2
        assert panel.exit_framework_model.rowCount() == 2
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# The sort is the lower bound - CLAUDE.md's headline rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attribute", ["control_discovery_rows", "study_discovery_rows"])
def test_the_sort_key_is_the_wilson_lower_bound(panel_module, tmp_path, attribute):
    _write(tmp_path / CONTROL_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    _write(tmp_path / STUDY_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    panel = panel_module.SetupTrackerPanel()
    try:
        rows = getattr(panel, attribute)
        assert [row["setup_family"] for row in rows] == [
            "fat_and_good",
            "thin_but_perfect",
        ], "sorted by the raw rate, not by the bound"
    finally:
        panel.deleteLater()


def test_the_ranking_function_is_pure_and_ranks_by_the_bound(panel_module):
    ranked = panel_module._rank_discovery_rows([THIN, FAT])
    assert [row["setup_family"] for row in ranked] == ["fat_and_good", "thin_but_perfect"]
    # A row with no bound has nothing to rank on and sorts LAST rather than
    # being treated as a bound of zero.
    ungraded = _discovery_row("never_graded", n=0, wins=0, win_rate="", win_rate_lb="", avg_r="")
    ranked = panel_module._rank_discovery_rows([ungraded, THIN, FAT])
    assert ranked[-1]["setup_family"] == "never_graded"


def test_the_framework_rows_rank_by_the_bound_too(panel_module):
    ranked = panel_module._rank_exit_frameworks(list(reversed(FRAMEWORK_ROWS)))
    assert ranked[0]["framework_family"] == "baseline"
    assert ranked[1]["framework_family"] == "comparison_apr2026"


# ---------------------------------------------------------------------------
# The population sentence, and the empty state
# ---------------------------------------------------------------------------


def test_each_tab_says_what_its_population_is(panel_module, tmp_path):
    _write(tmp_path / CONTROL_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    _write(tmp_path / STUDY_CSV, DISCOVERY_FIELDS, DISCOVERY_ROWS)
    _write(tmp_path / FRAMEWORK_CSV, FRAMEWORK_FIELDS, FRAMEWORK_ROWS)
    panel = panel_module.SetupTrackerPanel()
    try:
        control = panel.control_discovery_status_label.text().lower()
        assert "reject" in control
        assert "92 " in control or "92," in control or "92" in control  # 2 + 90 graded
        study = panel.study_discovery_status_label.text().lower()
        assert "study" in study or "never" in study
        framework = panel.exit_framework_status_label.text().lower()
        assert "experimental" in framework
    finally:
        panel.deleteLater()


@pytest.mark.parametrize(
    "label_attribute,sentence_attribute",
    [
        ("control_discovery_status_label", "CONTROL_DISCOVERY_NO_EXPORT_SENTENCE"),
        ("study_discovery_status_label", "STUDY_DISCOVERY_NO_EXPORT_SENTENCE"),
        ("exit_framework_status_label", "EXIT_FRAMEWORK_NO_EXPORT_SENTENCE"),
    ],
)
def test_an_absent_export_renders_the_empty_state_sentence(
    panel_module, tmp_path, label_attribute, sentence_attribute
):
    """Never a blank table: a page with no words on it reads as 'no edge'."""
    panel = panel_module.SetupTrackerPanel()
    try:
        expected = getattr(panel_module, sentence_attribute)
        assert getattr(panel, label_attribute).text() == expected
        assert expected.strip()
    finally:
        panel.deleteLater()


def test_the_experimental_label_is_in_the_framework_row(panel_module, tmp_path):
    _write(tmp_path / FRAMEWORK_CSV, FRAMEWORK_FIELDS, FRAMEWORK_ROWS)
    panel = panel_module.SetupTrackerPanel()
    try:
        keys = {key for key, _label in panel_module.EXIT_FRAMEWORK_COLUMNS}
        assert "experimental" in keys
        assert "framework_family" in keys
        experimental = [
            row
            for row in panel.exit_framework_rows
            if str(row.get("framework_family")) == "comparison_apr2026"
        ]
        assert experimental and str(experimental[0]["experimental"]).lower().startswith("t")
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Structural: one path per file, and the page still writes nothing
# ---------------------------------------------------------------------------


def test_the_tabs_read_the_files_the_exporter_writes():
    from master_avwap_lib import legacy
    from ui.panels import setup_tracker_panel

    pairs = (
        ("CONTROL_DISCOVERY_STATS_FILE", "master_avwap_control_discovery.csv"),
        ("STUDY_DISCOVERY_STATS_FILE", "master_avwap_study_discovery.csv"),
        ("EXIT_FRAMEWORK_STATS_FILE", "master_avwap_exit_framework_stats.csv"),
    )
    for name, filename in pairs:
        assert (
            Path(getattr(setup_tracker_panel, name)).name
            == Path(getattr(legacy, name)).name
            == filename
        )


def test_the_new_tabs_never_write_anything():
    import ast

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ui"
        / "panels"
        / "setup_tracker_panel.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    writers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"write_text", "to_csv", "write_bytes"}
    ]
    assert writers == []


def test_the_new_tables_carry_the_ten_row_floor(panel_module, tmp_path):
    panel = panel_module.SetupTrackerPanel()
    try:
        for table in (
            panel.control_discovery_table,
            panel.study_discovery_table,
            panel.exit_framework_table,
        ):
            assert table.minimumHeight() >= panel_module.TABLE_TEN_ROWS_PX
    finally:
        panel.deleteLater()
