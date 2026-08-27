"""Phase 0.10 B-2 item 5 - the Setup Tracker's "Band variant" section.

A pure CSV reader, like every other section on this page. It shows the champion
and the challenger side by side per (family, side, bucket) and it shows an
HONEST EMPTY STATE when the export is absent: a page that renders zeros for a
file that does not exist teaches the trader to read noise as evidence.

Read on the Qt thread, matching this panel's existing pattern - `refresh()`
already calls `_load_csv_rows` for nine other exports inline. `setup_tracker_panel`
is the first of the eight panels listed under plan.md Phase 0.9 G-P2.3 that still
read on the Qt thread; moving them is that packet's work, and doing it here for
one section alone would leave the page half on each thread.
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


ROWS = [
    {
        "setup_family": "top_pattern",
        "side": "LONG",
        "priority_bucket": "favorite_setup",
        "exit_template_id": "full_band2",
        "n": "12",
        "n_variant": "10",
        "n_variant_unmeasured": "2",
        "n_closed_champion": "9",
        "n_closed_variant": "8",
        "avg_total_r_champion": "0.31",
        "avg_total_r_variant": "0.44",
        "stop_out_rate_champion": "0.44",
        "stop_out_rate_variant": "0.25",
        "target_hit_rate_champion": "0.33",
        "target_hit_rate_variant": "0.50",
        "mean_stop_distance_atr_champion": "1.10",
        "mean_stop_distance_atr_variant": "2.40",
    },
    {
        "setup_family": "post_earnings_candle_break",
        "side": "SHORT",
        "priority_bucket": "near_favorite_zone",
        "exit_template_id": "full_band2",
        "n": "3",
        "n_variant": "0",
        "n_variant_unmeasured": "3",
        "n_closed_champion": "1",
        "n_closed_variant": "0",
        "avg_total_r_champion": "-0.20",
        # Blank, not zero: nothing measured this cell.
        "avg_total_r_variant": "",
        "stop_out_rate_champion": "1.0",
        "stop_out_rate_variant": "",
        "target_hit_rate_champion": "0.0",
        "target_hit_rate_variant": "",
        "mean_stop_distance_atr_champion": "0.9",
        "mean_stop_distance_atr_variant": "",
    },
]


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def panel_module(monkeypatch, tmp_path):
    if _qt_app() is None:
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    monkeypatch.setattr(
        setup_tracker_panel, "BAND_VARIANT_STATS_FILE", tmp_path / "band_variant.csv"
    )
    return setup_tracker_panel


def test_the_section_renders_the_export(panel_module, tmp_path):
    _write(tmp_path / "band_variant.csv", ROWS)
    panel = panel_module.SetupTrackerPanel()
    try:
        rows = panel.band_variant_rows
        assert len(rows) == 2
        # Best challenger edge first, so the interesting row is the top one.
        assert rows[0]["setup_family"] == "top_pattern"
        assert panel.band_variant_model.rowCount() == 2
        labels = [label for _key, label in panel_module.BAND_VARIANT_COLUMNS]
        assert any("Champ" in label for label in labels)
        assert any("Variant" in label for label in labels)
    finally:
        panel.deleteLater()


def test_a_blank_cell_stays_blank(panel_module, tmp_path):
    """An unmeasured cell must not become 0.0 on the way to the screen."""
    _write(tmp_path / "band_variant.csv", ROWS)
    panel = panel_module.SetupTrackerPanel()
    try:
        short_row = next(row for row in panel.band_variant_rows if row["side"] == "SHORT")
        assert short_row["avg_total_r_variant"] == ""
        assert short_row["stop_out_rate_variant"] == ""
        assert short_row["n_variant_unmeasured"] == "3"
    finally:
        panel.deleteLater()


def test_an_absent_export_shows_an_honest_empty_state(panel_module, tmp_path):
    assert not (tmp_path / "band_variant.csv").exists()
    panel = panel_module.SetupTrackerPanel()
    try:
        assert panel.band_variant_rows == []
        assert panel.band_variant_model.rowCount() == 0
    finally:
        panel.deleteLater()


def test_the_section_reads_the_file_the_exporter_writes():
    """One path, named once on each side - not two strings that happen to match."""
    from master_avwap_lib import legacy
    from ui.panels import setup_tracker_panel

    assert (
        Path(setup_tracker_panel.BAND_VARIANT_STATS_FILE).name
        == Path(legacy.SETUP_BAND_VARIANT_STATS_FILE).name
        == "master_avwap_band_variant_stats.csv"
    )


def test_the_section_never_scores_anything():
    """Shadow evidence: the panel reads a CSV and writes nothing at all."""
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
        and node.attr in {"write_text", "to_csv", "open"}
        and "band_variant" in ast.unparse(node).lower()
    ]
    assert writers == []
