"""Packet M1.3 - the Band variant view says what it actually measured.

`master_avwap_band_variant_stats.csv` carried 40 rows and 11,292 setups on
2026-09-05 with `n_variant = 0` on every one of them. The table rendered: a
family, a side, a bucket, a champion R - and blank challenger cells that read as
"the challenger did nothing special" rather than "the challenger was never
computed". An empty comparison must never again look like a comparison.

Two halves:

* the stats builder names the top reason the unmeasured setups gave, so the
  panel can say it without reading the 1.1 GB tracker JSON;
* the panel renders ONE sentence above the table, built from the file's own
  `n_variant` / `n` / `n_variant_unmeasured` sums.

Display only. Nothing here scores, ranks, gates or writes.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402

SHORT_WINDOW_REASON = "fewer than the lookback's closes before this bar"
PLACEHOLDER = "no band-variant block on the scan entry"


# --- the stats builder ----------------------------------------------------


def _setup(setup_id: str, *, variant_reason: str | None, measured: bool) -> dict:
    """One tracker setup with a champion scenario and, optionally, a challenger."""
    scenarios = {
        "champ": {
            "scenario_id": "champ",
            "tradeable": True,
            "experimental": False,
            "stop_reference_label": "LOWER_1",
            "stop_source_type": "current_anchor",
            "stop_reference_level": 95.0,
            "exit_template_id": "full_band2",
            "status": "STOPPED",
            "total_r": -1.0,
        }
    }
    if measured:
        scenarios["variant"] = {
            "scenario_id": "variant",
            "tradeable": True,
            "experimental": False,
            "stop_reference_label": "VARIANT_LOWER_1",
            "stop_source_type": legacy.BAND_VARIANT_STOP_SOURCE,
            "stop_reference_level": 92.0,
            "exit_template_id": "full_band2",
            "status": "OPEN",
            "total_r": 0.4,
        }
    return {
        "setup_id": setup_id,
        "symbol": "TEST",
        "side": "LONG",
        "setup_family": "top_pattern",
        "priority_bucket": "favorite_setup",
        "entry_price": 100.0,
        "entry_feature_snapshot": {"atr20": 2.0},
        "current_anchor_variant": {
            "formula_version": "" if variant_reason else "avwap_bands_oneoption_bb20_v1",
            "date": "2026-08-10",
            "vwap": 100.0,
            "stdev": None if variant_reason else 4.0,
            "bands": {} if variant_reason else {"LOWER_1": 96.0},
            "reason": variant_reason or "",
        },
        "scenarios": scenarios,
    }


def test_the_stats_row_names_the_top_unmeasured_reason():
    setups = {
        "a": _setup("a", variant_reason=SHORT_WINDOW_REASON, measured=False),
        "b": _setup("b", variant_reason=SHORT_WINDOW_REASON, measured=False),
        "c": _setup("c", variant_reason=PLACEHOLDER, measured=False),
        "d": _setup("d", variant_reason=None, measured=True),
    }
    rows = legacy.build_band_variant_stats_rows(setups)

    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 4
    assert row["n_variant"] == 1
    assert row["n_variant_unmeasured"] == 3
    assert row["top_unmeasured_reason"] == SHORT_WINDOW_REASON
    assert "top_unmeasured_reason" in legacy.BAND_VARIANT_STATS_COLUMNS


def test_a_fully_measured_group_names_no_reason():
    setups = {"d": _setup("d", variant_reason=None, measured=True)}
    row = legacy.build_band_variant_stats_rows(setups)[0]

    assert row["n_variant"] == 1
    assert row["n_variant_unmeasured"] == 0
    assert row["top_unmeasured_reason"] == ""


def test_an_open_challenger_scenario_still_counts_as_measured():
    """`n_variant` counts a paired scenario; the closed counts are separate.

    A comparison that only counted closed scenarios would report zero coverage
    for the first weeks of forward accrual, which is exactly the blindness this
    packet is fixing.
    """
    row = legacy.build_band_variant_stats_rows(
        {"d": _setup("d", variant_reason=None, measured=True)}
    )[0]

    assert row["n_variant"] == 1
    assert row["n_closed_variant"] == 0


# --- the panel sentence ---------------------------------------------------


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _csv_row(**overrides) -> dict:
    row = {column: "" for column in legacy.BAND_VARIANT_STATS_COLUMNS}
    row.update(
        {
            "setup_family": "top_pattern",
            "side": "LONG",
            "priority_bucket": "favorite_setup",
            "exit_template_id": "full_band2",
        }
    )
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(legacy.BAND_VARIANT_STATS_COLUMNS))
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


def test_the_sentence_is_a_pure_function_of_the_rows(panel_module):
    sentence = panel_module.band_variant_coverage_sentence(
        [
            _csv_row(n="10", n_variant="0", n_variant_unmeasured="10",
                     top_unmeasured_reason=PLACEHOLDER),
            _csv_row(n="5", n_variant="0", n_variant_unmeasured="5",
                     top_unmeasured_reason=SHORT_WINDOW_REASON),
        ]
    )
    assert sentence == (
        f"Measured 0 of 15 setups (15 unmeasured: {PLACEHOLDER})."
    )


def test_a_fully_measured_file_drops_the_unmeasured_clause(panel_module):
    sentence = panel_module.band_variant_coverage_sentence(
        [_csv_row(n="8", n_variant="8", n_variant_unmeasured="0")]
    )
    assert sentence == "Measured 8 of 8 setups."


def test_an_absent_export_says_so_rather_than_measuring_nothing(panel_module):
    assert panel_module.band_variant_coverage_sentence([]) == (
        "No band-variant comparison has been written yet."
    )


def test_the_panel_renders_the_sentence_above_the_table(panel_module, tmp_path):
    """An empty comparison must announce itself on the screen the trader reads."""
    _write(
        tmp_path / "band_variant.csv",
        [
            _csv_row(n="10", n_variant="0", n_variant_unmeasured="10",
                     top_unmeasured_reason=PLACEHOLDER),
            _csv_row(side="SHORT", n="5", n_variant="0", n_variant_unmeasured="5",
                     top_unmeasured_reason=PLACEHOLDER),
        ],
    )
    panel = panel_module.SetupTrackerPanel()
    try:
        assert panel.band_variant_status_label.text() == (
            f"Measured 0 of 15 setups (15 unmeasured: {PLACEHOLDER})."
        )
    finally:
        panel.deleteLater()
