"""S13 exit review: three exit models per family x side, display only.

`exit_model_review` computes the tracker's R, a 1 ATR stop held 10 sessions and a
1 ATR trail from the same session-horizon outcomes; the Setup Tracker prints the
worker's cells and sentence and computes nothing.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import exit_model_review as emr  # noqa: E402


def _rows(symbol, side, scan_day, family, returns, *, entry_close=100.0, unmeasured=()):
    out = []
    for horizon, side_return in zip(emr.CHECKPOINTS, returns, strict=True):
        measured = horizon not in unmeasured
        out.append(
            {
                "symbol": symbol,
                "side": side,
                "scan_date": scan_day,
                "horizon_sessions": str(horizon),
                "entry_close": str(entry_close),
                "side_return_pct": str(side_return) if measured else "",
                "measured": "True" if measured else "False",
                "maturity": "mature" if measured else "immature",
                "outcome_kind": emr.OUTCOME_KIND,
                "setup_family": family,
            }
        )
    return out


FAMILY = "avwap_retest_followthrough"


def _fixture_rows():
    rows = []
    # Runner that gives back: ATR moves 0.5, 2.0, 0.75, 1.5 -> trail exits at 0.75.
    rows += _rows("RUN", "SHORT", "2026-09-01", FAMILY, [1.0, 4.0, 1.5, 3.0])
    # Stop first, then runs: -1.5 ATR on session 1 stops both models out.
    rows += _rows("STP", "SHORT", "2026-09-01", FAMILY, [-3.0, 2.0, 6.0, 8.0])
    # All closes measured but no ATR known on the scan date: unknown.
    rows += _rows("NOA", "SHORT", "2026-09-01", FAMILY, [1.0, 1.0, 1.0, 1.0])
    # Session 10 not complete yet: unknown.
    rows += _rows("IMM", "SHORT", "2026-09-01", FAMILY, [1.0, 1.0, 1.0, 0.0], unmeasured=(10,))
    return rows


_ATRS = {
    ("RUN", "2026-09-01"): 2.0,
    ("STP", "2026-09-01"): 2.0,
    ("IMM", "2026-09-01"): 2.0,
    # Known only the day AFTER the scan: never used for a 09-01 entry.
    ("NOA", "2026-09-02"): 2.0,
}


def test_exit_model_r_stop_only_and_trail_paths():
    runner = emr.exit_model_r({1: 0.5, 3: 2.0, 5: 0.75, 10: 1.5})
    assert runner == {"stop_only": 1.5, "trail": 0.75}
    stop_first = emr.exit_model_r({1: -1.5, 3: 1.0, 5: 3.0, 10: 4.0})
    assert stop_first == {"stop_only": -1.5, "trail": -1.5}
    assert emr.exit_model_r({1: 0.5, 3: None, 5: 1.0, 10: 1.0}) is None


def test_review_covers_all_three_models_and_counts_unknowns():
    currents = {("RUN", "SHORT", "2026-09-01"): -0.2}
    summary = emr.review(_fixture_rows(), _ATRS, currents)
    (cell,) = summary["cells"]
    assert cell["family"] == FAMILY and cell["side"] == "SHORT"
    assert cell["n"] == 2
    assert cell["stop_only_r"] == pytest.approx((1.5 - 1.5) / 2)
    assert cell["trail_r"] == pytest.approx((0.75 - 1.5) / 2)
    assert cell["current_r"] == pytest.approx(-0.2) and cell["current_n"] == 1
    assert cell["unknown_no_atr"] == 1
    assert cell["unknown_not_measured"] == 1
    sentence = emr.review_sentence(summary)
    assert "2 entries measured" in sentence and "1 with no ATR" in sentence


def test_atr_index_is_the_scan_days_last_row_and_skips_bad_values():
    rows = [
        {"run_date": "2026-09-01", "symbol": "aaa", "atr20": "1.0"},
        {"run_date": "2026-09-01", "symbol": "AAA", "atr20": "1.5"},
        {"run_date": "2026-09-01", "symbol": "BBB", "atr20": "nan"},
        {"run_date": "2026-09-01", "symbol": "CCC", "atr20": "0"},
    ]
    assert emr.atr_index(rows) == {("AAA", "2026-09-01"): 1.5}


def test_current_r_index_averages_setups_of_one_session():
    setups = {
        "a": {"symbol": "X", "side": "SHORT", "scan_date": "2026-09-01",
              "_scoring_outcome_summary": {"avg_total_r": 0.4}},
        "b": {"symbol": "X", "side": "SHORT", "scan_date": "2026-09-01",
              "_scoring_outcome_summary": {"avg_total_r": -0.2}},
        "c": {"symbol": "Y", "side": "LONG", "scan_date": "2026-09-01",
              "_scoring_outcome_summary": {"avg_total_r": None}},
    }
    assert emr.current_r_index(setups) == {("X", "SHORT", "2026-09-01"): pytest.approx(0.1)}


def test_no_rows_is_the_no_data_sentence():
    assert emr.review((), {}, {})["cells"] == []
    assert emr.review_sentence({}) == emr.NO_DATA_SENTENCE


def test_the_worker_reader_is_cached_and_unknown_without_a_file(tmp_path, monkeypatch):
    from ui.services import working_lately_service as service

    horizon = tmp_path / "horizon.csv"
    features = tmp_path / "features.csv"
    monkeypatch.setattr(service, "_horizon_outcomes_path", lambda: horizon)
    monkeypatch.setattr(service, "_features_history_path", lambda: features)
    monkeypatch.setattr(service, "_scoring_snapshot_path", lambda: tmp_path / "snap.json")
    service._LOOKING_BACK_CACHE.clear()
    try:
        assert service.read_exit_model_review()["cells"] == []

        rows = _fixture_rows()
        with horizon.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with features.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run_date", "symbol", "atr20", "other"])
            for (symbol, day), atr in _ATRS.items():
                writer.writerow([day, symbol, atr, "x"])
        calls = []
        real = emr.review
        monkeypatch.setattr(emr, "review", lambda *a, **k: calls.append(1) or real(*a, **k))
        first = service.read_exit_model_review()
        second = service.read_exit_model_review()
        assert first["cells"][0]["n"] == 2 and first["cells"][0]["current_n"] == 0
        assert second is first and len(calls) == 1
    finally:
        service._LOOKING_BACK_CACHE.clear()


# ---------------------------------------------------------------------------
# Setup Tracker: the page only formats
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _boom(*_args, **_kwargs):
    raise AssertionError("the page must not read or compute this")


_CELLS = [
    {"family": FAMILY, "side": "SHORT", "n": 2, "current_r": -0.2, "current_n": 1,
     "stop_only_r": 0.0, "trail_r": -0.375, "unknown_no_atr": 1, "unknown_not_measured": 1},
]


@pytest.mark.qt
def test_setup_tracker_prints_the_exit_models_it_was_handed(qapp, monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    panel = module.SetupTrackerPanel()
    monkeypatch.setattr(emr, "review", _boom)
    monkeypatch.setattr(emr, "exit_model_r", _boom)
    monkeypatch.setattr(working_lately_service, "read_exit_model_review", _boom)
    try:
        panel._on_exports_loaded(
            {
                "signatures": {"exit_models": "sig"},
                "ranked": {"exit_models": [dict(c) for c in _CELLS]},
                "raw": {},
                "exit_model_sentence": "Scan dates a to b: 2 entries measured.",
            }
        )
        assert panel.exit_model_status_label.text() == "Scan dates a to b: 2 entries measured."
        model = panel.exit_model_model
        assert model.rowCount() == 1
        columns = [key for key, _label in module.EXIT_MODEL_COLUMNS]
        shown = {
            key: model.data(model.index(0, i)) for i, key in enumerate(columns)
        }
        assert shown["family"] == FAMILY
        assert shown["current_r"] == "-0.20"
        assert shown["stop_only_r"] == "+0.00"
        assert shown["trail_r"] == "-0.38"
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "min_closed": 1})
        assert panel.exit_model_status_label.text() == emr.NO_DATA_SENTENCE
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_setup_tracker_worker_read_carries_the_cells(monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    summary = {"cells": [dict(c) for c in _CELLS], "first": "2026-09-01", "last": "2026-09-01"}
    monkeypatch.setattr(working_lately_service, "read_exit_model_review", lambda: summary)
    data = module._read_tracker_exports(1)
    assert data["ranked"]["exit_models"] == _CELLS
    assert data["exit_model_sentence"] == emr.review_sentence(summary)

    monkeypatch.setattr(working_lately_service, "read_exit_model_review", _boom)
    data = module._read_tracker_exports(1)
    assert data["ranked"]["exit_models"] == []
    assert data["exit_model_sentence"] == emr.NO_DATA_SENTENCE
