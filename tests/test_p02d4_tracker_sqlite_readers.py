"""P0-2 2d(4): compression calibration reads the tracker's SQLite mirror when it is fresh.

Parity: the report built from the mirror equals the report built by streaming
the JSON. A stale or missing mirror falls back to the JSON and logs why.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import project_paths  # noqa: E402
from tracker_store import TrackerStore, file_stamp, load_fresh_projection  # noqa: E402

from test_pct3_calibration_accounting import SINCE, scratch  # noqa: E402,F401 - fixture


def _mirror(json_path: Path, db_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    TrackerStore(db_path).save_payload(payload, source_path=json_path, source_stamp=file_stamp(json_path))


def test_projection_reads_several_sections_in_json_order(tmp_path):
    tracker = tmp_path / "tracker.json"
    tracker.write_text(
        json.dumps(
            {
                "setups": {"b": {"symbol": "B", "x": 2}, "a": {"symbol": "A", "x": 1}, "bad": 3},
                "control_setups": {"c": {"symbol": "C"}},
                "study_setups": {"d": {"symbol": "D", "x": 0.1234567890123456789}},
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "tracker.sqlite"
    _mirror(tracker, db)
    rows, reason = load_fresh_projection(
        tracker, ("symbol", "x"), sections=("setups", "control_setups", "study_setups"), db_path=db
    )
    assert reason == ""
    assert rows == [
        {"symbol": "B", "x": 2},
        {"symbol": "A", "x": 1},
        {"symbol": "C"},
        {"symbol": "D", "x": 0.1234567890123456789},
    ]


def _add_control_record(tracker: Path) -> None:
    """A second section, so the SQLite path must read past `setups` like the stream does."""
    payload = json.loads(tracker.read_text(encoding="utf-8"))
    first = next(iter(payload["setups"].values()))
    twin = dict(first)
    twin["setup_id"] = "control:" + str(first.get("setup_id"))
    twin["anchor_date"] = "2026-08-01"
    payload["control_setups"] = {twin["setup_id"]: twin}
    tracker.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def test_compression_rows_from_sqlite_equal_rows_from_the_json(scratch, tmp_path, monkeypatch):  # noqa: F811
    import compression_calibration

    tracker = Path(project_paths.MASTER_AVWAP_SETUP_TRACKER_FILE)
    _add_control_record(tracker)
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_SETUP_TRACKER_DB", tmp_path / "none.sqlite", raising=False)
    json_rows, json_summary = compression_calibration.build_rows(since=SINCE)
    assert json_rows, "the fixture must produce rows for the parity to mean anything"

    db = tmp_path / "tracker.sqlite"
    _mirror(tracker, db)
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_SETUP_TRACKER_DB", db, raising=False)

    def _no_stream(*_args, **_kwargs):
        raise AssertionError("the JSON was streamed although the mirror is fresh")

    monkeypatch.setattr(compression_calibration, "iter_tracker_records", _no_stream)
    db_rows, db_summary = compression_calibration.build_rows(since=SINCE)
    assert db_rows == json_rows
    assert db_summary == json_summary


def test_a_stale_mirror_falls_back_to_the_json_and_says_why(scratch, tmp_path, monkeypatch, caplog):  # noqa: F811
    import compression_calibration

    tracker = Path(project_paths.MASTER_AVWAP_SETUP_TRACKER_FILE)
    db = tmp_path / "tracker.sqlite"
    _mirror(tracker, db)
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_SETUP_TRACKER_DB", db, raising=False)
    expected_rows, _ = compression_calibration.build_rows(since=SINCE)
    tracker.write_text(tracker.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        rows, _ = compression_calibration.build_rows(since=SINCE)
    assert rows == expected_rows
    assert "streamed the setup tracker JSON" in caplog.text
    assert "changed after the last mirror" in caplog.text


@pytest.mark.parametrize("fields", [("symbol",), ("symbol", "anchor_date")])
def test_single_section_callers_are_unchanged(tmp_path, fields):
    tracker = tmp_path / "tracker.json"
    tracker.write_text(
        json.dumps({"setups": {"a": {"symbol": "A", "anchor_date": "2026-01-02"}}, "control_setups": {"c": {"symbol": "C"}}}),
        encoding="utf-8",
    )
    db = tmp_path / "tracker.sqlite"
    _mirror(tracker, db)
    rows, reason = load_fresh_projection(tracker, fields, db_path=db)
    assert reason == ""
    assert rows == [{name: {"symbol": "A", "anchor_date": "2026-01-02"}[name] for name in fields}]
