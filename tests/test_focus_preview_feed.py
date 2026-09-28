"""Daytime scans publish a focus-feed preview the swing list reads before the close."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ui.services import data_feed  # noqa: E402

REPORT_AT = datetime(2026, 9, 29, 7, 50, 12)


def _write_report(path, generated_at=REPORT_AT):
    path.write_text(
        "Master AVWAP priority setups\n"
        f"Generated at {generated_at:%Y-%m-%d %H:%M:%S}\n\n"
        "TYRA SHORT ExpR=+0.10R score=60 family=avwap band bounce bucket=favorite_setup\n",
        encoding="utf-8",
    )


def _write_focus(path, symbol, generated_at):
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at.isoformat(timespec="seconds"),
                "run_date": generated_at.date().isoformat(),
                "high_conviction": [
                    {"symbol": symbol, "side": "SHORT", "priority_bucket": "high_conviction", "priority_score": 70}
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def feed(tmp_path, monkeypatch):
    paths = {
        "report": tmp_path / "priority.txt",
        "focus": tmp_path / "focus.json",
        "preview": tmp_path / "focus_preview.json",
    }
    monkeypatch.setattr(data_feed, "MASTER_AVWAP_PRIORITY_SETUPS_FILE", paths["report"])
    monkeypatch.setattr(data_feed, "MASTER_AVWAP_FOCUS_FILE", paths["focus"])
    monkeypatch.setattr(data_feed, "MASTER_AVWAP_FOCUS_PREVIEW_FILE", paths["preview"])
    monkeypatch.setattr(data_feed, "read_priority_report_date", lambda path=None: REPORT_AT.date().isoformat())
    monkeypatch.setattr(
        data_feed, "load_setup_rows_from_priority_report", lambda path=None: _real_report_loader(paths["report"])
    )
    monkeypatch.setattr(data_feed, "cached_points_projection", lambda rows: None)
    monkeypatch.setattr(data_feed, "enrich_report_rows_with_cached_scan", lambda rows, projection: None)
    monkeypatch.setattr(data_feed, "enrich_setup_rows_for_display", lambda rows, supplemental_rows=None: None)
    _write_report(paths["report"])
    _write_focus(paths["focus"], "IBM", REPORT_AT - timedelta(days=1))
    return paths


_real_report_loader = data_feed.load_setup_rows_from_priority_report


def _symbols(meta):
    return [row.symbol for row in meta["rows"]]


def test_daytime_preview_from_the_newest_report_wins(feed):
    _write_focus(feed["preview"], "TDOC", REPORT_AT + timedelta(seconds=4))

    meta = data_feed.load_latest_setup_rows_with_meta()

    assert meta["source"] == "focus_preview"
    assert _symbols(meta) == ["TDOC"]
    assert meta["data_date"] == "2026-09-29"


def test_preview_older_than_the_report_is_ignored(feed):
    _write_focus(feed["preview"], "TDOC", REPORT_AT - timedelta(minutes=2))

    meta = data_feed.load_latest_setup_rows_with_meta()

    assert meta["source"] == "priority_report"
    assert _symbols(meta) == ["TYRA"]


def test_preview_long_after_the_report_belongs_to_another_scan(feed):
    _write_focus(feed["preview"], "TDOC", REPORT_AT + timedelta(minutes=30))

    assert data_feed.load_latest_setup_rows_with_meta()["source"] == "priority_report"


def test_close_scan_focus_feed_beats_the_preview(feed):
    _write_focus(feed["focus"], "IBM", REPORT_AT + timedelta(seconds=30))
    _write_focus(feed["preview"], "TDOC", REPORT_AT + timedelta(seconds=4))

    meta = data_feed.load_latest_setup_rows_with_meta()

    assert meta["source"] == "focus"
    assert _symbols(meta) == ["IBM"]


def test_callers_can_skip_the_preview(feed):
    _write_focus(feed["preview"], "TDOC", REPORT_AT + timedelta(seconds=4))

    meta = data_feed.load_latest_setup_rows_with_meta(include_preview=False)

    assert meta["source"] == "priority_report"


def test_near_hod_adds_do_not_read_the_preview(monkeypatch):
    from ui.services.autopilot_service import AutopilotService

    seen = {}

    def fake_feed(*, include_preview=True):
        seen["include_preview"] = include_preview
        return {"rows": []}

    monkeypatch.setattr(AutopilotService, "_load_swing_feed", staticmethod(fake_feed))

    assert AutopilotService._load_swing_rows() == []
    assert seen == {"include_preview": False}


def test_runner_writes_the_preview_and_survives_a_failed_write(tmp_path, monkeypatch):
    from master_avwap_lib import runner

    calls = []
    monkeypatch.setattr(
        runner, "write_master_avwap_focus_feed",
        lambda path, rows, ai_state, study_rows=None: calls.append((path, rows, study_rows)),
    )
    target = tmp_path / "preview.json"

    assert runner.write_focus_preview([{"symbol": "TDOC"}], {}, [], path=target) is True
    assert calls == [(target, [{"symbol": "TDOC"}], [])]

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_master_avwap_focus_feed", boom)
    assert runner.write_focus_preview([], {}, path=target) is False


def test_preview_file_is_not_the_close_only_focus_feed():
    import project_paths

    assert project_paths.MASTER_AVWAP_FOCUS_PREVIEW_FILE != project_paths.MASTER_AVWAP_FOCUS_FILE
    assert project_paths.MASTER_AVWAP_FOCUS_PREVIEW_FILE.parent == project_paths.MASTER_AVWAP_FOCUS_FILE.parent
