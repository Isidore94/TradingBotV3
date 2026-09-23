"""Point-in-time history for the Day Recap: setup grades and the opening regime.

Pins: every grades write also lands a dated, tz-aware snapshot; unchanged
content adds no line; `grades_as_of` never returns a snapshot written after
`when`; a failed history write never breaks the grades write. The opening
regime keeps the session's first read and its directional anchor per day.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = timezone(timedelta(hours=-4))


def _grades(as_of: str, grade: str = "A") -> dict:
    return {
        "schema": "setup_grades_v1",
        "as_of": as_of,
        "rules": "r",
        "swing": [{"key": "pullback|long", "grade": grade}],
        "daytrade": [],
    }


# --- setup grades history -------------------------------------------------


def test_append_snapshot_writes_dated_tz_aware_line_and_skips_unchanged(tmp_path):
    import setup_grades_history as h

    folder = tmp_path / "hist"
    t0 = datetime(2026, 9, 23, 10, 0, tzinfo=ET)
    assert h.append_snapshot(_grades("2026-09-22"), history_dir=folder, now=t0) is True
    assert h.append_snapshot(_grades("2026-09-22"), history_dir=folder, now=t0 + timedelta(minutes=30)) is False
    assert h.append_snapshot(_grades("2026-09-22", "B"), history_dir=folder, now=t0 + timedelta(hours=1)) is True

    files = sorted(p.name for p in folder.iterdir())
    assert files == [f"{t0.astimezone().date().isoformat()}.jsonl"]
    rows = [json.loads(line) for line in (folder / files[0]).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert datetime.fromisoformat(row["written_at"]).tzinfo is not None
    assert rows[1]["grades"]["swing"][0]["grade"] == "B"


def test_grades_as_of_returns_newest_at_or_before_and_never_after(tmp_path):
    import setup_grades_history as h

    folder = tmp_path / "hist"
    day1 = datetime(2026, 9, 21, 12, 0, tzinfo=ET)
    day2 = datetime(2026, 9, 22, 12, 0, tzinfo=ET)
    h.append_snapshot(_grades("2026-09-18", "A"), history_dir=folder, now=day1)
    h.append_snapshot(_grades("2026-09-21", "C"), history_dir=folder, now=day2)

    assert h.grades_as_of(day1 - timedelta(seconds=1), history_dir=folder) is None
    at_day1 = h.grades_as_of(day1, history_dir=folder)
    assert at_day1["swing"][0]["grade"] == "A"
    assert datetime.fromisoformat(at_day1["written_at"]) == day1
    # Between the two writes: still the first, never the later one.
    assert h.grades_as_of(day2 - timedelta(minutes=1), history_dir=folder)["swing"][0]["grade"] == "A"
    assert h.grades_as_of(day2 + timedelta(days=3), history_dir=folder)["swing"][0]["grade"] == "C"
    assert h.grades_as_of(day2, history_dir=tmp_path / "missing") is None


def test_unchanged_content_on_a_new_day_adds_no_line_and_is_still_found(tmp_path):
    import setup_grades_history as h

    folder = tmp_path / "hist"
    day1 = datetime(2026, 9, 21, 12, 0, tzinfo=ET)
    h.append_snapshot(_grades("2026-09-18"), history_dir=folder, now=day1)
    assert h.append_snapshot(_grades("2026-09-18"), history_dir=folder, now=day1 + timedelta(days=1)) is False
    assert h.grades_as_of(day1 + timedelta(days=2), history_dir=folder)["as_of"] == "2026-09-18"


def test_old_history_files_are_pruned_but_400_sessions_are_kept(tmp_path):
    import setup_grades_history as h

    assert h.KEEP_DAYS >= 580  # 400 sessions is about 580 calendar days
    folder = tmp_path / "hist"
    folder.mkdir()
    today = datetime(2026, 9, 23, 10, 0, tzinfo=ET)
    old = folder / f"{(today.date() - timedelta(days=h.KEEP_DAYS + 5)).isoformat()}.jsonl"
    kept = folder / f"{(today.date() - timedelta(days=h.KEEP_DAYS - 5)).isoformat()}.jsonl"
    old.write_text("", encoding="utf-8")
    kept.write_text("", encoding="utf-8")
    h.append_snapshot(_grades("2026-09-22"), history_dir=folder, now=today)
    assert not old.exists()
    assert kept.exists()


def _service(tmp_path: Path):
    from ui.services.working_lately_service import WorkingLatelyService

    return WorkingLatelyService(store_dir=tmp_path / "working_lately")


def test_service_grades_write_also_appends_history(tmp_path):
    import setup_grades_history as h

    service = _service(tmp_path)
    service._write_grades(_grades("2026-09-22"))
    service._write_grades(_grades("2026-09-22"))

    history = service.store_dir / "setup_grades_history"
    rows = [line for p in history.glob("*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    found = h.grades_as_of(datetime.now().astimezone(), history_dir=history)
    assert found["as_of"] == "2026-09-22"


def test_failed_history_write_never_breaks_the_grades_write(tmp_path, monkeypatch, caplog):
    import setup_grades_history as h

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(h, "append_snapshot", boom)
    service = _service(tmp_path)
    with caplog.at_level("WARNING"):
        service._write_grades(_grades("2026-09-22"))
    latest = json.loads((service.store_dir / "setup_grades_latest.json").read_text(encoding="utf-8"))
    assert latest["as_of"] == "2026-09-22"
    assert "grades history" in caplog.text.lower()


def test_default_history_dir_sits_beside_the_default_grades_file():
    from project_paths import SETUP_GRADES_HISTORY_DIR
    from ui.services.working_lately_service import default_store_dir

    assert Path(SETUP_GRADES_HISTORY_DIR).parent == default_store_dir()


# --- opening regime history -----------------------------------------------


def test_opening_regime_keeps_first_read_and_directional_anchor(tmp_path):
    from autopilot_core import record_opening_environment
    from opening_regime_history import opening_regime_for

    path = tmp_path / "opening_env.json"
    history = tmp_path / "auto_opening_regime_history.jsonl"
    morning = datetime(2026, 7, 17, 9, 31)

    record_opening_environment("neutral_chop", path=path, now=morning)
    record_opening_environment("bearish_strong", path=path, now=morning + timedelta(minutes=5))
    record_opening_environment("neutral_chop", path=path, now=morning + timedelta(hours=2))
    record_opening_environment("bullish_weak", path=path, now=morning + timedelta(hours=3))

    got = opening_regime_for(date(2026, 7, 17), path=history)
    assert got["label"] == "neutral_chop"
    assert got["source"] == "first_read"
    assert got["directional_anchor"] == "bearish_strong"
    assert datetime.fromisoformat(got["written_at"]).tzinfo is not None
    assert datetime.fromisoformat(got["anchor_written_at"]).tzinfo is not None
    rows = history.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2  # one first read, one anchor; later reads add nothing

    next_day = datetime(2026, 7, 20, 9, 31)
    record_opening_environment("bullish_strong", path=path, now=next_day)
    nxt = opening_regime_for("2026-07-20", path=history)
    assert nxt["label"] == "bullish_strong" and nxt["directional_anchor"] == "bullish_strong"
    assert opening_regime_for("2026-07-18", path=history) is None


def test_opening_regime_history_failure_never_breaks_the_opening_write(tmp_path, monkeypatch):
    import opening_regime_history
    from autopilot_core import load_opening_environment, record_opening_environment

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(opening_regime_history, "record", boom)
    path = tmp_path / "opening_env.json"
    now = datetime(2026, 7, 17, 9, 31)
    assert record_opening_environment("bearish_strong", path=path, now=now) == "bearish_strong"
    assert load_opening_environment(path=path, now=now) == "bearish_strong"


def test_opening_regime_default_history_is_beside_the_opening_file():
    from project_paths import AUTO_OPENING_ENV_FILE, AUTO_OPENING_REGIME_HISTORY_FILE

    assert Path(AUTO_OPENING_REGIME_HISTORY_FILE).parent == Path(AUTO_OPENING_ENV_FILE).parent


@pytest.mark.parametrize("label", ["", "   "])
def test_empty_regime_read_is_not_recorded(tmp_path, label):
    from opening_regime_history import opening_regime_for, record

    history = tmp_path / "h.jsonl"
    assert record("2026-07-17", label, "first_read", path=history) is False
    assert opening_regime_for("2026-07-17", path=history) is None
