"""P12 (plan to 8/10): permutation report history, weak-variant verdicts and their chips.

Rank and annotate only: a verdict never hides a row, and nothing here feeds a
detector, a score, an alert, Focus, the queue or `review_policy.json`.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import setup_permutation_search as search  # noqa: E402


def _report(generated_at="2026-09-12T12:00:00+00:00", **extra):
    return {"schema": search.REPORT_SCHEMA, "generated_at": generated_at, "populations": {}, **extra}


# --- 1. report history ------------------------------------------------------


def test_history_writes_one_file_per_run_date_and_skips_unchanged_content(tmp_path):
    folder = tmp_path / "permutation_report_history"
    first = search.append_history(_report(), folder, today=date(2026, 9, 12))
    assert first == folder / "2026-09-12.json"
    assert json.loads(first.read_text(encoding="utf-8"))["schema"] == search.REPORT_SCHEMA
    # Same content, new run stamp: nothing written.
    assert search.append_history(_report("2026-09-19T12:00:00+00:00"), folder, today=date(2026, 9, 19)) is None
    changed = search.append_history(_report(source="x"), folder, today=date(2026, 9, 19))
    assert changed == folder / "2026-09-19.json"
    assert [day.isoformat() for day, _path in search.history_files(folder)] == ["2026-09-12", "2026-09-19"]


def test_history_prunes_files_older_than_600_days(tmp_path):
    folder = tmp_path / "permutation_report_history"
    folder.mkdir()
    (folder / "2024-01-06.json").write_text(json.dumps(_report(source="old")), encoding="utf-8")
    (folder / "2025-06-07.json").write_text(json.dumps(_report(source="kept")), encoding="utf-8")
    (folder / "notes.txt").write_text("not a report", encoding="utf-8")
    search.append_history(_report(source="new"), folder, today=date(2026, 9, 26))
    names = sorted(path.name for path in folder.iterdir())
    assert names == ["2025-06-07.json", "2026-09-26.json", "notes.txt"]


def test_the_cli_keeps_history_beside_its_out_and_the_live_default_is_the_constant(tmp_path, monkeypatch):
    import project_paths

    live_out = Path(project_paths.SETUP_PERMUTATION_REPORT_FILE)
    assert live_out.parent / search.HISTORY_DIR_NAME == Path(project_paths.SETUP_PERMUTATION_REPORT_HISTORY_DIR)
    monkeypatch.setattr(search, "read_outcomes", lambda _path: [])
    out = tmp_path / "scratch" / "permutation_report.json"
    assert search.main(["--outcomes", "x.parquet", "--ledger-root", str(tmp_path / "lake"), "--out", str(out)]) == 0
    assert len(search.history_files(out.parent / search.HISTORY_DIR_NAME)) == 1
