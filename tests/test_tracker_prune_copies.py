"""P0-2 2f: ``tracker_store.py --prune-copies`` removes exactly two stale copies.

Decided 2026-09-24: the tracker ``.bak`` and the ``.damaged-20260905T200233``
SQLite (2.5 GB) go. The CLI lists by default, deletes only with ``--yes``,
refuses any other name, and writes a keyless job-ledger row. Scratch dirs only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tracker_store  # noqa: E402

BAK = "master_avwap_setup_tracker.json.bak"
DAMAGED = "master_avwap_setup_tracker.sqlite.damaged-20260905T200233"
LIVE_JSON = "master_avwap_setup_tracker.json"
LIVE_DB = "master_avwap_setup_tracker.sqlite"


@pytest.fixture
def tracker_dir(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    for name, size in ((BAK, 11), (DAMAGED, 7), (LIVE_JSON, 5), (LIVE_DB, 3)):
        (root / name).write_bytes(b"x" * size)
    ledger = tmp_path / "diag" / "job_ledger.jsonl"
    monkeypatch.setattr("job_ledger.default_ledger_path", lambda: ledger)
    return root, ledger


def _ledger_rows(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_without_yes_it_only_lists(tracker_dir, capsys):
    root, ledger = tracker_dir
    assert tracker_store._main(["--prune-copies", "--tracker-dir", str(root)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert {item["name"]: item["bytes"] for item in report["candidates"]} == {BAK: 11, DAMAGED: 7}
    assert report["total_bytes"] == 18
    assert (root / BAK).exists() and (root / DAMAGED).exists()
    rows = [row for row in _ledger_rows(ledger) if row["event"] == "tracker_prune_copies"]
    assert len(rows) == 1 and rows[0]["deleted"] is False and rows[0]["bytes"] == 18
    assert "key" not in rows[0]


def test_dry_run_wins_over_yes(tracker_dir):
    root, _ledger = tracker_dir
    assert tracker_store._main(["--prune-copies", "--yes", "--dry-run", "--tracker-dir", str(root)]) == 0
    assert (root / BAK).exists() and (root / DAMAGED).exists()


def test_with_yes_it_deletes_exactly_the_two_copies(tracker_dir, capsys):
    root, ledger = tracker_dir
    assert tracker_store._main(["--prune-copies", "--yes", "--tracker-dir", str(root)]) == 0
    assert not (root / BAK).exists() and not (root / DAMAGED).exists()
    assert (root / LIVE_JSON).exists() and (root / LIVE_DB).exists()
    row = [row for row in _ledger_rows(ledger) if row["event"] == "tracker_prune_copies"][-1]
    assert row["deleted"] is True
    assert sorted(row["names"]) == sorted([BAK, DAMAGED]) and row["bytes"] == 18
    assert "+" in row["ts"] or "-" in row["ts"][19:]


@pytest.mark.parametrize("name", [LIVE_JSON, LIVE_DB, "../" + BAK, "other.bak"])
def test_any_other_name_is_refused_and_nothing_is_deleted(tracker_dir, name, capsys):
    root, ledger = tracker_dir
    code = tracker_store._main(
        ["--prune-copies", "--yes", "--tracker-dir", str(root), "--name", BAK, "--name", name]
    )
    assert code != 0
    for kept in (BAK, DAMAGED, LIVE_JSON, LIVE_DB):
        assert (root / kept).exists()
    assert not [row for row in _ledger_rows(ledger) if row["event"] == "tracker_prune_copies"]


def test_a_directory_with_a_prunable_name_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    (root / BAK).mkdir(parents=True)
    monkeypatch.setattr("job_ledger.default_ledger_path", lambda: tmp_path / "ledger.jsonl")
    code, report = tracker_store.prune_copies(root, delete=True)
    assert code == 2 and report["refused"]
    assert (root / BAK).is_dir()


def test_missing_copies_are_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("job_ledger.default_ledger_path", lambda: tmp_path / "ledger.jsonl")
    code, report = tracker_store.prune_copies(tmp_path, delete=True)
    assert code == 0 and report["total_bytes"] == 0
