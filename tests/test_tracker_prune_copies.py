"""P0-2 2f: ``tracker_store.py --prune-copies`` removes exactly the named stale copies.

Decided 2026-09-24: the tracker ``.bak`` and the ``.damaged-20260905T200233``
SQLite (2.5 GB) go; the next P0-2 pass adds the three small leftovers from the
same 2026-09-05 repair (``-shm``, ``-wal``, ``_digests.json``). The CLI lists by
default, deletes only with ``--yes``, refuses any other name, and writes a
keyless job-ledger row. Scratch dirs only.
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
SHM = "master_avwap_setup_tracker.sqlite-shm.damaged-20260905T200233"
WAL = "master_avwap_setup_tracker.sqlite-wal.damaged-20260905T200233"
DIGESTS = "master_avwap_setup_tracker_digests.json.damaged-20260905T200233"
LEFTOVERS = (SHM, WAL, DIGESTS)
LIVE_JSON = "master_avwap_setup_tracker.json"
LIVE_DB = "master_avwap_setup_tracker.sqlite"


@pytest.fixture
def tracker_dir(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    for name, size in ((BAK, 11), (DAMAGED, 7), (SHM, 2), (WAL, 0), (DIGESTS, 1), (LIVE_JSON, 5), (LIVE_DB, 3)):
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
    assert {item["name"]: item["bytes"] for item in report["candidates"]} == {
        BAK: 11, DAMAGED: 7, SHM: 2, WAL: 0, DIGESTS: 1
    }
    assert report["total_bytes"] == 21
    assert all((root / name).exists() for name in (BAK, DAMAGED, *LEFTOVERS))
    rows = [row for row in _ledger_rows(ledger) if row["event"] == "tracker_prune_copies"]
    assert len(rows) == 1 and rows[0]["deleted"] is False and rows[0]["bytes"] == 21
    assert "key" not in rows[0]


def test_dry_run_wins_over_yes(tracker_dir):
    root, _ledger = tracker_dir
    assert tracker_store._main(["--prune-copies", "--yes", "--dry-run", "--tracker-dir", str(root)]) == 0
    assert (root / BAK).exists() and (root / DAMAGED).exists()


def test_with_yes_it_deletes_exactly_the_named_copies(tracker_dir, capsys):
    root, ledger = tracker_dir
    assert tracker_store._main(["--prune-copies", "--yes", "--tracker-dir", str(root)]) == 0
    assert not any((root / name).exists() for name in (BAK, DAMAGED, *LEFTOVERS))
    assert (root / LIVE_JSON).exists() and (root / LIVE_DB).exists()
    row = [row for row in _ledger_rows(ledger) if row["event"] == "tracker_prune_copies"][-1]
    assert row["deleted"] is True
    assert sorted(row["names"]) == sorted([BAK, DAMAGED, *LEFTOVERS]) and row["bytes"] == 21
    assert "+" in row["ts"] or "-" in row["ts"][19:]


@pytest.mark.parametrize("name", [LIVE_JSON, LIVE_DB, "../" + BAK, "other.bak"])
def test_any_other_name_is_refused_and_nothing_is_deleted(tracker_dir, name, capsys):
    root, ledger = tracker_dir
    code = tracker_store._main(
        ["--prune-copies", "--yes", "--tracker-dir", str(root), "--name", BAK, "--name", name]
    )
    assert code != 0
    for kept in (BAK, DAMAGED, *LEFTOVERS, LIVE_JSON, LIVE_DB):
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


def test_the_three_leftovers_can_be_pruned_by_name(tracker_dir):
    root, _ledger = tracker_dir
    argv = ["--prune-copies", "--yes", "--tracker-dir", str(root)]
    for name in LEFTOVERS:
        argv += ["--name", name]
    assert tracker_store._main(argv) == 0
    assert not any((root / name).exists() for name in LEFTOVERS)
    assert (root / BAK).exists() and (root / DAMAGED).exists()
    assert (root / LIVE_JSON).exists() and (root / LIVE_DB).exists()


@pytest.mark.parametrize(
    "name",
    [
        "master_avwap_setup_tracker.sqlite-shm",
        "master_avwap_setup_tracker.sqlite-wal",
        "master_avwap_setup_tracker_digests.json",
        "master_avwap_tracker_scoring_snapshot.json.damaged-20260905T200233",
    ],
)
def test_the_live_siblings_of_the_leftovers_are_refused(tracker_dir, name):
    root, _ledger = tracker_dir
    (root / name).write_bytes(b"live")
    code = tracker_store._main(["--prune-copies", "--yes", "--tracker-dir", str(root), "--name", name])
    assert code == 2
    assert (root / name).exists()
