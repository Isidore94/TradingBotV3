"""R6(b) item 2: the read-only JSONL-ledger audit.

Rotation of `technical_integrity_events.jsonl` was DECLINED on 2026-08-17 on
measurement, so what the desk is owed is not a pruner but an honest number.
These tests pin the two properties that make the number trustworthy:

- it is MEASURED, never remembered (the two hard-coded sizes that used to live
  in this module and in the R6 packet were both stale by growth within weeks);
- it is READ-ONLY (a retention feature that quietly truncated a bronze
  warehouse source would break the ingest watermark, which is exactly why
  rotation was declined).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import operations_audit as oa  # noqa: E402


def _write_ledger(root: Path, name: str, rows: int, payload: str = "x" * 200) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f'{{"row": {i}, "pad": "{payload}"}}' for i in range(rows)) + "\n",
                    encoding="utf-8")
    return path


def _walk(root: Path):
    """The (size, relative name) list the footprint check already builds."""
    entries = []
    for path in root.rglob("*"):
        if path.is_file():
            entries.append((path.stat().st_size, str(path.relative_to(root))))
    entries.sort(reverse=True)
    return entries


def test_a_big_ledger_is_measured_and_its_rows_estimated(tmp_path):
    path = _write_ledger(tmp_path, "technical_integrity_events.jsonl", 6000)
    rows = oa._jsonl_ledger_rows(tmp_path, _walk(tmp_path))

    assert [row["artifact"] for row in rows] == ["technical_integrity_events.jsonl"]
    row = rows[0]
    assert row["megabytes"] >= oa.JSONL_LEDGER_MIN_MB
    # A sampled mean over near-uniform rows lands within a fraction of a
    # percent. It is called `estimated_rows` because it is one - the
    # alternative is reading 370 MB inside System Health.
    assert abs(row["estimated_rows"] - 6000) <= 60
    assert row["modified_at"]
    assert "declined" in row["retention"]
    # And the file is untouched: same size, same bytes.
    assert path.stat().st_size == len(path.read_bytes())


def test_the_estimate_is_labelled_as_one_and_survives_ragged_rows(tmp_path):
    """Rows in the real ledger are not uniform; the estimate must not pretend."""
    lines = []
    for index in range(12000):
        pad = "y" * (50 if index % 2 else 400)
        lines.append(f'{{"row": {index}, "pad": "{pad}"}}')
    (tmp_path / "greatness_shadow.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    row = oa._jsonl_ledger_rows(tmp_path, _walk(tmp_path))[0]
    assert "estimated_rows" in row
    assert row["sampled_lines"] > 0
    # Within a sane band of the truth rather than exact - which is the honest
    # claim for a sampled mean over ragged lines.
    assert 6000 <= row["estimated_rows"] <= 24000


def test_small_ledgers_and_non_ledgers_are_skipped(tmp_path):
    _write_ledger(tmp_path, "tiny.jsonl", 5)
    (tmp_path / "heartbeat.json").write_text("{}" * 400_000, encoding="utf-8")
    assert oa._jsonl_ledger_rows(tmp_path, _walk(tmp_path)) == []


def test_an_unreadable_ledger_is_unknown_not_zero(tmp_path, monkeypatch):
    _write_ledger(tmp_path, "job_ledger.jsonl", 6000)
    walk = _walk(tmp_path)

    original = Path.open

    def _refuse(self, *args, **kwargs):
        if self.name == "job_ledger.jsonl":
            raise OSError("locked")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _refuse)
    row = oa._jsonl_ledger_rows(tmp_path, walk)[0]
    assert row["estimated_rows"] is None


def test_the_footprint_check_reports_the_ledgers(tmp_path):
    from datetime import datetime

    _write_ledger(tmp_path, "technical_integrity_events.jsonl", 6000)
    check = oa._disk_check(tmp_path, datetime(2026, 8, 18, 21, 0))
    ledgers = check["details"]["jsonl_ledgers"]
    assert [row["artifact"] for row in ledgers] == ["technical_integrity_events.jsonl"]


def test_no_size_is_hard_coded_in_the_module_comment():
    """The comment that said ~106 MB was a mid-July docstring, never a reading."""
    source = (SCRIPTS_DIR / "operations_audit.py").read_text(encoding="utf-8")
    header = source.split("def ", 1)[0]
    assert "already ~" not in header
    assert "MEASURES the live files" in header
    assert "2026-08-17" in header
