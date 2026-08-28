"""A gigabyte snapshot must not be loaded twice to decide it is unchanged.

Measured on the desk 2026-08-27: `master_avwap_setup_tracker.json` is
**1,026,057,028 bytes**. `ingest_artifact` did `source.read_bytes()` and hashed
it BEFORE consulting the watermark, so every bronze ingest - including the ones
that concluded UNCHANGED - allocated 1.03 GB inside the desk process. When the
sha HAD changed it then decoded the whole thing to text and ran `json.loads`
over it, which is several GB more, on top of the warehouse build's own peak.

What the parse actually bought for this artifact is nothing. `setup_tracker`
declares no `event_keys` and no `id_keys`, so `_parse_event_at` returns None on
its first line (`if not keys`) and `_first_value` returns "" without looking at
the payload - whether or not it parsed. The ONLY column the parse influences is
`quality`. That is what makes skipping it above a size threshold a defensible
trade rather than a silent loss, and it is why these tests pin the derived
columns as well as the memory behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("pyarrow", reason="the research lake is parquet")


@pytest.fixture
def lake(tmp_path, monkeypatch):
    from research_warehouse import config
    from research_warehouse.store import ResearchStore

    root = tmp_path / "research_lake"
    monkeypatch.setattr(config, "get_research_store_dir", lambda: root)
    monkeypatch.setattr(config, "warehouse_enabled", lambda: True)
    config.ensure_lake_layout(root)
    return ResearchStore(root)


@pytest.fixture
def tracker_artifact():
    from research_warehouse.ingest_existing import BRONZE_ARTIFACTS

    found = next(a for a in BRONZE_ARTIFACTS if a.artifact == "setup_tracker")
    assert found.event_keys == () and found.id_keys == (), (
        "this whole packet rests on setup_tracker deriving nothing from the "
        "parse; if it gained event_keys/id_keys the threshold skip must be "
        "reconsidered, not silently kept"
    )
    return found


def _write_json(path: Path, *, padding: int = 0) -> Path:
    payload = {"schema": "tracker", "symbols": {"AAA": {"note": "x" * padding}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------- the UNCHANGED path


def test_an_unchanged_snapshot_is_decided_without_reading_the_file(
    lake, tracker_artifact, tmp_path, monkeypatch
):
    """The 1.03 GB allocation that bought nothing. The watermark comparison
    needs the file's HASH, which can be computed in chunks; it never needed
    the file's CONTENT in memory."""
    from research_warehouse import ingest_existing

    source = _write_json(tmp_path / "tracker.json", padding=2048)
    first = ingest_existing.ingest_artifact(lake, tracker_artifact, path=source)
    assert first.rows_ingested == 1, "the first ingest must actually publish"

    reads: list[str] = []
    real_read_bytes = Path.read_bytes

    def counting_read_bytes(self):
        reads.append(str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    again = ingest_existing.ingest_artifact(lake, tracker_artifact, path=source)

    assert again.status == "UNCHANGED"
    assert str(source) not in reads, (
        "an unchanged 1 GB snapshot was read into memory just to be discarded"
    )
    assert again.source_sha256 == first.source_sha256, "same file, same hash"


def test_the_chunked_hash_is_the_same_hash_as_before(tmp_path):
    """The watermark on disk was written by the old whole-file hash, so a
    different digest would make every artifact look changed forever."""
    from research_warehouse.ingest_existing import _sha256_bytes, _sha256_path

    source = _write_json(tmp_path / "tracker.json", padding=100_000)
    assert _sha256_path(source) == _sha256_bytes(source.read_bytes())


# ------------------------------------------------------ the threshold skip


def test_a_huge_snapshot_is_not_json_parsed(lake, tracker_artifact, tmp_path, monkeypatch):
    from research_warehouse import ingest_existing

    monkeypatch.setattr(ingest_existing, "SNAPSHOT_PARSE_MAX_BYTES", 512)
    source = _write_json(tmp_path / "tracker.json", padding=4096)
    assert source.stat().st_size > 512

    parsed_calls: list[int] = []
    real_loads = json.loads

    def counting_loads(text, *args, **kwargs):
        parsed_calls.append(len(text))
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(ingest_existing.json, "loads", counting_loads)
    report = ingest_existing.ingest_artifact(lake, tracker_artifact, path=source)

    assert report.rows_ingested == 1
    assert parsed_calls == [], "a file over the threshold must not be parsed"

    row = lake.read_table(tracker_artifact.dataset).to_pylist()[0]
    assert row["payload"] == source.read_text(encoding="utf-8"), "stored in FULL"
    assert row["payload_format"] == "JSON"
    assert row["quality"] == "COMPLETE", "JSON-shaped and stored whole"


def test_a_small_snapshot_is_still_parsed(lake, tracker_artifact, tmp_path, monkeypatch):
    """The threshold must not quietly turn the parse off everywhere."""
    from research_warehouse import ingest_existing

    monkeypatch.setattr(ingest_existing, "SNAPSHOT_PARSE_MAX_BYTES", 10_000_000)
    source = _write_json(tmp_path / "tracker.json", padding=8)

    parsed_calls: list[int] = []
    real_loads = json.loads

    def counting_loads(text, *args, **kwargs):
        parsed_calls.append(len(text))
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(ingest_existing.json, "loads", counting_loads)
    ingest_existing.ingest_artifact(lake, tracker_artifact, path=source)
    assert parsed_calls, "a small file must still be parsed"


def test_the_skip_changes_only_the_quality_column(lake, tracker_artifact, tmp_path, monkeypatch):
    """Equivalence, stated exactly. For an artifact with no event_keys and no
    id_keys the parse feeds nothing else, so a parsed row and a skipped row
    must be identical everywhere but `quality` - and here, not even there."""
    from research_warehouse import ingest_existing

    source = _write_json(tmp_path / "tracker.json", padding=4096)

    monkeypatch.setattr(ingest_existing, "SNAPSHOT_PARSE_MAX_BYTES", 10_000_000)
    parsed = ingest_existing._snapshot_row(
        tracker_artifact, source, source.read_bytes(), "sha", None, "run", offset=0
    )
    monkeypatch.setattr(ingest_existing, "SNAPSHOT_PARSE_MAX_BYTES", 8)
    skipped = ingest_existing._snapshot_row(
        tracker_artifact, source, source.read_bytes(), "sha", None, "run", offset=0
    )

    assert parsed == skipped, (
        "for setup_tracker the parse influences nothing at all; a difference "
        "here means the threshold is losing information"
    )


def test_a_huge_file_that_is_not_json_shaped_is_marked_invalid(
    lake, tracker_artifact, tmp_path, monkeypatch
):
    """The cheap check must be able to say NO. Missing data is uncertainty,
    never confirmation - a file we cannot even show to be JSON-shaped must not
    be published as COMPLETE."""
    from research_warehouse import ingest_existing

    monkeypatch.setattr(ingest_existing, "SNAPSHOT_PARSE_MAX_BYTES", 8)
    source = tmp_path / "tracker.json"
    source.write_text("this is not json at all, it is prose", encoding="utf-8")

    ingest_existing.ingest_artifact(lake, tracker_artifact, path=source)
    row = lake.read_table(tracker_artifact.dataset).to_pylist()[0]
    assert row["quality"] == "INVALID_DATA"
    assert row["payload"] == source.read_text(encoding="utf-8"), "still stored whole"


def test_a_small_file_that_is_not_json_is_still_marked_invalid(
    lake, tracker_artifact, tmp_path, monkeypatch
):
    """The pre-existing behaviour under the threshold is unchanged."""
    from research_warehouse import ingest_existing

    monkeypatch.setattr(ingest_existing, "SNAPSHOT_PARSE_MAX_BYTES", 10_000_000)
    source = tmp_path / "tracker.json"
    source.write_text("{not json", encoding="utf-8")

    ingest_existing.ingest_artifact(lake, tracker_artifact, path=source)
    row = lake.read_table(tracker_artifact.dataset).to_pylist()[0]
    assert row["quality"] == "INVALID_DATA"


def test_json_shape_check_handles_whitespace_and_both_containers(tmp_path):
    from research_warehouse.ingest_existing import _looks_like_json

    assert _looks_like_json('  \n {"a": 1} \n ')
    assert _looks_like_json("[1, 2, 3]\n")
    assert not _looks_like_json("plain text")
    assert not _looks_like_json("")
    assert not _looks_like_json("   ")
    assert not _looks_like_json('{"a": 1')


def test_the_default_threshold_is_well_above_normal_artifacts_and_below_the_tracker():
    """64 MB: every other bronze snapshot on this desk is orders of magnitude
    smaller, and the tracker (1.03 GB measured 2026-08-27) is far above it."""
    from research_warehouse.ingest_existing import SNAPSHOT_PARSE_MAX_BYTES

    assert 1_000_000 < SNAPSHOT_PARSE_MAX_BYTES < 1_026_057_028
