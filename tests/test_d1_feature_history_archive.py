"""Lossless packing store for `d1_features_history.csv` (trader 2026-10-02).

The contract: for any state - never archived, archived but untrimmed, trimmed,
trimmed then appended to, schema widened after archiving - `read_history()`
equals reading the original untrimmed CSV row for row, in the same order, with
the same text. Archive and trim change nothing when they cannot prove the rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import d1_feature_history_archive as arc  # noqa: E402

TODAY = date(2026, 10, 2)

META = [
    "feature_history_schema_version",
    "run_id",
    "run_timestamp",
    "run_date",
    "watchlist_label",
    "scoring_config_hash",
    "scoring_config_updated_at",
]
FEATURES = ["symbol", "side", "last_close", "atr20", "note", "flag", "count"]
TRICKY = [
    "plain",
    "",
    "NA",
    "nan",
    "a,b",
    'say "hi"',
    "line1\nline2",
    "ünïcode",
    " lead space",
    "null",
]

#: (run_date, run_id, number of symbols). Two scans on 07-31 and on 10-01: the
#: repeat same-day scans the store must keep.
RUNS = [
    ("2026-07-30", "2026-07-30-150000", 7),
    ("2026-07-31", "2026-07-31-100000", 6),
    ("2026-07-31", "2026-07-31-150000", 6),
    ("2026-08-03", "2026-08-03-150000", 9),
    ("2026-08-28", "2026-08-28-150000", 5),
    ("2026-09-01", "2026-09-01-150000", 8),
    ("2026-09-25", "2026-09-25-150000", 7),
    ("2026-10-01", "2026-10-01-100000", 4),
    ("2026-10-01", "2026-10-01-150000", 4),
]


def _run_frame(run_date: str, run_id: str, n: int, *, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "feature_history_schema_version": "1",
            "run_id": run_id,
            "run_timestamp": run_id[:10] + "T" + run_id[11:13] + ":00:00",
            "run_date": run_date,
            "watchlist_label": "home folder watchlists",
            "scoring_config_hash": "abc123",
            "scoring_config_updated_at": "2026-07-01T00:00:00",
            "symbol": f"S{i:02d}",
            "side": "LONG" if i % 2 else "SHORT",
            "last_close": "1.50" if i % 3 == 0 else f"{10 + i}.25",
            "atr20": "" if i % 4 == 0 else "0.10",
            "note": TRICKY[(i + sum(map(ord, run_id))) % len(TRICKY)],
            "flag": "True" if i % 2 else "False",
            "count": "" if i % 5 == 0 else str(i),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def _append(path: Path, frame: pd.DataFrame) -> None:
    """The writer's own append: `to_csv(mode="a", header=False)` after the first write."""
    if not path.exists() or path.stat().st_size == 0:
        frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, mode="a", header=False, index=False)


def _build(path: Path, runs=RUNS) -> None:
    for run_date, run_id, n in runs:
        _append(path, _run_frame(run_date, run_id, n))


def _ref(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _snapshot(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


@pytest.fixture
def store(tmp_path):
    csv_path = tmp_path / "runtime" / "d1_features_history.csv"
    csv_path.parent.mkdir(parents=True)
    archive_dir = tmp_path / "runtime" / "d1_features_history_archive"
    _build(csv_path)
    return csv_path, archive_dir


def _read(store, **kwargs):
    csv_path, archive_dir = store
    return arc.read_history(csv_path=csv_path, archive_dir=archive_dir, **kwargs)


def _archive(store, **kwargs):
    csv_path, archive_dir = store
    return arc.archive(csv_path=csv_path, archive_dir=archive_dir, **kwargs)


def _trim(store, **kwargs):
    csv_path, archive_dir = store
    kwargs.setdefault("today", TODAY)
    return arc.trim(csv_path=csv_path, archive_dir=archive_dir, **kwargs)


# ---------------------------------------------------------------------------
# read_history == the original CSV, in every state
# ---------------------------------------------------------------------------


def test_never_archived_reads_back_the_csv_exactly(store):
    original = _ref(store[0])
    assert len(original) == sum(n for *_x, n in RUNS)
    pd.testing.assert_frame_equal(_read(store), original)


def test_archived_untrimmed_reads_back_the_csv_exactly(store):
    original = _ref(store[0])
    result = _archive(store)
    assert result["archived_rows"] == len(original)
    pd.testing.assert_frame_equal(_read(store), original)


def test_the_archive_keeps_cell_text_and_original_order(store):
    import pyarrow.parquet as pq

    csv_path, archive_dir = store
    original = _ref(csv_path)
    _archive(store)
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert sorted(manifest["files"]) == ["2026-07", "2026-08", "2026-09", "2026-10"]
    frames = []
    for entry in manifest["files"].values():
        frames.append(pq.read_table(archive_dir / entry["file"]).to_pandas())
    packed = pd.concat(frames).sort_values(arc.SEQ_COLUMN)
    assert packed[arc.SEQ_COLUMN].tolist() == list(range(len(original)))
    # Text, not numbers: "1.50" stays "1.50", "" stays "", "NA" stays "NA".
    assert packed["last_close"].tolist() == original["last_close"].tolist()
    assert "1.50" in packed["last_close"].tolist()
    assert "" in packed["atr20"].tolist()
    assert {"NA", "nan", "null", "line1\nline2", 'say "hi"'} <= set(packed["note"])


def test_manifest_records_sha256_and_row_count_per_file(store):
    _csv, archive_dir = store
    _archive(store)
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    total = 0
    for entry in manifest["files"].values():
        data = (archive_dir / entry["file"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        total += entry["rows"]
    assert total == manifest["archived_rows"] == len(_ref(store[0]))


def test_a_second_archive_packs_nothing_and_changes_no_bytes(store):
    _csv, archive_dir = store
    _archive(store)
    before = _snapshot(archive_dir)
    csv_before = store[0].read_bytes()
    again = _archive(store)
    assert again["archived_rows"] == 0
    assert _snapshot(archive_dir) == before
    assert store[0].read_bytes() == csv_before


def test_trim_dry_run_changes_nothing(store):
    csv_path, archive_dir = store
    _archive(store)
    before_csv = csv_path.read_bytes()
    before_archive = _snapshot(archive_dir)
    plan = _trim(store)
    assert plan["applied"] is False
    assert plan["would_remove"] == 7 + 6 + 6 + 9 + 5 + 8  # every row before 2026-09-02
    assert csv_path.read_bytes() == before_csv
    assert _snapshot(archive_dir) == before_archive


def test_trim_keeps_the_last_n_days_and_reads_back_the_original(store):
    csv_path, _ = store
    original = _ref(csv_path)
    _archive(store)
    result = _trim(store, apply=True)
    assert result["applied"] is True and result["removed"] == 41
    live = _ref(csv_path)
    assert list(live.columns) == list(original.columns)
    assert sorted(set(live["run_date"])) == ["2026-09-25", "2026-10-01"]
    # The kept rows are byte-for-byte the tail of the original file.
    assert original.tail(len(live)).reset_index(drop=True).equals(live)
    pd.testing.assert_frame_equal(_read(store), original)


def test_trimmed_then_appended_then_archived_reads_back_everything(store):
    csv_path, _ = store
    _archive(store)
    _trim(store, apply=True)
    _append(csv_path, _run_frame("2026-10-02", "2026-10-02-100000", 5))
    _append(csv_path, _run_frame("2026-10-02", "2026-10-02-150000", 5))
    expected = pd.concat(
        [
            _ref_untrimmed(RUNS),
            _run_frame("2026-10-02", "2026-10-02-100000", 5),
            _run_frame("2026-10-02", "2026-10-02-150000", 5),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(_read(store), expected)
    assert _archive(store)["archived_rows"] == 10
    pd.testing.assert_frame_equal(_read(store), expected)
    # And a second trim later on keeps it lossless.
    _trim(store, apply=True, today=date(2026, 11, 20))
    pd.testing.assert_frame_equal(_read(store), expected)


def _ref_untrimmed(runs) -> pd.DataFrame:
    return pd.concat([_run_frame(d, r, n) for d, r, n in runs], ignore_index=True)


def test_the_fixture_round_trips_through_the_csv(store):
    # Guard on the fixture itself: what the writer put down is what we built.
    pd.testing.assert_frame_equal(_ref(store[0]), _ref_untrimmed(RUNS))


def test_schema_widened_after_archiving_through_the_real_writer(store, monkeypatch):
    """The live writer widens by rewriting the whole file through pandas.

    Rows archived BEFORE the widening keep the text they had when packed (the
    writer's rewrite turns "1.50" into "1.5"; the archive does not) and come
    back with the new column empty - exactly what the widened file holds there.
    """
    from master_avwap_lib import legacy

    csv_path, _ = store
    before = _ref(csv_path)
    _archive(store)
    _trim(store, apply=True)
    monkeypatch.setattr(legacy, "D1_FEATURE_HISTORY_FILE", csv_path)
    widened_run = _run_frame("2026-10-02", "2026-10-02-100000", 3, extra={"new_col": "x1.50"})
    features = widened_run.drop(columns=META)
    legacy.append_d1_feature_history(
        features,
        {
            "run_id": "2026-10-02-100000",
            "run_timestamp": "2026-10-02T10:00:00",
            "run_date": "2026-10-02",
            "watchlist_label": "home folder watchlists",
            "scoring_config_hash": "abc123",
            "scoring_config_updated_at": "2026-07-01T00:00:00",
        },
    )
    live = _ref(csv_path)
    assert list(live.columns)[-1] == "new_col"
    # The writer's rewrite is lossy for text: this is why the archive keeps its own.
    assert "1.50" in set(before["last_close"].tail(len(live) - 3))
    assert "1.50" not in set(live["last_close"].head(len(live) - 3))
    new_rows = live.tail(3).reset_index(drop=True)
    expected = pd.concat([before.assign(new_col=""), new_rows], ignore_index=True)
    pd.testing.assert_frame_equal(_read(store), expected)
    # Verify stays clean over the rewritten rows, archive packs only the 3 new
    # rows, and the next trim proves the rewritten rows against the archive.
    assert arc.verify(csv_path=csv_path, archive_dir=store[1])["ok"] is True
    assert _archive(store)["archived_rows"] == 3
    pd.testing.assert_frame_equal(_read(store), expected)
    _trim(store, apply=True, today=date(2026, 11, 20))
    pd.testing.assert_frame_equal(_read(store), expected)
    assert len(_ref(csv_path)) == 0


def test_rows_with_a_blank_or_unparseable_run_date_are_never_trimmed(tmp_path):
    csv_path = tmp_path / "h.csv"
    archive_dir = tmp_path / "arc"
    _build(csv_path, RUNS[:2])
    _append(csv_path, _run_frame("", "blank-run", 2))
    _append(csv_path, _run_frame("not-a-date", "odd-run", 2))
    _build(csv_path, RUNS[3:])
    original = _ref(csv_path)
    store = (csv_path, archive_dir)
    _archive(store)
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["files"][arc.UNDATED]["rows"] == 4
    result = _trim(store, apply=True)
    live = _ref(csv_path)
    assert {"", "not-a-date"} <= set(live["run_date"])
    assert result["removed"] == 7 + 6  # stops at the first undated row
    assert result["stopped_by"] == "undated"
    pd.testing.assert_frame_equal(_read(store), original)


# ---------------------------------------------------------------------------
# Refusals: nothing changes, last good archive kept, loud
# ---------------------------------------------------------------------------


def test_a_failed_verify_changes_nothing_and_is_loud(store, monkeypatch):
    csv_path, archive_dir = store
    _archive(store)
    _append(csv_path, _run_frame("2026-10-02", "2026-10-02-100000", 3))
    before = _snapshot(archive_dir)
    real = arc._archive_table

    def corrupt(*args, **kwargs):
        table = real(*args, **kwargs)
        import pyarrow as pa

        index = table.schema.get_field_index("last_close")
        bad = pa.array(["9.99"] * table.num_rows, pa.string())
        return table.set_column(index, "last_close", bad)

    monkeypatch.setattr(arc, "_archive_table", corrupt)
    with pytest.raises(arc.VerifyFailed):
        _archive(store)
    assert _snapshot(archive_dir) == before
    monkeypatch.setattr(arc, "_archive_table", real)
    assert _archive(store)["archived_rows"] == 3


def test_a_tampered_archive_file_fails_verify_and_blocks_archive_and_trim(store):
    csv_path, archive_dir = store
    _archive(store)
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    target = archive_dir / manifest["files"]["2026-08"]["file"]
    target.write_bytes(target.read_bytes() + b"x")
    report = arc.verify(csv_path=csv_path, archive_dir=archive_dir)
    assert report["ok"] is False and any("sha256" in p for p in report["problems"])
    csv_before = csv_path.read_bytes()
    _append(csv_path, _run_frame("2026-10-02", "2026-10-02-100000", 3))
    csv_before = csv_path.read_bytes()
    with pytest.raises(arc.ArchiveError):
        _archive(store)
    with pytest.raises(arc.ArchiveError):
        _trim(store, apply=True)
    assert csv_path.read_bytes() == csv_before


def test_lost_live_rows_are_refused_loudly(store):
    csv_path, _ = store
    _archive(store)
    frame = _ref(csv_path).drop(index=[3]).reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    report = arc.verify(csv_path=csv_path, archive_dir=store[1])
    assert report["ok"] is False
    with pytest.raises(arc.ArchiveError):
        _archive(store)
    with pytest.raises(arc.ArchiveError):
        _trim(store, apply=True)


def test_trim_refuses_when_the_writer_lock_is_unavailable(store, monkeypatch):
    from local_writer_lock import LocalLockUnavailable

    csv_path, _ = store
    _archive(store)
    before = csv_path.read_bytes()

    @contextmanager
    def busy(key, **kwargs):
        raise LocalLockUnavailable("held by a scan")
        yield  # pragma: no cover

    monkeypatch.setattr(arc, "local_writer_lock", busy)
    with pytest.raises(arc.ArchiveRefused):
        _trim(store, apply=True)
    assert csv_path.read_bytes() == before


def test_trim_takes_the_writers_own_lock_key(store, monkeypatch):
    from local_writer_lock import lock_key_for_path

    csv_path, _ = store
    _archive(store)
    keys = []
    real = arc.local_writer_lock

    @contextmanager
    def spy(key, **kwargs):
        keys.append(key)
        with real(key, **kwargs) as info:
            yield info

    monkeypatch.setattr(arc, "local_writer_lock", spy)
    _trim(store, apply=True)
    assert keys[0] == lock_key_for_path(csv_path)


def test_trim_refuses_an_unreadable_header(tmp_path):
    csv_path = tmp_path / "h.csv"
    csv_path.write_bytes(b"")
    with pytest.raises(arc.ArchiveRefused):
        arc.trim(csv_path=csv_path, archive_dir=tmp_path / "arc", today=TODAY, apply=True)


def test_trim_never_removes_unarchived_rows(store):
    csv_path, _ = store
    before = csv_path.read_bytes()
    result = _trim(store, apply=True)
    assert result["removed"] == 0 and result["stopped_by"] == "unarchived"
    assert csv_path.read_bytes() == before


def test_a_failed_replace_leaves_the_live_file_and_the_reads_whole(store, monkeypatch):
    csv_path, archive_dir = store
    original = _ref(csv_path)
    _archive(store)
    before = csv_path.read_bytes()
    real_replace = arc.os.replace

    def refuse_csv(src, dst):
        if Path(dst) == csv_path:
            raise PermissionError("file in use")
        return real_replace(src, dst)

    monkeypatch.setattr(arc.os, "replace", refuse_csv)
    with pytest.raises(arc.ArchiveError):
        _trim(store, apply=True)
    monkeypatch.setattr(arc.os, "replace", real_replace)
    assert csv_path.read_bytes() == before
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest.get("pending_live_offset") is None
    assert not [p for p in csv_path.parent.iterdir() if ".tmp" in p.name]
    pd.testing.assert_frame_equal(_read(store), original)


def test_a_crash_after_the_replace_is_settled_from_the_live_file(store, monkeypatch):
    """Manifest says a trim is pending and the live file was already replaced."""
    csv_path, archive_dir = store
    original = _ref(csv_path)
    _archive(store)
    real_write = arc._write_manifest
    calls = {"n": 0}

    def die_on_commit(path, manifest):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("power cut")
        return real_write(path, manifest)

    monkeypatch.setattr(arc, "_write_manifest", die_on_commit)
    with pytest.raises((OSError, arc.ArchiveError)):
        _trim(store, apply=True)
    monkeypatch.setattr(arc, "_write_manifest", real_write)
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["pending_live_offset"] == 41
    assert len(_ref(csv_path)) == len(original) - 41
    pd.testing.assert_frame_equal(_read(store), original)
    assert arc.verify(csv_path=csv_path, archive_dir=archive_dir)["ok"] is True
    _archive(store)  # settles the pending trim under the lock
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["pending_live_offset"] is None and manifest["live_offset"] == 41
    pd.testing.assert_frame_equal(_read(store), original)


# ---------------------------------------------------------------------------
# Narrow reads, chunking, dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["live", "archived", "trimmed"])
def test_columns_and_date_bounds_match_a_filtered_csv_read(store, state):
    csv_path, _ = store
    original = _ref(csv_path)
    if state in ("archived", "trimmed"):
        _archive(store)
    if state == "trimmed":
        _trim(store, apply=True)
    cols = ["run_id", "symbol", "note"]
    got = _read(store, columns=cols, since="2026-07-31", until="2026-09-25")
    mask = (original["run_date"] >= "2026-07-31") & (original["run_date"] <= "2026-09-25")
    expected = original.loc[mask, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)
    got_dates = _read(store, since=date(2026, 10, 1))
    expected_dates = original[original["run_date"] >= "2026-10-01"].reset_index(drop=True)
    pd.testing.assert_frame_equal(got_dates, expected_dates)


def test_a_narrow_read_pushes_columns_and_months_into_parquet(store, monkeypatch):
    import pyarrow.parquet as pq

    _archive(store)
    seen = []
    real = pq.read_table

    def spy(source, *args, **kwargs):
        seen.append((Path(str(source)).name, tuple(kwargs.get("columns") or ())))
        return real(source, *args, **kwargs)

    monkeypatch.setattr(arc.pq, "read_table", spy)
    _read(store, columns=["symbol"], since="2026-08-01", until="2026-08-31")
    assert seen and all(name.startswith("2026-08") for name, _ in seen)
    for _name, cols in seen:
        assert set(cols) <= {"symbol", "run_date", arc.SEQ_COLUMN}


def test_an_unknown_column_is_an_error(store):
    with pytest.raises(ValueError):
        _read(store, columns=["no_such_column"])


def test_small_blocks_across_month_boundaries_stay_lossless(store):
    csv_path, _ = store
    original = _ref(csv_path)
    _archive(store, block_bytes=1024)
    pd.testing.assert_frame_equal(_read(store), original)
    _trim(store, apply=True)
    pd.testing.assert_frame_equal(_read(store), original)


@pytest.mark.parametrize("state", ["live", "archived", "trimmed"])
def test_typed_read_gives_pandas_own_dtypes(store, state):
    csv_path, _ = store
    original_typed = pd.read_csv(csv_path, low_memory=False)
    if state in ("archived", "trimmed"):
        _archive(store)
    if state == "trimmed":
        _trim(store, apply=True)
    pd.testing.assert_frame_equal(_read(store, typed=True), original_typed)
    # Like `usecols`, the columns come back in file order.
    narrow = _read(store, columns=["count", "flag", "atr20"], typed=True)
    pd.testing.assert_frame_equal(narrow, original_typed[["atr20", "flag", "count"]])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_status_archive_verify_and_trim_is_a_dry_run_without_apply(store, capsys):
    csv_path, archive_dir = store
    base = ["--csv", str(csv_path), "--archive-dir", str(archive_dir)]
    assert arc.main(["status", *base]) == 0
    assert arc.main(["archive", *base]) == 0
    assert arc.main(["verify", *base]) == 0
    before = csv_path.read_bytes()
    assert arc.main(["trim", *base, "--today", TODAY.isoformat()]) == 0
    assert csv_path.read_bytes() == before
    assert arc.main(["trim", *base, "--today", TODAY.isoformat(), "--apply"]) == 0
    assert csv_path.read_bytes() != before
    out = capsys.readouterr().out
    assert "dry run" in out.lower()


def test_cli_verify_exits_nonzero_on_a_bad_archive(store):
    csv_path, archive_dir = store
    _archive(store)
    manifest = json.loads((archive_dir / arc.MANIFEST_NAME).read_text(encoding="utf-8"))
    target = archive_dir / manifest["files"]["2026-07"]["file"]
    target.write_bytes(b"garbage")
    assert arc.main(["verify", "--csv", str(csv_path), "--archive-dir", str(archive_dir)]) != 0


def test_paths_live_beside_the_history_csv():
    import project_paths as pp

    assert pp.D1_FEATURES_HISTORY_ARCHIVE_DIR.parent == pp.D1_FEATURES_HISTORY_FILE.parent
    assert pp.D1_FEATURES_HISTORY_ARCHIVE_DIR.parent == pp.RUNTIME_DATA_DIR
    assert arc.DEFAULT_TRIM_KEEP_DAYS == 30
