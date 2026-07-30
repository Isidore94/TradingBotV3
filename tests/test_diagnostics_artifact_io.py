"""Shared diagnostics artifact I/O primitives (plan.md sec 4 auditability, sec 6.1 retention).

Covers the properties the ad-hoc per-module copies get wrong: temp cleanup on
every failure path, hashes that do not depend on dict/set ordering, retention
boundaries, and an archive step that never destroys evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnostics import (  # noqa: E402
    append_jsonl,
    append_jsonl_rows,
    archive_dated,
    atomic_write_json,
    canonical_json,
    config_hash,
    diagnostics_dir,
    diagnostics_path,
    prune_by_age,
    prune_by_size,
    read_jsonl,
    sweep_stale_temp_files,
)
from diagnostics import artifact_io  # noqa: E402

DAY = 86400.0


def _touch(path: Path, *, size: int = 0, mtime: float | None = None) -> Path:
    path.write_bytes(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _temp_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.tmp"))


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------
def test_atomic_write_json_round_trips_and_leaves_no_temp(tmp_path):
    target = tmp_path / "nested" / "state.json"
    returned = atomic_write_json(target, {"b": 2, "a": [1, 2, 3]})

    assert returned == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"b": 2, "a": [1, 2, 3]}
    assert _temp_files(target.parent) == []


def test_atomic_write_json_replaces_previous_content(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"generation": 1})
    atomic_write_json(target, {"generation": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 2}
    assert _temp_files(tmp_path) == []


def test_atomic_write_json_stages_temp_in_same_directory(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    seen: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(artifact_io.os, "replace", spy)
    atomic_write_json(target, {"ok": True})

    assert len(seen) == 1
    src, dst = seen[0]
    # Same-directory temp is what makes os.replace an atomic same-volume rename.
    assert Path(src).parent == target.parent
    assert Path(dst) == target


def test_atomic_write_json_failure_keeps_original_and_removes_temp(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"generation": 1})
    original = target.read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("rename failed (sync client lock)")

    saved = artifact_io.os.replace
    artifact_io.os.replace = boom
    try:
        with pytest.raises(OSError, match="rename failed"):
            atomic_write_json(target, {"generation": 2})
    finally:
        artifact_io.os.replace = saved

    assert target.read_text(encoding="utf-8") == original
    assert _temp_files(tmp_path) == [], "a failed write must not orphan a temp file"


def test_atomic_write_json_serialization_failure_touches_nothing(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"generation": 1})

    class Exploding:
        @property
        def __dict__(self):  # pragma: no cover - exercised via json.dumps default
            raise RuntimeError("cannot introspect")

        def __str__(self):
            raise RuntimeError("cannot stringify")

    with pytest.raises(RuntimeError):
        atomic_write_json(target, {"bad": Exploding()})

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 1}
    assert _temp_files(tmp_path) == []


def test_atomic_write_json_write_failure_removes_temp(tmp_path, monkeypatch):
    target = tmp_path / "state.json"

    class FailingHandle:
        def __init__(self, real):
            self._real = real

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._real.close()
            return False

        def write(self, _text):
            raise OSError("no space left on device")

    real_fdopen = artifact_io.os.fdopen
    monkeypatch.setattr(
        artifact_io.os, "fdopen", lambda *a, **k: FailingHandle(real_fdopen(*a, **k))
    )

    with pytest.raises(OSError, match="no space left"):
        atomic_write_json(target, {"generation": 1})

    assert not target.exists()
    assert _temp_files(tmp_path) == []


# ---------------------------------------------------------------------------
# append_jsonl
# ---------------------------------------------------------------------------
def test_append_jsonl_writes_one_record_per_line_in_order(tmp_path):
    path = tmp_path / "shadow.jsonl"
    for i in range(3):
        append_jsonl(path, {"seq": i, "state": "PULLBACK"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["seq"] for line in lines] == [0, 1, 2]
    assert read_jsonl(path) == [{"seq": i, "state": "PULLBACK"} for i in range(3)]


def test_append_jsonl_rows_batches_and_creates_parent(tmp_path):
    path = tmp_path / "deep" / "dir" / "shadow.jsonl"
    append_jsonl_rows(path, [{"i": 1}, {"i": 2}])
    append_jsonl_rows(path, [])  # no-op, must not add a blank line

    assert [row["i"] for row in read_jsonl(path)] == [1, 2]
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_append_jsonl_recovers_from_a_missing_trailing_newline(tmp_path):
    path = tmp_path / "shadow.jsonl"
    # A previous process died before writing its newline.
    path.write_text('{"seq": 0}', encoding="utf-8")
    append_jsonl(path, {"seq": 1})

    assert [row["seq"] for row in read_jsonl(path)] == [0, 1]


def test_read_jsonl_skips_a_corrupt_tail_line(tmp_path):
    path = tmp_path / "shadow.jsonl"
    append_jsonl(path, {"seq": 0})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 1, "trunc\n')

    assert read_jsonl(path) == [{"seq": 0}]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path, skip_bad=False)


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "absent.jsonl") == []


def test_append_jsonl_serializes_exotic_types(tmp_path):
    path = tmp_path / "shadow.jsonl"
    append_jsonl(path, {"at": datetime(2026, 7, 30, 9, 30), "window": timedelta(minutes=5)})

    row = read_jsonl(path)[0]
    assert row["at"] == "2026-07-30T09:30:00"
    assert row["window"] == "timedelta:300.0"


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------
def test_config_hash_ignores_dict_insertion_order():
    a = {"alpha": 1, "beta": {"x": [1, 2], "y": "z"}}
    b = {"beta": {"y": "z", "x": [1, 2]}, "alpha": 1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_ignores_set_ordering_but_not_list_ordering():
    assert config_hash({"s": {"a", "b", "c"}}) == config_hash({"s": {"c", "b", "a"}})
    assert config_hash({"l": [1, 2]}) != config_hash({"l": [2, 1]})


def test_config_hash_distinguishes_values_and_is_a_full_sha256():
    base = config_hash({"near_trigger_pct": 0.5})
    assert base != config_hash({"near_trigger_pct": 0.6})
    assert base != config_hash({"near_trigger_pctx": 0.5})
    assert len(base) == 64
    assert config_hash({"a": 1}, length=12) == config_hash({"a": 1})[:12]
    with pytest.raises(ValueError):
        config_hash({"a": 1}, length=0)


def test_config_hash_handles_config_objects_dataclasses_and_temporal_types():
    class Mode(Enum):
        SHADOW = "shadow"

    @dataclass
    class Nested:
        pct: float
        label: str

    class Config:
        def __init__(self):
            self.window = timedelta(minutes=30)
            self.anchor = date(2026, 7, 30)
            self.path = Path("a/b")
            self.mode = Mode.SHADOW
            self.nested = Nested(pct=0.25, label="near")
            self._cache = object()  # private/volatile state must not leak in

    first, second = config_hash(Config()), config_hash(Config())
    assert first == second
    assert len(first) == 64
    payload = json.loads(canonical_json(Config()))
    assert "_cache" not in payload
    assert payload["window"] == "timedelta:1800.0"
    assert payload["mode"] == "shadow"
    assert payload["nested"] == {"label": "near", "pct": 0.25}


def test_canonical_json_is_sorted_and_compact():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_config_hash_is_stable_across_processes_and_hash_seeds():
    payload = {"zeta": 1, "alpha": {"tags": {"b", "a"}}, "list": [3, 1, 2]}
    expected = config_hash(payload)

    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        from diagnostics import config_hash
        # Deliberately different insertion order from the parent process.
        payload = {"list": [3, 1, 2], "alpha": {"tags": {"a", "b"}}, "zeta": 1}
        print(config_hash(payload))
        """
    )
    for seed in ("0", "1", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", script, str(SCRIPTS_DIR)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert proc.stdout.strip() == expected, f"hash drifted with PYTHONHASHSEED={seed}"


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
def test_prune_by_age_keeps_the_exact_boundary_and_drops_older(tmp_path):
    now = 1_800_000_000.0
    fresh = _touch(tmp_path / "fresh.json", mtime=now - 0.5 * DAY)
    boundary = _touch(tmp_path / "boundary.json", mtime=now - 1.0 * DAY)
    stale = _touch(tmp_path / "stale.json", mtime=now - 1.0 * DAY - 1.0)
    ancient = _touch(tmp_path / "ancient.json", mtime=now - 30 * DAY)

    assert prune_by_age(tmp_path, 1.0, now=now) == 2
    assert fresh.exists() and boundary.exists()
    assert not stale.exists() and not ancient.exists()


def test_prune_by_age_respects_pattern_keep_newest_and_guards(tmp_path):
    now = 1_800_000_000.0
    _touch(tmp_path / "a.jsonl", mtime=now - 10 * DAY)
    _touch(tmp_path / "b.jsonl", mtime=now - 9 * DAY)
    other = _touch(tmp_path / "c.json", mtime=now - 10 * DAY)

    assert prune_by_age(tmp_path, 1.0, pattern="*.jsonl", keep_newest=1, now=now) == 1
    assert not (tmp_path / "a.jsonl").exists()
    assert (tmp_path / "b.jsonl").exists(), "keep_newest must spare the newest match"
    assert other.exists(), "pattern must not touch other artifacts"

    # Guards: a mis-set retention config must never wipe the evidence base.
    assert prune_by_age(tmp_path, 0, now=now) == 0
    assert prune_by_age(tmp_path, -5, now=now) == 0
    assert prune_by_age(tmp_path / "missing", 1.0) == 0
    assert (tmp_path / "b.jsonl").exists() and other.exists()


def test_prune_by_size_removes_oldest_until_under_budget(tmp_path):
    now = 1_800_000_000.0
    files = [
        _touch(tmp_path / f"{i}.json", size=100, mtime=now - (10 - i) * DAY) for i in range(4)
    ]

    assert prune_by_size(tmp_path, 250) == 2
    assert not files[0].exists() and not files[1].exists()
    assert files[2].exists() and files[3].exists()
    assert sum(p.stat().st_size for p in tmp_path.glob("*.json")) <= 250


def test_prune_by_size_never_deletes_the_newest_or_a_fitting_set(tmp_path):
    now = 1_800_000_000.0
    only = _touch(tmp_path / "today.jsonl", size=5000, mtime=now)

    assert prune_by_size(tmp_path, 100) == 0
    assert only.exists(), "the current artifact must survive an impossible budget"

    _touch(tmp_path / "old.jsonl", size=10, mtime=now - DAY)
    assert prune_by_size(tmp_path, 1_000_000) == 0
    assert prune_by_size(tmp_path, 0) == 0
    assert prune_by_size(tmp_path / "missing", 10) == 0
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_sweep_stale_temp_files_only_takes_old_temps(tmp_path):
    now = 1_800_000_000.0
    stale = _touch(tmp_path / "tmp8fj2k1.tmp", mtime=now - 7200)
    inflight = _touch(tmp_path / "artifact-abc.tmp", mtime=now - 5)
    real = _touch(tmp_path / "spy_state_shadow.jsonl", mtime=now - 7200)

    assert sweep_stale_temp_files(tmp_path, min_age_seconds=3600, now=now) == 1
    assert not stale.exists()
    assert inflight.exists(), "a concurrent write must never be swept"
    assert real.exists()


# ---------------------------------------------------------------------------
# archive_dated (must never destroy evidence)
# ---------------------------------------------------------------------------
def test_archive_dated_copies_every_row_and_preserves_the_source(tmp_path):
    log = tmp_path / "spy_state_shadow.jsonl"
    rows = [{"seq": i, "state": "PULLBACK"} for i in range(25)]
    append_jsonl_rows(log, rows)

    archived = archive_dated(log, "2026-07-30")

    assert archived is not None
    assert archived == tmp_path / "archive" / "spy_state_shadow-2026-07-30.jsonl"
    assert read_jsonl(archived) == rows
    assert log.exists(), "plan.md sec 6.1: rotate prior shadow logs WITHOUT deleting them"
    assert read_jsonl(log) == rows
    assert _temp_files(archived.parent) == []


def test_archive_dated_never_overwrites_an_existing_archive(tmp_path):
    log = tmp_path / "shadow.jsonl"
    append_jsonl(log, {"run": 1})
    first = archive_dated(log, date(2026, 7, 30))

    log.write_text("", encoding="utf-8")
    append_jsonl(log, {"run": 2})
    second = archive_dated(log, datetime(2026, 7, 30, 16, 5))

    assert first is not None and second is not None
    assert first != second
    assert second.name == "shadow-2026-07-30-2.jsonl"
    assert read_jsonl(first) == [{"run": 1}], "the earlier archive must be untouched"
    assert read_jsonl(second) == [{"run": 2}]


def test_archive_dated_accepts_explicit_dir_and_missing_source(tmp_path):
    log = tmp_path / "shadow.jsonl"
    append_jsonl(log, {"run": 1})
    destination = tmp_path / "cold" / "storage"

    archived = archive_dated(log, "2026-07-30", archive_dir=destination)
    assert archived is not None and archived.parent == destination

    assert archive_dated(tmp_path / "never_written.jsonl", "2026-07-30") is None


def test_archive_dated_rejects_an_empty_session_date(tmp_path):
    log = tmp_path / "shadow.jsonl"
    append_jsonl(log, {"run": 1})
    with pytest.raises(ValueError):
        archive_dated(log, "   ")
    assert read_jsonl(log) == [{"run": 1}]


def test_archive_dated_sanitizes_a_path_bearing_stamp(tmp_path):
    log = tmp_path / "shadow.jsonl"
    append_jsonl(log, {"run": 1})

    archived = archive_dated(log, "../../2026-07-30")

    assert archived is not None
    assert archived.parent == tmp_path / "archive"
    assert ".." not in archived.name


# ---------------------------------------------------------------------------
# location resolution
# ---------------------------------------------------------------------------
def test_diagnostics_dir_honors_the_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(tmp_path))

    assert diagnostics_dir() == tmp_path
    assert diagnostics_path("archive", "shadow.jsonl") == tmp_path / "archive" / "shadow.jsonl"


def test_end_to_end_shadow_day_rotation_under_the_override(tmp_path, monkeypatch):
    """The Milestone-1 shape: append all day, archive, then start the next day."""
    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(tmp_path))
    log = diagnostics_path("spy_state_shadow.jsonl")
    stamp = config_hash({"version": "v1", "near_trigger_pct": 0.5}, length=12)

    for seq in range(5):
        append_jsonl(log, {"seq": seq, "config_hash": stamp})
    atomic_write_json(log.with_name("spy_state_shadow_status.json"), {"rows_written": 5})

    archived = archive_dated(log, "2026-07-30")
    assert archived is not None and len(read_jsonl(archived)) == 5
    assert len(read_jsonl(log)) == 5

    append_jsonl(log, {"seq": 5, "config_hash": stamp})
    assert len(read_jsonl(log)) == 6
    assert len(read_jsonl(archived)) == 5, "the archive is immutable evidence"
    assert _temp_files(tmp_path) == [] and _temp_files(archived.parent) == []
