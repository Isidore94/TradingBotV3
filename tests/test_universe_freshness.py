"""Plan.md 23.5: multi-file generation freshness must not use max(mtime)."""

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autopilot_core as core  # noqa: E402


def _touch(path: Path, epoch: float) -> Path:
    path.write_text("AAPL\n", encoding="utf-8")
    os.utime(path, (epoch, epoch))
    return path


def test_one_fresh_file_cannot_hide_a_stale_companion(tmp_path):
    old = time.time() - 86_400
    new = time.time()
    paths = [
        _touch(tmp_path / "universe_all.txt", old),
        _touch(tmp_path / "universe_longs.txt", new),
        _touch(tmp_path / "universe_shorts.txt", new),
    ]
    built = core.universe_built_at(paths)
    assert built is not None
    assert abs(built.timestamp() - old) < 2, "generation is only as fresh as its stalest member"


def test_missing_required_file_means_no_valid_generation(tmp_path):
    paths = [
        _touch(tmp_path / "universe_all.txt", time.time()),
        tmp_path / "universe_longs.txt",  # missing
        _touch(tmp_path / "universe_shorts.txt", time.time()),
    ]
    assert core.universe_built_at(paths) is None
