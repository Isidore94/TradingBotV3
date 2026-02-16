#!/usr/bin/env python3
"""Utility script to reset the repository to a tidy default state."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EXCLUDED_PYCACHE_ROOTS = {".git", ".venv"}

# Paths that should be emptied (but kept in place) when tidying.
DIRECTORIES_TO_CLEAR = (
    "data",
    "logs",
    "output",
)

PRESERVED_FILENAMES = {".gitkeep"}

# Individual generated artifacts that should be removed if present.
FILES_TO_DELETE = (
    Path("output") / "master_avwap_events.txt",
    Path("logs") / "bouncers.txt",
)


def _resolve_repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


def empty_directory(path: Path) -> None:
    """Delete all contents within *path* while preserving the directory itself."""
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name in PRESERVED_FILENAMES:
            continue
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)


def delete_file(path: Path) -> None:
    """Delete *path* if it exists."""
    if path.exists():
        path.unlink()


def _pycache_directories(root: Path):
    """Yield every __pycache__ directory that should be removed."""
    for pycache_dir in root.rglob("__pycache__"):
        if any(excluded in pycache_dir.parts for excluded in EXCLUDED_PYCACHE_ROOTS):
            continue
        yield pycache_dir


def remove_pycache(root: Path) -> int:
    """Remove every __pycache__ directory below *root*."""
    removed = 0
    for pycache_dir in _pycache_directories(root):
        shutil.rmtree(pycache_dir)
        removed += 1
    return removed


def tidy_workspace(dry_run: bool = False) -> None:
    """Reset generated artifacts to match the repository's default state."""
    repo_root = _resolve_repo_root()

    directories_cleared = []
    for rel_dir in DIRECTORIES_TO_CLEAR:
        directory = repo_root / rel_dir
        if dry_run:
            if directory.exists():
                directories_cleared.append(str(directory.relative_to(repo_root)))
            continue
        empty_directory(directory)
        directories_cleared.append(str(directory.relative_to(repo_root)))

    files_deleted = []
    for rel_path in FILES_TO_DELETE:
        file_path = repo_root / rel_path
        if file_path.exists():
            files_deleted.append(str(rel_path))
            if not dry_run:
                delete_file(file_path)

    if dry_run:
        pycache_removed = sum(1 for _ in _pycache_directories(repo_root))
    else:
        pycache_removed = remove_pycache(repo_root)

    print("Directories cleared:")
    if directories_cleared:
        for directory in directories_cleared:
            print(f"  - {directory}")
    else:
        print("  (none)")

    print("Files deleted:")
    if files_deleted:
        for file in files_deleted:
            print(f"  - {file}")
    else:
        print("  (none)")

    print(f"__pycache__ directories removed: {pycache_removed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove generated files and folders so the repository matches its "
            "default tidy state."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tidy_workspace(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
