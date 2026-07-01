"""One-time compaction of the Master AVWAP setup tracker.

Strips per-bar replay detail (daily_marks + scenario events) from *sealed* records
-- CLOSED setups whose forward window ended past the daily-bar restatement buffer,
which are already recompute no-ops. Scalar scenario outcomes are kept, so every
ranking/stats aggregation is unaffected. The regular after-close scan does this
automatically now; this script just applies it immediately to the existing file.

Usage:
    python scripts/compact_setup_tracker.py --dry-run   # report only, no write
    python scripts/compact_setup_tracker.py             # compact + save (rotates .bak)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_paths import MASTER_AVWAP_SETUP_TRACKER_FILE  # noqa: E402
from master_avwap_lib import legacy as m  # noqa: E402


def _record_count(tracker: dict) -> int:
    return sum(
        len(tracker.get(ns) or {})
        for ns in ("setups", "control_setups", "study_setups")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only; do not save")
    args = parser.parse_args()

    path = MASTER_AVWAP_SETUP_TRACKER_FILE
    size_before = path.stat().st_size if path.exists() else 0
    print(f"Tracker: {path}")
    print(f"Size before: {size_before / 1e6:.1f} MB")

    tracker = m.load_setup_tracker_payload()
    total = _record_count(tracker)
    scan_date = datetime.now().date().isoformat()

    compacted = m._compact_sealed_tracker_setups(tracker, scan_date)
    print(f"Records: {total} | sealed+compacted this run: {compacted}")

    if args.dry_run:
        print("Dry run -- nothing written.")
        return 0
    if not compacted:
        print("Nothing to compact -- file left untouched.")
        return 0

    m.save_setup_tracker_payload(tracker)
    size_after = path.stat().st_size if path.exists() else 0
    print(f"Size after:  {size_after / 1e6:.1f} MB  (saved {(size_before - size_after) / 1e6:.1f} MB)")
    print(f"Backup rotated to: {path.with_name(path.name + '.bak')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
