"""What the outcome step actually simulated, firing by firing (P3 item 4).

`cli._run_outcomes` picks ONE of `OUTCOME_BUCKETS` symbol buckets per firing -
`(day.toordinal() + hour) % 32` - so a family can be absent from a fact pack for
two entirely different reasons: it was measured and produced nothing, or its
symbols have not been in a covered bucket yet. The pack could not tell those
apart, and they are opposite conclusions.

This is the record that separates them. Every firing appends one line naming
the bucket it covered; the pack reads the last window of lines and reports how
much of the ring has been walked.

Three rules it keeps:

* **Append-only.** A firing is a fact about a moment; a rewritten history would
  make the coverage claim unfalsifiable. Reading tolerates a truncated last
  line, because a crash mid-append must cost one record and never the file.
* **It never costs the build.** A failed append returns False and is logged at
  debug. The outcome rows are the product; this is evidence about them.
* **It answers "unknown" rather than "zero".** No file, or an unreadable one,
  yields `None` coverage and a stated reason - never 0 of 32, which reads as a
  measured claim that nothing has been covered.

Location: under the research store's own root, beside the lake it describes,
because this is warehouse operational history. The packet asked for it "beside
the packs" in the AI store; that would have made `research_warehouse.cli` -
the data layer - import `ai_jobs.store`, inverting the one-way dependency the
rest of the tree keeps (`ai_jobs` reads `research_warehouse`, never the
reverse). The reader is `ai_jobs.setup_research`, which already imports this
package, so the pack still gets the number.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

try:  # package import
    from .manifest import utc_now
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    from manifest import utc_now  # type: ignore

_log = logging.getLogger(__name__)

SCHEMA = "outcome_bucket_coverage_v1"
#: Directory and file name under the store root.
COVERAGE_DIRNAME = "_diagnostics"
COVERAGE_FILENAME = "outcome_bucket_coverage.jsonl"
#: How many firings back the default window looks. One full ring: with 32
#: buckets and one firing a day, this is "the last month of nightly runs".
DEFAULT_WINDOW = 32
#: Never read more than this many lines from the tail, whatever the window.
MAX_LINES_READ = 4096


def coverage_path(root: Path) -> Path:
    return Path(root) / COVERAGE_DIRNAME / COVERAGE_FILENAME


def record_firing(
    root: Path | None,
    outcomes_step: Mapping[str, Any] | None,
    *,
    run_id: str = "",
    now: datetime | None = None,
) -> bool:
    """Append one firing's bucket. Returns whether a line was written.

    Takes the step RESULT rather than the bucket, so the caller cannot get the
    field name wrong and so a step that never reached a bucket (NO_OCCURRENCES,
    a refused lock) is skipped here rather than recorded as covering bucket 0.
    """
    if root is None or not isinstance(outcomes_step, Mapping):
        return False
    bucket = outcomes_step.get("bucket")
    total = outcomes_step.get("bucket_count")
    if bucket is None or not total:
        return False
    row = {
        "schema": SCHEMA,
        "at": (now or utc_now()).isoformat(timespec="seconds"),
        "run_id": str(run_id or ""),
        "bucket": int(bucket),
        "bucket_count": int(total),
        "status": str(outcomes_step.get("status") or ""),
        "symbols": int(outcomes_step.get("symbols") or 0),
        "occurrences": int(outcomes_step.get("occurrences") or 0),
    }
    target = coverage_path(root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True
    except OSError as exc:
        # The outcome rows are the product; this is evidence about them.
        _log.debug("Outcome bucket coverage not recorded: %s", exc)
        return False


def read_firings(root: Path | None, *, limit: int = DEFAULT_WINDOW) -> list[dict]:
    """The most recent `limit` firings, oldest first. Never raises."""
    if root is None:
        return []
    target = coverage_path(root)
    try:
        if not target.is_file():
            return []
        with target.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-MAX_LINES_READ:]
    except OSError as exc:
        _log.debug("Outcome bucket coverage unreadable: %s", exc)
        return []
    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A crash mid-append costs one record, never the file.
            continue
        if isinstance(row, dict) and row.get("bucket") is not None:
            rows.append(row)
    return rows[-int(limit):] if limit else rows


def coverage_state(root: Path | None, *, limit: int = DEFAULT_WINDOW) -> dict[str, Any]:
    """How much of the bucket ring the recent firings walked.

    An absent record answers UNKNOWN, with the reason. "0 of 32 covered" would
    be a measured claim, and nobody measured it.
    """
    firings = read_firings(root, limit=limit)
    if not firings:
        return {
            "outcome_buckets_covered": None,
            "outcome_bucket_count": None,
            "outcome_firings_considered": 0,
            "outcome_bucket_coverage_note": (
                "no firing history recorded yet - coverage is UNKNOWN, not zero. "
                "The record starts at the first warehouse build after P3."
            ),
        }
    counts = {int(row.get("bucket_count") or 0) for row in firings if row.get("bucket_count")}
    total = max(counts) if counts else None
    covered = sorted({int(row["bucket"]) for row in firings})
    state: dict[str, Any] = {
        "outcome_buckets_covered": len(covered),
        "outcome_bucket_count": total,
        "outcome_firings_considered": len(firings),
        "outcome_buckets_seen": covered,
        "outcome_coverage_first_at": str(firings[0].get("at") or ""),
        "outcome_coverage_last_at": str(firings[-1].get("at") or ""),
    }
    if total and len(counts) > 1:
        # The ring was resized at some point in this window; say so rather than
        # reporting a fraction of two different denominators.
        state["outcome_bucket_coverage_note"] = (
            f"bucket_count changed within this window ({sorted(counts)}); the "
            "denominator shown is the largest seen."
        )
    return state


__all__ = [
    "SCHEMA",
    "DEFAULT_WINDOW",
    "coverage_path",
    "coverage_state",
    "read_firings",
    "record_firing",
]
