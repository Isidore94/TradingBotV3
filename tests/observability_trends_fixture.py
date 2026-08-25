"""Freeze a representative observability trend report — P1.4 (packet W7).

The inputs are a HAND-WRITTEN, representative set of run manifests and job-ledger
rows rather than a copy of one machine's diagnostics. That is deliberate: a
fixture cut from live files would drift the moment the desk ran again, would
carry that machine's symbol counts into the repo, and would freeze whatever
happened to be true one evening rather than the shapes the reader has to handle.

The shapes it deliberately contains, because each one is a rule in the reader:

* a phase present in the recent window and absent from the baseline (no change
  may be computed);
* a phase present in both, slower (a change must be);
* a phase missing from one recent manifest (missing, never zero);
* a provider family with attempts and failures, and one with none at all (no
  failure rate, rather than a 0% one);
* a failed run carrying an error string;
* an overnight job with a mixed record.

Regenerate with:

    .venv\\Scripts\\python.exe tests/observability_trends_fixture.py --note "why"

It refuses to write a CHANGED expectation without a note, the same rule the
journal characterization fixture keeps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for path in (str(SCRIPTS_DIR), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from diagnostics import observability_trends as trends  # noqa: E402

AS_OF = "2026-08-25T03:00:00+00:00"
WINDOW = 3

MANIFESTS: list[dict[str, Any]] = [
    # --- baseline window ---------------------------------------------------
    {
        "schema": "run_manifest_v1", "run_id": "master_scan-baseline-1",
        "job_type": "master_scan", "started_at": "2026-08-18T20:05:00+00:00",
        "ended_at": "2026-08-18T20:11:00+00:00", "status": "ok", "error": "",
        "total_seconds": 360.0,
        "phases": [
            {"label": "prep+fetch+priority", "seconds": 200.0},
            {"label": "output writes", "seconds": 88.0},
        ],
        "counters": {
            "symbols_processed": 1100,
            "provider.daily_bars.lookup": 1150,
            "provider.daily_bars.cache_hit": 1145,
            "provider.daily_bars.attempt.yahoo": 5,
            "provider.daily_bars.failure.yahoo": 0,
            "provider.daily_bars.success.yahoo": 5,
        },
    },
    {
        "schema": "run_manifest_v1", "run_id": "master_scan-baseline-2",
        "job_type": "master_scan", "started_at": "2026-08-19T20:05:00+00:00",
        "ended_at": "2026-08-19T20:11:30+00:00", "status": "ok", "error": "",
        "total_seconds": 390.0,
        "phases": [
            {"label": "prep+fetch+priority", "seconds": 210.0},
            {"label": "output writes", "seconds": 90.0},
        ],
        "counters": {
            "symbols_processed": 1110,
            "provider.daily_bars.lookup": 1160,
            "provider.daily_bars.cache_hit": 1150,
            "provider.daily_bars.attempt.yahoo": 10,
            "provider.daily_bars.failure.yahoo": 1,
            "provider.daily_bars.success.yahoo": 9,
        },
    },
    {
        "schema": "run_manifest_v1", "run_id": "master_scan-baseline-3",
        "job_type": "master_scan", "started_at": "2026-08-20T20:05:00+00:00",
        "ended_at": "2026-08-20T20:11:00+00:00", "status": "ok", "error": "",
        "total_seconds": 372.0,
        "phases": [
            {"label": "prep+fetch+priority", "seconds": 205.0},
            {"label": "output writes", "seconds": 92.0},
        ],
        "counters": {
            "symbols_processed": 1105,
            "provider.daily_bars.lookup": 1155,
            "provider.daily_bars.cache_hit": 1150,
            "provider.daily_bars.attempt.yahoo": 5,
            "provider.daily_bars.failure.yahoo": 0,
            "provider.daily_bars.success.yahoo": 5,
        },
    },
    # --- recent window -----------------------------------------------------
    {
        "schema": "run_manifest_v1", "run_id": "master_scan-recent-1",
        "job_type": "master_scan", "started_at": "2026-08-21T20:05:00+00:00",
        "ended_at": "2026-08-21T20:13:00+00:00", "status": "ok", "error": "",
        "total_seconds": 470.0,
        "phases": [
            {"label": "prep+fetch+priority", "seconds": 260.0},
            {"label": "output writes", "seconds": 95.0},
            # New in the recent window: no baseline to compare against.
            {"label": "output/scan-factors", "seconds": 62.0},
            {"label": "TOTAL (theta enrichment deferred)", "seconds": 470.0},
        ],
        "counters": {
            "symbols_processed": 1120,
            "provider.daily_bars.lookup": 1176,
            "provider.daily_bars.cache_hit": 1171,
            "provider.daily_bars.attempt.yahoo": 5,
            "provider.daily_bars.failure.yahoo": 2,
            "provider.daily_bars.success.yahoo": 3,
            # A family that never dialled out: no failure rate, not a 0% one.
            "provider.earnings_calendar.lookup": 342,
            "provider.earnings_calendar.cache_hit": 342,
        },
    },
    {
        "schema": "run_manifest_v1", "run_id": "master_scan-recent-2",
        "job_type": "master_scan", "started_at": "2026-08-22T20:05:00+00:00",
        "ended_at": "2026-08-22T20:13:20+00:00", "status": "ok", "error": "",
        "total_seconds": 480.0,
        "phases": [
            {"label": "prep+fetch+priority", "seconds": 265.0},
            # `output writes` deliberately missing from this one.
            {"label": "output/scan-factors", "seconds": 64.0},
        ],
        "counters": {
            "symbols_processed": 1122,
            "provider.daily_bars.lookup": 1180,
            "provider.daily_bars.cache_hit": 1172,
            "provider.daily_bars.attempt.yahoo": 8,
            "provider.daily_bars.failure.yahoo": 1,
            "provider.daily_bars.success.yahoo": 7,
            "provider.earnings_calendar.lookup": 340,
            "provider.earnings_calendar.cache_hit": 340,
        },
    },
    {
        "schema": "run_manifest_v1", "run_id": "master_scan-recent-3",
        "job_type": "master_scan", "started_at": "2026-08-24T20:05:00+00:00",
        "ended_at": "2026-08-24T20:09:00+00:00", "status": "failed",
        "error": "IB refused the connection after 3 attempts",
        "total_seconds": 240.0,
        "phases": [
            {"label": "prep+fetch+priority", "seconds": 240.0},
            {"label": "output writes", "seconds": 0.5},
            {"label": "output/scan-factors", "seconds": 60.0},
        ],
        "counters": {
            "symbols_processed": 400,
            "provider.daily_bars.lookup": 420,
            "provider.daily_bars.cache_hit": 400,
            "provider.daily_bars.attempt.ibkr": 20,
            "provider.daily_bars.failure.ibkr": 20,
            "provider.earnings_calendar.lookup": 120,
            "provider.earnings_calendar.cache_hit": 120,
        },
    },
]

JOB_ROWS: list[dict[str, Any]] = [
    {"job": "journal_import", "status": "ok", "session_date": "2026-08-20"},
    {"job": "journal_import", "status": "ok", "session_date": "2026-08-21"},
    {"job": "ai_summary", "status": "ok", "session_date": "2026-08-20"},
    {"job": "ai_summary", "status": "degraded_no_narrative", "session_date": "2026-08-21"},
    {"job": "ticker_briefs", "status": "failed", "session_date": "2026-08-21"},
    {"job": "ticker_briefs", "status": "ok", "session_date": "2026-08-20"},
    {"job": "veto_cohort_grading", "status": "skipped", "session_date": "2026-08-22"},
]

EXPECTED_KEYS = ("stage_latency", "providers", "jobs", "runs", "coverage")


def _generate() -> dict[str, Any]:
    report = trends.build_trends(
        manifests=MANIFESTS,
        job_rows=JOB_ROWS,
        window=WINDOW,
        now=datetime.fromisoformat(AS_OF),
    )
    return {key: report[key] for key in EXPECTED_KEYS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", default="",
                        help="why the expected output changed. Required whenever it did.")
    args = parser.parse_args(argv)

    from conftest import FIXTURES_DIR, _canonical_json, validate_fixture_contract

    captured = _generate()
    if _generate() != captured:  # pragma: no cover - a golden must be reproducible
        print("REFUSED: the trend report differed between two runs", file=sys.stderr)
        return 1

    path = FIXTURES_DIR / f"{trends.FIXTURE_NAME}.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    changed = [key for key in EXPECTED_KEYS if key in previous and previous[key] != captured[key]]
    if changed and not args.note.strip():
        print(
            "REFUSED: the expected output changed in section(s) "
            f"{', '.join(changed)} and no --note was given.\n"
            'Re-run with --note "why this changed and who approved it".',
            file=sys.stderr,
        )
        return 1

    payload: dict[str, Any] = {
        "schema": "observability_trends/v1",
        "feature_version": previous.get("feature_version") or "p1.4-trends-v1",
        "universe_version": "n/a (diagnostics fixture; no symbol universe is read)",
        "provider_assumptions": (
            "No broker, network or filesystem access. The inputs are hand-written "
            "run manifests and job-ledger rows chosen to contain each shape the "
            "reader has a rule for, not a copy of any machine's diagnostics."
        ),
        "acquired_at": previous.get("acquired_at") or "2026-08-25T03:00:00+00:00",
        "as_of": AS_OF,
        "numeric_tolerance": 0.0,
        "intentional_difference": args.note.strip() or previous.get("intentional_difference") or "",
        "raw_input_keys": ["manifests", "job_rows"],
        "expected_keys": list(EXPECTED_KEYS),
        "configuration": {"window": WINDOW},
        "manifests": MANIFESTS,
        "job_rows": JOB_ROWS,
        **captured,
    }
    import hashlib

    payload["raw_input_sha256"] = hashlib.sha256(
        _canonical_json({"manifests": MANIFESTS, "job_rows": JOB_ROWS})
    ).hexdigest()
    validate_fixture_contract(payload, trends.FIXTURE_NAME)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(captured['stage_latency'])} stage rows, "
          f"{len(captured['providers'])} provider families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
