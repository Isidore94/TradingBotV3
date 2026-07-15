#!/usr/bin/env python3
"""Replay Technical Integrity outcomes and compare fixed scoring candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from technical_integrity import (
    technical_integrity_calibration_path,
    technical_integrity_events_path,
    write_technical_integrity_calibration_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Point-in-time replay of the advisory Technical Integrity score. "
            "It writes research evidence and never changes the live configuration."
        )
    )
    parser.add_argument("--events", type=Path, default=technical_integrity_events_path())
    parser.add_argument("--output", type=Path, default=technical_integrity_calibration_path())
    args = parser.parse_args(argv)

    report = write_technical_integrity_calibration_report(
        events_path=args.events,
        output_path=args.output,
    )
    gate = report["review_gate"]
    print(
        f"Technical Integrity replay: {report['event_count']} resolved events across "
        f"{report['session_count']} sessions"
    )
    for row in report["configs"]:
        score = "n/a" if row["brier_score"] is None else f"{row['brier_score']:.6f}"
        print(f"  {row['name']}: Brier {score} (n={row['event_count']})")
    if gate["eligible"]:
        print(f"Review eligible; best fixed replay candidate: {report['best_replay_config']}")
    else:
        print(
            "Still building evidence; review requires 100 outcomes / 5 sessions / "
            "20 intact / 20 breaks."
        )
    print(f"Research-only report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
