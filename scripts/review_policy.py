#!/usr/bin/env python3
"""The AI-authored review policy: the contract between reviewer and desk.

Phase 2 of the review-learning loop deliberately keeps the *decisions* out of
code: an AI (Fable or Sol - see AGENTS.md "Review-learning loop") reads the
scoreboard documents and writes this policy file; the Alert Center merely
applies it. Each rule targets one (dimension, segment) pair from the
scoreboard's segmentation and may carry:

- ``priority_delta``  - review-queue ordering nudge, clamped to +/-5. Positive
  floats matching alerts toward the front of the visual review queue.
- ``annotation``      - one line shown on the chart when a matching alert is
  under review ("you usually skip these; passed avg +0.6R").
- ``watch_kind`` / ``fill_source`` - the trader's usual arm for this segment,
  surfaced as a hint (never auto-armed).

``draft_policy_from_state`` translates the scoreboard's blind spots and leaks
into a mechanical draft (blind spot -> boost + note, leak -> demote + note).
The AI reviewer curates that draft - edits, prunes, adds context the counting
cannot see - and saves the result as ``review_policy.json``. A hand-written
policy without a draft is equally valid.

Hard rule inherited from the house style: the policy RANKS AND ANNOTATES
ONLY. Nothing in it can hide an alert, and the Alert Center must never grow
a code path that lets it.

Run:
    .venv/Scripts/python.exe scripts/review_policy.py --draft   # state -> draft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import (  # noqa: E402
    REVIEW_POLICY_DRAFT_FILE,
    REVIEW_POLICY_FILE,
    REVIEW_PREFERENCE_STATE_FILE,
)

REVIEW_POLICY_SCHEMA = "review_policy_v1"
MAX_PRIORITY_DELTA = 5
DRAFT_BLIND_SPOT_DELTA = 2
DRAFT_LEAK_DELTA = -2


@dataclass
class PolicyRule:
    """One segment-level directive from the AI reviewer."""

    dimension: str
    segment: str
    priority_delta: int = 0
    annotation: str = ""
    watch_kind: str = ""
    fill_source: str = ""

    def key(self) -> tuple[str, str]:
        return (self.dimension, self.segment)


def _clamp_delta(value) -> int:
    try:
        delta = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(-MAX_PRIORITY_DELTA, min(MAX_PRIORITY_DELTA, delta))


def load_review_policy(path: Path = REVIEW_POLICY_FILE) -> list[PolicyRule]:
    """Rules from disk; malformed rows are skipped, never fatal."""
    path = Path(path)
    try:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    rules = []
    for row in payload.get("rules") or []:
        if not isinstance(row, dict):
            continue
        dimension = str(row.get("dimension") or "").strip()
        segment = str(row.get("segment") or "").strip()
        if not dimension or not segment:
            continue
        rules.append(
            PolicyRule(
                dimension=dimension,
                segment=segment,
                priority_delta=_clamp_delta(row.get("priority_delta")),
                annotation=str(row.get("annotation") or "").strip(),
                watch_kind=str(row.get("watch_kind") or "").strip(),
                fill_source=str(row.get("fill_source") or "").strip(),
            )
        )
    return rules


def save_review_policy(
    rules: Iterable[PolicyRule],
    path: Path = REVIEW_POLICY_FILE,
    *,
    author: str = "",
    notes: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": REVIEW_POLICY_SCHEMA,
        "generated_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "author": author,
        "notes": notes,
        "rules": [asdict(rule) for rule in rules],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        os.replace(temp_name, path)
    except OSError:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return payload


def draft_policy_from_state(state: dict[str, Any]) -> list[PolicyRule]:
    """Mechanical draft from the scoreboard callouts, for the AI to curate."""
    rules: dict[tuple[str, str], PolicyRule] = {}
    for entry in state.get("blind_spots") or []:
        dimension = str(entry.get("dimension") or "")
        segment = str(entry.get("segment") or "")
        if not dimension or not segment:
            continue
        if "passed_r_avg" in entry:
            measured = f"passed avg {entry['passed_r_avg']:+.2f}R (n={entry['passed_r_n']})"
        else:
            measured = (
                f"passed avg {entry.get('passed_fwd_avg_pct', 0):+.1f}% "
                f"(n={entry.get('passed_fwd_n', 0)})"
            )
        rules[(dimension, segment)] = PolicyRule(
            dimension=dimension,
            segment=segment,
            priority_delta=DRAFT_BLIND_SPOT_DELTA,
            annotation=(
                f"Blind spot: you take {entry.get('take_rate', 0) * 100:.0f}% of "
                f"{segment}; {measured}."
            ),
        )
    for entry in state.get("leaks") or []:
        dimension = str(entry.get("dimension") or "")
        segment = str(entry.get("segment") or "")
        if not dimension or not segment:
            continue
        if "taken_r_avg" in entry:
            measured = f"taken avg {entry['taken_r_avg']:+.2f}R (n={entry['taken_r_n']})"
        else:
            measured = (
                f"taken avg {entry.get('taken_fwd_avg_pct', 0):+.1f}% "
                f"(n={entry.get('taken_fwd_n', 0)})"
            )
        rules[(dimension, segment)] = PolicyRule(
            dimension=dimension,
            segment=segment,
            priority_delta=DRAFT_LEAK_DELTA,
            annotation=(
                f"Leak: you take {entry.get('take_rate', 0) * 100:.0f}% of "
                f"{segment}; {measured}."
            ),
        )
    return list(rules.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review policy tooling.")
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Generate review_policy_draft.json from the scoreboard state.",
    )
    parser.add_argument("--state", type=Path, default=REVIEW_PREFERENCE_STATE_FILE)
    parser.add_argument("--out", type=Path, default=REVIEW_POLICY_DRAFT_FILE)
    args = parser.parse_args(argv)

    if not args.draft:
        rules = load_review_policy()
        print(f"{len(rules)} active rule(s) in {REVIEW_POLICY_FILE}")
        for rule in rules:
            print(f"  {rule.dimension}={rule.segment}: delta {rule.priority_delta:+d} {rule.annotation}")
        return 0

    try:
        state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"No readable scoreboard state at {args.state}; run review_learning.py first.")
        return 1
    rules = draft_policy_from_state(state)
    save_review_policy(
        rules,
        args.out,
        author="draft_policy_from_state",
        notes="Mechanical draft from scoreboard callouts. Curate before promoting to review_policy.json.",
    )
    print(f"{len(rules)} draft rule(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
