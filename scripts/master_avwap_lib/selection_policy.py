"""Which observation of a thesis becomes the episode that gets graded.

Packet ST4 (2026-09-06). The Setup Tracker rescans the same thesis every day
it still looks like a setup, so one *(symbol, side, anchor_date,
setup_family)* thesis leaves many rows behind. Something has to pick which of
those rows is "the trade", and today that choice reads the outcome:
``_dedupe_recent_tracker_family_rows`` sorts ``(0 if closed_setups > 0 else 1,
scan_date)``, so a LATER rescan that happens to have closed beats the EARLIER
entry the trader could actually have taken. The 08-10 open entry loses to the
08-15 closed one; the recorded episode is the one that resolved.

This module names both policies so the choice can be compared instead of
assumed:

``closed_first_v1``
    Today's rule, byte-identical, and still the default everywhere. Prefer a
    row that has closed, then the earliest scan date. One row per thesis, so a
    genuine second entry on the same anchor is invisible.

``first_actionable_v2``
    The challenger, evidence-only. The episode identity gains an
    ``attempt_index`` and the selected row is fixed BEFORE any outcome is
    known: attempt 1 is the earliest scan row of the thesis, and a later row
    opens attempt k+1 only when the previous attempt's representative scenario
    had already CLOSED. A rescan of a live attempt is one more observation of
    the same episode, never a new one, and an attempt still running stays
    pending.

Nothing here reaches a detector, a score, a rank, an alert, a watchlist, Focus,
the review queue or ``review_policy.json``. ``first_actionable_v2`` is reachable
only through an explicit keyword argument and through
``scripts/tracker_selection_compare.py``; the trader's decision on whether to
switch is a separate question built on that comparison.
"""

from __future__ import annotations

from typing import Iterable

SELECTION_CLOSED_FIRST_V1 = "closed_first_v1"
SELECTION_FIRST_ACTIONABLE_V2 = "first_actionable_v2"

#: Today's policy. Every default call site resolves to this and moves no
#: number; the golden `tests/fixtures/st4_family_rows_golden.csv` was pinned
#: from `main` before this module existed and proves it.
DEFAULT_SELECTION_POLICY = SELECTION_CLOSED_FIRST_V1

SELECTION_POLICIES = (SELECTION_CLOSED_FIRST_V1, SELECTION_FIRST_ACTIONABLE_V2)

#: The row that OPENED an attempt - the observation a trader could have acted
#: on - and every later observation of that same live attempt.
ATTEMPT_ROLE_FIRST_ACTIONABLE = "first_actionable"
ATTEMPT_ROLE_RESCAN = "rescan"

#: The four fields that name one thesis. `attempt_index` is the fifth field of
#: an episode identity under v2; under v1 every row of a thesis is attempt 1.
EPISODE_KEY_FIELDS = ("symbol", "side", "anchor_date", "setup_family")

REENTRY_RULE_V2 = """first_actionable_v2 re-entry rule (packet ST4, 2026-09-06).

An episode is (symbol, side, anchor_date, setup_family, attempt_index).

* Attempt 1 is the EARLIEST scan row of the thesis. It is chosen without
  reading any outcome: only the scan date decides.
* A later scan row starts attempt k+1 ONLY when the previous attempt's
  representative scenario has CLOSED - its recorded exit date is strictly
  before the new scan date. That is a declared entry rule: the trade was out
  before the next one could be taken.
* A rescan while the attempt is still open is the SAME attempt. It is counted
  as an observation of that episode and never as a new episode.
* An attempt whose representative scenario has not closed stays PENDING. It
  contributes neither a win nor a loss, and its R is never replaced by the
  mean of any alternate exit plan that did close.

The exit date the rule reads is `representative_exit_date` on the row, which
`build_recent_tracker_setup_family_rows` stamps from the representative
scenario's own last recorded event. A scenario carries no scalar exit field;
the recorded exit is the `trade_date` of the last entry of its `events` list,
and an open scenario has that list PRESENT and EMPTY.
"""


def _text(value: object) -> str:
    return str(value or "").strip()


def _resolve(policy: object) -> str:
    text = _text(policy) or DEFAULT_SELECTION_POLICY
    if text not in SELECTION_POLICIES:
        raise ValueError(
            f"unknown selection policy {text!r}; expected one of {SELECTION_POLICIES}"
        )
    return text


def episode_key(row: dict) -> tuple[str, str, str, str]:
    """The thesis key. Identical to the key `_dedupe_recent_tracker_family_rows`
    has always used, including its un-normalized `side` - v1 must not move."""
    return (
        str(row.get("symbol") or ""),
        str(row.get("side") or ""),
        str(row.get("anchor_date") or ""),
        str(row.get("setup_family") or "general"),
    )


def _scan_date(row: dict) -> str:
    return str(row.get("scan_date") or "")


def representative_exit_date(row: dict) -> str:
    """The ISO date the row's representative scenario CLOSED, or '' while open."""
    return _text(row.get("representative_exit_date"))


def _grouped(rows: Iterable[dict]) -> dict[tuple[str, str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(episode_key(row), []).append(row)
    return groups


def _v1_pick(group: list[dict]) -> dict:
    """Today's choice: a closed record first, then the earliest scan date.

    Moved here verbatim from `_dedupe_recent_tracker_family_rows` so there is
    exactly ONE implementation of the shipped policy and it cannot drift from
    the one the challenger is compared against. `sorted` is stable, so ties
    still keep input order.
    """
    return sorted(
        group,
        key=lambda r: (
            0 if int(r.get("closed_setups", 0) or 0) > 0 else 1,
            str(r.get("scan_date") or ""),
        ),
    )[0]


def assign_attempts(rows, *, policy: str = DEFAULT_SELECTION_POLICY) -> list[dict]:
    """Stamp `attempt_index` and `role` on every row, in input order.

    Every row is returned - this names observations, it never drops one. The
    rows are mutated in place and also returned, because the family builder
    wants the stamps on the rows it is about to aggregate.
    """
    policy = _resolve(policy)
    row_list = list(rows)
    groups = _grouped(row_list)

    if policy == SELECTION_CLOSED_FIRST_V1:
        for group in groups.values():
            chosen = _v1_pick(group)
            for row in group:
                row["attempt_index"] = 1
                row["role"] = (
                    ATTEMPT_ROLE_FIRST_ACTIONABLE
                    if row is chosen
                    else ATTEMPT_ROLE_RESCAN
                )
                row["selection_policy"] = policy
        return row_list

    for group in groups.values():
        ordered = sorted(group, key=_scan_date)
        attempt = 0
        open_attempt_exit = ""
        for index, row in enumerate(ordered):
            starts_new_attempt = index == 0 or (
                bool(open_attempt_exit) and open_attempt_exit < _scan_date(row)
            )
            if starts_new_attempt:
                attempt += 1
                open_attempt_exit = representative_exit_date(row)
                row["role"] = ATTEMPT_ROLE_FIRST_ACTIONABLE
            else:
                row["role"] = ATTEMPT_ROLE_RESCAN
            row["attempt_index"] = attempt
            row["selection_policy"] = policy
    return row_list


def select_episode_rows(rows, *, policy: str = DEFAULT_SELECTION_POLICY) -> list[dict]:
    """One row per episode: the v1 representative, or every v2 attempt opener.

    The returned rows are the SAME dict objects that came in, so the caller's
    downstream weighting reads exactly what it read before.
    """
    policy = _resolve(policy)
    row_list = list(rows)
    if policy == SELECTION_CLOSED_FIRST_V1:
        selected = [_v1_pick(group) for group in _grouped(row_list).values()]
        for row in selected:
            row.setdefault("attempt_index", 1)
            row["selection_policy"] = policy
        return selected

    assign_attempts(row_list, policy=policy)
    return [
        row for row in row_list if row.get("role") == ATTEMPT_ROLE_FIRST_ACTIONABLE
    ]
