"""Attach the day's environment label to outcome rows, BY SCAN DATE (WS-ENV).

Point-in-time, and that is the whole module: a swing observation is labelled
with the environment of the session it was DECIDED in, never the session it
exited in. Joining on the exit date would tell the trader what kind of tape the
trade finished in - which they cannot know when they take it - and would do it
confidently, because both dates are usually in the store. A wrong label here is
not a blank; it is a plausible number pointing the wrong way.

Cheap by construction: ONE `labels_by_session` read (itself mtime-cached), a
dict lookup per row, and the CALLER'S OWN list and dicts mutated in place. The
Results worker joins tens of thousands of rows on a redraw and a second copy of
that list is pure cost.

Shadow only: nothing here detects, scores, ranks, gates or alerts (plan.md
sec 5). It adds one string to a row a readout is about to group by.
"""

from __future__ import annotations

from typing import Any, Iterable, MutableMapping

import d1_environment_store
from indicators.d1_environment import RULE_VERSION

#: The column the label lands in. One name, everywhere.
ENVIRONMENT_FIELD = "d1_environment"

#: What a row nobody labelled reads. `unknown` is its OWN cell on every surface
#: and is never pooled into a measured one.
UNKNOWN = "unknown"


def attach_environment(
    rows: Iterable[MutableMapping[str, Any]],
    *,
    date_field: str = "scan_date",
    benchmark: str = "SPY",
    rule_version: str = RULE_VERSION,
    labels: dict[str, str] | None = None,
    path: Any = None,
) -> Any:
    """Add `d1_environment` to each row IN PLACE and return the same object.

    `date_field` defaults to `scan_date` - `swing_evidence.SwingOutcomePolicy`'s
    own clock field for the tier outcome file - and a caller that passes another
    column is saying so out loud.

    A present-and-empty date, a missing column and a session the store never
    labelled all read `unknown`: none of them is a claim about the tape. A
    timestamped date (`2026-09-09T13:02:12`) is read as its session.
    """
    table = (
        labels
        if labels is not None
        else d1_environment_store.labels_by_session(
            benchmark=benchmark, rule_version=rule_version, path=path
        )
    )
    for row in rows or ():
        try:
            stamp = str(row.get(date_field) or "")[:10]
        except AttributeError:
            continue
        row[ENVIRONMENT_FIELD] = table.get(stamp, UNKNOWN) if stamp else UNKNOWN
    return rows


def environment_counts(rows: Iterable[MutableMapping[str, Any]]) -> dict[str, int]:
    """`{label: rows}` over already-joined rows. Counting, never a statistic."""
    counts: dict[str, int] = {}
    for row in rows or ():
        label = str(row.get(ENVIRONMENT_FIELD) or UNKNOWN)
        counts[label] = counts.get(label, 0) + 1
    return counts
