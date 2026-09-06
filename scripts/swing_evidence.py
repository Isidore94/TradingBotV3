"""One declared reading of the swing outcome files - packet ST1 item 3.

The trader, 2026-09-06: *"Centralize eligible-row reading across family docs,
tier reports, desk, and Away: one declared outcome definition, horizon,
knowledge basis, maturity rule, window, and missingness policy."*

Before this module there were three readings of ONE file
(`master_avwap_tier_outcomes.csv`):

* `setup_docs._all_family_outcomes` - horizon 5, drop `stale_horizon` True,
  bound to the lately window;
* `autopilot_core.swing_family_records` - the same three rules, written out a
  second time;
* `master_avwap_lib.legacy.build_bot_tier_performance_rows` - the same rows with
  NO stale filter, so the tier report counted what the two trader-facing
  surfaces had thrown away. Measured on the live file (2026-09-04, 19,558 rows):
  5,005 rows carry `stale_horizon` True, 2,989 of them at horizon 5 in the last
  20 sessions - one reader's n and another's differ by that much off one file.

**A policy is DECLARED, never inferred.** :class:`SwingOutcomePolicy` names the
outcome kind, the horizon, the knowledge basis, the maturity rule, the window
and the missingness rule in one frozen object, and
:func:`read_eligible_rows` is the only thing that applies them.

**Every source row lands in exactly one bucket and the bucket has a name.**
`rows` (eligible), `pending` (the horizon has not arrived yet) and `excluded`
(a Counter keyed by reason) always add up to `source_rows` -
:attr:`EligibleRead.reconciles`. A reader that quietly drops rows publishes a
number nobody can check.

**The two policies.** `POLICY_SCANROW_V1` is the file every surface reads
today and its numbers are unchanged by this module. `POLICY_SESSION_V2` reads
the new exact-session file and **has no production caller** - a
`read_eligible_rows` call with it is the seam a later decision flips.

Pure arithmetic and one optional CSV read. No Qt, no network, no scoring.
Nothing here reaches a detector, a score, a rank that gates, an alert, a
watchlist, Focus, the review queue or `review_policy.json`.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidence_stats import LATELY_SESSIONS, SWING_HORIZON_SESSIONS, lately_window

#: The v1 outcome kind: the SIGN OF A CLOSE-TO-CLOSE PERCENT MOVE between two of
#: a symbol's own scan rows. It is not a stop-rule verdict and it is not R.
#: `master_avwap_lib.legacy.build_scan_factor_observation_rows` stamps it.
OUTCOME_KIND_SCANROW_V1 = "favorable_direction_scanrow_v1"

#: The v2 outcome kind: the same question asked of the EXCHANGE CALENDAR - the
#: close on the entry session against the close on the N-th session after it.
#: `master_avwap_lib.session_horizon_outcomes` stamps it.
OUTCOME_KIND_SESSION_V2 = "favorable_direction_session_v2"

#: `tier_for_tracker_row` writes exactly these two, and nothing else may be read
#: as a tier somebody recorded (`legacy.py`, the `ASSIGNED_TIER_VALUES` block).
ASSIGNED_TIER_SOURCE = "assigned"
DERIVED_TIER_SOURCE = "derived_from_bucket"


def outcome_kind_of(row: Mapping[str, Any] | Any) -> str:
    """What kind of outcome this row IS, for a reader that must not guess.

    **An absent or empty `outcome_kind` reads as the v1 kind.** Every row in the
    live file was written before the column existed, and a row whose meaning is
    "unknown" would put every historical number outside every policy. The
    column is additive: it declares what `win` always meant, it does not change
    it.
    """
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    value = str(getter("outcome_kind", "") or "").strip()
    return value or OUTCOME_KIND_SCANROW_V1


@dataclass(frozen=True)
class SwingOutcomePolicy:
    """One declared way of reading a swing outcome file.

    The first six fields are the trader's six words - outcome definition,
    horizon, knowledge basis, maturity rule, window and missingness. The rest
    are how those sentences are APPLIED, kept beside them so a policy cannot say
    one thing and do another.
    """

    #: Which outcome kind these rows are. A row of another kind is not read.
    outcome_kind: str
    #: The ONE declared horizon. The files carry 1/3/5/10 and pooling them
    #: counts one decision up to four times.
    horizon_sessions: int
    #: What was compared with what, in words.
    knowledge_basis: str
    #: When a row is ripe enough to count.
    maturity_rule: str
    #: How far back "lately" reaches, in exchange sessions.
    window_sessions: int
    #: What happens to a row that could not be measured.
    missingness: str
    #: What the horizon is COUNTED IN. This is the whole packet in two words:
    #: v1 counts scan rows, v2 counts exchange sessions.
    horizon_unit: str = "scan rows"
    #: The date column the window is measured on. v1 has only the scan date; v2
    #: knows the session it was MEASURED on, which is what "lately" asks about.
    clock_field: str = "scan_date"
    #: An explicit `stale_horizon` True is dropped (None is kept - uncertainty
    #: is not grounds for deletion).
    drop_stale_horizon: bool = False
    #: A row must say `measured` truthy, else it is excluded as
    #: `unmeasured:<reason>`.
    require_measured: bool = False
    #: The value of `maturity` that means "the horizon has not arrived yet".
    #: Empty disables the check - a v1 row cannot exist until its future scan
    #: row does, so every row in that file is mature by construction.
    immature_value: str = ""


POLICY_SCANROW_V1 = SwingOutcomePolicy(
    outcome_kind=OUTCOME_KIND_SCANROW_V1,
    horizon_sessions=SWING_HORIZON_SESSIONS,
    knowledge_basis="entry_scan_row_close_to_future_scan_row_close",
    maturity_rule=(
        "mature by construction: the row does not exist until the symbol's "
        "own N-th later scan row does"
    ),
    window_sessions=LATELY_SESSIONS,
    missingness="an explicit stale_horizon True is dropped; an unmeasured drift (None) is kept",
    horizon_unit="scan rows",
    clock_field="scan_date",
    drop_stale_horizon=True,
)

POLICY_SESSION_V2 = SwingOutcomePolicy(
    outcome_kind=OUTCOME_KIND_SESSION_V2,
    horizon_sessions=SWING_HORIZON_SESSIONS,
    knowledge_basis="entry_session_close_to_target_session_close",
    maturity_rule="the target session must have closed on or before the last completed session",
    window_sessions=LATELY_SESSIONS,
    missingness="an unmeasured row is excluded BY ITS REASON and never counted as a loss",
    horizon_unit="exchange sessions",
    # The v2 row knows the session it was measured on, so "lately" is asked of
    # the target session rather than of the entry - the measurement is the event
    # the window is about.
    clock_field="target_session",
    require_measured=True,
    immature_value="immature",
)


@dataclass(frozen=True)
class EligibleRead:
    """What one policy made of one set of rows. Everything is accounted for."""

    policy: SwingOutcomePolicy
    rows: list[dict] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    excluded: Counter = field(default_factory=Counter)
    source_rows: int = 0
    window: tuple[str, str] = ("", "")

    @property
    def reconciles(self) -> bool:
        """Eligible + pending + excluded == source rows. Never a silent drop."""
        return len(self.rows) + len(self.pending) + sum(self.excluded.values()) == self.source_rows

    @property
    def coverage(self) -> str:
        return f"{len(self.rows)} eligible / {len(self.pending)} pending / {sum(self.excluded.values())} excluded"


def _as_rows(path_or_rows: Any) -> list[dict]:
    if isinstance(path_or_rows, (str, Path)):
        try:
            with open(path_or_rows, newline="", encoding="utf-8-sig") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except OSError:
            # A missing file is zero rows, not a raised reader: every surface
            # here renders reference material the trader opens mid-session.
            return []
    if isinstance(path_or_rows, Mapping):
        return [dict(path_or_rows)]
    return [dict(row) for row in (path_or_rows or ()) if isinstance(row, Mapping)]


def _int_or_none(value: Any) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def read_eligible_rows(
    path_or_rows: Any,
    policy: SwingOutcomePolicy,
    *,
    end: Any = None,
    window: Sequence[str] | None = None,
) -> EligibleRead:
    """The ONE eligible-row read. Every surface goes through it.

    `path_or_rows` is a CSV path or an iterable of row mappings. `end` closes
    the lately window (default: today); `window` overrides it outright for a
    caller that already declared one.

    The order of the checks is deliberate:

    1. **unreadable** - a present-and-empty `horizon_sessions` is not a zero.
    2. **duplicate** - the same `observation_id` twice is one decision.
    3. **wrong_horizon** - the file carries four horizons and pooling them
       counts one decision up to four times.
    4. **pending** - an immature row is checked BEFORE the window, because "the
       horizon has not arrived" and "the measurement is old" are different facts
       and the immature one is not an exclusion at all.
    5. **outside_window**.
    6. **the missingness rule** - `stale_horizon` for v1, `unmeasured:<reason>`
       for v2.
    """
    rows_in = _as_rows(path_or_rows)
    if window is not None:
        first, last = str(window[0]), str(window[1])
    else:
        first, last = lately_window(end, sessions=int(policy.window_sessions))

    eligible: list[dict] = []
    pending: list[dict] = []
    excluded: Counter = Counter()
    seen_ids: set[str] = set()

    for row in rows_in:
        horizon = _int_or_none(row.get("horizon_sessions"))
        if horizon is None:
            excluded["unreadable"] += 1
            continue
        observation_id = str(row.get("observation_id") or "").strip()
        if observation_id:
            if observation_id in seen_ids:
                excluded["duplicate"] += 1
                continue
            seen_ids.add(observation_id)
        if horizon != int(policy.horizon_sessions):
            excluded["wrong_horizon"] += 1
            continue
        if policy.immature_value:
            maturity = str(row.get("maturity") or "").strip().lower()
            if maturity == policy.immature_value:
                pending.append(row)
                continue
        stamp = str(row.get(policy.clock_field) or "")[:10]
        if stamp and not (first <= stamp <= last):
            excluded["outside_window"] += 1
            continue
        if policy.drop_stale_horizon and str(row.get("stale_horizon") or "").strip().lower() == "true":
            excluded["stale_horizon"] += 1
            continue
        if policy.require_measured and not _is_true(row.get("measured")):
            reason = str(row.get("unmeasured_reason") or "").strip() or "unspecified"
            excluded[f"unmeasured:{reason}"] += 1
            continue
        eligible.append(row)

    return EligibleRead(
        policy=policy,
        rows=eligible,
        pending=pending,
        excluded=excluded,
        source_rows=len(rows_in),
        window=(first, last),
    )


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def is_stale_horizon(row: Mapping[str, Any]) -> bool:
    """An EXPLICIT `stale_horizon` True. None means unmeasured, and stays."""
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    return str(getter("stale_horizon", "") or "").strip().lower() == "true"


def tier_split(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """`{assigned, derived, unknown}` from `tier_source`.

    The trader, 2026-09-06: *"Keep bucket-derived historic tiers separate from
    tiers actually assigned at the decision time. Never use reconstructed labels
    to validate shipped S/A performance."*

    A row whose `tier_source` cell is present and empty is UNKNOWN, never
    assigned: the live file was written before the column existed, and reading a
    blank as a recorded decision is exactly the mistake this counts.
    """
    counts = {"assigned": 0, "derived": 0, "unknown": 0}
    for row in rows or ():
        getter = row.get if hasattr(row, "get") else lambda key, default=None: default
        source = str(getter("tier_source", "") or "").strip().lower()
        if source == ASSIGNED_TIER_SOURCE:
            counts["assigned"] += 1
        elif source.startswith("derived"):
            counts["derived"] += 1
        else:
            counts["unknown"] += 1
    return counts


def describe(policy: SwingOutcomePolicy, read: EligibleRead, *, max_reasons: int = 3) -> str:
    """One short line for a surface that shows the rate.

    Says the outcome kind, the horizon IN ITS OWN UNIT, the window and the
    coverage - so a reader can tell a 5-scan-row move from a 5-session one
    without opening the file.
    """
    horizon = f"{int(policy.horizon_sessions)} {policy.horizon_unit}"
    first, last = read.window
    span = f"{first}..{last}" if first and last else f"last {int(policy.window_sessions)} sessions"
    line = (
        f"{policy.outcome_kind} over {horizon}, "
        f"last {int(policy.window_sessions)} sessions ({span}): {read.coverage}"
    )
    if read.excluded:
        ordered = sorted(read.excluded.items(), key=lambda item: (-item[1], item[0]))
        shown = ordered[:max_reasons]
        detail = ", ".join(f"{name} {count}" for name, count in shown)
        if len(ordered) > len(shown):
            detail += f", +{len(ordered) - len(shown)} more"
        line += f" ({detail})"
    return line
