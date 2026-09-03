"""The look-counter: one append-only row per registered parameter grid.

Phase 0.13 packet P7. Spec: `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` sec 15.1 (the
registered research question and the trial ledger) and sec 12.4 (combination
control - "a record of every attempted recipe, including failures").

**What it is for.** Sec 15.1 replaces formal multiplicity machinery with a
count: every family records `n_variants_examined` as a FAMILY-LIFETIME number,
and splitting the search into several files never resets it. The widening rule
is `k > 10` implies a 99% interval on holdout AND beating the family-median
holdout. None of that is computable unless something wrote down, BEFORE any
outcome was inspected, how many cells the grid had and what would count as a
pass. This is that something.

**Written at REGISTRATION time.** The row is the pre-declaration: question,
cells, floors, window, who authorized it. A row appended after the numbers came
back is not a pre-declaration, it is a description - which is why `register`
refuses to rewrite an existing trial and why nothing here reads an outcome. The
`outcome` field exists for a LATER packet to fill by appending a superseding
row; this packet only ever writes `status="registered"`.

**Read-only for now.** Nothing in production imports this module. The three
grids that already exist are backfilled from their real authorization pointers
so the ledger starts complete rather than starting empty and pretending the
looks before it did not happen - a lifetime count that begins today would
understate every family it covers.

The file lives beside the warehouse's other diagnostics
(`<store root>/_diagnostics/trial_ledger.jsonl`) and is JSONL because it is
append-only: a row is added, never edited, and a correction is another row.

Every row carries `registered_at` (UTC, aware). Backfilled rows carry the date
their work was AUTHORIZED in plan.md rather than the date this module was
written - a backfilled row stamped "today" would claim the look happened today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # package import
    from .manifest import utc_now
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    from manifest import utc_now  # type: ignore

SCHEMA = "trial_ledger_v1"
LEDGER_FILENAME = "trial_ledger.jsonl"
DIAGNOSTICS_DIRNAME = "_diagnostics"

STATUS_REGISTERED = "registered"
#: Declared, and its window is open. Distinct from `registered` on purpose (P8):
#: `registered` says the question was written down, `collecting` says the clock
#: is running, and only a trial whose declared window has CLOSED may be read for
#: a verdict.
STATUS_COLLECTING = "collecting"
STATUS_ABANDONED = "abandoned"
STATUS_CONCLUDED = "concluded"
STATUSES = (STATUS_REGISTERED, STATUS_COLLECTING, STATUS_ABANDONED, STATUS_CONCLUDED)

#: The grids that already existed when the ledger was written, with the real
#: authorization pointer for each. Backfilled rather than started from zero: sec
#: 15.1's count is family-lifetime, and a ledger that began today would claim
#: these families had been looked at k=0 times.
#:
#: `recipe_id_prefix` is how a row claims its recipes. The alternative - listing
#: all 54 M5-close ids - would put the grid's SIZE in two places, and the two
#: would eventually disagree; the prefix is derived from the same f-string that
#: builds the ids.
BACKFILL_TRIALS: tuple[Mapping[str, Any], ...] = (
    {
        "trial_id": "m5_close_recipe_grid_v1",
        "schema": SCHEMA,
        # The date the work was AUTHORIZED, from plan.md, not the date this
        # file was written. A backfilled row stamped "today" would claim the
        # look happened today, which is the understatement the backfill exists
        # to prevent.
        "registered_at": "2026-08-27T00:00:00+00:00",
        "family": "M5_OPPORTUNITY",
        "question": (
            "For an M5 opportunity entered at the next session's first completed M5 "
            "close, which structural stop (ranked tracker level and close-failure "
            "count) and which fixed-R target produce the best forward record, "
            "against an ATR-placement control?"
        ),
        "failure_mode": (
            "The structural stops win only because the ATR control is placed too "
            "tight, or one stop source dominates because its level is nearly always "
            "unavailable and the recipe silently skips those opportunities."
        ),
        "declared_cells": {
            "stop_source": ["current_anchor", "sma", "ema", "post_earnings_anchor", "post_earnings_candle"],
            "stop_rank": [1, 2, 3],
            "target_r": [1.0, 2.0, 3.0],
            "control_atr_multiple": [0.5, 1.0, 1.5],
        },
        "declared_cell_count": 54,
        "recipe_id_prefix": "m5close_",
        "declared_floors": {
            "min_opportunities_per_cell": 30,
            "note": (
                "`evidence_stats.MIN_REPORTABLE_N` is the desk's one statistics "
                "contract; a cell under it is reported as unmeasured, never as a "
                "weak edge."
            ),
        },
        "declared_window": {
            "kind": "forward_shadow",
            "note": "accrues forward from warehouse capture; no in-sample re-fit",
        },
        "authorization": "docs/ULTIMATE_SETUP_DATABASE_PLAN.md sec 12.4 + sec 19.3 (registered recipe grid)",
        "analysis_unit": "opportunity",
        "status": STATUS_REGISTERED,
        "outcome": "",
        "registered_by": "P7 backfill (grid predates the ledger)",
    },
    {
        "trial_id": "htf_lrsi_entry_grid_v1",
        "schema": SCHEMA,
        # The date the work was AUTHORIZED, from plan.md, not the date this
        # file was written. A backfilled row stamped "today" would claim the
        # look happened today, which is the understatement the backfill exists
        # to prevent.
        "registered_at": "2026-09-01T00:00:00+00:00",
        "family": "HTF_LRSI",
        "question": (
            "Does an efficiency-LRSI cross on a higher timeframe (M30/H1/H2/H4), "
            "through the levels the trader already reads, mark a better entry than "
            "the M5-close entry the same opportunity would otherwise take?"
        ),
        "failure_mode": (
            "The higher timeframes agree with each other so strongly that four "
            "timeframes are one look, not four; or the session-stub exclusion leaves "
            "H4 with too few completed bars to measure at all."
        ),
        "declared_cells": {
            "timeframe": ["M30", "H1", "H2", "H4"],
            "entry": [
                "LONG up 50", "LONG up 20", "SHORT down 50", "SHORT down 80",
            ],
            "target_r": [2.0],
        },
        "declared_cell_count": 16,
        "recipe_id_prefix": "htf_lrsi_",
        "declared_floors": {
            "min_opportunities_per_cell": 30,
            "note": "same statistics contract; sixteen cells also keep the nightly inside its reserve",
        },
        "declared_window": {
            "kind": "forward_shadow",
            "note": "shadow only; the study writes no alert and no score",
        },
        "authorization": "plan.md Phase 0.12 B (higher-timeframe LRSI entry study, shadow only)",
        "analysis_unit": "opportunity",
        "status": STATUS_REGISTERED,
        "outcome": "",
        "registered_by": "P7 backfill (grid predates the ledger)",
    },
    {
        "trial_id": "avwap_band_challenger_v1",
        "schema": SCHEMA,
        # The date the work was AUTHORIZED, from plan.md, not the date this
        # file was written. A backfilled row stamped "today" would claim the
        # look happened today, which is the understatement the backfill exists
        # to prevent.
        "registered_at": "2026-08-26T00:00:00+00:00",
        "family": "AVWAP_BAND_SIGMA",
        "question": (
            "Does OneOption's band - AVWAP(HLC/3) +/- k * stdev(close, 20, "
            "population) - describe the level a move respects better than the "
            "champion's running-deviation sigma?"
        ),
        "failure_mode": (
            "A wider band is read as a further stop. It is not: it is stopped out "
            "less often only when entry sits INSIDE it, and on the parity fixture's "
            "short - entered above both upper bands - the challenger's stop lands "
            "six times TIGHTER. Any stop-out or respect-rate comparison that does "
            "not condition on entry position relative to the band is measuring the "
            "entry, not the band."
        ),
        "declared_cells": {"band_multiple": [1, 2, 3]},
        "declared_cell_count": 3,
        "recipe_id_prefix": "",
        "declared_floors": {
            "min_forward_sessions": 20,
            "note": "plan.md Phase 0.10: >= 20 sessions of forward accrual before T3 counts",
        },
        "declared_window": {
            "kind": "forward_shadow",
            "note": (
                "shadow only; `calc_anchored_vwap_bands` stays frozen (decision 0008) "
                "and every champion aggregate is fenced from the challenger scenario"
            ),
        },
        "authorization": "plan.md Phase 0.10 (AVWAP band challenger, authorized 2026-08-26); docs/AVWAP_BAND_VARIANT_STUDY.md T4",
        "analysis_unit": "tracker_setup",
        "status": STATUS_REGISTERED,
        "outcome": "",
        "registered_by": "P7 backfill (grid predates the ledger)",
    },
    {
        "trial_id": "setup_entry_timing_avwape_first_dev_long_v1",
        "schema": SCHEMA,
        # The paste date, which IS this grid's authorization pointer. Not
        # backfill: this is the one row whose declaration really was made on the
        # day it is stamped with.
        "registered_at": "2026-09-02T00:00:00+00:00",
        "family": "AVWAPE_TO_FIRST_DEV",
        "question": (
            "For a D1 AVWAPE-to-first-dev LONG occurrence, does an entry that WAITS "
            "for confirmation - an M15 acceptance close, an M5 retest of the trigger "
            "level, or an M30 EMA15/21 controlled pullback - earn more net R per "
            "episode than entering at the first completed M5 close of the next "
            "session, under one structural stop?"
        ),
        "failure_mode": (
            "A waiting entry looks better only because it SKIPS the episodes that "
            "went straight down - the confirmation never printed, so no row exists "
            "and the loss is missing from the average rather than counted. The "
            "control's row count per cluster is therefore the denominator to read "
            "first, and a challenger with materially fewer rows is reporting "
            "survivorship, not edge. Second failure: the three challengers agree "
            "with each other so strongly that they are one look and not three."
        ),
        "declared_cells": {
            "entry": [
                "m5_first_close (control)",
                "m15_acceptance_close",
                "m5_retest_trigger",
                "m30_ema15_21_pullback",
            ],
            "target_r": [1.0, 2.0, 3.0],
        },
        "declared_cell_count": 12,
        "recipe_id_prefix": "setupentry_",
        "declared_floors": {
            "min_episodes_per_cell": 30,
            "min_symbols": 5,
            "min_entry_sessions": 5,
            "counted_on": "dependency_cluster_id",
            "note": (
                "`evidence_stats`' contract, counted on the EPISODE and not the row: "
                "twelve readings of one occurrence are twelve views of one trade."
            ),
        },
        "declared_window": {
            "kind": "fixed_forward_sessions",
            "sessions": 20,
            "opens_at": "the first overnight run after 2026-09-02",
            "note": (
                "Fixed at REGISTRATION. No cell is read for a verdict before the "
                "window closes - the point of declaring a window is that it cannot be "
                "shortened once a cell starts looking good."
            ),
        },
        "authorization": (
            "trader packet pasted 2026-09-02 (Phase 0.13 P8); recorded in plan.md as "
            "the Phase 6.1 addendum"
        ),
        "analysis_unit": "opportunity",
        "status": STATUS_COLLECTING,
        "outcome": "",
        "registered_by": "P8 registration (written before any outcome was inspected)",
    },
    {
        "trial_id": "after_like_entry_grid_v1",
        "schema": SCHEMA,
        # The paste date. Declared BEFORE any after-like outcome exists - the
        # simulator that produces them lands in the same commit as this row and
        # runs for the first time on the night after it.
        "registered_at": "2026-09-02T00:00:00+00:00",
        "family": "AFTER_LIKE",
        "question": (
            "For a D1 name the trader LIKED, which day after the like and which "
            "entry rule gives the best net R? Trader, 2026-09-02: *\"if I like a "
            "stock one day it may not be for 3-5 days later that the best entry "
            "is.\"*"
        ),
        "failure_mode": (
            "THREE, and the first is the one that will actually happen. (1) A later "
            "offset looks better because the episodes that went straight down have "
            "no bars left to enter on - the same survivorship the P8 grid names, "
            "with five chances to commit it instead of three, so the DAY-0 row "
            "count is the denominator every other offset is read against. (2) The "
            "unlinked bucket and the linked one are compared as if they were the "
            "same population; they are not - an unlinked like is one the scanner "
            "never found, which is a different kind of name. (3) Twenty cells over "
            "one like episode are twenty views of one decision: the episode is the "
            "unit, and a cell whose n counts ROWS is counting the same opinion "
            "twenty times."
        ),
        "declared_cells": {
            "day_offset": [0, 1, 2, 3, 5],
            "entry": [
                "first_m5_close",
                "m5_retest_trigger",
                "m15_acceptance",
                "m30_ema15_21_pullback",
            ],
            "target_r": [2.0],
            "stop": ["current_anchor:1"],
        },
        "declared_cell_count": 20,
        "recipe_id_prefix": "afterlike_",
        "declared_floors": {
            "min_episodes_per_cell": 30,
            "min_symbols": 5,
            "min_entry_sessions": 5,
            "counted_on": "like_episode",
            "note": (
                "The episode is ONE LIKE on one symbol-side, across all twenty "
                "cells - not one row per cell. `dependency_cluster_id` is keyed on "
                "(symbol, side, like_date) for the same reason: a name liked on "
                "consecutive days is one opinion held twice, not two."
            ),
        },
        "declared_window": {
            "kind": "fixed_forward_sessions",
            "sessions": 20,
            "opens_at": "the first overnight run after 2026-09-02",
            "note": (
                "Fixed at REGISTRATION. No cell is read for a verdict before the "
                "window closes, including by the agent that built it and including "
                "if an early cell looks good."
            ),
        },
        "authorization": (
            "trader packet pasted 2026-09-02 (Phase 0.13 P10 Part C); the trader's "
            "own question, quoted in `question` above"
        ),
        "analysis_unit": "like_episode",
        "status": STATUS_COLLECTING,
        "outcome": "",
        "registered_by": "P10 registration (written before any outcome was inspected)",
    },
    {
        "trial_id": "v1_recipe_library",
        "schema": SCHEMA,
        # The date the work was AUTHORIZED, from plan.md, not the date this
        # file was written. A backfilled row stamped "today" would claim the
        # look happened today, which is the understatement the backfill exists
        # to prevent.
        "registered_at": "2026-08-27T00:00:00+00:00",
        "family": "SHARED_RECIPES",
        "question": (
            "How does each setup perform under the shared house recipes and the "
            "matched controls, rather than under one management policy baked into "
            "the setup's name?"
        ),
        "failure_mode": (
            "The controls are treated as competitors and the 'best of' is reported "
            "across house and control together, which turns a baseline into a "
            "variant and inflates k."
        ),
        "declared_cells": {
            "recipe": [
                "swing_house_v1", "intraday_bounce_v1", "control_fixed_1r2r_v1",
                "control_time_only_v1", "diag_signal_bar_atr_stop_v1",
            ]
        },
        "declared_cell_count": 5,
        "recipe_ids": [
            "swing_house_v1", "intraday_bounce_v1", "control_fixed_1r2r_v1",
            "control_time_only_v1", "diag_signal_bar_atr_stop_v1",
        ],
        "recipe_id_prefix": "",
        "declared_floors": {
            "min_opportunities_per_cell": 30,
            "note": "controls are baselines, never counted as variants examined",
        },
        "declared_window": {"kind": "forward_shadow", "note": "the standing comparison set"},
        "authorization": "docs/ULTIMATE_SETUP_DATABASE_PLAN.md sec 19.3 (the concrete v1 recipes)",
        "analysis_unit": "mixed",
        "status": STATUS_REGISTERED,
        "outcome": "",
        "registered_by": "P7 backfill (library predates the ledger)",
    },
)


def ledger_path(store_root: Path | str) -> Path:
    """Where the ledger lives for one warehouse root."""
    return Path(store_root) / DIAGNOSTICS_DIRNAME / LEDGER_FILENAME


def owner_of(recipe_id: str, trials: Iterable[Mapping[str, Any]] | None = None) -> str:
    """Which trial claims this recipe id, or "" if none does.

    Pure: no store, no file. The membership rule is the row's explicit
    `recipe_ids` first, then its `recipe_id_prefix` - explicit beats derived, so
    a named recipe can never be captured by another row's prefix.
    """
    text = str(recipe_id or "").strip()
    if not text:
        return ""
    rows = list(trials if trials is not None else BACKFILL_TRIALS)
    for row in rows:
        if text in tuple(row.get("recipe_ids") or ()):
            return str(row["trial_id"])
    for row in rows:
        prefix = str(row.get("recipe_id_prefix") or "")
        if prefix and text.startswith(prefix):
            return str(row["trial_id"])
    return ""


def owners_of(recipe_id: str, trials: Iterable[Mapping[str, Any]] | None = None) -> tuple[str, ...]:
    """EVERY trial that would claim this recipe id.

    Separate from `owner_of` because the interesting failure is not "no owner"
    but "two owners": two grids claiming one recipe means one result would be
    counted against both families' lifetime look counts, which is the exact
    double-count sec 15.1's k exists to prevent.
    """
    text = str(recipe_id or "").strip()
    rows = list(trials if trials is not None else BACKFILL_TRIALS)
    found = []
    for row in rows:
        explicit = tuple(row.get("recipe_ids") or ())
        prefix = str(row.get("recipe_id_prefix") or "")
        if (text and text in explicit) or (prefix and text.startswith(prefix)):
            found.append(str(row["trial_id"]))
    return tuple(found)


def load(store_root: Path | str) -> list[dict[str, Any]]:
    """Every row, oldest first. A corrupt line is SKIPPED and counted by nobody -
    the reader's job is to answer what was declared, and a half-written trailing
    line from a killed process is not a declaration."""
    path = ledger_path(store_root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def register(store_root: Path | str, trial: Mapping[str, Any]) -> bool:
    """Append one pre-declaration. Returns whether it was written.

    Refuses to write a `trial_id` the ledger already carries. That is not a
    convenience: rewriting a declaration after the fact is how a grid of 54 cells
    becomes a grid of 3 in the record, and the whole point of writing it before
    the outcomes is that it cannot be edited afterwards. A genuine change of plan
    is a NEW trial id with its own row.
    """
    trial_id = str(trial.get("trial_id") or "").strip()
    if not trial_id:
        raise ValueError("a trial needs an id")
    status = str(trial.get("status") or STATUS_REGISTERED)
    if status not in STATUSES:
        raise ValueError(f"unknown trial status: {status!r}")
    if not str(trial.get("authorization") or "").strip():
        raise ValueError(
            "a trial needs an authorization pointer: an experiment nobody authorized "
            "is not a registered question"
        )
    if any(str(row.get("trial_id") or "") == trial_id for row in load(store_root)):
        return False
    path = ledger_path(store_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # WHEN the declaration was made, stamped here rather than taken from the
    # caller. The ledger's whole claim is that the question was written down
    # before the numbers arrived, and a row with no time cannot support it: an
    # undated declaration and a declaration written afterwards look identical
    # six months later. A row that already carries `registered_at` keeps it,
    # which is what lets the backfilled rows state the date they were really
    # authorized instead of the date this code first ran.
    row = {"schema": SCHEMA, "registered_at": utc_now().isoformat(timespec="seconds")}
    row.update(dict(trial))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return True


def backfill(store_root: Path | str) -> list[str]:
    """Write the grids that predate the ledger. Idempotent; returns what it wrote."""
    return [
        str(trial["trial_id"])
        for trial in BACKFILL_TRIALS
        if register(store_root, trial)
    ]


__all__ = [
    "BACKFILL_TRIALS",
    "LEDGER_FILENAME",
    "SCHEMA",
    "STATUSES",
    "STATUS_ABANDONED",
    "STATUS_COLLECTING",
    "STATUS_CONCLUDED",
    "STATUS_REGISTERED",
    "backfill",
    "ledger_path",
    "load",
    "owner_of",
    "owners_of",
    "register",
]
