"""The nightly after-like pass — P10 Part C, shadow only.

It takes the trader's likes, the occurrence each one links to (P10 B2), and the
canonical M5 bars the build has ALREADY materialised, and writes one outcome row
per (like episode, cell) into `outcome_path`. No second data pass, so it costs
simulation time and not another read of the lake.

**The unlinked bucket is a COUNT, not a set of graded cells, and this is a
measured limit rather than an omission.** The registered grid declares one
structural stop, `current_anchor:1`, and that level comes from the occurrence's
own tracker geometry. A like with no occurrence has no anchor, so under the
declared stop there is nothing to place a stop at. The alternatives were both
worse: a substitute stop for the unlinked bucket would mean the grid no longer
has one stop model, so an unlinked-vs-linked difference could be a difference in
stops; and dropping the unlinked likes silently would hide how many of the
trader's likes the scanner never found. So they are counted, named by reason, and
reported beside the graded cells — see `AfterLikeRun.excluded_by_reason`.

Every recipe is `is_diagnostic=True`, the trial is registered before any outcome
is inspected, and nothing here reaches a detector, score, alert, watchlist,
Focus, the review queue or `review_policy.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

try:  # package import
    from . import outcomes as outcomes_module
    from .like_links import BASIS_NONE
except ImportError:  # pragma: no cover - flat sys.path layout
    import outcomes as outcomes_module  # type: ignore
    from like_links import BASIS_NONE  # type: ignore

#: Why a like produced no rows. Every one is COUNTED and named; none is silent.
EXCLUDED_NO_OCCURRENCE = "no_linked_occurrence"
EXCLUDED_NO_BARS = "no_m5_bars_for_symbol"
EXCLUDED_NO_CELL_MEASURABLE = "no_cell_produced_a_row"


@dataclass
class AfterLikeRun:
    """What one pass did. Counts first, because the counts are the caveat."""

    likes_seen: int = 0
    episodes_graded: int = 0
    rows: list[dict] = field(default_factory=list)
    excluded_by_reason: dict[str, int] = field(default_factory=dict)

    def exclude(self, reason: str) -> None:
        self.excluded_by_reason[reason] = self.excluded_by_reason.get(reason, 0) + 1


def simulate_after_like_rows(
    like: Mapping[str, Any],
    occurrence: Mapping[str, Any],
    m5_bars,
    *,
    as_of: datetime,
    computed_at: datetime | None = None,
    run_id: str = "",
) -> list[dict]:
    """Every cell of the grid for ONE like. Cells that cannot measure are absent.

    The derived-series cache is shared across all twenty cells of one like, so
    the four entry rules that need M15/M30 series build each series once per
    like rather than once per cell.
    """
    series_cache: dict = {}
    rows = []
    cluster = outcomes_module.after_like_cluster_id(like)
    for recipe in outcomes_module.AFTER_LIKE_RECIPES:
        row = outcomes_module.simulate_after_like_entry(
            dict(like),
            dict(occurrence),
            m5_bars,
            recipe,
            as_of=as_of,
            computed_at=computed_at,
            run_id=run_id,
            series_cache=series_cache,
        )
        if row is None:
            continue
        # The LIKE is the episode, not the occurrence: twenty cells over one
        # like are twenty views of one decision, and a name liked on two days is
        # one opinion held twice.
        row["dependency_cluster_id"] = cluster
        row["like_event_id"] = str(like.get("event_id") or "")
        # A SYNTHETIC EPISODE ID, not the occurrence's own.
        #
        # `outcome_path`'s grain is (occurrence_id, recipe_id,
        # outcome_definition_id). Two likes on two days that link to the SAME
        # occurrence produce genuinely different rows - the offsets are measured
        # from each like's own session - and under the occurrence's id they would
        # collide on that grain, with the second silently replacing the first.
        #
        # It also keeps an after-like row from being mistaken for an occurrence
        # outcome by any join on `occurrence_id`. What was linked is not lost:
        # `linked_occurrence_id` carries it, and the bronze link dataset holds
        # the pair with its match basis.
        row["linked_occurrence_id"] = str(occurrence.get("occurrence_id") or "")
        row["occurrence_id"] = cluster
        rows.append(row)
    return rows


def run_after_like(
    likes: Iterable[Mapping[str, Any]],
    links_by_event: Mapping[str, Any],
    occurrences_by_id: Mapping[str, Mapping[str, Any]],
    m5_by_symbol: Mapping[str, list],
    *,
    as_of: datetime,
    computed_at: datetime | None = None,
    run_id: str = "",
) -> AfterLikeRun:
    """One pass over the likes. Read-only; the caller publishes the rows.

    Publishing is the caller's job on purpose: this function is pure enough to
    test without a lake, and the build already owns the one place rows are
    sealed into a dataset.
    """
    result = AfterLikeRun()
    for like in likes:
        result.likes_seen += 1
        event_id = str(like.get("event_id") or "")
        link = links_by_event.get(event_id)
        occurrence_id = str(getattr(link, "occurrence_id", "") or "")
        basis = str(getattr(link, "match_basis", BASIS_NONE) or BASIS_NONE)
        if basis == BASIS_NONE or not occurrence_id:
            result.exclude(EXCLUDED_NO_OCCURRENCE)
            continue
        occurrence = occurrences_by_id.get(occurrence_id)
        if not occurrence:
            result.exclude(EXCLUDED_NO_OCCURRENCE)
            continue
        bars = m5_by_symbol.get(str(like.get("symbol") or "").strip().upper()) or []
        if not bars:
            result.exclude(EXCLUDED_NO_BARS)
            continue
        rows = simulate_after_like_rows(
            like,
            occurrence,
            bars,
            as_of=as_of,
            computed_at=computed_at,
            run_id=run_id,
        )
        if not rows:
            result.exclude(EXCLUDED_NO_CELL_MEASURABLE)
            continue
        result.episodes_graded += 1
        result.rows.extend(rows)
    return result
