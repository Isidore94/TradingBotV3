"""Tag the journal's backlog once, provisionally, so the trader can review it.

Phase 0.13 packet P6a, authorized by the trader on 2026-09-01: *"let's get Opus
to do the tagging and I can review after"*.

**Why this exists.** 193 trades are in the journal and exactly ONE carries a
setup tag the trader typed. Every per-setup statistic in the desk - the
analytics group, the walkaway report, the preference join - therefore rests on a
single row, and no amount of nightly evidence fixes that, because the missing
thing is a human decision about 155 closed trades. Tagging them by hand is the
work nobody does; tagging them by machine is the work nobody trusts. This does
the machine half and marks it as such, so the trader's job shrinks from typing
155 tags to confirming or correcting them.

**The one authorized exception, and its boundary.** R7 invariant I7 says the
trader owns ``trade_annotations``: no import, no nightly job and no model writes
there. This module is the single exception, and it pays for that with a
permanent mark - every tag it writes carries ``tag_status='provisional'``, which
no analytics group counts as the trader's and which no rebuild erases. Nothing
here reads an outcome: the candidates come from the scanner's own output files
by symbol, date window and side, exactly as they do behind the Trades tab's
suggestion list, so no tag can ever be derived from whether the trade made
money.

**What it will not do.**

- It never overwrites a confirmed tag. The refusal lives in
  ``JournalStore.apply_provisional_tags``, not in this file, because an
  exception that depends on its caller remembering a rule is not a boundary.
- It never writes ``tag_corrections``. That table is the trader's feedback to
  the tagger - a row in it raises a tag's confidence for that symbol forever -
  and a machine writing into it would be the tagger teaching itself from its own
  guesses.
- It never guesses below the threshold. A low-confidence tag parked in
  ``setup_tags`` would be counted by every statistic that groups on setups,
  which is the exact circularity the tagging rules forbid; those trades get a
  ``needs_review`` marker and NO tag.
- It is idempotent. A second run over unchanged evidence writes nothing and
  appends no adjustment, so it is safe to run again after a scan.

Dry run by default. ``--apply`` is the only way a trader-owned row is written;
the dry run does re-derive ``auto_tag_candidates``, which is the machine's own
table and is what the threshold has to be measured against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from journal_analytics import is_link_candidate
from journal_store import (
    PROVISIONAL_TAG_ADJUSTMENT,
    TAG_STATUS_CONFIRMED,
    TAG_STATUS_NEEDS_REVIEW,
    TAG_STATUS_PROVISIONAL,
    TRADE_SHAPE_SOURCE,
    JournalStore,
)

#: The confidence a candidate must clear before it is written as a tag.
#:
#: Chosen from the live histogram rather than picked round: ``AutoTagger``'s
#: score is a sum of a source weight (tracker 0.28, focus favourite 0.24, avwap
#: or bounce 0.18), a date score that is 0.28 ONLY on the trade's own day and at
#: most 0.22 otherwise, a side agreement of +0.16 (or -0.10 when the sides
#: disagree), and small priority/bucket bonuses. 0.70 encodes a sentence rather
#: than a percentile: "the setup tracker or a focus favourite named this symbol,
#: on the day I traded it, on the side I traded". Tracker + same day + side is
#: 0.72 and a focus favourite is 0.68 before its 0.08 bucket bonus, so both
#: clear it; the SAME tracker row one day later reaches 0.66, and a weaker
#: source (avwap signal, intraday bounce) on the right day and side reaches 0.62
#: and needs a real priority score to get over. Everything below the line still
#: gets looked at - it gets a `needs_review` marker instead of a guess.
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

#: Which trades are eligible. An OPEN trade has no shape yet and its story is
#: not finished; tagging one is a claim about a position the trader is still in.
ELIGIBLE_STATUS = "CLOSED"

#: Half-finished, and there are SIX of them on the live journal (R1). A
#: CLOSED_PARTIAL trade is still open in part, so the OPEN rule applies - but it
#: was falling through the eligibility check into nothing at all: not tagged,
#: not marked, invisible in every count on the way past. It is now marked
#: `needs_review` and never tagged, so the trader can see the six rather than
#: wonder why 162 closed-ish trades produced 156 decisions.
PARTIAL_STATUS = "CLOSED_PARTIAL"

#: The bucket width of the printed histogram. Fine enough to see where the
#: threshold falls, coarse enough to read in a terminal.
HISTOGRAM_BUCKET = 0.05


@dataclass
class TagDecision:
    """What this run would do to one trade, and why."""

    trade_id: str
    symbol: str
    trade_date: str
    action: str  # "apply" | "needs_review" | "skip"
    tag: str = ""
    confidence: float | None = None
    source: str = ""
    rationale: str = ""
    reason: str = ""

    def as_payload(self) -> dict[str, Any]:
        """The adjustment payload: the candidate, verbatim, and this packet.

        The stored rationale is the tagger's own sentence (which file, which
        symbol, which context date), so six months from now the record answers
        "why does this trade say avwap-reclaim?" without re-deriving anything.
        """
        return {
            "packet": "P6a",
            "trade_id": self.trade_id,
            "tag": self.tag,
            "confidence": self.confidence,
            "candidate_source": self.source,
            "rationale": self.rationale,
            "tag_status": TAG_STATUS_PROVISIONAL,
        }


@dataclass
class BulkTagPlan:
    """Everything the dry run knows, which is everything ``--apply`` uses."""

    threshold: float
    decisions: list[TagDecision] = field(default_factory=list)
    histogram: Counter = field(default_factory=Counter)
    considered: int = 0
    already_confirmed: int = 0
    no_candidate: int = 0
    #: Half-closed trades: marked for review, never tagged (R1).
    partial: int = 0

    @property
    def to_apply(self) -> list[TagDecision]:
        return [item for item in self.decisions if item.action == "apply"]

    @property
    def to_review(self) -> list[TagDecision]:
        return [item for item in self.decisions if item.action == "needs_review"]


def _setup_lane(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the candidates that NAME A SETUP. Two lanes are excluded, not one.

    ``journal_trade_shape``'s tags (``midday``, ``swing``, ``scalp``) are
    derived from the trade's own clock and legs, carry a confidence of 1.0, and
    already reach the trade through ``auto_tag_summary``. Promoting one into
    ``setup_tags`` would answer "which of my setups was this?" with a fact about
    the clock - and would do it at a confidence no scanner match can beat.

    A LINK candidate is excluded for the same reason and a worse one (R2). It
    records that the trader added the chart to Focus or armed a level; it names
    no setup at all, and it arrives at 0.90-0.95 because the capture lane is the
    most confident lane there is. `build_plan` takes `max(confidence)`, so a
    link beat every scanner match beneath it - reproduced on a copy of the live
    journal: TRV lost `avwap_retest_followthrough` at 0.91 to
    `link:review:arm_level`, and VFC and UMAC the same way.
    """
    lane = []
    for candidate in candidates:
        source = str(candidate.get("source") or "")
        if source.startswith(f"{TRADE_SHAPE_SOURCE}:"):
            continue
        if is_link_candidate(candidate):
            continue
        lane.append(candidate)
    return lane


def _bucket(confidence: float) -> float:
    return round(round(float(confidence) / HISTOGRAM_BUCKET) * HISTOGRAM_BUCKET, 2)


def build_plan(
    store: JournalStore,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    refresh: bool = True,
) -> BulkTagPlan:
    """Decide what would be written, without writing anything.

    ``refresh`` re-derives the candidates first, and defaults to True for a
    reason the live data made obvious: most of this journal was imported from
    broker statements LONG after the scan files that could have explained it,
    so a trade's stored suggestions may predate the evidence that now exists.
    Deciding from stale candidates would tag the backlog from whatever the
    tagger believed on some earlier day.
    """
    if refresh:
        store.refresh_auto_tags()

    plan = BulkTagPlan(threshold=float(threshold))
    for trade in store.list_trades():
        status_text = str(trade.get("status") or "").upper()
        if status_text == PARTIAL_STATUS:
            plan.partial += 1
            if str(trade.get("tag_status") or TAG_STATUS_CONFIRMED) != TAG_STATUS_NEEDS_REVIEW:
                plan.decisions.append(
                    TagDecision(
                        trade_id=str(trade.get("trade_id") or ""),
                        symbol=str(trade.get("symbol") or ""),
                        trade_date=str(trade.get("trade_date") or "")[:10],
                        action="needs_review",
                        reason="partially closed - the trade is not finished, so no tag",
                    )
                )
            continue
        if status_text != ELIGIBLE_STATUS:
            continue
        plan.considered += 1
        trade_id = str(trade.get("trade_id") or "")
        status = str(trade.get("tag_status") or TAG_STATUS_CONFIRMED)
        has_tags = bool(str(trade.get("setup_tags") or "").strip())
        if status == TAG_STATUS_CONFIRMED and has_tags:
            # The trader's own answer. Nothing here improves on it.
            plan.already_confirmed += 1
            continue

        candidates = _setup_lane(store.list_auto_tag_candidates(trade_id))
        symbol = str(trade.get("symbol") or "")
        trade_date = str(trade.get("trade_date") or "")[:10]
        if not candidates:
            plan.no_candidate += 1
            if status != TAG_STATUS_NEEDS_REVIEW:
                plan.decisions.append(
                    TagDecision(
                        trade_id=trade_id,
                        symbol=symbol,
                        trade_date=trade_date,
                        action="needs_review",
                        reason="no scanner match for this symbol in the lookback",
                    )
                )
            continue

        top = max(candidates, key=lambda item: float(item.get("confidence") or 0.0))
        confidence = float(top.get("confidence") or 0.0)
        plan.histogram[_bucket(confidence)] += 1
        if confidence < plan.threshold:
            if status != TAG_STATUS_NEEDS_REVIEW:
                plan.decisions.append(
                    TagDecision(
                        trade_id=trade_id,
                        symbol=symbol,
                        trade_date=trade_date,
                        action="needs_review",
                        tag="",
                        confidence=confidence,
                        source=str(top.get("source") or ""),
                        reason=(
                            f"best candidate {top.get('tag')!r} at {confidence:.2f} "
                            f"is below {plan.threshold:.2f}"
                        ),
                    )
                )
            continue

        if status == TAG_STATUS_PROVISIONAL and str(trade.get("setup_tags") or "").strip() == str(
            top.get("tag") or ""
        ).strip():
            # Already applied, and the evidence has not changed its mind. This
            # is what makes a second run a no-op.
            continue

        plan.decisions.append(
            TagDecision(
                trade_id=trade_id,
                symbol=symbol,
                trade_date=trade_date,
                action="apply",
                tag=str(top.get("tag") or ""),
                confidence=confidence,
                source=str(top.get("source") or ""),
                rationale=str(top.get("rationale") or ""),
            )
        )
    return plan


def apply_plan(store: JournalStore, plan: BulkTagPlan) -> dict[str, int]:
    """Write the plan. Every application leaves an adjustment record behind it.

    The order is deliberate: the tag is written FIRST and the audit record
    second. If the record's append fails, a tag exists that the trail does not
    explain - visible, marked provisional, and correctable. The other order
    would leave a record claiming a tag that is not there, which reads as
    history rather than as a fault.
    """
    summary = {"applied": 0, "marked": 0, "refused": 0}
    for decision in plan.decisions:
        if decision.action == "apply":
            if not store.apply_provisional_tags(decision.trade_id, decision.tag):
                # The refusal is the store's, and it means a confirmed tag
                # appeared between the plan and the write.
                summary["refused"] += 1
                continue
            summary["applied"] += 1
            store.record_adjustment(
                action=PROVISIONAL_TAG_ADJUSTMENT,
                target_kind="TRADE",
                target_uid=decision.trade_id,
                reason=(
                    f"P6a bulk tag: {decision.tag!r} at {decision.confidence:.2f} "
                    f"from {decision.source or 'scanner output'}, provisional until reviewed"
                ),
                payload=decision.as_payload(),
                source="journal_bulk_tag",
            )
        elif decision.action == "needs_review":
            if store.mark_tags_needing_review(decision.trade_id):
                summary["marked"] += 1
    return summary


def format_histogram(plan: BulkTagPlan) -> list[str]:
    """The confidence distribution the threshold was chosen against."""
    if not plan.histogram:
        return ["  (no scanner candidates at all)"]
    lines = []
    drawn = False
    for bucket in sorted(plan.histogram):
        if not drawn and bucket >= plan.threshold:
            # A line THROUGH the distribution rather than a mark on one bucket:
            # the threshold may not land on a bucket that exists, and a cut
            # nobody can see in the histogram is not a justified cut.
            lines.append(f"  ----- threshold {plan.threshold:.2f} -----")
            drawn = True
        count = plan.histogram[bucket]
        lines.append(f"  {bucket:.2f}  {count:4d}  {'#' * min(count, 60)}")
    if not drawn:
        lines.append(f"  ----- threshold {plan.threshold:.2f} ----- (nothing reached it)")
    return lines


def format_plan(plan: BulkTagPlan, *, applied: dict[str, int] | None = None) -> str:
    lines = [
        "Journal bulk tagging (P6a)",
        "",
        f"Closed trades considered:        {plan.considered}",
        f"Already tagged by the trader:    {plan.already_confirmed}",
        f"No scanner candidate at all:     {plan.no_candidate}",
        f"Partly closed (marked, not tagged): {plan.partial}",
        f"Confidence threshold:            {plan.threshold:.2f}",
        "",
        "Top-candidate confidence, closed and untagged:",
        *format_histogram(plan),
        "",
        f"Would apply a provisional tag:   {len(plan.to_apply)}",
        f"Would mark needs_review:         {len(plan.to_review)}",
    ]
    if applied is not None:
        lines += [
            "",
            f"APPLIED:      {applied.get('applied', 0)}",
            f"MARKED:       {applied.get('marked', 0)}",
            f"REFUSED:      {applied.get('refused', 0)} (a confirmed tag appeared first)",
        ]
    else:
        lines += [
            "",
            "Dry run: no trader-owned row was written. The candidates WERE re-derived,",
            "because `auto_tag_candidates` is the machine's own table and the desk",
            "refreshes it on demand anyway - and deciding a threshold from stale",
            "suggestions would measure a different question. Re-run with --apply to write.",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the plan (default is a dry run that writes nothing)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"confidence a candidate must clear (default {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--db",
        default="",
        help=(
            "journal database to work on (default: the live one). A dry run "
            "against a copy is the safe way to try a different --threshold."
        ),
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="decide from the stored candidates instead of re-deriving them first",
    )
    args = parser.parse_args(argv)

    store = JournalStore(Path(args.db)) if args.db else JournalStore()
    store.initialize_schema()
    plan = build_plan(store, threshold=args.threshold, refresh=not args.no_refresh)
    applied = apply_plan(store, plan) if args.apply else None
    print(format_plan(plan, applied=applied))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main(sys.argv[1:]))
