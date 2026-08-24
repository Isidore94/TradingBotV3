"""The setup-claim picklist, read from the existing setup registry.

A claimed setup is the trader's answer to "what do you think this is?" - the
label whose forward record the program is ultimately trying to learn. It has
to name the same families the rest of the system names, so the list is derived
from :mod:`setup_docs` rather than restated here. Study families are included
deliberately: a claim on a measured-only setup is exactly the evidence that
decides whether it ever graduates (plan.md sec 7).

This module reads ``setup_docs``; it never writes to it and never influences
which setups the scanners detect or how they score.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Claims the trader can make that are not detector families. "none_of_these"
#: is the honest answer when the chart looks good for a reason the registry
#: has no name for yet - and a run of them is itself a finding.
EXTRA_CLAIMS = (
    ("none_of_these", "None of these", "Focus-worthy, but not a family the registry names."),
)
EXTRA_CLAIM_GROUP = "Unregistered"


@dataclass(frozen=True)
class SetupClaim:
    """One selectable claim."""

    setup_id: str
    label: str
    group: str
    summary: str


def _display_label(key: str, doc: dict) -> str:
    label = str(doc.get("label") or doc.get("title") or "").strip()
    return label or key.replace("_", " ")


def setup_claim_groups() -> list[tuple[str, list[SetupClaim]]]:
    """Claims grouped for display, in the registry's own reading order."""
    from setup_docs import all_setup_docs_by_group

    groups: list[tuple[str, list[SetupClaim]]] = []
    for group_name, docs in all_setup_docs_by_group():
        claims = [
            SetupClaim(
                setup_id=key,
                label=_display_label(key, doc),
                group=group_name,
                summary=str(doc.get("summary") or doc.get("what") or "").strip(),
            )
            for key, doc in docs
        ]
        if claims:
            groups.append((group_name, claims))
    groups.append(
        (
            EXTRA_CLAIM_GROUP,
            [
                SetupClaim(setup_id=key, label=label, group=EXTRA_CLAIM_GROUP, summary=summary)
                for key, label, summary in EXTRA_CLAIMS
            ],
        )
    )
    return groups


def all_setup_claims() -> list[SetupClaim]:
    """Every selectable claim, flattened, display order preserved."""
    return [claim for _group, claims in setup_claim_groups() for claim in claims]


# --- what the capture rail actually offers ---------------------------------
#
# These three lived in ``ui.widgets.capture_rail`` until 2026-08-24, which put
# them behind a PySide6 import. That was fine while the rail was the only
# reader; it stopped being fine when ``ai_summary`` had to state the offered
# list as a machine-written caveat, because the summary runs headless in the
# overnight slate and must never drag Qt in. So the DEFINITION lives here, with
# the registry it is derived from, and the rail imports it - the rail still
# owns the rendering, it no longer owns the fact.
#
# The rail re-exports all three, so ``from ui.widgets.capture_rail import
# MAIN_CLAIM_GROUP`` keeps working exactly as before.

#: The claim group the rail offers whole (trader, 2026-08-20: "only do the
#: main setups for now").
MAIN_CLAIM_GROUP = "Main swing"

#: Named claims from OTHER groups, in the order the trader asked for them
#: (2026-08-21: "add my post earnings setups and 2nd stdev breakout"). Ids, not
#: a group name, because that ask was specific: the three post-earnings
#: families and the 2nd-dev breakout, not the mid-earnings retests beside them
#: and not the rest of the study shelf. Adding one later is a line here.
EXTRA_CLAIM_IDS = (
    "post_earnings_52w_break",
    "post_earnings_candle_break",
    "post_earnings_avwap_bounce",
    "second_dev_breakout",
)


def offered_setup_claims() -> list[SetupClaim]:
    """The claims the capture rail offers, in display order.

    Main swing whole and in the registry's own order, then the named extras in
    the order they are listed. Reads the registry rather than restating it, so
    a label or summary edited in ``setup_docs`` shows up unchanged.

    An extra id the registry does not know is skipped rather than guessed at -
    and ``test_the_rail_offers_every_claim_the_trader_asked_for`` fails loudly
    if that ever happens, so a typo cannot quietly cost the trader a claim.
    """
    grouped = setup_claim_groups()
    offered: list[SetupClaim] = []
    for group_name, claims in grouped:
        if group_name == MAIN_CLAIM_GROUP:
            offered.extend(claims)
    by_id = {claim.setup_id: claim for _group, claims in grouped for claim in claims}
    for setup_id in EXTRA_CLAIM_IDS:
        claim = by_id.get(setup_id)
        if claim is not None and claim not in offered:
            offered.append(claim)
    return offered


def valid_setup_claim_ids() -> frozenset[str]:
    """Ids a ``claimed_setup_id`` may legally carry."""
    return frozenset(claim.setup_id for claim in all_setup_claims())


def is_valid_setup_claim(setup_id: str) -> bool:
    return str(setup_id or "").strip().lower() in valid_setup_claim_ids()
