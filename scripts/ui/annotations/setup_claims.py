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


def valid_setup_claim_ids() -> frozenset[str]:
    """Ids a ``claimed_setup_id`` may legally carry."""
    return frozenset(claim.setup_id for claim in all_setup_claims())


def is_valid_setup_claim(setup_id: str) -> bool:
    return str(setup_id or "").strip().lower() in valid_setup_claim_ids()
