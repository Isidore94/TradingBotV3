"""Generate the frozen setup registry from the four places that name a setup.

Phase 0.13 packet P7. Run once, review the diff, commit the JSON. The registry
is DATA at runtime (`setup_registry.py` loads it and never regenerates it),
because a crosswalk that recomputes itself from four moving sources is not a
crosswalk - it is a fourth source that changes silently when any of the others
does. Freezing it is what turns a divergence into a reviewable diff.

The four sources, and what each contributes:

- ``master_avwap_lib.setup_tagging._FAMILY_TAGS`` - the CANONICAL id. Warehouse
  upper-snake; this is the name `setup_occurrence` stores.
- ``setup_docs.SETUP_DOCS`` - the docs key, the display label, and (through its
  group) the status: "Study (measured only)" and "Playbook research" are not
  production families.
- ``setup_playbook_study.PLAYBOOK`` - the study families, which have no scanner
  tag at all and would be invisible to a registry built from tags alone.
- ``ui.annotations.setup_claims`` - what the trader can actually claim. A claim
  that resolves to nothing grades under a name no other table uses, which is the
  exact failure this registry exists to prevent.

Roles use Appendix C's vocabulary (`TRADE_SETUP`, `CONTEXT`, `WATCH_STATE`,
`CONTROL`, `FALLBACK`) and are assigned from Appendix C's own table rather than
inferred. Everything Appendix C requires that the four sources do NOT establish -
supported sides, timeframe roles, the exact completed-bar trigger, the primary
recipe - is left empty, and every row names the packet that fills it. An
invented side is worse than a blank one: the blank says "not established", and a
guess says "established" in exactly the column a later experiment would trust.

Usage::

    python scripts/build_setup_registry.py            # print what would change
    python scripts/build_setup_registry.py --write    # rewrite the frozen JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(ROOT))

REGISTRY_PATH = ROOT / "setup_registry_v1.json"
REGISTRY_VERSION = 1
SCHEMA = "setup_registry_v1"

#: Appendix C's role vocabulary, verbatim. `TRADE` - the fact pack's spelling -
#: is deliberately NOT here; see `setup_registry.fact_pack_role`.
ROLE_TRADE_SETUP = "TRADE_SETUP"
ROLE_CONTEXT = "CONTEXT"
ROLE_WATCH_STATE = "WATCH_STATE"
ROLE_CONTROL = "CONTROL"
ROLE_FALLBACK = "FALLBACK"

#: Roles Appendix C names explicitly, keyed by canonical id, with its own words
#: beside each. Anything absent is a TRADE_SETUP - the only default Appendix C's
#: table supports, since every row it lists is a trade setup unless it says
#: otherwise.
ROLE_BY_CANONICAL = {
    # "Favorite Zone Watch | Watch state | Never counted as a triggered trade setup"
    "FAVORITE_ZONE_WATCH": ROLE_WATCH_STATE,
    # "SMA50/100/200 Breakout and Retest | Reclaim/watch and confirmed retest are
    # separate states" - the tracking family IS the watch half.
    "SMA_BREAKOUT_WATCH": ROLE_WATCH_STATE,
    # "TOP Weekly Leader | Context/basket plus linked daily trigger | Weekly
    # pattern alone is not the entry" - the tracking family is that basket state.
    "TOP_PATTERN_WATCH": ROLE_WATCH_STATE,
    # "General/Untagged | Diagnostic fallback | Must not become a pooled 'setup' edge"
    "GENERAL": ROLE_FALLBACK,
    # The trader's spelling of untagged. Same rule, same reason.
    "NONE_OF_THESE": ROLE_FALLBACK,
    # "`baseline_every5` | CONTROL | Anchors every playbook comparison; never tradable"
    "BASELINE_EVERY5": ROLE_CONTROL,
}

#: Status, from the source that establishes it rather than from an opinion:
#: `setup_docs`' own group text carries it for everything it documents.
STATUS_BY_DOC_GROUP = {
    "Main swing": "production",
    "Earnings cycle": "production",
    "Study (measured only)": "study",
    "Playbook research": "research",
}

#: A playbook family with no documentation entry is a study family by
#: construction: `setup_playbook_study` is the study harness and no production
#: scanner path reaches it.
STATUS_PLAYBOOK_ONLY = "study"

#: A scanner family with no documentation entry is still production - it is what
#: the live scanner tags - and the missing doc is itself a divergence.
STATUS_SCANNER_ONLY = "production"

#: Exclusivity groups Appendix C states outright. Absent = its own group, which
#: is the safe default: wrongly SHARING a group would let two independent setups
#: suppress each other's samples.
EXCLUSIVITY_BY_CANONICAL = {
    # "Post-Earnings Candle Break | Mutually exclusive with the 52-week variant
    # for one trigger"; "Post-Earnings 52-week Break | Separate extreme-break
    # thesis and exclusivity group".
    "POST_EARNINGS_CANDLE_BREAK": "post_earnings_break",
    "POST_EARNINGS_52W_BREAK": "post_earnings_break",
    # "Mid-Earnings EMA21 Retest | Correlated with EMA15; explicit family wins if
    # both fire"; "Mid-Earnings 1st-Dev Retest | same episode dependence cluster".
    "MID_EARNINGS_EMA15_RETEST": "mid_earnings_retest",
    "MID_EARNINGS_EMA21_RETEST": "mid_earnings_retest",
    "MID_EARNINGS_FIRST_DEV_RETEST": "mid_earnings_retest",
    # "AVWAP Retest Followthrough | fold/compare with parent without
    # double-counting one move" - the parent is the favorite thesis.
    "AVWAPE_TO_FIRST_DEV": "avwap_favorite_thesis",
    "AVWAP_RETEST": "avwap_favorite_thesis",
}

#: Appendix C's vertical-slice mapping, which the alias table does not carry:
#: "`AVWAPE_TO_FIRST_DEV` <-> 'AVWAPE to 1st Dev Favorite'". Without it the
#: scanner's family and the doc describing that exact family land as two
#: entries, which is precisely the split this registry exists to close. The
#: MISSING alias row is still reported as a divergence - the join is Appendix
#: C's, not this generator's, and the alias table should eventually carry it.
DOC_KEY_BY_CANONICAL = {
    "AVWAPE_TO_FIRST_DEV": "avwape_to_1stdev",
}

#: The fifth naming site. `legacy.py` declares study families as
#: `*_STUDY_FAMILY` constants, and eight of them are named NOWHERE else - so a
#: registry built from the four sources P7 names would be missing detectors that
#: actually run. Read by REGEX rather than by importing a 27k-line module that
#: pulls the whole scanner in; the file is never written here.
LEGACY_STUDY_FAMILY_RE = r"^([A-Z0-9_]+_STUDY_FAMILY)\s*=\s*\"([a-z0-9_]+)\""
LEGACY_PATH = ROOT / "master_avwap_lib" / "legacy.py"

#: Playbook family -> the documentation key describing it, where the two spell
#: one setup differently. Read off `setup_docs` rather than guessed: each of
#: these docs keys exists and describes that family.
PLAYBOOK_DOC_KEY = {
    "volume_thrust": "playbook_volume_thrust",
    "quiet_pullback_resume": "playbook_quiet_pullback_resume",
    "post_earnings_volume_break": "playbook_post_earnings_volume_break",
    "second_dev_power_hold": "playbook_second_dev_power_hold",
    "golden_pullback_sma50_vol": "playbook_golden_pullback_vol",
    "first_dev_breakout": "first_dev_breakout",
    "second_dev_breakout": "second_dev_breakout",
}

#: What P7 does not establish, named on every row so an empty column is never
#: read as a measured "no".
UNESTABLISHED_FIELDS = (
    "supported_sides",
    "structural_timeframe",
    "context_timeframe",
    "trigger_timeframe",
    "completed_bar_trigger",
    "primary_recipe",
)
AUTHORITATIVE_WHEN = "plan.md P4.1 (identity-graph freeze)"

#: Notes Appendix C attaches to a specific family and that a later packet must
#: not have to rediscover.
APPENDIX_C_NOTES = {
    "POST_EARNINGS_AVWAP_BOUNCE": (
        "Side asymmetry: long confirm-only/weak evidence and short hypothesis are "
        "preserved SEPARATELY (Appendix C). The AVWAPE spelling and the "
        "'Pre-Earnings AVWAPE Reject' label are aliases of this one family."
    ),
    "MID_EARNINGS_SECOND_DEV_HOLD": (
        "Two linked ids: the context episode and the long-only research thesis. "
        "`mid_earnings_above_2nd_stdev` is an alias, not another sample."
    ),
    "POST_EARNINGS_GAP_HOLD3": (
        "NO 52-week condition in its identity - the detector has none, and "
        "re-adding one would recreate the spurious gating the trader flagged. "
        "Weekly-strong is a measured evidence segment, never part of identity."
    ),
    "BASELINE_EVERY5": "Anchors every playbook comparison; never tradable.",
    "GENERAL": "Must not become a pooled 'setup' edge.",
}


def _sources():
    from master_avwap_lib.setup_tagging import _FAMILY_TAGS, _TAG_ALIASES
    from setup_docs import SETUP_DOC_ALIASES, SETUP_DOCS
    from setup_playbook_study import PLAYBOOK
    from ui.annotations.setup_claims import all_setup_claims

    return (
        dict(_FAMILY_TAGS),
        dict(_TAG_ALIASES),
        dict(SETUP_DOCS),
        dict(SETUP_DOC_ALIASES),
        dict(PLAYBOOK),
        [claim.setup_id for claim in all_setup_claims()],
    )


def legacy_study_families() -> dict[str, str]:
    """`{family key: constant name}` for every `*_STUDY_FAMILY` in `legacy.py`."""
    import re

    found: dict[str, str] = {}
    if not LEGACY_PATH.is_file():  # pragma: no cover - the file is in the tree
        return found
    pattern = re.compile(LEGACY_STUDY_FAMILY_RE, re.MULTILINE)
    for constant, family in pattern.findall(LEGACY_PATH.read_text(encoding="utf-8")):
        found.setdefault(family, constant)
    return found


def canonical_from_key(key: str) -> str:
    """A docs/playbook key as a warehouse id: upper snake, no `playbook_` prefix.

    The prefix is a documentation namespace and not part of identity -
    `playbook_volume_thrust` and the playbook's own `volume_thrust` are one
    family, and two canonical ids would split every sample it ever produces.
    """
    text = str(key or "").strip().lower()
    if text.startswith("playbook_"):
        text = text[len("playbook_") :]
    return text.upper()


def build() -> dict[str, Any]:
    """The whole registry, assembled from the four sources in source order."""
    families, tag_aliases, docs, doc_aliases, playbook, claim_ids = _sources()

    entries: dict[str, dict[str, Any]] = {}
    divergences: list[dict[str, str]] = []

    def entry_for(canonical: str) -> dict[str, Any]:
        setup_id = canonical.lower()
        key = f"{setup_id}@{REGISTRY_VERSION}"
        found = entries.get(key)
        if found is None:
            found = {
                "setup_id": setup_id,
                "version": REGISTRY_VERSION,
                "canonical_setup_id": canonical,
                "docs_key": "",
                "playbook_family": "",
                "claim_id": "",
                "family_tag_key": "",
                "label": "",
                "role": ROLE_BY_CANONICAL.get(canonical, ROLE_TRADE_SETUP),
                "status": "",
                "exclusivity_group": EXCLUSIVITY_BY_CANONICAL.get(canonical, setup_id),
                "aliases": [],
                "sources": [],
                "supported_sides": [],
                "structural_timeframe": "",
                "context_timeframe": "",
                "trigger_timeframe": "",
                "completed_bar_trigger": "",
                "primary_recipe": "",
                "unestablished": list(UNESTABLISHED_FIELDS),
                "authoritative_when": AUTHORITATIVE_WHEN,
                "legacy_study_constant": "",
                "note": APPENDIX_C_NOTES.get(canonical, ""),
            }
            entries[key] = found
        return found

    def add_alias(entry: dict[str, Any], alias: str) -> None:
        text = str(alias or "").strip()
        if text and text != entry["setup_id"] and text not in entry["aliases"]:
            entry["aliases"].append(text)

    def add_source(entry: dict[str, Any], source: str) -> None:
        if source not in entry["sources"]:
            entry["sources"].append(source)

    # 1. The scanner families carry the canonical id, so they are first and the
    #    others attach to them.
    for family_key, canonical in families.items():
        entry = entry_for(canonical)
        entry["family_tag_key"] = family_key
        entry["status"] = STATUS_SCANNER_ONLY
        add_alias(entry, family_key)
        add_source(entry, "setup_tagging._FAMILY_TAGS")

    for alias_tag, canonical in tag_aliases.items():
        add_alias(entry_for(canonical), alias_tag)

    # 2. Documentation. A docs key reaches a family directly, through
    #    `SETUP_DOC_ALIASES`, or not at all - and "not at all" is recorded as a
    #    divergence, never resolved by inventing a second entry.
    canonical_by_family_key = dict(families)
    canonical_by_doc_key = {value: key for key, value in DOC_KEY_BY_CANONICAL.items()}
    for docs_key, doc in docs.items():
        canonical = (
            canonical_by_family_key.get(docs_key)
            or canonical_by_doc_key.get(docs_key)
            or canonical_from_key(docs_key)
        )
        entry = entry_for(canonical)
        entry["docs_key"] = docs_key
        entry["label"] = str(doc.get("label") or "").strip()
        status = STATUS_BY_DOC_GROUP.get(str(doc.get("group") or ""), "")
        if status:
            entry["status"] = status
        add_alias(entry, docs_key)
        add_source(entry, "setup_docs.SETUP_DOCS")

    # The alias table is CHECKED rather than trusted: it maps a spelling to a
    # docs key, and if that key belongs to a different canonical family then the
    # two tables disagree about what one setup is.
    for alias_key, docs_key in doc_aliases.items():
        alias_canonical = canonical_by_family_key.get(alias_key)
        target_canonical = canonical_by_family_key.get(docs_key) or canonical_from_key(docs_key)
        if alias_canonical and alias_canonical != target_canonical:
            divergences.append(
                {
                    "kind": "alias_points_at_another_family",
                    "alias": alias_key,
                    "docs_key": docs_key,
                    "tag_canonical": alias_canonical,
                    "docs_canonical": target_canonical,
                    "note": (
                        f"`setup_tagging` gives {alias_key!r} its own canonical id "
                        f"({alias_canonical}); `setup_docs` documents it under "
                        f"{docs_key!r} ({target_canonical}). One is identity and the "
                        "other is a display convenience, and which is which is not "
                        "P7's call."
                    ),
                    "resolved_by": AUTHORITATIVE_WHEN,
                }
            )
        add_alias(entry_for(alias_canonical or target_canonical), alias_key)

    for canonical, docs_key in DOC_KEY_BY_CANONICAL.items():
        divergences.append(
            {
                "kind": "alias_table_missing_the_pair",
                "canonical": canonical,
                "docs_key": docs_key,
                "note": (
                    f"Appendix C's vertical-slice mapping states {canonical} is the "
                    f"family {docs_key!r} documents, but `SETUP_DOC_ALIASES` carries no "
                    "row joining them - so every reader that goes through the alias "
                    "table alone sees two setups where there is one. This registry "
                    "applies Appendix C's join; the alias table has not been changed."
                ),
                "resolved_by": AUTHORITATIVE_WHEN,
            }
        )

    # 3. The study families. Most have no scanner tag at all.
    for family in playbook:
        docs_key = PLAYBOOK_DOC_KEY.get(family, family if family in docs else "")
        canonical = canonical_from_key(docs_key or family)
        entry = entry_for(canonical)
        entry["playbook_family"] = family
        if docs_key:
            entry["docs_key"] = entry["docs_key"] or docs_key
        entry["status"] = entry["status"] or STATUS_PLAYBOOK_ONLY
        entry["role"] = ROLE_BY_CANONICAL.get(canonical, entry["role"])
        entry["note"] = entry["note"] or APPENDIX_C_NOTES.get(canonical, "")
        add_alias(entry, family)
        add_source(entry, "setup_playbook_study.PLAYBOOK")

    # 3b. THE FIFTH NAMING SITE. `legacy.py`'s `*_STUDY_FAMILY` constants. P7's
    #     packet names four sources; the code has five, and eight of these
    #     families are named nowhere else - a registry without them would be a
    #     crosswalk that quietly omits detectors that run every scan.
    for family, constant in legacy_study_families().items():
        docs_key = family if family in docs else PLAYBOOK_DOC_KEY.get(family, "")
        alias_target = doc_aliases.get(family, "")
        canonical = canonical_from_key(docs_key or alias_target or family)
        entry = entry_for(canonical)
        entry["legacy_study_constant"] = constant
        if docs_key:
            entry["docs_key"] = entry["docs_key"] or docs_key
        entry["status"] = entry["status"] or STATUS_PLAYBOOK_ONLY
        entry["note"] = entry["note"] or APPENDIX_C_NOTES.get(canonical, "")
        add_alias(entry, family)
        add_source(entry, "legacy.py::*_STUDY_FAMILY")

    # 4. What the trader can claim. Every offered claim must land on a row.
    for claim_id in claim_ids:
        canonical = (
            canonical_by_family_key.get(claim_id)
            or canonical_by_doc_key.get(claim_id)
            or canonical_from_key(claim_id)
        )
        entry = entry_for(canonical)
        entry["claim_id"] = claim_id
        entry["role"] = ROLE_BY_CANONICAL.get(canonical, entry["role"])
        entry["status"] = entry["status"] or "not_a_setup"
        entry["note"] = entry["note"] or APPENDIX_C_NOTES.get(canonical, "")
        add_alias(entry, claim_id)
        add_source(entry, "setup_claims.all_setup_claims")

    # A documented family with no scanner tag and no playbook function is a
    # divergence too - something names it that nothing detects.
    for entry in entries.values():
        if entry["role"] == ROLE_FALLBACK:
            continue
        detected = (
            entry["family_tag_key"]
            or entry["playbook_family"]
            or entry.get("legacy_study_constant")
        )
        if not detected:
            divergences.append(
                {
                    "kind": "documented_but_undetected",
                    "canonical": entry["canonical_setup_id"],
                    "docs_key": entry["docs_key"],
                    "note": (
                        "documented and claimable, but no `_FAMILY_TAGS` entry, no "
                        "playbook detector and no `*_STUDY_FAMILY` constant names it - "
                        "so nothing writes an occurrence under this id today"
                    ),
                    "resolved_by": AUTHORITATIVE_WHEN,
                }
            )
        if entry["family_tag_key"] and not entry["docs_key"]:
            divergences.append(
                {
                    "kind": "detected_but_undocumented",
                    "canonical": entry["canonical_setup_id"],
                    "family_tag_key": entry["family_tag_key"],
                    "note": (
                        "the live scanner tags this family, but `setup_docs` has no "
                        "entry for it - so it can be detected and never claimed"
                    ),
                    "resolved_by": AUTHORITATIVE_WHEN,
                }
            )

    ordered = {key: entries[key] for key in sorted(entries)}
    divergences.sort(key=lambda item: (item["kind"], item.get("canonical", ""), item.get("alias", "")))
    return {
        "schema": SCHEMA,
        "registry_version": REGISTRY_VERSION,
        "generated_by": "scripts/build_setup_registry.py",
        "packet": "Phase 0.13 P7",
        "authoritative_when": AUTHORITATIVE_WHEN,
        "read_only": True,
        "sources": [
            "scripts/master_avwap_lib/setup_tagging.py::_FAMILY_TAGS/_TAG_ALIASES",
            "scripts/setup_docs.py::SETUP_DOCS/SETUP_DOC_ALIASES",
            "scripts/setup_playbook_study.py::PLAYBOOK",
            "scripts/ui/annotations/setup_claims.py::all_setup_claims",
        ],
        "entries": ordered,
        "known_divergences": divergences,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the frozen JSON")
    args = parser.parse_args(argv)

    payload = build()
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    existing = REGISTRY_PATH.read_text(encoding="utf-8") if REGISTRY_PATH.is_file() else ""

    print(f"entries: {len(payload['entries'])}")
    print(f"known divergences: {len(payload['known_divergences'])}")
    for item in payload["known_divergences"]:
        print(f"  [{item['kind']}] {item.get('canonical') or item.get('alias')}")
    if args.write:
        REGISTRY_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {REGISTRY_PATH}")
    elif text != existing:
        print("FROZEN FILE IS STALE - re-run with --write and review the diff")
        return 1
    else:
        print("frozen file matches")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
