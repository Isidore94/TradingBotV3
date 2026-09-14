"""The trader's claimed D1 picks, as rows in the Master AVWAP setups table.

Packet D1C-A (trader, 2026-09-14): *"A successful D1 'Like and claim' must add
the pick to Master AVWAP Setups immediately, even when it is absent from the
latest scan report. ... A setup belonging to several buckets appears once with
all its labels. Preserve distinct setups and directions for the same symbol."*

PURE. This module reads a claim store's rows and a list of `SetupRow` and
returns a new list. It writes nothing, reads no file, touches no Qt object and
computes no statistic: the point total a claimed row shows is
`setup_points.score_row`'s, from the measurements the claim carried, and this
module's whole job is to put those measurements where that function looks.

Two outcomes per active claim, and only two:

* **The scan already carries this opportunity** - same symbol, same side, same
  setup family. Then the claim is a LABEL: the row gains the bucket key
  `claimed_like` and the badge "My liked trade", and keeps everything it had.
  It appears ONCE with all its labels; its score, its bucket and its rank are
  the scan's, because the scan measured it and the like did not.
* **The scan does not carry it.** Then the claim is a NEW row, built from what
  the desk knew when the claim was made (`known_at_claim`, possibly `{}`) and
  honest about the rest: `score` is None because the scan never scored this
  name, `expected_r` is None unless the claim carried one, and the Points cell
  states what was not measured rather than filling the holes with zeros.

**A like grants nothing.** It never sets `favorite_setup` or `high_conviction`
on any row, never changes a score, and never reaches a detector, an alert, a
watchlist or Focus - a claim places exactly one row in one table (D1C0
decision 1).

The `none_of_these` case is deliberate and documented: it names no family, so
:func:`claimed_family` answers `""` and the claim matches NO scan row. It
becomes its own row instead of silently labelling a row it was never about.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ui.models.setup import SetupRow

#: The claim key the setups table reads as a bucket.
CLAIMED_BUCKET = "claimed_like"
CLAIMED_BADGE = "My liked trade"

#: Claims that name no detector family. `none_of_these` is an honest answer -
#: "focus-worthy, but not a family the registry names" - and it must never be
#: resolved to a family, because every family it could be resolved to would be
#: one the trader explicitly declined to pick.
NON_FAMILY_CLAIMS = frozenset({"none_of_these"})

#: The measurements copied from `known_at_claim` onto the row's `raw`, so
#: `setup_points.score_row` reads them exactly as it reads a scan row's.
#: `priority_score` is deliberately NOT promoted to `SetupRow.score`: the scan
#: did not score this name, and a claim-time number printed in the Score column
#: would be a measurement the desk never made.
_RAW_MEASUREMENT_FIELDS = (
    "setup_family",
    "expected_r",
    "priority_score",
    "priority_bucket",
    "key_level",
    "last_price",
    "d1_vs_sector",
    "d1_vs_industry",
)


def claimed_family(setup_id: object) -> str:
    """The detector family a claimed setup id names, or `""` when it names none.

    Not `setup_docs.resolve_setup_doc` alone: that falls back to `general` for
    anything it has never heard of, and a fallback here would hand an unknown
    id a family - and therefore a scan row to attach itself to - that it never
    earned. An id the registry cannot name matches nothing.
    """
    key = str(setup_id or "").strip().lower()
    if not key or key in NON_FAMILY_CLAIMS:
        return ""
    try:
        from setup_docs import resolve_setup_doc

        resolved, _doc = resolve_setup_doc(key)
    except Exception:  # noqa: BLE001 - an unreadable registry matches nothing
        return ""
    if resolved == "general" and key != "general":
        return ""
    return str(resolved or "").strip().lower()


def claimed_label(setup_id: object) -> str:
    """The trader-facing name of a claimed setup, from the registry itself."""
    key = str(setup_id or "").strip().lower()
    if not key:
        return ""
    try:
        from ui.annotations.setup_claims import all_setup_claims

        for claim in all_setup_claims():
            if str(claim.setup_id).strip().lower() == key:
                return str(claim.label or "").strip() or key.replace("_", " ")
    except Exception:  # noqa: BLE001 - a missing label never costs the row
        pass
    return key.replace("_", " ")


def _row_family(row: SetupRow) -> str:
    raw = row.raw if isinstance(row.raw, dict) else {}
    return claimed_family(raw.get("setup_family")) or str(
        raw.get("setup_family") or ""
    ).strip().lower()


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "" or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _copy_for_label(row: SetupRow) -> SetupRow:
    """A row we are about to label, copied so the caller's list is untouched.

    Shallow on `raw` and explicit about the two lists this module appends to -
    a deep copy of every scan row's payload on every refresh would be real work
    on the Qt thread for no gain (nothing expensive on the Qt thread).
    """
    import dataclasses

    raw = dict(row.raw or {})
    badges = raw.get("classification_badges")
    raw["classification_badges"] = list(badges) if isinstance(badges, list) else []
    keys = raw.get("bucket_keys")
    raw["bucket_keys"] = list(keys) if isinstance(keys, (list, tuple)) else []
    return dataclasses.replace(row, raw=raw, setup_tags=list(row.setup_tags or []))


def _label_as_claimed(row: SetupRow, claim: Mapping[str, Any]) -> SetupRow:
    labelled = _copy_for_label(row)
    # The claim that labelled it, carried on the row: "Drop my claim" needs the
    # setup id to end the right pick, and a labelled scan row is a claimed pick
    # just as much as a claimed-only row is. The scan's own measurements are
    # untouched - this adds provenance, never a number.
    labelled.raw["claimed_setup_id"] = str(claim.get("claimed_setup_id") or "").strip()
    labelled.raw["claim_at"] = str(claim.get("claim_at") or "")
    keys = labelled.raw["bucket_keys"]
    primary = str(labelled.bucket or "").strip().lower()
    if primary and primary not in keys:
        keys.append(primary)
    if CLAIMED_BUCKET not in keys:
        keys.append(CLAIMED_BUCKET)
    badges = labelled.raw["classification_badges"]
    if not badges:
        # The scan feed stamps this on every row it builds; a row that arrived
        # without one still has to show its own label beside the new one.
        label = labelled.bucket_label
        if label:
            badges.append(label)
    if CLAIMED_BADGE not in badges:
        badges.append(CLAIMED_BADGE)
    return labelled


def row_from_claim(claim: Mapping[str, Any]) -> SetupRow | None:
    """One claimed-only row: what the trader claimed, and what was known then."""
    symbol = str(claim.get("symbol") or "").strip().upper()
    side = str(claim.get("side") or "").strip().upper()
    if not symbol:
        return None
    setup_id = str(claim.get("claimed_setup_id") or "").strip()
    known = claim.get("known_at_claim")
    known = dict(known) if isinstance(known, Mapping) else {}
    raw: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "claimed_setup_id": setup_id,
        "claim_at": str(claim.get("claim_at") or ""),
        "session_date": str(claim.get("session_date") or ""),
        "source": str(claim.get("source") or ""),
        "note": str(claim.get("note") or ""),
        # The nested dict is the honest record of what the desk had; the
        # flattened copies below are the same numbers where the point system
        # looks for them. Both, deliberately - a reader asking "what did we
        # know?" and a reader asking "what does this score?" are different
        # readers (lead ruling, 2026-09-14).
        "known_at_claim": dict(known),
        "bucket_keys": [CLAIMED_BUCKET],
        "classification_badges": [CLAIMED_BADGE],
    }
    for field in _RAW_MEASUREMENT_FIELDS:
        if field in known and known.get(field) not in (None, ""):
            raw[field] = known.get(field)
    # Deliberately NOT `raw.setdefault("setup_family", claimed_family(setup_id))`.
    # The claimed id is what the TRADER said; a `setup_family` is what the scan
    # MEASURED, and the point system reads that field as a measurement - a
    # family called "..._bounce" scores `BOUNCE_NAMED` on its name alone. A
    # claimed name the scan never carried has no measured family, and the row
    # must say so by leaving the field absent rather than by scoring points the
    # desk never earned. `claimed_setup_id` above is where the claim lives.
    label = claimed_label(setup_id)
    return SetupRow(
        symbol=symbol,
        side=side,
        # A like never invents a score. The scan did not measure this name, so
        # the Score cell stays blank and says so by being blank.
        score=None,
        bucket=CLAIMED_BUCKET,
        setup_tags=[label] if label else [],
        key_level=str(known.get("key_level") or ""),
        expected_r=_float(known.get("expected_r")),
        d1_vs_sector=_float(known.get("d1_vs_sector")),
        d1_vs_industry=_float(known.get("d1_vs_industry")),
        source="claim",
        raw=raw,
    )


def _claim_sort_key(claim: Mapping[str, Any]) -> str:
    return str(claim.get("claim_at") or claim.get("session_date") or "")


def merge_claims(
    rows: Iterable[SetupRow], claims: Iterable[Mapping[str, Any]]
) -> list[SetupRow]:
    """The scan's rows with the trader's active claims folded in. Pure.

    Scan rows keep their order and their measurements; a claim that matches one
    labels it in place, and a claim that matches none becomes a new row after
    them, NEWEST CLAIM FIRST - with the Points switch off that is the order the
    trader reads them in, and the switch (item 6) is what re-orders them.
    """
    source = list(rows or [])
    active = [claim for claim in (claims or []) if isinstance(claim, Mapping)]
    if not active:
        return source

    by_opportunity: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(source):
        key = (
            str(row.symbol or "").strip().upper(),
            str(row.side or "").strip().upper(),
            _row_family(row),
        )
        by_opportunity.setdefault(key, index)

    merged = list(source)
    unmatched: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for claim in active:
        symbol = str(claim.get("symbol") or "").strip().upper()
        side = str(claim.get("side") or "").strip().upper()
        setup_id = str(claim.get("claimed_setup_id") or "").strip()
        identity = (symbol, side, setup_id.lower())
        if not symbol or identity in seen:
            continue
        seen.add(identity)
        family = claimed_family(setup_id)
        index = by_opportunity.get((symbol, side, family)) if family else None
        if index is not None:
            merged[index] = _label_as_claimed(merged[index], claim)
            continue
        unmatched.append(claim)

    for claim in sorted(unmatched, key=_claim_sort_key, reverse=True):
        row = row_from_claim(claim)
        if row is not None:
            merged.append(row)
    return merged
