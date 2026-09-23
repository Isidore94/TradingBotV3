"""The overnight day story, and the rolling D1 view (TJ-4 items 2 and 3).

`plan.md` §12.4 TJ-4 changes 2 and 3 as amended 2026-09-19; decision 0021
answer 14. The pattern is `market_story_narration.py` beside it: a closed
schema, an allowed-`source_id` grounding rule, an `inputs_hash` skip, and a
failure that leaves the last verified file byte-identical.

**The model narrates verdicts; it never makes them.** Every
`were_you_right[].verdict` must EQUAL the verdict of the TJ-10 read row its
`evidence_id` names. An output that disagrees with a measured row, grades a
claim no read row carries, calls an OBSERVATION a call, or cites an id the pack
does not carry is rejected WHOLE - not trimmed, not partially kept - and the
last verified story stays exactly as it was. Only a `prediction` may be called a
call; an `observation` is quoted as what the trader saw.

Two artifacts, two verdicts, and neither may destroy the other's last good file:

* `<DAY_REVIEW_DIR>/narration/<date>.json` - one day, narrated from that day's
  pack;
* `<DAY_REVIEW_DIR>/d1_view.json` - ONE rolling file, built from the trader's D1
  prediction clicks and D1 notes of the last `evidence_stats.LATELY_SESSIONS`
  exchange sessions. The thesis store is empty and stays so (packet TJ-4 item
  3): the rolling view is what the trader actually said on a D1 card, or it is
  nothing at all. Its own `inputs_hash` means an unchanged D1 week costs no
  second call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

_log = logging.getLogger(__name__)

#: A session folder's name. Anything else beside the packs is not a queue entry.
_SESSION_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# The stored story is deliberately still v1: readers own that shape.  v2 only
# changes the transport so code, rather than a model, owns the measured link.
PROMPT_VERSION = "day_review_narration_v2"
SCHEMA = "day_review_narration_v1"

D1_VIEW_PROMPT_VERSION = "d1_view_narration_v1"
D1_VIEW_SCHEMA = "d1_view_narration_v1"

#: The local tier this slot asks for - the same one the market-story narration
#: uses. A day story is a page of text, not a research pass.
MODEL_TIER = "medium"

#: How long one call may take. The slot reserves ten minutes (gate #158 wants
#: the day story finished before 23:30 Pacific), and the call itself is bounded
#: well inside that.
TIMEOUT_SECONDS = 540

#: How many QUEUED sessions one night may NARRATE on top of its own, oldest
#: first. The budget is spent on work ATTEMPTED - a model call - and never on a
#: name in a list: three markers whose sessions have no pack used to take the
#: whole budget and then sit at the head of the queue for ever, starving every
#: real request behind them (reviewer round 2, 2026-09-20; and today the live
#: home folder holds three session folders and ZERO packs, so it was every
#: request). A packless marker costs nothing, keeps its marker and is named.
REDO_SWEEP_LIMIT = 3

#: How many packless sessions one reason line names before it says `+N more`.
MAX_NAMED_UNBUILT = 5

#: The ABSOLUTE ceilings on what one reply may contain, per list. They are not
#: the working caps: the caps come FROM THE PACK (see `_narration_schema_for`),
#: because a regular session with every Mentor card answered already carries 8
#: read rows and 25 citable ids before a forecast, a trade, a walk-away row or
#: an internals mark is counted (reviewer round 2). These only bound what the
#: page can be asked to RENDER, because
#: `ai_summary.validate_structured_output` enforces no `maxItems` at all and a
#: 5,000-source reply was written whole (372 KB) and drawn line by line on the
#: Qt thread (reviewer round 1).
MAX_GRADED_CLAIMS = 64
MAX_SOURCES = 512
MAX_OPEN_THESES = 32

#: How long one model call may take, in MINUTES - the same number as
#: `TIMEOUT_SECONDS`, in the unit the launch window speaks. The slot's
#: `reserve_minutes` buys the FIRST call; every call after it asks the window
#: again for this much room, because a sweep begun at 07:50 ET with a
#: ten-minute reserve could otherwise still be loading a model at 08:35, in
#: front of the trader's own market prep (reviewer round 2).
SWEEP_CALL_MINUTES = 9.0

#: Bounds are deliberately NOT 2,000 anywhere in either schema: a `maxLength` of
#: exactly 2,000 is the grammar defect gate #144 found, where the constrained
#: decoder silently truncated mid-sentence.
NARRATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "headline",
        "what_happened",
        "what_you_thought",
        "were_you_right",
        "chased_against_news",
        "process",
        "sources",
    ],
    "properties": {
        "headline": {"type": "string", "maxLength": 160},
        "what_happened": {"type": "string", "maxLength": 1200},
        "what_you_thought": {"type": "string", "maxLength": 600},
        "were_you_right": {
            "type": "array",
            "maxItems": MAX_GRADED_CLAIMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "source_id", "verdict", "evidence_id"],
                "properties": {
                    "claim": {"type": "string", "maxLength": 240},
                    "source_id": {"type": "string", "maxLength": 160},
                    # NOT an enum: a measured verdict may be
                    # `unmeasured:<reason>`, and the rule that matters is
                    # equality with the read row, enforced below.
                    "verdict": {"type": "string", "maxLength": 80},
                    "evidence_id": {"type": "string", "maxLength": 160},
                },
            },
        },
        "chased_against_news": {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "evidence_id"],
            "properties": {
                "verdict": {"type": "string", "enum": ["yes", "no", "unknown"]},
                "evidence_id": {"type": "string", "maxLength": 160},
            },
        },
        "process": {"type": "string", "maxLength": 400},
        "sources": {
            "type": "array",
            "maxItems": MAX_SOURCES,
            "items": {"type": "string", "maxLength": 160},
        },
    },
}

D1_VIEW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["belief_now", "open_theses", "sources"],
    "properties": {
        "belief_now": {"type": "string", "maxLength": 600},
        "open_theses": {
            "type": "array",
            "maxItems": MAX_OPEN_THESES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "since", "still_true", "evidence_id"],
                "properties": {
                    "claim": {"type": "string", "maxLength": 240},
                    "since": {"type": "string", "maxLength": 40},
                    "still_true": {"type": "string", "enum": ["yes", "no", "unknown"]},
                    "evidence_id": {"type": "string", "maxLength": 160},
                },
            },
        },
        "sources": {
            "type": "array",
            "maxItems": MAX_SOURCES,
            "items": {"type": "string", "maxLength": 160},
        },
    },
}

DAY_INSTRUCTIONS = (
    "Narrate this ONE session from the pack below and nothing else. You may not "
    "calculate a statistic, grade a call, or turn an unmeasured item into a "
    "fact. `read_explanations` is a closed object: copy every offered read id "
    "as a key and write only its short explanation. Do not supply a verdict or "
    "choose a prediction source; code owns both measured links. Only a "
    "trader_said item whose kind is 'prediction' may be a call; an item whose "
    "kind is 'observation' is quoted as what the trader SAW. "
    "chased_against_news is 'unknown' unless the forecast section states the "
    "condition you are judging. Every id in sources must be copied exactly from "
    "allowed_source_ids."
)

D1_VIEW_INSTRUCTIONS = (
    "These are the trader's own D1 calls and D1 notes over the last twenty "
    "exchange sessions, oldest first. Say what they appear to believe about the "
    "bigger picture NOW, and list the theses still open. You may not measure "
    "anything: still_true is 'unknown' unless one of these items itself says "
    "otherwise. Every id you cite must be copied exactly from "
    "allowed_source_ids."
)


class NarrationRejected(ValueError):
    """The model's answer was not a narration of the evidence it was given."""


def _bounded_schema(schema: Mapping[str, Any], **limits: int) -> dict[str, Any]:
    """A copy of `schema` whose named arrays are capped at what the pack holds.

    The model is TOLD the real number rather than a fixed six. A regular
    session with every Mentor card answered carries eight read rows and
    twenty-five citable ids (two of the scheduled cards store two entries
    each), so a fixed cap of six made the model choose two reads to drop -
    silently - or made the whole answer unwritable on the busiest days
    (reviewer round 2, 2026-09-20). The ceilings above still apply, because a
    number that comes from the evidence is still rendered by a page.
    """
    body = json.loads(json.dumps(dict(schema)))
    for key, limit in limits.items():
        spec = (body.get("properties") or {}).get(key)
        if isinstance(spec, dict):
            spec["maxItems"] = max(0, int(limit))
    return body


def _read_explanation_contract(
    pack: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Return code-owned read/prediction links and named omissions.

    A prediction belongs to its journal entry, rather than merely being another
    prediction in the pack.  That makes a model incapable of linking one
    entry's measured result to another entry's call.  Multiple predictions for
    one read are ambiguous evidence, so a v2 request is refused whole instead
    of guessing which click was meant.
    """
    import day_review_pack

    reads = _by_source_id(pack.get("reads"), what="reads")
    said = _by_source_id(pack.get("trader_said"), what="trader_said")
    predictions: dict[str, list[Mapping[str, Any]]] = {}
    for item in said.values():
        if str(item.get("kind") or "") != day_review_pack.KIND_PREDICTION:
            continue
        entry_id = str(item.get("entry_id") or "")
        source_id = str(item.get("source_id") or "")
        if not entry_id or not source_id:
            raise NarrationRejected("a prediction offered to v2 has no entry or source id")
        predictions.setdefault(entry_id, []).append(item)

    pairs: dict[str, dict[str, str]] = {}
    not_offered: list[dict[str, str]] = []
    for read_id, read in reads.items():
        entry_id = str(read.get("entry_id") or "")
        if not read_id or not entry_id:
            raise NarrationRejected("a read offered to v2 has no entry or source id")
        matches = predictions.get(entry_id, [])
        if len(matches) > 1:
            raise NarrationRejected(
                f"read {read_id!r} has {len(matches)} prediction sources on its entry"
            )
        if not matches:
            not_offered.append({"read_source_id": read_id, "reason": "no_prediction"})
            continue
        if len(pairs) >= MAX_GRADED_CLAIMS:
            not_offered.append({"read_source_id": read_id, "reason": "render_cap"})
            continue
        prediction = matches[0]
        pairs[read_id] = {
            "read_source_id": read_id,
            "prediction_source_id": str(prediction["source_id"]),
            "verdict": str(read.get("verdict") or ""),
        }
    return pairs, not_offered


def _read_explanation_pairs(pack: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return _read_explanation_contract(pack)[0]


def _narration_schema_for(pack: Mapping[str, Any]) -> dict[str, Any]:
    """The v2 day-story schema, closed around this pack's offered read ids."""
    import day_review_pack

    sources = len(day_review_pack.allowed_source_ids(pack))
    pairs = _read_explanation_pairs(pack)
    reads = len([row for row in pack.get("reads") or () if isinstance(row, Mapping)])
    body = _bounded_schema(
        NARRATION_JSON_SCHEMA,
        were_you_right=min(reads, MAX_GRADED_CLAIMS),
        sources=min(sources, MAX_SOURCES),
    )
    properties = body["properties"]
    # The v2 request must have one transport only.  Legacy replies take their
    # own strict schema after they arrive; this provider schema never offers a
    # model the old verdict/source fields it must no longer choose.
    properties.pop("were_you_right", None)
    required = body["required"]
    required.remove("were_you_right")
    required.append("read_explanations")
    properties["read_explanations"] = {
        "type": "object",
        "additionalProperties": False,
        "required": list(pairs),
        "properties": {
            read_id: {"type": "string", "maxLength": 240}
            for read_id in pairs
        },
    }
    return body


def _legacy_narration_schema_for(pack: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen v1 provider-fixture path, with its original strict limits."""
    import day_review_pack

    reads = len([row for row in pack.get("reads") or () if isinstance(row, Mapping)])
    sources = len(day_review_pack.allowed_source_ids(pack))
    return _bounded_schema(
        NARRATION_JSON_SCHEMA,
        were_you_right=min(reads, MAX_GRADED_CLAIMS),
        sources=min(sources, MAX_SOURCES),
    )


def _d1_schema_for(items) -> dict[str, Any]:
    """The rolling view's schema for THIS window: one thesis per D1 thing said."""
    count = len(list(items or ()))
    return _bounded_schema(
        D1_VIEW_JSON_SCHEMA,
        open_theses=min(count, MAX_OPEN_THESES),
        sources=min(count, MAX_SOURCES),
    )


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def _root(root: Path | None) -> Path:
    import day_review_pack

    return Path(root) if root is not None else day_review_pack.default_root()


def narration_path(session_date: str, *, root: Path | None = None) -> Path:
    """`<root>/narration/<date>.json` - ONE verified story per session."""
    return _root(root) / "narration" / f"{str(session_date or '').strip()[:10]}.json"


def read_narration(session_date: str, *, root: Path | None = None) -> dict[str, Any] | None:
    return _read_json(narration_path(session_date, root=root))


def d1_view_path(*, root: Path | None = None) -> Path:
    """`<root>/d1_view.json` - ONE rolling file, not one per day."""
    return _root(root) / "d1_view.json"


def read_d1_view(*, root: Path | None = None) -> dict[str, Any] | None:
    return _read_json(d1_view_path(root=root))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _moment(now: datetime | None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate(payload: Any, schema: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """The closed schema, top level and one level into every declared object.

    `ai_summary.validate_structured_output` is the SHARED validator and it stops
    at the top level's own strings: it does NOT enforce `maxItems`, nor the
    `maxLength` of an array's ITEMS. That gap is inherited from
    `market_story_narration`'s pattern, and TJ-4 is the first consumer that
    RENDERS a per-item list from the answer - a 5,000-source, 500-claim reply
    was accepted, written whole (372 KB) and then drawn line by line on the Qt
    thread (reviewer, 2026-09-20). So the bounds this module declares are
    enforced HERE rather than by widening the shared validator under every
    other caller.
    """
    import ai_summary

    body = ai_summary.validate_structured_output(payload, schema, name=name)
    for key, spec in (schema.get("properties") or {}).items():
        if key not in body:
            continue
        kind = str(spec.get("type") or "")
        if kind == "object":
            ai_summary.validate_structured_output(body[key], spec, name=f"{name}.{key}")
            continue
        if kind != "array":
            continue
        rows = list(body[key] or ())
        limit = spec.get("maxItems")
        if isinstance(limit, int) and not isinstance(limit, bool) and len(rows) > limit:
            raise NarrationRejected(
                f"{name}.{key} carries {len(rows)} items; at most {limit} are allowed"
            )
        item_spec = spec.get("items") or {}
        item_kind = str(item_spec.get("type") or "")
        if item_kind == "object":
            for index, item in enumerate(rows):
                ai_summary.validate_structured_output(
                    item, item_spec, name=f"{name}.{key}[{index}]"
                )
        elif item_kind == "string":
            longest = item_spec.get("maxLength")
            for index, item in enumerate(rows):
                if isinstance(longest, int) and len(str(item)) > longest:
                    raise NarrationRejected(
                        f"{name}.{key}[{index}] is longer than {longest} characters"
                    )
    return body


def _by_source_id(rows, *, what: str) -> dict[str, Mapping[str, Any]]:
    """`source_id -> row`, RAISING when one id names two rows.

    A dict comprehension over a pack with a duplicated id silently keeps the
    LAST row, so a narration could quote the second row's verdict while naming
    the first and still pass the equality check (reviewer, 2026-09-20). The
    pack's own minter makes a duplicate unmintable; this is the reader's half of
    the same rule, for a pack an older build wrote or a hand edited.
    """
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        source_id = str(row.get("source_id") or "")
        if source_id in out:
            raise NarrationRejected(
                f"the pack's {what} carries {source_id!r} twice; one id must name one row"
            )
        out[source_id] = row
    return out


def _check_sources(narration: Mapping[str, Any], allowed: set[str]) -> None:
    cited = [str(item) for item in narration.get("sources") or ()]
    if not cited:
        raise NarrationRejected("the narration cited nothing at all")
    outside = [item for item in cited if item not in allowed]
    if outside:
        raise NarrationRejected(
            f"the narration cited id(s) the pack does not carry: {', '.join(sorted(outside))}"
        )


def _check_day_narration(narration: Mapping[str, Any], pack: Mapping[str, Any]) -> None:
    """The rule of this packet: a narrated verdict IS the measured verdict."""
    import day_review_pack

    allowed = set(day_review_pack.allowed_source_ids(pack))
    _check_sources(narration, allowed)
    # A story with no headline is written and then read as "No story yet" OVER a
    # story, because the page keeps its own line when there is nothing to put in
    # its place (reviewer, 2026-09-20). An answer that says nothing is not an
    # answer; it is rejected like any other breach.
    if not str(narration.get("headline") or "").strip():
        raise NarrationRejected("the narration carries no headline")

    reads = _by_source_id(pack.get("reads"), what="reads")
    said = _by_source_id(pack.get("trader_said"), what="trader_said")
    graded: set[str] = set()
    for claim in narration.get("were_you_right") or ():
        if not isinstance(claim, Mapping):
            raise NarrationRejected("a graded claim was not an object")
        evidence_id = str(claim.get("evidence_id") or "")
        row = reads.get(evidence_id)
        if row is None:
            raise NarrationRejected(
                f"the narration graded a claim no read row carries: {evidence_id!r}"
            )
        # One read, one verdict. Two claims on one read row are two readings of
        # a single measured thing, and the page would print both as if the desk
        # had measured twice.
        if evidence_id in graded:
            raise NarrationRejected(
                f"the narration graded {evidence_id!r} twice; one read carries one verdict"
            )
        graded.add(evidence_id)
        measured = str(row.get("verdict") or "")
        stated = str(claim.get("verdict") or "")
        if stated != measured:
            raise NarrationRejected(
                f"the narration said {stated!r} where the measured row says {measured!r}"
            )
        source_id = str(claim.get("source_id") or "")
        item = said.get(source_id)
        if item is None:
            raise NarrationRejected(
                f"a graded claim named {source_id!r}, which is not something the trader said"
            )
        if str(item.get("kind") or "") != day_review_pack.KIND_PREDICTION:
            raise NarrationRejected(
                "the narration graded an observation as a call; only a prediction "
                "is a call"
            )
        if str(item.get("entry_id") or "") != str(row.get("entry_id") or ""):
            raise NarrationRejected(
                "a graded claim linked a read to a prediction from another entry"
            )

    chased = narration.get("chased_against_news")
    if not isinstance(chased, Mapping):
        raise NarrationRejected("chased_against_news was not an object")
    verdict = str(chased.get("verdict") or "")
    if not pack.get("forecast") and verdict != "unknown":
        raise NarrationRejected(
            "the pack carries no pasted forecast, so chased_against_news can only "
            f"be 'unknown'; the narration said {verdict!r}"
        )
    evidence_id = str(chased.get("evidence_id") or "")
    if evidence_id and evidence_id not in allowed:
        raise NarrationRejected(
            f"chased_against_news cited an id the pack does not carry: {evidence_id!r}"
        )


def _materialize_v2_narration(
    narration: Mapping[str, Any], pack: Mapping[str, Any]
) -> dict[str, Any]:
    """Turn v2 explanations into the persisted v1 claims using canonical links."""
    pairs, _not_offered = _read_explanation_contract(pack)
    explanations = narration.get("read_explanations")
    if not isinstance(explanations, Mapping):
        raise NarrationRejected("read_explanations was not an object")
    keys = [str(key) for key in explanations]
    expected = list(pairs)
    if set(keys) != set(expected) or len(keys) != len(expected):
        raise NarrationRejected("read_explanations must contain exactly the offered read ids")
    claims: list[dict[str, str]] = []
    for read_id, canonical in pairs.items():
        explanation = explanations.get(read_id)
        if not isinstance(explanation, str) or len(explanation) > 240:
            raise NarrationRejected(
                f"the explanation for {read_id!r} must be a string of at most 240 characters"
            )
        claims.append(
            {
                "claim": explanation,
                "source_id": canonical["prediction_source_id"],
                "verdict": canonical["verdict"],
                "evidence_id": read_id,
            }
        )
    body = dict(narration)
    body.pop("read_explanations", None)
    body["were_you_right"] = claims
    return body


def _check_d1_narration(narration: Mapping[str, Any], allowed: set[str]) -> None:
    _check_sources(narration, allowed)
    for thesis in narration.get("open_theses") or ():
        if not isinstance(thesis, Mapping):
            raise NarrationRejected("an open thesis was not an object")
        evidence_id = str(thesis.get("evidence_id") or "")
        if evidence_id not in allowed:
            raise NarrationRejected(
                f"a standing thesis cited {evidence_id!r}, which is not a D1 thing "
                "the trader said inside the window"
            )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def _previous_story(session: str, root: Path) -> dict[str, Any]:
    """Yesterday's narration, READ-ONLY context. Never rewritten, never cited."""
    try:
        import market_calendar

        yesterday = market_calendar.previous_session(
            date.fromisoformat(str(session)[:10])
        ).isoformat()
    except Exception:  # noqa: BLE001 - no yesterday is simply less context
        return {}
    stored = read_narration(yesterday, root=root)
    if not isinstance(stored, Mapping):
        return {}
    narration = stored.get("narration")
    return {
        "session_date": yesterday,
        "narration": dict(narration) if isinstance(narration, Mapping) else {},
    }


#: Ceiling on the day evidence the model reads, in characters of JSON. The
#: medium model evaluates ~118 prompt tokens/s at ~2.1 chars/token, so this
#: keeps prompt evaluation near three minutes inside the call's timeout.
MAX_DAY_EVIDENCE_CHARS = 60_000

#: Longest pasted forecast text the model reads; its parsed fields stay whole.
MAX_FORECAST_TEXT_CHARS = 6_000

#: How many member ids one congruence row shows; the rest are counted.
MAX_CONGRUENCE_MEMBER_IDS = 8

#: The trade keys a day story can use. Account ids, FX and reconcile fields stay out.
_TRADE_KEYS = (
    "source_id", "trade_id", "symbol", "direction", "status", "trade_date",
    "opened_at", "closed_at", "quantity_opened", "quantity_closed",
    "average_entry_price", "average_exit_price", "net_pnl", "currency",
    "planned_entry", "planned_stop", "planned_risk", "setup_tags",
    "display_tags", "notes", "exit_note", "exit_fields",
)


def _trade_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """One journal trade reduced to what a story can narrate."""
    view = {key: row.get(key) for key in _TRADE_KEYS if key in row}
    review = row.get("trade_review")
    if isinstance(review, Mapping):
        entry_raw = review.get("entry_raw")
        exit_raw = review.get("exit_raw")
        entry_raw = entry_raw if isinstance(entry_raw, Mapping) else {}
        exit_raw = exit_raw if isinstance(exit_raw, Mapping) else {}
        view["trader_words"] = {
            "entry": str(entry_raw.get("text") or ""),
            "exit": str(exit_raw.get("text") or ""),
            "entry_answers": review.get("entry_answers") or {},
            "review_pnl_note": str(review.get("review_pnl_note") or ""),
        }
    return view


def _internals_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """One internals mark as its one-line summary; the per-symbol table stays out."""
    context = row.get("context")
    context = context if isinstance(context, Mapping) else {}
    common = context.get("common")
    common = common if isinstance(common, Mapping) else {}
    return {
        "source_id": row.get("source_id"),
        "kind": row.get("kind"),
        "at": row.get("at"),
        "availability": common.get("availability"),
        "internals": common.get("internals"),
        "reason": common.get("reason"),
    }


def _congruence_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """One congruence line with its member-id list capped and counted."""
    members = row.get("source_ids")
    if not isinstance(members, (list, tuple)) or len(members) <= MAX_CONGRUENCE_MEMBER_IDS:
        return dict(row)
    return {
        **row,
        "source_ids": list(members[:MAX_CONGRUENCE_MEMBER_IDS]),
        "source_ids_total": len(members),
    }


def _model_pack(pack: Mapping[str, Any]) -> dict[str, Any]:
    """The pack sections the model reads, trimmed to fit the medium model.

    Every citable source id stays visible. Only bulk a story cannot use is
    dropped: per-symbol internals tables, multi-session skill cells, trade
    bookkeeping, report-card counters, long congruence member lists, and
    forecast text past a cap.
    """
    import day_review_pack

    view: dict[str, Any] = {name: pack.get(name) for name in day_review_pack.SECTIONS}
    view["internals"] = [
        _internals_view(row) for row in pack.get("internals") or () if isinstance(row, Mapping)
    ]
    skill = pack.get("skill")
    if isinstance(skill, Mapping):
        view["skill"] = {key: value for key, value in skill.items() if key != "lately"}
    trades = pack.get("trades")
    if isinstance(trades, Mapping):
        view["trades"] = {
            **{key: value for key, value in trades.items() if key != "rows"},
            "rows": [
                _trade_view(row) for row in trades.get("rows") or () if isinstance(row, Mapping)
            ],
        }
    view["congruence"] = [
        _congruence_view(row) for row in pack.get("congruence") or () if isinstance(row, Mapping)
    ]
    card = pack.get("report_card")
    if isinstance(card, Mapping):
        view["report_card"] = {
            **{key: value for key, value in card.items() if key != "lines"},
            "lines": [
                {key: line.get(key) for key in ("source_id", "key", "text")}
                for line in card.get("lines") or ()
                if isinstance(line, Mapping)
            ],
        }
    forecast = pack.get("forecast")
    if isinstance(forecast, Mapping):
        text = str(forecast.get("text") or "")
        if len(text) > MAX_FORECAST_TEXT_CHARS:
            view["forecast"] = {
                **forecast,
                "text": text[:MAX_FORECAST_TEXT_CHARS],
                "text_truncated": f"first {MAX_FORECAST_TEXT_CHARS} of {len(text)} characters",
            }
    return view


def _previous_view(previous: Mapping[str, Any]) -> dict[str, Any]:
    """Yesterday's story as headline and summary only: continuity, not evidence."""
    narration = previous.get("narration")
    narration = narration if isinstance(narration, Mapping) else {}
    if not previous:
        return {}
    return {
        "session_date": previous.get("session_date"),
        "headline": narration.get("headline"),
        "what_happened": narration.get("what_happened"),
    }


def _day_evidence(pack: Mapping[str, Any], root: Path) -> dict[str, Any]:
    import day_review_pack

    session = str(pack.get("session_date") or "")
    allowed = list(day_review_pack.allowed_source_ids(pack))
    pairs, not_offered = _read_explanation_contract(pack)
    evidence = {
        "package_id": f"day-review:{str(pack.get('inputs_hash') or '')[:16]}",
        "evidence_hash": str(pack.get("inputs_hash") or ""),
        "instructions": DAY_INSTRUCTIONS,
        "allowed_source_ids": allowed,
        # The model sees only a closed, code-owned mapping.  An omission is
        # named and counted so an observation/extracted read never quietly
        # becomes a clicked prediction.
        "read_explanations": pairs,
        "read_explanations_not_offered": not_offered,
        "session_date": session,
        "pack": _model_pack(pack),
        # Read-only, and deliberately outside `allowed_source_ids`: last night's
        # story is context for continuity, never a fact this night may cite.
        "previous_day": _previous_view(_previous_story(session, root)),
    }
    size = len(json.dumps(evidence, sort_keys=True, default=str))
    if size > MAX_DAY_EVIDENCE_CHARS:
        # Refused before the model loads: a prompt this size times out instead.
        raise NarrationRejected(
            f"the day evidence is {size} characters; the medium model reads at most "
            f"{MAX_DAY_EVIDENCE_CHARS} inside the {TIMEOUT_SECONDS}s call"
        )
    return evidence


def _d1_window(session: str) -> list[str]:
    """The last `LATELY_SESSIONS` exchange sessions ending at `session`."""
    import evidence_stats
    import market_calendar

    cursor = date.fromisoformat(str(session)[:10])
    days = [cursor.isoformat()]
    while len(days) < int(evidence_stats.LATELY_SESSIONS):
        cursor = market_calendar.previous_session(cursor)
        days.append(cursor.isoformat())
    return list(reversed(days))


def _d1_items(session: str, root: Path) -> list[dict[str, Any]]:
    """The trader's D1 clicks and D1 notes across the window, oldest first.

    Never an M5 item - a rest-of-day read is a read about the tape, not a belief
    about the bigger picture - and never a session outside the window.
    """
    import day_review_pack

    items: list[dict[str, Any]] = []
    for day in _d1_window(session):
        pack = day_review_pack.read_pack(day, root=root)
        if not isinstance(pack, Mapping):
            continue
        for item in day_review_pack.said_items(pack, timeframe="D1"):
            items.append({**item, "session_date": day})
    return items


def _d1_evidence(session: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "package_id": f"d1-view:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": D1_VIEW_INSTRUCTIONS,
        "allowed_source_ids": [str(item.get("source_id") or "") for item in items],
        "session_date": session,
        "window_sessions": sorted({str(item.get("session_date") or "") for item in items}),
        "d1_said": items,
    }


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def _request_for(request: Callable[..., Mapping[str, Any]] | None):
    """The injected seam, or the local provider when there is one."""
    import ai_summary

    if request is not None:
        return request, ""
    if not ai_summary.local_provider_enabled():
        return None, "local AI is not configured; the prior narration was kept"
    return ai_summary.request_ai_summary, ""


def _call(request, *, evidence: Mapping[str, Any], schema, prompt_version: str, schema_name: str):
    import ai_summary

    return request(
        provider="local",
        model=ai_summary.local_model(MODEL_TIER),
        api_key="",
        evidence=dict(evidence),
        timeout_seconds=TIMEOUT_SECONDS,
        schema=schema,
        schema_name=schema_name,
        prompt_version=prompt_version,
    )


def _clock_for(now: datetime | None) -> Callable[[], datetime]:
    """A clock that starts at `now` and MOVES. One seam, injectable in a test.

    The window is re-asked between model calls, so the moment has to advance -
    and it may never advance by sleeping. This is `now` plus the wall time this
    run has actually spent, measured monotonically.
    """
    start = now or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    began = time.monotonic()

    def _at() -> datetime:
        return start + timedelta(seconds=time.monotonic() - began)

    return _at


def _window_allows(clock: Callable[[], datetime]) -> tuple[bool, str]:
    """May this run start ANOTHER model call right now?

    The slot's `reserve_minutes` buys the FIRST call and nothing more. A night
    that narrated its own session at 07:50 ET could otherwise sweep three more
    and be loading a model at 08:35, past the window's close, which is the
    night-only rule broken from inside (reviewer round 2, 2026-09-20). So every
    call after the first asks again, with one call's worth of room.
    """
    from ai_jobs import window

    try:
        return window.launch_allowed(clock(), reserve_minutes=SWEEP_CALL_MINUTES)
    except Exception as exc:  # noqa: BLE001 - an unanswerable window STOPS the run
        return False, f"the night window could not be read: {exc}"


def queued_sessions(root: Path | None = None, *, skip: str = "") -> list[str]:
    """Sessions with a `redo_requested` marker on disk, OLDEST first.

    The Day Review page's default pick during a session day is the PREVIOUS
    session, so the DEFAULT daytime Redo queues a day the night was never going
    to narrate. Nothing read those markers, so the trader was told "queued for
    tonight" and nothing ever ran (reviewer, 2026-09-20). This is what makes the
    sentence true: the night finds them.
    """
    import day_review_pack

    base = _root(root)
    try:
        names = sorted(child.name for child in (base / "sessions").iterdir() if child.is_dir())
    except OSError:
        return []
    ignore = str(skip or "")[:10]
    return [
        name
        for name in names
        # A folder that is not a session DATE is not a queue entry, whatever a
        # future writer leaves beside the packs. A marker is never retired by
        # age: uncertainty never deletes.
        if _SESSION_DATE.fullmatch(name)
        and name != ignore
        and day_review_pack.redo_requested(name, root=base)
    ]


def run_day_review_narration(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    only_this_session: bool = False,
    clock: Callable[[], datetime] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Narrate one session, refresh the D1 view, sweep what the trader queued.

    Never raises: a crash here is a lost night. Every failure path leaves the
    last verified file byte-identical and says what happened.

    ``only_this_session`` is set by the runner when the operator named a day
    (``--session``): a targeted redo does what it was asked for and nothing
    else. The unattended nightly run sweeps.

    The order is the night's own story, then the rolling D1 view, then the
    queue: the two things this session owes come first, and what is left of the
    window goes to the backlog. Every call after the first re-asks the window.
    """
    import day_review_pack

    base = _root(root)
    tick = clock or _clock_for(now)
    session = str(session_date or "").strip()[:10] or datetime.now().date().isoformat()
    pack = day_review_pack.read_pack(session, root=base)
    redo = day_review_pack.redo_requested(session, root=base)
    outputs: list[str] = []
    reasons: list[str] = []
    model = ""
    degraded = False
    own = "ok"

    if pack is None:
        # The post-close tick never reached this session. That is a `skipped`
        # row with a reason, not a failure and not an invented story - an
        # evidence job is never allowed to cost the night. It does not cost the
        # QUEUE either: what the trader asked for on OTHER days is still swept
        # below, or one missing pack tonight would strand it for ever.
        own = "skipped"
        reasons.append(f"no day pack for {session}; nothing to narrate")
    else:
        story = _run_day_story(
            session, pack, base, now=now, request=request, redo=redo
        )
        outputs.extend(story["outputs"])
        reasons.append(story["reason"])
        model = model or story["model"]
        degraded = degraded or story["status"] == "degraded_no_narrative"
        if story["status"] == "ok" and redo:
            day_review_pack.clear_redo(session, root=base)

    view = _run_d1_view(
        session, base, now=now, request=request, redo=redo, clock=tick
    )
    outputs.extend(view["outputs"])
    if view["reason"]:
        reasons.append(view["reason"])
    model = model or view["model"]
    degraded = degraded or view["status"] == "degraded_no_narrative"

    if not only_this_session:
        swept = _sweep_queued(session, base, now=now, request=request, clock=tick)
        outputs.extend(swept["outputs"])
        if swept["reason"]:
            reasons.append(swept["reason"])
            model = model or swept["model"]

    # The night's OWN story and the rolling view decide the status. A queued
    # session that was rejected is reported in the reason and keeps its marker
    # for tomorrow; it does not take this night's `ok` away, because the story
    # the trader opens in the morning was written.
    return {
        "status": "degraded_no_narrative" if degraded else own,
        "model": model,
        "reason": "; ".join(part for part in reasons if part),
        "outputs": outputs,
    }


def _sweep_queued(
    session: str,
    root: Path,
    *,
    now: datetime | None,
    request: Callable[..., Mapping[str, Any]] | None,
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    """Narrate the sessions a daytime Redo queued. At most `REDO_SWEEP_LIMIT`.

    Oldest first, so a queue that outgrows one night drains in order rather than
    starving its oldest entry. Six rules, and each of them is a test:

    * the marker OVERRIDES that session's unchanged-hash skip - the trader asked
      for the story to be written again and the pack has not moved;
    * **the budget is spent on sessions NARRATED, never on names in a list.**
      Three markers whose sessions had no pack used to consume the whole budget
      and, being the oldest, sat at the head of the queue every night after -
      so no real request behind them was ever attempted (reviewer round 2);
    * a marker is cleared only after a GOOD run. A rejected or raising redo
      leaves that session's prior story byte-identical AND keeps its marker, so
      the next night tries again;
    * a queued session with no pack is skipped with its marker KEPT and named
      (at most `MAX_NAMED_UNBUILT` of them, then `+N more`) - the post-close
      tick may simply not have reached it yet, and by TJ-4's review round 2 the
      page builds the pack before it queues anything;
    * every call re-asks the launch window, and a window that has closed stops
      the sweep CLEANLY with every remaining marker kept and said;
    * one bad queued session never costs the night's own story or the rolling
      D1 view. This returns a REPORT; it cannot fail the night.
    """
    import day_review_pack

    queued = queued_sessions(root, skip=session)
    if not queued:
        return {"status": "ok", "model": "", "reason": "", "outputs": []}
    outputs: list[str] = []
    narrated: list[str] = []
    kept: list[str] = []
    unbuilt: list[str] = []
    left: list[str] = []
    closed = ""
    model = ""
    for index, day in enumerate(queued):
        if len(narrated) + len(kept) >= REDO_SWEEP_LIMIT:
            left = queued[index:]
            break
        pack = day_review_pack.read_pack(day, root=root)
        if pack is None:
            # Costs no model call, so it costs no budget either.
            unbuilt.append(day)
            continue
        allowed, why = _window_allows(clock)
        if not allowed:
            closed = why
            left = queued[index:]
            break
        outcome = _run_day_story(day, pack, root, now=now, request=request, redo=True)
        outputs.extend(outcome["outputs"])
        model = model or outcome["model"]
        if outcome["status"] == "ok":
            day_review_pack.clear_redo(day, root=root)
            narrated.append(day)
        else:
            kept.append(day)
    parts: list[str] = []
    if narrated:
        parts.append("redo queued by the trader, narrated: " + ", ".join(narrated))
    if kept:
        parts.append("still queued after a rejected redo: " + ", ".join(kept))
    if unbuilt:
        named = ", ".join(unbuilt[:MAX_NAMED_UNBUILT])
        over = len(unbuilt) - MAX_NAMED_UNBUILT
        parts.append(
            "still queued, no pack yet: " + named + (f", +{over} more" if over > 0 else "")
        )
    if closed:
        parts.append(f"the night window closed mid-sweep ({closed})")
    if left:
        parts.append(
            f"{len(left)} more queued session(s) wait for tomorrow night "
            f"(at most {REDO_SWEEP_LIMIT} a night)"
        )
    return {
        "status": "ok",
        "model": model,
        "reason": "; ".join(parts),
        "outputs": outputs,
    }


def _run_day_story(
    session: str,
    pack: Mapping[str, Any],
    root: Path,
    *,
    now: datetime | None,
    request: Callable[..., Mapping[str, Any]] | None,
    redo: bool,
) -> dict[str, Any]:
    destination = narration_path(session, root=root)
    existing = _read_json(destination) or {}
    if (
        not redo
        and existing.get("inputs_hash") == pack.get("inputs_hash")
        and existing.get("prompt_version") == PROMPT_VERSION
    ):
        return {
            "status": "ok",
            "model": str(existing.get("model") or ""),
            "reason": "the verified day story is unchanged for this session",
            "outputs": [str(destination)],
        }

    caller, refusal = _request_for(request)
    if caller is None:
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": refusal,
            "outputs": [],
        }
    try:
        # A malformed/ambiguous pack is evidence failure too.  Keep the last
        # story and report it through the same no-raise nightly seam.
        schema = _narration_schema_for(pack)
        result = _call(
            caller,
            evidence=_day_evidence(pack, root),
            schema=schema,
            prompt_version=PROMPT_VERSION,
            schema_name="tradingbot_day_review_narration",
        )
        reply = result.get("summary") if isinstance(result, Mapping) else None
        if not isinstance(reply, Mapping):
            raise NarrationRejected("the day story was not an object")
        has_v2 = "read_explanations" in reply
        has_legacy = "were_you_right" in reply
        if has_v2 and has_legacy:
            raise NarrationRejected("the reply mixed v1 claims with v2 explanations")
        if has_v2:
            narration = _materialize_v2_narration(
                _validate(reply, schema, name="day story"), pack
            )
        else:
            # Existing saved/provider fixtures are still a hard v1 contract.
            # They are checked just as strictly as before and never repaired.
            narration = _validate(
                reply, _legacy_narration_schema_for(pack), name="legacy day story"
            )
        _check_day_narration(narration, pack)
        payload = {
            "schema": SCHEMA,
            "session_date": session,
            "generated_at": _moment(now),
            "inputs_hash": str(pack.get("inputs_hash") or ""),
            "prompt_version": PROMPT_VERSION,
            "model": str(result.get("model") or ""),
            # A SIZE statement, counted here so the page states it without
            # counting anything: how many of the session's measured reads this
            # story actually graded. No result is involved and nothing is
            # ranked - the page says `graded K of N reads` when K < N.
            "graded": {
                "reads_graded": len(narration.get("were_you_right") or ()),
                "reads_in_pack": len(
                    [row for row in pack.get("reads") or () if isinstance(row, Mapping)]
                ),
            },
            "narration": dict(narration),
        }
        _atomic_write(destination, payload)
    except Exception as exc:  # noqa: BLE001 - the prior verified file is the fallback
        _log.debug("The day story was not written.", exc_info=True)
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"the day story was rejected; the prior story was kept: {exc}",
            "outputs": [],
        }
    return {
        "status": "ok",
        "model": str(result.get("model") or ""),
        "reason": f"grounded day story written for {session}",
        "outputs": [str(destination)],
    }


def _run_d1_view(
    session: str,
    root: Path,
    *,
    now: datetime | None,
    request: Callable[..., Mapping[str, Any]] | None,
    redo: bool,
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    try:
        items = _d1_items(session, root)
    except Exception as exc:  # noqa: BLE001 - an unanswerable calendar names no window
        return {
            "status": "skipped",
            "model": "",
            "reason": f"the rolling D1 window could not be walked: {exc}",
            "outputs": [],
        }
    if not items:
        return {
            "status": "skipped",
            "model": "",
            "reason": "",
            "outputs": [],
        }
    evidence = _d1_evidence(session, items)
    destination = d1_view_path(root=root)
    existing = _read_json(destination) or {}
    if (
        not redo
        and existing.get("inputs_hash") == evidence["evidence_hash"]
        and existing.get("prompt_version") == D1_VIEW_PROMPT_VERSION
    ):
        return {
            "status": "ok",
            "model": str(existing.get("model") or ""),
            "reason": "",
            "outputs": [str(destination)],
        }

    # The SECOND call of the night asks the window again: the slot's reserve
    # bought the FIRST one and nothing more (reviewer round 2, 2026-09-20).
    # Asked here rather than at the top of this function, so a view that needs
    # no call at all - nothing said on a D1 card, or an unchanged hash - is
    # never reported as blocked by a window it never wanted.
    allowed, why = _window_allows(clock)
    if not allowed:
        return {
            "status": "skipped",
            "model": "",
            "reason": f"the rolling D1 view waits for the next night ({why})",
            "outputs": [],
        }

    caller, refusal = _request_for(request)
    if caller is None:
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": refusal,
            "outputs": [],
        }
    schema = _d1_schema_for(items)
    try:
        result = _call(
            caller,
            evidence=evidence,
            schema=schema,
            prompt_version=D1_VIEW_PROMPT_VERSION,
            schema_name="tradingbot_d1_view_narration",
        )
        narration = result.get("summary") if isinstance(result, Mapping) else None
        narration = _validate(narration, schema, name="D1 view")
        _check_d1_narration(narration, set(evidence["allowed_source_ids"]))
        payload = {
            "schema": D1_VIEW_SCHEMA,
            "session_date": session,
            "generated_at": _moment(now),
            "inputs_hash": evidence["evidence_hash"],
            "prompt_version": D1_VIEW_PROMPT_VERSION,
            "model": str(result.get("model") or ""),
            "narration": dict(narration),
        }
        _atomic_write(destination, payload)
    except Exception as exc:  # noqa: BLE001 - the prior rolling view is the fallback
        _log.debug("The rolling D1 view was not written.", exc_info=True)
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"the rolling D1 view was rejected; the prior view was kept: {exc}",
            "outputs": [],
        }
    return {
        "status": "ok",
        "model": str(result.get("model") or ""),
        "reason": "rolling D1 view refreshed",
        "outputs": [str(destination)],
    }


__all__ = [
    "D1_VIEW_JSON_SCHEMA",
    "D1_VIEW_PROMPT_VERSION",
    "D1_VIEW_SCHEMA",
    "MAX_GRADED_CLAIMS",
    "MAX_NAMED_UNBUILT",
    "MAX_OPEN_THESES",
    "MAX_SOURCES",
    "NARRATION_JSON_SCHEMA",
    "PROMPT_VERSION",
    "REDO_SWEEP_LIMIT",
    "SCHEMA",
    "SWEEP_CALL_MINUTES",
    "d1_view_path",
    "narration_path",
    "queued_sessions",
    "read_d1_view",
    "read_narration",
    "run_day_review_narration",
]
