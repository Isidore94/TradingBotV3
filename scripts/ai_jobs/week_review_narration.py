"""The week story - one grounded narration of five sessions (TJ-5 change 2).

`plan.md` §12.4 TJ-5 change 2 with its **AMENDED 2026-09-19** block, decision
0021 answers 14 and 20, decision 0018's stage order. The pattern is TJ-4's
`day_review_narration.py` beside it, and every rule it learned is repeated here
because each of them was a NO-GO on some packet the same week:

* **A JSON schema is a grammar hint, never a guard.** `maxItems` and
  `additionalProperties` tell the constrained decoder what shape to write; they
  do not stop a reply arriving in another shape. Every bound this module
  declares is RE-CHECKED in :func:`check_week_narration` and :func:`_validate`
  after the answer comes back.
* **The bounds come from the INPUT.** How many tendencies may be narrated is
  how many the contrast pack actually offers, not a fixed guess.
* **The model narrates measured rows and makes no verdict.** The
  ``were_you_right`` triple must EQUAL the one counted from the packs, a
  tendency must quote its cell's own ``n``, and every citation must be an id the
  week actually carries. A reply that breaks any of those is rejected WHOLE -
  not trimmed - and last Saturday's file stays byte-identical.
* **One call per week.** There is nothing to sweep: one week, one story.

What it may see: the five day PACKS, the five day NARRATIONS, the weekly
market-story rollup and the two contrast packs (TJ-15's misses, TJ-16's
tendencies). **Never bars and never the lake** - :data:`EVIDENCE_KEYS` is the
closed set, and a test asserts no bar or lake section can appear in it.

Where it is stored: ``<DAY_REVIEW_DIR>/week/<YYYY-Www>.json``, one file per ISO
week beside the day stories.

Fewer than :data:`MIN_NARRATED_DAYS` narrated days is not a story. It writes a
deterministic SCAFFOLD saying ``narrated K of 5`` and loads no model at all -
which is the state the trader's first Saturday is actually in: the live day
review folder held three session folders and ZERO packs on 2026-09-20.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai_jobs import ledger

_log = logging.getLogger(__name__)

PROMPT_VERSION = "week_review_narration_v1"
SCHEMA = "week_review_narration_v1"

#: The name the structured-output validator reports under.
SCHEMA_NAME = "tradingbot_week_review_narration"

#: `plan.md` TJ-5 change 2: *"Fewer than three narrated days -> a deterministic
#: scaffold and 'narrated K of 5'."* Three of five is the floor under a story
#: about a week; below it the honest answer is the count itself.
MIN_NARRATED_DAYS = 3

#: *"it may narrate at most three tendencies, each citing its cell and `n`"*
#: (packet TJ-5). The same number `prediction_contrast.TENDENCY_LIMIT` offers;
#: named here too because this module RE-CHECKS the bound after the reply and a
#: verifier that imported its own limit from the thing it is checking would be
#: checking nothing.
TENDENCY_LIMIT = 3

#: What separates a session from the id it qualifies. Each day pack mints its
#: ids with its OWN minter, so ``said:<entry>:prediction`` repeats across the
#: five packs and an unqualified citation in a week story would name two rows.
WEEK_SOURCE_SEPARATOR = "/"

#: The CLOSED set of evidence sections. Nothing else may be sent, and a test
#: asserts that no bar or lake section can ever appear in it.
EVIDENCE_KEYS: tuple[str, ...] = (
    "package_id",
    "evidence_hash",
    "instructions",
    "allowed_source_ids",
    "week_id",
    "sessions",
    "sessions_with_facts",
    "sessions_missing",
    "narrated_sessions",
    "days",
    "were_you_right",
    "tendencies",
    "misses",
    "rollup",
    "walkaway_totals",
)

#: How many of a day's own items travel in the week's evidence, per kind. The
#: week is five days wide, so a per-day budget is what keeps one busy session
#: from crowding out the other four. What does not fit is COUNTED and said
#: (`omitted`), never silently dropped - TJ-13A's bounded-package rule.
MAX_ITEMS_PER_DAY = 40

#: The ABSOLUTE ceilings on what one reply may contain. They are not the working
#: caps - those come from the input, below - and exist because the PAGE renders
#: these lists (TJ-4's reviewer round 1: a 5,000-source reply was accepted,
#: written whole and drawn line by line on the Qt thread).
MAX_EXAMPLES = 3
MAX_CHASED = 5
MAX_WATCH = 5
MAX_SOURCES = 512

#: How long one week-story call may take. The week story is one document on the
#: largest local model the desk owns, and the slot's `reserve_minutes` is
#: derived from the probe measurement rather than from this number.
TIMEOUT_SECONDS = 1800

#: What the slot reserves when the large model has NEVER been measured
#: (TJ-13B). Declared, not guessed at call time: with no probe row the week
#: story runs on the MEDIUM model, which is the tier `ai_summary` and the day
#: story already reserve minutes against.
DEFAULT_RESERVE_MINUTES = 30.0

WEEK_NARRATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "headline",
        "what_happened",
        "were_you_right",
        "chased",
        "tendencies",
        "process_pattern",
        "next_week_watch",
        "sources",
    ],
    "properties": {
        "headline": {"type": "string", "maxLength": 180},
        # Deliberately not 2,000: a `maxLength` of exactly 2,000 is the grammar
        # defect gate #144 found, where the decoder truncated mid-sentence.
        "what_happened": {"type": "string", "maxLength": 1500},
        "were_you_right": {
            "type": "object",
            "additionalProperties": False,
            "required": ["right", "wrong", "unresolved", "examples"],
            "properties": {
                "right": {"type": "integer"},
                "wrong": {"type": "integer"},
                "unresolved": {"type": "integer"},
                "examples": {
                    "type": "array",
                    "maxItems": MAX_EXAMPLES,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "source_id"],
                        "properties": {
                            "text": {"type": "string", "maxLength": 240},
                            "source_id": {"type": "string", "maxLength": 200},
                        },
                    },
                },
            },
        },
        "chased": {
            "type": "array",
            "maxItems": MAX_CHASED,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "source_id"],
                "properties": {
                    "text": {"type": "string", "maxLength": 240},
                    "source_id": {"type": "string", "maxLength": 200},
                },
            },
        },
        "tendencies": {
            "type": "array",
            "maxItems": TENDENCY_LIMIT,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "source_id", "n"],
                "properties": {
                    "text": {"type": "string", "maxLength": 300},
                    "source_id": {"type": "string", "maxLength": 200},
                    "n": {"type": "integer"},
                },
            },
        },
        "process_pattern": {"type": "string", "maxLength": 600},
        "next_week_watch": {
            "type": "array",
            "maxItems": MAX_WATCH,
            "items": {"type": "string", "maxLength": 240},
        },
        "sources": {
            "type": "array",
            "maxItems": MAX_SOURCES,
            "items": {"type": "string", "maxLength": 200},
        },
    },
}

WEEK_INSTRUCTIONS = (
    "Narrate this ONE exchange week from the evidence below and nothing else. "
    "You may not calculate a statistic, grade a call, or turn an unmeasured "
    "item into a fact. were_you_right must repeat the measured triple exactly "
    "as it is given; if you disagree with it, say nothing rather than changing "
    "it. A tendency may only be one of the cells you were handed, quoted with "
    "that cell's own n and its source_id. A day with no pack is a day nobody "
    "measured: name it, never describe it. Every id you cite must be copied "
    "exactly from allowed_source_ids."
)


class WeekNarrationRejected(ValueError):
    """The model's answer was not a narration of the evidence it was given."""


# ---------------------------------------------------------------------------
# identity and paths
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value or "").strip()


def week_id(session_date: Any) -> str:
    """The ISO week a session belongs to, ``YYYY-Www``.

    ISO rather than "the Friday", because the week story is one file per week
    and a Friday-holiday week has no Friday to name it with.
    """
    day = date.fromisoformat(_text(session_date)[:10])
    year, week, _weekday = day.isocalendar()
    return f"{year}-W{week:02d}"


def week_source_id(session: Any, source_id: Any) -> str:
    """``<session>/<source_id>`` - a citation that names ONE row of ONE day."""
    return f"{_text(session)[:10]}{WEEK_SOURCE_SEPARATOR}{_text(source_id)}"


def _root(root: Path | None) -> Path:
    import day_review_pack

    return Path(root) if root is not None else day_review_pack.default_root()


def narration_path(week: Any, *, root: Path | None = None) -> Path:
    """``<root>/week/<YYYY-Www>.json`` - ONE verified story per week."""
    return _root(root) / "week" / f"{_text(week)}.json"


def read_week_narration(week: Any, *, root: Path | None = None) -> dict[str, Any] | None:
    return _read_json(narration_path(week, root=root))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
# the week's own five days
# ---------------------------------------------------------------------------
def week_sessions(session_date: Any) -> tuple[str, ...]:
    """The exchange sessions of the week ``session_date`` falls in, in order.

    Monday to Friday of that ISO week, filtered through the exchange calendar,
    so a holiday week is four sessions and says so rather than padding itself to
    five. A calendar that cannot answer falls back to the five weekdays - fewer
    named days would be a silent narrowing of the week.
    """
    day = date.fromisoformat(_text(session_date)[:10])
    year, week, _weekday = day.isocalendar()
    weekdays = [date.fromisocalendar(year, week, index) for index in range(1, 6)]
    try:
        import market_calendar

        found = tuple(item.isoformat() for item in weekdays if market_calendar.is_session(item))
    except Exception:  # noqa: BLE001 - an unanswerable calendar names the weekdays
        _log.debug("The exchange calendar could not name this week.", exc_info=True)
        return tuple(item.isoformat() for item in weekdays)
    return found or tuple(item.isoformat() for item in weekdays)


def _day_narration(session: str, root: Path) -> Mapping[str, Any]:
    """TJ-4's stored day story for one session, or an empty mapping."""
    from ai_jobs import day_review_narration

    stored = day_review_narration.read_narration(session, root=root)
    if not isinstance(stored, Mapping):
        return {}
    body = stored.get("narration")
    return dict(body) if isinstance(body, Mapping) else {}


def _bounded(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> tuple[list[dict], int]:
    """At most :data:`MAX_ITEMS_PER_DAY` rows, narrowed to ``keys``, plus what was left."""
    kept = [
        {name: row.get(name) for name in keys}
        for row in rows[:MAX_ITEMS_PER_DAY]
        if isinstance(row, Mapping)
    ]
    return kept, max(0, len(rows) - len(kept))


def _day_block(session: str, pack: Mapping[str, Any] | None, story: Mapping[str, Any]):
    """One day's evidence, bounded, with every id session-qualified.

    Returns ``(block, ids)``: the ids are exactly what this block SHOWED, so a
    citation can never name a row the model was not handed.
    """
    if not isinstance(pack, Mapping) or not pack:
        return (
            {
                "session": session,
                "has_facts": False,
                "narrated": False,
                "note": "no day pack was built for this session; nobody measured it",
            },
            [],
        )
    said_rows = [row for row in pack.get("trader_said") or () if isinstance(row, Mapping)]
    read_rows = [row for row in pack.get("reads") or () if isinstance(row, Mapping)]
    card = pack.get("report_card") if isinstance(pack.get("report_card"), Mapping) else {}
    card_rows = [row for row in (card or {}).get("lines") or () if isinstance(row, Mapping)]

    said, said_over = _bounded(
        said_rows, ("kind", "timeframe", "at", "text", "direction", "horizon", "source_id")
    )
    reads, reads_over = _bounded(
        read_rows, ("timeframe", "direction", "horizon", "text", "verdict", "source_id")
    )
    lines, lines_over = _bounded(card_rows, ("key", "text", "n", "measured", "source_id"))

    ids: list[str] = []
    for rows in (said, reads, lines):
        for row in rows:
            row["source_id"] = week_source_id(session, row.get("source_id"))
            if row["source_id"] not in ids:
                ids.append(row["source_id"])

    walkaway = pack.get("walkaway") if isinstance(pack.get("walkaway"), Mapping) else {}
    trades = pack.get("trades") if isinstance(pack.get("trades"), Mapping) else {}
    block = {
        "session": session,
        "has_facts": True,
        "narrated": bool(story),
        "said": said,
        "reads": reads,
        "report_card": lines,
        # Counted over the day's WHOLE read list, never over the bounded copy
        # above: a tally taken from a truncated list is a number nobody
        # measured. TJ-5's day card prints this one.
        "tally": _tally_rows(read_rows),
        "said_counts": {
            "observations": sum(1 for row in said_rows if _text(row.get("kind")) == "observation"),
            "predictions": sum(1 for row in said_rows if _text(row.get("kind")) == "prediction"),
        },
        "trades": {"n": int(trades.get("n") or 0)},
        "walkaway_counts": dict(walkaway.get("counts") or {}),
        "story": {
            "headline": _text(story.get("headline")),
            "what_happened": _text(story.get("what_happened")),
            "process": _text(story.get("process")),
            "chased_against_news": story.get("chased_against_news") or {},
        },
        # What did not fit is COUNTED and said (TJ-13A's bounded-package rule).
        "omitted": {"said": said_over, "reads": reads_over, "report_card": lines_over},
    }
    return block, ids


def _tally_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """``right``/``wrong``/``unresolved``/``n`` over graded read rows.

    `pending <date>` and `unmeasured:<reason>` are unresolved, never wrong: a
    read nobody could close is not a read the trader got wrong. The verdict
    vocabulary is IMPORTED from its owner, never spelled here.
    """
    import market_read_grades as grades

    right = wrong = other = 0
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        verdict = _text(row.get("verdict"))
        if verdict == grades.VERDICT_RIGHT:
            right += 1
        elif verdict == grades.VERDICT_WRONG:
            wrong += 1
        else:
            other += 1
    return {"right": right, "wrong": wrong, "unresolved": other, "n": right + wrong + other}


def _tally(packs: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """The same tally over every pack the week HAS. A missing day adds nothing."""
    total = {"right": 0, "wrong": 0, "unresolved": 0, "n": 0}
    for pack in packs.values():
        for key, value in _tally_rows((pack or {}).get("reads") or ()).items():
            total[key] += value
    return total


def _walkaway_totals(packs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The week's A-D populations, SUMMED. Counts only, never a rate.

    A session with no pack contributes nothing and is named in
    ``sessions_missing``; a missing day is never a zero day.
    """
    import day_review_pack

    totals = {name: 0 for name in day_review_pack.WALKAWAY_POPULATIONS}
    for pack in packs.values():
        walkaway = (pack or {}).get("walkaway")
        counts = walkaway.get("counts") if isinstance(walkaway, Mapping) else None
        for name in totals:
            try:
                totals[name] += int((counts or {}).get(name) or 0)
            except (TypeError, ValueError):
                continue
    return {"counts": totals, "sessions": len(packs), "n": sum(totals.values())}


def _tendencies(session: str) -> list[dict[str, Any]]:
    """TJ-16's bounded view, each cell carrying its own citable id.

    `prediction_contrast.tendencies` already applies the floor and the SIZE
    order (n descending, then name). Nothing is re-ranked here - gate #43
    forbids an R statistic in that key - and nothing under
    `evidence_stats.MIN_REPORTABLE_N` is ever offered.
    """
    from ai_jobs import prediction_contrast

    try:
        pack = prediction_contrast.read_latest(session)
    except Exception:  # noqa: BLE001 - a missing contrast pack is simply no tendencies
        _log.debug("The prediction-contrast pack could not be read.", exc_info=True)
        return []
    if not isinstance(pack, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for item in prediction_contrast.tendencies(pack, limit=TENDENCY_LIMIT):
        cell = item.get("cell") or {}
        out.append(
            {
                "text": _text(item.get("text")),
                "n": int(item.get("n") or 0),
                "source_id": (
                    f"tendency:{_text(cell.get('horizon'))}:"
                    f"{_text(cell.get('table'))}:{_text(cell.get('key'))}"
                ),
                "cell": dict(cell),
            }
        )
    return out[:TENDENCY_LIMIT]


def _misses(session: str) -> dict[str, Any]:
    """TJ-15's miss contrast for the week's last session, bounded to its report."""
    from ai_jobs import miss_contrast

    try:
        pack = miss_contrast.read_latest(session)
    except Exception:  # noqa: BLE001 - a missing contrast pack is simply no misses
        _log.debug("The miss-contrast pack could not be read.", exc_info=True)
        return {}
    if not isinstance(pack, Mapping):
        return {}
    return {
        "session_date": _text(pack.get("session_date")),
        "groups": [dict(row) for row in pack.get("groups") or () if isinstance(row, Mapping)],
        "features": [dict(row) for row in pack.get("features") or () if isinstance(row, Mapping)],
        "thin_features": list(pack.get("thin_features") or ()),
        "excluded_by_timeframe": dict(pack.get("excluded_by_timeframe") or {}),
        "source_id": f"misses:{_text(pack.get('session_date'))}",
    }


def _rollup(week: str) -> dict[str, Any]:
    """WS-10D's weekly market-story pack, narrowed to its COVERAGE statement.

    The pack carries every daily story it rolled up; the week story needs what
    it covered and what it missed, not a second copy of five days.
    """
    try:
        from project_paths import MARKET_STORY_ROLLUPS_DIR

        stored = _read_json(Path(MARKET_STORY_ROLLUPS_DIR) / "weekly" / f"{week}.json")
    except Exception:  # noqa: BLE001 - no rollup is less context, never a failure
        _log.debug("The weekly market-story rollup could not be read.", exc_info=True)
        return {}
    if not isinstance(stored, Mapping):
        return {}
    return {
        name: stored.get(name)
        for name in (
            "period_id",
            "sessions_covered",
            "sessions_expected",
            "sessions_missing",
            "complete",
            "coverage_note",
        )
    }


def build_week_inputs(
    session_date: Any, *, root: Path | None = None, ledger_path: Any = None
) -> dict[str, Any]:
    """Everything the week story is allowed to see, already read. PURE-ish.

    It opens the five day packs, the five day narrations, the weekly rollup and
    the two contrast packs, and NOTHING else - no bars, no lake, no detector.
    ``ledger_path`` is accepted so a caller with its own ledger can hand one in;
    the week story itself reads no ledger (it is the SLOT's record).

    ``inputs_hash`` is over the sections and never over the clock, so the same
    unchanged week costs one model call and not one a night.
    """
    import day_review_pack

    base = _root(root)
    session = _text(session_date)[:10]
    sessions = week_sessions(session)
    week = week_id(session)

    packs: dict[str, Mapping[str, Any]] = {}
    for day in sessions:
        pack = day_review_pack.read_pack(day, root=base)
        if isinstance(pack, Mapping) and pack:
            packs[day] = pack

    days: list[dict[str, Any]] = []
    allowed: list[str] = []
    narrated: list[str] = []
    for day in sessions:
        story = _day_narration(day, base) if day in packs else {}
        if story:
            narrated.append(day)
        block, ids = _day_block(day, packs.get(day), story)
        days.append(block)
        for item in ids:
            if item not in allowed:
                allowed.append(item)

    tendencies = _tendencies(sessions[-1] if sessions else session)
    misses = _misses(sessions[-1] if sessions else session)
    for item in tendencies:
        if item["source_id"] not in allowed:
            allowed.append(item["source_id"])
    if misses.get("source_id") and misses["source_id"] not in allowed:
        allowed.append(misses["source_id"])

    body: dict[str, Any] = {
        "week_id": week,
        "sessions": list(sessions),
        "sessions_with_facts": [day for day in sessions if day in packs],
        "sessions_missing": [day for day in sessions if day not in packs],
        "narrated_sessions": narrated,
        "days": days,
        "allowed_source_ids": allowed,
        "were_you_right": _tally(packs),
        "tendencies": tendencies,
        "misses": misses,
        "rollup": _rollup(week),
        "walkaway_totals": _walkaway_totals(packs),
    }
    body["inputs_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return body


# ---------------------------------------------------------------------------
# the evidence, and the bounds that come out of it
# ---------------------------------------------------------------------------
def _evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    body = {name: inputs.get(name) for name in EVIDENCE_KEYS if name in inputs}
    body["package_id"] = f"week-review:{_text(inputs.get('inputs_hash'))[:16]}"
    body["evidence_hash"] = _text(inputs.get("inputs_hash"))
    body["instructions"] = WEEK_INSTRUCTIONS
    return body


def _schema_for(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """This week's schema: bounds taken FROM THE INPUT, under the ceilings.

    A fixed cap would make the model drop a tendency the week really offers, or
    make the answer unwritable on a busy week. The ceilings still apply, because
    a number that came from the evidence is still rendered by a page.
    """
    body = json.loads(json.dumps(WEEK_NARRATION_JSON_SCHEMA))
    properties = body["properties"]
    properties["tendencies"]["maxItems"] = min(
        TENDENCY_LIMIT, len(list(inputs.get("tendencies") or ()))
    )
    properties["sources"]["maxItems"] = min(
        MAX_SOURCES, len(list(inputs.get("allowed_source_ids") or ()))
    )
    reads = sum(
        len(list((day or {}).get("reads") or ()))
        for day in inputs.get("days") or ()
        if isinstance(day, Mapping)
    )
    properties["were_you_right"]["properties"]["examples"]["maxItems"] = min(
        MAX_EXAMPLES, reads
    )
    properties["chased"]["maxItems"] = min(MAX_CHASED, len(list(inputs.get("days") or ())))
    return body


def _validate(payload: Any, schema: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """The closed schema, top level and one level into every declared object.

    `ai_summary.validate_structured_output` is the SHARED validator and it stops
    at the top level's own strings: it enforces neither `maxItems` nor an
    array item's `maxLength`. TJ-4 found that gap the hard way; this is the same
    enforcement, kept out of the shared validator so no other caller moves.
    """
    import ai_summary

    body = ai_summary.validate_structured_output(payload, schema, name=name)
    _enforce(body, schema, name=name)
    return body


def _enforce(body: Mapping[str, Any], schema: Mapping[str, Any], *, name: str) -> None:
    import ai_summary

    for key, spec in (schema.get("properties") or {}).items():
        if key not in body:
            continue
        kind = str(spec.get("type") or "")
        if kind == "object":
            nested = ai_summary.validate_structured_output(
                body[key], spec, name=f"{name}.{key}"
            )
            _enforce(nested, spec, name=f"{name}.{key}")
            continue
        if kind != "array":
            continue
        rows = list(body[key] or ())
        limit = spec.get("maxItems")
        if isinstance(limit, int) and not isinstance(limit, bool) and len(rows) > limit:
            raise WeekNarrationRejected(
                f"{name}.{key} carries {len(rows)} items; at most {limit} are allowed"
            )
        item_spec = spec.get("items") or {}
        item_kind = str(item_spec.get("type") or "")
        if item_kind == "object":
            for index, item in enumerate(rows):
                nested = ai_summary.validate_structured_output(
                    item, item_spec, name=f"{name}.{key}[{index}]"
                )
                _enforce(nested, item_spec, name=f"{name}.{key}[{index}]")
        elif item_kind == "string":
            longest = item_spec.get("maxLength")
            for index, item in enumerate(rows):
                if isinstance(longest, int) and len(str(item)) > longest:
                    raise WeekNarrationRejected(
                        f"{name}.{key}[{index}] is longer than {longest} characters"
                    )


def check_week_narration(narration: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    """Re-check every bound and every number, against THIS week's input.

    Raises :class:`WeekNarrationRejected`. The schema handed to the provider is
    a grammar hint; this is the guard. Nothing is trimmed and nothing is partly
    kept - a story that breaks one of these rules is not published at all, and
    last Saturday's file stands.
    """
    allowed = set(_text(item) for item in inputs.get("allowed_source_ids") or ())

    cited = [_text(item) for item in narration.get("sources") or ()]
    if not cited:
        raise WeekNarrationRejected("the week story cited nothing at all")
    outside = sorted({item for item in cited if item not in allowed})
    if outside:
        raise WeekNarrationRejected(
            "the week story cited id(s) the week does not carry: " + ", ".join(outside)
        )

    # An answer that says nothing is not an answer: an empty headline is written
    # and then read as "no story yet" OVER a story (TJ-4, reviewer round 2).
    if not _text(narration.get("headline")):
        raise WeekNarrationRejected("the week story carries no headline")

    measured = inputs.get("were_you_right") or {}
    stated = narration.get("were_you_right")
    if not isinstance(stated, Mapping):
        raise WeekNarrationRejected("were_you_right was not an object")
    for key in ("right", "wrong", "unresolved"):
        try:
            said = int(stated.get(key))
        except (TypeError, ValueError):
            raise WeekNarrationRejected(
                f"were_you_right.{key} was not a count"
            ) from None
        if said != int(measured.get(key) or 0):
            raise WeekNarrationRejected(
                f"the week story said {key} {said} where the measured row says "
                f"{int(measured.get(key) or 0)}"
            )
    for index, example in enumerate(stated.get("examples") or ()):
        if not isinstance(example, Mapping):
            raise WeekNarrationRejected("a were_you_right example was not an object")
        source_id = _text(example.get("source_id"))
        if source_id not in allowed:
            raise WeekNarrationRejected(
                f"were_you_right example {index} cited {source_id!r}, which the week "
                "does not carry"
            )

    for index, item in enumerate(narration.get("chased") or ()):
        if not isinstance(item, Mapping):
            raise WeekNarrationRejected("a chased item was not an object")
        source_id = _text(item.get("source_id"))
        if source_id not in allowed:
            raise WeekNarrationRejected(
                f"chased item {index} cited {source_id!r}, which the week does not carry"
            )

    offered = {
        _text(item.get("source_id")): int(item.get("n") or 0)
        for item in inputs.get("tendencies") or ()
        if isinstance(item, Mapping)
    }
    told = list(narration.get("tendencies") or ())
    limit = min(TENDENCY_LIMIT, len(offered))
    if len(told) > limit:
        raise WeekNarrationRejected(
            f"the week story narrated {len(told)} tendencies; this week offers {limit}"
        )
    seen: set[str] = set()
    for item in told:
        if not isinstance(item, Mapping):
            raise WeekNarrationRejected("a tendency was not an object")
        source_id = _text(item.get("source_id"))
        if source_id not in offered:
            raise WeekNarrationRejected(
                f"the week story narrated a tendency the input does not hold: {source_id!r}"
            )
        if source_id in seen:
            raise WeekNarrationRejected(
                f"the week story narrated {source_id!r} twice; one cell is one tendency"
            )
        seen.add(source_id)
        try:
            said = int(item.get("n"))
        except (TypeError, ValueError):
            raise WeekNarrationRejected(
                f"the tendency {source_id!r} carries no count"
            ) from None
        if said != offered[source_id]:
            raise WeekNarrationRejected(
                f"the tendency {source_id!r} says n {said} where its cell says "
                f"{offered[source_id]}"
            )


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def reserve_minutes() -> float:
    """Minutes the week-story slot reserves, from TJ-13B's PROBE when one exists.

    A model nobody measured has no reserve (TJ-13B): with no probe row the week
    story runs on the MEDIUM model, and :data:`DEFAULT_RESERVE_MINUTES` is what
    that reserves. Read at slot-build time, like `ai_summary`'s, and guarded -
    an unreadable ledger costs the declared default, never the slot.
    """
    try:
        from ai_jobs import model_probe

        measured = model_probe.reserve_minutes_from_probe(tier="large")
    except Exception:  # noqa: BLE001 - an unreadable ledger is not a reserve
        _log.debug("The large-model probe could not be read.", exc_info=True)
        return DEFAULT_RESERVE_MINUTES
    if measured is None:
        return DEFAULT_RESERVE_MINUTES
    try:
        value = float(measured)
    except (TypeError, ValueError):
        return DEFAULT_RESERVE_MINUTES
    return value if value > 0 else DEFAULT_RESERVE_MINUTES


def _scaffold(inputs: Mapping[str, Any], *, now: datetime | None) -> dict[str, Any]:
    """The deterministic week, with no story in it and the count said out loud."""
    sessions = list(inputs.get("sessions") or ())
    narrated = list(inputs.get("narrated_sessions") or ())
    return {
        "schema": SCHEMA,
        "week_id": _text(inputs.get("week_id")),
        "generated_at": _moment(now),
        "inputs_hash": _text(inputs.get("inputs_hash")),
        "prompt_version": PROMPT_VERSION,
        "model": "",
        "scaffold": True,
        "narrated": f"narrated {len(narrated)} of {len(sessions)}",
        "sessions": sessions,
        "sessions_with_facts": list(inputs.get("sessions_with_facts") or ()),
        "sessions_missing": list(inputs.get("sessions_missing") or ()),
        "were_you_right": dict(inputs.get("were_you_right") or {}),
        "walkaway_totals": dict(inputs.get("walkaway_totals") or {}),
        # A scaffold is NOT a story. The page reads this key to decide whether
        # it has one, so it stays falsy rather than holding half an answer.
        "narration": {},
    }


def _ask(
    ask_provider: str,
    *,
    model: str,
    evidence: Mapping[str, Any],
    schema: Mapping[str, Any],
    request: Callable[..., Mapping[str, Any]] | None,
    ledger_path: Any,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """One model call. Returns ``(result, attribution)`` or RAISES.

    A provider that is not a local tier - ``openai``, which decision 0021
    answer 20 keeps as a setting that is off - is refused by the provider seam
    itself, whose sentence becomes the ledger row. That refusal happens here
    even when a test injects its own ``request``, so there is exactly ONE place
    that decides what this desk will speak to.
    """
    from ai_jobs import provider as provider_seam

    if ask_provider not in (provider_seam.LOCAL_LARGE, provider_seam.LOCAL_MEDIUM):
        # Raises without sending anything; nothing is spent.
        provider_seam.request_with_fallback(
            provider=ask_provider,
            evidence=evidence,
            schema=schema,
            schema_name=SCHEMA_NAME,
            ledger_path=ledger_path,
        )
        raise ValueError(f"unknown week-review provider {ask_provider!r}")

    if request is not None:
        result = request(
            provider="local",
            model=model,
            api_key="",
            evidence=dict(evidence),
            timeout_seconds=TIMEOUT_SECONDS,
            schema=schema,
            schema_name=SCHEMA_NAME,
            prompt_version=PROMPT_VERSION,
        )
        attribution = {
            "provider": ask_provider,
            "model_asked": model,
            "model_answered": _text((result or {}).get("model")) or model,
            "fallback_reason": "",
        }
        return (result or {}), attribution

    answer = provider_seam.request_with_fallback(
        provider=ask_provider,
        evidence=dict(evidence),
        schema=schema,
        schema_name=SCHEMA_NAME,
        ledger_path=ledger_path,
    )
    attribution = dict(answer.get("attribution") or {})
    result = answer.get("result")
    if not isinstance(result, Mapping):
        raise WeekNarrationRejected(
            attribution.get("fallback_reason") or "no local model answered the week story"
        )
    return result, attribution


def run_week_review_narration(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    ledger_path: Any = None,
    force: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    """Write ONE week story. Never raises: a crash here is a lost weekend night.

    ``force`` re-spends the unchanged-hash skip and NOTHING else. It never buys
    the night window - that is the runner's gate and TJ-13A item 1's rule, and
    a 27 GB model load in front of a trader at their desk on a Saturday
    afternoon is exactly what that rule is about.
    """
    base = _root(root)
    session = _text(session_date)[:10] or datetime.now().date().isoformat()
    try:
        inputs = build_week_inputs(session, root=base, ledger_path=ledger_path)
    except Exception as exc:  # noqa: BLE001 - an unreadable week is a recorded row
        _log.debug("The week's inputs could not be read.", exc_info=True)
        return {
            "status": ledger.STATUS_FAILED,
            "model": "",
            "reason": f"the week's inputs could not be read: {exc}",
            "outputs": [],
            "model_attribution": {},
        }

    week = inputs["week_id"]
    destination = narration_path(week, root=base)
    total = len(inputs["sessions"])
    narrated = len(inputs["narrated_sessions"])
    count = f"narrated {narrated} of {total}"

    if narrated < MIN_NARRATED_DAYS:
        # No model is loaded to say this. The deterministic scaffold IS the
        # honest answer, and on the trader's first Saturday it is the only one.
        outputs: list[str] = []
        try:
            _atomic_write(destination, _scaffold(inputs, now=now))
            outputs.append(str(destination))
        except OSError as exc:
            _log.debug("The week scaffold could not be written.", exc_info=True)
            return {
                "status": ledger.STATUS_FAILED,
                "model": "",
                "reason": f"{count}; the scaffold could not be written: {exc}",
                "outputs": [],
                "model_attribution": {},
            }
        return {
            "status": ledger.STATUS_SKIPPED,
            "model": "",
            "reason": (
                f"{count} day(s) of this week have a story, under the floor of "
                f"{MIN_NARRATED_DAYS}; the deterministic scaffold was written and no "
                "model was loaded"
            ),
            "outputs": outputs,
            "model_attribution": {},
        }

    existing = _read_json(destination) or {}
    if (
        not force
        and not existing.get("scaffold")
        and _text(existing.get("inputs_hash")) == _text(inputs.get("inputs_hash"))
        and _text(existing.get("prompt_version")) == PROMPT_VERSION
    ):
        return {
            "status": ledger.STATUS_OK,
            "model": _text(existing.get("model")),
            "reason": f"the verified week story is unchanged for {week}",
            "outputs": [str(destination)],
            "model_attribution": dict(existing.get("model_attribution") or {}),
        }

    from ai_jobs import provider as provider_seam

    plan = provider_seam.week_review_plan(ledger_path=ledger_path)
    configured = _text(plan.get("provider")) or provider_seam.LOCAL_LARGE
    if configured == provider_seam.LOCAL_LARGE and not plan.get("may_run_large"):
        # Lead decision, 2026-09-19: the trader wants a week story every
        # Saturday. A model nobody measured has no reserve, so that costs the
        # LARGE model and never the story - and the reason reaches the record.
        ask_provider = provider_seam.LOCAL_MEDIUM
    else:
        ask_provider = configured
    model = _text(plan.get("model"))
    if not model and ask_provider in (provider_seam.LOCAL_LARGE, provider_seam.LOCAL_MEDIUM):
        import ai_summary

        model = ai_summary.local_model(
            "large" if ask_provider == provider_seam.LOCAL_LARGE else "medium"
        )

    evidence = _evidence(inputs)
    schema = _schema_for(inputs)
    attribution: dict[str, Any] = {}
    try:
        result, attribution = _ask(
            ask_provider,
            model=model,
            evidence=evidence,
            schema=schema,
            request=request,
            ledger_path=ledger_path,
        )
        if not plan.get("may_run_large") and not _text(attribution.get("fallback_reason")):
            attribution["fallback_reason"] = _text(plan.get("reason"))
        narration = _validate(result.get("summary"), schema, name="week story")
        check_week_narration(narration, inputs)
        payload = {
            "schema": SCHEMA,
            "week_id": week,
            "generated_at": _moment(now),
            "inputs_hash": _text(inputs.get("inputs_hash")),
            "prompt_version": PROMPT_VERSION,
            "model": _text(result.get("model")) or model,
            "model_attribution": dict(attribution),
            "narrated": count,
            "sessions": list(inputs["sessions"]),
            "sessions_with_facts": list(inputs["sessions_with_facts"]),
            "sessions_missing": list(inputs["sessions_missing"]),
            "were_you_right": dict(inputs["were_you_right"]),
            "walkaway_totals": dict(inputs["walkaway_totals"]),
            "narration": dict(narration),
        }
        _atomic_write(destination, payload)
    except ValueError as exc:
        # A refused provider, a rejected answer, a malformed reply: one
        # sentence in the record, and the last verified week story - the file
        # the trader read last Saturday - byte-identical.
        _log.debug("The week story was not written.", exc_info=True)
        return {
            "status": ledger.STATUS_FAILED,
            "model": "",
            "reason": f"the week story was not written; the prior story was kept: {exc}",
            "outputs": [],
            "model_attribution": dict(attribution),
        }
    except Exception as exc:  # noqa: BLE001 - the prior verified file is the fallback
        _log.debug("The week story was not written.", exc_info=True)
        return {
            "status": ledger.STATUS_DEGRADED,
            "model": "",
            "reason": f"the week story was rejected; the prior story was kept: {exc}",
            "outputs": [],
            "model_attribution": dict(attribution),
        }
    return {
        "status": ledger.STATUS_OK,
        "model": _text(result.get("model")) or model,
        "reason": f"grounded week story written for {week} ({count})",
        "outputs": [str(destination)],
        "model_attribution": dict(attribution),
        "extra": {provider_seam.LEDGER_FIELD: dict(attribution)},
    }


__all__ = [
    "DEFAULT_RESERVE_MINUTES",
    "EVIDENCE_KEYS",
    "MAX_ITEMS_PER_DAY",
    "MIN_NARRATED_DAYS",
    "PROMPT_VERSION",
    "SCHEMA",
    "SCHEMA_NAME",
    "TENDENCY_LIMIT",
    "TIMEOUT_SECONDS",
    "WEEK_NARRATION_JSON_SCHEMA",
    "WEEK_SOURCE_SEPARATOR",
    "WeekNarrationRejected",
    "build_week_inputs",
    "check_week_narration",
    "narration_path",
    "read_week_narration",
    "reserve_minutes",
    "run_week_review_narration",
    "week_id",
    "week_sessions",
    "week_source_id",
]
