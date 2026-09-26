"""Grounded local-AI narration of deterministic Market Journal rollups.

The model sees the latest weekly, monthly and quarterly packs and nothing
else.  A failed call leaves the last verified narration untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

PROMPT_VERSION = "market_story_narration_v1"
SCHEMA = "market_story_narration_v1"

#: How many click options the overnight question may carry (TJ-14B). Four is a
#: card row; a fifth is a list, and a list is not a click.
MENTOR_QUESTION_OPTIONS_MAX = 4

#: The symbols `market_story` measures; the only ones a direction claim is checked for.
_BENCHMARKS = {"SPY", "QQQ", "IWM", "VXX", "TLT", "USO"}

NARRATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "changes", "open_questions", "mentor_question", "sources"],
    "properties": {
        "summary": {"type": "string", "maxLength": 1800},
        "changes": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 400},
        },
        "open_questions": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 400},
        },
        "mentor_question": {"type": "string", "maxLength": 300},
        # TJ-14B. The overnight question becomes a CLICK on tomorrow's Mentor
        # card, so the options are a CLOSED set the card can draw: at most four,
        # each a short string. Optional - a night that offers none still asks a
        # question, in words. `maxLength` is deliberately nowhere near 2,000
        # (gate #144's grammar-compile defect).
        "mentor_question_options": {
            "type": "array",
            "maxItems": MENTOR_QUESTION_OPTIONS_MAX,
            "items": {"type": "string", "maxLength": 60},
        },
        "sources": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 160},
        },
    },
}


def _roots(rollups_dir: Path | None, out_dir: Path | None) -> tuple[Path, Path]:
    if rollups_dir is None or out_dir is None:
        from project_paths import MARKET_STORY_NARRATIONS_DIR, MARKET_STORY_ROLLUPS_DIR

        rollups_dir = Path(rollups_dir or MARKET_STORY_ROLLUPS_DIR)
        out_dir = Path(out_dir or MARKET_STORY_NARRATIONS_DIR)
    return Path(rollups_dir), Path(out_dir)


def _latest_pack(root: Path, kind: str) -> dict[str, Any] | None:
    paths = sorted((root / kind).glob("*.json"))
    for path in reversed(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _source_ids(packs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for kind, pack in packs.items():
        ids.add(f"rollup:{kind}:{pack.get('period_id')}")
        for session in pack.get("sessions") or ():
            if not isinstance(session, Mapping):
                continue
            for entry in session.get("entries") or ():
                if isinstance(entry, Mapping) and str(entry.get("entry_id") or "").strip():
                    ids.add("journal:" + str(entry["entry_id"]).strip())
    return sorted(ids)


def _evidence(
    packs: Mapping[str, Mapping[str, Any]], frame: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    hashed: Any = packs if not frame else {"packs": packs, "regime_frame": frame}
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    source_ids = _source_ids(packs)
    evidence = {
        "package_id": f"market-story:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": (
            "Narrate only these deterministic packs. Do not calculate new statistics or "
            "turn an unmeasured item into a fact. Distinguish the trader's words from measured "
            "market facts. Name changes across covered sessions, keep uncertainty, and ask one "
            "short coaching question that tests an open thesis. Every source must be copied "
            "exactly from allowed_source_ids."
        ),
        "allowed_source_ids": source_ids,
        "rollups": {kind: dict(pack) for kind, pack in packs.items()},
    }
    if frame:
        # S16.4: the desk prints the regime line itself; the model tells the story inside it.
        evidence["instructions"] += (
            " The story is told inside regime_frame, the trader's own regime: the desk "
            "already prints its regime line, the trader's notes and the structure facts "
            "above your summary, so do not restate them; relate the week to that regime."
        )
        evidence["regime_frame"] = dict(frame)
    return evidence


# ---------------------------------------------------------------------------
# S16.4: the regime the story opens with - trader's words, then facts
# ---------------------------------------------------------------------------
#: At most this many of the trader's own notes from the current regime.
REGIME_NOTES_MAX = 5
REGIME_NOTE_CHARS = 200


def _clip(text: Any, limit: int) -> str:
    words = " ".join(str(text or "").split())
    return words if len(words) <= limit else words[: limit - 3].rstrip() + "..."


def _journal_regime_rows() -> list[dict[str, Any]]:
    from journal_store import JournalStore
    from project_paths import JOURNAL_DB_FILE

    return JournalStore(Path(JOURNAL_DB_FILE)).list_structural_regime()


def _regime_table_rows() -> list[dict[str, Any]]:
    import market_regimes
    from project_paths import MARKET_REGIME_TABLE_FILE

    return market_regimes.read_table(MARKET_REGIME_TABLE_FILE)


def _trader_notes(start: str, day: str, root: Path | None) -> list[dict[str, str]]:
    """The trader's observations in the day packs of [start, day], newest first."""
    import day_review_pack

    base = Path(root) if root is not None else day_review_pack.default_root()
    sessions = sorted(
        (path.name for path in (base / "sessions").glob("*") if path.is_dir() and start <= path.name <= day),
        reverse=True,
    )
    notes: list[dict[str, str]] = []
    for session in sessions:
        pack = day_review_pack.read_pack(session, root=base)
        items = day_review_pack.said_items(pack or {}, kind=day_review_pack.KIND_OBSERVATION)
        for item in sorted(items, key=lambda row: str(row.get("at") or ""), reverse=True):
            text = _clip(item.get("text"), REGIME_NOTE_CHARS)
            if text:
                notes.append({"session": session, "text": text, "source_id": str(item.get("source_id") or "")})
            if len(notes) >= REGIME_NOTES_MAX:
                return notes
    return notes


def _structure_facts(row: Mapping[str, Any]) -> list[str]:
    """Plain lines from one regime-table row's structure facts; unknown facts say nothing."""
    symbol = str(row.get("symbol") or "")
    structure = row.get("structure") if isinstance(row.get("structure"), Mapping) else {}
    facts: list[str] = []
    channel = structure.get("daily_channel") if isinstance(structure.get("daily_channel"), Mapping) else {}
    pivot_high = channel.get("last_pivot_high") if isinstance(channel.get("last_pivot_high"), Mapping) else {}
    pivot_low = channel.get("last_pivot_low") if isinstance(channel.get("last_pivot_low"), Mapping) else {}
    if channel.get("label") == "lh_ll" and pivot_high.get("date"):
        facts.append(f"{symbol} daily lower high on {pivot_high['date']} (lower-high / lower-low channel)")
    elif channel.get("label") == "hh_hl" and pivot_low.get("date"):
        facts.append(f"{symbol} daily higher low on {pivot_low['date']} (higher-high / higher-low channel)")
    weekly = structure.get("weekly") if isinstance(structure.get("weekly"), Mapping) else {}
    if weekly.get("status") == "ok":
        facts.append(
            f"{symbol} weekly: {weekly.get('higher_highs')} higher highs and {weekly.get('lower_highs')} "
            f"lower highs in the last {weekly.get('weeks')} weeks"
        )
    atr = structure.get("atr") if isinstance(structure.get("atr"), Mapping) else {}
    if atr.get("status") == "ok" and atr.get("compressed") is True:
        facts.append(f"{symbol} daily range compressed (ATR percentile {atr.get('percentile')})")
    sma = structure.get("sma20") if isinstance(structure.get("sma20"), Mapping) else {}
    if sma.get("status") == "ok":
        facts.append(
            f"{symbol} {abs(float(sma.get('distance_pct') or 0.0)):.2f}% {sma.get('side')} its 20-day, "
            f"20-day slope {float(sma.get('slope_pct') or 0.0):+.2f}% over 5 sessions"
        )
    return facts


def regime_frame(
    session_date: str,
    *,
    journal_rows: Any = None,
    table_rows: Any = None,
    day_review_root: Path | None = None,
) -> dict[str, Any] | None:
    """What the story opens with: the trader's regime and day count, their notes, the facts.

    Facts come from the regime journal and the regime table, never the model.
    None when there is neither a typed regime nor a table row to say.
    """
    import market_regimes
    import structural_regime

    day = str(session_date or "")[:10]
    rows = _journal_regime_rows() if journal_rows is None else list(journal_rows)
    table = _regime_table_rows() if table_rows is None else list(table_rows)
    regime = structural_regime.regime_at(rows, day)
    spy = [
        row for row in table
        if str(row.get("symbol") or "").upper() == market_regimes.PRIMARY
        and str(row.get("session_date") or "")[:10] <= day
    ]
    latest = max(spy, key=lambda row: str(row.get("session_date") or "")) if spy else None
    if regime is None and latest is None:
        return None
    if regime is None:
        line = "Regime: unknown - the trader has not typed one."
        notes: list[dict[str, str]] = []
    else:
        line = f"{regime['label']}, day {regime['day_count']} (since {regime['start_date']})"
        note = str(regime.get("structure_note") or "").strip()
        line += f": {note}" if note else ""
        notes = _trader_notes(str(regime["start_date"]), day, day_review_root)
    return {
        "session_date": day,
        "regime": {
            key: regime.get(key)
            for key in ("regime", "label", "start_date", "day_count", "session_count", "structure_note")
        } if regime else None,
        "regime_line": line,
        "trader_words": notes,
        "facts": _structure_facts(latest) if latest else [],
        "facts_as_of": str(latest.get("session_date") or "") if latest else "",
    }


def story_lines(payload: Mapping[str, Any] | None) -> list[str]:
    """The stored story in reading order: regime line, the trader's words, facts, model prose."""
    body = payload if isinstance(payload, Mapping) else {}
    frame = body.get("opening") if isinstance(body.get("opening"), Mapping) else {}
    lines: list[str] = []
    if frame.get("regime_line"):
        lines.append(str(frame["regime_line"]))
    for note in frame.get("trader_words") or ():
        if isinstance(note, Mapping) and note.get("text"):
            lines.append(f"You, {note.get('session')}: \"{note['text']}\"")
    lines.extend(str(fact) for fact in frame.get("facts") or () if str(fact).strip())
    narration = body.get("narration") if isinstance(body.get("narration"), Mapping) else {}
    if str(narration.get("summary") or "").strip():
        lines.append(str(narration["summary"]).strip())
    return lines


_UP_WORDS = r"up|higher|rose|rising|rises|increased|increasing|increases|gained|gaining|climbed|climbing|rallied|rallying"
_DOWN_WORDS = r"down|lower|fell|falling|falls|decreased|decreasing|decreases|dropped|dropping|declined|declining|slid|sliding"
#: After a ticker: up to three words, then a direction word ("USO decreased", "VXX going up").
_AFTER_SYMBOL = re.compile(
    rf"(?:\W+\w+){{0,3}}?\W+(?:(?P<up>{_UP_WORDS})|(?P<down>{_DOWN_WORDS}))\b",
    re.IGNORECASE,
)
_SYMBOL = re.compile(r"\b[A-Z]{2,5}\b")
_INLINE_SOURCE = re.compile(r"\b(?:journal|rollup):[\w:.-]+")


def _moves(text: str, symbols: set[str]) -> list[tuple[str, int]]:
    """(symbol, +1/-1) for every benchmark the text says went up or down."""
    out = []
    for found in _SYMBOL.finditer(text):
        if found.group(0) not in symbols:
            continue
        move = _AFTER_SYMBOL.match(text, found.end())
        if move is not None:
            out.append((found.group(0), 1 if move.group("up") else -1))
    return out


def _check_directions(narration: Mapping[str, Any], packs: Mapping[str, Mapping[str, Any]]) -> None:
    """A benchmark's up/down claim must match a measured change or the trader's own words."""
    measured: dict[str, set[int]] = {}
    said: dict[str, set[int]] = {}
    for pack in packs.values():
        for session in pack.get("sessions") or ():
            if not isinstance(session, Mapping):
                continue
            for cell in session.get("measured") or ():
                change = cell.get("change_pct") if isinstance(cell, Mapping) else None
                if isinstance(change, (int, float)) and change:
                    measured.setdefault(str(cell.get("symbol") or "").upper(), set()).add(1 if change > 0 else -1)
            for entry in session.get("entries") or ():
                if isinstance(entry, Mapping):
                    for symbol, sign in _moves(str(entry.get("text") or "").upper(), _BENCHMARKS):
                        said.setdefault(symbol, set()).add(sign)
    texts = [str(narration.get("summary") or "")] + [str(item) for item in narration.get("changes") or ()]
    for text in texts:
        for symbol, sign in _moves(text, _BENCHMARKS):
            if sign not in measured.get(symbol, set()) | said.get(symbol, set()):
                word = "rose" if sign > 0 else "fell"
                raise ValueError(f"narration says {symbol} {word}, which no measured bar or journal note says")


def _check_inline_sources(narration: Mapping[str, Any], allowed: set[str]) -> None:
    texts = [str(narration.get("summary") or "")] + [str(item) for item in narration.get("changes") or ()]
    for text in texts:
        for cited in _INLINE_SOURCE.findall(text):
            if cited.rstrip(".") not in allowed:
                raise ValueError(f"narration cited {cited!r} in its text, outside its fact packs")


def _check_mentor_question_options(narration: Mapping[str, Any]) -> None:
    """The click options are validated HERE, not only advertised to the model.

    A schema the local grammar is compiled from is still only a request; an
    output that fails it is rejected WHOLE and the last verified file stays
    (plan.md 12.3). Five options is not a closed set a card row can draw, so the
    whole narration goes - a truncated list would be the desk quietly deciding
    which of the trader's choices to drop.
    """
    options = narration.get("mentor_question_options")
    if options is None:
        return
    if not isinstance(options, (list, tuple)):
        raise ValueError("mentor_question_options is not a list")
    if len(options) > MENTOR_QUESTION_OPTIONS_MAX:
        raise ValueError(
            f"mentor_question_options carried {len(options)} options; "
            f"at most {MENTOR_QUESTION_OPTIONS_MAX} may be offered as clicks"
        )
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("every mentor_question option must be a non-empty string")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_market_story_narration(
    *,
    session_date: str = "",
    now: datetime | None = None,
    rollups_dir: Path | None = None,
    out_dir: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    frame_reader: Callable[[str], Mapping[str, Any] | None] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Write one verified narration; model failure preserves the prior file."""
    root, target = _roots(rollups_dir, out_dir)
    packs = {
        kind: pack
        for kind in ("weekly", "monthly", "quarterly")
        if (pack := _latest_pack(root, kind)) is not None
    }
    if not packs:
        return {
            "status": "skipped",
            "model": "",
            "reason": "no market-story rollups exist yet",
            "outputs": [],
        }
    day = str(session_date or datetime.now().date().isoformat())
    try:
        frame = (frame_reader or regime_frame)(day)
    except Exception:  # noqa: BLE001 - an unreadable regime never costs the story
        frame = None
    evidence = _evidence(packs, frame)
    destination = target / f"{day}.json"
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    if (
        isinstance(existing, Mapping)
        and existing.get("inputs_hash") == evidence["evidence_hash"]
        and existing.get("prompt_version") == PROMPT_VERSION
    ):
        return {
            "status": "ok",
            "model": str(existing.get("model") or ""),
            "reason": "verified narration unchanged",
            "outputs": [str(destination)],
        }

    import ai_summary

    if request is None:
        if not ai_summary.local_provider_enabled():
            return {
                "status": "degraded_no_narrative",
                "model": "",
                "reason": "local AI is not configured; prior narration was kept",
                "outputs": [],
            }
        request = ai_summary.request_ai_summary
    try:
        result = request(
            provider="local",
            model=ai_summary.local_model("medium"),
            api_key="",
            evidence=evidence,
            timeout_seconds=900,
            schema=NARRATION_JSON_SCHEMA,
            schema_name="tradingbot_market_story_narration",
            prompt_version=PROMPT_VERSION,
        )
        narration = result.get("summary") if isinstance(result, Mapping) else None
        if not isinstance(narration, Mapping):
            raise ValueError("local AI returned no narration")
        allowed = set(evidence["allowed_source_ids"])
        cited = [str(item) for item in narration.get("sources") or ()]
        if not cited or any(source not in allowed for source in cited):
            raise ValueError("narration cited a source outside its fact packs")
        _check_inline_sources(narration, allowed)
        _check_directions(narration, packs)
        _check_mentor_question_options(narration)
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        payload = {
            "schema": SCHEMA,
            "session_date": day,
            "generated_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "inputs_hash": evidence["evidence_hash"],
            "periods": {kind: str(pack.get("period_id") or "") for kind, pack in packs.items()},
            "model": str(result.get("model") or ""),
            "prompt_version": PROMPT_VERSION,
            # S16.4: the regime line, the trader's words and the facts, printed before the prose.
            "opening": dict(frame) if frame else None,
            "narration": dict(narration),
        }
        _atomic_write(destination, payload)
    except Exception as exc:  # noqa: BLE001 - prior verified file is the fallback
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"local AI narration failed; prior narration was kept: {exc}",
            "outputs": [],
        }
    return {
        "status": "ok",
        "model": str(result.get("model") or ""),
        "reason": "grounded market-story narration written",
        "outputs": [str(destination)],
    }


def read_story(
    session_date: str = "", *, out_dir: Path | None = None, reads_dir: Path | None = None
) -> dict[str, Any]:
    """The market story as the trader reads it, joined with the night's regime read.

    `lines` is the stored story in order (regime line, the trader's words, facts,
    model prose); `regime_read` is the newest verified regime read on or before
    the story's session (S17.2), or None.
    """
    import market_regimes

    _root, target = _roots(None, out_dir)
    limit = str(session_date or "")[:10]
    story: dict[str, Any] | None = None
    for path in sorted(target.glob("*.json"), reverse=True):
        if limit and path.stem > limit:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("narration"), Mapping):
            story = payload
            break
    on = str((story or {}).get("session_date") or limit)[:10]
    return {
        "story": story,
        "lines": story_lines(story),
        "regime_read": market_regimes.latest_regime_read(on or None, root=reads_dir),
    }


def latest_coaching_question(out_dir: Path | None = None) -> str:
    """The last verified nightly question, for the existing Mentor card."""
    _root, target = _roots(None, out_dir)
    for path in reversed(sorted(target.glob("*.json"))):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        narration = payload.get("narration") if isinstance(payload, Mapping) else None
        if isinstance(narration, Mapping):
            question = str(narration.get("mentor_question") or "").strip()
            if question:
                return question
    return ""
