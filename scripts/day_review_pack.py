"""The day pack — one session's deterministic evidence, hash-stable (TJ-4 item 1).

`plan.md` §12.4 TJ-4 change 1, as amended 2026-09-19. This module is what the
overnight day story is allowed to read, and nothing else: twelve sections, every
item naming its own `source_id`, and a hash over the INPUTS so the same session
built twice is the same pack.

Three rules hold it together:

* **PURE.** It opens no store, starts no thread, has no clock of its own and
  reads no detector, score or tracker file. Everything it needs is handed in by
  the caller that already read it on a worker (`DayReviewService.build_pack_for`)
  or by the nightly slot. A builder that read its own inputs would be a second
  reader of the day, and the two would disagree on the first rounding decision.
* **A machine row is never in it.** `market_journal.is_machine_entry` is the ONE
  filter (TJ-1 item 2), and the pasted forecast is somebody ELSE's words, so it
  has its own section and never appears under `trader_said`.
* **An observation and a prediction are SEPARATE items.** TJ-14A keeps what the
  trader SAW apart from what they CALLED at the writer; folding them here would
  let the night's story quote a description as a call, which is the one thing
  TJ-4's amendment forbids. An EMPTY observation emits no item at all - an empty
  quote is a source id the model can cite and say nothing about.

Where it is stored: `DAY_REVIEW_DIR/sessions/<date>/pack.json`, beside that
session's index. `day_review_index._prune` deletes every child of a session
folder older than `KEEP_SESSIONS` (40) - including this file - and that is safe
precisely because the pack is REBUILDABLE from durable inputs, which
`tests/test_tj4_day_pack.py::test_two_builds_of_the_same_session_hash_equal_...`
is what proves. Never put anything unrebuildable in that folder.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import project_paths

_log = logging.getLogger(__name__)

#: A session folder's name, exactly. `session_dir` slices to ten characters, so
#: without this `"2026-09-18-extra"` would write to a REAL day's folder and
#: `".."` would write outside `sessions/` altogether (reviewer, 2026-09-20).
_SESSION_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

SCHEMA = "day_review_pack_v1"

#: The twelve sections, in the order `plan.md` TJ-4 change 1 names them.
#: `report_card` (TJ-12) and `mood` (TJ-7) are HOOKS: present and empty, so a
#: reader has one shape whether or not those packets have landed.
SECTIONS: tuple[str, ...] = (
    "trader_said",
    "forecast",
    "environment",
    "measured",
    "internals",
    "walkaway",
    "skill",
    "reads",
    "congruence",
    "trades",
    "report_card",
    "mood",
)

#: The sections that are a LIST of items, each of which names its source.
LIST_SECTIONS: tuple[str, ...] = (
    "trader_said",
    "environment",
    "measured",
    "internals",
    "reads",
    "congruence",
)

#: The walk-away populations, in the order the page shows them.
WALKAWAY_POPULATIONS: tuple[str, ...] = (
    "liked_not_traded",
    "rejected",
    "traded_left_early",
    "claimed_d1",
    "earlier_calls",
)

#: How many walk-away rows the pack carries. A SIZE rule, ordered by how far the
#: name ran after the decision - never a ranking that decides anything.
WALKAWAY_TOP_N = 3

#: The `forecast_brief` fields the story may cite one by one.
FORECAST_FIELDS: tuple[str, ...] = (
    "playbook_bullish",
    "playbook_bearish",
    "bottom_line",
    "turbulence",
)

KIND_OBSERVATION = "observation"
KIND_PREDICTION = "prediction"


# ---------------------------------------------------------------------------
# plain values
# ---------------------------------------------------------------------------
def _plain(value: Any) -> Any:
    """A JSON-safe copy of `value`, deterministic for the same input.

    The pack is hashed and written; a `datetime` or a frozen dataclass in it
    would hash by `repr` in one place and by `str` in another.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        rows = list(value)
        if isinstance(value, (set, frozenset)):
            rows = sorted(rows, key=str)
        return [_plain(item) for item in rows]
    return str(value)


def _text(value: Any) -> str:
    return str(value or "").strip()


class _Minter:
    """Mints a `source_id` that no other item in this pack carries.

    Every id is derived from what it points AT - a read's `read_id`, a symbol,
    a congruence kind - and derived ids can collide: a store that appended the
    same row twice, two congruence lines of one kind, a duplicated entry. A
    collision is not harmless here. `_check_day_narration` resolves a cited id
    against ONE row, so two rows under one id would let a narration quote the
    SECOND row's verdict while claiming the first (reviewer, 2026-09-20).

    So the nth item to ask for an id gets `<base>#n`. Nothing is dropped and
    nothing raises - an evidence store is never allowed to cost the thing it
    records - and the suffix still points at exactly one row.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def mint(self, base: str) -> str:
        text = _text(base) or "item"
        count = self._seen.get(text, 0) + 1
        self._seen[text] = count
        return text if count == 1 else f"{text}#{count}"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


# ---------------------------------------------------------------------------
# the sections
# ---------------------------------------------------------------------------
def _entry_rows(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The trader's own entries, oldest first, with the machine's dropped."""
    import market_journal

    rows: list[dict[str, Any]] = []
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("event_type") or "entry") != "entry":
            continue
        if market_journal.is_machine_entry(entry):
            continue
        if str(entry.get("origin") or "") == market_journal.ORIGIN_EXTERNAL_FORECAST:
            continue
        rows.append(dict(entry))
    rows.sort(key=lambda row: str(row.get("created_at") or ""))
    return rows


def _trader_said(
    entries: Iterable[Mapping[str, Any]], mint: _Minter
) -> list[dict[str, Any]]:
    """One item per observation AND one per prediction (TJ-14A's rule, kept)."""
    import market_journal

    items: list[dict[str, Any]] = []
    for entry in _entry_rows(entries):
        entry_id = _text(entry.get("entry_id"))
        timeframe = _text(entry.get("timeframe")).upper()
        stamp = _text(entry.get("created_at"))
        mentor = entry.get("mentor") if isinstance(entry.get("mentor"), Mapping) else {}
        words = _text((mentor or {}).get("observation")) or _text(entry.get("text"))
        if words:
            items.append({
                "kind": KIND_OBSERVATION,
                "entry_id": entry_id,
                "timeframe": timeframe,
                "at": stamp,
                "text": words,
                "direction": "",
                "horizon": "",
                "source_id": mint.mint(f"said:{entry_id}:{KIND_OBSERVATION}"),
            })
        call = market_journal.prediction_of(entry)
        if call is not None:
            items.append({
                "kind": KIND_PREDICTION,
                "entry_id": entry_id,
                "timeframe": timeframe,
                "at": stamp,
                "text": "",
                "direction": call.direction,
                "horizon": call.horizon,
                "confidence": call.confidence,
                "because": call.because,
                "source_id": mint.mint(f"said:{entry_id}:{KIND_PREDICTION}"),
            })
    return items


def _brief_field(brief: Any, name: str) -> Any:
    if isinstance(brief, Mapping):
        return brief.get(name)
    return getattr(brief, name, None)


def _forecast_section(forecast: Any, mint: _Minter) -> dict[str, Any]:
    """The pasted brief, verbatim, with each parsed field citable on its own.

    `chased_against_news` is judged against what the brief STATED, so the
    bearish playbook has to be a source of its own rather than a sentence buried
    in 4 KB of somebody else's markdown.
    """
    if not isinstance(forecast, Mapping) or not forecast:
        return {}
    entry_id = _text(forecast.get("entry_id"))
    text = str(forecast.get("text") or "")
    if not entry_id and not text.strip():
        return {}
    brief = forecast.get("brief")
    fields: dict[str, Any] = {}
    for name in FORECAST_FIELDS:
        fields[name] = {
            "value": _plain(_brief_field(brief, name)),
            "source_id": mint.mint(f"forecast:{entry_id}:{name}"),
        }
    return {
        "entry_id": entry_id,
        "text": text,
        "created_at": _text(forecast.get("created_at")),
        "source_model": _text(forecast.get("source_model")),
        "fields": fields,
    }


def _environment(
    environment: Iterable[Mapping[str, Any]], d1_label: str, mint: _Minter
) -> list[dict[str, Any]]:
    """The day's regime shifts, whole, then the desk's D1 label for the session."""
    items: list[dict[str, Any]] = []
    for index, row in enumerate(environment or ()):
        if not isinstance(row, Mapping):
            continue
        items.append({
            **_plain(dict(row)),
            "kind": "regime_shift",
            "source_id": mint.mint(f"env:regime_shift:{index}"),
        })
    label = _text(d1_label)
    if label:
        items.append({
            "kind": "d1_label",
            "label": label,
            "source_id": mint.mint("env:d1_label"),
        })
    return items


def _measured(story: Any, mint: _Minter) -> list[dict[str, Any]]:
    """`market_story.build_daily_story`'s own cells. The pack measures nothing."""
    cells = getattr(story, "measured", None)
    if cells is None and isinstance(story, Mapping):
        cells = story.get("measured")
    items: list[dict[str, Any]] = []
    for index, cell in enumerate(cells or ()):
        if not isinstance(cell, Mapping):
            continue
        symbol = _text(cell.get("symbol")) or str(index)
        items.append({**_plain(dict(cell)), "source_id": mint.mint(f"measured:{symbol}")})
    return items


def _internals(marks: Iterable[Mapping[str, Any]], mint: _Minter) -> list[dict[str, Any]]:
    """The open, each Mentor hour and the close, in time order.

    The context travels COMPACTED (`trade_mentor_context.compact_for_ai`), so
    `common.internals` - the one SCALAR that survives the evidence package's
    depth cut - is what the model reads. Measured 2026-09-19: `ai_summary._bounded`
    stops six levels down, exactly where a derived line's inputs sit, so the raw
    v2 block reaches the model as "[nested content omitted]".
    """
    import trade_mentor_context

    items: list[dict[str, Any]] = []
    for mark in marks or ():
        if not isinstance(mark, Mapping):
            continue
        kind = _text(mark.get("kind"))
        at = _text(mark.get("at"))
        items.append({
            "kind": kind,
            "at": at,
            "context": _plain(trade_mentor_context.compact_for_ai(mark.get("context"))),
            "source_id": mint.mint(f"internals:{kind}:{at}"),
        })
    items.sort(key=lambda item: str(item.get("at") or ""))
    return items


def _walkaway(walkaway: Any, mint: _Minter) -> dict[str, Any]:
    """Counts per population, and the three rows that ran furthest after.

    A SIZE rule with an order, never a ranking that decides anything: the story
    needs the names it can talk about, and three is what one paragraph holds.
    """
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for population in WALKAWAY_POPULATIONS:
        found = getattr(walkaway, population, None)
        if found is None and isinstance(walkaway, Mapping):
            found = walkaway.get(population)
        found = tuple(found or ())
        counts[population] = len(found)
        for index, row in enumerate(found):
            plain = _plain(row)
            if not isinstance(plain, Mapping):
                continue
            rows.append({**plain, "population": population, "_order": (population, index)})
    rows.sort(
        key=lambda row: (
            -(_number(row.get("ran_after_pct")) if _number(row.get("ran_after_pct")) is not None else float("-inf")),
            str(row.get("symbol") or ""),
            str(row.get("_order")),
        )
    )
    top: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:WALKAWAY_TOP_N]):
        body = {key: value for key, value in row.items() if key != "_order"}
        body["source_id"] = mint.mint(f"walkaway:{index}:{_text(body.get('symbol'))}")
        top.append(body)
    return {"counts": counts, "top": top}


def _skill(walkaway: Any, session: str, mint: _Minter) -> dict[str, Any]:
    """TJ-11's own skill block, whole. The pack computes no rate of its own."""
    skill = getattr(walkaway, "skill", None)
    if skill is None and isinstance(walkaway, Mapping):
        skill = walkaway.get("skill")
    body = _plain(skill) if isinstance(skill, Mapping) else {}
    if not isinstance(body, dict):
        body = {}
    return {**body, "source_id": mint.mint(f"skill:{session}")}


def _report_card(card: Any, session: str, mint: _Minter) -> dict[str, Any]:
    """TJ-12's six lines, each citable on its own.

    The pack computes nothing here: the card arrives BUILT from
    `day_report_card.build`, on the Day Review worker, and this only mints the
    ids a narration may quote. A caller with no card leaves the hook exactly as
    TJ-4 shipped it - an empty mapping, never an invented card.
    """
    lines = getattr(card, "lines", None)
    if lines is None and isinstance(card, Mapping):
        lines = card.get("lines")
    if not lines:
        return {}
    items: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not isinstance(line, Mapping):
            continue
        key = _text(line.get("key")) or str(index)
        items.append({**_plain(dict(line)), "source_id": mint.mint(f"report_card:{key}")})
    if not items:
        return {}
    stamped = getattr(card, "session", "")
    if not stamped and isinstance(card, Mapping):
        stamped = card.get("session") or ""
    return {"session": _text(stamped) or session, "lines": items}


def _reads(reads: Iterable[Mapping[str, Any]], mint: _Minter) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, row in enumerate(reads or ()):
        if not isinstance(row, Mapping):
            continue
        read_id = _text(row.get("read_id")) or str(index)
        items.append({**_plain(dict(row)), "source_id": mint.mint(f"read:{read_id}")})
    return items


def _congruence(lines: Iterable[Mapping[str, Any]], mint: _Minter) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, line in enumerate(lines or ()):
        if not isinstance(line, Mapping):
            continue
        kind = _text(line.get("kind")) or str(index)
        items.append({**_plain(dict(line)), "source_id": mint.mint(f"congruence:{kind}")})
    return items


def _trades(trades: Iterable[Mapping[str, Any]], mint: _Minter) -> dict[str, Any]:
    """The day's closed money, counted once per row and never recomputed."""
    rows: list[dict[str, Any]] = []
    wins = losses = 0
    net = 0.0
    for index, trade in enumerate(trades or ()):
        if not isinstance(trade, Mapping):
            continue
        trade_id = _text(trade.get("trade_id")) or str(index)
        pnl = _number(trade.get("realized_pnl"))
        if pnl is not None:
            net += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
        rows.append({**_plain(dict(trade)), "source_id": mint.mint(f"trade:{trade_id}")})
    return {
        "n": len(rows),
        "wins": wins,
        "losses": losses,
        "net_pnl": round(net, 6),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# the pack
# ---------------------------------------------------------------------------
def build_pack(
    session_date: str,
    *,
    entries: Iterable[Mapping[str, Any]] = (),
    forecast: Any = None,
    story: Any = None,
    environment: Iterable[Mapping[str, Any]] = (),
    d1_label: str = "",
    internals: Iterable[Mapping[str, Any]] = (),
    walkaway: Any = None,
    reads: Iterable[Mapping[str, Any]] = (),
    congruence: Iterable[Mapping[str, Any]] = (),
    trades: Iterable[Mapping[str, Any]] = (),
    report_card: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One session's evidence, as the night's story is allowed to see it.

    `inputs_hash` is over the SECTIONS and never over the clock: the post-close
    tick and the nightly slot build the same pack hours apart, and a clock in
    the hash would make change 2's skip unreachable and pay the model every
    night for a session that had not moved.
    """
    session = str(session_date or "")[:10]
    moment = now or datetime.now().astimezone()
    # ONE minter for the whole pack: no two items in it may share a source_id,
    # whatever the stores handed in (reviewer, 2026-09-20).
    mint = _Minter()
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "session_date": session,
        "trader_said": _trader_said(entries, mint),
        "forecast": _forecast_section(forecast, mint),
        "environment": _environment(environment, d1_label, mint),
        "measured": _measured(story, mint),
        "internals": _internals(internals, mint),
        "walkaway": _walkaway(walkaway, mint),
        "skill": _skill(walkaway, session, mint),
        "reads": _reads(reads, mint),
        "congruence": _congruence(congruence, mint),
        "trades": _trades(trades, mint),
        # TJ-12's six lines when the caller built them, and TJ-7's hook. Both
        # present and falsy when nobody handed one in, never absent.
        "report_card": _report_card(report_card, session, mint),
        "mood": {},
    }
    body["inputs_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    body["built_at"] = moment.isoformat()
    return body


def allowed_source_ids(pack: Mapping[str, Any]) -> tuple[str, ...]:
    """Every id a narration of this pack may cite, once each, in pack order."""
    seen: list[str] = []

    def _add(value: Any) -> None:
        text = _text(value)
        if text and text not in seen:
            seen.append(text)

    for name in LIST_SECTIONS:
        for item in (pack or {}).get(name) or ():
            if isinstance(item, Mapping):
                _add(item.get("source_id"))
    forecast = (pack or {}).get("forecast")
    if isinstance(forecast, Mapping):
        for cell in (forecast.get("fields") or {}).values():
            if isinstance(cell, Mapping):
                _add(cell.get("source_id"))
    walkaway = (pack or {}).get("walkaway")
    if isinstance(walkaway, Mapping):
        for row in walkaway.get("top") or ():
            if isinstance(row, Mapping):
                _add(row.get("source_id"))
    skill = (pack or {}).get("skill")
    if isinstance(skill, Mapping):
        _add(skill.get("source_id"))
    card = (pack or {}).get("report_card")
    if isinstance(card, Mapping):
        for line in card.get("lines") or ():
            if isinstance(line, Mapping):
                _add(line.get("source_id"))
    trades = (pack or {}).get("trades")
    if isinstance(trades, Mapping):
        for row in trades.get("rows") or ():
            if isinstance(row, Mapping):
                _add(row.get("source_id"))
    return tuple(seen)


def said_items(pack: Mapping[str, Any], *, kind: str = "", timeframe: str = "") -> list[dict[str, Any]]:
    """`trader_said` items, optionally narrowed by kind and timeframe."""
    wanted_kind = _text(kind)
    wanted_timeframe = _text(timeframe).upper()
    out: list[dict[str, Any]] = []
    for item in (pack or {}).get("trader_said") or ():
        if not isinstance(item, Mapping):
            continue
        if wanted_kind and _text(item.get("kind")) != wanted_kind:
            continue
        if wanted_timeframe and _text(item.get("timeframe")).upper() != wanted_timeframe:
            continue
        out.append(dict(item))
    return out


# ---------------------------------------------------------------------------
# where it lives
# ---------------------------------------------------------------------------
def default_root() -> Path:
    """`DAY_REVIEW_DIR`, read at CALL time so a test can redirect it."""
    return Path(project_paths.DAY_REVIEW_DIR)


def session_dir(session_date: str, *, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else default_root()
    return base / "sessions" / str(session_date or "").strip()[:10]


def pack_path(session_date: str, *, root: Path | None = None) -> Path:
    """`<root>/sessions/<date>/pack.json`, beside that session's index.

    `day_review_index._prune` deletes every child of a session folder older than
    its `KEEP_SESSIONS` and removes the folder, so this file goes with it. That
    is safe because the pack is REBUILDABLE from durable inputs and nothing
    unrebuildable may be written here.
    """
    return session_dir(session_date, root=root) / "pack.json"


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_pack(pack: Mapping[str, Any], *, root: Path | None = None) -> Path:
    path = pack_path(str((pack or {}).get("session_date") or ""), root=root)
    _atomic_write(path, pack)
    return path


def read_pack(session_date: str, *, root: Path | None = None) -> dict[str, Any] | None:
    """The stored pack, or `None`. A page and a night both ask for one that may
    not exist yet, and neither may be costed an exception for asking."""
    path = pack_path(session_date, root=root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Redo, queued for tonight
# ---------------------------------------------------------------------------
def redo_path(session_date: str, *, root: Path | None = None) -> Path:
    return session_dir(session_date, root=root) / "redo_requested.json"


def validated_session(session_date: str, *, now: datetime | None = None) -> str:
    """`session_date` as a real, closed exchange session, or RAISE.

    This is the one place a Redo turns a string into a PATH, and it fails
    CLOSED - unlike a reader, which answers `unmeasured`. A write the trader
    asked for is not evidence: it must land where they meant it or not at all.
    Measured by the reviewer on the round-1 build: `request_redo("..")` wrote
    `redo_requested.json` at the `day_review` ROOT, outside `sessions/`, and
    `"2026-09-18-extra"` was silently truncated to a real day.

    A future session has no pack and no story, and a day the exchange never
    opened has neither either, so both are refused rather than queued for ever.
    """
    text = str(session_date or "").strip()
    if not _SESSION_DATE.fullmatch(text):
        raise ValueError(f"{session_date!r} is not a YYYY-MM-DD session date")
    try:
        day = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{session_date!r} is not a calendar date: {exc}") from exc
    try:
        import market_calendar

        if not market_calendar.is_session(day):
            raise ValueError(f"{text} is not a trading session")
        last = market_calendar.last_completed_session(now or datetime.now().astimezone())
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - an unanswerable calendar refuses
        raise ValueError(f"the exchange calendar cannot place {text}: {exc}") from exc
    if day > last:
        raise ValueError(f"{text} has not closed yet; there is nothing to narrate")
    return text


def request_redo(
    session_date: str, *, root: Path | None = None, now: datetime | None = None
) -> Path:
    """Ask the night to narrate this session again (plan TJ-4 change 4).

    Without this the daytime "queued for tonight" would be a lie: the pack's
    hash has not moved, so the night would skip the very session the trader
    asked to have redone.

    The session is VALIDATED first and a bad one raises without writing: see
    :func:`validated_session`.
    """
    session_date = validated_session(session_date, now=now)
    path = redo_path(session_date, root=root)
    moment = now or datetime.now().astimezone()
    _atomic_write(path, {
        "session_date": str(session_date or "")[:10],
        "requested_at": moment.isoformat(),
    })
    return path


def redo_requested(session_date: str, *, root: Path | None = None) -> bool:
    return redo_path(session_date, root=root).exists()


def clear_redo(session_date: str, *, root: Path | None = None) -> bool:
    """Drop the marker after a successful run. Quiet on every failure path."""
    path = redo_path(session_date, root=root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        _log.debug("The Day Review redo marker could not be cleared.", exc_info=True)
        return False
    return True


__all__ = [
    "FORECAST_FIELDS",
    "KIND_OBSERVATION",
    "KIND_PREDICTION",
    "LIST_SECTIONS",
    "SCHEMA",
    "SECTIONS",
    "WALKAWAY_POPULATIONS",
    "allowed_source_ids",
    "build_pack",
    "clear_redo",
    "default_root",
    "pack_path",
    "read_pack",
    "redo_path",
    "redo_requested",
    "request_redo",
    "said_items",
    "validated_session",
    "session_dir",
    "write_pack",
]
