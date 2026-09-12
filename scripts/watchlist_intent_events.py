"""A watchlist edit is a dated event, never a verdict (WISHLIST 5D).

The four plain watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`,
`shortswings.txt`) are the oldest surface on the desk and the only one that
kept no history at all: a name appeared, a name vanished, and nothing recorded
who did it or when. Every other trader act - a like, a veto, a pass, a Focus
pick - already has a forward record, so the one act the trader performs most
often was the one the evidence loop could not see.

This module is that record, and its restraint is the point:

* **Membership means interest.** It is not a setup claim, not a position, not
  a prediction and not a verdict. A `remove` is not a dislike - the trader's
  dislike has its own store (`pick_feedback.jsonl`) and its own words.
* **Nothing is invented.** `ts` is the OBSERVATION time: when the desk saw the
  change. An edit made in Notepad while the app was closed is recorded at the
  moment the panel next loaded the file, labelled `observed_external`, and
  never back-dated to a time nobody measured.
* **A machine write is distinguishable from a trader write.** The Focus
  store's injection into the shared lists is `machine_inject`; the trader's
  own typing is `trader_edit` and a clipboard drop is `trader_paste`. A reader
  that cannot tell them apart would read the desk's own bookkeeping as the
  trader's conviction.
* **The evidence never costs the save.** Every writer here returns a COUNT and
  swallows its own failure. The trader's list is the product; this is evidence
  about it (ground rule: an evidence store never costs the event it records).
* **Nothing consumes it yet.** No detector, score, alert, Focus list, scanner
  or `review_policy.json` reads this file. 10G's Watchlist tab will show source
  badges from it and the WS-DR work may join on it; until then it is a stream
  the trader can tail.

The baseline row is how a stream that started late stays honest. Reconstructing
membership needs a starting point, and the file's current contents are not one:
they are today's state, not the state when the stream began. So the first time a
list is seen with no reconstructable history, ONE `baseline_recorded` row is
written naming the symbols then present - and no `add` rows, because nobody
observed those additions. It stays small because it is written at most once per
list (never per load), encodes the symbols as one comma-joined string rather
than a list of objects, and carries a count and a short digest so a later reader
can check it cheaply. After that the stream grows with CHANGES, never with
loads.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from project_paths import WATCHLIST_INTENT_EVENTS_FILE

logger = logging.getLogger(__name__)

#: Schema by NAME, never by number - a changed meaning is a new name.
SCHEMA_WATCHLIST_INTENT_EVENT = "watchlist_intent_event_v1"

#: Resolved at CALL time by every writer and reader, so a test (or a future
#: per-machine store) can redirect the stream without reaching into callers.
EVENTS_FILE = WATCHLIST_INTENT_EVENTS_FILE

ACTION_ADD = "add"
ACTION_REMOVE = "remove"
ACTION_BASELINE = "baseline_recorded"
ACTIONS = (ACTION_ADD, ACTION_REMOVE, ACTION_BASELINE)

SOURCE_TRADER_EDIT = "trader_edit"
SOURCE_TRADER_PASTE = "trader_paste"
SOURCE_MACHINE_INJECT = "machine_inject"
SOURCE_MACHINE_UNINJECT = "machine_uninject"
SOURCE_OBSERVED_EXTERNAL = "observed_external"
SOURCES = (
    SOURCE_TRADER_EDIT,
    SOURCE_TRADER_PASTE,
    SOURCE_MACHINE_INJECT,
    SOURCE_MACHINE_UNINJECT,
    SOURCE_OBSERVED_EXTERNAL,
)

#: The four plain watchlists, and what membership on each one means. `side` and
#: `horizon` are derived here rather than passed in, so no caller can label a
#: swing list as a day-trade interest.
LIST_SPECS: dict[str, tuple[str, str]] = {
    "longs": ("long", "day"),
    "shorts": ("short", "day"),
    "swinglongs": ("long", "swing"),
    "shortswings": ("short", "swing"),
}
LISTS = tuple(LIST_SPECS)

#: `FocusPickStore` category -> the shared list a side injects into.
_CATEGORY_LISTS: dict[str, dict[str, str]] = {
    "m5": {"long": "longs", "short": "shorts"},
    "swing": {"long": "swinglongs", "short": "shortswings"},
}


def default_events_path() -> Path:
    """The stream's path, read at call time (never bound at import)."""
    return Path(EVENTS_FILE)


def list_for_path(path: Path | str) -> str:
    """Which of the four lists a file is, or `''` when it is none of them."""
    name = Path(path).name.strip().lower()
    if name.endswith(".txt"):
        name = name[:-4]
    return name if name in LIST_SPECS else ""


def list_for(category: object, side: object) -> str:
    """Which shared list a Focus (category, side) injects into, or `''`."""
    cat = str(category or "").strip().lower()
    sided = str(side or "").strip().lower()
    return _CATEGORY_LISTS.get(cat, {}).get(sided, "")


def normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def _market_tz():
    try:
        from market_calendar import MARKET_TZ

        return MARKET_TZ
    except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.12
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")


def event_time(now: datetime | None = None) -> datetime:
    """The observation instant, aware and market-local.

    A naive `now` is ATTACHED to the machine zone with `astimezone()` and then
    converted - never `replace(tzinfo=...)`, which would relabel a Pacific
    afternoon as an Eastern one and silently move the session it belongs to.
    """
    moment = now or datetime.now()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(_market_tz())


def _symbols_digest(symbols: Sequence[str]) -> str:
    joined = ",".join(sorted(symbols))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def build_row(
    *,
    list_name: str,
    symbol: object,
    action: str,
    source: str,
    writer: str = "",
    reason: str = "",
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """One stream row, or `None` when the list, action or source is unusable."""
    spec = LIST_SPECS.get(str(list_name or "").strip().lower())
    action_text = str(action or "").strip().lower()
    source_text = str(source or "").strip().lower()
    if spec is None or action_text not in ACTIONS or source_text not in SOURCES:
        return None
    sym = normalize_symbol(symbol)
    if action_text in (ACTION_ADD, ACTION_REMOVE) and not sym:
        return None
    side, horizon = spec
    stamp = event_time(now)
    return {
        "schema": SCHEMA_WATCHLIST_INTENT_EVENT,
        "ts": stamp.isoformat(timespec="seconds"),
        "market_date": stamp.date().isoformat(),
        "list": str(list_name).strip().lower(),
        "side": side,
        "horizon": horizon,
        "symbol": sym,
        "action": action_text,
        "source": source_text,
        "reason": str(reason or "").strip(),
        "writer": str(writer or "").strip(),
    }


def append_rows(rows: Iterable[Mapping[str, Any]], path: Path | str | None = None) -> int:
    """Append rows, one complete JSON line each. Returns how many were written.

    Never raises: a failure is logged and costs the evidence, never the edit
    that produced it.
    """
    payload = [dict(row) for row in rows if row]
    if not payload:
        return 0
    target = Path(path) if path is not None else default_events_path()
    try:
        from diagnostics.artifact_io import append_jsonl_rows

        append_jsonl_rows(target, payload)
    except Exception:  # pragma: no cover - exercised by the failure test
        logger.warning("watchlist intent events not recorded (%s rows)", len(payload), exc_info=True)
        return 0
    return len(payload)


def record_changes(
    *,
    list_name: str,
    added: Iterable[object] = (),
    removed: Iterable[object] = (),
    source: str,
    writer: str = "",
    reason: str = "",
    now: datetime | None = None,
    path: Path | str | None = None,
) -> int:
    """Append one row per symbol that joined or left a list. Returns the count.

    Removals are written before adds so a replay of a swap reads the way it
    happened. An unknown list writes nothing rather than guessing a side.
    """
    rows: list[dict[str, Any]] = []
    for symbol in removed or ():
        row = build_row(
            list_name=list_name, symbol=symbol, action=ACTION_REMOVE,
            source=source, writer=writer, reason=reason, now=now,
        )
        if row is not None:
            rows.append(row)
    for symbol in added or ():
        row = build_row(
            list_name=list_name, symbol=symbol, action=ACTION_ADD,
            source=source, writer=writer, reason=reason, now=now,
        )
        if row is not None:
            rows.append(row)
    return append_rows(rows, path)


def record_baseline(
    *,
    list_name: str,
    symbols: Iterable[object],
    writer: str = "",
    now: datetime | None = None,
    path: Path | str | None = None,
) -> int:
    """Record the starting point of a list, once, without inventing any adds."""
    row = build_row(
        list_name=list_name, symbol="", action=ACTION_BASELINE,
        source=SOURCE_OBSERVED_EXTERNAL, writer=writer,
        reason="no earlier observation exists for this list", now=now,
    )
    if row is None:
        return 0
    names = [normalize_symbol(item) for item in (symbols or ()) if normalize_symbol(item)]
    row["symbol_count"] = len(names)
    row["symbols"] = ",".join(names)
    row["symbols_digest"] = _symbols_digest(names)
    return append_rows([row], path)


# --------------------------------------------------------------------------- reading
def _parse_ts(value: object) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value or ""))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=_market_tz())
    return stamp


def read_events(
    list_name: str | None = None,
    *,
    since: datetime | str | None = None,
    path: Path | str | None = None,
    **aliases: Any,
) -> list[dict[str, Any]]:
    """Rows in file order, optionally narrowed to one list and/or a start time.

    `list=` is accepted as a spelling of `list_name` because that is how the
    packet names it; `list` is a builtin, so the parameter itself is not.
    """
    if aliases:
        alias = aliases.pop("list", None)
        if aliases:
            raise TypeError(f"read_events() got unexpected keyword arguments: {sorted(aliases)}")
        if alias is not None and list_name is None:
            list_name = alias

    target = Path(path) if path is not None else default_events_path()
    try:
        from diagnostics.artifact_io import read_jsonl

        rows = read_jsonl(target)
    except Exception:  # pragma: no cover - a torn store is empty, never fatal
        logger.warning("watchlist intent events unreadable at %s", target, exc_info=True)
        return []

    wanted = str(list_name or "").strip().lower()
    floor = since if isinstance(since, datetime) else _parse_ts(since)
    if floor is not None and floor.tzinfo is None:
        floor = floor.astimezone()

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if wanted and str(row.get("list") or "").strip().lower() != wanted:
            continue
        if floor is not None:
            stamp = _parse_ts(row.get("ts"))
            if stamp is None or stamp < floor:
                continue
        out.append(row)
    return out


def reconstruct_membership(
    list_name: str,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    path: Path | str | None = None,
) -> list[str] | None:
    """The membership the STREAM can account for, or `None` when it cannot.

    `None` means "no baseline": the stream has never observed this list, so the
    file's current contents are a starting point, not a diff. Returning an
    empty list there would turn every existing name into an invented `add`.
    """
    source = list(rows) if rows is not None else read_events(list_name, path=path)
    wanted = str(list_name or "").strip().lower()
    start = -1
    for index, row in enumerate(source):
        if str(row.get("list") or "").strip().lower() != wanted:
            continue
        if str(row.get("action") or "") == ACTION_BASELINE:
            start = index
    if start < 0:
        return None

    baseline = source[start]
    names = [
        normalize_symbol(item)
        for item in str(baseline.get("symbols") or "").split(",")
        if normalize_symbol(item)
    ]
    members: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            members.append(name)
    for row in source[start + 1:]:
        if str(row.get("list") or "").strip().lower() != wanted:
            continue
        symbol = normalize_symbol(row.get("symbol"))
        action = str(row.get("action") or "")
        if not symbol:
            continue
        if action == ACTION_ADD and symbol not in seen:
            seen.add(symbol)
            members.append(symbol)
        elif action == ACTION_REMOVE and symbol in seen:
            seen.discard(symbol)
            members = [item for item in members if item != symbol]
    return members


def observe_list(
    *,
    list_name: str,
    symbols: Iterable[object],
    writer: str = "",
    now: datetime | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Reconcile what is on disk with what the stream can account for.

    Called when a panel LOADS a list. A difference is somebody else's edit -
    a text editor, the DAS, another machine - so it is recorded as
    `observed_external` at THIS moment, which is the only time anyone measured.
    A list the stream cannot reconstruct gets one baseline row and no adds.
    """
    result: dict[str, Any] = {"baseline": False, "added": [], "removed": [], "written": 0}
    if str(list_name or "").strip().lower() not in LIST_SPECS:
        return result

    current: list[str] = []
    seen: set[str] = set()
    for item in symbols or ():
        name = normalize_symbol(item)
        if name and name not in seen:
            seen.add(name)
            current.append(name)

    known = reconstruct_membership(list_name, path=path)
    if known is None:
        result["baseline"] = True
        result["written"] = record_baseline(
            list_name=list_name, symbols=current, writer=writer, now=now, path=path
        )
        return result

    known_set = set(known)
    added = [name for name in current if name not in known_set]
    removed = [name for name in known if name not in seen]
    if not added and not removed:
        return result
    result["added"] = added
    result["removed"] = removed
    result["written"] = record_changes(
        list_name=list_name,
        added=added,
        removed=removed,
        source=SOURCE_OBSERVED_EXTERNAL,
        writer=writer,
        reason="difference seen when the list was loaded; the edit itself was not observed",
        now=now,
        path=path,
    )
    return result


# --------------------------------------------------------------------------- CLI
def _format_row(row: Mapping[str, Any]) -> str:
    action = str(row.get("action") or "")
    if action == ACTION_BASELINE:
        body = f"{row.get('symbol_count', 0)} symbols"
    else:
        body = str(row.get("symbol") or "")
    parts = [
        str(row.get("ts") or ""),
        f"{str(row.get('list') or ''):<11}",
        f"{action:<17}",
        f"{body:<10}",
        str(row.get("source") or ""),
    ]
    reason = str(row.get("reason") or "")
    writer = str(row.get("writer") or "")
    if writer:
        parts.append(f"[{writer}]")
    if reason:
        parts.append(f"- {reason}")
    return "  ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="watchlist_intent_events",
        description="Tail the append-only record of watchlist adds and removes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    tail = sub.add_parser("tail", help="print the most recent rows")
    tail.add_argument("--list", dest="list_name", default=None, choices=list(LISTS))
    tail.add_argument("--limit", type=int, default=40)
    tail.add_argument("--since", default=None, help="ISO timestamp; rows at or after it")
    tail.add_argument("--path", default=None, help="stream path (defaults to the shared home)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = read_events(args.list_name, since=args.since, path=args.path)
    if args.limit and args.limit > 0:
        rows = rows[-args.limit:]
    if not rows:
        print("no watchlist intent events recorded yet")
        return 0
    for row in rows:
        print(_format_row(row))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
