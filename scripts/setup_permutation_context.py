"""Setup permutation keys (WISHLIST P1-4 / 4a): the other-store ``ctx``, read-only.

`setup_permutations.facets_for_row(row, ctx)` reads four facets from stores
other than the scan row. This module builds that ``ctx`` for one session and
stamps the key on a scan's feature rows:

- ``discovery_slot``: the first dated scan report copy of the session that put
  the name in a bucket (`master_avwap_lib.scan_replay`), mapped from its
  exchange-clock checkpoint to the desk slot label.
- ``entry_trigger``: the first ``watch_fired`` review event on the name and
  side that session (`review_events`), as ``kind`` or ``kind_trigger``.
- ``m5_bounce_type``: the first M5 alert on the name and side that session
  (`intraday_bounce_outcomes.csv`), or ``none`` when the log was read and holds
  none.
- ``d1_environment``: the session's label in `d1_environment_store`.

Point in time: each source only holds what happened up to the moment it is
read, and every lookup is keyed to the row's own session. A source that cannot
be read gives nothing, and the facet reads ``unknown``. Nothing here writes a
store; the only write is onto the caller's own feature-row dicts.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import setup_permutations as sp

#: scan_replay checkpoint (exchange clock) -> desk slot label the facet knows.
CHECKPOINT_SLOTS = {"open": "0730", "midday": "1000", "final hour": "1245", "close": "close"}

#: What ``entry_trigger`` reads when the review log was read and nothing fired.
NO_ENTRY_TRIGGER = "no_trigger"

#: Bytes read per step when the M5 log is read backwards from its end.
_TAIL_CHUNK = 1 << 20


def _session_text(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "").strip()[:10]


def _side_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"LONG", "SHORT"} else ""


def covered(session: Any, coverage_start: str | None) -> bool:
    """True when ``session`` is on or after the source's first logged session.

    Before a source's first event the source did not exist yet, so "nothing
    logged" there is unknown - never a confirmed "no trigger" / "no M5 alert".
    """
    start = _session_text(coverage_start)
    return bool(start) and _session_text(session) >= start


def watch_fired_coverage_start(events: Iterable[Mapping[str, Any]] | None) -> str | None:
    """The first session with a logged ``watch_fired``, or None when there is none."""
    days = [
        _session_text(event.get("trade_date"))
        for event in events or ()
        if isinstance(event, Mapping) and str(event.get("action") or "") == "watch_fired"
        and _session_text(event.get("trade_date"))
    ]
    return min(days) if days else None


# --- discovery slot


def discovery_slots(snapshots: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """``{(SYMBOL, SIDE): slot}`` from one session's snapshots, oldest first."""
    ordered = sorted(
        (snap for snap in snapshots if isinstance(snap, Mapping)),
        key=lambda snap: str(snap.get("_exchange") or snap.get("finished_at") or ""),
    )
    out: dict[tuple[str, str], str] = {}
    for snap in ordered:
        slot = CHECKPOINT_SLOTS.get(str(snap.get("_checkpoint") or ""))
        if not slot:
            continue
        for row in snap.get("rows") or []:
            if not isinstance(row, Mapping) or not str(row.get("bucket") or "").strip():
                continue
            key = (str(row.get("symbol") or "").strip().upper(), _side_text(row.get("side")))
            if key[0] and key[1]:
                out.setdefault(key, slot)
    return out


def load_discovery_slots(session: Any, *, reports_dir: Path | None = None) -> dict[tuple[str, str], str]:
    from master_avwap_lib import scan_replay

    try:
        day = date.fromisoformat(_session_text(session))
    except ValueError:
        return {}
    return discovery_slots(scan_replay.load_snapshots(day, reports_dir=reports_dir))


# --- entry trigger


def entry_triggers(events: Iterable[Mapping[str, Any]], session: Any) -> dict[tuple[str, str], str]:
    """``{(SYMBOL, SIDE): trigger}``: the first ``watch_fired`` of the session per name and side."""
    wanted = _session_text(session)
    fired = []
    for event in events or ():
        if not isinstance(event, Mapping) or str(event.get("action") or "") != "watch_fired":
            continue
        if _session_text(event.get("trade_date")) != wanted:
            continue
        detail = event.get("detail") if isinstance(event.get("detail"), Mapping) else {}
        kind = str(detail.get("kind") or event.get("chart_watch_kind") or "").strip().lower()
        if not kind:
            continue
        trigger = str(detail.get("trigger") or "").strip().lower()
        fired.append((str(event.get("ts") or ""), event, f"{kind}_{trigger}" if trigger else kind))
    out: dict[tuple[str, str], str] = {}
    for _ts, event, value in sorted(fired, key=lambda item: item[0]):
        key = (str(event.get("symbol") or "").strip().upper(), _side_text(event.get("side")))
        if key[0] and key[1]:
            out.setdefault(key, value)
    return out


def entry_trigger_checkpoints(events: Iterable[Mapping[str, Any]], session: Any) -> dict[tuple[str, str], str]:
    """``{(SYMBOL, SIDE): checkpoint}``: the exchange-clock window of the session's first ``watch_fired``."""
    from datetime import datetime as _datetime

    from master_avwap_lib import scan_replay

    wanted = _session_text(session)
    stamped = []
    for event in events or ():
        if not isinstance(event, Mapping) or str(event.get("action") or "") != "watch_fired":
            continue
        if _session_text(event.get("trade_date")) != wanted:
            continue
        try:
            moment = _datetime.fromisoformat(str(event.get("ts") or ""))
        except ValueError:
            continue
        stamped.append((str(event.get("ts") or ""), event, moment))
    out: dict[tuple[str, str], str] = {}
    for _ts, event, moment in sorted(stamped, key=lambda item: item[0]):
        key = (str(event.get("symbol") or "").strip().upper(), _side_text(event.get("side")))
        if key[0] and key[1] and key not in out:
            out[key] = scan_replay._checkpoint_for(scan_replay._exchange_time(moment))
    return out


def load_entry_triggers(session: Any, *, path: Path | None = None) -> dict[tuple[str, str], str] | None:
    """The session's triggers, or None when the review log could not be read."""
    events = load_review_events(path=path)
    return None if events is None else entry_triggers(events, session)


def load_review_events(*, path: Path | None = None) -> list[dict] | None:
    """Every review event, or None when the log could not be read."""
    import review_events

    try:
        return review_events.load_review_events(path) if path else review_events.load_review_events()
    except Exception:  # noqa: BLE001 - an unreadable log is "unknown", never a crash
        logging.debug("setup permutations: review events unreadable", exc_info=True)
        return None


def load_session_watch_fired(session: Any, *, path: Path | None = None) -> list[dict] | None:
    """The session's ``watch_fired`` events, streamed; None before the log's first one.

    Same answer as `load_review_events` + `watch_fired_coverage_start` for one
    session, without holding every review event (or filling that module's
    whole-log cache) in memory: only ``watch_fired`` lines are parsed.
    """
    import review_events

    wanted = _session_text(session)
    sources = review_events.review_event_sources(path) if path else review_events.review_event_sources()
    first_day: str | None = None
    kept: list[dict] = []
    seen: set[str] = set()
    for source in sources:
        try:
            handle = Path(source).open("rb")
        except OSError:
            continue
        with handle:
            for raw in handle:
                if b"watch_fired" not in raw:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(event, dict) or str(event.get("action") or "") != "watch_fired":
                    continue
                record_id = str(event.get("review_record_id") or "").strip()
                if record_id:
                    if record_id in seen:
                        continue
                    seen.add(record_id)
                day = _session_text(event.get("trade_date"))
                if day and (first_day is None or day < first_day):
                    first_day = day
                if day == wanted:
                    kept.append(event)
    return kept if covered(wanted, first_day) else None


# --- M5 confirmation


def m5_bounce_types(rows: Iterable[Mapping[str, Any]], session: Any) -> dict[tuple[str, str], str]:
    """``{(SYMBOL, SIDE): bounce_type}``: the first M5 alert of the session per name and side."""
    from setup_scoreboard import bounce_type_from_event_id

    wanted = _session_text(session)
    found = []
    for row in rows or ():
        if _session_text(row.get("trade_date")) != wanted:
            continue
        bounce = bounce_type_from_event_id(str(row.get("event_id") or ""))
        side = _side_text(row.get("direction"))
        symbol = str(row.get("symbol") or "").strip().upper()
        if bounce and side and symbol:
            found.append((str(row.get("entry_time") or ""), (symbol, side), bounce))
    out: dict[tuple[str, str], str] = {}
    for _entry, key, bounce in sorted(found, key=lambda item: item[0]):
        out.setdefault(key, bounce)
    return out


class TailReadCapped(RuntimeError):
    """The backwards read passed its byte cap before it left the session behind."""


def _tail_lines_for_session(
    path: Path,
    session: str,
    *,
    date_columns: tuple[str, ...] = ("trade_date",),
    stamp_columns: tuple[str, ...] = (),
    max_bytes: int | None = None,
) -> tuple[list[str], list[bytes]]:
    """``(fieldnames, raw lines of the session in file order)``, read backwards from the end.

    A row's date is its first non-blank ``date_columns`` value; its stamp is the
    first non-blank ``stamp_columns`` value (the moment the row was written).
    Reading stops after one whole chunk holds no row dated ``session`` or later
    AND no row stamped on or after ``session``'s calendar day. The stamp half is
    what makes an out-of-order append safe: a scan run on day D can write rows
    dated before D, so a block of older dates is not the end of ``session``, but
    rows written before ``session``'s day cannot be dated ``session``. Only the
    session's own lines are kept. ``max_bytes`` raises `TailReadCapped`.
    """
    with path.open("rb") as handle:
        header = handle.readline()
        fieldnames = next(csv.reader(io.StringIO(header.decode("utf-8-sig", errors="replace"))), [])
        date_indexes = [fieldnames.index(column) for column in date_columns if column in fieldnames]
        stamp_indexes = [fieldnames.index(column) for column in stamp_columns if column in fieldnames]
        if not date_indexes:
            return fieldnames, []

        def first(values: list[str], indexes: list[int]) -> str:
            for index in indexes:
                text = values[index].strip() if len(values) > index else ""
                if text:
                    return text
            return ""

        handle.seek(0, 2)
        end = handle.tell()
        start_of_body = len(header)
        position = end
        carry = b""
        chunks: list[list[bytes]] = []
        while position > start_of_body:
            step = min(_TAIL_CHUNK, position - start_of_body)
            if max_bytes is not None and end - (position - step) > max_bytes:
                raise TailReadCapped(f"{path.name}: more than {max_bytes} bytes back to {session}")
            position -= step
            handle.seek(position)
            block = handle.read(step) + carry
            parts = block.split(b"\n")
            carry = parts[0] if position > start_of_body else b""
            chunk_lines = parts[1:] if position > start_of_body else parts
            recent = False
            kept: list[bytes] = []
            for raw in chunk_lines:
                text = raw.decode("utf-8", errors="replace")
                if not text.strip():
                    continue
                values = next(csv.reader(io.StringIO(text)), [])
                day = first(values, date_indexes)
                if day >= session or (stamp_indexes and first(values, stamp_indexes)[:10] >= session):
                    recent = True
                if _session_text(day) == session:
                    kept.append(raw)
            chunks.append(kept)
            if not recent:
                break
    return fieldnames, [raw for kept in reversed(chunks) for raw in kept]


def _tail_rows_for_session(
    path: Path,
    session: str,
    *,
    date_columns: tuple[str, ...] = ("trade_date",),
    stamp_columns: tuple[str, ...] = (),
    max_bytes: int | None = None,
) -> list[dict]:
    """The log's rows for ``session`` as dicts, in file order (see `_tail_lines_for_session`)."""
    fieldnames, lines = _tail_lines_for_session(
        path, session, date_columns=date_columns, stamp_columns=stamp_columns, max_bytes=max_bytes
    )
    body = b"\n".join(lines).decode("utf-8", errors="replace")
    return [dict(zip(fieldnames, values, strict=False)) for values in csv.reader(io.StringIO(body))]


_M5_COLUMNS = ("trade_date", "event_id", "direction", "symbol", "entry_time")


def load_m5_bounce_types(session: Any, *, path: Path | None = None) -> dict[tuple[str, str], str] | None:
    """The session's first M5 alert per name and side, or None when the log could not be read."""
    import project_paths

    target = Path(path or project_paths.INTRADAY_BOUNCE_OUTCOMES_FILE)
    wanted = _session_text(session)
    try:
        if not target.is_file():
            return None
        if not covered(wanted, m5_coverage_start(target)):
            return None
        fieldnames, lines = _tail_lines_for_session(target, wanted)
        # Only the five columns the answer needs, one line at a time (the log's context_json is large).
        wanted_columns = [(name, fieldnames.index(name)) for name in _M5_COLUMNS if name in fieldnames]
        rows = []
        for raw in lines:
            values = next(csv.reader(io.StringIO(raw.decode("utf-8", errors="replace"))), [])
            rows.append({name: values[index] if index < len(values) else "" for name, index in wanted_columns})
    except (OSError, csv.Error):
        logging.debug("setup permutations: M5 outcome log unreadable", exc_info=True)
        return None
    return m5_bounce_types(rows, wanted)


def m5_coverage_start(path: Path) -> str | None:
    """The first logged ``trade_date`` of the (append-ordered) M5 log, or None when it has no row."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            day = _session_text(row.get("trade_date"))
            if day:
                return day
    return None


# --- D1 environment


def load_d1_environment(session: Any, *, path: Any = None) -> str:
    import d1_environment_store

    try:
        return d1_environment_store.label_for_session(_session_text(session), path=path)
    except Exception:  # noqa: BLE001 - an unreadable store is "unknown"
        return sp.UNKNOWN


# --- S15: market caps (the universe builder's weekly yfinance cache, read-only)

#: A cache older than this at scan time gives no caps (unknown), never a stale bucket.
MARKET_CAP_MAX_AGE_DAYS = 30


def market_cap_cache_path() -> Path:
    """The same file as `universe_builder.MARKET_CAP_CACHE`."""
    import project_paths

    return Path(project_paths.CACHE_DIR) / "universe" / "market_caps.json"


def load_market_caps(*, path: Path | None = None, now: datetime | None = None) -> dict[str, float]:
    """Symbol -> market cap ($M) from the cache; empty when it is missing, unreadable or too old."""
    try:
        payload = json.loads(Path(path or market_cap_cache_path()).read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(str(payload.get("fetched_at")))
    except Exception:  # noqa: BLE001 - an unreadable cache is "unknown"
        return {}
    reference = now or datetime.now()
    if fetched > reference or reference - fetched > timedelta(days=MARKET_CAP_MAX_AGE_DAYS):
        return {}
    caps = payload.get("caps") if isinstance(payload, dict) else None
    if not isinstance(caps, dict):
        return {}
    out: dict[str, float] = {}
    for symbol, value in caps.items():
        try:
            cap = float(value)
        except (TypeError, ValueError):
            continue
        if cap > 0:
            out[str(symbol).strip().upper()] = cap
    return out


# --- ctx and stamping


class SessionContext:
    """The four lookups for one session; ``ctx_for`` gives one row's ``ctx``."""

    def __init__(
        self,
        *,
        slots: Mapping[tuple[str, str], str] | None = None,
        triggers: Mapping[tuple[str, str], str] | None = None,
        m5: Mapping[tuple[str, str], str] | None = None,
        environment: str = sp.UNKNOWN,
        trigger_times: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self.slots = slots
        self.triggers = triggers
        self.m5 = m5
        self.environment = environment or sp.UNKNOWN
        self.trigger_times = trigger_times

    @classmethod
    def load(cls, session: Any, *, stream_review_events: bool = False, **paths: Any) -> "SessionContext":
        """``stream_review_events`` reads only the session's watch_fired lines (same answer, less memory)."""
        if stream_review_events:
            events = load_session_watch_fired(session, path=paths.get("review_events_path"))
        else:
            events = load_review_events(path=paths.get("review_events_path"))
            # Before the first logged watch_fired the log could not have held one: unknown, not "none".
            if events is not None and not covered(session, watch_fired_coverage_start(events)):
                events = None
        return cls(
            slots=load_discovery_slots(session, reports_dir=paths.get("reports_dir")),
            triggers=entry_triggers(events, session) if events is not None else None,
            trigger_times=entry_trigger_checkpoints(events, session) if events is not None else None,
            m5=load_m5_bounce_types(session, path=paths.get("m5_outcomes_path")),
            environment=load_d1_environment(session, path=paths.get("environment_path")),
        )

    def ctx_for(self, symbol: Any, side: Any) -> dict[str, str]:
        key = (str(symbol or "").strip().upper(), _side_text(side))
        ctx: dict[str, str] = {"d1_environment": self.environment}
        if self.slots is not None and key in self.slots:
            ctx["discovery_slot"] = self.slots[key]
        if self.triggers is not None:
            ctx["entry_trigger"] = self.triggers.get(key, NO_ENTRY_TRIGGER)
        if self.trigger_times is not None and key in self.trigger_times:
            ctx["entry_trigger_checkpoint"] = self.trigger_times[key]
        if self.m5 is not None:
            ctx["m5_bounce_type"] = self.m5.get(key, "none")
        return ctx


def stamp_scan_rows(
    feature_rows: Iterable[MutableMapping[str, Any]],
    *,
    session: Any,
    context: SessionContext | None = None,
) -> int:
    """Write the permutation stamp onto each feature row in place. Returns rows stamped.

    Shadow only: adds the stamp columns and ``perm_weekly_ema8_hold_weeks`` (the
    scan's own streak under a name no legacy reader knows) and nothing else. Any
    failure leaves the rows unstamped and the scan untouched.
    """
    rows = [row for row in feature_rows or () if isinstance(row, MutableMapping)]
    if not rows:
        return 0
    try:
        lookups = context if context is not None else SessionContext.load(session)
    except Exception:  # noqa: BLE001 - no ctx is unknown facets, never a failed scan
        logging.warning("setup permutations: session context unavailable", exc_info=True)
        lookups = SessionContext()
    stamped = 0
    for row in rows:
        if sp.WEEKLY_STREAK_COLUMN not in row and "weekly_ema8_hold_weeks" in row:
            row[sp.WEEKLY_STREAK_COLUMN] = row.get("weekly_ema8_hold_weeks")
        view = dict(row)
        view.setdefault("run_date", _session_text(session))
        try:
            fields = sp.stamp_fields(view, lookups.ctx_for(row.get("symbol"), row.get("side")))
        except Exception:  # noqa: BLE001 - one bad row loses its stamp, never the row
            logging.debug("setup permutations: stamp failed for %s", row.get("symbol"), exc_info=True)
            continue
        row.update(fields)
        stamped += 1
    return stamped


__all__ = [
    "CHECKPOINT_SLOTS",
    "covered",
    "m5_coverage_start",
    "watch_fired_coverage_start",
    "NO_ENTRY_TRIGGER",
    "SessionContext",
    "discovery_slots",
    "entry_trigger_checkpoints",
    "entry_triggers",
    "load_d1_environment",
    "load_discovery_slots",
    "load_entry_triggers",
    "load_m5_bounce_types",
    "m5_bounce_types",
    "stamp_scan_rows",
]
