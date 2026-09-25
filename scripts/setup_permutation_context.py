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
import logging
from datetime import date
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


def _tail_rows_for_session(path: Path, session: str) -> list[dict]:
    """The log's rows for ``session``, read backwards from the end (the log is append-ordered).

    Stops after one whole chunk holds no row of the session or later, so a
    582 MB log costs a few MB per scan.
    """
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="replace")
        fieldnames = next(csv.reader(io.StringIO(header)), [])
        date_index = fieldnames.index("trade_date") if "trade_date" in fieldnames else -1
        if date_index < 0:
            return []
        handle.seek(0, 2)
        end = handle.tell()
        start_of_body = len(header.encode("utf-8"))
        position = end
        carry = b""
        lines: list[bytes] = []
        while position > start_of_body:
            step = min(_TAIL_CHUNK, position - start_of_body)
            position -= step
            handle.seek(position)
            block = handle.read(step) + carry
            parts = block.split(b"\n")
            carry = parts[0] if position > start_of_body else b""
            chunk_lines = parts[1:] if position > start_of_body else parts
            recent = False
            for raw in chunk_lines:
                text = raw.decode("utf-8", errors="replace")
                if not text.strip():
                    continue
                reader_row = next(csv.reader(io.StringIO(text)), [])
                trade_date = reader_row[date_index] if len(reader_row) > date_index else ""
                if trade_date >= session:
                    recent = True
                lines.append(raw)
            if not recent:
                break
    body = b"\n".join(lines).decode("utf-8", errors="replace")
    return [
        row
        for row in csv.DictReader(io.StringIO(body), fieldnames=fieldnames)
        if _session_text(row.get("trade_date")) == session
    ]


def load_m5_bounce_types(session: Any, *, path: Path | None = None) -> dict[tuple[str, str], str] | None:
    """The session's first M5 alert per name and side, or None when the log could not be read."""
    import project_paths

    target = Path(path or project_paths.INTRADAY_BOUNCE_OUTCOMES_FILE)
    wanted = _session_text(session)
    try:
        if not target.is_file():
            return None
        rows = _tail_rows_for_session(target, wanted)
    except (OSError, csv.Error):
        logging.debug("setup permutations: M5 outcome log unreadable", exc_info=True)
        return None
    return m5_bounce_types(rows, wanted)


# --- D1 environment


def load_d1_environment(session: Any, *, path: Any = None) -> str:
    import d1_environment_store

    try:
        return d1_environment_store.label_for_session(_session_text(session), path=path)
    except Exception:  # noqa: BLE001 - an unreadable store is "unknown"
        return sp.UNKNOWN


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
    def load(cls, session: Any, **paths: Any) -> "SessionContext":
        events = load_review_events(path=paths.get("review_events_path"))
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

    Shadow only: adds the three stamp columns and nothing else. Any failure
    leaves the rows unstamped and the scan untouched.
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
