"""Finish a capture sidecar after the close, so the intraday grade is reachable.

Phase 0.13 packet P9, item 4.

**THE PROBLEM THIS EXISTS FOR, MEASURED.** `pass_cohort`'s intraday grade returns
blank on every live pass, with the reason `sidecar_ends_before_the_entry_bar`.
That is not a bug in the grade - it is the shape of the evidence. The sidecar is
written from the bars the desk was ALREADY HOLDING at the moment of the click, so
every bar in it starts BEFORE the click, and the entry bar the rule asks for -
the first completed M5 close AFTER the click - is by construction never inside
it. Gate 34 recorded this as an open definition question: should entry instead be
the last completed close AT the click?

**It does not have to be.** After the session closes, the rest of that session's
bars exist; they were simply not in the desk's hands when the trader pressed the
key. Completing the sidecar overnight makes "the first completed close after the
click" a real bar, and the definition the packet declared stays the definition.

**THE ORIGINAL SNAPSHOT IS NEVER REWRITTEN.** The completed bars go to a NEW file
(`<event_id>.completed.json`) and a NEW field (`m5_bars_completed_ref`) points at
it. The row's original `m5_bars_ref` still names exactly what the desk was
holding at the click, which is a fact about that moment and is not ours to edit -
and a reader comparing the two can see precisely how much of the session the
trader could actually see.

**Sources, in order, and what each refusal means.** The research lake first,
narrowed Arrow-side by symbol and interval range through `ResearchStore.read_rows`
- never a materialised list, which is BD-74's rule. Then the desk's own bar cache,
for a session the lake has not ingested yet. If neither can answer, the row is
left UNCOMPLETED with a stated reason: an unfinished sidecar is a gap, and a
sidecar padded from nowhere would be worse than a blank grade.

Fail-open throughout. No research store configured, an unreachable share, a
missing cache - each records its reason and moves to the next row. This runs in a
nightly slot behind the grading it feeds, and it may never cost that grading.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from project_paths import TRADER_ANNOTATIONS_FILE

from ui.annotations.pass_bars import (
    SIDECAR_DIRNAME,
    SIDECAR_SCHEMA_VERSION,
    read_pass_bars,
    sidecar_path,
)

_log = logging.getLogger(__name__)

#: The field a completed sidecar is referenced by. Separate from `m5_bars_ref`
#: on purpose: the original reference must keep meaning "what the desk held at
#: the click".
COMPLETED_REF_FIELD = "m5_bars_completed_ref"
COMPLETED_AT_FIELD = "sidecar_completed_at"
COMPLETED_SOURCE_FIELD = "sidecar_completed_source"

#: Why a sidecar was NOT completed. Every one is a real, different absence.
REASON_NO_SIDECAR = "no_sidecar_to_complete"
REASON_ALREADY_COMPLETE = "already_reaches_the_session_close"
REASON_ALREADY_COMPLETED = "already_completed"
REASON_NO_STORE = "research_store_not_configured"
REASON_STORE_UNREACHABLE = "research_store_unreachable"
REASON_NO_BARS_ANYWHERE = "no_bars_in_the_lake_or_the_cache"

SOURCE_LAKE = "research_lake"
SOURCE_CACHE = "desk_bar_cache"

#: RTH close, market-local. A bar starting at or after this is the next session.
_SESSION_CLOSE_HOUR = 16
_SESSION_CLOSE_MINUTE = 0


def completed_sidecar_path(event_id: str, annotations_path: Any = None) -> Path:
    """`<event_id>.completed.json`, beside the snapshot it completes."""
    base = sidecar_path(event_id, annotations_path)
    return base.with_name(base.name.replace(".json", ".completed.json"))


def read_completed_bars(row: Mapping[str, Any], *, annotations_path: Any = None) -> dict:
    """The completed sidecar if there is one, else the original snapshot.

    ONE reader for both, so a grader never has to remember which file to open -
    remembering is what produces two graders that disagree.
    """
    ref = str(row.get(COMPLETED_REF_FIELD) or "").strip()
    if not ref:
        return read_pass_bars(row, annotations_path=annotations_path)
    base = Path(annotations_path or TRADER_ANNOTATIONS_FILE).parent
    try:
        return json.loads((base / ref).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A completed file that cannot be read falls back to the snapshot that
        # certainly can. Never an exception on a grading path.
        return read_pass_bars(row, annotations_path=annotations_path)


def _bar_moment(bar: Mapping[str, Any]) -> datetime | None:
    raw = bar.get("dt") or bar.get("interval_start")
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _session_close(moment: datetime) -> datetime:
    return moment.replace(
        hour=_SESSION_CLOSE_HOUR, minute=_SESSION_CLOSE_MINUTE,
        second=0, microsecond=0,
    )


def _normalise(bar: Mapping[str, Any]) -> dict[str, Any]:
    """One bar in the sidecar's own shape, whatever shape it arrived in."""
    moment = _bar_moment(bar)
    return {
        "dt": moment.isoformat() if moment else "",
        "open": bar.get("open"),
        "high": bar.get("high"),
        "low": bar.get("low"),
        "close": bar.get("close"),
        "volume": bar.get("volume"),
    }


def _lake_bars(symbol: str, start: datetime, end: datetime) -> tuple[list[dict], str]:
    """Completed M5 bars for one symbol between two moments, from the lake.

    Narrowed ARROW-SIDE by symbol and interval range (BD-74): a month-keyed
    partition materialised and then filtered is what put 10 GB in the desk.
    """
    try:
        from research_warehouse.store import ResearchStore
    except Exception:  # noqa: BLE001 - the warehouse is optional
        return [], REASON_NO_STORE
    try:
        store = ResearchStore.open()
    except Exception:  # noqa: BLE001
        return [], REASON_STORE_UNREACHABLE
    if store is None:
        return [], REASON_NO_STORE
    try:
        rows = store.read_rows(
            "bar_m5",
            columns=["symbol", "interval_start", "open", "high", "low", "close", "volume"],
            symbols=[symbol],
            interval_start_range=(start, end),
        )
    except Exception:  # noqa: BLE001 - an unreachable share is a reason, not a crash
        return [], REASON_STORE_UNREACHABLE
    return [dict(row) for row in rows], ""


def complete_sidecar(
    row: Mapping[str, Any],
    *,
    annotations_path: Any = None,
    lake_reader=_lake_bars,
    cache_reader=None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Complete ONE row's sidecar. Returns the fields to merge, or a reason.

    Idempotent: a row that already carries `m5_bars_completed_ref` is left
    exactly alone, so a second night adds nothing and rewrites nothing.
    """
    if str(row.get(COMPLETED_REF_FIELD) or "").strip():
        return {"completed": False, "reason": REASON_ALREADY_COMPLETED}

    snapshot = read_pass_bars(row, annotations_path=annotations_path)
    bars = list(snapshot.get("bars") or ())
    if not bars:
        return {"completed": False, "reason": REASON_NO_SIDECAR}

    last = _bar_moment(bars[-1])
    if last is None:
        return {"completed": False, "reason": REASON_NO_SIDECAR}
    close = _session_close(last)
    if last + timedelta(minutes=5) >= close:
        return {"completed": False, "reason": REASON_ALREADY_COMPLETE}

    symbol = str(snapshot.get("symbol") or row.get("symbol") or "").strip().upper()
    start = last + timedelta(minutes=5)
    fetched, reason = lake_reader(symbol, start, close)
    source = SOURCE_LAKE
    if not fetched and cache_reader is not None:
        # The lake has not ingested this session yet - which is the NORMAL case
        # the morning after, since the warehouse build runs on its own cadence.
        try:
            fetched = list(cache_reader(symbol, start, close) or ())
        except Exception:  # noqa: BLE001
            fetched = []
        if fetched:
            source, reason = SOURCE_CACHE, ""
    if not fetched:
        return {"completed": False, "reason": reason or REASON_NO_BARS_ANYWHERE}

    extra = []
    for bar in fetched:
        moment = _bar_moment(bar)
        if moment is None or moment < start or moment >= close:
            continue
        extra.append(_normalise(bar))
    if not extra:
        return {"completed": False, "reason": reason or REASON_NO_BARS_ANYWHERE}

    event_id = str(row.get("event_id") or "").strip()
    try:
        target = completed_sidecar_path(event_id, annotations_path)
    except ValueError:
        return {"completed": False, "reason": REASON_NO_SIDECAR}

    payload = {
        **{key: value for key, value in snapshot.items() if key != "bars"},
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "completes_ref": str(row.get("m5_bars_ref") or ""),
        "completed_source": source,
        "completed_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "bar_count": len(bars) + len(extra),
        "bars": [_normalise(bar) for bar in bars] + extra,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")
        import os

        os.replace(tmp, target)
    except OSError:
        _log.debug("Sidecar completion write failed for %s.", event_id, exc_info=True)
        return {"completed": False, "reason": "write_failed"}

    return {
        "completed": True,
        "added_bars": len(extra),
        COMPLETED_REF_FIELD: f"{SIDECAR_DIRNAME}/{target.name}",
        COMPLETED_AT_FIELD: payload["completed_at"],
        COMPLETED_SOURCE_FIELD: source,
    }


def complete_sidecars(
    rows: Iterable[Mapping[str, Any]],
    *,
    annotations_path: Any = None,
    lake_reader=_lake_bars,
    cache_reader=None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Complete every eligible row. Never raises; every refusal is counted.

    The annotation LOG is not rewritten - it is append-only. What this returns
    is the mapping a reader applies: `event_id -> the completion fields`, which
    the cohort merge folds into its picks rows.
    """
    completed: dict[str, dict] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        try:
            result = complete_sidecar(
                row,
                annotations_path=annotations_path,
                lake_reader=lake_reader,
                cache_reader=cache_reader,
                now=now,
            )
        except Exception:  # noqa: BLE001 - one bad row never stops the night
            _log.debug("Sidecar completion failed for a row.", exc_info=True)
            reasons["unexpected_error"] = reasons.get("unexpected_error", 0) + 1
            continue
        if result.get("completed"):
            completed[str(row.get("event_id") or "")] = {
                key: value
                for key, value in result.items()
                if key not in {"completed", "added_bars"}
            }
        else:
            reason = str(result.get("reason") or "unstated")
            reasons[reason] = reasons.get(reason, 0) + 1
    return {"completed": completed, "reasons": reasons}


__all__ = [
    "COMPLETED_AT_FIELD",
    "COMPLETED_REF_FIELD",
    "COMPLETED_SOURCE_FIELD",
    "REASON_ALREADY_COMPLETE",
    "REASON_ALREADY_COMPLETED",
    "REASON_NO_BARS_ANYWHERE",
    "REASON_NO_SIDECAR",
    "REASON_NO_STORE",
    "REASON_STORE_UNREACHABLE",
    "SOURCE_CACHE",
    "SOURCE_LAKE",
    "complete_sidecar",
    "complete_sidecars",
    "completed_sidecar_path",
    "read_completed_bars",
]
