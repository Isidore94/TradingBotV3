"""The M5 bars a day-trade pass is attached to, stored beside the row.

Trader, 2026-08-31: *"if the M5 data for the symbol is already in memory at
that moment, attach it, so an AI can later see the chart as it was"* - and,
explicitly, *"if that is hard, just store the exact timestamp and let a future
AI read the charts by it."* Both halves are honoured here: bars ride along when
the desk already holds them, and their absence costs the row nothing.

Why a SIDECAR and not the row itself. ``ui.annotations.store`` writes every
annotation as one bounded (:data:`~ui.annotations.store.MAX_ROW_BYTES`, 4096)
single buffered write, which is what keeps cooperating writers from
interleaving and what makes a torn tail cost exactly one decision. One session
of M5 bars is ~78 RTH bars and well over 8 KB serialised - it would blow that
cap, and raising the cap would trade the store's confinement property for a
convenience. So the bars go to their own file keyed by ``event_id`` and the row
carries a reference. The decision stream stays small, append-only and
line-oriented; the evidence hangs off it.

Ordering rule: the sidecar is written FIRST and the row second, so a reference
in the stream always has a file behind it. A sidecar whose row never landed is
an orphan that costs a few KB and nothing else; a row pointing at a file that
does not exist would be a lie in the permanent record.

Never fetches. The caller hands in whatever the desk already materialised for
the chart in front of the trader (``SymbolSnapshotWidget.cached_m5_bars``);
an empty list is an ordinary, expected outcome and simply means the row is
written with its timestamp alone.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from project_paths import TRADER_ANNOTATIONS_FILE

_log = logging.getLogger(__name__)

SIDECAR_SCHEMA_VERSION = 1
#: One directory beside the stream, one file per annotation.
SIDECAR_DIRNAME = "trader_annotation_bars"
#: Hard stop on top of the one-session rule. An RTH session is 78 M5 bars and
#: a pre/post session is under 300; anything past this is a malformed series,
#: not a trading day, and the newest bars are the ones worth keeping.
MAX_SIDECAR_BARS = 500

_BAR_FIELDS = ("open", "high", "low", "close", "volume")


def sidecar_dir(annotations_path: Any = None) -> Path:
    """Where sidecars live for a given annotation stream."""
    target = Path(annotations_path or TRADER_ANNOTATIONS_FILE)
    return target.parent / SIDECAR_DIRNAME


def sidecar_path(event_id: str, annotations_path: Any = None) -> Path:
    """The sidecar file for one annotation id."""
    key = "".join(ch for ch in str(event_id or "") if ch.isalnum() or ch in "-_")
    if not key:
        raise ValueError("event_id is required to name a bar sidecar")
    return sidecar_dir(annotations_path) / f"{key}.json"


def _bar_date(bar: Mapping[str, Any]) -> date | None:
    stamp = bar.get("dt")
    if isinstance(stamp, datetime):
        return stamp.date()
    if isinstance(stamp, date):
        return stamp
    text = str(stamp or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def one_session(bars: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The newest session's bars only, capped at :data:`MAX_SIDECAR_BARS`.

    The desk hands out two sessions (``m5_chart_bars(max_sessions=2)``) because
    an ATR(14) needs warm-up bars forty minutes after the open. A pass is a
    judgement about TODAY's chart, so the sidecar keeps today - the trader
    asked for "the chart as it was", not a week of it.

    Bars with an unreadable timestamp are kept rather than dropped: uncertainty
    is never a reason to delete evidence (plan.md sec 5), and the worst case is
    a slightly longer file.
    """
    rows = [bar for bar in bars if isinstance(bar, Mapping)]
    if not rows:
        return []
    dated = [(bar, _bar_date(bar)) for bar in rows]
    known = [day for _bar, day in dated if day is not None]
    if known:
        newest = max(known)
        rows = [bar for bar, day in dated if day is None or day == newest]
    return rows[-MAX_SIDECAR_BARS:]


def _serialisable_bar(bar: Mapping[str, Any]) -> dict[str, Any]:
    stamp = bar.get("dt")
    if isinstance(stamp, (datetime, date)):
        stamp_text = stamp.isoformat()
    else:
        stamp_text = str(stamp or "")
    row: dict[str, Any] = {"dt": stamp_text}
    for field in _BAR_FIELDS:
        value = bar.get(field)
        if value is None:
            continue
        try:
            row[field] = float(value)
        except (TypeError, ValueError):
            continue
    return row


def write_pass_bars(
    event_id: str,
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str = "",
    side: str = "",
    created_at: str = "",
    annotations_path: Any = None,
) -> dict[str, Any]:
    """Store one session of M5 bars. Returns the row fields that refer to it.

    An empty result means "no bars are attached" - because there were none
    cached, or because the write failed. Both are the same thing to the
    caller: write the row anyway. An evidence store is never allowed to cost
    the thing it records, and here the thing recorded is the trader's reason
    for passing, which stands with or without a chart behind it.
    """
    kept = one_session(bars)
    if not kept:
        return {}
    try:
        path = sidecar_path(event_id, annotations_path)
    except ValueError:
        return {}
    payload = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "event_id": str(event_id),
        "symbol": str(symbol or "").strip().upper(),
        "side": str(side or "").strip().upper(),
        "interval": "M5",
        "created_at": str(created_at or ""),
        "bar_count": len(kept),
        "bars": [_serialisable_bar(bar) for bar in kept],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, sort_keys=True, default=str), encoding="utf-8"
        )
    except OSError:
        _log.debug("Pass-bar sidecar write failed for %s.", event_id, exc_info=True)
        return {}
    first = payload["bars"][0]["dt"]
    last = payload["bars"][-1]["dt"]
    return {
        "m5_bars_ref": f"{SIDECAR_DIRNAME}/{path.name}",
        "m5_bar_count": len(kept),
        "m5_first_bar": first,
        "m5_last_bar": last,
    }


def read_pass_bars(row: Mapping[str, Any], *, annotations_path: Any = None) -> dict:
    """The sidecar behind one annotation row, or ``{}``.

    Offline analysis entry point. A missing or unreadable sidecar reads as
    empty for the same reason a corrupt annotation line is skipped rather than
    fatal: one lost chart must never make the rest of the record unreadable.
    """
    ref = str(row.get("m5_bars_ref") or "").strip()
    if not ref:
        return {}
    base = Path(annotations_path or TRADER_ANNOTATIONS_FILE).parent
    path = base / ref
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
