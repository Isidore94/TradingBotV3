"""Where the trader's own words sit on a chart — TJ-3 item 2.

Trader, 2026-09-17: *"it also needs some sort of chart system to show me when I
commented on it so I can see exactly where I went wrong."*

This module answers one question and nothing else: **which drawn candle was on
screen when the trader said or did that?** It is PURE - no Qt, no store, no
clock, no network. Everything it needs (the bars, the journal rows, the
decisions, the trades) is handed in, because it runs on the Day Review worker
and must be able to run on a nightly slot with no event loop.

Three rules it keeps:

* **A stamp resolves to the LAST completed bar at or before it.** 10:12 belongs
  to the bar that opened at 10:10; nothing is ever placed on a bar that had not
  happened yet.
* **An unknown stamp yields NO marker.** A thought written before the first bar,
  a row with no timestamp and a stamp nobody can parse are all `None` - never
  bar zero, which is the one place the trader was certainly not looking.
* **Zones are converted, never stripped.** The journal stores UTC with an
  offset and the durable session tape is market-local (`day_review_bars`
  persists `America/Los_Angeles`), so every comparison goes through
  `astimezone`. Stripping the zone puts a 17:12 UTC note on the last bar of the
  day. A naive stamp has market-local ATTACHED to it - the adoption gate's rule,
  kept here.

The durable DAILY store (`market_story_rollups.load_index_bars`) carries an ISO
date STRING in `dt`, so daily bars are matched by market-local DATE. A reader
that assumes a datetime cannot place a marker on a daily chart at all.

Nothing here reaches a detector, a score, an alert, a watchlist, Focus, the
review queue or `review_policy.json`: a marker is a place on a chart.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

#: The zone the durable session tape is persisted in (`day_review_bars.MARKET_ZONE`).
#: Named again rather than imported: that module pulls in the bar downloader, and
#: this one is pure by contract.
MARKET_ZONE = ZoneInfo("America/Los_Angeles")

#: Every kind a marker may carry. `plan.md` TJ-3 change 1 names ten; the
#: eleventh is `prediction` - a Mentor read where the trader actually CLICKED a
#: direction is a call, and a described read is not (TJ-14 item 1). Drawing both
#: with one glyph would let the chart claim a call nobody made.
MARKER_KINDS: tuple[str, ...] = (
    "note",
    "mentor",
    "prediction",
    "forecast",
    "like",
    "pass",
    "veto",
    "click_away",
    "claim",
    "trade_open",
    "trade_close",
)

#: A recorded verdict -> the family it draws in. `dislike` and `not_today` are
#: refusals and draw beside `veto`; `swing_favorite` is an endorsement and draws
#: with `like` (lead, 2026-09-19). The verdict itself is still spelled out in the
#: marker's label, so the chart never hides WHICH refusal it was.
VERDICT_KINDS: dict[str, str] = {
    "veto": "veto",
    "dislike": "veto",
    "not_today": "veto",
    "pass": "pass",
    "m5_click_away": "click_away",
    "like": "like",
    "swing_favorite": "like",
    "claim": "claim",
}

#: How much of a written thought a marker's label carries. The label is read in a
#: tooltip beside a candle, not in the reader pane.
LABEL_LIMIT = 80


# -- stamps and bars ---------------------------------------------------------


def _as_datetime(value: Any) -> datetime | None:
    """A readable moment, or None. Never raises, never guesses."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _as_date(value: Any) -> date | None:
    """The DATE a daily bar names, from a date, a datetime or an ISO string."""
    if isinstance(value, datetime):
        return _aware(value).astimezone(MARKET_ZONE).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def _aware(moment: datetime) -> datetime:
    """Market-local ATTACHED to a naive stamp; an aware one is left alone."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=MARKET_ZONE)


def _is_daily(bars: Iterable[Mapping[str, Any]]) -> bool:
    """Whether this tape is the durable DAILY store rather than an M5 session.

    Decided by the first bar that carries anything readable: the daily store's
    `dt` is an ISO date string (or a `date`), and the session tape's is an aware
    datetime.
    """
    for bar in bars:
        stamp = bar.get("dt") if isinstance(bar, Mapping) else None
        if stamp is None:
            continue
        if isinstance(stamp, datetime):
            return False
        if isinstance(stamp, date):
            return True
        if isinstance(stamp, str):
            return _as_datetime(stamp) is None or len(stamp.strip()) <= 10
        return False
    return False


def bar_index_for(bars: Any, stamp: Any) -> int | None:
    """The index of the LAST bar at or before ``stamp``.

    ``None`` when the stamp sits before the first bar, carries no readable
    moment, or is absent - an unknown stamp yields NO marker and is never placed
    at an invented one.
    """
    rows = [bar for bar in (bars or ()) if isinstance(bar, Mapping)]
    if not rows:
        return None
    moment = _as_datetime(stamp)
    if moment is None:
        return None
    moment = _aware(moment)
    found: int | None = None
    if _is_daily(rows):
        target_date = moment.astimezone(MARKET_ZONE).date()
        for index, bar in enumerate(rows):
            bar_date = _as_date(bar.get("dt"))
            if bar_date is not None and bar_date <= target_date:
                found = index
        return found
    for index, bar in enumerate(rows):
        bar_moment = _as_datetime(bar.get("dt"))
        if bar_moment is not None and _aware(bar_moment) <= moment:
            found = index
    return found


# -- labels ------------------------------------------------------------------


def _excerpt(text: Any, limit: int = LABEL_LIMIT) -> str:
    body = " ".join(str(text or "").split())
    if len(body) <= limit:
        return body
    return f"{body[:limit].rstrip()}…"


def _marker(
    bars: Any, stamp: Any, *, kind: str, label: str, ref_id: str
) -> dict[str, Any] | None:
    """One placed marker, or None when the stamp has no bar to stand on."""
    index = bar_index_for(bars, stamp)
    if index is None:
        return None
    ref = str(ref_id or "").strip()
    if not ref:
        return None
    moment = _as_datetime(stamp)
    return {
        "stamp": moment.isoformat() if moment is not None else "",
        "index": index,
        "kind": kind,
        "label": str(label or "").strip() or kind.replace("_", " "),
        "ref_id": ref,
    }


def _ordered(markers: Iterable[Mapping[str, Any] | None]) -> tuple[dict[str, Any], ...]:
    """Bar order, ties keeping the order they were handed in (a stable sort)."""
    kept = [dict(marker) for marker in markers if marker]
    kept.sort(key=lambda marker: int(marker["index"]))
    return tuple(kept)


def _is_machine(entry: Mapping[str, Any]) -> bool:
    """`market_journal.is_machine_entry`, the ONE filter, imported lazily.

    Lazily so this module stays importable anywhere: the constant lives in the
    same repo, and the fallback is the same string it is defined as.
    """
    try:
        import market_journal

        return bool(market_journal.is_machine_entry(entry))
    except Exception:  # noqa: BLE001 - the constant is in the same repo
        return str(entry.get("origin") or "") == "auto_mode_flip"


def _entry_kind(entry: Mapping[str, Any]) -> str:
    """Which family one journal row draws in.

    `mentor` is PRESENT AND EMPTY on every row no prompt asked for, and ABSENT on
    every row written before Phase 0.31 - two different absences, neither of them
    a reason to lose the trader's words, and neither of them a Mentor read.
    """
    origin = str(entry.get("origin") or "")
    try:
        import market_journal

        forecast_origin = market_journal.ORIGIN_EXTERNAL_FORECAST
        mentor_origin = market_journal.ORIGIN_TRADE_MENTOR
    except Exception:  # noqa: BLE001
        forecast_origin, mentor_origin = "external_forecast", "trade_mentor"
    if origin == forecast_origin:
        return "forecast"
    mentor = entry.get("mentor")
    mentor = dict(mentor) if isinstance(mentor, Mapping) else {}
    if origin != mentor_origin and not mentor:
        return "note"
    prediction = mentor.get("prediction")
    if isinstance(prediction, Mapping) and str(prediction.get("direction") or "").strip():
        return "prediction"
    if str(prediction or "").strip() and not isinstance(prediction, Mapping):
        return "prediction"
    return "mentor"


def _entry_stamp(entry: Mapping[str, Any], kind: str) -> Any:
    """When the trader SAID it.

    A Mentor read is placed at `responded_at` - the hour it was answered, not
    the hour it was asked (lead, 2026-09-19) - falling back to the row's own
    `created_at`. `scheduled_at` is never used: an unanswered prompt is not a
    thing the trader said.
    """
    if kind in {"mentor", "prediction"}:
        mentor = entry.get("mentor")
        if isinstance(mentor, Mapping):
            responded = mentor.get("responded_at")
            if _as_datetime(responded) is not None:
                return responded
    return entry.get("created_at")


def _entry_label(entry: Mapping[str, Any], kind: str) -> str:
    body = _excerpt(entry.get("text"))
    prefix = {
        "forecast": "forecast",
        "mentor": "Mentor read",
        "prediction": "Mentor call",
    }.get(kind, "")
    if prefix and body:
        return f"{prefix}: {body}"
    return prefix or body or "note"


# -- the two payloads --------------------------------------------------------


def benchmark_markers(
    bars: Any, *, entries: Any = (), trades: Any = ()
) -> tuple[dict[str, Any], ...]:
    """The benchmark chart: the day's words and every trade of the day.

    The question this chart answers is *"what was the tape doing when I acted"*
    (lead, 2026-09-19), so a trade of ANY name gets an entry and an exit marker
    here, labelled with its symbol. A machine row never gets one.
    """
    markers: list[dict[str, Any] | None] = []
    for entry in entries or ():
        if not isinstance(entry, Mapping) or _is_machine(entry):
            continue
        kind = _entry_kind(entry)
        markers.append(
            _marker(
                bars,
                _entry_stamp(entry, kind),
                kind=kind,
                label=_entry_label(entry, kind),
                ref_id=str(entry.get("entry_id") or ""),
            )
        )
    markers.extend(_trade_markers(bars, trades, symbol=""))
    return _ordered(markers)


def symbol_markers(
    symbol: Any, bars: Any, *, decisions: Any = (), trades: Any = (), claims: Any = ()
) -> tuple[dict[str, Any], ...]:
    """One name's chart: that name's decisions and that name's trades.

    Nobody else's. A name is matched however it was written, because the stores
    disagree about case and the trader does not.
    """
    name = str(symbol or "").strip().upper()
    if not name:
        return ()
    markers: list[dict[str, Any] | None] = []
    for row in decisions or ():
        if not isinstance(row, Mapping):
            continue
        if str(row.get("symbol") or "").strip().upper() != name:
            continue
        markers.append(_decision_marker(bars, row))
    for row in claims or ():
        if not isinstance(row, Mapping):
            continue
        if str(row.get("symbol") or "").strip().upper() != name:
            continue
        markers.append(_claim_marker(bars, row))
    markers.extend(_trade_markers(bars, trades, symbol=name))
    return _ordered(markers)


def name_charts(
    bars_by_symbol: Any,
    *,
    decisions: Any = (),
    trades: Any = (),
    claims: Any = (),
) -> dict[str, dict[str, Any]]:
    """`{SYMBOL: {"bars": [...], "markers": (...)}}` for the names acted on.

    Only names the trader actually decided on, claimed or traded, and only where
    the session tape HOLDS that name: a symbol the bars file never got is absent
    rather than drawn on somebody else's tape.
    """
    tapes = bars_by_symbol if isinstance(bars_by_symbol, Mapping) else {}
    wanted: list[str] = []
    for rows in (decisions, claims, trades):
        for row in rows or ():
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("symbol") or "").strip().upper()
            if name and name not in wanted:
                wanted.append(name)
    charts: dict[str, dict[str, Any]] = {}
    for name in wanted:
        bars = tapes.get(name)
        if not bars:
            continue
        charts[name] = {
            "bars": list(bars),
            "markers": symbol_markers(
                name, bars, decisions=decisions, trades=trades, claims=claims
            ),
        }
    return charts


# -- the rows ----------------------------------------------------------------


def _decision_stamp(row: Mapping[str, Any]) -> Any:
    for key in ("stamp", "observed_at", "created_at", "ts"):
        value = row.get(key)
        if _as_datetime(value) is not None:
            return value
    return None


def _decision_marker(bars: Any, row: Mapping[str, Any]) -> dict[str, Any] | None:
    verdict = str(row.get("verdict") or "").strip().lower()
    kind = VERDICT_KINDS.get(verdict, "")
    if not kind:
        return None
    stamp = _decision_stamp(row)
    reason = _excerpt(row.get("reason"), 40)
    spelled = verdict.replace("_", " ")
    label = f"{spelled} · {reason}" if reason else spelled
    # `capture_id` is PRESENT AND EMPTY on every pick-feedback and swing-favorite
    # row, so the id falls back to what the row IS. Two decisions on one name
    # must never share one id: a marker nothing can tell apart is a marker the
    # page cannot select with.
    ref = str(row.get("capture_id") or "").strip()
    if not ref:
        moment = _as_datetime(stamp)
        ref = "decision:" + ":".join(
            (
                str(row.get("symbol") or "").strip().upper(),
                str(row.get("side") or "").strip().upper(),
                str(row.get("category") or "").strip(),
                verdict,
                moment.isoformat() if moment is not None else "",
            )
        )
    return _marker(bars, stamp, kind=kind, label=label, ref_id=ref)


def _claim_marker(bars: Any, row: Mapping[str, Any]) -> dict[str, Any] | None:
    stamp = row.get("claim_at") or row.get("claim_at_utc")
    horizon = str(row.get("horizon") or "").strip()
    label = f"claim · {horizon}" if horizon else "claim"
    ref = str(row.get("claimed_setup_id") or "").strip()
    if not ref:
        moment = _as_datetime(stamp)
        ref = "claim:" + ":".join(
            (
                str(row.get("symbol") or "").strip().upper(),
                str(row.get("side") or "").strip().upper(),
                moment.isoformat() if moment is not None else "",
            )
        )
    return _marker(bars, stamp, kind="claim", label=label, ref_id=ref)


def _trade_markers(bars: Any, trades: Any, *, symbol: str) -> list[dict[str, Any] | None]:
    """Both legs of a trade. An OPEN position is never given an invented exit."""
    out: list[dict[str, Any] | None] = []
    for row in trades or ():
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("symbol") or "").strip().upper()
        if symbol and name != symbol:
            continue
        ref = str(row.get("trade_id") or "").strip()
        if not ref:
            continue
        direction = str(row.get("direction") or "").strip().upper()
        stem = " ".join(part for part in (name, direction) if part)
        out.append(
            _marker(
                bars,
                row.get("opened_at"),
                kind="trade_open",
                label=f"in · {stem}" if stem else "in",
                ref_id=ref,
            )
        )
        closed = row.get("closed_at") or row.get("last_closing_leg_at")
        if _as_datetime(closed) is not None:
            out.append(
                _marker(
                    bars,
                    closed,
                    kind="trade_close",
                    label=f"out · {stem}" if stem else "out",
                    ref_id=ref,
                )
            )
    return out


__all__ = [
    "LABEL_LIMIT",
    "MARKER_KINDS",
    "MARKET_ZONE",
    "VERDICT_KINDS",
    "bar_index_for",
    "benchmark_markers",
    "name_charts",
    "symbol_markers",
]
