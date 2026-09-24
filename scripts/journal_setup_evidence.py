"""Setup suggestions for a trade from what the desk logged BEFORE the entry.

Reads three append-only logs the desk already keeps and joins them to a trade
by (underlying symbol, side, time):

* review events (`alert_review_events*`): alerts that fired, watches the trader
  armed and cards the trader liked, each carrying a setup kind;
* claimed picks (`claimed_picks.jsonl`): the setup the trader claimed on a D1 row;
* Focus daily picks (`human_focus_daily_picks.csv`): corroboration only - a
  Focus membership names no setup, so it raises a match and never makes one.

Point in time: a row counts only if it was stamped strictly before the trade's
first fill. Nothing here reads an outcome. Every candidate is a SUGGESTION for
the scanner lane of `AutoTagger`; the bulk tagger's threshold still decides what
is written, and the Trade Mentor offers the rest for one-click confirmation.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")

#: `auto_tag_candidates.source` prefix for this lane: `evidence:<family>`.
EVIDENCE_SOURCE = "evidence"

#: How far back a D1 (swing) signal still explains an entry. M5 signals count
#: only on the entry's own session.
D1_LOOKBACK = timedelta(days=7)
#: A fired alert this close to the entry is the entry's trigger.
TRIGGER_WINDOW = timedelta(minutes=90)
#: A claim older than this is about the ticker, not this trade.
CLAIM_LOOKBACK = timedelta(days=14)

#: Base confidence per evidence family and timing. The 0.70 bulk-tag threshold
#: is cleared only by a claim, or by an alert that fired (or a card liked) on
#: the entry's own session and side.
CONFIDENCE = {
    ("claimed_pick", "any"): 0.86,
    ("alert_fired", "trigger"): 0.78,
    ("alert_fired", "session"): 0.72,
    ("alert_fired", "earlier"): 0.62,
    ("card_liked", "trigger"): 0.76,
    ("card_liked", "session"): 0.72,
    ("card_liked", "earlier"): 0.60,
    ("watch_armed", "trigger"): 0.68,
    ("watch_armed", "session"): 0.64,
    ("watch_armed", "earlier"): 0.56,
    ("alert_shown", "trigger"): 0.64,
    ("alert_shown", "session"): 0.60,
    ("alert_shown", "earlier"): 0.50,
}
#: Evidence that states no side matches either side, a little less surely.
SIDE_SILENT_PENALTY = 0.06
#: Being on the trader's Focus list for that side on the entry session.
FOCUS_BONUS = 0.04

#: Review actions and the evidence family each one is.
FIRED_ACTIONS = {"focus_d1_flag", "watch_fired", "d1_event_fired", "any_bounce_fired"}
ARMED_ACTIONS = {"arm_watch", "arm_d1_event", "arm_any_bounce"}
LIKED_ACTIONS = {"like_advance"}
SHOWN_ACTIONS = {"shown"}

#: Card tags that say where a card came from, not which setup it is.
GENERIC_CARD_TAGS = frozenset(
    {
        "",
        "manual_chart",
        "auto_pick",
        "focus_review",
        "focus_d1_event",
        "d1_flag_long",
        "d1_flag_short",
        "green",
        "red",
        "chart_watch",
        "watch",
    }
)
#: Kinds that are D1 (swing) signals; everything else is intraday.
D1_KINDS = frozenset(
    {
        "new_5d_high",
        "new_5d_low",
        "new_20d_high",
        "new_20d_low",
        "avwape_dev1_bounce",
        "avwape_dev1_break",
        "avwape_bounce",
        "avwape_break",
        "ema15_reject",
        "sma_break",
        "d1_ema15",
        "d1_level_above",
        "d1_level_below",
        "trendline_break",
        "range_breakout",
        "line_break",
    }
)

_OCC_OPTION = re.compile(r"^([A-Z][A-Z.]{0,5})(\d{6})([CP])(\d{8})$")
_QUESTRADE_OPTION = re.compile(r"^([A-Z][A-Z.]{0,5})(\d{1,2}[A-Z]{3}\d{2})([CP])(\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class EvidenceRow:
    """One logged fact about a symbol, stamped with when the desk knew it."""

    symbol: str
    side: str  # LONG | SHORT | "" (side-silent)
    at: datetime  # tz-aware
    family: str  # claimed_pick | alert_fired | card_liked | watch_armed | alert_shown | focus_pick
    kind: str  # setup slug; "" for a focus pick
    horizon: str  # d1 | m5
    ref: str = ""
    detail: str = ""


def option_underlying(symbol: Any) -> tuple[str, str]:
    """`(underlying, right)` for an option symbol, else `(symbol, "")`.

    Reads OCC (`BK250905C00102000`) and Questrade (`QQQ26JUN26P700.00`) forms.
    """
    text = str(symbol or "").strip().upper().replace(" ", "")
    for pattern in (_OCC_OPTION, _QUESTRADE_OPTION):
        match = pattern.match(text)
        if match:
            return match.group(1), match.group(3)
    return text, ""


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"LONG", "BUY", "BOT", "BTO", "COVER"}:
        return "LONG"
    if text in {"SHORT", "SELL", "SLD", "STO", "SSHORT"}:
        return "SHORT"
    return ""


def underlying_view(trade: Mapping[str, Any]) -> tuple[str, str]:
    """`(underlying, side on the underlying)` for a trade.

    A long put or a short call is SHORT the underlying; everything else keeps
    the trade's own direction.
    """
    underlying, right = option_underlying(trade.get("symbol"))
    side = _side(trade.get("direction"))
    if right == "P" and side:
        side = "SHORT" if side == "LONG" else "LONG"
    return underlying, side


def aware_moment(value: Any, naive_tz: tzinfo | None = None) -> datetime | None:
    """A tz-aware datetime, or None. Naive text is read in `naive_tz` (machine-local by default)."""
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=naive_tz) if naive_tz is not None else moment.astimezone()
    return moment


def _is_date_only(moment: datetime) -> bool:
    local = moment.astimezone(MARKET_TZ)
    return (local.hour, local.minute, local.second, local.microsecond) == (0, 0, 0, 0)


def _event_kind(event: Mapping[str, Any]) -> str:
    detail = event.get("detail") if isinstance(event.get("detail"), Mapping) else {}
    for value in (detail.get("kind"), event.get("chart_watch_kind")):
        text = str(value or "").strip().lower()
        if text:
            return text
    bounce = str(event.get("bounce_types") or "").strip().lower()
    if bounce:
        return bounce.split(";", 1)[0].strip()
    tag = str(event.get("tag") or "").strip().lower()
    return "" if tag in GENERIC_CARD_TAGS else tag


def _event_horizon(event: Mapping[str, Any], kind: str) -> str:
    if kind in D1_KINDS or kind.startswith("d1_"):
        return "d1"
    if str(event.get("action") or "") in {"focus_d1_flag", "d1_event_fired", "arm_d1_event"}:
        return "d1"
    if str(event.get("timeframe") or "").upper() == "D1" or event.get("is_d1") is True:
        return "d1"
    return "m5"


def review_event_rows(
    events: Iterable[Mapping[str, Any]], *, naive_tz: tzinfo | None = None
) -> list[EvidenceRow]:
    """Evidence rows from review events that name a setup kind."""
    rows: list[EvidenceRow] = []
    for event in events or ():
        action = str(event.get("action") or "")
        if action in FIRED_ACTIONS:
            family = "alert_fired"
        elif action in LIKED_ACTIONS:
            family = "card_liked"
        elif action in ARMED_ACTIONS:
            family = "watch_armed"
        elif action in SHOWN_ACTIONS:
            family = "alert_shown"
        else:
            continue
        kind = _event_kind(event)
        symbol = str(event.get("symbol") or "").strip().upper()
        at = aware_moment(event.get("ts"), naive_tz)
        if not kind or not symbol or at is None:
            continue
        detail = event.get("detail") if isinstance(event.get("detail"), Mapping) else {}
        message = str(detail.get("message") or event.get("trigger") or "").strip()
        rows.append(
            EvidenceRow(
                symbol=symbol,
                side=_side(event.get("side")),
                at=at,
                family=family,
                kind=kind,
                horizon=_event_horizon(event, kind),
                ref=str(event.get("review_record_id") or event.get("event_id") or ""),
                detail=message[:120],
            )
        )
    return rows


def claimed_pick_rows(rows: Iterable[Mapping[str, Any]]) -> list[EvidenceRow]:
    """Claim rows that name a setup. Drops are applied per trade (`_claim_drops`)."""
    claims: list[EvidenceRow] = []
    for row in rows or ():
        symbol = str(row.get("symbol") or "").strip().upper()
        side = _side(row.get("side"))
        setup = str(row.get("claimed_setup_id") or "").strip().lower()
        at = aware_moment(row.get("claim_at_utc") or row.get("claim_at"))
        if not symbol or at is None:
            continue
        if str(row.get("action") or "").strip().lower() != "claim" or not setup:
            continue
        claims.append(
            EvidenceRow(
                symbol=symbol,
                side=side,
                at=at,
                family="claimed_pick",
                kind=setup,
                horizon=str(row.get("horizon") or "d1").strip().lower() or "d1",
                ref=str(row.get("annotation_ref") or ""),
                detail=str(row.get("source") or ""),
            )
        )
    return claims


def focus_pick_rows(
    rows: Iterable[Mapping[str, Any]], *, naive_tz: tzinfo | None = None
) -> list[EvidenceRow]:
    """Focus memberships from `human_focus_daily_picks.csv` rows."""
    result: list[EvidenceRow] = []
    for row in rows or ():
        symbol = str(row.get("symbol") or "").strip().upper()
        at = aware_moment(row.get("snapshotted_at"), naive_tz)
        session = str(row.get("trade_date") or "").strip()[:10]
        if not symbol or at is None or not session:
            continue
        source = str(row.get("source") or "").strip().lower()
        result.append(
            EvidenceRow(
                symbol=symbol,
                side=_side(row.get("side")),
                at=at,
                family="focus_pick",
                kind="",
                horizon="d1" if "swing" in source else "m5",
                ref=session,
                detail=source,
            )
        )
    return result


class EvidenceIndex:
    """Evidence rows grouped by symbol, plus the claim drops that retire claims."""

    def __init__(self, rows: Iterable[EvidenceRow] = (), claim_drops=None) -> None:
        self.by_symbol: dict[str, list[EvidenceRow]] = defaultdict(list)
        for row in rows:
            self.by_symbol[row.symbol].append(row)
        self.claim_drops: dict[tuple[str, str, str], list[datetime]] = dict(claim_drops or {})

    @classmethod
    def load(
        cls,
        *,
        review_events: Iterable[Mapping[str, Any]] | None = None,
        claimed_rows: Iterable[Mapping[str, Any]] | None = None,
        focus_rows: Iterable[Mapping[str, Any]] | None = None,
        naive_tz: tzinfo | None = None,
    ) -> "EvidenceIndex":
        """Build from the live logs (read-only). A source that cannot be read adds nothing."""
        if review_events is None:
            review_events = _read_review_events()
        if claimed_rows is None:
            claimed_rows = _read_claimed_rows()
        if focus_rows is None:
            focus_rows = _read_focus_rows()
        claimed_rows = list(claimed_rows)
        rows = [
            *review_event_rows(review_events, naive_tz=naive_tz),
            *claimed_pick_rows(claimed_rows),
            *focus_pick_rows(focus_rows, naive_tz=naive_tz),
        ]
        return cls(rows, claim_drops=_claim_drops(claimed_rows))

    def candidates_for(self, trade: Mapping[str, Any]) -> list[dict[str, Any]]:
        return evidence_candidates(trade, self.by_symbol.get(underlying_view(trade)[0], ()), self.claim_drops)


def _claim_drops(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[datetime]]:
    drops: dict[tuple[str, str, str], list[datetime]] = defaultdict(list)
    for row in rows or ():
        if str(row.get("action") or "").strip().lower() != "drop":
            continue
        at = aware_moment(row.get("claim_at_utc") or row.get("claim_at"))
        symbol = str(row.get("symbol") or "").strip().upper()
        if at is None or not symbol:
            continue
        key = (symbol, _side(row.get("side")), str(row.get("claimed_setup_id") or "").strip().lower())
        drops[key].append(at)
    return dict(drops)


def _timing(row: EvidenceRow, entry: datetime, entry_session) -> str | None:
    """`trigger` / `session` / `earlier`, or None when the row cannot explain the entry."""
    if row.at >= entry:
        return None
    gap = entry - row.at
    same_session = row.at.astimezone(MARKET_TZ).date() == entry_session
    if row.family == "claimed_pick":
        return "any" if gap <= CLAIM_LOOKBACK else None
    if same_session:
        return "trigger" if gap <= TRIGGER_WINDOW and not _is_date_only(entry) else "session"
    if row.horizon == "d1" and gap <= D1_LOOKBACK:
        return "earlier"
    return None


def evidence_candidates(
    trade: Mapping[str, Any],
    rows: Iterable[EvidenceRow],
    claim_drops: Mapping[tuple[str, str, str], list[datetime]] | None = None,
) -> list[dict[str, Any]]:
    """Scanner-lane candidates for one trade, best first.

    Each is `{tag, confidence, source, rationale, context_row_id}`. Only rows
    stamped before the first fill count; opposite-side rows never match.
    """
    entry = aware_moment(trade.get("opened_at"))
    underlying, side = underlying_view(trade)
    if entry is None or not underlying:
        return []
    entry_session = entry.astimezone(MARKET_TZ).date()
    rows = list(rows)
    on_focus = any(
        row.family == "focus_pick"
        and row.at < entry
        and row.ref == entry_session.isoformat()
        and (not row.side or not side or row.side == side)
        for row in rows
    )
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.family == "focus_pick" or not row.kind:
            continue
        if row.side and side and row.side != side:
            continue
        timing = _timing(row, entry, entry_session)
        if timing is None:
            continue
        if row.family == "claimed_pick" and any(
            row.at < dropped < entry for dropped in (claim_drops or {}).get((row.symbol, row.side, row.kind), ())
        ):
            continue
        confidence = CONFIDENCE.get((row.family, timing))
        if confidence is None:
            continue
        if not row.side or not side:
            confidence -= SIDE_SILENT_PENALTY
        if on_focus:
            confidence += FOCUS_BONUS
        confidence = round(min(0.95, max(0.01, confidence)), 4)
        minutes = int((entry - row.at).total_seconds() // 60)
        when = f"{minutes} min before entry" if minutes < 24 * 60 else f"{minutes // (24 * 60)} day(s) before entry"
        rationale = f"{row.family.replace('_', ' ')} {row.kind} on {underlying} {when}"
        if row.detail:
            rationale += f": {row.detail}"
        if on_focus:
            rationale += "; on your Focus list that session"
        current = best.get(row.kind)
        if current is not None and current["confidence"] >= confidence:
            continue
        best[row.kind] = {
            "tag": row.kind,
            "confidence": confidence,
            "source": f"{EVIDENCE_SOURCE}:{row.family}",
            "rationale": rationale,
            "context_row_id": row.ref,
        }
    return sorted(best.values(), key=lambda item: (-item["confidence"], item["tag"]))


def _read_review_events() -> list[dict[str, Any]]:
    try:
        from review_events import load_review_events

        return load_review_events()
    except Exception:  # noqa: BLE001 - a suggestion source is never fatal
        logging.debug("Review events unavailable to the evidence lane.", exc_info=True)
        return []


def _read_claimed_rows() -> list[dict[str, Any]]:
    try:
        from claimed_picks import load_rows

        return load_rows()
    except Exception:  # noqa: BLE001
        logging.debug("Claimed picks unavailable to the evidence lane.", exc_info=True)
        return []


def _read_focus_rows(path: Path | None = None) -> list[dict[str, Any]]:
    try:
        if path is None:
            from project_paths import HUMAN_FOCUS_DAILY_PICKS_FILE

            path = Path(HUMAN_FOCUS_DAILY_PICKS_FILE)
        if not Path(path).exists():
            return []
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:  # noqa: BLE001
        logging.debug("Focus daily picks unavailable to the evidence lane.", exc_info=True)
        return []


__all__ = [
    "EVIDENCE_SOURCE",
    "EvidenceIndex",
    "EvidenceRow",
    "aware_moment",
    "claimed_pick_rows",
    "evidence_candidates",
    "focus_pick_rows",
    "option_underlying",
    "review_event_rows",
    "underlying_view",
]
