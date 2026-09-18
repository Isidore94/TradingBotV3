"""Pure, durable-data walk-away rows for the Day Review page (TJ-2B)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence


REJECTS = frozenset({"veto", "pass", "not_today", "dislike", "m5_click_away"})
LIKES = frozenset({"like", "swing_favorite"})


@dataclass(frozen=True)
class WalkawayRow:
    decision_id: tuple[str, str, str, str, str, str, str]
    time: datetime | None
    symbol: str
    side: str
    category: str
    what_you_did: str
    ran_after_pct: float | None = None
    held_at_close_pct: float | None = None
    traded: str = "no"
    you_made: float | None = None
    left_on_table_pct: float | None = None
    state: str = "unmeasured"


@dataclass(frozen=True)
class WalkawayDay:
    liked_not_traded: tuple[WalkawayRow, ...] = ()
    rejected: tuple[WalkawayRow, ...] = ()
    traded_left_early: tuple[WalkawayRow, ...] = ()
    claimed_d1: tuple[WalkawayRow, ...] = ()

    def __iter__(self):
        return iter(("liked_not_traded", "rejected", "traded_left_early", "claimed_d1"))

    def __contains__(self, key: object) -> bool:
        return key in tuple(self)


def _moment(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _bars_for(bars: Mapping[str, Any], symbol: str, session: str, *, allow_direct: bool = True) -> Sequence[Mapping[str, Any]]:
    direct = bars.get(symbol)
    if allow_direct and isinstance(direct, Sequence):
        return direct
    session_rows = bars.get(session)
    if isinstance(session_rows, Sequence):
        return session_rows
    if isinstance(session_rows, Mapping):
        rows = session_rows.get(symbol)
        return rows if isinstance(rows, Sequence) else ()
    # Tests and the durable reader both use a session map; tolerate a single-symbol
    # tape too, which is the old TJ-2A handoff shape.
    return ()


def _after_move(rows: Sequence[Mapping[str, Any]], stamp: datetime | None, side: str):
    eligible = [(bar, _moment(bar.get("dt"))) for bar in rows]
    eligible = [(bar, dt) for bar, dt in eligible if dt is not None and (stamp is None or dt > stamp)]
    if not eligible:
        return None
    first = eligible[0][0]
    start = _number(first.get("open"))
    if not start:
        return None
    if side == "SHORT":
        best = min((_number(bar.get("low")) for bar, _ in eligible if _number(bar.get("low")) is not None), default=None)
        return ((start - best) / start * 100) if best is not None else None
    best = max((_number(bar.get("high")) for bar, _ in eligible if _number(bar.get("high")) is not None), default=None)
    return ((best - start) / start * 100) if best is not None else None


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _identity(session: str, row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (session, str(row.get("symbol") or "").upper(), str(row.get("side") or "").upper(), str(row.get("category") or "pick"), str(row.get("verdict") or ""), str(row.get("timeframe") or "M5").upper(), str(row.get("stamp") or row.get("created_at") or ""))


def _preference_row(rows: Sequence[Mapping[str, Any]], session: str, symbol: str, side: str, source: str, verdict: str) -> Mapping[str, Any] | None:
    channels = {("annotations", "like"): "annotation:like_claim", ("annotations", "pass"): "annotation:pass", ("annotations", "veto"): "annotation:veto", ("pick_feedback", "like"): "pick_feedback:like", ("pick_feedback", "dislike"): "pick_feedback:dislike", ("pick_feedback", "not_today"): "pick_feedback:not_today", ("swing_favorites", "swing_favorite"): "swing_favorite", ("review_events", "m5_click_away"): "review_event:m5_click_away"}
    channel = channels.get((source, verdict), "")
    for row in rows:
        row_session = str(row.get("session_date") or "")[:10]
        row_channel = str(row.get("channel") or "")
        if ((row_session == session or not row_session) and str(row.get("symbol") or "").upper() == symbol and str(row.get("side") or row.get("direction") or "").upper() == side and (row_channel == channel or not row_channel)):
            return row
    return None


def _claim_events(claims: Sequence[Mapping[str, Any]]):
    events: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in claims:
        key = (str(row.get("symbol") or "").upper(), str(row.get("side") or "").upper(), str(row.get("claimed_setup_id") or ""))
        events.setdefault(key, []).append(row)
    return events


def build(session: str, sources: Mapping[str, Any], bars: Mapping[str, Any], *, trades=(), claims=(), now: datetime | None = None) -> WalkawayDay:
    """Build the four populations without touching a store, clock, or network."""
    session = str(session)[:10]
    decisions = list(sources.get("decisions") or ())
    unique: dict[tuple[str, str, str, str, str, str, str], Mapping[str, Any]] = {}
    for row in decisions:
        key = _identity(session, row)
        unique.setdefault(key, row)  # source duplicates are one decision; times are not.
    claim_events = _claim_events(tuple(claims))
    claimed_refs = {str(row.get("annotation_ref") or "") for row in claims if str(row.get("annotation_ref") or "")}
    preference = tuple(sources.get("preference") or ())
    liked: list[WalkawayRow] = []
    rejected: list[WalkawayRow] = []
    early: list[WalkawayRow] = []
    early_trade_ids: set[str] = set()

    for ident, row in unique.items():
        symbol, side, verdict = ident[1], ident[2], ident[4]
        stamp = _moment(ident[-1])
        ran = _after_move(_bars_for(bars, symbol, session), stamp, side)
        state = "measured" if ran is not None else "unmeasured no_bars"
        capture = str(row.get("capture_id") or row.get("event_id") or "")
        pref_row = _preference_row(preference, session, symbol, side, str(row.get("source") or ""), verdict)
        pref = str((pref_row or {}).get("match_state") or "")
        wanted_trade_id = str((pref_row or {}).get("trade_id") or "")
        matches = [trade for trade in trades if (str(trade.get("trade_id") or "") == wanted_trade_id if wanted_trade_id else True) and (_moment(trade.get("opened_at")) or datetime.min) > (stamp or datetime.min)]
        claimed = capture and capture in claimed_refs
        if verdict in REJECTS:
            rejected.append(WalkawayRow(ident, stamp, symbol, side, ident[3], verdict.replace("_", " "), ran_after_pct=ran, state=state))
        elif verdict in LIKES and not claimed:
            if matches and pref == "matched":
                trade = matches[0]
                exit_stamp = _moment(trade.get("last_closing_leg_at") or trade.get("closed_at"))
                if str(trade.get("status") or "").lower() != "closed":
                    early.append(WalkawayRow(ident, stamp, symbol, side, ident[3], f"liked {session[5:]}, entered {str(trade.get('opened_at') or '')[:10][5:]}", traded="yes", you_made=_number(trade.get("net_pnl")), state="pending trade open"))
                else:
                    exit_day = str((trade.get("last_closing_leg_at") or trade.get("closed_at") or ""))[:10]
                    left = _after_move(_bars_for(bars, symbol, exit_day, allow_direct=False), exit_stamp, side)
                    exit_state = "measured" if left is not None else f"unmeasured no_bars (exit {exit_day})"
                    early.append(WalkawayRow(ident, stamp, symbol, side, ident[3], f"liked {session[5:]}, entered {str(trade.get('opened_at') or '')[:10][5:]}", traded="yes", you_made=_number(trade.get("net_pnl")), left_on_table_pct=left, state=exit_state))
                    early_trade_ids.add(str(trade.get("trade_id") or ""))
            else:
                traded = "window" if pref == "window_open" else "no"
                liked.append(WalkawayRow(ident, stamp, symbol, side, ident[3], verdict.replace("_", " "), ran_after_pct=ran, traded=traded, state=state))

    # Every position closed on the selected day belongs in C, even if no
    # earlier like was linked to it. A linked trade remains one row.
    for trade in trades:
        exit_stamp = _moment(trade.get("last_closing_leg_at") or trade.get("closed_at"))
        if str(trade.get("status") or "").lower() != "closed" or not exit_stamp or exit_stamp.date().isoformat() != session:
            continue
        trade_id = str(trade.get("trade_id") or "")
        if trade_id in early_trade_ids:
            continue
        symbol = str(trade.get("symbol") or "").upper()
        side = str(trade.get("direction") or trade.get("side") or "").upper()
        left = _after_move(_bars_for(bars, symbol, session, allow_direct=False), exit_stamp, side)
        state = "measured" if left is not None else f"unmeasured no_bars (exit {session})"
        early.append(WalkawayRow((session, symbol, side, "trade", "trade_close", "M5", exit_stamp.isoformat()), exit_stamp, symbol, side, "trade", "closed trade", traded="yes", you_made=_number(trade.get("net_pnl")), left_on_table_pct=left, state=state))

    claimed_rows: list[WalkawayRow] = []
    for claim in claims:
        if str(claim.get("action") or "").lower() != "claim" or str(claim.get("session_date") or "")[:10] != session:
            continue
        symbol, side, setup = (str(claim.get("symbol") or "").upper(), str(claim.get("side") or "").upper(), str(claim.get("claimed_setup_id") or ""))
        horizon = str(claim.get("horizon") or "").lower()
        events = claim_events.get((symbol, side, setup), ())
        start = list(events).index(claim)
        drop = next((row for row in events[start + 1:] if str(row.get("action") or "").lower() in {"drop", "expire"}), None)
        decision = next((ident for ident, row in unique.items() if str(row.get("capture_id") or "") == str(claim.get("annotation_ref") or "")), (session, symbol, side, setup or "claim", "claim", "D1", ""))
        horizon_sessions = {"d1": 5, "m5": 1}.get(horizon)
        outcome = next((row for row in sources.get("outcomes") or ()
                        if str(row.get("scan_date") or row.get("session_date") or "")[:10] == session
                        and str(row.get("symbol") or "").upper() == symbol
                        and str(row.get("side") or "").upper() == side
                        and (horizon_sessions is None or int(row.get("horizon_sessions") or horizon_sessions) == horizon_sessions)), None)
        held = _number((outcome or {}).get("eod_move_pct") if isinstance(outcome, Mapping) else None)
        if not horizon:
            state = "unmeasured no_horizon"
        elif outcome and (horizon != "d1" or bool(outcome.get("measured"))):
            state = "measured"
        else:
            maturity = str((outcome or {}).get("maturity_date") or "")
            state = f"pending {maturity}" if maturity else "unmeasured no_outcome"
        if drop:
            action = str(drop.get("action") or "dropped").lower()
            action = "dropped" if action == "drop" else action
            state = f"{action} {str(drop.get('session_date') or '')[:10]}; {state}"
        claimed_rows.append(WalkawayRow(decision, _moment(decision[-1]), symbol, side, setup or "claim", "claimed D1 pick", held_at_close_pct=held, state=state))

    sort = lambda row: (row.ran_after_pct is None, -(row.ran_after_pct or 0), row.symbol)
    return WalkawayDay(tuple(sorted(liked, key=sort)), tuple(sorted(rejected, key=sort)), tuple(early), tuple(claimed_rows))


__all__ = ["WalkawayDay", "WalkawayRow", "build"]
