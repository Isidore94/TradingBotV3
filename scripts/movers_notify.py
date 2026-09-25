"""Movers phone/desk notices (P8 P8): which NEW board names to announce, and where.

Pure: no Qt, no I/O. `MoversService` calls `MoversNotifier.decide` once per tick
(the final publish) on its worker, sends the push lines, hands DESK lines to the
Alert Center's sound path, and logs every notice with `notice_records`.

- Lists: Pop (both sides, biggest move first), Dip-strong (dip long) and
  Rip-weak (rip short).
- New = not on that list at the previous tick. After the list's SPY state
  changes (another turn, another start bar) every name counts as new.
- At most TOP_NEW names per list, one notice per list per bar, and at most
  MAX_NOTICES_PER_WINDOW notices per NOTICE_WINDOW_MINUTES overall.
- AWAY / EVENING -> phone push; DESK -> desk sound + status line; any other
  mode (OFF, unknown) -> nothing. Hidden names (the Oil & Gas / Real Estate
  switch, the board's Hide for today) never count and never take a slot.
- M5 watch (trader 2026-09-25): the same names, in DESK/AWAY/EVENING, become
  `adoption_candidates` for the M5 Focus AUTO lane, but only when
  `focus_adoption_gate` holds on the bar: long above the previous session's
  high (durable daily bars) and above session VWAP, short below both. Unknown
  levels never pass. `adopt_records` are the `kind: adopt` log rows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Mapping

import focus_adoption_gate

#: Names per list per notice.
TOP_NEW = 3
#: At most this many notices (push or desk) in any NOTICE_WINDOW_MINUTES.
MAX_NOTICES_PER_WINDOW = 6
NOTICE_WINDOW_MINUTES = 10
#: Modes that push to the phone; the mode that sounds on the desk.
PUSH_MODES = ("AWAY", "EVENING")
DESK_MODE = "DESK"
#: Modes whose notices also feed the M5 Focus watch (the bot watches either way).
ADOPT_MODES = ("DESK", "AWAY", "EVENING")
#: ntfy priority for a Movers line (never urgent).
PUSH_PRIORITY = "default"
CHANNEL_PUSH = "push"
CHANNEL_DESK = "desk"
#: list key -> label.
LISTS = {"pop": "Pop", "dip_strong": "Dip-strong", "rip_weak": "Rip-weak"}
#: The outcome log each list's notice rows go to.
LOG_FOR_LIST = {"pop": "pop", "dip_strong": "dip", "rip_weak": "dip"}
BAR_MINUTES = 5


@dataclass(frozen=True)
class MoversNotice:
    list_key: str
    label: str
    channel: str
    mode: str
    bar: str  # the board's as_of (SPY's last completed bar start)
    state: str
    line: str
    rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"list": self.list_key, "label": self.label, "channel": self.channel,
                "mode": self.mode, "bar": self.bar, "state": self.state, "line": self.line,
                "rows": [dict(r) for r in self.rows]}


def list_rows(board: Mapping[str, Any], list_key: str) -> list[dict[str, Any]]:
    """One list's rows in board order, each tagged `_side`."""
    def side_rows(mode: str, side: str) -> list[dict[str, Any]]:
        return [dict(r, _side=side) for r in (((board.get(mode) or {}).get(side)) or [])]

    if list_key == "pop":
        both = side_rows("pop", "long") + side_rows("pop", "short")
        return sorted(both, key=lambda r: -abs(float(r.get("pop_score") or 0.0)))
    if list_key == "dip_strong":
        return side_rows("dip", "long")
    if list_key == "rip_weak":
        return side_rows("rip", "short")
    return []


def list_state(board: Mapping[str, Any], list_key: str) -> tuple[str, str]:
    """(state name, start stamp) that lights the list; ("", "") when it is dark."""
    state = board.get("state") or {}
    start = str(state.get("start_dt") or state.get("extreme_time") or "")
    if list_key == "pop":
        return str(state.get("state") or "unknown"), ""
    if list_key == "dip_strong":
        if state.get("pullback"):
            return "pullback", start
        if state.get("bounce"):
            return "bounce", start
        return "", ""
    if list_key == "rip_weak":
        return ("rally", start) if state.get("rally") else ("", "")
    return "", ""


def row_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('symbol') or '').strip().upper()}|{row.get('_side') or 'long'}"


def _score(row: Mapping[str, Any], list_key: str) -> float | None:
    value = row.get("pop_score" if list_key == "pop" else "dip_score")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _bar_clock(bar: str, local_tz=None) -> str:
    """HH:MM on the desk clock when the bar closed ('' when unreadable)."""
    try:
        start = datetime.fromisoformat(bar)
    except ValueError:
        return ""
    end = start + timedelta(minutes=BAR_MINUTES)
    if end.tzinfo is not None:
        end = end.astimezone(local_tz if local_tz is not None else desk_zone())
    return end.strftime("%H:%M")


def desk_zone():
    """The desk's one display clock (Day Review's `market_session` zone); None = system."""
    try:
        from market_session import get_market_local_timezone

        return get_market_local_timezone()[0]
    except Exception:
        return None


def format_line(label: str, rows: Iterable[Mapping[str, Any]], list_key: str, clock: str) -> str:
    """'Pop: NVDA +1.2 ATR rvol 3.1 · AMD -0.9 ATR rvol — · 10:35'."""
    parts = []
    for row in rows:
        score = _score(row, list_key)
        rvol = row.get("rvol")
        score_text = "—" if score is None else f"{score:+.1f}"
        rvol_text = "—" if rvol is None else f"{float(rvol):.1f}"
        parts.append(f"{row.get('symbol')} {score_text} ATR rvol {rvol_text}")
    if clock:
        parts.append(clock)
    return f"{label}: " + " · ".join(parts)


class MoversNotifier:
    """Remembers the last tick's lists and the notice budget. One per service."""

    def __init__(self) -> None:
        self._previous: dict[str, set[str]] = {}
        self._signature: dict[str, tuple[str, str]] = {}
        self._last_bar: dict[str, str] = {}
        self._session = ""
        self._sent: deque[datetime] = deque()

    def decide(
        self,
        board: Mapping[str, Any],
        *,
        mode: str | None,
        now: datetime,
        hidden_keys: Iterable[str] = (),
        is_sector_hidden: Callable[[str], bool] | None = None,
        local_tz=None,
    ) -> list[MoversNotice]:
        """The notices for this tick. Always advances the list memory, whatever the mode."""
        bar = str(board.get("as_of") or "")
        if not bar or board.get("as_of_stale"):
            return []  # SPY unknown: no bar to anchor on, nothing is new
        session = bar[:10]
        if session != self._session:
            self._session = session
            self._previous, self._signature, self._last_bar = {}, {}, {}
        hidden = {str(k) for k in hidden_keys or ()}  # the board's "SYM|side" keys
        text_mode = str(mode or "").strip().upper()
        channel = (CHANNEL_PUSH if text_mode in PUSH_MODES
                   else CHANNEL_DESK if text_mode == DESK_MODE else "")
        clock = _bar_clock(bar, local_tz)
        out: list[MoversNotice] = []
        for list_key, label in LISTS.items():
            rows = list_rows(board, list_key)
            signature = list_state(board, list_key)
            before = self._previous.get(list_key, set())
            if self._signature.get(list_key) != signature:
                before = set()  # a new turn: everything on it is new
            self._signature[list_key] = signature
            self._previous[list_key] = {row_key(r) for r in rows}
            if not channel or self._last_bar.get(list_key) == bar:
                continue
            fresh = []
            for row in rows:
                key = row_key(row)
                if key in before or key in hidden:
                    continue
                symbol = key.split("|", 1)[0]
                if is_sector_hidden is not None and _safe_hidden(is_sector_hidden, symbol):
                    continue
                fresh.append(row)
                if len(fresh) >= TOP_NEW:
                    break
            if not fresh or not self._budget_ok(now):
                continue
            self._sent.append(now)
            self._last_bar[list_key] = bar
            picked = tuple(
                {"symbol": str(r.get("symbol") or "").upper(), "side": r.get("_side") or "long",
                 "score": _score(r, list_key), "rvol": r.get("rvol"),
                 "last": r.get("last"), "session_vwap": r.get("session_vwap")}
                for r in fresh
            )
            out.append(MoversNotice(
                list_key=list_key, label=label, channel=channel, mode=text_mode, bar=bar,
                state=signature[0], line=format_line(label, fresh, list_key, clock), rows=picked,
            ))
        return out

    def _budget_ok(self, now: datetime) -> bool:
        window = timedelta(minutes=NOTICE_WINDOW_MINUTES)
        while self._sent and now - self._sent[0] >= window:
            self._sent.popleft()
        return len(self._sent) < MAX_NOTICES_PER_WINDOW


def _safe_hidden(test: Callable[[str], bool], symbol: str) -> bool:
    try:
        return bool(test(symbol))
    except Exception:
        return False  # unknown classification = shown


def notice_records(notice: MoversNotice, *, pushed_at: datetime, result: str = "") -> list[dict]:
    """One `kind: notice` log row per announced name (outcome readers skip this kind)."""
    return [
        {"kind": "notice", "session": notice.bar[:10], "symbol": row["symbol"],
         "side": row["side"], "list": notice.list_key, "state": notice.state,
         "channel": notice.channel, "mode": notice.mode, "bar": notice.bar,
         "pushed_at": pushed_at.isoformat(timespec="seconds"), "result": result,
         "score": row.get("score"), "rvol": row.get("rvol")}
        for row in notice.rows
    ]


def adoption_candidates(
    notices: Iterable[MoversNotice],
    levels: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Each noticed name with its level gate verdict for the M5 Focus AUTO lane.

    `levels` is {symbol: {"prev_high", "prev_low"}} from the durable daily bars;
    a missing symbol or level is UNKNOWN and never passes. Only DESK, AWAY and
    EVENING notices count (OFF makes no notices at all)."""
    out: list[dict[str, Any]] = []
    for notice in notices:
        if notice.mode not in ADOPT_MODES:
            continue
        for row in notice.rows:
            symbol = str(row.get("symbol") or "").upper()
            side = "short" if row.get("side") == "short" else "long"
            level = levels.get(symbol) or {}
            gate = {"prev_high": level.get("prev_high"), "prev_low": level.get("prev_low"),
                    "vwap": row.get("session_vwap"), "last": row.get("last")}
            state, reason = focus_adoption_gate.focus_adoption_gate_state(
                side, gate["last"], gate["prev_high"], gate["prev_low"], gate["vwap"]
            )
            out.append({"symbol": symbol, "side": side, "list": notice.list_key,
                        "label": notice.label, "state": notice.state, "mode": notice.mode,
                        "bar": notice.bar, "level_gate": gate, "gate": state,
                        "gate_reason": reason, "passes": state == focus_adoption_gate.OPEN})
    return out


def adopt_records(results: Iterable[Mapping[str, Any]], *, at: str) -> list[dict[str, Any]]:
    """One `kind: adopt` log row per candidate, with what the M5 Focus lane did."""
    return [
        {"kind": "adopt", "session": str(c.get("bar") or "")[:10], "symbol": c.get("symbol"),
         "side": c.get("side"), "list": c.get("list"), "state": c.get("state"),
         "mode": c.get("mode"), "bar": c.get("bar"), "level_gate": dict(c.get("level_gate") or {}),
         "gate": c.get("gate"), "gate_reason": c.get("gate_reason"),
         "result": c.get("result") or "", "adopted_at": at}
        for c in results
    ]
