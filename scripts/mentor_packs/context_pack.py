"""Context pack: the desk's state right now, every row with a citable id. Read-only.

Auto mode, the D1 environment label, the trader's structural regime, open journal
positions, Focus names, econ events for the next 7 days, and the clock. A source
that cannot be read gives an "unknown" row, never a guess. The journal is opened
``mode=ro`` and the Focus files are read directly: constructing ``JournalStore`` or
``FocusPickStore`` would migrate or rewrite the desk's stores.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "context_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The desk right now: Auto mode, D1 market environment, the trader's regime, open "
            "positions, Focus names, econ events in the next 7 days, and the clock."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
ECON_DAYS_AHEAD = 7


@dataclass(frozen=True)
class Sources:
    """Where each section reads from; tests pass fakes, the app uses :func:`live_sources`."""

    auto_mode: Callable[[], str]
    d1_env: Callable[[date], str]
    regime_rows: Callable[[], list[Mapping[str, Any]]]
    open_positions: Callable[[], list[Mapping[str, Any]]]
    focus: Callable[[], Mapping[str, Mapping[str, list[str]]]]
    econ: Callable[[str], Mapping[str, Any]]
    #: P13: "Today so far: ..." from the journal (``journal_pack.today_summary``); None = not read.
    today: Callable[[datetime], str] | None = None
    #: P17: "Tape now: ..." (top 2 leading / lagging industries, today's alert counts); None = not read.
    tape_now: Callable[[datetime], str] | None = None


def _journal_rows(sql: str) -> list[dict[str, Any]]:
    from project_paths import JOURNAL_DB_FILE

    path = Path(JOURNAL_DB_FILE)
    if not path.exists():
        return []
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
    try:
        conn.row_factory = sqlite3.Row
        return [{key: row[key] for key in row.keys()} for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _live_auto_mode() -> str:
    from autopilot_core import read_auto_pilot_mode

    return read_auto_pilot_mode()


def _live_d1_env(day: date) -> str:
    import d1_environment_store

    return d1_environment_store.label_for_session(day.isoformat())


def _live_regime_rows() -> list[Mapping[str, Any]]:
    return _journal_rows("SELECT * FROM structural_regime ORDER BY segment_id")


def _live_open_positions() -> list[Mapping[str, Any]]:
    return _journal_rows(
        "SELECT trade_id, symbol, direction, quantity_opened, quantity_closed, average_entry_price, opened_at, "
        "account_label FROM trades WHERE status = 'OPEN' ORDER BY opened_at"
    )


def _live_focus() -> dict[str, dict[str, list[str]]]:
    from project_paths import FOCUS_LONGS_FILE, FOCUS_SHORTS_FILE
    from watchlist_utils import read_watchlist_symbols

    longs, shorts = Path(FOCUS_LONGS_FILE), Path(FOCUS_SHORTS_FILE)
    return {
        "m5": {"long": read_watchlist_symbols(longs), "short": read_watchlist_symbols(shorts)},
        "swing": {
            "long": read_watchlist_symbols(longs.with_name("focus_swing_longs.txt")),
            "short": read_watchlist_symbols(shorts.with_name("focus_swing_shorts.txt")),
        },
    }


def _live_econ(session: str) -> Mapping[str, Any]:
    import econ_brief

    return econ_brief.today_view(session)


def _live_today(moment: datetime) -> str:
    from mentor_packs import journal_pack

    return journal_pack.today_summary(now=moment)


#: P17: the tape-now line is rebuilt at most every 5 minutes (two board / alert file reads).
TAPE_NOW_TTL_S = 300.0
_tape_now_cache: dict[str, Any] = {}


def tape_now_text(rs: Pack, alerts: Pack) -> str:
    """One line from an rs_pack (top 2) and an alerts_pack: leaders, laggards and alert counts."""
    lead = [str(row.get("group")) for row in rs.rows if row.get("kind") == "rs_lead"][:2]
    lag = [str(row.get("group")) for row in rs.rows if row.get("kind") == "rs_lag"][:2]
    if lead or lag:
        board = f"leading {', '.join(lead) or 'n/a'}; lagging {', '.join(lag) or 'n/a'}"
    else:
        board = next((str(row.get("text")) for row in rs.rows if row.get("kind") == "none"), "industry board unknown")
    summary = next((str(row.get("text")) for row in alerts.rows if row.get("kind") == "summary"), "")
    if summary:
        alerts_text = "alerts today " + summary.split(": ", 1)[-1].split("; in your book")[0]
    elif alerts.rows:
        alerts_text = "no alerts today yet"
    else:
        alerts_text = "alerts unknown"
    return f"Tape now: {board}; {alerts_text}"


def _live_tape_now(moment: datetime) -> str:
    import time as _time

    from mentor_packs import alerts_pack, rs_pack

    cached = _tape_now_cache.get("value")
    if cached is not None and _time.monotonic() - float(_tape_now_cache.get("at") or 0.0) < TAPE_NOW_TTL_S:
        return str(cached)
    text = tape_now_text(rs_pack.build(top=2, now=moment), alerts_pack.build(now=moment))
    _tape_now_cache.update(value=text, at=_time.monotonic())
    return text


def live_sources() -> Sources:
    return Sources(
        auto_mode=_live_auto_mode,
        d1_env=_live_d1_env,
        regime_rows=_live_regime_rows,
        open_positions=_live_open_positions,
        focus=_live_focus,
        econ=_live_econ,
        today=_live_today,
        tape_now=_live_tape_now,
    )


def econ_events(view: Mapping[str, Any], session: date) -> list[dict[str, Any]]:
    """Today's and this week's events from an ``econ_brief.today_view`` dict, up to 7 days ahead."""
    horizon = (session + timedelta(days=ECON_DAYS_AHEAD)).isoformat()
    return [
        dict(event)
        for event in list(view.get("today") or ()) + list(view.get("week") or ())
        if str(event.get("date") or "") <= horizon
    ]


def _unknown(row_id: str, what: str, exc: BaseException) -> dict[str, Any]:
    return {"id": row_id, "kind": "unknown", "text": f"{what}: unknown ({type(exc).__name__})"}


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment


def build(*, now: datetime | None = None, sources: Sources | None = None) -> Pack:
    """Build the context pack. File and DB reads: call it on a worker."""
    moment = _now(now)
    src = sources or live_sources()
    local, market = moment.astimezone(PT), moment.astimezone(ET)
    session = market.date()
    rows: list[dict[str, Any]] = [
        {
            "id": "ctx:clock",
            "kind": "clock",
            "text": f"Now {local:%a %Y-%m-%d %H:%M} PT ({market:%H:%M} ET)",
            "at_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        }
    ]
    try:
        rows.append({"id": "ctx:auto_mode", "kind": "auto_mode", "text": f"Auto mode: {src.auto_mode() or 'unknown'}"})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("ctx:auto_mode", "Auto mode", exc))
    try:
        label = src.d1_env(session) or "unknown"
        rows.append({"id": "ctx:d1_env", "kind": "d1_env", "text": f"D1 environment (SPY, {session}): {label}"})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("ctx:d1_env", "D1 environment", exc))
    try:
        import structural_regime

        current = structural_regime.current_regime(src.regime_rows(), local.date())
        if current:
            text = (
                f"Trader's regime: {current.get('label') or current.get('regime')} since "
                f"{current.get('start_date')} (day {current.get('day_count')})"
            )
        else:
            text = "Trader's regime: none typed yet"
        rows.append({"id": "ctx:regime", "kind": "regime", "text": text})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("ctx:regime", "Trader's regime", exc))
    try:
        positions = list(src.open_positions())
        if not positions:
            rows.append({"id": "ctx:positions", "kind": "positions", "text": "Open journal positions: none"})
        for pos in positions:
            qty = float(pos.get("quantity_opened") or 0) - float(pos.get("quantity_closed") or 0)
            # The journal trade id keeps a position's evidence id stable across rebuilds.
            trade_id = str(pos.get("trade_id") or "").strip() or f"{pos.get('symbol')}@{pos.get('opened_at')}"
            rows.append(
                {
                    "id": f"ctx:pos:{trade_id}",
                    "kind": "position",
                    "symbol": str(pos.get("symbol") or ""),
                    # P16: the side the planner reads for "my shorts" (the text alone carried it before).
                    "side": str(pos.get("direction") or "").strip().upper(),
                    "direction": str(pos.get("direction") or "").strip().upper(),
                    "text": (
                        f"Open {pos.get('direction')} {pos.get('symbol')} {qty:g} @ "
                        f"{float(pos.get('average_entry_price') or 0):.2f} since {str(pos.get('opened_at'))[:16]}"
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("ctx:positions", "Open journal positions", exc))
    if src.today is not None:
        try:
            rows.append({"id": "ctx:today", "kind": "today", "text": str(src.today(moment))})
        except Exception as exc:  # noqa: BLE001
            rows.append(_unknown("ctx:today", "Today so far", exc))
    if src.tape_now is not None:
        try:
            rows.append({"id": "ctx:tape_now", "kind": "tape_now", "text": str(src.tape_now(moment))})
        except Exception as exc:  # noqa: BLE001
            rows.append(_unknown("ctx:tape_now", "Tape now", exc))
    try:
        focus = src.focus()
        for category in ("swing", "m5"):
            for side in ("long", "short"):
                names = list((focus.get(category) or {}).get(side) or ())
                rows.append(
                    {
                        "id": f"ctx:focus:{category}:{side}",
                        "kind": "focus",
                        "category": category,
                        "side": side,
                        "names": names,
                        "text": f"Focus {category} {side}s ({len(names)}): {', '.join(names) or 'none'}",
                    }
                )
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("ctx:focus", "Focus names", exc))
    try:
        view = src.econ(session.isoformat()) or {}
        events = econ_events(view, session)
        if not events:
            note = str(view.get("note") or "no events listed")
            rows.append({"id": "ctx:econ", "kind": "econ", "text": f"Econ next {ECON_DAYS_AHEAD} days: {note}"})
        for event in events:
            rows.append(
                {
                    "id": f"ctx:econ:{event.get('id') or len(rows)}",
                    "kind": "econ",
                    "text": f"Econ {event.get('date')} {event.get('time_et') or '--:--'} ET: {event.get('label')}",
                }
            )
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("ctx:econ", "Econ events", exc))
    return make_pack(NAME, rows)


def fixture_sources() -> Sources:
    return Sources(
        auto_mode=lambda: "DESK",
        d1_env=lambda day: "bearish_trend",
        regime_rows=lambda: [
            {"segment_id": 1, "start_date": "2026-09-01", "regime": "weak", "structure_note": "", "supersedes": None}
        ],
        open_positions=lambda: [
            {
                "trade_id": "T-NVDA-1",
                "symbol": "NVDA",
                "direction": "SHORT",
                "quantity_opened": 100,
                "quantity_closed": 0,
                "average_entry_price": 120.5,
                "opened_at": "2026-09-29T07:05:00-07:00",
            }
        ],
        focus=lambda: {"swing": {"long": ["AAPL"], "short": ["TSLA", "AMD"]}, "m5": {"long": [], "short": ["NVDA"]}},
        econ=lambda session: {
            "today": [{"id": "t1", "date": session, "time_et": "10:00", "label": "ISM Manufacturing"}],
            "week": [{"id": "w1", "date": "2026-10-02", "time_et": "08:30", "label": "Nonfarm payrolls"}],
        },
        today=lambda moment: "Today so far: 2 closed trade(s), 1 win(s), net +40.00 $, 1 open",
        tape_now=lambda moment: ("Tape now: leading Cybersecurity, Semiconductors; lagging Autos, Insurance; "
                                 "alerts today M5 2 (1 long, 1 short); D1 2 (0 long, 2 short)"),
    )


def fixture() -> Pack:
    return build(now=datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc), sources=fixture_sources())
