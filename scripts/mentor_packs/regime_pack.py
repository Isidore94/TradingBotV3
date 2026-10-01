"""Regime pack (the tape): Auto mode, D1 environment, the trader's regime, the night's regime
read, econ events for the next 7 days, the sector board and SPY-pause observations.

Read-only and deterministic: every source is a file or the journal opened ``mode=ro``;
nothing is computed from bars here. A source that cannot be read gives an "unknown"
row, never a guess. Index RRS for SPY/QQQ/IWM has no file-based source yet, so the live
reader returns nothing and those rows are simply absent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs import context_pack
from mentor_packs.registry import Pack, make_pack

NAME = "regime_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The tape right now: Auto mode, D1 market environment, the trader's regime, last night's "
            "regime read, econ events in the next 7 days, the sector board and SPY-pause observations."
        ),
        "parameters": {"type": "object", "properties": {
            "diff": {"type": "boolean", "description": (
                "true = also what changed since the previous session's snapshot (regime, D1 env, sector "
                "leaders/laggards, night read date, new econ events today).")},
        }, "required": []},
    },
}

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
INDEXES = ("SPY", "QQQ", "IWM")
BOARD_EACH_SIDE = 3
NIGHT_MAX_LINES = 6


@dataclass(frozen=True)
class Sources:
    """Where each section reads from; tests pass fakes, the app uses :func:`live_sources`."""

    auto_state: Callable[[], Mapping[str, Any]]
    d1_env: Callable[[date], str]
    regime_rows: Callable[[], list[Mapping[str, Any]]]
    night_read: Callable[[str], Mapping[str, Any] | None]
    econ: Callable[[str], Mapping[str, Any]]
    index_rrs: Callable[[], Mapping[str, float]]
    sector_board: Callable[[], Mapping[str, Any]]
    spy_pause: Callable[[str], Mapping[str, Any] | None]
    #: The plan file's hash (``plan_lines.plan_digest``), carried on the as-of row so a plan edit
    #: re-narrates the tape; None = not carried.
    plan_sha: Callable[[], str] | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _live_auto_state() -> dict[str, str]:
    from autopilot_core import read_auto_pilot_mode
    from project_paths import AUTOPILOT_STATE_FILE

    payload = _read_json(Path(AUTOPILOT_STATE_FILE))
    profile = str(payload.get("profile") or "") if isinstance(payload, dict) else ""
    return {"mode": read_auto_pilot_mode(Path(AUTOPILOT_STATE_FILE)), "profile": profile.strip().upper()}


def _live_night_read(session: str) -> Mapping[str, Any] | None:
    import market_regimes

    return market_regimes.latest_regime_read(on=session)


def _live_index_rrs() -> dict[str, float]:
    return {}  # no file-based index RRS exists yet; computing it from bars is not this pack's job


def _live_sector_board() -> dict[str, Any]:
    from industry_context import _read_csv_rows
    from project_paths import INDUSTRY_BOARD_STATE_FILE

    snapshot = _read_json(Path(INDUSTRY_BOARD_STATE_FILE))
    if not isinstance(snapshot, dict) or not snapshot.get("sector_path"):
        return {}
    return {"as_of": str(snapshot.get("last_success_at") or ""), "rows": _read_csv_rows(snapshot["sector_path"])}


def _live_spy_pause(session: str) -> Mapping[str, Any] | None:
    from project_paths import REGIME_PAUSE_OBSERVATIONS_FILE

    payload = _read_json(Path(REGIME_PAUSE_OBSERVATIONS_FILE))
    return payload if isinstance(payload, dict) else None


def live_sources() -> Sources:
    return Sources(
        auto_state=_live_auto_state,
        d1_env=context_pack._live_d1_env,
        regime_rows=context_pack._live_regime_rows,
        night_read=_live_night_read,
        econ=context_pack._live_econ,
        index_rrs=_live_index_rrs,
        sector_board=_live_sector_board,
        spy_pause=_live_spy_pause,
        plan_sha=_live_plan_sha,
    )


def _live_plan_sha() -> str:
    from mentor_packs.plan_lines import plan_digest

    return plan_digest()


def _unknown(row_id: str, what: str, exc: BaseException) -> dict[str, Any]:
    return {"id": row_id, "kind": "unknown", "text": f"{what}: unknown ({type(exc).__name__})"}


def _float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


_BRACKET_ID = re.compile(r"\s*\[[^\]]+\]")


def _sentences(paragraph: str) -> list[str]:
    """The read's sentences without the night model's own bracketed ids (they are foreign to the tape)."""
    clean = _BRACKET_ID.sub("", paragraph)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", clean) if part.strip()]
    return parts[:NIGHT_MAX_LINES]


def _econ_key(event: Mapping[str, Any], index: int) -> str:
    raw = str(event.get("id") or "").strip()
    if raw:
        return raw
    return f"{event.get('date')}-{index}"


#: P16: the app's ``app_state`` key for one PT day's tape snapshot (rows only), written once at first build.
SNAPSHOT_KEY = "tape:snapshot:{day}"


def snapshot_key(day: date | str) -> str:
    return SNAPSHOT_KEY.format(day=str(day)[:10])


def snapshot_clean(pack: Pack) -> bool:
    """A tape with every source read (no ``unknown`` row) is the only one worth keeping for tomorrow's diff."""
    return bool(pack.rows) and not any(row.get("kind") == "unknown" for row in pack.rows)


def snapshot_json(pack: Pack) -> str:
    """The rows of a built tape, as stored for tomorrow's diff (no as-of stamp)."""
    return json.dumps([row for row in pack.rows if row.get("kind") != "asof"], sort_keys=True, default=str)


def read_previous_snapshot(chat_db: Path | str | None, today: date) -> tuple[str, list[dict[str, Any]]] | None:
    """(day, rows) of the newest snapshot before ``today`` in the app's store (``mode=ro``); None when none."""
    if chat_db is None:
        return None
    from mentor_packs.journal_read import connect_ro

    try:
        conn = connect_ro(chat_db)
    except Exception:  # noqa: BLE001 - an unreadable store is no snapshot, never a guess
        return None
    if conn is None:
        return None
    try:
        rows = conn.execute("SELECT key, value FROM app_state WHERE key LIKE 'tape:snapshot:%' AND key < ? "
                            "ORDER BY key DESC LIMIT 1", (snapshot_key(today),)).fetchall()
    except Exception:  # noqa: BLE001
        return None
    finally:
        conn.close()
    if not rows:
        return None
    try:
        return str(rows[0]["key"]).rsplit(":", 1)[-1], list(json.loads(rows[0]["value"]))
    except (ValueError, TypeError):
        return None


def _previous_weekday(day: date) -> date:
    from datetime import timedelta

    back = day - timedelta(days=1)
    while back.weekday() >= 5:
        back -= timedelta(days=1)
    return back


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """An econ event by (date, time ET, label in lower-case words): its row id is positional and renumbers daily."""
    words = " ".join(re.findall(r"[a-z0-9]+", str(row.get("label") or "").lower()))
    return str(row.get("date") or "")[:10], str(row.get("time_et") or ""), words


def diff_rows(rows: list[dict[str, Any]], previous: tuple[str, list[dict[str, Any]]] | None,
              today: date) -> list[dict[str, Any]]:
    """P16: ``tape:diff:<k>`` rows, today's tape against the previous session's snapshot."""
    if previous is None:
        missing = _previous_weekday(today).isoformat()
        return [{"id": "tape:diff:none", "kind": "diff",
                 "text": f"Tape diff: no snapshot for {missing} (first day); what changed is unknown"}]
    day, old = previous
    now_by, old_by = {str(r.get("id")): r for r in rows}, {str(r.get("id")): r for r in old}
    out: list[dict[str, Any]] = []
    expected = _previous_weekday(today).isoformat()
    if day < expected:
        out.append({"id": "tape:diff:gap", "kind": "diff",
                    "text": f"Tape diff: no clean snapshot for {expected}; comparing with {day} instead"})

    def changed(key: str, what: str, before: str, after: str) -> None:
        same = before == after
        out.append({"id": f"tape:diff:{key}", "kind": "diff", "changed": not same, "before": before, "after": after,
                    "text": (f"Since {day}: {what} unchanged ({after})" if same
                             else f"Since {day}: {what} CHANGED from {before} to {after}")})

    changed("regime", "the trader's regime", str((old_by.get("tape:regime") or {}).get("regime") or "unknown"),
            str((now_by.get("tape:regime") or {}).get("regime") or "unknown"))
    changed("d1env", "the D1 environment", str((old_by.get("tape:d1env") or {}).get("label") or "unknown"),
            str((now_by.get("tape:d1env") or {}).get("label") or "unknown"))
    for side in ("leaders", "laggards"):
        before = ", ".join((old_by.get("tape:breadth") or {}).get(side) or ()) or "unknown"
        after = ", ".join((now_by.get("tape:breadth") or {}).get(side) or ()) or "unknown"
        changed(side, f"the sector {side}", before, after)

    def night_date(by: Mapping[str, Any]) -> str:
        return next((str(r.get("date")) for r in by.values() if r.get("kind") == "night" and r.get("date")), "none")

    changed("night", "the night read's date", night_date(old_by), night_date(now_by))
    old_econ = {_event_key(r) for r in old_by.values() if r.get("kind") == "econ" and r.get("label")}
    new_today = [r for r in now_by.values() if r.get("kind") == "econ" and r.get("label")
                 and _event_key(r) not in old_econ and str(r.get("date") or "") == today.isoformat()]
    out.append({"id": "tape:diff:econ", "kind": "diff", "changed": bool(new_today),
                "text": (f"Since {day}: new econ events today: " + "; ".join(str(r.get("text")) for r in new_today)
                         if new_today else f"Since {day}: no new econ events listed for today")})
    return out


def build(diff: bool = False, *, now: datetime | None = None, sources: Sources | None = None,
          chat_db: Path | str | None = None) -> Pack:
    """Build the regime pack. File and DB reads: call it on a worker.

    P16: ``diff`` adds ``tape:diff:<k>`` rows against the newest earlier snapshot in ``chat_db`` (``mode=ro``)."""
    moment = context_pack._now(now)
    src = sources or live_sources()
    local, market = moment.astimezone(PT), moment.astimezone(ET)
    session = market.date()
    rows: list[dict[str, Any]] = [
        {
            "id": "tape:asof",
            "kind": "asof",
            "text": f"Tape as of {local:%a %Y-%m-%d %H:%M} PT ({market:%H:%M} ET)",
            "at_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        }
    ]
    if src.plan_sha is not None:
        try:
            rows[0]["plan_sha"] = str(src.plan_sha())
        except Exception:  # noqa: BLE001 - an unreadable plan hashes as "unknown"
            rows[0]["plan_sha"] = "unknown"
    try:
        state = src.auto_state() or {}
        mode = str(state.get("mode") or "unknown")
        profile = str(state.get("profile") or "")
        text = f"Auto mode: {mode}" + (f" (profile {profile})" if profile and profile != mode else "")
        rows.append({"id": "tape:mode", "kind": "mode", "mode": mode, "profile": profile, "text": text})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:mode", "Auto mode", exc))
    try:
        label = src.d1_env(session) or "unknown"
        rows.append({"id": "tape:d1env", "kind": "d1env", "label": label,
                     "text": f"D1 environment (SPY, {session}): {label}"})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:d1env", "D1 environment", exc))
    try:
        import structural_regime

        current = structural_regime.current_regime(src.regime_rows(), local.date())
        if current:
            text = (f"Trader's regime: {current.get('label') or current.get('regime')} since "
                    f"{current.get('start_date')} (day {current.get('day_count')})")
            # P16: the day count moves daily; the diff compares the regime and its start date only.
            which = f"{current.get('label') or current.get('regime')} since {current.get('start_date')}"
        else:
            text, which = "Trader's regime: none typed yet", "none typed"
        rows.append({"id": "tape:regime", "kind": "regime", "regime": which, "text": text})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:regime", "Trader's regime", exc))
    try:
        read = src.night_read(session.isoformat())
        paragraph = str(((read or {}).get("read") or {}).get("paragraph") or "").strip()
        if not paragraph:
            rows.append({"id": "tape:night:none", "kind": "night", "text": "Night read: none on file"})
        else:
            day = str(read.get("session_date") or "")[:10] or "unknown date"
            for index, line in enumerate(_sentences(paragraph), start=1):
                rows.append({"id": f"tape:night:{index}", "kind": "night", "date": day,
                             "text": f"(night read, {day}) {line}"})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:night:none", "Night read", exc))
    try:
        view = src.econ(session.isoformat()) or {}
        events = context_pack.econ_events(view, session)
        if not events:
            note = str(view.get("note") or "no events listed")
            rows.append({"id": "tape:econ:none", "kind": "econ",
                         "text": f"Econ next {context_pack.ECON_DAYS_AHEAD} days: {note}"})
        seen: set[str] = set()
        for index, event in enumerate(events):
            key = _econ_key(event, index)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"id": f"tape:econ:{key}", "kind": "econ", "date": str(event.get("date") or ""),
                         "time_et": str(event.get("time_et") or ""), "label": str(event.get("label") or ""),
                         "text": f"Econ {event.get('date')} {event.get('time_et') or '--:--'} ET: {event.get('label')}"})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:econ:none", "Econ events", exc))
    try:
        rrs = src.index_rrs() or {}
        for symbol in INDEXES:
            value = _float(rrs.get(symbol))
            if value is not None:
                rows.append({"id": f"tape:rrs:{symbol}", "kind": "rrs", "symbol": symbol, "value": value,
                             "text": f"{symbol} rolling RRS: {value:+.2f}"})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:rrs:none", "Index RRS", exc))
    try:
        board = src.sector_board() or {}
        ranked = sorted(
            (row for row in board.get("rows") or () if _float(row.get("rs_rank")) is not None),
            key=lambda row: _float(row.get("rs_rank")) or 0.0,
        )
        if ranked:
            top, bottom = ranked[:BOARD_EACH_SIDE], ranked[-BOARD_EACH_SIDE:][::-1]

            def _names(group: list[Mapping[str, Any]]) -> str:
                return ", ".join(
                    f"{row.get('sector')} ({row.get('etf')}, 5d {(_float(row.get('return_5d_pct')) or 0.0):+.1f}%)"
                    for row in group
                )

            as_of = str(board.get("as_of") or "")[:16] or "unknown time"
            rows.append({"id": "tape:breadth", "kind": "breadth",
                         "leaders": [str(row.get("sector")) for row in top],
                         "laggards": [str(row.get("sector")) for row in bottom],
                         "text": f"Sector board ({as_of}): strongest {_names(top)}; weakest {_names(bottom)}"})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:breadth", "Sector board", exc))
    try:
        pause = src.spy_pause(session.isoformat())
        if pause and str(pause.get("date") or "")[:10] == session.isoformat():
            sides = pause.get("sides") or {}
            longs, shorts = len(sides.get("long") or {}), len(sides.get("short") or {})
            updated = str(pause.get("updated_at") or "")[11:16] or "--:--"
            rows.append({"id": "tape:spy:pause", "kind": "spy_pause",
                         "text": (f"SPY pause observations today (updated {updated} desk time): {longs} long and "
                                  f"{shorts} short names held up through a SPY pause")})
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown("tape:spy:pause", "SPY pause observations", exc))
    if diff in (True, "true", "True", 1, "1", "yes"):
        rows.extend(diff_rows(rows, read_previous_snapshot(chat_db, local.date()), local.date()))
    return make_pack(NAME, rows)


def push_line(pack: Pack, *, limit: int = 200) -> str:
    """One factual line from the pack only (mode, D1 env, next econ event, index RRS); never model text."""
    rows = {str(row.get("id")): row for row in pack.rows}
    parts: list[str] = []
    mode = rows.get("tape:mode")
    if mode and mode.get("kind") == "mode":
        parts.append(f"Auto {mode.get('mode')}")
    env = rows.get("tape:d1env")
    if env and env.get("kind") == "d1env":
        parts.append(f"D1 {env.get('label')}")
    econ = [row for row in pack.rows if row.get("kind") == "econ" and row.get("label")]
    if econ:
        first = min(econ, key=lambda row: (str(row.get("date")), str(row.get("time_et") or "99:99")))
        parts.append(f"next {first.get('label')} {str(first.get('date'))[5:]} {first.get('time_et') or ''}".rstrip())
    rrs = [f"{row['symbol']} {row['value']:+.1f}" for row in pack.rows if row.get("kind") == "rrs"]
    if rrs:
        parts.append("RRS " + " ".join(rrs))
    line = "Tape: " + " | ".join(parts) if parts else "Tape: nothing known yet"
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


def fixture_sources() -> Sources:
    return Sources(
        auto_state=lambda: {"mode": "DESK", "profile": "DESK"},
        d1_env=lambda day: "bearish_trend",
        regime_rows=lambda: [
            {"segment_id": 1, "start_date": "2026-09-01", "regime": "weak", "structure_note": "", "supersedes": None}
        ],
        night_read=lambda session: {
            "session_date": "2026-09-28",
            "read": {"paragraph": "SPY stayed below its D1 line. QQQ held the weekly trend. IWM was weakest."},
        },
        econ=lambda session: {
            "today": [{"id": "t1", "date": session, "time_et": "10:00", "label": "ISM Manufacturing"}],
            "week": [{"id": "w1", "date": "2026-10-02", "time_et": "08:30", "label": "Nonfarm payrolls"}],
        },
        index_rrs=lambda: {"SPY": 0.0, "QQQ": 0.42, "IWM": -1.3},
        sector_board=lambda: {
            "as_of": "2026-09-29T13:50:19",
            "rows": [
                {"etf": etf, "sector": name, "rs_rank": str(rank), "return_5d_pct": str(ret)}
                for rank, (etf, name, ret) in enumerate(
                    (("XLK", "Technology", -0.9), ("XLV", "Health Care", 0.5), ("XLU", "Utilities", 0.2),
                     ("XLF", "Financials", -1.1), ("XLE", "Energy", -2.4), ("XLB", "Materials", -3.0)),
                    start=1,
                )
            ],
        },
        spy_pause=lambda session: {"date": session, "updated_at": f"{session}T10:55:27",
                                   "sides": {"long": {"AI": {}, "ASML": {}}, "short": {"XOM": {}}}},
    )


def fixture() -> Pack:
    return build(now=datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc), sources=fixture_sources())
