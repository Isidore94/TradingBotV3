"""Gate pack: what the desk knows before one trade the trader is about to take. Read-only.

Sections, every row under ``gate:<SYM>:``: the request as typed (``req``); the dollar
risk against the Settings > General "Risk per trade ($)" value (``risk``); the pick
pack's rows and the regime pack's mode / D1 environment / regime / SPY pause rows,
embedded VERBATIM with ``gate:<SYM>:`` put in front of their own ids (so
``pick:NVDA:cell`` becomes ``gate:NVDA:pick:NVDA:cell``; the pick pack's own plan rows
are left out because the plan has its own section); the open book (``book:<TRADE_ID>``
and ``book:industry``); and the plan lines (``plan:<line>``). It never sizes, orders or
writes. A source that cannot be read gives an "unknown" row, never a guess.
"""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from mentor_packs import pick_pack, plan_lines, regime_pack
from mentor_packs.registry import Pack, make_pack

NAME = "gate_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The pre-trade check for one trade the trader is about to take: his request, the dollar risk "
            "against his risk-per-trade setting, the pick pack and tape for the name, his open book "
            "(same symbol, same industry), and his plan lines. Advice only; never sizes or orders."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "side": {"type": "string", "description": "LONG or SHORT."},
                "symbol": {"type": "string", "description": "Ticker, e.g. NVDA."},
                "size": {"type": "number", "description": "Shares, if typed."},
                "stop": {"type": "number", "description": "Stop price, if typed."},
                "entry": {"type": "number", "description": "Entry price, if typed."},
            },
            "required": ["side", "symbol"],
        },
    },
}

#: The regime pack's row kinds the gate carries.
TAPE_KINDS = frozenset({"mode", "d1env", "regime", "spy_pause"})
#: Their ids, so an "unknown" row in their place is carried too.
TAPE_IDS = frozenset({"tape:mode", "tape:d1env", "tape:regime", "tape:spy:pause"})
NOT_SET_TEXT = "risk per trade not set (Settings > General)"
_VOLATILE_KINDS = frozenset({"asof"})


@dataclass(frozen=True)
class Sources:
    """Where each section reads from; tests pass fixtures, the app uses :func:`live_sources`."""

    risk_setting: Callable[[], Any]
    open_trades: Callable[[], list[Mapping[str, Any]]]
    industry_map: Callable[[], Mapping[str, Mapping[str, Any]]]
    pick_paths: pick_pack.PickPaths | None = None
    regime_sources: regime_pack.Sources | None = None
    plan: Path | None = None
    #: Tests pin the pack's clock; None = now.
    now: datetime | None = field(default=None, compare=False)


def _live_risk() -> Any:
    from entry_plan import RISK_SETTING
    from project_paths import get_local_setting

    return get_local_setting(RISK_SETTING, None)


def _live_open_trades() -> list[Mapping[str, Any]]:
    from project_paths import JOURNAL_DB_FILE

    return read_open_trades(Path(JOURNAL_DB_FILE))


def read_open_trades(path: Path) -> list[dict[str, Any]]:
    """Open journal trades with their planned stop (``mode=ro``; the store class is never built)."""
    if not path.exists():
        return []
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
    try:
        conn.row_factory = sqlite3.Row
        has_annotations = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trade_annotations'"
        ).fetchone()
        stop_sql = "a.planned_stop" if has_annotations else "NULL"
        join = "LEFT JOIN trade_annotations a ON a.trade_id = t.trade_id" if has_annotations else ""
        sql = (
            "SELECT t.trade_id, t.symbol, t.direction, t.quantity_opened, t.quantity_closed, "
            f"t.average_entry_price, t.opened_at, {stop_sql} AS planned_stop FROM trades t {join} "
            "WHERE t.status = 'OPEN' ORDER BY t.opened_at, t.trade_id"
        )
        return [{key: row[key] for key in row.keys()} for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _live_industry_map() -> Mapping[str, Mapping[str, Any]]:
    from industry_context import load_industry_context_map

    return load_industry_context_map()


def live_sources() -> Sources:
    return Sources(risk_setting=_live_risk, open_trades=_live_open_trades, industry_map=_live_industry_map)


# ---------------------------------------------------------------- small helpers
def _sym(value: Any) -> str:
    return str(value or "").strip().upper()


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return {"BUY": "LONG", "SELL": "SHORT"}.get(text, text if text in ("LONG", "SHORT") else "")


def _num(value: Any) -> float | None:
    from entry_plan import _number

    return _number(value)


def _fmt(value: float | None, money: bool = False) -> str:
    if value is None:
        return "not given"
    text = f"{value:,.2f}".rstrip("0").rstrip(".") if not money else f"${value:,.2f}"
    return text


def _unknown(row_id: str, what: str, exc: BaseException) -> dict[str, Any]:
    return {"id": row_id, "kind": "unknown", "text": f"{what}: unknown ({type(exc).__name__})"}


def _industry(symbol: str, imap: Mapping[str, Mapping[str, Any]]) -> str:
    return str((imap.get(symbol) or {}).get("industry") or "").strip()


def request_key(side: Any, symbol: Any, size: Any = None, stop: Any = None, entry: Any = None) -> str:
    """The request as one stable string: two different requests never share a card."""
    parts = [_side(side), _sym(symbol)] + [("" if _num(v) is None else repr(_num(v))) for v in (size, stop, entry)]
    return "|".join(parts)


# ---------------------------------------------------------------- sections
def _risk_row(prefix: str, side: str, size: float | None, stop: float | None, entry: float | None,
              src: Sources) -> dict[str, Any]:
    from entry_plan import parse_risk_dollars

    setting = parse_risk_dollars(src.risk_setting())
    row: dict[str, Any] = {"id": f"{prefix}:risk", "kind": "risk", "setting": setting}
    at_risk = abs(entry - stop) * size if None not in (entry, stop, size) else None
    row["at_risk"] = None if at_risk is None else round(at_risk, 2)
    wrong_side = (entry is not None and stop is not None
                  and ((side == "LONG" and stop >= entry) or (side == "SHORT" and stop <= entry)))
    parts = [f"Risk per trade setting: {_fmt(setting, money=True)}" if setting is not None else NOT_SET_TEXT]
    if at_risk is None:
        parts.append("$ at risk: unknown (needs entry, stop and size)")
    else:
        parts.append(f"$ at risk: |{_fmt(entry)} - {_fmt(stop)}| x {_fmt(size)} = ${at_risk:,.2f}")
        if setting is not None:
            row["ratio"] = round(at_risk / setting, 4)
            parts.append(f"= {at_risk / setting:.2f}x the setting")
    if wrong_side:
        row["wrong_side"] = True
        parts.append(f"the stop is on the wrong side of the entry for a {side.lower()}")
    row["text"] = "; ".join(parts)
    return row


def _book_rows(prefix: str, symbol: str, side: str, src: Sources) -> list[dict[str, Any]]:
    trades = list(src.open_trades() or ())
    try:
        imap = src.industry_map() or {}
    except Exception:  # noqa: BLE001 - industry is a label; unreadable = unknown
        imap = {}
    own_industry = _industry(symbol, imap)
    rows: list[dict[str, Any]] = []
    same_industry: list[str] = []
    for trade in trades:
        sym = _sym(trade.get("symbol"))
        trade_side = _side(trade.get("direction"))
        qty = (_num(trade.get("quantity_opened")) or 0.0) - (_num(trade.get("quantity_closed")) or 0.0)
        entry, stop = _num(trade.get("average_entry_price")), _num(trade.get("planned_stop"))
        industry = _industry(sym, imap)
        at_risk = round(abs(entry - stop) * qty, 2) if entry is not None and stop is not None and qty else None
        same_symbol = sym == symbol
        text = (f"Open {trade_side or '?'} {sym} {qty:g} sh @ {_fmt(entry)}, industry {industry or 'unknown'}, "
                + (f"$ at risk ${at_risk:,.2f} (stop {_fmt(stop)})" if at_risk is not None else "no stop on file"))
        if same_symbol:
            text += " -- SAME SYMBOL as this request"
        if own_industry and industry == own_industry:
            same_industry.append(sym)
        rows.append({"id": f"{prefix}:book:{trade.get('trade_id')}", "kind": "book_trade", "symbol": sym,
                     "side": trade_side, "industry": industry, "at_risk": at_risk,
                     "same_symbol": same_symbol, "text": text})
    if own_industry:
        names = sorted(set(same_industry))
        text = (f"Open book: {len(trades)} open trade(s); {len(names)} open name(s) in {own_industry}"
                + (f" ({', '.join(names)})" if names else ""))
    else:
        names = []
        text = f"Open book: {len(trades)} open trade(s); {symbol}'s industry unknown"
    rows.append({"id": f"{prefix}:book:industry", "kind": "book_industry", "industry": own_industry,
                 "same_industry_count": len(names), "open_count": len(trades),
                 "same_symbol_open": any(row.get("same_symbol") for row in rows), "text": text})
    return rows


def _plan_rows(prefix: str, src: Sources) -> list[dict[str, Any]]:
    plan = plan_lines.build(path=src.plan)
    if not plan.rows:
        return [{"id": f"{prefix}:plan:none", "kind": "plan_empty", "text": f"Plan: {plan.empty_text or plan_lines.EMPTY_TEXT}"}]
    return [{"id": f"{prefix}:{line['id']}", "kind": "plan_line", "plan_id": str(line["id"]),
             "text": f"Plan [{line['id']}]: {line.get('text', '')}"} for line in plan.rows]


def _embed(prefix: str, rows: Any) -> list[dict[str, Any]]:
    return [{**dict(row), "id": f"{prefix}:{row['id']}", "source_id": str(row["id"])} for row in rows]


# ---------------------------------------------------------------- build
def build(side: str = "", symbol: str = "", size: Any = None, stop: Any = None, entry: Any = None, *,
          now: datetime | None = None, sources: Sources | None = None) -> Pack:
    """Build the gate pack for one request. File and DB reads: call it on a worker."""
    sym, chosen = _sym(symbol), _side(side)
    if not sym or not sym.replace(".", "").replace("-", "").isalnum():
        return make_pack(NAME, (), empty_text="gate_pack needs a ticker, e.g. /check short NVDA")
    if not chosen:
        return make_pack(NAME, (), empty_text="gate_pack needs a side: LONG or SHORT")
    src = sources or live_sources()
    moment = pick_pack._now(now or src.now)
    prefix = f"gate:{sym}"
    size_v, stop_v, entry_v = _num(size), _num(stop), _num(entry)
    rows: list[dict[str, Any]] = [{
        "id": f"{prefix}:req", "kind": "request", "side": chosen, "symbol": sym,
        "size": size_v, "stop": stop_v, "entry": entry_v, "key": request_key(chosen, sym, size, stop, entry),
        "at_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "text": (f"Request: {chosen} {sym}, size {_fmt(size_v)}, stop {_fmt(stop_v)}, entry {_fmt(entry_v)}"),
    }]
    try:
        rows.append(_risk_row(prefix, chosen, size_v, stop_v, entry_v, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"{prefix}:risk", "Risk", exc))
    try:
        pick = pick_pack.build(sym, chosen, now=moment, paths=src.pick_paths)
        keep = [row for row in pick.rows if row.get("kind") not in ("plan_line", "plan_empty")]
        rows.extend(_embed(prefix, keep) if keep else [{"id": f"{prefix}:pick:none", "kind": "pick_empty",
                                                         "text": f"Pick: {pick.empty_text or 'nothing'}"}])
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"{prefix}:pick:none", "Pick pack", exc))
    try:
        tape = regime_pack.build(now=moment, sources=src.regime_sources)
        rows.extend(_embed(prefix, [row for row in tape.rows
                                    if row.get("kind") in TAPE_KINDS or str(row.get("id")) in TAPE_IDS]))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"{prefix}:tape:none", "Tape", exc))
    try:
        rows.extend(_book_rows(prefix, sym, chosen, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"{prefix}:book:industry", "Open book", exc))
    try:
        rows.extend(_plan_rows(prefix, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"{prefix}:plan:none", "Plan", exc))
    return make_pack(NAME, rows)


def pack_hash(pack: Pack) -> str:
    """What the pack SAYS, the request row included (so two requests never share a card)."""
    import hashlib
    import json

    stable = [(row.get("id"), row.get("text")) for row in pack.rows if row.get("kind") not in _VOLATILE_KINDS]
    body = json.dumps({"name": pack.name, "rows": stable, "empty": pack.empty_text}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def plan_ids(pack: Pack) -> set[str]:
    """The plan ids (``plan:...``) the pack carried."""
    return {str(row["plan_id"]) for row in pack.rows if row.get("kind") == "plan_line"}


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = pick_pack.FIXTURE_NOW


def fixture_sources(root: Path | str, *, risk: Any = 100, trades: list[Mapping[str, Any]] | None = None,
                    plan_text: str | None = None) -> Sources:
    """The pick pack's fixture world plus a set risk, an open book and the regime fixture."""
    paths = pick_pack.write_fixture_world(root, plan_text=plan_text)
    book = [
        {"trade_id": "T1", "symbol": "AMD", "direction": "SHORT", "quantity_opened": 100, "quantity_closed": 0,
         "average_entry_price": 150.0, "planned_stop": 152.5, "opened_at": "2026-09-28T07:00:00-07:00"},
    ] if trades is None else list(trades)
    return Sources(risk_setting=lambda: risk, open_trades=lambda: book,
                   industry_map=paths.industry_map or (lambda: {}), pick_paths=paths,
                   regime_sources=regime_pack.fixture_sources(), plan=paths.plan)


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build("SHORT", "NVDA", 400, 3.20, 3.05, now=FIXTURE_NOW, sources=fixture_sources(tmp))
