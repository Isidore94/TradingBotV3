"""The `market_regime_table` night slot (S17 half 1): deterministic, no model.

Appends one `market_regimes.session_row` per (session, symbol) for the last
`BACKFILL_SESSIONS` sessions that are not in `MARKET_REGIME_TABLE_FILE` yet.
A session waits until SPY's D1 store holds its bar (the day is recorded).
Rows already in the file are never rewritten; a failure appends nothing and
leaves the file as it was. M5 comes from the lake first, then the cached Day
Review Yahoo store; each row names its `m5_source` (lake/day_review_yahoo/none).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

_log = logging.getLogger(__name__)

BACKFILL_SESSIONS = 20


def _recent_sessions(last: date, count: int) -> list[date]:
    from research_warehouse import exchange_calendar as xcal

    start = last - timedelta(days=count * 2 + 10)
    return [session.session_date for session in xcal.sessions_between(start, last)][-count:]


def run_market_regime_table(
    *,
    session_date: str = "",
    out_path: Any = None,
    loader: Callable[..., tuple[dict, dict, str]] | None = None,
    sessions: int = BACKFILL_SESSIONS,
    symbols: Sequence[str] | None = None,
    now: datetime | None = None,
    day_review_root: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    import market_regimes as mr

    try:
        if out_path is None:
            from project_paths import MARKET_REGIME_TABLE_FILE

            out_path = MARKET_REGIME_TABLE_FILE
        out = Path(out_path)
        last = date.fromisoformat(str(session_date)[:10]) if session_date else datetime.now(mr.MARKET_TZ).date()
        wanted = tuple(symbols or mr.table_symbols())
        days = _recent_sessions(last, max(1, int(sessions)))
        have = {(str(row.get("session_date")), str(row.get("symbol") or "").upper()) for row in mr.read_table(out)}
        missing = [(day, symbol) for day in days for symbol in wanted if (day.isoformat(), symbol) not in have]
        if not missing:
            return {"status": "ok", "model": "", "reason": "nothing new: every session is in the table", "outputs": []}
        first = min(day for day, _symbol in missing)
        d1_by_symbol, m5_by_symbol, source = (loader or mr.load_bars)(
            wanted, wanted, m5_since=first - mr.INTRADAY_LOOKBACK
        )
        # QQQ/IWM have no lake M5: fill those sessions from the cached Day Review store.
        from research_warehouse import exchange_calendar as xcal

        m5_by_symbol = {symbol: list(m5_by_symbol.get(symbol) or ()) for symbol in wanted}
        m5_days = [session.session_date for session in xcal.sessions_between(first - mr.INTRADAY_LOOKBACK, last)]
        m5_sources = mr.fill_m5_from_day_review(m5_by_symbol, wanted, m5_days, root=day_review_root, now=now)
        recorded = {mr.session_day(row) for row in d1_by_symbol.get(mr.PRIMARY) or ()}
        stamp = (now or datetime.now(mr.MARKET_TZ)).astimezone(mr.MARKET_TZ)
        rows = [
            {
                **mr.session_row(
                    symbol, day,
                    d1_rows=d1_by_symbol.get(symbol) or (),
                    m5_rows=m5_by_symbol.get(symbol) or (),
                    computed_at=stamp,
                ),
                "m5_source": m5_sources.get((day.isoformat(), symbol), mr.M5_SOURCE_NONE),
            }
            for day, symbol in missing
            if day in recorded
        ]
        waiting = sorted({day.isoformat() for day, _symbol in missing if day not in recorded})
        written = mr.append_rows(out, rows)
    except Exception as exc:  # noqa: BLE001 - the night goes on; the table keeps what it had
        _log.exception("market_regime_table: build failed")
        return {
            "status": "failed", "model": "",
            "reason": f"regime table not built ({type(exc).__name__}: {exc}); last file kept",
            "outputs": [],
        }
    reason = f"{written} rows appended ({len(wanted)} symbols, bars {source})"
    if waiting:
        reason += f"; waiting for the D1 bar of {', '.join(waiting)}"
    return {"status": "ok", "model": "", "reason": reason, "outputs": [str(out)] if written else []}
