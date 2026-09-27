"""The `market_regime_daily` night slot (P10): deterministic, no model.

Appends the missing point-in-time regime rows for SPY/QQQ/IWM to the research
lake's `market_regime_daily` dataset from the lake's provider history
(`research_warehouse.history_reader`). Idempotent: a row already in the lake is
never rewritten, and a session whose D1 bar is not in the lake yet is simply
picked up on a later night. It holds the research build's single-flight lock.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)


def run_market_regime_daily(
    *,
    session_date: str = "",
    store: Any = None,
    now: datetime | None = None,
    lock_path: Path | None = None,
    d1_loader: Callable | None = None,
    intraday_loader: Callable | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    try:
        from research_warehouse import cli
        from research_warehouse.store import ResearchStore

        lake = store if store is not None else ResearchStore.open()
        if lake is None:
            return {"status": "ok", "model": "", "reason": "research lake not configured; nothing to do", "outputs": []}
        until = date.fromisoformat(str(session_date)[:10]) if session_date else None
        report = cli.run_build_regimes(
            lake, apply=True, until=until, now=now, lock_path=lock_path,
            d1_loader=d1_loader, intraday_loader=intraday_loader,
        )
    except Exception as exc:  # noqa: BLE001 - the night goes on; the lake keeps what it had
        _log.exception("market_regime_daily: build failed")
        return {
            "status": "failed", "model": "",
            "reason": f"regime rows not built ({type(exc).__name__}: {exc}); lake unchanged",
            "outputs": [],
        }
    status = report.get("status")
    if status == "REFUSED":
        return {"status": "ok", "model": "", "reason": f"lake busy ({report.get('reason')}); rows wait for the next night",
                "outputs": []}
    if status not in {"OK", "PARTIAL"}:
        return {"status": "failed", "model": "", "reason": f"regime build {status}: {report.get('message', '')}",
                "outputs": []}
    written = int(report.get("rows_published") or 0)
    reason = f"{written} regime rows appended through {report.get('last_session')}"
    if report.get("held"):
        reason += f"; holding {len(report['held'])} until their ^VIX/H1 bars land"
    if report.get("rows_quarantined"):
        reason += f"; {report['rows_quarantined']} rows quarantined"
    return {"status": "ok", "model": "", "reason": reason, "outputs": ["market_regime_daily"] if written else []}
