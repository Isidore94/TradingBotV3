"""Night slot: keep the lake's provider history current (P10, 2026-09-27).

One owner for the daily top-up: D1 for the last ~10 sessions (plus full pulls
for new names and split re-pulls), H1 (+H4) and M30 for the last 5 days, a
slice of the weekly earnings-date refresh, then the series checks. Network
only, no model, no desk: it runs with the desk down. Idempotent, so a retry or
a second run the same night adds nothing.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def run_lake_history_topup(*, store: Any = None, client: Any = None, **_ignored: Any) -> dict[str, Any]:
    try:
        from research_warehouse import cli
        from research_warehouse.store import ResearchStore

        lake = store if store is not None else ResearchStore.open()
        if lake is None:
            return {"status": "ok", "model": "", "reason": "research lake not configured; nothing to top up", "outputs": []}
        report = cli.run_history_topup(lake, client=client, log=_log.info)
    except Exception as exc:  # noqa: BLE001 - the night goes on; the lake keeps what it had
        _log.exception("lake_history_topup failed")
        return {"status": "failed", "model": "", "reason": f"history top-up failed ({type(exc).__name__}: {exc})", "outputs": []}
    published = {}
    for kind in ("d1", "h1", "m30", "earnings", "quality"):
        for dataset, rows in (report.get(kind, {}).get("rows_published") or {}).items():
            published[dataset] = published.get(dataset, 0) + int(rows or 0)
    reason = "history top-up: " + (", ".join(f"{name} +{rows}" for name, rows in sorted(published.items())) or "nothing new")
    repulled = sum(len(report.get(kind, {}).get("repulled") or []) for kind in ("d1", "h1", "m30"))
    if repulled:
        reason += f"; {repulled} split/basis re-pulls"
    if report.get("status") != "OK":
        reason += "; some provider batches failed (retried tomorrow)"
    return {"status": "ok", "model": "", "reason": reason, "outputs": []}
