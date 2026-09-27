"""Night slot: keep the lake's provider history current (P10, 2026-09-27).

After the Yahoo pass, a best-effort IB M30 top-up (client 1011, never in
market hours, skipped when TWS is unreachable) keeps an IBKR intraday basis fresh.

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


def _ib_topup(cli, lake, ib_fetcher) -> str:
    """Best-effort IB M30 top-up after Yahoo's; any failure is a note, never the slot's status."""
    try:
        report = cli.run_history_ib_topup(lake, fetcher=ib_fetcher, log=_log.info)
    except Exception as exc:  # noqa: BLE001 - IB is optional for the night
        _log.exception("IB M30 top-up failed")
        return f"; IB M30 top-up failed ({type(exc).__name__})"
    added = int((report.get("rows_published") or {}).get("bar_m30", 0) or 0)
    if report.get("status") in {"OK", "PARTIAL"} or added:
        return f"; IB M30 top-up +{added}"
    notes = "; ".join(str(note) for note in report.get("notes") or []) or report.get("status", "")
    return f"; IB M30 top-up skipped ({notes})"


def run_lake_history_topup(
    *, store: Any = None, client: Any = None, ib_fetcher: Any = None, **_ignored: Any
) -> dict[str, Any]:
    """Yahoo top-up, then IB M30. An injected Yahoo ``client`` (tests) skips IB unless ``ib_fetcher`` is given."""
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
    ib_note = _ib_topup(cli, lake, ib_fetcher) if (ib_fetcher is not None or client is None) else ""
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
    reason += ib_note
    return {"status": "ok", "model": "", "reason": reason, "outputs": []}
