from __future__ import annotations

import traceback
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class _LookupWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, symbol: str, days_ahead: int) -> None:
        super().__init__()
        self._symbol = symbol
        self._days_ahead = days_ahead

    @Slot()
    def run(self) -> None:
        try:
            from market_prep.services.ticker_lookup_service import lookup_ticker_context

            payload = lookup_ticker_context(self._symbol, days_ahead=self._days_ahead)
            self.finished.emit(payload if isinstance(payload, dict) else {})
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class TickerLookupService(QObject):
    started = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _LookupWorker | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def lookup(self, symbol: str, days_ahead: int) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol or self.running:
            return

        thread = QThread(self)
        worker = _LookupWorker(symbol, days_ahead)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.finished)
        worker.failed.connect(self.failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_thread)

        self._thread = thread
        self._worker = worker
        self.started.emit(symbol)
        thread.start()

    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)

    @Slot()
    def _clear_thread(self) -> None:
        self._thread = None
        self._worker = None


def format_earnings_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scope, key in (("Target", "target_earnings"), ("Peer", "peer_earnings")):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "scope": scope,
                    "date": str(item.get("date") or item.get("report_date") or ""),
                    "ticker": str(item.get("ticker") or "").upper(),
                    "company": str(item.get("company") or item.get("name") or ""),
                    "time": str(item.get("time") or item.get("session") or ""),
                    "importance": str(item.get("importance") or item.get("risk") or ""),
                }
            )
    return rows


def format_headlines(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for label, key in (("Ticker headlines", "target_headlines"), ("Industry headlines", "industry_headlines")):
        items = payload.get(key)
        if not isinstance(items, list) or not items:
            continue
        chunks.append(f"== {label} ==")
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("headline") or "").strip()
            source = str(item.get("source") or "").strip()
            published = str(item.get("published") or item.get("date") or "").strip()
            meta = " · ".join(part for part in (source, published) if part)
            chunks.append(f"• {title}" + (f"  [{meta}]" if meta else ""))
        chunks.append("")
    return "\n".join(chunks).strip() or "No headlines returned."


def lookup_summary(payload: dict[str, Any]) -> str:
    ticker = str(payload.get("ticker") or "").upper()
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    company = str(metadata.get("company") or metadata.get("name") or "")
    sector = str(metadata.get("sector") or metadata.get("industry") or "")
    peer_reason = str(payload.get("peer_reason") or "")
    parts = [part for part in (ticker, company, sector) if part]
    summary = " · ".join(parts) if parts else (ticker or "No ticker loaded.")
    if peer_reason:
        summary += f"\nPeer context: {peer_reason}"
    return summary
