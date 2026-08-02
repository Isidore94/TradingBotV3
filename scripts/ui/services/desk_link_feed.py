"""Satellite desk feed: drive the FULL Trading Desk UI from Desk Link.

Runs on a satellite machine instead of (not alongside) the standalone
satellite window: relayed alert popups are reconstructed into real
BounceAlert objects and injected into the Alert Center exactly where the
live bot's alerts enter, so the desk behaves as if it were connected to
the API. Chart data comes from two places: D1 from the Drive-synced
durable store (already on this machine), M5 from the relayed payloads via
``payload_bot()`` — a stand-in for the live bot's in-memory cache.

This service never scans, never talks to TWS, and never re-relays (the
local Desk Link server is not serving in this mode).
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal

from desk_link import protocol
from desk_link.client import DeskLinkClient
from desk_link.popup_payload import restore_alert_popup_payload
from ui.models.bounce import BounceAlert

log = logging.getLogger(__name__)

_M5_CACHE_SYMBOLS = 200


class _PayloadBot:
    """Duck-types the one bot method the chart surfaces use."""

    def __init__(self) -> None:
        self._m5: dict[str, list[dict[str, Any]]] = {}

    def store(self, symbol: str, bars: list[dict[str, Any]]) -> None:
        symbol = str(symbol or "").strip().upper()
        if not symbol or not bars:
            return
        self._m5[symbol] = bars
        while len(self._m5) > _M5_CACHE_SYMBOLS:
            self._m5.pop(next(iter(self._m5)))

    def m5_chart_bars(self, symbol: str, max_sessions: int = 2) -> list[dict[str, Any]]:
        bars = self._m5.get(str(symbol or "").strip().upper()) or []
        return [dict(bar) for bar in bars]


class DeskLinkFeedService(QObject):
    """Owns the satellite-desk DeskLinkClient and republishes desk inputs."""

    alertReceived = Signal(object)  # BounceAlert, same contract as BounceService
    linkStatusChanged = Signal(str, str)  # (state, detail)

    _messageArrived = Signal(dict)  # client thread -> GUI thread bridge

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._client: DeskLinkClient | None = None
        self._payload_bot = _PayloadBot()
        self._messageArrived.connect(self._handle_message)

    def payload_bot(self) -> _PayloadBot:
        return self._payload_bot

    @property
    def running(self) -> bool:
        return self._client is not None

    def start(self, *, host: str, port: int, token: str, machine_name: str) -> None:
        if self._client is not None:
            return
        self._client = DeskLinkClient(
            host=host,
            port=port,
            token=token,
            machine_name=machine_name,
            on_message=self._messageArrived.emit,
            on_status=self.linkStatusChanged.emit,
        )
        self._client.start()

    def stop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.stop()

    def _handle_message(self, message: dict) -> None:
        if message.get("type") != protocol.TYPE_ALERT_POPUP:
            return
        payload = message.get("payload") or {}
        try:
            restored = restore_alert_popup_payload(payload)
        except ValueError as exc:
            log.warning("Satellite desk dropped an incompatible alert payload: %s", exc)
            return
        # Cache chart data first so the Alert Center's render finds it.
        symbol = str(restored["alert"].get("symbol") or "").strip().upper()
        m5_bars = (restored.get("m5") or {}).get("bars") or []
        self._payload_bot.store(symbol, m5_bars)
        self.alertReceived.emit(_rebuild_alert(restored["alert"]))


def _rebuild_alert(fields: dict[str, Any]) -> BounceAlert:
    """BounceAlert from its asdict() wire form, ignoring unknown keys so a
    newer main can add fields without breaking an older satellite."""
    known = {f.name for f in BounceAlert.__dataclass_fields__.values()}
    kwargs = {key: value for key, value in fields.items() if key in known}
    kwargs.setdefault("time_text", "")
    payload = kwargs.get("payload")
    if not isinstance(payload, dict):
        kwargs["payload"] = {}
    return BounceAlert(**kwargs)
