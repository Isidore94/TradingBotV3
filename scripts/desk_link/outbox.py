"""Satellite intent outbox: journal first, send second, ack third.

Every decision made on a satellite is appended to a machine-local JSONL
journal BEFORE it goes on the wire, so a Wi-Fi drop or main-desk restart
can never lose one: unacked intents are resent when control is regained.
The applied actions are idempotent on the main (remove-for-day / focus
add / focus remove), which makes at-least-once delivery safe.

Journal rows: {"kind": "intent", "seq": n, ...payload} and
{"kind": "ack", "seq": n}. Sequence numbers continue across restarts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from desk_link.protocol import utc_now_iso


class IntentOutbox:
    def __init__(self, journal_path: Path) -> None:
        self._path = Path(journal_path)
        self._pending: dict[int, dict[str, Any]] = {}
        self._next_seq = 1
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            seq = row.get("seq")
            if not isinstance(seq, int):
                continue
            self._next_seq = max(self._next_seq, seq + 1)
            if row.get("kind") == "intent":
                payload = {key: value for key, value in row.items() if key != "kind"}
                self._pending[seq] = payload
            elif row.get("kind") == "ack":
                self._pending.pop(seq, None)

    def _append(self, row: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        except OSError:
            # The in-memory pending set still guards this session; only
            # crash-recovery durability is degraded.
            pass

    def create(self, action: str, symbol: str, **extra: Any) -> dict[str, Any]:
        """Journal a new intent and return the wire payload."""
        intent = {
            "seq": self._next_seq,
            "action": str(action),
            "symbol": str(symbol or "").strip().upper(),
            "ts": utc_now_iso(),
            **extra,
        }
        self._next_seq += 1
        self._pending[intent["seq"]] = intent
        self._append({"kind": "intent", **intent})
        return intent

    def mark_acked(self, seq: Any) -> None:
        if isinstance(seq, int) and seq in self._pending:
            self._pending.pop(seq)
            self._append({"kind": "ack", "seq": seq})

    def unacked(self) -> list[dict[str, Any]]:
        return [self._pending[seq] for seq in sorted(self._pending)]
