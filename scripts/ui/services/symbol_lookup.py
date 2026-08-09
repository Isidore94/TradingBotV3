"""Look up any symbol for review, without ever joining a watchlist.

The Chart Review workspace has to open names the scans never surfaced - a
symbol somebody mentioned, a name off a screener, anything the trader wants a
look at. That is a strictly *read* act, and the distinction is load-bearing:

    plan.md sec 5 - user-entered watchlist names are never auto-removed.

The mirror of that invariant is the one this module enforces. Looking at a
name must never add it either. A lookup that quietly wrote to longs.txt or the
CandidateRegistry would put symbols into the scan universe that the trader
never chose, and the next writer to reconcile those files would be deciding
what to do about entries nobody made. So this module has no writer
dependencies at all: it imports no watchlist path, no registry, no focus
store, and the only thing it persists is a machine-local recents list under
%LOCALAPPDATA% - never the shared home where the watchlists live.

Adding a looked-up name to a watchlist or focus list stays an explicit,
separate trader action through the surfaces that already own those files.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from project_paths import LOCAL_SETTINGS_DIR

#: Tickers, class shares (BRK.B), and the odd hyphenated listing. Deliberately
#: strict: this string reaches provider requests and filenames.
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9]{0,9}(?:[.\-][A-Z0-9]{1,4})?$")

#: Machine-local on purpose - a per-machine convenience cache, not shared
#: state, and categorically not next to the watchlists.
RECENT_LOOKUPS_FILE = LOCAL_SETTINGS_DIR / "chart_review_recent_lookups.json"

MAX_RECENT_LOOKUPS = 12


def normalize_symbol(text: object) -> str:
    """The symbol a lookup would open, or "" when it is not a symbol at all."""
    candidate = str(text or "").strip().upper().lstrip("$")
    # Only the surrounding whitespace comes off. Stripping spaces *inside* the
    # text would turn "not a symbol" into the perfectly valid-looking ticker
    # NOTASYMBOL and happily open a chart for it.
    return candidate if _SYMBOL_RE.fullmatch(candidate) else ""


def is_lookupable(text: object) -> bool:
    return bool(normalize_symbol(text))


class RecentLookups:
    """Most-recent-first symbol history, persisted machine-locally.

    Deliberately not a watchlist: nothing reads this file to decide what to
    scan, alert on, or track. It exists so returning to a name the trader
    looked at ten minutes ago is one keystroke.
    """

    def __init__(self, path: Path = RECENT_LOOKUPS_FILE, *, limit: int = MAX_RECENT_LOOKUPS) -> None:
        self._path = Path(path)
        self._limit = max(1, int(limit))
        self._symbols: list[str] = []
        self.reload()

    @property
    def path(self) -> Path:
        return self._path

    def symbols(self) -> list[str]:
        return list(self._symbols)

    def reload(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        symbols: list[str] = []
        if isinstance(payload, dict):
            raw = payload.get("symbols")
            if isinstance(raw, list):
                for entry in raw:
                    symbol = normalize_symbol(entry)
                    if symbol and symbol not in symbols:
                        symbols.append(symbol)
        self._symbols = symbols[: self._limit]

    def remember(self, symbol: object) -> str:
        """Move ``symbol`` to the front. Returns the normalized symbol or ""."""
        resolved = normalize_symbol(symbol)
        if not resolved:
            return ""
        self._symbols = [resolved] + [s for s in self._symbols if s != resolved]
        del self._symbols[self._limit :]
        self._save()
        return resolved

    def clear(self) -> None:
        self._symbols = []
        self._save()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps({"symbols": self._symbols}, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self._path)
        except OSError:
            pass  # a convenience cache never blocks a lookup
