"""Small helpers for parsing ticker watchlists."""

from __future__ import annotations

import re
from pathlib import Path

WATCHLIST_SYMBOL_RE = re.compile(r"[A-Z0-9.\-]+")
TC2000_HEADER_PREFIX = "SYMBOLS FROM TC2000"


def extract_watchlist_symbols(raw_text: str, *, skip_tokens: set[str] | None = None) -> list[str]:
    """Return ordered, de-duplicated symbols from pasted or file watchlist text."""
    symbols: list[str] = []
    seen: set[str] = set()
    ignored = {token.upper() for token in (skip_tokens or set())}

    for raw_line in str(raw_text or "").splitlines():
        upper = raw_line.strip().upper()
        if not upper or upper.startswith(TC2000_HEADER_PREFIX):
            continue
        for symbol in WATCHLIST_SYMBOL_RE.findall(upper):
            if symbol in ignored or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)

    return symbols


def read_watchlist_symbols(path: Path, *, skip_tokens: set[str] | None = None) -> list[str]:
    try:
        return extract_watchlist_symbols(path.read_text(encoding="utf-8"), skip_tokens=skip_tokens)
    except Exception:
        return []


def count_watchlist_symbols(path: Path, *, skip_tokens: set[str] | None = None) -> int:
    return len(read_watchlist_symbols(path, skip_tokens=skip_tokens))
