from __future__ import annotations

"""Persistent, presentation-only Alert Center symbol suppression."""

import os
from pathlib import Path
from typing import Iterable

from project_paths import ALERT_CENTER_IGNORED_SYMBOLS_FILE
from watchlist_utils import extract_watchlist_symbols


def load_ignored_alert_symbols(
    path: Path = ALERT_CENTER_IGNORED_SYMBOLS_FILE,
) -> set[str]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return set()
    return set(extract_watchlist_symbols(text))


def save_ignored_alert_symbols(
    symbols: Iterable[str],
    path: Path = ALERT_CENTER_IGNORED_SYMBOLS_FILE,
) -> set[str]:
    normalized = set(extract_watchlist_symbols(" ".join(str(value or "") for value in symbols)))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    try:
        staged.write_text(
            "".join(f"{symbol}\n" for symbol in sorted(normalized)),
            encoding="utf-8",
        )
        os.replace(staged, target)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
    return normalized
