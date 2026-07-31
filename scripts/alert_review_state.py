from __future__ import annotations

"""Trading-day-scoped, presentation-only Alert Center symbol suppression."""

import json
import os
from datetime import date
from pathlib import Path
from typing import Iterable

from project_paths import ALERT_CENTER_IGNORED_SYMBOLS_FILE
from watchlist_utils import extract_watchlist_symbols


def load_ignored_alert_symbols(
    path: Path = ALERT_CENTER_IGNORED_SYMBOLS_FILE,
    *,
    market_date: date | str | None = None,
) -> set[str]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return set()
    if not text.strip():
        return set()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        # The original feature stored a permanent newline list. Permanent
        # suppression is no longer valid, so do not carry that state forward.
        return set()
    if not isinstance(payload, dict):
        return set()
    if str(payload.get("market_date") or "") != _market_date_text(market_date):
        return set()
    values = payload.get("symbols")
    if not isinstance(values, list):
        return set()
    return set(
        extract_watchlist_symbols(
            " ".join(str(value or "") for value in values)
        )
    )


def save_ignored_alert_symbols(
    symbols: Iterable[str],
    path: Path = ALERT_CENTER_IGNORED_SYMBOLS_FILE,
    *,
    market_date: date | str | None = None,
) -> set[str]:
    normalized = set(extract_watchlist_symbols(" ".join(str(value or "") for value in symbols)))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    try:
        staged.write_text(
            json.dumps(
                {
                    "market_date": _market_date_text(market_date),
                    "symbols": sorted(normalized),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged, target)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
    return normalized


def load_day_scoped_flags(path: Path, *, market_date: date | str | None = None) -> set[str]:
    """Day-scoped set of opaque flag strings (e.g. "SYM|event_kind").

    Same lifecycle as the ignored-symbols store - a stored date other than
    today reads empty - but values are kept verbatim, not run through the
    symbol extractor.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return set()
    if not text.strip():
        return set()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    if str(payload.get("market_date") or "") != _market_date_text(market_date):
        return set()
    values = payload.get("flags")
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if str(value or "").strip()}


def save_day_scoped_flags(
    flags: Iterable[str],
    path: Path,
    *,
    market_date: date | str | None = None,
) -> set[str]:
    normalized = {str(value) for value in flags if str(value or "").strip()}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    try:
        staged.write_text(
            json.dumps(
                {
                    "market_date": _market_date_text(market_date),
                    "flags": sorted(normalized),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged, target)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
    return normalized


def _market_date_text(value: date | str | None) -> str:
    return value.isoformat() if isinstance(value, date) else str(value or date.today().isoformat())
