"""View-level "Hide Oil & Gas / Real Estate" filter (display only).

One switch, shared by the setups table, the Alert Center and the phone report.
It hides names from what the trader SEES; the scan, alerts, evidence, journal
and research stores keep recording every name. Unknown classification = shown.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path

import project_paths

#: Machine-local setting; default True (hide).
SETTING_HIDE_OIL_GAS_REAL_ESTATE = "view_hide_oil_gas_real_estate"
#: The short label every surface uses for its checkbox and its count.
HIDE_LABEL = "Hide Oil & Gas / Real Estate"
HIDDEN_NOUN = "oil & gas / real estate"

# How often the lookup re-stats the classification CSV (seconds).
_STAT_INTERVAL_SECONDS = 30.0

_lock = threading.Lock()
_cache: dict = {"path": None, "stamp": None, "checked": 0.0, "rows": {}}


def is_excluded(sector: str, industry: str, sector_key: str = "", industry_key: str = "") -> bool:
    """True for Real Estate or an Oil & Gas industry; False when unknown."""
    sector_text = str(sector or "").strip().lower()
    industry_text = str(industry or "").strip().lower()
    sector_slug = str(sector_key or "").strip().lower()
    industry_slug = str(industry_key or "").strip().lower()
    if sector_text == "real estate" or sector_slug == "real-estate":
        return True
    if industry_text.startswith("reit") or industry_slug.startswith("reit-"):
        return True
    if "oil & gas" in industry_text or industry_slug.startswith("oil-gas-"):
        return True
    return False


def hide_enabled() -> bool:
    """The shared switch (default ON); an unreadable setting keeps the default."""
    try:
        return bool(project_paths.get_local_setting(SETTING_HIDE_OIL_GAS_REAL_ESTATE, True))
    except Exception:  # noqa: BLE001 - a preference read never costs a surface
        return True


def set_hide_enabled(value: bool) -> None:
    project_paths.save_local_setting(SETTING_HIDE_OIL_GAS_REAL_ESTATE, bool(value))


def _classification_path() -> Path:
    return Path(project_paths.SYMBOL_CLASSIFICATION_CACHE_FILE)


def _read_rows(path: Path) -> dict[str, tuple[str, str, str, str]]:
    rows: dict[str, tuple[str, str, str, str]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol:
                rows[symbol] = (
                    str(row.get("sector") or ""),
                    str(row.get("industry") or ""),
                    str(row.get("sectorKey") or ""),
                    str(row.get("industryKey") or ""),
                )
    return rows


def _classifications() -> dict[str, tuple[str, str, str, str]]:
    """Symbol -> (sector, industry, sectorKey, industryKey), re-read on mtime change."""
    path = _classification_path()
    now = time.monotonic()
    with _lock:
        if (
            _cache["path"] == path
            and _cache["stamp"] is not None
            and now - _cache["checked"] < _STAT_INTERVAL_SECONDS
        ):
            return _cache["rows"]
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = ("missing",)
    with _lock:
        if _cache["path"] == path and _cache["stamp"] == stamp:
            _cache["checked"] = now
            return _cache["rows"]
    rows: dict[str, tuple[str, str, str, str]] = {}
    if stamp != ("missing",):
        try:
            rows = _read_rows(path)
        except Exception as exc:  # noqa: BLE001 - unknown means shown
            logging.warning("Sector filter: classification cache unreadable: %s", exc)
    with _lock:
        _cache.update(path=path, stamp=stamp, checked=now, rows=rows)
    return rows


def clear_cache() -> None:
    with _lock:
        _cache.update(path=None, stamp=None, checked=0.0, rows={})


def symbol_is_excluded(symbol: str) -> bool:
    """Classification says Oil & Gas / Real Estate. Unknown symbol -> False."""
    key = str(symbol or "").strip().upper()
    if not key:
        return False
    row = _classifications().get(key)
    if row is None:
        return False
    return is_excluded(*row)


def symbol_is_hidden(symbol: str) -> bool:
    """The switch is on AND the symbol is excluded."""
    return hide_enabled() and symbol_is_excluded(symbol)


def hidden_line(count: int) -> str:
    """The one short line a surface prints when it hid names; "" for none."""
    return f"Hidden: {int(count)} {HIDDEN_NOUN}" if count else ""
