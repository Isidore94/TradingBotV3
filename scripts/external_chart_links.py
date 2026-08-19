"""Deep-link a symbol into an external charting tool (WISHLIST, 2026-08-18).

The trader's wishlist entry asked for "deep-link a symbol/timeframe into
TradingView or TC2000", with one prerequisite recorded against it: *confirm the
supported URL/application schemes and the failure behaviour; no scraping or
browser-automation dependency*. That is what this module is - a link builder
and nothing else. It opens a URL. It never scrapes, never drives a browser, and
never reads anything back, so there is no second source of truth about a
symbol anywhere in the system.

**TradingView is built; TC2000 is deliberately not.** TradingView has a stable,
documented chart URL with a symbol query parameter, so a link can be built
without guessing. TC2000 is a desktop application whose documented automation
surface is its own scripting layer, not a URL scheme; inventing a
``tc2000://`` handler that silently does nothing on the trader's machine would
be worse than the honest gap. The template is a setting, so the trader can
point it at TC2000 - or anything else - the moment they know the scheme their
install answers to, without a code change.

The exchange prefix is the one real subtlety. TradingView resolves a bare
symbol, but a bare symbol can be ambiguous across exchanges; the trader's
universe is US equities, so the default template passes the plain ticker and
lets TradingView resolve it, which is what its own search does. A trader who
wants ``NASDAQ:AAPL`` sets the template to include the prefix.
"""

from __future__ import annotations

import re

#: The local setting that overrides the template. Machine-local, like every
#: other desk preference, and absent by default.
EXTERNAL_CHART_URL_SETTING = "external_chart_url_template"

#: ``{symbol}`` and ``{interval}`` are substituted; anything else is left alone.
DEFAULT_CHART_URL_TEMPLATE = "https://www.tradingview.com/chart/?symbol={symbol}&interval={interval}"

#: Desk timeframe -> the external tool's interval token. Only the timeframes the
#: desk actually charts are mapped; an unknown one falls back to the daily view
#: rather than producing a URL whose meaning nobody can predict.
INTERVAL_BY_TIMEFRAME = {
    "M5": "5",
    "5M": "5",
    "5m": "5",
    "M15": "15",
    "H1": "60",
    "1H": "60",
    "D1": "D",
    "D": "D",
    "W1": "W",
    "M1": "M",
}
DEFAULT_INTERVAL = "D"

_SYMBOL_OK = re.compile(r"^[A-Z0-9.\-:]{1,16}$")


def interval_for_timeframe(timeframe: str) -> str:
    """The external interval token for a desk timeframe."""
    return INTERVAL_BY_TIMEFRAME.get(str(timeframe or "").strip(), DEFAULT_INTERVAL)


def chart_url(symbol: str, timeframe: str = "D1", template: str | None = None) -> str:
    """The deep link for ``symbol``, or ``""`` when there is nothing to link to.

    A symbol that does not look like a ticker returns an empty string rather
    than a URL built out of whatever text was in the box. The desk's symbol box
    is free text, and "open a browser at a URL assembled from unvalidated
    input" is a category of bug worth refusing outright.
    """
    ticker = str(symbol or "").strip().upper()
    if not ticker or not _SYMBOL_OK.match(ticker):
        return ""
    pattern = str(template or DEFAULT_CHART_URL_TEMPLATE)
    if "{symbol}" not in pattern:
        # A template that cannot carry the symbol is a misconfiguration, and
        # opening it would chart whatever the trader last looked at.
        return ""
    try:
        return pattern.format(symbol=ticker, interval=interval_for_timeframe(timeframe))
    except (IndexError, KeyError, ValueError):
        # An unknown placeholder is the trader's typo, not a reason to guess.
        return ""


def configured_template() -> str:
    """The trader's template, or the built-in TradingView one."""
    try:
        from project_paths import get_local_setting

        value = str(get_local_setting(EXTERNAL_CHART_URL_SETTING, "") or "").strip()
    except Exception:
        value = ""
    return value or DEFAULT_CHART_URL_TEMPLATE


def open_chart(symbol: str, timeframe: str = "D1") -> tuple[bool, str]:
    """Open the external chart. Returns (opened, message for the status bar).

    Failure is reported, never swallowed: if the desktop refuses to open the
    URL the trader needs to know the click did nothing, because the alternative
    is staring at an unchanged screen wondering which window has focus.
    """
    url = chart_url(symbol, timeframe, configured_template())
    if not url:
        return False, f"No external chart link for {symbol or '(no symbol)'}."
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        opened = bool(QDesktopServices.openUrl(QUrl(url)))
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not open the external chart: {exc}"
    if not opened:
        return False, f"The desktop refused to open {url}"
    return True, f"Opened {url}"
