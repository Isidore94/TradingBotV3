"""The hourly instance of `intraday_history`, under its original name.

PCT-1 generalised the H1 fallback cache by `interval_minutes` so the Pullback
alert's M15 and M30 legs could ride the same cadence rule (see
`intraday_history`). Nothing about the hourly behaviour changed, so the module
that shipped with WISHLIST 10C keeps its name and its exports: `chart_watch`
imports `h1_bucket_end` from here, the panel builds `H1HistoryCache()` here,
and the five `test_rv_h1_*` files read `frame_to_h1_bars`,
`last_completed_h1_bucket` and `h1_bucket_end` here.

One rule, one implementation: this module adds no behaviour of its own.
"""

from __future__ import annotations

from typing import Any, Callable

from intraday_history import (  # noqa: F401 - the module's public surface
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    H1_MINUTES,
    H1_SPAN,
    SOURCE_CACHE,
    SOURCE_YFINANCE,
    IntradayHistoryCache,
    frame_to_h1_bars,
    h1_bucket_end,
    last_completed_h1_bucket,
)


class H1HistoryCache(IntradayHistoryCache):
    """`IntradayHistoryCache(60)`, with the hourly keyword signature.

    A subclass rather than an alias so `H1HistoryCache(downloader=...)` -
    which is how the panel and four test files build it - keeps working with
    no positional argument, and so `interval="60m"` stays accepted by name.
    """

    def __init__(
        self,
        *,
        downloader: Callable[..., Any] | None = None,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> None:
        super().__init__(
            H1_MINUTES, downloader=downloader, period=period, interval=interval
        )
