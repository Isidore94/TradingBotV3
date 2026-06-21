from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "LONGS_FILENAME",
        "SHORTS_FILENAME",
        "BOUNCE_LOG_FILENAME",
        "TRADING_BOT_LOG_FILENAME",
        "INTRADAY_BOUNCES_CSV",
        "INTRADAY_BOUNCE_CANDIDATES_CSV",
        "INTRADAY_BOUNCE_OUTCOMES_CSV",
        "INTRADAY_BOUNCE_FEEDBACK_CSV",
        "MARKET_ENVIRONMENTS",
        "RRS_TIMEFRAMES",
        "BOUNCE_TYPE_DEFAULTS",
        "BOUNCE_TYPE_LABELS",
        "USE_GUI",
    ),
)
