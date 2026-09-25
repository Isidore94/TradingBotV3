"""The app's root logging setup, moved out of ``legacy`` unchanged (P2-11e).

Importing the ``master_avwap_lib`` package has always configured root logging
(console + the rotating ``trading_bot.log``), because the package imported
``legacy`` eagerly. ``legacy`` now loads lazily, so the package calls
``configure_logging`` itself; ``legacy`` re-exports these same names.
"""

from __future__ import annotations

import logging

from project_paths import APP_LOG_BACKUP_COUNT, MASTER_AVWAP_LOG_FILE, SafeRotatingFileHandler

YF_EARNINGS_LOGGER_NAMES = (
    "yfinance",
    "yfinance.base",
    "yfinance.scrapers",
    "yfinance.scrapers.calendar",
    "yfinance.scrapers.quote",
)
IBAPI_NOISY_LOGGER_NAMES = (
    "ibapi",
    "ibapi.client",
    "ibapi.comm",
    "ibapi.connection",
    "ibapi.decoder",
    "ibapi.orderdecoder",
    "ibapi.reader",
    "ibapi.utils",
    "ibapi.wrapper",
)
APP_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s]: %(message)s"


def _configure_third_party_loggers() -> None:
    # The IB API emits one INFO log per socket send/request, which overwhelms
    # the console and makes it look like the watchlist itself exploded.
    for logger_name in IBAPI_NOISY_LOGGER_NAMES:
        ib_logger = logging.getLogger(logger_name)
        ib_logger.setLevel(logging.WARNING)
        ib_logger.propagate = True
    for logger_name in YF_EARNINGS_LOGGER_NAMES:
        yf_logger = logging.getLogger(logger_name)
        yf_logger.setLevel(logging.CRITICAL)
        yf_logger.propagate = True


def configure_logging():
    logger = logging.getLogger()
    if logger.handlers:
        _configure_third_party_loggers()
        return  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(APP_LOG_FORMAT)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)
    try:
        MASTER_AVWAP_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = SafeRotatingFileHandler(
            MASTER_AVWAP_LOG_FILE,
            maxBytes=2_000_000,
            backupCount=APP_LOG_BACKUP_COUNT,
        )
    except OSError as exc:
        logger.warning(f"File logging disabled for {MASTER_AVWAP_LOG_FILE}: {exc}")
        return

    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    _configure_third_party_loggers()
