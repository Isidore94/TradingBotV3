from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config_loader import load_market_prep_config


LOGGER_NAME = "market_prep"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s]: %(message)s"


def get_market_prep_logger() -> logging.Logger:
    config = load_market_prep_config()
    log_dir = config.resolved_paths().get("log_dir") or Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "market_prep.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    target = str(log_path.resolve())
    for handler in logger.handlers:
        handler_path = getattr(handler, "baseFilename", "")
        if handler_path and str(Path(handler_path).resolve()) == target:
            return logger

    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=1, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    return logger
