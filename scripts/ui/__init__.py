"""PySide6 presentation layer for TradingBotV3.

The package keeps imports lazy so the legacy Tk GUI and headless scripts do not
need Qt installed until the new UI is explicitly launched.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SCRIPT_DIR.parent

# scripts/ stays at the front for the bot libraries; the repo root is appended as
# a fallback so root-level packages (market_prep, used by Research → Ticker Lookup)
# resolve without shadowing any same-named module under scripts/.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
