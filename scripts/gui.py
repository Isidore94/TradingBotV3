#!/usr/bin/env python3
"""Compatibility entrypoint for the consolidated TradingBot GUI."""

from __future__ import annotations

import sys

from gui_app import app as _app

if __name__ == "__main__":
    try:
        _app.main()
    except KeyboardInterrupt:
        pass
else:
    sys.modules[__name__] = _app
