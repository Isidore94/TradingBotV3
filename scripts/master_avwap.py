#!/usr/bin/env python3
"""Compatibility entrypoint for the Master AVWAP package.

The implementation lives in :mod:`master_avwap_lib`.  This file stays tiny so
old imports and launch commands keep working while the internals are split into
focused modules.
"""

from __future__ import annotations

import sys

from master_avwap_lib import legacy as _legacy

if __name__ == "__main__":
    _legacy.main()
else:
    sys.modules[__name__] = _legacy
