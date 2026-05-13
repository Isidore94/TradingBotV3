# -*- coding: utf-8 -*-
"""Compatibility entrypoint for the BounceBot package."""

from __future__ import annotations

from bounce_bot_lib import legacy as _legacy

if __name__ == "__main__":
    _legacy.main()
else:
    import sys

    sys.modules[__name__] = _legacy
