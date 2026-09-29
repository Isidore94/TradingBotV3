"""Stand-in `build_board` targets for the strength board child-process tests.

Loaded by file path inside the spawned child, so none of these touch the network.
"""

from __future__ import annotations

import os
import time


def board_with_pid(**kwargs):
    return {
        "long": [{"symbol": "NVDA"}],
        "short": [],
        "offered": 1,
        "measured": 1,
        "pid": os.getpid(),
        "fraction": kwargs.get("fraction"),
    }


def sleep_long(**_kwargs):
    time.sleep(8.0)
    return {"long": [{"symbol": "SLOW"}], "short": [], "offered": 1, "measured": 1}


def raise_error(**_kwargs):
    raise RuntimeError("boom in child")


def hard_exit(**_kwargs):
    os._exit(3)
