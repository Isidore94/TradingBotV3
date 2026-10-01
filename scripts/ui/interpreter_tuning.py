"""Interpreter knobs the desk sets once at launch.

The Setup Tracker is painted by a Python delegate: every cell is a hop from
Qt into Python, and every hop has to take the interpreter lock. While a worker
thread runs pure Python (shadow setups, the autopilot wrap-up, a Movers
refresh), each hop waits up to one switch interval for that lock. At CPython's
default 5 ms a repaint of a few thousand hops is tens of seconds of waiting
with almost no CPU on the Qt thread. A shorter interval bounds every wait;
the workers pay a little more switching for it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping

#: Milliseconds; overrides the default for one run. Clamped to the range below.
ENV_SWITCH_INTERVAL_MS = "TRADINGBOTV3_SWITCH_INTERVAL_MS"
DEFAULT_SWITCH_INTERVAL_MS = 1.0
MIN_SWITCH_INTERVAL_MS = 0.1
MAX_SWITCH_INTERVAL_MS = 100.0


def resolve_switch_interval_ms(env: Mapping[str, str] | None = None) -> float:
    """The switch interval to use, in ms: the env override when valid, else the default."""
    source = os.environ if env is None else env
    raw = str(source.get(ENV_SWITCH_INTERVAL_MS, "") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_SWITCH_INTERVAL_MS
    except ValueError:
        value = DEFAULT_SWITCH_INTERVAL_MS
    if value != value:  # NaN
        value = DEFAULT_SWITCH_INTERVAL_MS
    return min(MAX_SWITCH_INTERVAL_MS, max(MIN_SWITCH_INTERVAL_MS, value))


def configure_switch_interval(
    env: Mapping[str, str] | None = None,
    *,
    setter: Callable[[float], None] = sys.setswitchinterval,
) -> float:
    """Apply the resolved interval (seconds to the setter) and return it in ms."""
    interval_ms = resolve_switch_interval_ms(env)
    setter(interval_ms / 1000.0)
    return interval_ms


__all__ = [
    "DEFAULT_SWITCH_INTERVAL_MS",
    "ENV_SWITCH_INTERVAL_MS",
    "MAX_SWITCH_INTERVAL_MS",
    "MIN_SWITCH_INTERVAL_MS",
    "configure_switch_interval",
    "resolve_switch_interval_ms",
]
