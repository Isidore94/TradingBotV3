"""Master AVWAP package.

This package is intentionally compatibility-first: focused modules expose the
current behavior while implementation is migrated out of ``legacy`` in stages.
"""

from __future__ import annotations

from . import legacy as _legacy


def __getattr__(name: str):
    return getattr(_legacy, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy)))


__all__ = [name for name in dir(_legacy) if not (name.startswith("__") and name.endswith("__"))]
