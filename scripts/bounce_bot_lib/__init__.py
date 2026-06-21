"""BounceBot package.

Focused modules expose the current runtime while the large implementation is
split in behavior-preserving stages.
"""

from __future__ import annotations

from importlib import import_module

_legacy = None


def _legacy_module():
    global _legacy
    if _legacy is None:
        _legacy = import_module(f"{__name__}.legacy")
    return _legacy


def __getattr__(name: str):
    return getattr(_legacy_module(), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy_module())))


__all__: list[str] = []
