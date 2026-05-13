from __future__ import annotations

from collections.abc import Iterable

from . import legacy as _legacy


def expose_legacy_names(module_globals: dict, names: Iterable[str]) -> None:
    exported = tuple(dict.fromkeys(names))
    allowed = set(exported)

    def __getattr__(name: str):
        if name in allowed:
            return getattr(_legacy, name)
        raise AttributeError(f"module {module_globals.get('__name__')!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted(set(module_globals) | allowed)

    module_globals["__all__"] = list(exported)
    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
