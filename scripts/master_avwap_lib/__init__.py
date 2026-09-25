"""Master AVWAP package.

This package is intentionally compatibility-first: focused modules expose the
current behavior while implementation is migrated out of ``legacy`` in stages.

``legacy`` (with pandas, ~1.5 s) loads on first use rather than with the
package, so importing a small submodule such as ``setup_tagging`` or
``d1_zone_arms`` no longer pays for it at desk boot (P2-11e). Any attribute not
found here still resolves through ``legacy``, exactly as before.
"""

from __future__ import annotations

import importlib

# Imported eagerly so ``from master_avwap_lib import execution_convention``
# resolves to the real submodule rather than falling through ``__getattr__``
# into ``legacy``'s namespace.
from . import execution_convention  # noqa: F401
from .app_logging import configure_logging as _configure_logging

# Importing the package has always set up root logging (it used to happen as a
# side effect of loading ``legacy``); keep that without paying for ``legacy``.
_configure_logging()


def _legacy():
    return importlib.import_module(f"{__name__}.legacy")


def __getattr__(name: str):
    if name == "legacy":
        return _legacy()
    if name == "__all__":
        legacy = _legacy()
        return [n for n in dir(legacy) if not (n.startswith("__") and n.endswith("__"))]
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(_legacy(), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy())))
