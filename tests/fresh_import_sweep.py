"""Import every ``scripts/`` module as if it were the first one (P2-11e review).

Run as a script in a clean interpreter with scratch stores. For each module it
drops every first-party module from ``sys.modules`` and imports that one name,
so an import cycle that only bites when a given module is the entry point (the
scan child's ``master_avwap_lib.runner``) fails here. Prints one JSON object:
``{"failed": {module: error}, "count": n}``.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

#: Not importable on purpose, with the reason. Everything else must import.
SKIP_PREFIXES = (
    "gui_app",  # the retired Tk app, outside the bundle (pyproject excludes it too)
)


def module_names() -> list[str]:
    names = []
    for path in sorted(SCRIPTS.rglob("*.py")):
        rel = path.relative_to(SCRIPTS)
        if any(part.startswith((".", "__pycache__")) for part in rel.parts):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        # A package directory without __init__ is not importable as a package.
        if len(parts) > 1 and not (SCRIPTS.joinpath(*parts[:-1]) / "__init__.py").exists():
            continue
        name = ".".join(parts)
        if name.startswith(SKIP_PREFIXES):
            continue
        names.append(name)
    return names


def _first_party(module) -> bool:
    file = getattr(module, "__file__", None) or ""
    try:
        return Path(file).resolve().is_relative_to(ROOT)
    except (OSError, ValueError):
        return False


def main() -> int:
    sys.path.insert(0, str(SCRIPTS))
    failed: dict[str, str] = {}
    names = module_names()
    for name in names:
        for key in [k for k, m in list(sys.modules.items()) if m is not None and _first_party(m)]:
            sys.modules.pop(key, None)
        try:
            importlib.import_module(name)
        except BaseException as exc:  # SystemExit from a CLI module counts as a failure too
            failed[name] = f"{type(exc).__name__}: {exc}"[:300]
            if "-v" in sys.argv:
                traceback.print_exc()
    print(json.dumps({"failed": failed, "count": len(names)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
