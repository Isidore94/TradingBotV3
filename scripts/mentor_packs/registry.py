"""The pack registry: the tool list, the citation ids and the prefetch queue all read it."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Iterable, Mapping

#: Adding a capability = one module here plus its test file.
PACK_MODULES: tuple[str, ...] = (
    "mentor_packs.context_pack",
    "mentor_packs.plan_lines",
    "mentor_packs.recall",
    "mentor_packs.pick_pack",
    "mentor_packs.veto_pack",
    "mentor_packs.regime_pack",
    "mentor_packs.gate_pack",
    "mentor_packs.news_pack",
    "mentor_packs.book_pack",
    "mentor_packs.mirror_pack",
    "mentor_packs.tilt_pack",
    "mentor_packs.hypothesis_pack",
    "mentor_packs.journal_pack",
    "mentor_packs.earnings_pack",
    "mentor_packs.night_pack",
    "mentor_packs.fundamentals_pack",
    "mentor_packs.recaps_pack",
)


@dataclass(frozen=True)
class Pack:
    """What one pack build returns. ``rows`` each carry an ``id`` and a ``text``."""

    name: str
    rows: tuple[dict[str, Any], ...] = ()
    empty_text: str = ""
    built_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(str(row["id"]) for row in self.rows if row.get("id"))

    def as_text(self) -> str:
        """One line per row, each prefixed by its citable id."""
        if not self.rows:
            return f"## {self.name}\n{self.empty_text or 'nothing'}"
        lines = [f"## {self.name}"]
        lines.extend(f"[{row['id']}] {row.get('text', '')}" for row in self.rows)
        return "\n".join(lines)

    def as_json(self) -> str:
        return json.dumps(
            {"name": self.name, "rows": list(self.rows), "empty_text": self.empty_text, "built_utc": self.built_utc},
            sort_keys=True,
            default=str,
        )


def pack_from_json(text: str) -> Pack:
    payload = json.loads(text)
    return Pack(
        name=str(payload.get("name") or ""),
        rows=tuple(dict(row) for row in payload.get("rows") or ()),
        empty_text=str(payload.get("empty_text") or ""),
        built_utc=str(payload.get("built_utc") or ""),
    )


def make_pack(name: str, rows: Iterable[Mapping[str, Any]], *, empty_text: str = "") -> Pack:
    return Pack(name=name, rows=tuple(dict(row) for row in rows), empty_text=empty_text)


def modules() -> dict[str, ModuleType]:
    """``{NAME: module}`` for every registered pack."""
    found: dict[str, ModuleType] = {}
    for dotted in PACK_MODULES:
        module = importlib.import_module(dotted)
        found[str(module.NAME)] = module
    return found


def names() -> tuple[str, ...]:
    return tuple(modules())


def tool_schemas() -> list[dict[str, Any]]:
    """The Ollama ``tools`` list, in registry order."""
    return [dict(module.SCHEMA) for module in modules().values()]


def build(name: str, **args: Any) -> Pack:
    """Build one pack; an unknown name or a failing build is an empty pack, never a raise."""
    module = modules().get(str(name))
    if module is None:
        return make_pack(str(name), (), empty_text=f"there is no pack called {name!r}")
    try:
        return module.build(**args)
    except TypeError as exc:
        return make_pack(str(name), (), empty_text=f"bad arguments for {name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - a broken pack reads as unknown, never as a claim
        logging.warning("mentor pack %s failed: %s", name, exc)
        return make_pack(str(name), (), empty_text=f"{name} could not be built ({type(exc).__name__}); unknown")
