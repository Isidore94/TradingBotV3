"""Plan lines pack: the trader's ``trading_plan.md`` lines with their plan ids. Read-only.

``read_plan(create=False, snapshot=False)``: a missing plan is never created and no
history snapshot is written from here. An empty plan is "no plan lines", nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mentor_packs.registry import Pack, make_pack

NAME = "plan_lines"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": "The trader's written trading plan, one citable line per rule.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
EMPTY_TEXT = "no plan lines"


def build(*, path: Path | str | None = None) -> Pack:
    import trading_plan

    result = trading_plan.read_plan(create=False, snapshot=False, path=Path(path) if path else None)
    if result.get("error"):
        return make_pack(NAME, (), empty_text=f"the plan could not be read; unknown ({result['error']})")
    rows = [
        {"id": str(line["id"]), "kind": "plan_line", "section": line.get("section", ""), "text": str(line["text"])}
        for line in trading_plan.plan_lines(result.get("parsed") or {})
        if str(line.get("text") or "").strip()
    ]
    return make_pack(NAME, rows, empty_text=EMPTY_TEXT)


def plan_digest(path: Path | str | None = None) -> str:
    """A short hash of the plan file's whole text ("none" when missing, "unknown" when unreadable).

    Card caches key on it, so any edit to the plan (a comment or a heading too) re-narrates.
    """
    import trading_plan

    result = trading_plan.read_plan(create=False, snapshot=False, path=Path(path) if path else None)
    if result.get("error"):
        return "unknown"
    if not result.get("exists"):
        return "none"
    return trading_plan.content_hash(str(result.get("text") or ""))[:16]


FIXTURE_PLAN = """# Trading plan

## Risk
- Max 3 open shorts at once.
- No new entries after 12:30 PT.

## Setups
- Only short below the D1 AVWAP.
"""


def fixture() -> Pack:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "trading_plan.md"
        target.write_text(FIXTURE_PLAN, encoding="utf-8")
        return build(path=target)
