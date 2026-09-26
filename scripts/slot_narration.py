"""B11: the two night narrations that had no desk page, read for display.

`daily_digest` writes `digests/narration/<yyyy>/<session>[.N].json`;
`setup_research` writes `retros/setup_research/<yyyy>/<date>[.N].narration[.M].json`
beside its packs. Both hold the `ai_summary` shape: `executive_summary` plus
lists of `{statement}`. These readers run on a page's worker and return a dict
whose `text` the page prints as is. Absent and stale say so in words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

#: The narration's list sections, in print order, and their labels.
SECTIONS = (
    ("what_is_working", "Working"),
    ("what_is_not_working", "Not working"),
    ("lessons_for_tomorrow", "Lessons"),
    ("risk_notes", "Risks"),
)
PER_SECTION = 3

_DATE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})(?P<rest>.*)$")


def _order(path: Path) -> tuple[str, tuple[int, ...]]:
    """`(date, ordinals)`: a `.N` re-run sorts after the file it superseded."""
    match = _DATE.match(path.name)
    if match is None:
        return ("", ())
    numbers = tuple(int(part) for part in match.group("rest").split(".") if part.isdigit())
    return (match.group("date"), numbers)


def _newest(paths) -> Path | None:
    dated = [path for path in paths if _order(path)[0]]
    return max(dated, key=_order) if dated else None


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def narration_body(payload: Mapping[str, Any] | None, *, per_section: int = PER_SECTION) -> str:
    """The summary and up to `per_section` statements per section, as plain text."""
    summary = (payload or {}).get("narration")
    if not isinstance(summary, Mapping):
        return ""
    lines: list[str] = []
    head = str(summary.get("executive_summary") or "").strip()
    if head:
        lines.append(head)
    for key, label in SECTIONS:
        items = [
            str(item.get("statement") or "").strip()
            for item in (summary.get(key) or ())
            if isinstance(item, Mapping) and str(item.get("statement") or "").strip()
        ]
        for text in items[:per_section]:
            lines.append(f"- {label}: {text}")
    return "\n".join(lines)


def _written(payload: Mapping[str, Any]) -> str:
    stamp = str(payload.get("generated_at") or "")[:16].replace("T", " ")
    model = str(payload.get("model") or "")
    return ", ".join(part for part in (f"written {stamp}" if stamp else "", model) if part)


def _default_store(name: str) -> Path | None:
    try:
        from ai_jobs import store

        return Path(store.get_ai_store_dir()) / name
    except Exception:  # noqa: BLE001 - no store configured is "absent"
        return None


def read_digest_narration(session: str, root: Path | None = None) -> dict[str, Any]:
    """The `daily_digest` narration for `session`. Worker side.

    `root` is the digests dir. state: present | absent | unreadable.
    """
    day = str(session or "")[:10]
    base = Path(root) if root is not None else _default_store("digests")
    folder = base / "narration" if base is not None else None
    files = sorted(folder.rglob("*.json")) if folder is not None and folder.is_dir() else []
    mine = _newest(path for path in files if _order(path)[0] == day)
    if mine is None:
        newest = _newest(path for path in files if _order(path)[0] <= day) if day else None
        tail = f" The newest is for {_order(newest)[0]}." if newest is not None else ""
        return {
            "state": "absent",
            "session": day,
            "text": f"Night digest: no narration for {day or 'this session'}.{tail}",
        }
    payload = _load(mine)
    body = narration_body(payload)
    if payload is None or not body:
        return {
            "state": "unreadable",
            "session": day,
            "text": f"Night digest for {day}: the narration file could not be read ({mine.name}).",
        }
    written = _written(payload)
    return {
        "state": "present",
        "session": day,
        "text": f"Night digest for {day}" + (f" ({written})" if written else "") + ":\n" + body,
    }


def read_setup_research_narration(root: Path | None = None) -> dict[str, Any]:
    """The newest `setup_research` narration, stale when a newer pack has none.

    `root` is `retros/setup_research`. state: present | stale | absent | unreadable.
    """
    base = Path(root) if root is not None else _default_store("retros")
    folder = base if root is not None else (base / "setup_research" if base is not None else None)
    files = sorted(folder.rglob("*.json")) if folder is not None and folder.is_dir() else []
    narrations = [path for path in files if ".narration" in path.name]
    packs = [path for path in files if ".narration" not in path.name]
    narration, pack = _newest(narrations), _newest(packs)
    if narration is None:
        tail = f" The newest research pack is {_order(pack)[0]}." if pack is not None else ""
        return {"state": "absent", "text": f"Setup research: no narration yet.{tail}"}
    payload = _load(narration)
    body = narration_body(payload)
    day = _order(narration)[0]
    if payload is None or not body:
        return {
            "state": "unreadable",
            "text": f"Setup research {day}: the narration file could not be read ({narration.name}).",
        }
    newer = _order(pack)[0] if pack is not None else ""
    stale = bool(newer) and newer > day
    written = _written(payload)
    head = f"Setup research for {day}" + (f" ({written})" if written else "")
    if stale:
        head += f" - STALE: the {newer} research has no narration"
    return {"state": "stale" if stale else "present", "text": f"{head}:\n{body}"}
