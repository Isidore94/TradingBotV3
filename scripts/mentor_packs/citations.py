"""Citation checks for the Trade Mentor's structured replies.

Same shape as ``ai_jobs.plan_review.check_challenges``: a bullet that cites an id
the packs did not carry rejects the WHOLE reply (the model invented evidence); a
bullet that cites nothing is dropped on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: An inline citation in free text, e.g. ``[ctx:mode]`` or ``[plan:risk:2]``.
CITATION_RE = re.compile(r"\[([a-z][a-z0-9_]*:[A-Za-z0-9_.:\-]+)\]")


class CitationRejected(ValueError):
    """The reply cited an id no pack carried."""


@dataclass(frozen=True)
class CitationResult:
    kept: tuple[dict[str, Any], ...]
    dropped: tuple[dict[str, Any], ...]


def _text(value: Any) -> str:
    return str(value or "").strip()


def cited_ids(text: str) -> list[str]:
    """Ids cited inline in free text, in order, without repeats."""
    seen: list[str] = []
    for match in CITATION_RE.finditer(str(text or "")):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen


def check_citations(reply: Any, allowed_ids: Iterable[str]) -> CitationResult:
    """Validate ``{"bullets": [{"text", "evidence_refs": [...]}]}`` against the packs' ids.

    Raises :class:`CitationRejected` for a malformed reply or a foreign id.
    """
    if not isinstance(reply, Mapping):
        raise CitationRejected("the reply was not an object")
    rows = reply.get("bullets")
    if rows is None or not isinstance(rows, (list, tuple)):
        raise CitationRejected("the reply carried no bullets array")
    allowed = {_text(item) for item in allowed_ids}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CitationRejected(f"bullet {index} was not an object")
        refs = row.get("evidence_refs") or ()
        if isinstance(refs, str):
            refs = [refs]
        refs = [_text(ref) for ref in refs if _text(ref)]
        for ref in refs:
            if ref not in allowed:
                raise CitationRejected(f"bullet {index} cited {ref!r}, which no pack carries")
        if not refs or not _text(row.get("text")):
            dropped.append(dict(row))
            continue
        kept.append({**dict(row), "evidence_refs": refs})
    return CitationResult(kept=tuple(kept), dropped=tuple(dropped))
