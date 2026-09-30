"""The pre-trade checklist the app guarantees: the model words it, the app makes sure it is all there.

A "thinking of taking" answer must touch six sections of the gate pack: earnings (own +
peers), plan lines, tape/regime, setup cell/cohort, book exposure and news. A section counts
as covered when the reply cites any id of that section. For every section it did not cite,
the app appends "Not covered: ..." with the gate pack's own rows. Pure and Qt-free.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from mentor_app.attach import cited_ids

SECTIONS: tuple[tuple[str, str], ...] = (
    ("earnings", "earnings (own + peers)"),
    ("plan", "plan lines"),
    ("tape", "tape / regime"),
    ("cell", "setup cell / cohort"),
    ("book", "book exposure"),
    ("news", "news"),
)
_TAPE_CONTEXT_IDS = frozenset({"ctx:regime", "ctx:d1_env", "ctx:auto_mode"})


def section_of(row_id: str) -> str:
    """Which checklist section an evidence id belongs to ("" = none)."""
    text = str(row_id or "").lower()
    if "book:" in text:
        return "book"
    if "tape:" in text or text in _TAPE_CONTEXT_IDS:
        return "tape"
    if ":plan" in text or text.startswith("plan:"):
        return "plan"
    if ":earn" in text or ":peer" in text or ":industry" in text:
        return "earnings"
    if ":cell" in text or ":cohort" in text or ":m5cell" in text or ":branch" in text:
        return "cell"
    if ":news" in text or text.startswith("news:"):
        return "news"
    return ""


def covered(reply: str) -> set[str]:
    return {section for section in (section_of(row_id) for row_id in cited_ids(reply)) if section}


def missing(reply: str) -> list[str]:
    done = covered(reply)
    return [key for key, _ in SECTIONS if key not in done]


def appendix(reply: str, pack: Any) -> str:
    """"" when the reply covered all six sections; else "Not covered: ..." plus the pack's rows for each."""
    gaps = missing(reply)
    if not gaps:
        return ""
    labels = dict(SECTIONS)
    rows: Iterable[Mapping[str, Any]] = getattr(pack, "rows", ()) or ()
    rows = list(rows)
    parts = [f"**Not covered:** {', '.join(labels[key] for key in gaps)}"]
    for key in gaps:
        found = [row for row in rows if section_of(str(row.get("id") or "")) == key]
        bullets = [f"- [{row['id']}] {row.get('text', '')}" for row in found] or [
            "- unknown: the gate pack has no rows for this"]
        parts.append(f"*{labels[key]}*\n" + "\n".join(bullets))
    return "\n\n".join(parts)
