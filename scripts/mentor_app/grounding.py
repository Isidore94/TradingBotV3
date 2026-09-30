"""Free-chat guardrail 2: a number the packs never said is shown grey, never hidden. Qt-free.

A number in the model's reply is grounded when the same number appears in some pack
text sent this turn (the desk context plus every tool pack). Citations (``[pick:NVDA:cell]``)
are left alone. Ungrounded numbers are wrapped in ``<span class="uncited">`` with an
inline grey colour, which the transcript's markdown renders as HTML.
"""

from __future__ import annotations

import re
from typing import Iterable

from mentor_packs.citations import CITATION_RE

NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?![\w])")
UNCITED_CLASS = "uncited"
UNCITED_COLOR = "#8a8a8a"


def normalize(token: str) -> str:
    """``+$1,234.50%`` -> ``1234.5``: sign, currency, grouping, percent and trailing zeros go."""
    text = token.strip().lstrip("+-").replace("$", "").replace(",", "").rstrip("%")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    whole, dot, frac = text.partition(".")
    return (whole.lstrip("0") or "0") + (dot + frac if frac else "")


def grounded_numbers(pack_texts: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for text in pack_texts:
        for match in NUMBER_RE.finditer(str(text or "")):
            found.add(normalize(match.group(0)))
    return found


def mark_uncited_numbers(reply: str, pack_texts: Iterable[str]) -> str:
    """The reply with every ungrounded number wrapped in a grey span; citations untouched."""
    allowed = grounded_numbers(pack_texts)
    text = str(reply or "")
    out: list[str] = []
    position = 0
    for citation in CITATION_RE.finditer(text):
        out.append(_mark(text[position:citation.start()], allowed))
        out.append(citation.group(0))  # an id's digits are never a claim
        position = citation.end()
    out.append(_mark(text[position:], allowed))
    return "".join(out)


def _mark(chunk: str, allowed: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if normalize(token) in allowed:
            return token
        return f'<span class="{UNCITED_CLASS}" style="color:{UNCITED_COLOR}">{token}</span>'

    return NUMBER_RE.sub(replace, chunk)
