"""Read the trader's pasted morning brief by HEADING MATCH alone — TJ-1 item 5.

The trader has a scheduled ChatGPT prompt that produces one Markdown brief a
day, and they paste it into the desk ("rename paste weekly forecast to paste
daily forecast, that's where I paste the output from my scheduled chatgpt
prompt", 2026-09-17). The whole text is what gets STORED; this module is a small
deterministic VIEW over it, so the Day Review page can show the two intraday
conditions and the ranked signals without a model reading the document.

Three rules, and they are the whole design:

* **Heading match only.** A section is found by its heading, case-insensitively,
  and nothing else. No sentence classification, no keyword scoring, no model.
* **Anything the document does not say is empty, never guessed** (plan.md §12.4
  TJ-1 item 5). A brief with no `Bottom line` heading has no bottom line; a
  playbook that names neither condition has two empty conditions. An inferred
  field here would become "what the brief said" on a page the trader reads.
* **Nothing is fetched.** It is a pure function over a string. The golden test
  monkeypatches `socket.connect` to prove it.

The golden input is the brief the trader actually pasted on 2026-09-17,
`tests/fixtures/day_review/forecast_2026-09-17.md`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

#: The section headings this reader knows, matched on the heading TEXT with its
#: `#` markers and surrounding whitespace removed, casefolded.
PLAYBOOK_HEADING = "intraday playbook"
BOTTOM_LINE_HEADING = "bottom line"

#: The two intraday conditions, as the brief bolds them. Matched inside a
#: paragraph rather than at its start: the trader's brief writes "The **bullish
#: continuation** requires…", and a reader anchored to the first character
#: would have found neither. Either word names either side ("**Bullish
#: reversal:**" / "**Bearish continuation:**" on 2026-09-24), colon optional.
_BULLISH_MARKER = re.compile(
    r"\*\*\s*bullish\s+(?:continuation|reversal)\s*:?\s*\*\*", re.IGNORECASE
)
_BEARISH_MARKER = re.compile(
    r"\*\*\s*bearish\s+(?:continuation|reversal)\s*:?\s*\*\*", re.IGNORECASE
)

#: A Markdown heading of any level.
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

#: `September 17, 2026`, with or without a weekday in front of it, and an ISO
#: date as written. Nothing else: a brief whose heading carries no date has no
#: date, and the page falls back to the session the trader is looking at.
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTH_DATE = re.compile(
    r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

#: The ranked-signals line: the one bold line with arrows in it.
ARROW = "→"

#: A turbulence score. `~8/10` is one number; `6–7/10` is TWO (a range, both
#: ends scored), and `September 29–30` on a Turbulence line is NEITHER, because
#: it carries no `/10`. That last case is why this is a pattern and not a scrape
#: of every number on the line.
_TURBULENCE_LINE = re.compile(r"turbulence\s*:", re.IGNORECASE)
_TURBULENCE_SCORE = re.compile(
    r"(?:(\d{1,2})\s*[–—-]\s*)?(\d{1,2})\s*/\s*10"
)


@dataclass(frozen=True)
class ForecastBrief:
    """What one pasted brief SAYS, with nothing added.

    Frozen because it is evidence about a document: a field edited after the
    parse would be a claim about the brief that the brief never made.
    """

    title_date: str = ""
    playbook_bullish: str = ""
    playbook_bearish: str = ""
    bottom_line: str = ""
    ranked_signals: tuple[str, ...] = field(default_factory=tuple)
    turbulence: tuple[int, ...] = field(default_factory=tuple)
    source_hash: str = ""


def normalize(text: str) -> str:
    """One newline convention, so a Windows paste parses like a Unix one.

    A Qt paste delivers whatever the clipboard held, and the fixture is a
    Windows-newline file. A reader that split paragraphs on `"\\n\\n"` without
    this saw one paragraph where the document has two.
    """
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def parse(text: str) -> ForecastBrief:
    """Read one brief. Every field is what the document says, or empty."""
    body = normalize(text)
    sections = _sections(body)
    bullish, bearish = _playbook(sections.get(PLAYBOOK_HEADING, ""))
    return ForecastBrief(
        title_date=_title_date(body),
        playbook_bullish=bullish,
        playbook_bearish=bearish,
        bottom_line=sections.get(BOTTOM_LINE_HEADING, ""),
        ranked_signals=_ranked_signals(body),
        turbulence=_turbulence(body),
        source_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------
def _sections(body: str) -> dict[str, str]:
    """Heading text (casefolded) -> the lines under it, up to the next heading.

    The next heading of ANY level closes a section: `### Bottom line` sits under
    `## Two-week outlook` in the trader's brief, and a reader that only stopped
    at the same level would have put the two-week outlook inside the bottom
    line. A repeated heading keeps the FIRST occurrence: a document that says
    "Intraday playbook" twice is one the reader has no rule for, and quietly
    preferring the later one would change the answer for no stated reason.
    """
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in body.splitlines():
        match = _HEADING.match(line)
        if match is not None:
            key = match.group(2).strip().casefold()
            if key in out:
                current = None
                continue
            out[key] = []
            current = out[key]
            continue
        if current is not None:
            current.append(line)
    return {key: "\n".join(lines).strip() for key, lines in out.items()}


def _paragraphs(section: str) -> list[str]:
    """A section's paragraphs, blank-line separated, as written."""
    out: list[str] = []
    buffer: list[str] = []
    for line in section.splitlines():
        if line.strip():
            buffer.append(line.rstrip())
            continue
        if buffer:
            out.append("\n".join(buffer).strip())
            buffer = []
    if buffer:
        out.append("\n".join(buffer).strip())
    return out


def _playbook(section: str) -> tuple[str, str]:
    """The bullish and the bearish paragraph of the playbook, kept apart.

    The section has a third paragraph in the trader's own brief (the 10:00
    catalyst). It is NEITHER condition, so a reader that returned "the section"
    would print a schedule note as a bullish trigger.
    """
    bullish = ""
    bearish = ""
    for paragraph in _paragraphs(section):
        if not bullish and _BULLISH_MARKER.search(paragraph):
            bullish = paragraph
            continue
        if not bearish and _BEARISH_MARKER.search(paragraph):
            bearish = paragraph
    return bullish, bearish


def _title_date(body: str) -> str:
    """The date in the FIRST heading, as ISO, or `""`.

    From the document, never from the clock: the trader may paste Thursday's
    brief on Thursday evening or Friday morning, and the brief itself is the
    only thing that knows which day it is about.
    """
    for line in body.splitlines():
        match = _HEADING.match(line)
        if match is None:
            continue
        return _iso_date(match.group(2))
    return ""


def _iso_date(text: str) -> str:
    iso = _ISO_DATE.search(text)
    if iso is not None:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
    named = _MONTH_DATE.search(text)
    if named is None:
        return ""
    month = _MONTHS.index(named.group(1).casefold()) + 1
    try:
        day = int(named.group(2))
        year = int(named.group(3))
    except ValueError:  # pragma: no cover - the pattern is digits
        return ""
    if not 1 <= day <= 31:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _ranked_signals(body: str) -> tuple[str, ...]:
    """The bold line of arrows, split in the order it was written.

    A tuple, not a set: the ranking IS the content of that line. The first bold
    arrow line in the document wins; a brief with none has no ranking.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if ARROW not in stripped or "**" not in stripped:
            continue
        cleaned = stripped.replace("**", "").strip()
        parts = [part.strip() for part in cleaned.split(ARROW)]
        parts = [part for part in parts if part]
        if len(parts) < 2:
            continue
        return tuple(parts)
    return ()


def _turbulence(body: str) -> tuple[int, ...]:
    """Every `N/10` on a Turbulence line, in document order, as ints."""
    out: list[int] = []
    for line in body.splitlines():
        if _TURBULENCE_LINE.search(line) is None:
            continue
        for match in _TURBULENCE_SCORE.finditer(line):
            low, high = match.group(1), match.group(2)
            if low is not None:
                out.append(int(low))
            out.append(int(high))
    return tuple(out)


def headline(brief: ForecastBrief) -> str:
    """One line for a collapsed block: what the brief is about, and how big.

    A label, never a judgement: it counts what was read and names nothing the
    document did not say.
    """
    parts: list[str] = []
    if brief.title_date:
        parts.append(f"brief for {brief.title_date}")
    if brief.turbulence:
        parts.append("turbulence " + "/".join(str(value) for value in brief.turbulence))
    if brief.ranked_signals:
        parts.append(f"{len(brief.ranked_signals)} ranked signals")
    if not parts:
        return "a pasted brief with no headings this reader knows"
    return " · ".join(parts)


def collapse(text: str, *, lines: int = 6) -> tuple[str, int]:
    """The first `lines` non-empty lines of a text, and how many are hidden.

    The page shows a forecast collapsed and says how much more there is; it
    never silently shows a fragment.
    """
    body = normalize(text).strip()
    if not body:
        return "", 0
    rows: Sequence[str] = body.splitlines()
    kept = [row for row in rows if row.strip()][: max(1, int(lines))]
    hidden = max(0, len([row for row in rows if row.strip()]) - len(kept))
    return "\n".join(kept), hidden


__all__ = [
    "ARROW",
    "BOTTOM_LINE_HEADING",
    "ForecastBrief",
    "PLAYBOOK_HEADING",
    "collapse",
    "headline",
    "normalize",
    "parse",
]
