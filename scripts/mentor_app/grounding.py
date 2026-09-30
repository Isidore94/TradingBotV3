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


def count_numbers(reply: str) -> int:
    """Numbers in the reply outside citations (the denominator of the uncited-number rate)."""
    return len(NUMBER_RE.findall(CITATION_RE.sub(" ", str(reply or ""))))


def _mark(chunk: str, allowed: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if normalize(token) in allowed:
            return token
        return f'<span class="{UNCITED_CLASS}" style="color:{UNCITED_COLOR}">{token}</span>'

    return NUMBER_RE.sub(replace, chunk)


# ---------------------------------------------------------------- P14: symbols in earnings claims
UNCITED_CLAIM_CLASS = "uncited_claim"
_EARNINGS_WORDS = re.compile(r"\bearnings\b|\breport(?:s|ing|ed)?\b|\bEPS\b", re.I)
_TICKER = re.compile(r"(?<![A-Za-z0-9$.\-])\$?([A-Z]{1,5}(?:\.[A-Z]{1,2})?)(?![A-Za-z0-9])")
_TAG = re.compile(r"<[^>]+>")
_BRACKETS = re.compile(r"\[[^\[\]]*\]")
#: Capitalised words that are not tickers in an earnings sentence.
NOT_SYMBOLS = frozenset({
    "I", "A", "AM", "PM", "ET", "PT", "EPS", "CEO", "CFO", "AI", "US", "USA", "EU", "UK", "ID", "IPO", "ETF", "OK",
    "YES", "NO", "EOD", "USD", "CAD", "TFSA", "RRSP", "IRA", "SEC", "FDA", "GDP", "CPI", "PPI", "FOMC", "FED", "YOY",
    "QOQ", "TTM", "NOTE", "LONG", "SHORT", "LONGS", "SHORTS", "M5", "D1", "R", "N", "NA", "TBD", "BMO", "AMC",
    "SPY", "QQQ", "IWM", "DIA", "VIX", "NYSE", "NASDAQ", "WIN", "LOSS", "NET", "OPEN", "HOLD",
})


def earnings_symbols(pack_texts: Iterable[str]) -> set[str]:
    """Tickers the packs said something about earnings for (lines naming earnings/reports, or earn/peer ids)."""
    found: set[str] = set()
    for text in pack_texts:
        for line in str(text or "").split("\n"):
            if _EARNINGS_WORDS.search(line) or re.search(r"\[[^\]]*(?:earn|peer)[^\]]*\]", line):
                found.update(match.group(1) for match in _TICKER.finditer(line.replace(":", " ")))
    return found


def mark_ungrounded_earnings(reply: str, pack_texts: Iterable[str]) -> str:
    """Grey a whole line that makes an earnings claim about a ticker no pack gave earnings for."""
    allowed = earnings_symbols(pack_texts)
    out: list[str] = []
    for line in str(reply or "").split("\n"):
        plain = _BRACKETS.sub(" ", _TAG.sub(" ", line))
        if line.strip() and _EARNINGS_WORDS.search(plain):
            named = {match.group(1) for match in _TICKER.finditer(plain)} - NOT_SYMBOLS
            if named - allowed:
                lead = line[: len(line) - len(line.lstrip())]
                out.append(f'{lead}<span class="{UNCITED_CLAIM_CLASS}" style="color:{UNCITED_COLOR}">'
                           f"{line.strip()}</span>")
                continue
        out.append(line)
    return "\n".join(out)
