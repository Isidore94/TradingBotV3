"""Style guard (P14): measure a reply's wrapper and strip headers and closing offers. Pure, Qt-free.

``measure(reply, question)`` counts what makes a reply long without adding substance:
characters, markdown headers (``# X`` lines and bold label lines like ``**Today:**``),
bullets, a closing question, offer phrases ("would you like", ...), and regime / breadth /
SPY-pause sentences when the question had no market or pre-trade cue.

``guard(reply)`` rewrites only the wrapper: a ``### X`` header becomes ``**X**``, a bold
label line with no substance is dropped, and a trailing offer or closing question on a line with
no citation, number or ticker is removed. Every citation, number and ticker is kept byte for byte.
"""

from __future__ import annotations

import re
from typing import Any

#: Phrases that offer more instead of answering (the style score's "offer").
OFFER_PHRASES = ("would you like", "let me know if", "anything specific", "want me to", "shall i", "should i pull",
                 "do you want me", "i can pull", "i can also", "happy to", "feel free to")
#: Question sentences that end a reply by asking the trader for direction (no data in them).
_CLOSING_ASK = re.compile(r"\b(do you|would you|are you|is there|anything|what would|which would|want)\b", re.I)
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
#: A bold label alone on its line, ending in a colon (``**Today's Performance:**``): a header in disguise.
_BOLD_LABEL = re.compile(r"^\s*(?:\*\*|__)[^*_\n]{1,60}?(?::(?:\*\*|__)|(?:\*\*|__)\s*:)\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
_CITATION = re.compile(r"\[[A-Za-z][^\[\]\s]{2,}\]")
#: A sentence ends at . ! or ? followed by a space, so "$11.75" and "e.g." mostly stay whole.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
#: Market-context talk: the lines the trader called spam on a question that did not ask for them.
_CONTEXT = re.compile(r"\bregime\b|\bbear channel\b|\bbull channel\b|\bbreadth\b|\bspy pause\b|\bheld through a spy\b"
                      r"|\blower highs\b|\bhigher lows\b|\bauto mode\b|\[(?:ctx:regime|ctx:auto_mode|tape:[^\]]*)\]", re.I)


def _is_header(line: str) -> bool:
    return bool(_HEADER.match(line) or _BOLD_LABEL.match(line))


def _sentences(text: str) -> list[str]:
    return [piece.strip() for line in str(text or "").split("\n") for piece in _SENTENCE_END.split(line)
            if piece.strip()]


def _has_offer(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in OFFER_PHRASES)


def _last_sentence(text: str) -> str:
    body = [line for line in str(text or "").rstrip().split("\n") if line.strip()]
    if not body:
        return ""
    parts = _sentences(body[-1])
    return parts[-1] if parts else ""


def _is_bare_question(sentence: str) -> bool:
    """A question with no citation and no number: it asks the trader instead of telling him."""
    return (sentence.rstrip("*_ ").endswith("?") and not _CITATION.search(sentence)
            and not re.search(r"\d", sentence))


def _droppable_ending(sentence: str) -> bool:
    """An offer, or a question back to the trader that carries no citation and no number."""
    if not sentence:
        return False
    if _has_offer(sentence):
        return True
    return _is_bare_question(sentence) and bool(_CLOSING_ASK.search(sentence))


def measure(reply: str, question: str = "", *, market_cue: bool | None = None) -> dict[str, Any]:
    """The style numbers for one reply. ``market_cue`` None = read it from ``question``."""
    text = str(reply or "")
    lines = text.split("\n")
    if market_cue is None:
        from mentor_app.attach import market_cue as cue

        market_cue = cue(question)
    last = _last_sentence(text)
    context = 0 if market_cue else sum(1 for sentence in _sentences(text) if _CONTEXT.search(sentence))
    return {
        "chars": len(text),
        "headers": sum(1 for line in lines if _is_header(line)),
        "bullets": sum(1 for line in lines if _BULLET.match(line)),
        "closing_question": _is_bare_question(last),
        "offer_phrases": sum(1 for sentence in _sentences(text) if _has_offer(sentence)),
        "context_lines_unasked": context,
        "market_cue": bool(market_cue),
    }


def carries_substance(text: str) -> bool:
    """A citation, a number or a ticker: text the guard never removes."""
    from mentor_app.grounding import NOT_SYMBOLS

    if _CITATION.search(text) or re.search(r"\d", text):
        return True
    return any(match.group(1) not in NOT_SYMBOLS for match in _TICKER.finditer(text))


#: An all-capitals word of 2-5 letters (``AMD``, ``$PLTR``): a ticker until proven otherwise.
_TICKER = re.compile(r"(?<![A-Za-z0-9])\$?([A-Z]{2,5})(?![A-Za-z0-9])")


def guard(reply: str) -> tuple[str, list[str]]:
    """(the reply with headers turned bold and a trailing offer removed, the text that was removed).

    Only three rewrites, and never on a line with a citation, a number or a ticker except (a):
    (a) ``### X`` -> ``**X**``; (b) a bold label line ending in ":" is dropped; (c) a trailing offer or
    closing question is dropped, a sentence at a time, from a last line that carries no substance.
    """
    removed: list[str] = []
    out: list[str] = []
    for line in str(reply or "").split("\n"):
        header = _HEADER.match(line)
        if header:
            out.append(f"**{header.group(1).strip().strip('*').rstrip(':')}**")
            continue
        if _BOLD_LABEL.match(line) and not carries_substance(line):
            removed.append(line.strip())
            continue
        out.append(line)
    # Trailing offers / closing questions: whole sentences at the end of the last paragraph, repeatedly.
    while True:
        while out and not out[-1].strip():
            out.pop()
        if not out:
            break
        last_line = out[-1]
        parts = _sentences(last_line)
        if not parts or carries_substance(last_line) or not _droppable_ending(parts[-1]):
            break
        cut = last_line.rstrip().rfind(parts[-1])
        removed.append(parts[-1])
        kept = last_line[:cut].rstrip() if cut >= 0 else ""
        if kept.strip() and kept.strip() not in ("-", "*", "**"):
            out[-1] = kept
            break  # one sentence off a line with substance; the rest of that line is the answer
        out.pop()
    text = "\n".join(out)
    # A dropped label can leave three blank lines in a row; keep at most one.
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    return text, removed


def passes(style: dict[str, Any], *, simple: bool, max_chars: int = 600) -> bool:
    """The eval's style pass: no header, no offer or closing question, and short when the question is simple."""
    if style.get("headers") or style.get("offer_phrases") or style.get("closing_question"):
        return False
    return not simple or int(style.get("chars") or 0) <= max_chars
