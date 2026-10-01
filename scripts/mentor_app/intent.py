"""Trade intent: which ticker each verb acts on, and what it means against the open book. Pure, table-driven.

One resolver replaces the regex chain (P18 review round 5). Steps:

1. Tokenize; find tickers (the caller's symbols) and verb phrases from ``PHRASES`` (longest match first, so
   "sell short" is never "sell", "buy back" never "buy"; "sell-off" is one token, never a verb). "close" is an
   EXIT only when its object is a ticker, my/the/our, position, all/half, it or out - never "the close",
   "close is", "closing price" or "close above/below/at/near/red/green/strong/weak/today/yesterday/price".
2. Split into clauses on , ; ? — and "and"/"but"/"then". A clause with one ticker binds all its verbs to it;
   a sentence with one ticker binds everything to it; otherwise each verb binds to the nearest ticker after
   it in its clause, else before it.
3. Resolve per ticker, the BOOK first (``RULES``):
   held LONG:  exit verb (sell/trim/take profit/scale out/get out/dump/lighten/exit/cover/close) -> EXIT LONG;
               buy/add -> ADD LONG (an add word always wins over an exit word).
   held SHORT: cover/buy back/buy/close/exit/trim/take profit/scale out/get out/dump/lighten -> EXIT SHORT;
               sell/short/add -> ADD SHORT. ("dump" on a held short closes it: it is an exit verb.)
   not held:   buy/long -> NEW LONG; sell/short -> NEW SHORT; add alone -> nothing; an exit verb -> no gate,
               only "no position in X to exit".
   A held name with no verb bound to it gets no gate (it is context).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

EXIT, COVER, SELL, ADD, NEW_LONG, NEW_SHORT, CLOSE = "exit", "cover", "sell", "add", "long", "short", "close"

#: (token sequence, class), longest first at each position.
PHRASES: tuple[tuple[tuple[str, ...], str], ...] = tuple(sorted((
    (("buy", "back"), COVER), (("buying", "back"), COVER), (("buy", "it", "back"), COVER),
    (("cover",), COVER), (("covering",), COVER),
    (("sell", "short"), NEW_SHORT), (("selling", "short"), NEW_SHORT), (("go", "short"), NEW_SHORT),
    (("going", "short"), NEW_SHORT), (("short",), NEW_SHORT), (("shorting",), NEW_SHORT),
    (("sell",), SELL), (("selling",), SELL), (("sold",), SELL),
    (("take", "profit"), EXIT), (("take", "profits"), EXIT), (("taking", "profit"), EXIT),
    (("taking", "profits"), EXIT), (("take", "some", "profit"), EXIT), (("take", "some", "profits"), EXIT),
    (("scale", "out"), EXIT), (("scaling", "out"), EXIT), (("get", "out"), EXIT), (("getting", "out"), EXIT),
    (("trim",), EXIT), (("trimming",), EXIT), (("dump",), EXIT), (("dumping",), EXIT),
    (("lighten",), EXIT), (("lightening",), EXIT), (("exit",), EXIT), (("exiting",), EXIT),
    (("add", "more"), ADD), (("size", "up"), ADD), (("sizing", "up"), ADD), (("add",), ADD), (("adding",), ADD),
    (("press",), ADD), (("pressing",), ADD),
    (("go", "long"), NEW_LONG), (("going", "long"), NEW_LONG), (("enter", "long"), NEW_LONG),
    (("entering", "long"), NEW_LONG), (("buy",), NEW_LONG), (("buying",), NEW_LONG), (("long",), NEW_LONG),
), key=lambda item: -len(item[0])))
CLOSE_WORDS = frozenset({"close", "closing", "closed"})
#: "close" followed by one of these is a price word, never an exit.
CLOSE_PRICE_NEXT = frozenset({"above", "below", "at", "near", "red", "green", "strong", "weak", "today",
                              "yesterday", "price", "is", "was", "of", "higher", "lower", "up", "down", "flat"})
CLOSE_OBJECT_NEXT = frozenset({"my", "the", "our", "position", "all", "half", "it", "out", "this", "that"})
CLAUSE_WORDS = frozenset({"and", "but", "then"})
_TOKEN = re.compile(r"\$?[A-Za-z][A-Za-z'\-]*|[,;?—]")


@dataclass(frozen=True)
class Intent:
    """One gate: ``kind`` is ``new``, ``add`` or ``exit``; ``none_to_exit`` = an exit verb on a name not held."""

    symbol: str
    kind: str
    side: str = ""


def _tokens(text: str) -> list[str]:
    return [tok.lstrip("$") if tok[0] == "$" else tok for tok in _TOKEN.findall(text or "")]


def verbs(text: str, tickers: Iterable[str]) -> list[tuple[int, str]]:
    """``[(token index, class)]`` for every verb phrase in ``text``."""
    toks = _tokens(text)
    lowered = [t.lower() for t in toks]
    upper = {str(t).upper() for t in tickers}
    found: list[tuple[int, str]] = []
    i = 0
    while i < len(toks):
        word = lowered[i]
        if word in CLOSE_WORDS:
            prev = lowered[i - 1] if i else ""
            nxt = lowered[i + 1] if i + 1 < len(toks) else ""
            if prev not in ("the", "a") and nxt not in CLOSE_PRICE_NEXT and (
                    nxt in CLOSE_OBJECT_NEXT or (i + 1 < len(toks) and toks[i + 1].upper() in upper)):
                found.append((i, CLOSE))
            i += 1
            continue
        for phrase, kind in PHRASES:
            if tuple(lowered[i:i + len(phrase)]) == phrase:
                found.append((i, kind))
                i += len(phrase) - 1
                break
        i += 1
    return found


def bind(text: str, tickers: Iterable[str]) -> dict[str, set[str]]:
    """``{TICKER: {verb classes}}`` by the clause rules above."""
    toks = _tokens(text)
    upper = [str(t).upper() for t in tickers]
    places = [(i, tok.upper()) for i, tok in enumerate(toks) if tok.upper() in upper]
    out: dict[str, set[str]] = {sym: set() for sym in upper}
    # Clause number of every token.
    clause, number = [], 0
    for tok in toks:
        if tok in (",", ";", "?", "—") or tok.lower() in CLAUSE_WORDS:
            number += 1
        clause.append(number)
    names_in_sentence = {sym for _, sym in places}
    for index, kind in verbs(text, upper):
        if len(names_in_sentence) == 1:
            out[next(iter(names_in_sentence))].add(kind)
            continue
        here = [(i, sym) for i, sym in places if clause[i] == clause[index]]
        if len({sym for _, sym in here}) == 1:
            out[here[0][1]].add(kind)
            continue
        after = [(i - index, sym) for i, sym in here if i > index]
        before = [(index - i, sym) for i, sym in here if i < index]
        if after:
            out[min(after)[1]].add(kind)
        elif before:
            out[min(before)[1]].add(kind)
    return out


def resolve(text: str, tickers: Iterable[str], held: Mapping[str, str]) -> list[Intent]:
    """One ``Intent`` per ticker a verb acts on, read against ``held`` (``{SYM: LONG|SHORT}`` of the open book)."""
    out: list[Intent] = []
    for sym, kinds in bind(text, tickers).items():
        if not kinds:
            continue
        side = str(held.get(sym) or "").upper()
        exits = kinds & {EXIT, COVER, CLOSE}
        if side == "LONG":
            if ADD in kinds or (NEW_LONG in kinds and not exits and SELL not in kinds):
                out.append(Intent(sym, "add", "LONG"))
            elif exits or SELL in kinds or NEW_SHORT in kinds:
                out.append(Intent(sym, "exit", "LONG"))
        elif side == "SHORT":
            if ADD in kinds and not exits:
                out.append(Intent(sym, "add", "SHORT"))
            elif exits or NEW_LONG in kinds:
                out.append(Intent(sym, "exit", "SHORT"))
            elif SELL in kinds or NEW_SHORT in kinds:
                out.append(Intent(sym, "add", "SHORT"))
        elif NEW_LONG in kinds and not (SELL in kinds or NEW_SHORT in kinds):
            out.append(Intent(sym, "new", "LONG"))
        elif SELL in kinds or NEW_SHORT in kinds:
            out.append(Intent(sym, "new", "SHORT"))
        elif exits:
            out.append(Intent(sym, "none_to_exit"))
    order = {str(t).upper(): n for n, t in enumerate(tickers)}
    return sorted(out, key=lambda item: order.get(item.symbol, 0))


def has_verb(text: str, tickers: Iterable[str]) -> bool:
    return bool(verbs(text, tickers))
