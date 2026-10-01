"""Trade intent: which ticker each verb acts on, and what it means against the open book. Pure, table-driven.

One resolver (P18 review rounds 5-6). Steps:

1. Tokenize; find tickers (the caller's symbols) and verb phrases from ``PHRASES`` (longest match first, so
   "sell short" is never "sell", "buy back" never "buy"; "sell-off" is one token, never a verb).
   * "close/closing" is an EXIT only when its object is a ticker, my/the/our, position, all/half, it or out -
     never "the close", "close is", "closing price" or "close above/below/at/near/red/green/strong/weak/...".
   * bare "short"/"long" are verbs only in a verb frame: first word of a clause or right after I/I'm/I'd/to/of
     ("should I short", "want to short", "thinking of long..."), AND followed by a ticker, it/them or here/now/at.
     Never after how/the/a/an/any, never before interest/squeeze/ideas/term/side/setup/trade. "go/get short|long"
     and "take a short|long" are always verbs.
   * Past tense is history, not a live intent (``PAST``: sold, closed, covered, bought, trimmed, exited, got out,
     stopped out, "I'm out of"): the name gets journal + pick rows, never a gate; history wins over any verb.
2. Split into clauses on , ; ? — and "and"/"but"/"then". One ticker in the sentence or in the clause takes every
   verb; otherwise each verb binds to the nearest ticker after it in its clause, else before it.
3. Resolve per ticker, the BOOK first:
   held LONG:  exit verb (sell/trim/take profit/take some off/cut/flat/scale out/get out/dump/lighten/exit/cover/
               close) -> EXIT LONG; buy/add/go long -> ADD LONG (an add word wins over an exit word);
               go short / short X / sell short -> FLIP: EXIT LONG and NEW SHORT.
   held SHORT: cover/buy back/buy/close/exit/trim/... -> EXIT SHORT ("dump" closes it too); sell/short/add -> ADD
               SHORT; go long / long X -> FLIP: EXIT SHORT and NEW LONG.
   not held:   buy/long -> NEW LONG; sell/short -> NEW SHORT; add alone -> nothing; an exit verb -> no gate,
               only "no position in X to exit".
   A held name with no verb bound to it gets no gate (it is context).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

EXIT, COVER, SELL, ADD, BUY, GO_LONG, GO_SHORT, CLOSE, PAST = (
    "exit", "cover", "sell", "add", "buy", "go_long", "go_short", "close", "past")

#: (token sequence, class), longest first at each position.
PHRASES: tuple[tuple[tuple[str, ...], str], ...] = tuple(sorted((
    # history: what he already did
    (("sold",), PAST), (("covered",), PAST), (("bought",), PAST), (("trimmed",), PAST),
    (("exited",), PAST), (("got", "out"), PAST), (("stopped", "out"), PAST), (("stop", "out"), PAST),
    (("i'm", "out"), PAST), (("im", "out"), PAST), (("i", "am", "out"), PAST),
    # covering a short
    (("buy", "back"), COVER), (("buying", "back"), COVER), (("buy", "it", "back"), COVER),
    (("cover",), COVER), (("covering",), COVER),
    # new or flipped sides
    (("sell", "short"), GO_SHORT), (("selling", "short"), GO_SHORT), (("go", "short"), GO_SHORT),
    (("going", "short"), GO_SHORT), (("get", "short"), GO_SHORT), (("getting", "short"), GO_SHORT),
    (("take", "a", "short"), GO_SHORT), (("taking", "a", "short"), GO_SHORT), (("shorting",), GO_SHORT),
    (("go", "long"), GO_LONG), (("going", "long"), GO_LONG), (("get", "long"), GO_LONG),
    (("getting", "long"), GO_LONG), (("enter", "long"), GO_LONG), (("entering", "long"), GO_LONG),
    (("take", "a", "long"), GO_LONG), (("taking", "a", "long"), GO_LONG),
    (("buy",), BUY), (("buying",), BUY),
    (("sell",), SELL), (("selling",), SELL),
    # quiet exits
    (("take", "profit"), EXIT), (("take", "profits"), EXIT), (("taking", "profit"), EXIT),
    (("taking", "profits"), EXIT), (("take", "some", "profit"), EXIT), (("take", "some", "profits"), EXIT),
    (("take", "some", "off"), EXIT), (("taking", "some", "off"), EXIT), (("go", "flat"), EXIT),
    (("scale", "out"), EXIT), (("scaling", "out"), EXIT), (("get", "out"), EXIT), (("getting", "out"), EXIT),
    (("trim",), EXIT), (("trimming",), EXIT), (("dump",), EXIT), (("dumping",), EXIT), (("cut",), EXIT),
    (("lighten",), EXIT), (("lightening",), EXIT), (("exit",), EXIT), (("exiting",), EXIT),
    # adds
    (("add", "more"), ADD), (("size", "up"), ADD), (("sizing", "up"), ADD), (("add",), ADD), (("adding",), ADD),
    (("press",), ADD), (("pressing",), ADD),
), key=lambda item: -len(item[0])))
CLOSE_WORDS = frozenset({"close", "closing"})
#: "close" followed by one of these is a price word, never an exit.
CLOSE_PRICE_NEXT = frozenset({"above", "below", "at", "near", "red", "green", "strong", "weak", "today",
                              "yesterday", "price", "is", "was", "of", "higher", "lower", "up", "down", "flat"})
CLOSE_OBJECT_NEXT = frozenset({"my", "the", "our", "position", "all", "half", "it", "out", "this", "that"})
#: Bare short/long: who may say it before, what may follow, and the nouns it is never a verb before.
SIDE_FRAME_PREV = frozenset({"i", "i'm", "im", "i'd", "id", "to", "of", "gonna", "wanna", "just", "then", "and"})
SIDE_NEVER_PREV = frozenset({"how", "the", "a", "an", "any", "my", "your", "his", "more", "very", "too", "so"})
SIDE_OBJECT_NEXT = frozenset({"it", "them", "here", "now", "at", "this", "that"})
SIDE_NEVER_NEXT = frozenset({"interest", "squeeze", "squeezes", "idea", "ideas", "term", "side", "sides", "setup",
                             "setups", "trade", "trades", "position", "positions", "list", "book", "bias"})
#: "the TSLA short", "my AMD long": a held position named, not a trade.
HELD_NOUN_PREV = frozenset({"the", "my", "our", "your", "his", "her", "their", "this", "that", "how"})
#: "buy back" before one of these is a company's buyback, not a cover.
BUYBACK_NOUN_NEXT = frozenset({"program", "programs", "plan", "authorization", "announcement"})
#: "flat"/"cut"/"take X off": only with a ticker as object.
OBJECT_ONLY = frozenset({"flat", "cut"})
CLAUSE_WORDS = frozenset({"and", "but", "then"})
_TOKEN = re.compile(r"\$?[A-Za-z][A-Za-z'\-]*|[,;?—]")


@dataclass(frozen=True)
class Intent:
    """One gate. ``kind``: ``new``, ``add`` or ``exit`` (``flip`` marks the two halves of a side flip);
    ``none_to_exit`` = an exit verb on a name not held; ``history`` = past tense, no gate."""

    symbol: str
    kind: str
    side: str = ""
    flip: bool = False


def _tokens(text: str) -> list[str]:
    return [tok.lstrip("$") if tok[0] == "$" else tok for tok in _TOKEN.findall(text or "")]


def _clause_starts(toks: list[str]) -> set[int]:
    starts, fresh = set(), True
    for i, tok in enumerate(toks):
        if tok in (",", ";", "?", "—") or tok.lower() in CLAUSE_WORDS:
            fresh = True
            continue
        if fresh:
            starts.add(i)
            fresh = False
    return starts


def verbs(text: str, tickers: Iterable[str]) -> list[tuple[int, str]]:
    """``[(token index, class)]`` for every verb phrase in ``text``."""
    toks = _tokens(text)
    lowered = [t.lower() for t in toks]
    upper = {str(t).upper() for t in tickers}
    starts = _clause_starts(toks)

    def is_ticker(j: int) -> bool:
        return 0 <= j < len(toks) and toks[j].upper() in upper

    found: list[tuple[int, str]] = []
    i = 0
    while i < len(toks):
        word = lowered[i]
        prev = lowered[i - 1] if i else ""
        nxt = lowered[i + 1] if i + 1 < len(toks) else ""
        if word in CLOSE_WORDS:
            if prev not in ("the", "a") and nxt not in CLOSE_PRICE_NEXT and (nxt in CLOSE_OBJECT_NEXT or is_ticker(i + 1)):
                found.append((i, CLOSE))
            i += 1
            continue
        if word == "closed":
            # "I closed AMD" is history; "AMD closed red / above vwap" is a price.
            if prev not in ("the", "a") and nxt not in CLOSE_PRICE_NEXT:
                found.append((i, PAST))
            i += 1
            continue
        if word in ("short", "long"):
            framed = (i in starts or prev in SIDE_FRAME_PREV) and prev not in SIDE_NEVER_PREV
            # "a QCOM long at 230", "size up TSLA short": the side right after its ticker, at the end or before
            # at/here/now - never "the TSLA short", "my AMD long".
            trailing = (is_ticker(i - 1) and (lowered[i - 2] if i >= 2 else "") not in HELD_NOUN_PREV
                        and nxt in ("", "at", "here", "now", ",", "?"))
            if (framed and nxt not in SIDE_NEVER_NEXT and (is_ticker(i + 1) or nxt in SIDE_OBJECT_NEXT)) or trailing:
                found.append((i, GO_SHORT if word == "short" else GO_LONG))
            i += 1
            continue
        if word in OBJECT_ONLY:
            if is_ticker(i + 1):
                found.append((i, EXIT))
            i += 1
            continue
        if word in ("take", "taking") and is_ticker(i + 1) and i + 2 < len(toks) and lowered[i + 2] == "off":
            found.append((i, EXIT))  # "take NVDA off"
            i += 3
            continue
        for phrase, kind in PHRASES:
            if tuple(lowered[i:i + len(phrase)]) != phrase:
                continue
            after = lowered[i + len(phrase)] if i + len(phrase) < len(toks) else ""
            if kind == COVER and phrase[-1] == "back" and after in BUYBACK_NOUN_NEXT:
                break  # a buyback program, not a cover
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


def _one(sym: str, kinds: set[str], side: str) -> list[Intent]:
    if PAST in kinds:
        return [Intent(sym, "history")]
    exits = kinds & {EXIT, COVER, CLOSE}
    if side == "LONG":
        if GO_SHORT in kinds:
            return [Intent(sym, "exit", "LONG", flip=True), Intent(sym, "new", "SHORT", flip=True)]
        if ADD in kinds or ((BUY in kinds or GO_LONG in kinds) and not exits and SELL not in kinds):
            return [Intent(sym, "add", "LONG")]
        if exits or SELL in kinds:
            return [Intent(sym, "exit", "LONG")]
        return []
    if side == "SHORT":
        if GO_LONG in kinds:
            return [Intent(sym, "exit", "SHORT", flip=True), Intent(sym, "new", "LONG", flip=True)]
        if ADD in kinds and not exits and BUY not in kinds:
            return [Intent(sym, "add", "SHORT")]
        if exits or BUY in kinds:
            return [Intent(sym, "exit", "SHORT")]
        if SELL in kinds or GO_SHORT in kinds:
            return [Intent(sym, "add", "SHORT")]
        return []
    longish, shortish = bool(kinds & {BUY, GO_LONG}), bool(kinds & {SELL, GO_SHORT})
    if longish and not shortish:
        return [Intent(sym, "new", "LONG")]
    if shortish:
        return [Intent(sym, "new", "SHORT")]
    if exits:
        return [Intent(sym, "none_to_exit")]
    return []


def resolve(text: str, tickers: Iterable[str], held: Mapping[str, str]) -> list[Intent]:
    """The ``Intent``s for every ticker a verb acts on, read against ``held`` (``{SYM: LONG|SHORT}``)."""
    names = [str(t).upper() for t in tickers]
    out: list[Intent] = []
    for sym, kinds in bind(text, names).items():
        if kinds:
            out += _one(sym, kinds, str(held.get(sym) or "").upper())
    order = {sym: n for n, sym in enumerate(names)}
    return sorted(out, key=lambda item: order.get(item.symbol, 0))


def has_verb(text: str, tickers: Iterable[str]) -> bool:
    return bool(verbs(text, tickers))
