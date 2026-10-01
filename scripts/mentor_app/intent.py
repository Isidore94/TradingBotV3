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
#: "I'm long AMD" / "I'm short TSLA": a status about his own side; TAKE: "take/enter/get into X" (no side said).
STATUS_LONG, STATUS_SHORT, TAKE = "status_long", "status_short", "take"
#: "I'm flat AMD": he holds nothing in it now (whatever the book file says); never a gate.
STATUS_FLAT = "status_flat"
STATUSES = (STATUS_LONG, STATUS_SHORT, STATUS_FLAT)

#: (token sequence, class), longest first at each position.
PHRASES: tuple[tuple[tuple[str, ...], str], ...] = tuple(sorted((
    # history: what he already did
    (("sold",), PAST), (("covered",), PAST), (("bought",), PAST), (("trimmed",), PAST),
    (("exited",), PAST), (("got", "out"), PAST), (("stopped", "out"), PAST), (("stop", "out"), PAST),
    (("i'm", "out"), PAST), (("im", "out"), PAST), (("i", "am", "out"), PAST),
    (("got", "into"), PAST), (("got", "in"), PAST), (("went", "flat"), PAST),
    (("buy", "back", "in"), BUY), (("buying", "back", "in"), BUY),
    (("get", "me", "out"), EXIT), (("getting", "me", "out"), EXIT),
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
    (("size", "down"), EXIT), (("sizing", "down"), EXIT),
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
#: "flat"/"cut"/"take X off": only with a ticker as object ("cut my losses on AMD": these words may sit between).
OBJECT_ONLY = frozenset({"flat", "cut"})
OBJECT_FILLER = frozenset({"my", "the", "some", "half", "all", "losses", "loss", "on", "in", "of", "position", "out"})
#: "take/enter/get into X": a trade with no side said; it takes the known side (only with a ticker right after).
TAKE_WORDS = frozenset({"take", "taking", "enter", "entering"})
TAKE_INTO = frozenset({"get", "getting"})
CLAUSE_WORDS = frozenset({"and", "but", "then"})
_TOKEN = re.compile(r"\$?[A-Za-z][A-Za-z'\-]*|[,;:?—]")
SEPARATORS = (",", ";", ":", "?", "—")
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')


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


def _quoted(text: str) -> set[int]:
    """Indices of the tokens inside quotation marks ('"sell AMD" he said')."""
    ranges = [(m.start(), m.end()) for m in _QUOTED.finditer(text or "")]
    return {n for n, m in enumerate(_TOKEN.finditer(text or ""))
            if any(a < m.start() < b for a, b in ranges)}


#: (f) a third-person subject before the verb in its clause: the verb is someone else's, never his ask.
THIRD_PERSON = frozenset({"he", "she", "they", "him", "her", "them", "he's", "she's", "they're", "hes", "shes",
                          "anyone", "someone", "somebody", "everyone", "everybody", "nobody", "people", "guys",
                          "analysts", "analyst", "who", "who's", "whos", "street", "guy", "twitter", "wife",
                          "husband", "brother", "sister", "friend", "buddy", "dad", "mom", "boss", "cathie",
                          "funds", "traders", "experts", "everybody's"})
SAY_WORDS = frozenset({"says", "said", "say", "told", "tells", "recommends", "recommended", "thinks", "wants",
                       "suggests", "suggested", "reckons"})
FIRST_PERSON_WORDS = frozenset({"i", "i'm", "im", "i'd", "i've", "i'll", "id", "ive"})
#: "I'm buying AMD" is a present intent; "... all morning / since the open" is narration.
DURATION = re.compile(r"\b(?:all (?:morning|day|week|session|afternoon)|since|for (?:the )?(?:last|past)|"
                      r"every (?:day|morning)|lately|recently)\b")


def _clause_starts(toks: list[str]) -> set[int]:
    starts, fresh = set(), True
    for i, tok in enumerate(toks):
        if tok in SEPARATORS or tok.lower() in CLAUSE_WORDS:
            fresh = True
            continue
        if fresh:
            starts.add(i)
            fresh = False
    return starts


#: The final guard (P18 review round 8). A gate opens only for a present-tense verb whose ticker is its direct
#: object (within OBJECT_WINDOW filler words) or, with no object at all, the one ticker the clause rules give;
#: never for a ticker used as an adjective ("the AMD noise"); a clause with a past-time marker turns its verbs
#: into history. A quiet miss is acceptable; a wrong gate is not.
OBJECT_WINDOW = 4
FILLER = frozenset({"my", "the", "some", "half", "all", "of", "on", "in", "into", "to", "position", "more", "a",
                    "little", "bit", "out", "back", "me", "losses", "loss", "like", "rest", "remaining", "entire",
                    "whole", "shares", "bunch", "this", "that"})
#: After the verb these mean "no object": the clause rules pick the ticker.
OBJECTLESS = frozenset({"it", "them", "here", "now", "at", "?", ",", ";", ":", "\u2014", "too", "already", "today",
                        "please", "or", "and", "but", "then", "soon", "first"})
#: A ticker followed by one of these is an adjective, never an object.
ADJ_NOUNS = frozenset({"noise", "chart", "charts", "setup", "setups", "news", "earnings", "call", "calls", "story",
                       "long", "short", "position", "trade", "trades", "idea", "ideas", "side", "squeeze",
                       "interest", "buyback", "buybacks", "move", "levels", "level", "thesis", "puts", "dip", "dips",
                       "pop", "bounce", "breakout", "breakdown", "rip", "run", "gap", "flush", "weakness", "strength"})
PAST_MARKER = re.compile(r"\b(?:yesterday|earlier|ago|this morning|last (?:week|night|month|friday|monday|tuesday|"
                         r"wednesday|thursday|session)|on (?:monday|tuesday|wednesday|thursday|friday))\b")
#: "at the open" is past only with no future word in the sentence ("add AMD at the open tomorrow").
AT_THE_OPEN = re.compile(r"\bat the open\b")
FUTURE_WORDS = re.compile(r"\b(?:tomorrow|next|will|should i|gonna|going to)\b|\?")
#: Past frames: "I was buying", "I was going to buy", "I wanted to sell" are history, not a live intent.
PAST_FRAME = re.compile(r"\b(?:i was|i were|was going to|was gonna|i wanted|wanted to|i had|i meant|i did)\b")
#: Condition (e): the clause must ASK (or the verb is imperative, first in its clause).
ASK_CUE = re.compile(r"\b(?:should|shall|can|could|would|do|does|is it|time|ok|okay) (?:i|we|my|to)\b|\bwhat should\b"
                     r"|\bthinking (?:of|about)\b|\babout to\b|\b(?:want|wanna|looking|going|need|like|ready|"
                     r"planning|plan) to\b|\bwanna\b|\bgonna\b|\bthoughts\b|\bcheck(?:list)?\b|\bpre-trade\b"
                     r"|\bwalk me through\b|\bworth\b")
IMPERATIVE_LEAD = frozenset({"just", "please", "ok", "okay", "so", "now", "then", "maybe", "should"})
#: A verb word after one of these is a noun ("a buy", "my exit", "the sell").
NOUN_PREV = frozenset({"a", "an", "my", "the", "or", "your", "his", "her", "our", "their"})
#: "TSLA shorts covering": a market phrase - the verb's subject is the crowd, not him.
MARKET_SUBJECTS = frozenset({"shorts", "longs", "bears", "bulls", "buyers", "sellers", "funds", "people",
                             "everyone", "traders"})
#: A clause starting with one of these has its own subject: a carried status frame stops.
NEW_SUBJECT = frozenset({"i", "it", "it's", "its", "he", "she", "we", "they", "you", "this", "that", "what",
                         "how", "should", "is", "does", "do", "can"})
WATCHLIST_WORDS = frozenset({"watchlist", "list", "radar", "screen", "focus"})


#: Every single-word verb in the table ("a buy or sell": the word after "or").
VERB_WORDS = frozenset(phrase[0] for phrase, _kind in PHRASES) | {"close", "short", "long", "hold"}


def spans(text: str, tickers: Iterable[str]) -> list[tuple[int, int, str]]:
    """``[(first token, last token, class)]`` for every verb phrase in ``text``."""
    toks = _tokens(text)
    lowered = [t.lower() for t in toks]
    upper = {str(t).upper() for t in tickers}
    starts = _clause_starts(toks)

    def is_ticker(j: int) -> bool:
        return 0 <= j < len(toks) and toks[j].upper() in upper

    def at(j: int) -> str:
        return lowered[j] if 0 <= j < len(toks) else ""

    found: list[tuple[int, int, str]] = []
    carry = ""  # a status frame ("I'm long ...") carried across coordinated side words
    i = 0
    while i < len(toks):
        word, prev, nxt = lowered[i], at(i - 1), at(i + 1)
        if i in starts and carry and word in NEW_SUBJECT:
            carry = ""
        if word in CLOSE_WORDS:
            if prev not in ("the", "a") and nxt not in CLOSE_PRICE_NEXT and (nxt in CLOSE_OBJECT_NEXT or is_ticker(i + 1)):
                found.append((i, i, CLOSE))
            i += 1
            continue
        if word == "closed":
            # "I closed AMD" is history; "AMD closed red / above vwap" is a price.
            if prev not in ("the", "a") and nxt not in CLOSE_PRICE_NEXT:
                found.append((i, i, PAST))
            i += 1
            continue
        if word == "out" and i in starts and at(i + 1) in ("", "?", ",", ";"):
            found.append((i, i, EXIT))  # "..., out?"
            i += 1
            continue
        if word == "flat":
            # "I'm flat AMD", "I'm already flat on NVDA", "I am flat AMD": a status, never an exit.
            back = at(i - 2) if prev in ("already", "now", "totally", "fully") else prev
            frame = back in ("i'm", "im") or (back == "am" and at(i - 3 if prev != back else i - 2) == "i")
            if (frame or (carry and i in starts)) and (is_ticker(i + 1) or (nxt == "on" and is_ticker(i + 2))
                                                      or nxt in ("", "now", "here", ",", "?")):
                found.append((i, i, STATUS_FLAT))
                carry = STATUS_FLAT
                i += 1
                continue
        if word in ("short", "long"):
            side_status = STATUS_SHORT if word == "short" else STATUS_LONG
            obj = is_ticker(i + 1) or nxt in SIDE_OBJECT_NEXT or (nxt == "on" and is_ticker(i + 2))
            if carry and i in starts and obj:
                found.append((i, i, side_status))  # "I'm short TSLA and long AMD"
                i += 1
                continue
            if prev == "a" and nxt in ("on", "in", "for") and is_ticker(i + 2):
                found.append((i, i, GO_SHORT if word == "short" else GO_LONG))  # "a long on TGT"
                i += 1
                continue
            status = prev in ("i'm", "im") or (prev == "am" and at(i - 2) == "i")
            if status and obj:
                found.append((i, i, side_status))  # "I'm long AMD", "I'm long on AMD"
                carry = side_status
                i += 1
                continue
            framed = (i in starts or prev in SIDE_FRAME_PREV) and prev not in SIDE_NEVER_PREV
            # "a QCOM long at 230", "size up TSLA short": the side right after its ticker, at the end or before
            # at/here/now - never "the TSLA short", "my AMD long".
            trailing = (is_ticker(i - 1) and at(i - 2) not in HELD_NOUN_PREV
                        and nxt in ("", "at", "here", "now", ",", "?", ":", ";", "\u2014"))
            if (framed and nxt not in SIDE_NEVER_NEXT and (is_ticker(i + 1) or nxt in SIDE_OBJECT_NEXT)) or trailing:
                found.append((i, i, GO_SHORT if word == "short" else GO_LONG))
            i += 1
            continue
        if word in OBJECT_ONLY:
            j = i + 1
            while j < len(toks) and j <= i + 4 and lowered[j] in OBJECT_FILLER:
                j += 1
            if is_ticker(j):
                found.append((i, i, EXIT))
            i += 1
            continue
        if word in ("take", "taking") and is_ticker(i + 1) and at(i + 2) == "off":
            if at(i + 3) not in ("my", "the") and at(i + 3) not in WATCHLIST_WORDS:
                found.append((i, i + 2, EXIT))  # "take NVDA off" - never "off my watchlist"
            i += 3
            continue
        if word in TAKE_WORDS and (is_ticker(i + 1) or (nxt in ("it", "them") and at(i + 2) in ("", "?", ","))):
            found.append((i, i, TAKE))  # "should I take TSLA", "entering TSLA"
            i += 1
            continue
        if word in TAKE_INTO and nxt in ("into", "in") and is_ticker(i + 2):
            found.append((i, i + 1, TAKE))  # "getting into ALL"
            i += 2
            continue
        for phrase, kind in PHRASES:
            if tuple(lowered[i:i + len(phrase)]) != phrase:
                continue
            after = at(i + len(phrase))
            if kind == COVER and phrase[-1] == "back" and after in BUYBACK_NOUN_NEXT:
                break  # a buyback program, not a cover
            if prev in MARKET_SUBJECTS and kind != PAST:
                break  # "TSLA shorts covering": the crowd, not him
            if kind != PAST and (prev in NOUN_PREV or (is_ticker(i - 1) and at(i - 2) in NOUN_PREV)
                                 or (after == "or" and at(i + len(phrase) + 1) in VERB_WORDS)):
                break  # a noun: "is NVDA a buy or sell", "what's my AMD exit"
            found.append((i, i + len(phrase) - 1, kind))
            i += len(phrase) - 1
            break
        i += 1
    return found


def verbs(text: str, tickers: Iterable[str]) -> list[tuple[int, str]]:
    """``[(token index, class)]`` for every verb phrase in ``text``."""
    return [(first, kind) for first, _last, kind in spans(text, tickers)]


def bind(text: str, tickers: Iterable[str]) -> dict[str, set[str]]:
    """``{TICKER: {verb classes}}``: the object rule first, then the clause rules, under the final guard."""
    toks = _tokens(text)
    lowered = [t.lower() for t in toks]
    upper = [str(t).upper() for t in tickers]
    out: dict[str, set[str]] = {sym: set() for sym in upper}
    clause, number = [], 0
    for tok in toks:
        if tok in SEPARATORS or tok.lower() in CLAUSE_WORDS:
            number += 1
        clause.append(number)
    words_of: dict[int, list[str]] = {}
    for n, c in enumerate(clause):
        words_of.setdefault(c, []).append(lowered[n])
    sentence = " ".join(lowered)
    future = bool(FUTURE_WORDS.search(sentence))
    past_clauses = {c for c, words in words_of.items()
                    if PAST_MARKER.search(" ".join(words)) or PAST_FRAME.search(" ".join(words))
                    or (AT_THE_OPEN.search(" ".join(words)) and not future)}
    # (e): a clause asks when it carries a cue or ends with "?"; or its verb is first in it (an imperative).
    asked = {c for c, words in words_of.items() if ASK_CUE.search(" ".join(words))}
    for n, tok in enumerate(toks):
        if tok == "?" and n:
            asked.add(clause[n - 1])
    clause_start: dict[int, int] = {}
    for n, c in enumerate(clause):
        if toks[n] in SEPARATORS or lowered[n] in CLAUSE_WORDS:
            continue
        clause_start.setdefault(c, n)
    # A ticker followed by a noun ("the AMD noise", "QCOM short interest") is an adjective: never an object.
    def adjective(i: int) -> bool:
        nxt = lowered[i + 1] if i + 1 < len(toks) else ""
        if nxt in ("long", "short"):  # "the TSLA short" is his position; "QCOM short interest" is a noun phrase
            return (lowered[i + 2] if i + 2 < len(toks) else "") in SIDE_NEVER_NEXT
        return nxt in ADJ_NOUNS

    places = [(i, tok.upper()) for i, tok in enumerate(toks) if tok.upper() in upper and not adjective(i)]
    ticker_at = dict(places)
    # The objectless fallback may land on a ticker used as an adjective ("QCOM buyback - buy?").
    names_in_sentence = {tok.upper() for tok in toks if tok.upper() in upper}
    every_place = [(i, tok.upper()) for i, tok in enumerate(toks) if tok.upper() in upper]
    found = spans(text, upper)
    quoted = _quoted(text)
    imperative = set()
    for first, _last, _kind in found:
        lead = clause_start.get(clause[first], -1)
        if (first == lead or (first == lead + 1 and lowered[lead] in IMPERATIVE_LEAD)
                or (first >= 2 and lowered[first - 2] in FIRST_PERSON_WORDS and lowered[first - 1] == "say")):
            imperative.add(clause[first])  # "sell AMD", "just trim NVDA", "I say sell AMD"
    for first, last, kind in found:
        c = clause[first]
        lead = clause_start.get(c, first)
        before = range(lead, first)
        # (f) someone else's verb: a third-person subject or a sayer before it, or inside quotation marks.
        sayer = any(lowered[n] in SAY_WORDS and not (n and lowered[n - 1] in FIRST_PERSON_WORDS) for n in before)
        third = any(lowered[n] in THIRD_PERSON or (toks[n][0].isupper() and not toks[n].isupper() and n > 0
                                                    and lowered[n] not in FIRST_PERSON_WORDS
                                                    and toks[n].upper() not in upper) for n in before)
        if kind not in STATUSES and (first in quoted or sayer or third):
            continue
        progressive = (first >= 1 and lowered[first - 1] in ("i'm", "im") or (
            first >= 2 and lowered[first - 2:first] == ["i", "am"])) and lowered[first].endswith("ing")
        if kind not in (PAST, *STATUSES) and progressive and not DURATION.search(" ".join(words_of[c])):
            imperative.add(c)  # "I'm buying AMD": a present intent
        if kind not in (PAST, *STATUSES) and c in past_clauses:
            kind = PAST  # "I cut half my AMD this morning" is what he did
        if kind not in (PAST, *STATUSES) and c not in asked and c not in imperative:
            if kind in (GO_LONG, GO_SHORT) and first - 1 in ticker_at:
                continue  # "ALL short - take it?": a side tag after its ticker, not a verb of its own
            kind = PAST  # (e) plain narration with no ask cue ("I cut AMD at 155") is history, never a gate
        j, skipped = last + 1, 0
        while j < len(toks) and lowered[j] in FILLER and toks[j].upper() not in upper and skipped < OBJECT_WINDOW:
            j, skipped = j + 1, skipped + 1
        if j < len(toks) and j in ticker_at and clause[j] == c:
            out[ticker_at[j]].add(kind)  # the direct object
            k = j + 1
            while k + 1 < len(toks) and lowered[k] in ("or", "and", "&") and k + 1 in ticker_at:
                out[ticker_at[k + 1]].add(kind)  # "should I buy QCOM or AMD": coordinated objects
                k += 2
            continue
        nxt = lowered[j] if j < len(toks) else ""
        if j < len(toks) and toks[j].upper() in upper and j not in ticker_at:
            continue  # the object is a ticker used as an adjective: no gate
        if j < len(toks) and nxt not in OBJECTLESS and clause[j] == c and not (
                kind in (GO_LONG, GO_SHORT) and first > 0 and first - 1 in ticker_at):
            continue  # a non-ticker object ("buy the dip"): no gate, a quiet miss
        # No object: the clause rules.
        if len(names_in_sentence) == 1:
            out[next(iter(names_in_sentence))].add(kind)
            continue
        here = [(i, sym) for i, sym in every_place if clause[i] == c]
        if len({sym for _, sym in here}) == 1:
            out[here[0][1]].add(kind)
            continue
        before = [(first - i, sym) for i, sym in here if i < first]
        if before:
            out[min(before)[1]].add(kind)
    # A carried status frame covers coordinated tickers with no verb of their own ("I'm long NVDA and AMD").
    status_kind, status_clause = "", -1
    verb_clauses = {clause[first]: kind for first, _last, kind in found}
    for c in sorted(words_of):
        kind = verb_clauses.get(c)
        if kind in STATUSES:
            status_kind, status_clause = kind, c
            continue
        if kind is not None or not status_kind or (words_of[c] and words_of[c][0] in NEW_SUBJECT):
            status_kind = "" if kind is not None or (words_of[c] and words_of[c][0] in NEW_SUBJECT) else status_kind
            continue
        for i, sym in places:
            if clause[i] == c and c > status_clause:
                out[sym].add(status_kind)
    return out


def _one(sym: str, kinds: set[str], side: str, known_side: str = "") -> list[Intent]:
    """Past tense wins ("sold AMD") unless an add follows it ("I bought NVDA yesterday, add?")."""
    if PAST in kinds and ADD not in kinds:
        return [Intent(sym, "history")]
    out = _decide(sym, set(kinds) - {PAST}, side, known_side)
    return out or ([Intent(sym, "history")] if PAST in kinds else [])


def _decide(sym: str, kinds: set[str], side: str, known_side: str) -> list[Intent]:
    if STATUS_FLAT in kinds:
        # "I'm flat AMD": he holds none now - the book file is behind; what follows is about a name not held.
        kinds.discard(STATUS_FLAT)
        side = ""
        if not kinds:
            return [Intent(sym, "status", "FLAT")]
    # "I'm long AMD": about a held name on that side it is a status (no gate); otherwise it says what he wants.
    for status, go, own in ((STATUS_LONG, GO_LONG, "LONG"), (STATUS_SHORT, GO_SHORT, "SHORT")):
        if status in kinds:
            kinds.discard(status)
            if side != own:
                kinds.add(go)
    if TAKE in kinds:
        kinds.discard(TAKE)
        if not kinds:
            if side:
                return [Intent(sym, "add", side)]
            if known_side in ("LONG", "SHORT"):
                return [Intent(sym, "new", known_side)]
            return []
    if not kinds:
        return [Intent(sym, "status", side)] if side else []
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


def resolve(text: str, tickers: Iterable[str], held: Mapping[str, str],
            known: Mapping[str, str] | None = None) -> list[Intent]:
    """The ``Intent``s for every ticker a verb acts on, read against ``held`` (``{SYM: LONG|SHORT}``); ``known``
    sides (Focus, likes) only side a TAKE verb ("entering TSLA") on a name not held."""
    names = [str(t).upper() for t in tickers]
    out: list[Intent] = []
    for sym, kinds in bind(text, names).items():
        if kinds:
            out += _one(sym, kinds, str(held.get(sym) or "").upper(), str((known or {}).get(sym) or "").upper())
    order = {sym: n for n, sym in enumerate(names)}
    return sorted(out, key=lambda item: order.get(item.symbol, 0))


def has_verb(text: str, tickers: Iterable[str]) -> bool:
    return bool(verbs(text, tickers))
