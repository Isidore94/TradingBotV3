"""What a journal trade was EXPOSED to - ST5.3.

The journal knows two things about a position's direction and they are not the
same thing:

* ``trades.direction`` is **ownership**. It is the sign of the opening
  quantity, nothing else. A trade that SOLD to open is ``SHORT``.
* what the trader was betting on is a function of the CONTRACT, and the journal
  has never asked that question.

Read the first as the second and the live journal says the trader is a bearish
trader with a bullish record: 53 of the 89 option trades are ``SHORT`` and 39 of
those 53 were winners (2026-09-06). They are **sold puts**. A sold put is a
bullish-to-neutral position, and calling it "short" on a summary of market bias
is a claim the data does not support.

So this module separates four questions and answers each one only as far as the
stores allow:

``instrument``
    ``STK`` / ``OPT`` / ``UNKNOWN`` / ``BAG`` / ``CASH`` / whatever
    ``security_type`` holds. Passed through, never guessed. ``BAG`` and ``CASH``
    are real values in the live store and both are uncertain: a BAG is a
    combination order the journal stores as one symbol.

``ownership_direction``
    ``LONG`` / ``SHORT`` / ``UNKNOWN`` - exactly ``trades.direction``.

``market_bias``
    ``bullish`` / ``bearish`` / ``bullish_or_neutral`` / ``bearish_or_neutral``
    / ``unknown``. **A LONG option is never read as a bullish setup**; a bought
    put is BEARISH. Anything the contract does not establish stays ``unknown``,
    which is a bucket of its own and never folded into either side.

``structure``
    ``single`` / ``multi_leg`` / ``partial_of_spread_candidate`` / ``unknown``.
    **A ``trade_legs`` row is a FILL, not a contract leg** - every closed option
    trade in the live journal carries at least two of them and 39 carry exactly
    two. So ``multi_leg`` means *more than one distinct option CONTRACT among
    the legs*, and a two-fill single-contract trade is emphatically not one.

``certainty``
    ``known`` when the instrument and the contract together establish a bias,
    ``uncertain`` otherwise. The uncertain rows are surfaced as their own
    population by :func:`journal_analytics.personal_evidence_summary`, never
    pooled into a result.

WHERE THE CONTRACT LIVES. There are no right/strike/expiry columns on
``trades``. For an ``OPT`` row the identity is the OCC ``trades.symbol``
(``AA260522P00062000``), and the same facts also ride in
``raw_executions.raw_json["option"]``, which is what
``journal_statement_import._execution_from_row`` writes. This module reads both
and neither is required: a trade handed over without its legs is classified from
its symbol alone, and a BAG whose symbol carries no contract stays uncertain.

**Pure.** No store is opened here, nothing is written, and every unmeasurable
answer is a named "unknown" rather than a default.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: The instruments a bias can be read from at all. Everything else - the live
#: store's ``UNKNOWN`` (86 rows), ``BAG`` (1) and ``CASH`` (1) - is uncertain.
DIRECTIONAL_INSTRUMENTS = ("STK", "OPT")

#: Structures that make a position's opinion unreadable from ownership alone.
UNCERTAIN_STRUCTURES = ("multi_leg", "partial_of_spread_candidate")

BIAS_BULLISH = "bullish"
BIAS_BEARISH = "bearish"
BIAS_BULLISH_OR_NEUTRAL = "bullish_or_neutral"
BIAS_BEARISH_OR_NEUTRAL = "bearish_or_neutral"
BIAS_UNKNOWN = "unknown"

#: Every bias that names a side of the market. The rest is `unknown`, and the
#: summary prints it as its own bucket rather than dropping it.
DIRECTIONAL_BIASES = (
    BIAS_BULLISH,
    BIAS_BEARISH,
    BIAS_BULLISH_OR_NEUTRAL,
    BIAS_BEARISH_OR_NEUTRAL,
)

#: ``SPY260918P00600000`` - root, yymmdd, right, strike x 1000 in 8 digits.
_OCC = re.compile(
    r"^(?P<root>[A-Z][A-Z0-9./\-]{0,5}?)(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$"
)


@dataclass(frozen=True)
class OptionContract:
    """One option contract, however it was spelled. Immutable, comparable."""

    underlying: str
    expiry: str
    right: str  # "CALL" / "PUT" / "" when the source did not say
    strike: float | None

    @property
    def key(self) -> tuple[str, str, str, float | None]:
        return (self.underlying, self.expiry, self.right, self.strike)


@dataclass(frozen=True)
class Exposure:
    """What a trade owned, what it was betting on, and how sure that is."""

    instrument: str
    ownership_direction: str
    market_bias: str
    structure: str
    certainty: str
    contracts: tuple[OptionContract, ...] = ()

    @property
    def is_uncertain(self) -> bool:
        return self.certainty != "known"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _right(value: Any) -> str:
    """``CALL`` / ``PUT`` / ``""``. An unrecognised right is not a guess."""
    text = _upper(value)
    if text.startswith("C"):
        return "CALL"
    if text.startswith("P"):
        return "PUT"
    return ""


def _strike(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_occ_symbol(symbol: Any) -> OptionContract | None:
    """The contract inside an OCC symbol, or ``None`` when it is not one.

    ``AA260522P00062000`` -> AA / 260522 / PUT / 62.0. A plain ticker returns
    ``None``, which is how a stock row and a BAG row both stay out of the
    contract count without a special case.
    """
    match = _OCC.match(_upper(symbol).replace(" ", ""))
    if match is None:
        return None
    return OptionContract(
        underlying=match.group("root"),
        expiry=match.group("expiry"),
        right="CALL" if match.group("right") == "C" else "PUT",
        strike=int(match.group("strike")) / 1000.0,
    )


def _contract_from_payload(payload: Any, *, fallback_symbol: Any = None) -> OptionContract | None:
    """The contract inside a ``raw_executions.raw_json`` payload, if any."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return None
    if not isinstance(payload, Mapping):
        return None
    option = payload.get("option")
    if not isinstance(option, Mapping):
        return None
    underlying = _upper(option.get("underlying") or option.get("symbol"))
    if not underlying:
        parsed = parse_occ_symbol(fallback_symbol)
        underlying = parsed.underlying if parsed else _upper(fallback_symbol)
    return OptionContract(
        underlying=underlying,
        expiry=_text(option.get("expiry") or option.get("expiration")),
        right=_right(option.get("right") or option.get("put_call")),
        strike=_strike(option.get("strike")),
    )


def contracts_for(trade: Mapping[str, Any]) -> tuple[OptionContract, ...]:
    """Every DISTINCT option contract this trade touched, in first-seen order.

    The legs are fills. Two fills of one contract collapse to one entry here,
    which is the whole reason ``multi_leg`` can be trusted.
    """
    seen: dict[tuple[str, str, str, float | None], OptionContract] = {}

    def _remember(contract: OptionContract | None) -> None:
        if contract is None:
            return
        if not contract.underlying and not contract.expiry and contract.strike is None:
            return
        seen.setdefault(contract.key, contract)

    for leg in trade.get("legs") or ():
        if not isinstance(leg, Mapping):
            continue
        _remember(parse_occ_symbol(leg.get("symbol")))
        _remember(
            _contract_from_payload(leg.get("raw_json"), fallback_symbol=leg.get("symbol"))
        )
    if not seen:
        _remember(parse_occ_symbol(trade.get("symbol")))
    return tuple(seen.values())


def _session_of(trade: Mapping[str, Any]) -> str:
    for key in ("trade_date", "opened_at"):
        stamp = _text(trade.get(key))
        if len(stamp) >= 10:
            return stamp[:10]
    return ""


def _underlying_of(trade: Mapping[str, Any], contracts: Sequence[OptionContract]) -> str:
    for contract in contracts:
        if contract.underlying:
            return contract.underlying
    parsed = parse_occ_symbol(trade.get("symbol"))
    return parsed.underlying if parsed else _upper(trade.get("symbol"))


def _looks_like_a_spread_sibling(
    trade: Mapping[str, Any],
    contracts: Sequence[OptionContract],
    other: Mapping[str, Any],
) -> bool:
    """Best effort, and it says so in the label it produces.

    THE STORE HAS NO SIBLING SEAM. Two legs of one spread arrive as two separate
    ``trades`` rows keyed by their own OCC symbols; nothing links them, and this
    packet may not invent an identifier (`plan.md` P5.3/P5.4 own that). What can
    be OBSERVED is a second option trade on the same underlying and the same
    expiry opened in the same session on a different contract - which is what a
    spread looks like from outside, and also what two independent ideas on one
    name look like. Hence ``partial_of_spread_candidate``, never
    ``partial_of_spread``: the row is flagged as UNCERTAIN so its P&L is kept
    out of the clean populations, and no claim is made that it is half a spread.
    """
    if _upper(other.get("security_type")) != "OPT":
        return False
    if _text(other.get("trade_id")) == _text(trade.get("trade_id")):
        return False
    if _session_of(other) != _session_of(trade) or not _session_of(trade):
        return False
    other_contracts = contracts_for(other)
    if not other_contracts or not contracts:
        return False
    mine = {(c.underlying, c.expiry) for c in contracts if c.underlying and c.expiry}
    theirs = {(c.underlying, c.expiry) for c in other_contracts if c.underlying and c.expiry}
    if not (mine & theirs):
        return False
    # Same underlying and expiry but a DIFFERENT contract. The same contract
    # twice is two trades in one name, not a structure.
    return {c.key for c in contracts} != {c.key for c in other_contracts}


def classify_exposure(
    trade: Mapping[str, Any],
    *,
    siblings: Iterable[Mapping[str, Any]] | None = None,
) -> Exposure:
    """What this trade owned and what it was betting on.

    ``siblings`` is optional and defaults to none, so the answer for one trade
    never depends on a list a caller might not have. When it is supplied,
    :func:`_looks_like_a_spread_sibling` may raise the structure to
    ``partial_of_spread_candidate`` - the best the store supports.
    """
    instrument = _upper(trade.get("security_type")) or "UNKNOWN"
    ownership = _upper(trade.get("direction")) or "UNKNOWN"
    contracts = contracts_for(trade)

    structure = "single"
    if len(contracts) > 1:
        structure = "multi_leg"
    elif instrument == "OPT" and siblings is not None:
        for other in siblings:
            if isinstance(other, Mapping) and _looks_like_a_spread_sibling(
                trade, contracts, other
            ):
                structure = "partial_of_spread_candidate"
                break

    bias = _market_bias(instrument, ownership, contracts, structure)
    known = (
        instrument in DIRECTIONAL_INSTRUMENTS
        and structure == "single"
        and bias in DIRECTIONAL_BIASES
    )
    return Exposure(
        instrument=instrument,
        ownership_direction=ownership,
        market_bias=bias,
        structure=structure,
        certainty="known" if known else "uncertain",
        contracts=contracts,
    )


def _market_bias(
    instrument: str,
    ownership: str,
    contracts: Sequence[OptionContract],
    structure: str,
) -> str:
    """The one table this module exists for. Everything else is plumbing."""
    if structure in UNCERTAIN_STRUCTURES:
        # Two contracts under one trade: neither leg's direction is the
        # position's opinion, and a straddle has no side at all.
        return BIAS_UNKNOWN
    if instrument == "STK":
        if ownership == "LONG":
            return BIAS_BULLISH
        if ownership == "SHORT":
            return BIAS_BEARISH
        return BIAS_UNKNOWN
    if instrument != "OPT":
        # UNKNOWN, BAG, CASH and anything a future broker file invents.
        return BIAS_UNKNOWN
    right = contracts[0].right if contracts else ""
    if not right or ownership not in ("LONG", "SHORT"):
        return BIAS_UNKNOWN
    if ownership == "LONG":
        # A LONG option is never a bullish setup. It is a bought CALL that is
        # bullish, and a bought PUT is bearish.
        return BIAS_BULLISH if right == "CALL" else BIAS_BEARISH
    # Sold. Premium collected, and the bet is that the contract expires
    # worthless - which is a directional lean WITH a neutral half.
    return BIAS_BULLISH_OR_NEUTRAL if right == "PUT" else BIAS_BEARISH_OR_NEUTRAL


def classify_all(trades: Iterable[Mapping[str, Any]]) -> dict[str, Exposure]:
    """``trade_id -> Exposure`` over a whole list, siblings visible.

    One pass, and the sibling search sees the same list the caller is
    summarising - which is the only place ``partial_of_spread_candidate`` can be
    observed at all.
    """
    rows = [trade for trade in trades if isinstance(trade, Mapping)]
    option_rows = [row for row in rows if _upper(row.get("security_type")) == "OPT"]
    out: dict[str, Exposure] = {}
    for trade in rows:
        out[_text(trade.get("trade_id"))] = classify_exposure(
            trade, siblings=option_rows if option_rows else None
        )
    return out


__all__ = [
    "BIAS_BEARISH",
    "BIAS_BEARISH_OR_NEUTRAL",
    "BIAS_BULLISH",
    "BIAS_BULLISH_OR_NEUTRAL",
    "BIAS_UNKNOWN",
    "DIRECTIONAL_BIASES",
    "DIRECTIONAL_INSTRUMENTS",
    "Exposure",
    "OptionContract",
    "UNCERTAIN_STRUCTURES",
    "classify_all",
    "classify_exposure",
    "contracts_for",
    "parse_occ_symbol",
]
