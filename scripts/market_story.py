"""The session's story: what the trader said, what the market did — WISHLIST 10D step 1.

The Market Journal already holds the trader's words and the desk already holds
the tape. What nobody could read back was the SEQUENCE: what was expected, what
happened, what changed, what is still open. This module builds that sequence for
one session, and it builds it out of three kinds of statement that are never
allowed to blur into one another:

* :data:`KIND_TRADER` — the trader's own entries, verbatim, in the order they
  were written. Nothing here is summarised, reconciled or averaged. Two opposing
  notes on one day are two notes; the trader is allowed to change their mind
  inside a session and the record must be able to say so.
* :data:`KIND_MEASURED` — arithmetic over completed daily bars, with the rule
  version that produced each number written beside it. A benchmark with no bars
  is ``unmeasured`` with every field ``None`` and a reason naming the symbol:
  missing data is uncertainty, never confirmation (plan.md sec 5).
* :data:`KIND_AI` — model narration. **Always empty in this packet.** Code
  computes; the model explains, later, in the narration stage. `ai_said` exists
  as a field so the surface that renders it never has to guess whether a
  sentence was written by a person or by a model.

**No note means no thesis.** A session the trader never wrote about produces an
EMPTY ``trader_said`` and a sentence saying so. The one thing this module must
never do is summarise the measured part into a sentence and attribute it to the
trader - that would manufacture a view they never held, which is exactly the
failure the three labelled kinds exist to prevent.

PURE: no Qt, no clock, no I/O, no model. Everything it reports, it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: The trader's primary scope (WISHLIST 10D). Explicitly named markets are
#: passed in by the caller; these six are the ones the story always speaks to,
#: in this order, because a story that names a different set each day cannot be
#: compared with yesterday's.
BENCHMARKS = ("SPY", "QQQ", "IWM", "VXX", "TLT", "USO")

#: The three kinds, visibly distinct wherever this is rendered.
KIND_TRADER = "trader"
KIND_MEASURED = "measured"
KIND_AI = "ai"

#: What a measured cell can say about itself.
STATUS_MEASURED = "measured"
STATUS_UNMEASURED = "unmeasured"

#: Named rule versions. They travel ON the row, so a later change to any of
#: these three formulas is visible in the evidence rather than invisible in a
#: diff - the same reason every other versioned computation on this desk names
#: itself. Never assert a literal version in a test.
RULE_CHANGE_PCT = "close_over_prior_close_pct_v1"
RULE_RANGE_ATR = "session_range_over_wilder_atr14_v1"
RULE_POSITION_VS_SMA20 = "close_minus_sma20_in_atr_v1"

#: How many completed sessions the ATR and the SMA want. Fewer is answered
#: honestly (the cell says how many bars it used) rather than refused.
ATR_LENGTH = 14
SMA_LENGTH = 20

_SIDE_ABOVE = "above"
_SIDE_BELOW = "below"
_SIDE_AT = "at"
_SIDE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class DailyStory:
    """One session, told in three kinds plus its sources.

    Every sequence is a tuple: a story handed to a Qt renderer is read on
    another thread and must not be something a caller can append to.
    """

    session_date: str
    trader_said: tuple[dict[str, Any], ...] = ()
    external_forecasts: tuple[dict[str, Any], ...] = ()
    measured: tuple[dict[str, Any], ...] = ()
    #: ALWAYS empty in this packet. See the module docstring.
    ai_said: tuple[dict[str, Any], ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def build_daily_story(
    session_date: str,
    *,
    entries: Iterable[Mapping[str, Any]] = (),
    captures: Mapping[str, Any] | None = None,
    context_row: Mapping[str, Any] | None = None,
    index_bars: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    benchmarks: Sequence[str] | None = None,
) -> DailyStory:
    """The story of one session, from the entries it is HANDED.

    It does not re-filter the entries by their stored ``session_date``. That
    field is stamped by :class:`evidence_ledger.EvidenceLedger` from the market-
    local date of the WRITE, so a note typed at 21:00 Pacific - 00:00 the next
    day in New York - is stored under tomorrow's date while being about today.
    Selecting the day's entries is the caller's job and there is one right way
    to do it (`market_journal.session_date_for(created_at)`); doing it twice, in
    two places, with two answers, is how an evening note disappears.
    """
    session = str(session_date or "").strip()
    # TJ-1 item 2: the desk's OWN rows are dropped here, defensively, as well as
    # in the readers that select the entries. A story is an AI input (TJ-4 reads
    # it) and a caller that builds its own entry list must not have to remember
    # the filter; the ledger itself is never rewritten.
    rows = [dict(row) for row in (entries or ()) if not _is_machine_entry(row)]
    # Stable: two entries written in the same second keep the order they came in.
    rows.sort(key=lambda row: str(row.get("created_at") or ""))

    said: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    for row in rows:
        rendered = _entry_row(row)
        if _is_external_forecast(row):
            forecasts.append(rendered)
        else:
            said.append(rendered)

    scope = tuple(
        str(name).strip().upper()
        for name in (benchmarks if benchmarks is not None else BENCHMARKS)
        if str(name).strip()
    )
    measured = tuple(
        _measured_cell(symbol, session, (index_bars or {}).get(symbol)) for symbol in scope
    )

    capture_ids = tuple(
        str(row.get("entry_id") or "")
        for row in rows
        if str(row.get("entry_id") or "") in dict(captures or {})
    )
    context_id = str((context_row or {}).get("event_at") or "")
    sources = {
        "entry_ids": tuple(str(row.get("entry_id") or "") for row in rows),
        "capture_entry_ids": capture_ids,
        "context_row_id": context_id,
        "benchmarks_measured": tuple(
            str(cell["symbol"]) for cell in measured if cell["status"] == STATUS_MEASURED
        ),
    }

    notes: list[str] = []
    if not said:
        notes.append(
            f"No note was written for {session}, so this story has no thesis in it - "
            "only what the desk measured."
        )
    if forecasts and not said:
        notes.append(
            "An external forecast was imported for this session. It is outside "
            "commentary, not the trader's adopted view."
        )
    unmeasured = [str(cell["symbol"]) for cell in measured if cell["status"] == STATUS_UNMEASURED]
    if unmeasured:
        notes.append(
            "No completed daily bars were available for " + ", ".join(unmeasured) + "."
        )

    return DailyStory(
        session_date=session,
        trader_said=tuple(said),
        external_forecasts=tuple(forecasts),
        measured=measured,
        ai_said=(),
        context=_context_block(session, context_row),
        sources=sources,
        notes=tuple(notes),
    )


def _is_machine_entry(row: Mapping[str, Any]) -> bool:
    """Did the desk write this row rather than the trader? (TJ-1 item 2.)

    `market_journal.is_machine_entry` is the ONE definition; it is imported
    lazily with a literal fallback for the same reason `_is_external_forecast`
    is - this module is pure by contract and is imported by the nightly slots.
    """
    try:
        from market_journal import is_machine_entry
    except Exception:  # pragma: no cover - the function is in the same repo
        return str(row.get("origin") or "") == "auto_mode_flip"
    return is_machine_entry(row)


def _is_external_forecast(row: Mapping[str, Any]) -> bool:
    try:
        from market_journal import ORIGIN_EXTERNAL_FORECAST
    except Exception:  # pragma: no cover - the constant is in the same repo
        ORIGIN_EXTERNAL_FORECAST = "external_forecast"
    return str(row.get("origin") or "") == ORIGIN_EXTERNAL_FORECAST


def _entry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """One entry as the story shows it: the words, and how to read them.

    ``predicts_this_session`` is the whole reason `written_after_the_session`
    is COMPUTED rather than claimed. The same sentence typed while the tape was
    moving and typed eight hours after the close are two different statements,
    and only one of them is a prediction.
    """
    after = bool(row.get("written_after_the_session"))
    return {
        "kind": KIND_TRADER,
        "entry_id": str(row.get("entry_id") or ""),
        "session_date": str(row.get("session_date") or ""),
        "created_at": str(row.get("created_at") or ""),
        "timeframe": str(row.get("timeframe") or ""),
        "symbols": tuple(str(item) for item in (row.get("symbols") or ())),
        "origin": str(row.get("origin") or ""),
        "text": str(row.get("text") or ""),
        "written_after_the_session": after,
        "predicts_this_session": not after,
    }


def _context_block(session: str, context_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not context_row:
        return {
            "measured": False,
            "reason": f"no daily context row exists for {session}; the desk did not measure it",
        }
    return {"measured": True, "row": dict(context_row)}


# ---------------------------------------------------------------------------
# the measured part
# ---------------------------------------------------------------------------
def _measured_cell(
    symbol: str, session: str, bars: Sequence[Mapping[str, Any]] | None
) -> dict[str, Any]:
    """One benchmark's arithmetic, or an honest refusal.

    COMPLETED BARS ONLY (plan.md sec 5): every bar dated after the session is
    dropped before a single number is computed, so tomorrow's forming bar can
    never become today's fact.
    """
    kept = _completed_bars(bars, session)
    if len(kept) < 2:
        return _unmeasured(symbol, kept, session)

    last = kept[-1]
    close = _number(last.get("close"))
    prior = _number(kept[-2].get("close"))
    high = _number(last.get("high"))
    low = _number(last.get("low"))
    if close is None or prior is None or prior == 0 or high is None or low is None:
        return _unmeasured(symbol, kept, session)

    atr = _wilder_atr(kept, ATR_LENGTH)
    sma20 = _sma(kept, SMA_LENGTH)
    change_pct = (close - prior) / prior * 100.0
    range_atr = ((high - low) / atr) if atr else None

    distance_atr = None
    side = _SIDE_UNKNOWN
    if sma20 is not None:
        side = _SIDE_ABOVE if close > sma20 else (_SIDE_BELOW if close < sma20 else _SIDE_AT)
        if atr:
            distance_atr = (close - sma20) / atr

    return {
        "kind": KIND_MEASURED,
        "symbol": symbol,
        "status": STATUS_MEASURED,
        "close": close,
        "change_pct": change_pct,
        "range_atr": range_atr,
        "atr": atr,
        "position_vs_sma20": {"sma20": sma20, "distance_atr": distance_atr, "side": side},
        "bars_through": str(last.get("dt") or "")[:10],
        "bars_used": len(kept),
        "completed_only": True,
        "reason": "",
        "rule_versions": {
            "change_pct": RULE_CHANGE_PCT,
            "range_atr": RULE_RANGE_ATR,
            "position_vs_sma20": RULE_POSITION_VS_SMA20,
        },
    }


def _unmeasured(symbol: str, kept: list[dict[str, Any]], session: str) -> dict[str, Any]:
    reason = (
        f"no completed daily bars for {symbol} through {session}"
        if not kept
        else f"only {len(kept)} completed daily bar(s) for {symbol} through {session}"
    )
    return {
        "kind": KIND_MEASURED,
        "symbol": symbol,
        "status": STATUS_UNMEASURED,
        "close": None,
        "change_pct": None,
        "range_atr": None,
        "atr": None,
        "position_vs_sma20": {"sma20": None, "distance_atr": None, "side": _SIDE_UNKNOWN},
        "bars_through": "",
        "bars_used": len(kept),
        "completed_only": True,
        "reason": reason,
        "rule_versions": {
            "change_pct": RULE_CHANGE_PCT,
            "range_atr": RULE_RANGE_ATR,
            "position_vs_sma20": RULE_POSITION_VS_SMA20,
        },
    }


def _completed_bars(
    bars: Sequence[Mapping[str, Any]] | None, session: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bar in bars or ():
        stamp = str(bar.get("dt") or bar.get("datetime") or bar.get("date") or "")[:10]
        if not stamp:
            continue
        if session and stamp > session:
            continue
        out.append({**dict(bar), "dt": stamp})
    out.sort(key=lambda row: str(row.get("dt") or ""))
    return out


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN is not a measurement


def _true_ranges(bars: Sequence[Mapping[str, Any]]) -> list[float]:
    ranges: list[float] = []
    prior_close: float | None = None
    for bar in bars:
        high = _number(bar.get("high"))
        low = _number(bar.get("low"))
        close = _number(bar.get("close"))
        if high is None or low is None:
            prior_close = close if close is not None else prior_close
            continue
        span = high - low
        if prior_close is not None:
            span = max(span, abs(high - prior_close), abs(low - prior_close))
        ranges.append(span)
        if close is not None:
            prior_close = close
    return ranges


def _wilder_atr(bars: Sequence[Mapping[str, Any]], length: int = ATR_LENGTH) -> float | None:
    """Wilder's ATR. Fewer bars than `length` is answered from what there is."""
    ranges = _true_ranges(bars)
    if not ranges:
        return None
    window = min(length, len(ranges))
    atr = sum(ranges[:window]) / window
    for value in ranges[window:]:
        atr = (atr * (length - 1) + value) / length
    return atr


def _sma(bars: Sequence[Mapping[str, Any]], length: int = SMA_LENGTH) -> float | None:
    closes = [value for value in (_number(bar.get("close")) for bar in bars) if value is not None]
    if len(closes) < length:
        return None
    window = closes[-length:]
    return sum(window) / length
