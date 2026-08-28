"""Which regime-pause rows the machine may place on M5 Focus (trader rule 2026-08-27).

On 2026-08-27 the trader reviewed 21 "holding highs" charts between 07:09 and
07:18 - on a day the tape had opened bullish - and put twelve of them on M5
Focus by hand, one click each, while 74 more charts waited. The rule that came
out of it: **a swing long holding its highs on a bullish day, or a swing short
pressing its lows on a bearish day, goes straight to M5 Focus.** The machine
makes the click the trader was making anyway.

This module is the whole decision and nothing else: given the day's directional
label and the row's side, which M5 Focus list (if any) the row belongs on. It
reads nothing, writes nothing and knows no clock. The Alert Center does the
placement (that panel owns the Focus store), stamps the entry with the auto-pick
marker so "Not today" and the desync repair can reach it, and keeps the row out
of the chart-review queue - the decision has been made.

What is deliberately NOT here:

- The mirror cases (a long holding highs on a BEARISH day, a short on a
  bullish day) stay on the review queue exactly as before. The trader named the
  with-trend cases and nothing else.
- A non-directional day (no opening read yet, or "neutral") admits nothing.
  Missing data is uncertainty, never confirmation (plan.md sec 5).
- No eviction. The regime-pause row already passed the M5 Focus adoption gate
  in the detector (prior-day extreme + session VWAP side, 2026-08-21); whether
  the name later stops holding is the queue's 15-minute rule, and a Focus entry
  stays until the trader or the desync repair says otherwise.
"""

from __future__ import annotations

from typing import Any

#: The day label a with-trend LONG needs; a SHORT needs the other family.
BULLISH_PREFIX = "bullish"
BEARISH_PREFIX = "bearish"


def day_bias(env: Any) -> str:
    """Collapse an env label to ``"bullish"``, ``"bearish"`` or ``""``.

    ``bullish_weak`` and ``bullish_strong`` are both a bullish day - the
    opening-environment record keeps the strength, this rule does not care.
    Anything else (``neutral``, blank, garbage) is "no directional read".
    """
    label = str(env or "").strip().lower()
    if label.startswith(BULLISH_PREFIX):
        return BULLISH_PREFIX
    if label.startswith(BEARISH_PREFIX):
        return BEARISH_PREFIX
    return ""


def focus_side_for(env: Any, side: Any) -> str | None:
    """The M5 Focus side a regime-pause row auto-joins, or None.

    ``"long"`` for a LONG row on a bullish day, ``"short"`` for a SHORT row on
    a bearish day. Everything else - counter-trend, non-directional day,
    unknown side - is None, which means "leave it on the review queue".
    """
    bias = day_bias(env)
    side_text = str(side or "").strip().lower()
    if bias == BULLISH_PREFIX and side_text == "long":
        return "long"
    if bias == BEARISH_PREFIX and side_text == "short":
        return "short"
    return None
