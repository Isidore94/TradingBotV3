"""Market breadth for one session from daily bars (`breadth_v1`, WISHLIST P2-8 8a).

Counts, over a universe of names, how many closed up / down / unchanged on the
session and how many closed above their SMA20 and SMA50. Display and grading
only: nothing here reaches a detector, score, alert, watchlist or Focus list.

Pure: bars in, an immutable reading out. No clock, no I/O.

Per name, point-in-time (bars after `session` are ignored):

* no bar dated `session`                   -> every fact unknown
* the bar before it is not `prior_session` -> advance/decline unknown
* fewer than N closes ending at `session`  -> the SMA-N fact unknown

An unknown name is counted as unknown, never as a decliner or "below".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .d1_environment import session_of

#: The rule's name. Every stored row carries it; a changed rule is a new name.
RULE_VERSION = "breadth_v1"

SMA_FAST = 20
SMA_SLOW = 50


@dataclass(frozen=True)
class BreadthReading:
    session: str
    rule_version: str
    names_total: int
    advancers: int
    decliners: int
    unchanged: int
    ad_unknown: int
    above_sma20: int
    sma20_known: int
    above_sma50: int
    sma50_known: int

    @property
    def pct_above_sma20(self) -> float | None:
        return _pct(self.above_sma20, self.sma20_known)

    @property
    def pct_above_sma50(self) -> float | None:
        return _pct(self.above_sma50, self.sma50_known)

    def as_row(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "rule_version": self.rule_version,
            "names_total": self.names_total,
            "advancers": self.advancers,
            "decliners": self.decliners,
            "unchanged": self.unchanged,
            "ad_unknown": self.ad_unknown,
            "above_sma20": self.above_sma20,
            "sma20_known": self.sma20_known,
            "pct_above_sma20": _round(self.pct_above_sma20),
            "above_sma50": self.above_sma50,
            "sma50_known": self.sma50_known,
            "pct_above_sma50": _round(self.pct_above_sma50),
        }


def _pct(part: int, whole: int) -> float | None:
    return (part / whole * 100.0) if whole else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _close(bar: Any) -> float | None:
    raw = bar.get("close") if isinstance(bar, Mapping) else getattr(bar, "close", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value == value and value > 0 else None


def name_facts(bars: Sequence[Any], *, session: str, prior_session: str) -> dict[str, Any]:
    """One name's facts for `session`: `change` and the two SMA sides.

    `change` is "up" / "down" / "flat" / "unknown"; `above_sma20` and
    `above_sma50` are True / False / None (None = unknown).
    """
    day = str(session or "")[:10]
    prior = str(prior_session or "")[:10]
    unknown = {"change": "unknown", "above_sma20": None, "above_sma50": None}
    cut: list[tuple[str, float]] = []
    for bar in bars or ():
        stamp = session_of(bar)
        close = _close(bar)
        if not stamp or close is None or stamp > day:
            continue
        cut.append((stamp, close))
    cut.sort(key=lambda item: item[0])
    if not cut or cut[-1][0] != day:
        return unknown
    last = cut[-1][1]
    facts = dict(unknown)
    if len(cut) >= 2 and cut[-2][0] == prior:
        before = cut[-2][1]
        facts["change"] = "up" if last > before else "down" if last < before else "flat"
    closes = [close for _stamp, close in cut]
    for length, key in ((SMA_FAST, "above_sma20"), (SMA_SLOW, "above_sma50")):
        if len(closes) >= length:
            average = sum(closes[-length:]) / length
            facts[key] = last > average
    return facts


def compute_breadth(
    bars_by_symbol: Mapping[str, Sequence[Any]],
    *,
    universe: Sequence[str],
    session: str,
    prior_session: str,
) -> BreadthReading:
    """Breadth over `universe` for `session`. A name with no bars is unknown."""
    names = sorted({str(name or "").strip().upper() for name in universe or () if str(name or "").strip()})
    up = down = flat = ad_unknown = 0
    above20 = known20 = above50 = known50 = 0
    for name in names:
        facts = name_facts(bars_by_symbol.get(name) or (), session=session, prior_session=prior_session)
        change = facts["change"]
        if change == "up":
            up += 1
        elif change == "down":
            down += 1
        elif change == "flat":
            flat += 1
        else:
            ad_unknown += 1
        if facts["above_sma20"] is not None:
            known20 += 1
            above20 += int(bool(facts["above_sma20"]))
        if facts["above_sma50"] is not None:
            known50 += 1
            above50 += int(bool(facts["above_sma50"]))
    return BreadthReading(
        session=str(session or "")[:10],
        rule_version=RULE_VERSION,
        names_total=len(names),
        advancers=up,
        decliners=down,
        unchanged=flat,
        ad_unknown=ad_unknown,
        above_sma20=above20,
        sma20_known=known20,
        above_sma50=above50,
        sma50_known=known50,
    )


__all__ = ["RULE_VERSION", "SMA_FAST", "SMA_SLOW", "BreadthReading", "compute_breadth", "name_facts"]
