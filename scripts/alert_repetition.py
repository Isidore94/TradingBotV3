"""R4 section 6.3: display-only repetition and open-burst control.

The trader's ask, 2026-08-14: *"I don't want to be constantly seeing the same
stocks over and over ... less spam and more quality ... I def find bangers
though."*

That last clause is the whole constraint. This module decides how a feed ROW is
presented. It does not decide whether an alert exists, whether it is real, or
whether anything downstream hears about it:

- No detector, score, or threshold is touched.
- The evidence streams, History, and the AWAY phone push read the alert list,
  not this ledger, so a folded row is still fully recorded and still pushed.
- ``review_policy.json`` is not involved, and **no suppression field exists
  here or anywhere in this chain** - a folded alert is displayed differently,
  never withheld.
- Focus-privileged and trader-armed hits are exempt from every rule below.
  Quieting an alarm the trader set themselves is the one failure this section
  is not permitted to cause, so the exemption is checked FIRST, before the
  digest window and before any fold.

Pure logic, no Qt and no I/O, so the decisions are testable without a desk.

The three tunables were confirmed by the trader on 2026-08-16 and are decisions,
not defaults to re-litigate: a 30-minute open digest (zero disables), no reason
prompt on a like, and an exhaustive escalation list of exactly three entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

#: The open-burst digest window, in minutes. Trader-confirmed 2026-08-16.
#: Zero disables the digest entirely.
OPEN_DIGEST_MINUTES = 30
OPEN_DIGEST_SETTING = "alert_open_digest_minutes"

#: A row's outcomes.
ACTION_NEW = "new"  # a row of its own, sounds if it otherwise would
ACTION_FOLD = "fold"  # updates the existing row in place, silent
ACTION_ESCALATE = "escalate"  # updates in place, re-floats and re-sounds
ACTION_DIGEST = "digest"  # joins the open-burst digest row, silent

#: Tier strength. Untiered is 0, so an untiered repeat can never read as an
#: upgrade over a tiered one - and a tiered alert after an untiered one can.
_TIER_RANK = {"S": 4, "A": 3, "B": 2, "C": 1}


@dataclass
class RepeatDecision:
    """What the feed should do with one alert."""

    action: str
    repeat_count: int
    first_seen: datetime | None = None
    best_tier: str = ""
    reason: str = ""

    @property
    def sounds(self) -> bool:
        """Whether this outcome may make a noise.

        A fold and a digest never do. A new row and an escalation are allowed
        to - the caller still applies its own loudness rules on top, so this is
        a permission, not an instruction.
        """
        return self.action in (ACTION_NEW, ACTION_ESCALATE)

    @property
    def is_repeat(self) -> bool:
        return self.repeat_count > 1


@dataclass
class _Row:
    first_seen: datetime
    count: int = 1
    best_rank: int = 0
    best_tier: str = ""
    had_proven: bool = False


@dataclass
class RepetitionLedger:
    """One market day's worth of feed rows, keyed by symbol + side.

    Deliberately NOT keyed by alert text or trigger type: the trader's
    complaint was about seeing the same *ticker* repeatedly, and two different
    bounce types on one name at one moment are still one name to look at. The
    side stays in the key because a long and a short on one symbol are
    different trades, not a repeat of each other.
    """

    session_open: datetime | None = None
    market_date: str = ""
    digest_minutes: int = OPEN_DIGEST_MINUTES
    _rows: dict[tuple[str, str], _Row] = field(default_factory=dict)
    _digest: list[str] = field(default_factory=list)

    # -- day scoping ----------------------------------------------------
    def set_market_date(self, market_date: str, *, session_open: datetime | None = None) -> None:
        """Roll to a new market day, clearing everything.

        'One row per symbol per side per market DAY' means yesterday's sighting
        cannot silence the first alert of a new session.
        """
        market_date = str(market_date or "")
        if market_date == self.market_date:
            return
        self.market_date = market_date
        self._rows.clear()
        self._digest.clear()
        if session_open is not None:
            self.session_open = session_open

    def digest_symbols(self) -> list[str]:
        """What the open-burst digest row currently holds, in arrival order."""
        return list(self._digest)

    def reset(self) -> None:
        self._rows.clear()
        self._digest.clear()

    # -- the decision ---------------------------------------------------
    def _in_digest_window(self, now: datetime) -> bool:
        minutes = int(self.digest_minutes or 0)
        if minutes <= 0 or self.session_open is None:
            # Fail open. If the digest is disabled, or we cannot say when the
            # session started, digesting anyway would quietly become
            # suppression - which this section is explicitly not.
            return False
        return self.session_open <= now < self.session_open + timedelta(minutes=minutes)

    def consider(
        self,
        *,
        symbol: str,
        side: str = "",
        tier: str = "",
        is_proven: bool = False,
        privileged: bool = False,
        now: datetime | None = None,
    ) -> RepeatDecision:
        """How to present this alert. ``privileged`` short-circuits everything.

        ``privileged`` is the caller's word for "Focus-privileged, trader-armed,
        entry-assist or ready-D1" - anything the trader either put on a list or
        armed by hand. Those always surface and always sound; they are never
        folded into a stale row and never swept into the digest.
        """
        now = now or datetime.now()
        symbol = str(symbol or "").strip().upper()
        key = (symbol, str(side or "").strip().upper())
        row = self._rows.get(key)
        rank = _TIER_RANK.get(str(tier or "").strip().upper(), 0)

        if row is None:
            row = _Row(first_seen=now, best_rank=rank, best_tier=str(tier or "").upper())
            row.had_proven = bool(is_proven)
            self._rows[key] = row
            if privileged:
                return RepeatDecision(ACTION_NEW, 1, now, row.best_tier, "trader-armed or focus")
            if is_proven:
                return RepeatDecision(ACTION_NEW, 1, now, row.best_tier, "proven")
            if self._in_digest_window(now):
                if symbol and symbol not in self._digest:
                    self._digest.append(symbol)
                return RepeatDecision(ACTION_DIGEST, 1, now, row.best_tier, "open burst")
            return RepeatDecision(ACTION_NEW, 1, now, row.best_tier, "first sighting")

        row.count += 1

        if privileged:
            # Not folded, not counted against - the trader is waiting on this.
            return RepeatDecision(
                ACTION_NEW, row.count, row.first_seen, row.best_tier, "trader-armed or focus"
            )

        # The escalation list, exhaustive by trader decision 2026-08-16:
        # a strictly higher best tier and the FIRST proven. "First" matters -
        # without it a repeatedly-proven name re-sounds forever, which is the
        # exact spam this exists to remove. The third member of this list was
        # the first BANGER; that class was retired 2026-09-01 (trader: "We can
        # probably remove this because idk what it is") - it had a matcher and
        # no producer, so the branch could never fire.
        reasons: list[str] = []
        if rank > row.best_rank:
            reasons.append(f"tier {tier}")
            row.best_rank = rank
            row.best_tier = str(tier or "").upper()
        if is_proven and not row.had_proven:
            reasons.append("first proven")
        row.had_proven = row.had_proven or bool(is_proven)

        if reasons:
            return RepeatDecision(
                ACTION_ESCALATE, row.count, row.first_seen, row.best_tier, ", ".join(reasons)
            )
        return RepeatDecision(
            ACTION_FOLD, row.count, row.first_seen, row.best_tier, "repeat"
        )


def configured_digest_minutes() -> int:
    """The open-digest window from machine-local settings, clamped.

    An unreadable settings file falls back to the trader-confirmed default
    rather than to zero: silently disabling the control would be a surprise,
    but silently digesting all day would be worse, so the failure mode leans
    toward the value the trader actually chose.
    """
    try:
        from project_paths import get_local_setting

        raw = get_local_setting(OPEN_DIGEST_SETTING, OPEN_DIGEST_MINUTES)
    except Exception:
        return OPEN_DIGEST_MINUTES
    try:
        minutes = int(float(raw))
    except (TypeError, ValueError):
        return OPEN_DIGEST_MINUTES
    return max(0, min(minutes, 6 * 60))
