"""The persistent any-bounce watches of the Alert Center: arm, disarm, poll and fire.

Moved verbatim out of `AlertCenterPanel`, which inherits this mixin.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from chart_watch import (
    ANY_BOUNCE_KINDS,
    AnyBounceWatch,
    any_bounce_levels,
    d1_event_levels,
    evaluate_any_bounce_watch,
    save_any_bounce_watches,
)
from ui.services.m5_bar_cache import is_process_proxy, shared_m5_cache
from ui.models.bounce import BounceAlert
from swallowed import note_swallowed


class AnyBounceWatchMixin:
    """`AlertCenterPanel` methods that own the any-bounce watch store and its poll."""

    # ------------------------------------------------------------------
    # Persistent any-bounce watches (R5 section 4): one armed request per
    # symbol+side covering the whole level set. Same rails as the D1 event
    # watches - same 60s poll, same red chart-watch alert, same one-shot
    # retire - and the same single owner, because a second writer to a watch
    # store is how two components start disagreeing about what is armed.
    def any_bounce_armed_for(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        return any(watch.symbol == symbol for watch in self._any_bounce_watches)

    def _save_any_bounce_watches(self) -> None:
        if self._any_bounce_watches_path is not None:
            try:
                save_any_bounce_watches(
                    self._any_bounce_watches, self._any_bounce_watches_path
                )
            except Exception as exc:
                note_swallowed("any-bounce watches write failed", exc)

    def _toggle_any_bounce_watch(self, alert: BounceAlert) -> None:
        if alert is None or not alert.symbol:
            return
        symbol = str(alert.symbol).strip().upper()
        if self._cancel_pending_arm(("any_bounce", symbol, "")):
            return
        if self.any_bounce_armed_for(symbol):
            self.disarm_any_bounce_watch(symbol)
        elif not self._arms_async():
            self.arm_any_bounce_watch(symbol, alert.side or "long")
        else:
            side = alert.side or "long"
            self._queue_arm(
                "any_bounce",
                symbol,
                "",
                "Any bounce",
                None,
                lambda _result: self.arm_any_bounce_watch(symbol, side)
                or self.any_bounce_armed_for(symbol),
            )

    def arm_any_bounce_watch(self, symbol: str, side: str = "long") -> bool:
        symbol = str(symbol or "").strip().upper()
        side = "short" if str(side or "").strip().lower().startswith("short") else "long"
        if not symbol:
            return False
        if self.any_bounce_armed_for(symbol):
            self.statusChanged.emit(f"{symbol}: any-bounce alert already armed.")
            return False
        self._any_bounce_watches.append(
            AnyBounceWatch(
                symbol=symbol,
                side=side,
                kinds=tuple(ANY_BOUNCE_KINDS),
                armed_at=datetime.now(),
            )
        )
        self._save_any_bounce_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event(
            "arm_any_bounce",
            alert=self._arm_review_alert(symbol),
            symbol=symbol,
            side=side,
            dwell_ms=self._arm_dwell_ms(symbol),
            detail={"kinds": list(ANY_BOUNCE_KINDS)},
        )
        self.statusChanged.emit(
            f"{symbol}: any-bounce alert armed - it fires once, on whichever "
            "of your levels holds, and then disarms."
        )
        return True

    def disarm_any_bounce_watch(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        remaining = [
            watch for watch in self._any_bounce_watches if watch.symbol != symbol
        ]
        if len(remaining) == len(self._any_bounce_watches):
            return False
        self._any_bounce_watches = remaining
        self._save_any_bounce_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event("disarm_any_bounce", symbol=symbol)
        self.statusChanged.emit(f"{symbol}: any-bounce alert disarmed.")
        return True

    def _zone_arms_unknown(self) -> bool:
        """True while a proxy bot's `d1_zone_arms` have never been fetched."""
        bot = self._current_bot()
        return is_process_proxy(bot) and shared_m5_cache().peek_value(bot, "d1_zone_arms") is None

    def _any_bounce_levels_for(
        self, symbol: str, moment: datetime, *, m5_bars: list | None = None
    ) -> dict:
        """The armed level set from whatever the desk already has cached.

        The D1 side comes from the scan's zone-arms file (which is where the
        prior-anchor AVWAP now rides, R5 section 8.3); the session and hourly
        EMAs are aggregated from the cached M5 bars. Nothing here fetches.
        """
        entry = None
        try:
            bot = self._current_bot()
            if is_process_proxy(bot):
                arms = shared_m5_cache().peek_value(bot, "d1_zone_arms") or {}
            else:
                arms = getattr(bot, "d1_zone_arms", None) or {}
            candidate = arms.get(symbol)
            if isinstance(candidate, Mapping):
                entry = candidate
        except Exception:
            entry = None
        d1_levels = None
        try:
            d1_bars = self._d1_bars_for(symbol)
            if d1_bars:
                d1_levels = d1_event_levels(d1_bars, session=moment.date())
        except Exception:
            d1_levels = None
        return any_bounce_levels(
            zone_arm_entry=entry,
            m5_bars=self._m5_bars_for(symbol) if m5_bars is None else m5_bars,
            d1_levels=d1_levels,
            now=moment,
        )

    def _poll_any_bounce_watches(self, now: datetime | None = None) -> None:
        if not self._any_bounce_watches:
            return
        moment = now or datetime.now()
        # An any-bounce watch covers a SET of levels, so it has no single
        # `kind`; it files under the default 10-session window by name.
        kept, expired = self._expire_armed_watches(
            "any_bounce_watches",
            self._any_bounce_watches,
            kind_of=lambda watch: "any_bounce",
            now=moment,
        )
        if expired:
            self._any_bounce_watches = kept
            self._save_any_bounce_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
            if not self._any_bounce_watches:
                return
        remaining: list[AnyBounceWatch] = []
        triggered = []
        zone_arms_unknown = self._zone_arms_unknown()
        for watch in self._any_bounce_watches:
            hit = None
            if zone_arms_unknown or self._m5_unknown(watch.symbol):
                remaining.append(watch)  # bars not fetched yet: unknown, never judged
                continue
            try:
                # Once per watch, not twice: the levels builder and the
                # evaluation both need today's M5 bars.
                m5_bars = self._m5_bars_for(watch.symbol)
                levels = self._any_bounce_levels_for(watch.symbol, moment, m5_bars=m5_bars)
                if levels:
                    hit = evaluate_any_bounce_watch(watch, m5_bars, levels, now=moment)
            except Exception:
                hit = None
            if hit is None:
                remaining.append(watch)
            else:
                triggered.append(hit)
        self._any_bounce_watches = remaining
        if triggered:
            self._save_any_bounce_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
        for hit in triggered:
            self._record_review_event(
                "any_bounce_fired",
                symbol=hit.watch.symbol,
                side=hit.resolved_side,
                detail={"kind": hit.kind, "level": hit.level, "message": hit.message},
            )
            self.add_alert(self._chart_watch_alert(hit, moment))

