"""The wall gate half of the Alert Center: measure the wall, arm its follow-up, hide only what it armed.

Moved verbatim out of `AlertCenterPanel`, which inherits this mixin.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from chart_watch import (
    D1EventWatch,
    arm_chart_watch,
    completed_session_bars,
    save_d1_event_watches,
    PULLBACK_KIND,
)
import wall_gate
import wall_gate_arms
from prev_day_gate import CLOSED as PREV_DAY_CLOSED, UNKNOWN as PREV_DAY_UNKNOWN
from project_paths import MASTER_AVWAP_AI_STATE_FILE
from ui.models.bounce import BounceAlert
from ui.panels.alert_center.gates import _bar_close


class WallGateMixin:
    """`AlertCenterPanel` methods that run the wall gate and its auto-armed follow-ups."""

    # ------------------------------------------------------------------
    # Wall gate (trader, 2026-09-23): "Don't show me stocks right at SMAs
    # (within 1 ATR). Instead auto-set an alert ... Same with trendline
    # breaks." The rule is `wall_gate`; this feeds it bars the desk already
    # holds, arms the follow-up and hides only what it could arm.

    #: At most this many symbols carry live wall follow-up watches at once.
    WALL_AUTO_ARM_CAP = 20
    #: SMA wall follow-ups: break through it, or reject and retest the 15 EMA.
    WALL_SMA_FOLLOW_UP_KINDS = ("sma_break", "ema15_reject")
    #: Trendline wall follow-up: the M15/M30 Pullback alert.
    WALL_PULLBACK_TIMEFRAMES = ("M15", "M30")
    #: The shared ai_state parse's projection key for the wall gate's records.
    WALL_TRENDLINE_PROJECTION = "alert_center.wall_gate_trendlines"

    @staticmethod
    def _wall_trendline_records(entry) -> list | None:
        """One ai_state record -> both of its trendline candidates, or None."""
        records = []
        for key in ("priority_trendline_candidate", "priority_trendline_break_candidate"):
            candidate = entry.get(key)
            if isinstance(candidate, dict):
                records.append(dict(candidate))
        return records or None

    def _wall_trendlines_for(self, symbol: str) -> list:
        """The scan's trendline records for a symbol, from the shared parse
        cache only - never a parse on this thread. Empty before the first one."""
        try:
            from d1_level_feed import peek_ai_state_projection

            feed = peek_ai_state_projection(
                self.WALL_TRENDLINE_PROJECTION,
                self._wall_trendline_records,
                Path(MASTER_AVWAP_AI_STATE_FILE),
            )
        except Exception:
            return []
        return list((feed or {}).get(symbol) or [])

    def wall_verdict(self, symbol: str, side: str):
        """The pure wall verdict for one name, memoized on its bars. Never raises."""
        symbol = str(symbol or "").strip().upper()
        side_key = str(side or "").strip().lower()
        if not symbol or side_key not in ("long", "short"):
            return wall_gate.WallVerdict(state=PREV_DAY_UNKNOWN, reason="no side")
        try:
            moment = datetime.now()
            if self._m5_unknown(symbol):
                return wall_gate.WallVerdict(state=PREV_DAY_UNKNOWN, reason="M5 bars not fetched yet")
            d1_bars = self._d1_bars_for(symbol)
            m5_bars = self._m5_bars_for(symbol)
            lines = self._wall_trendlines_for(symbol)
            stamp = (
                moment.date(),
                self._series_stamp(d1_bars),
                self._series_stamp(m5_bars),
                tuple(
                    (
                        str(line.get("line_id") or ""),
                        str(line.get("lookback_end") or ""),
                        str(line.get("break_date") or ""),
                    )
                    for line in lines
                ),
                wall_gate.WALL_GATE_ENABLED,
                wall_gate.WALL_ATR_MULTIPLE,
            )
            remembered = self._wall_measure_cache.get((symbol, side_key))
            if remembered is not None and remembered[0] == stamp:
                return remembered[1]
            completed = completed_session_bars(m5_bars, now=moment)
            price = _bar_close(completed[-1]) if completed else None
            if price is None and d1_bars:
                price = _bar_close(d1_bars[-1])
            verdict = wall_gate.wall_state(
                side_key, price, d1_bars, today=moment.date(), trendlines=lines
            )
            self._wall_measure_cache[(symbol, side_key)] = (stamp, verdict)
            return verdict
        except Exception:
            logging.debug("Wall verdict unavailable for %s.", symbol, exc_info=True)
            return wall_gate.WallVerdict(state=PREV_DAY_UNKNOWN, reason="unreadable")

    def wall_state(self, alert: BounceAlert) -> str:
        """The wall leg for one review chart: OPEN / CLOSED / UNKNOWN.

        CLOSED only when a wall is verified AND its follow-up alerts are armed
        (now or already). A wall nothing can follow up on - cap reached, a
        declined watch, a failed save, no cached M5 bars for the pullback -
        SHOWS as UNKNOWN, tagged "at wall": never hide what we cannot follow.
        """
        symbol = str(alert.symbol or "").strip().upper()
        verdict = self.wall_verdict(symbol, alert.side)
        if verdict.state != PREV_DAY_CLOSED:
            self._wall_uncovered.pop(symbol, None)
            return verdict.state
        try:
            covered, why = self._ensure_wall_follow_up(symbol, alert.side, verdict)
        except Exception:
            logging.debug("Wall follow-up failed for %s.", symbol, exc_info=True)
            covered, why = False, "arm failed"
        if not covered:
            self._wall_uncovered[symbol] = why
            self._log_wall_decision("wall_shown_uncovered", alert, verdict, why=why)
            return PREV_DAY_UNKNOWN
        self._wall_uncovered.pop(symbol, None)
        self._log_wall_decision("wall_hidden", alert, verdict)
        return PREV_DAY_CLOSED

    def _log_wall_decision(
        self, action: str, alert: BounceAlert, verdict, *, why: str = ""
    ) -> None:
        """One decision-log row per name, side, wall and day. Best-effort."""
        key = (
            datetime.now().date(),
            str(alert.symbol or "").upper(),
            str(alert.side or "").upper(),
            action,
            verdict.wall,
        )
        if key in self._wall_logged:
            return
        self._wall_logged.add(key)
        detail = {
            "reason": verdict.reason,
            "wall": verdict.wall,
            "level": verdict.level,
            "distance_atr": verdict.distance_atr,
            "atr": verdict.atr,
            "follow_up": verdict.follow_up,
        }
        if why:
            detail["why_shown"] = why
        self._record_review_event(
            action, alert=alert, queue_len=len(self._review_queue), detail=detail
        )

    def _live_wall_symbols(self) -> set[str]:
        """Symbols whose wall follow-up watch is still armed (the cap's count)."""
        live: set[str] = set()
        for symbol, entry in self._wall_arms.items():
            for kind, record in wall_gate_arms.armed_kinds(entry).items():
                if kind == PULLBACK_KIND:
                    watch_id = str(record.get("watch_id") or "")
                    if any(
                        watch.watch_id == watch_id
                        and not bool(getattr(watch, "declined", False))
                        for watch in self._chart_watches
                    ):
                        live.add(symbol)
                elif kind in self.armed_d1_event_kinds(symbol):
                    live.add(symbol)
        return live

    def _save_wall_arms(self) -> None:
        wall_gate_arms.save(self._wall_arms, self._wall_arms_path)

    def _is_wall_pullback_watch(self, watch) -> bool:
        """A Pullback alert the wall gate armed (its id is in the ledger)."""
        if str(getattr(watch, "kind", "") or "") != PULLBACK_KIND:
            return False
        entry = self._wall_arms.get(str(getattr(watch, "symbol", "") or ""))
        record = wall_gate_arms.armed_kinds(entry).get(PULLBACK_KIND) or {}
        watch_id = str(getattr(watch, "watch_id", "") or "")
        return bool(watch_id) and record.get("watch_id") == watch_id

    def _ensure_wall_follow_up(self, symbol: str, side: str, verdict) -> tuple[bool, str]:
        """Arm what follows this wall up. (covered, why-not)."""
        side = str(side or "").strip().upper()
        if side not in ("LONG", "SHORT"):
            return False, "no side"
        moment = datetime.now()
        if wall_gate_arms.prune_expired(self._wall_arms, today=moment.date()):
            self._save_wall_arms()
        if verdict.follow_up == wall_gate.FOLLOW_UP_SMA:
            return self._ensure_wall_d1_events(symbol, side, verdict, moment)
        return self._ensure_wall_pullback(symbol, side, verdict, moment)

    def _wall_cap_reached(self, symbol: str) -> bool:
        live = self._live_wall_symbols()
        return symbol not in live and len(live) >= self.WALL_AUTO_ARM_CAP

    def _remember_wall_arm(self, symbol, side, verdict, moment, kinds: dict) -> None:
        entry = self._wall_arms.get(symbol)
        if entry is None:
            entry = {
                "side": side,
                "wall": verdict.wall,
                "source_text": verdict.source_text(),
                "armed_at": moment.isoformat(timespec="seconds"),
                "kinds": {},
            }
            self._wall_arms[symbol] = entry
        entry["kinds"].update(kinds)
        self._save_wall_arms()

    def _ensure_wall_d1_events(self, symbol, side, verdict, moment) -> tuple[bool, str]:
        live_kinds = self.armed_d1_event_kinds(symbol)
        declined = wall_gate_arms.declined_kinds(self._wall_arms.get(symbol))
        wanted = self.WALL_SMA_FOLLOW_UP_KINDS
        covered = any(kind in live_kinds for kind in wanted)
        to_arm = [kind for kind in wanted if kind not in live_kinds and kind not in declined]
        if not to_arm:
            return (True, "") if covered else (False, "follow-up declined")
        if self._wall_cap_reached(symbol):
            if covered:
                return True, ""
            return False, f"cap of {self.WALL_AUTO_ARM_CAP} reached"
        new = [D1EventWatch(symbol=symbol, kind=kind, armed_at=moment) for kind in to_arm]
        candidate = list(self._d1_event_watches) + new
        if self._d1_event_watches_path is not None:
            try:
                save_d1_event_watches(candidate, self._d1_event_watches_path)
            except Exception:
                logging.debug("Wall follow-up save failed for %s.", symbol, exc_info=True)
                return (True, "") if covered else (False, "arm failed")
        self._d1_event_watches = candidate
        self._remember_wall_arm(
            symbol,
            side,
            verdict,
            moment,
            {kind: {"state": wall_gate_arms.STATE_ARMED} for kind in to_arm},
        )
        self._record_review_events(
            [
                {
                    "action": "auto_arm_watch",
                    "symbol": symbol,
                    "side": side,
                    "detail": {
                        "kind": kind,
                        "source_text": verdict.source_text(),
                        "auto": True,
                        "wall": verdict.wall,
                    },
                }
                for kind in to_arm
            ]
        )
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        return True, ""

    def _ensure_wall_pullback(self, symbol, side, verdict, moment) -> tuple[bool, str]:
        if not self._m5_bars_for(symbol):
            return False, "no cached M5 bars"
        existing = next(
            (
                watch
                for watch in self._chart_watches
                if watch.symbol == symbol and watch.kind == PULLBACK_KIND
            ),
            None,
        )
        if existing is not None:
            if bool(getattr(existing, "declined", False)):
                return False, "follow-up declined"
            # The sweep can retire its own pullback later, so it never covers a hide.
            if self._is_auto_pullback_watch(existing):
                return False, "pullback owned by the claim/Focus sweep"
            if str(getattr(existing, "side", "") or "").upper() == side:
                return True, ""
            return False, "other-side pullback armed"
        if self._wall_cap_reached(symbol):
            return False, f"cap of {self.WALL_AUTO_ARM_CAP} reached"
        watch = arm_chart_watch(
            PULLBACK_KIND,
            symbol,
            side,
            (),  # a pullback watch takes no M5 baseline
            now=moment,
            source_text=verdict.source_text(),
            timeframes=self.WALL_PULLBACK_TIMEFRAMES,
        )
        candidate = list(self._chart_watches) + [watch]
        if not self._save_chart_watches(candidate):
            return False, "arm failed"
        self._chart_watches = candidate
        self._remember_wall_arm(
            symbol,
            side,
            verdict,
            moment,
            {
                PULLBACK_KIND: {
                    "state": wall_gate_arms.STATE_ARMED,
                    "watch_id": watch.watch_id,
                }
            },
        )
        self._record_review_events(
            [
                {
                    "action": "auto_arm_watch",
                    "symbol": symbol,
                    "side": side,
                    "detail": {
                        "kind": PULLBACK_KIND,
                        "watch_id": watch.watch_id,
                        "source_text": verdict.source_text(),
                        "timeframes": list(self.WALL_PULLBACK_TIMEFRAMES),
                        "auto": True,
                        "wall": verdict.wall,
                    },
                }
            ]
        )
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        return True, ""

    def _decline_wall_d1_event(self, symbol: str, kind: str) -> None:
        """The trader turned off a D1 watch the wall gate armed: remember it."""
        entry = self._wall_arms.get(symbol)
        if kind in wall_gate_arms.armed_kinds(entry):
            entry["kinds"][kind] = {"state": wall_gate_arms.STATE_DECLINED}
            self._save_wall_arms()

