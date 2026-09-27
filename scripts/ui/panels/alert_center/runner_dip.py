"""The runner dip watch on the Alert Center (p9, the trader 2026-09-27).

Rides the 60 s D1 watch tick. The runner file is read by a worker (`long_setups_store`
cache); this thread reads memory and the cached M5 bars only. A fire is recorded for grading,
posted as a chart-watch row, and pushed to the phone only in AWAY.
"""

from __future__ import annotations

from datetime import datetime

import long_setups_store
import runner_dip_watch
from completed_bars import completed_m5_bars
from swallowed import note_swallowed
from ui.models.bounce import CHART_WATCH_TAG, BounceAlert

#: The chart-watch kind a runner-dip fire carries in its payload.
RUNNER_DIP_KIND = "runner_dip"


class RunnerDipWatchMixin:
    """`AlertCenterPanel` methods that poll the armed runner names and fire once per session."""

    def runner_dip_status_text(self) -> str:
        return str(getattr(self, "_runner_dip_status", "") or "")

    def _set_runner_dip_status(self, text: str) -> None:
        if text == self.runner_dip_status_text():
            return
        self._runner_dip_status = text
        label = getattr(self, "runner_dips_label", None)
        if label is not None:
            label.setText(text)
            label.setVisible(bool(text))

    def _poll_runner_dips(self, now: datetime | None = None) -> None:
        """Evaluate each armed runner on its cached M5 bars; fire once per name per session."""
        try:
            long_setups_store.refresh_runner_dip_async()
        except Exception as exc:  # the cached payload stays
            note_swallowed("runner dip watch refresh not started", exc, quiet=True)
        moment = now or datetime.now()
        payload = long_setups_store.runner_dip_snapshot()
        self._set_runner_dip_status(runner_dip_watch.status_line(payload, today=moment.date()))
        members = runner_dip_watch.armed_members(payload, today=moment.date())
        if not members:
            return
        fired = getattr(self, "_runner_dips_fired", None)
        if fired is None:
            fired = self._runner_dips_fired = set()
        day = moment.date().isoformat()
        for member in members:
            symbol = str(member.get("symbol") or "").strip().upper()
            if not symbol or (symbol, day) in fired:
                continue
            # Two sessions: the 20-bar M5 ATR needs warm-up bars early in the day.
            bars = self._m5_bars_for(symbol, sessions=2)
            if not bars:
                continue
            try:
                hit = runner_dip_watch.evaluate(member, bars, now=moment)
            except Exception as exc:
                note_swallowed("runner dip evaluation failed", exc, quiet=True)
                continue
            if hit is None:
                continue
            fired.add((symbol, day))
            self._fire_runner_dip(hit, moment)

    def _spy_m5_close(self, moment: datetime):
        """SPY's last completed cached M5 close (the grading's SPY baseline), or None."""
        try:
            done = completed_m5_bars(self._m5_bars_for("SPY"), now=moment)
            return float(done[-1]["close"]) if done else None
        except Exception:
            return None

    def _fire_runner_dip(self, hit, moment: datetime) -> None:
        detail = runner_dip_watch.fire_detail(hit, spy_price=self._spy_m5_close(moment))
        self._record_review_event(runner_dip_watch.FIRED_ACTION, symbol=hit.symbol, side="LONG", detail=detail)
        if self._auto_mode_now() == "AWAY":
            self._push_runner_dip(hit)
        self.add_alert(BounceAlert(
            time_text=moment.strftime("%H:%M:%S"),
            symbol=hit.symbol,
            side="LONG",
            trigger=hit.line,
            timeframe="M5",
            tag=CHART_WATCH_TAG,
            raw_text=f"CHART WATCH {hit.symbol} (LONG): {hit.line}",
            payload={"chart_watch_kind": RUNNER_DIP_KIND, **detail},
        ))

    def _push_runner_dip(self, hit) -> None:
        """One phone push per fire through the armed-watch sender (AWAY only, the caller's gate)."""
        service = getattr(self, "price_alert_service", None)
        if service is None:
            return
        try:
            service.notify_armed_watch(
                watch_id=f"{RUNNER_DIP_KIND}:{hit.symbol}:{hit.session}",
                title=f"Runner dip: {hit.symbol}",
                message=hit.line,
            )
        except Exception as exc:
            note_swallowed("runner dip push failed", exc, quiet=True)
