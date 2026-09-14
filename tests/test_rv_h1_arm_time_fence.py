"""RV-H1-ARM-TIME - a new arm never fires on an old bounce (review blocker B2).

WS-10C shipped `chart_watch.evaluate_h1_bars` running the frozen
`h1_ema_bounce_v1` rule over the whole H1 series and returning whatever it
says.  The rule anchors its verdict at the LAST completed bar, so a series
whose last bar already holds a confirmed reclaim fires the instant a watch is
armed - on a bounce that finished BEFORE the trader armed.  The review
reproduced it: golden bars whose confirm bar is 11:30-12:30, `armed_at` 13:30,
one poll -> `new_alerts 1  watches_left 0`.

The M5 kinds on this very store already fence exactly this
(`chart_watch._evaluate_extreme`: `_bar_end(bar) <= armed_at` is a PRE-ARM
bar).  These tests pin the same convention for the H1 kind:

* warm-up keeps every bar - the EMA and the ATR are still computed over the
  whole series, old bars included;
* only a POST-ARM event finishes the watch.  The event bar is the rule's
  `confirm_bar_dt` (the reclaim bar for a confirmation, the closing-through bar
  for an invalidation) and it is eligible when its END is strictly after
  `armed_at`; a bar ending exactly at `armed_at` is pre-arm;
* a bar that was FORMING when the trader armed (started before, ends after) is
  post-arm once it completes - the same courtesy the M5 kinds give;
* a pre-arm confirmation or invalidation is NOT an event: the watch stays
  armed, nothing fires, no `watch_fired` / `watch_invalidated` row, no push,
  no feed row - the poll treats it exactly like `awaiting_reclaim`;
* same-candle ambiguity stays ambiguous whatever the arm time (frozen rule);
* the comparison ATTACHES the desk's market-local zone to a naive value and
  keeps an aware value as the instant it is.  It never strips: stripping a
  zone three hours behind the desk turns a pre-arm arm into a post-arm one.

`scripts/indicators/h1_ema_bounce.py` is the frozen rule sheet and is NOT
touched - `pre_arm` is a `chart_watch`-level verdict, never an indicator
reason.

Every test here drives a real seam: the panel's `_poll_h1_bounce_watches` with
`ChartWatch` rows carrying an explicit `armed_at`, `chart_watch.evaluate_h1_bars`,
and `chart_watch.load_chart_watches` for the restart case.  Nothing touches the
network: the M5 bars are the pinned golden series and the yfinance fallback
cache is replaced by `None`.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_ws_10c_h1_retester import (  # noqa: E402
    ARM_REASON_LONG,
    GOLDEN_CONFIRM_DT,
    GOLDEN_TOUCH_DT,
    WATCH_KIND,
    _bucket_dt,
    _close_for_ema_offset,
    _ema_series,
    _ramp_bar,
    _ramp_closes,
    golden_long_h1_bars,
    golden_m5_series,
)

#: The golden confirm bar is 11:30-12:30 on 2026-08-26 (a 60-minute bucket, not
#: the short closing one), so its END - the moment that decides pre/post arm -
#: is 12:30.
GOLDEN_CONFIRM_END = GOLDEN_CONFIRM_DT + timedelta(hours=1)
#: The review's own arm time: an hour after the bounce had finished printing.
LATE_ARM = datetime(2026, 8, 26, 13, 30)
#: Well before the touch bar, so the whole episode is post-arm.
EARLY_ARM = datetime(2026, 8, 26, 9, 0)


# ---------------------------------------------------------------------------
# Fixtures: the golden series, and the same shape printed a session later.
# ---------------------------------------------------------------------------
def _retest_h1_bars(count: int) -> tuple[list[dict], list[float]]:
    """`count` completed H1 bars: a ramp, a touch at -2 and a reclaim at -1.

    The generalisation of `test_ws_10c_h1_retester.golden_long_h1_bars`, which
    is this function at `count = 55`; the premise test below pins them
    bar-for-bar so this file cannot drift off the packet's golden.  A larger
    `count` walks the same episode forward onto a later session, which is how a
    POST-ARM bounce is built for a watch armed after the golden one.
    """
    closes = _ramp_closes(count - 2)
    ema_before_touch = _ema_series(closes)[-1]
    closes.append(_close_for_ema_offset(ema_before_touch, -0.01))
    ema_at_touch = _ema_series(closes)[-1]
    reclaim_close = _ramp_closes(count)[-1]
    closes.append(reclaim_close)
    ema = _ema_series(closes)

    bars: list[dict] = []
    for index, close in enumerate(closes):
        if index == count - 2:
            bars.append(
                {
                    "dt": _bucket_dt(index),
                    "open": close + 0.05,
                    "high": close + 0.20,
                    "low": ema_at_touch - 0.02,
                    "close": close,
                }
            )
        elif index == count - 1:
            bars.append(
                {
                    "dt": _bucket_dt(index),
                    "open": close - 1.0,
                    "high": close + 0.20,
                    "low": close - 1.5,
                    "close": close,
                }
            )
        else:
            bars.append(_ramp_bar(index, close))
    return bars, ema


def _invalidated_h1_bars() -> list[dict]:
    """The golden through its touch, then a close 1.5 ATR the wrong way.

    The same series `tests/test_ws_10c_h1_retester_builder.py` uses for the
    invalidation path, so "it disarms" is the shipped behaviour being fenced,
    not a new fixture.  The closing-through bar is index 54 - 11:30, ending
    12:30 - exactly like the golden's reclaim bar.
    """
    bars, ema = golden_long_h1_bars()
    broken = list(bars[:54])
    close = _close_for_ema_offset(ema[53], -1.5)
    broken.append(
        {
            "dt": _bucket_dt(54),
            "open": close + 0.30,
            "high": close + 0.40,
            "low": close - 0.10,
            "close": close,
        }
    )
    return broken


def _ambiguous_h1_bars() -> list[dict]:
    """A single candle that both tags the EMA and closes half a point above it."""
    closes = _ramp_closes(54)
    ema53 = _ema_series(closes)[-1]
    closes.append(_close_for_ema_offset(ema53, 0.50))
    ema = _ema_series(closes)
    bars = [_ramp_bar(index, close) for index, close in enumerate(closes[:-1])]
    bars.append(
        {
            "dt": _bucket_dt(54),
            "open": ema[54] - 0.03,
            "high": closes[54] + 0.10,
            "low": ema[54] - 0.05,
            "close": closes[54],
        }
    )
    return bars


def test_the_generalised_fixture_reproduces_the_packets_golden_bar_for_bar():
    """Premise check: `_retest_h1_bars(55)` IS `golden_long_h1_bars()`."""
    mine, my_ema = _retest_h1_bars(55)
    golden, golden_ema = golden_long_h1_bars()

    assert len(mine) == len(golden) == 55
    assert mine == golden
    assert my_ema == golden_ema
    assert mine[-1]["dt"] == GOLDEN_CONFIRM_DT
    assert mine[-2]["dt"] == GOLDEN_TOUCH_DT

    # The later-session episode used below lands on 2026-08-27, one session on.
    later, _ = _retest_h1_bars(62)
    assert later[-1]["dt"] == datetime(2026, 8, 27, 11, 30)
    assert later[-2]["dt"] == datetime(2026, 8, 27, 10, 30)


# ---------------------------------------------------------------------------
# The desk: a real panel, a real poll, watches with an explicit armed_at.
# ---------------------------------------------------------------------------
def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless without Qt
        pytest.skip("PySide6 not installed")
    return QApplication.instance() or QApplication([])


class _Phone:
    """Stands in for the ONE armed price-alert sender. Never reaches ntfy."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def notify_armed_watch(self, *, watch_id, title, message):
        self.calls.append({"watch_id": watch_id, "title": title, "message": message})
        return {"ok": True}


class _Desk:
    """A live `AlertCenterPanel` wired to a fixed H1 series and a fake phone."""

    def __init__(self, panel, phone, events, holder):
        self.panel = panel
        self.phone = phone
        self.events = events
        self._holder = holder

    def set_h1_bars(self, h1_bars) -> None:
        self._holder["m5"] = golden_m5_series(list(h1_bars))

    def arm(self, armed_at, *, watch_id, side="LONG", symbol="AAPL"):
        from chart_watch import ChartWatch

        watch = ChartWatch(
            symbol=symbol,
            kind=WATCH_KIND,
            armed_at=armed_at,
            side=side,
            source_text="",
            watch_id=watch_id,
            reason=ARM_REASON_LONG,
        )
        self.panel._chart_watches = list(self.panel._chart_watches) + [watch]
        return watch

    def poll(self, now) -> None:
        self.panel._poll_h1_bounce_watches(now=now)

    @property
    def armed_ids(self) -> list[str]:
        return [
            watch.watch_id
            for watch in self.panel._chart_watches
            if watch.kind == WATCH_KIND
        ]

    @property
    def alert_count(self) -> int:
        return len(self.panel._alerts)

    def event_names(self) -> list[str]:
        return [name for name, _detail in self.events]


def _desk(monkeypatch, tmp_path, h1_bars) -> _Desk:
    _qt_app()
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    panel._chart_watches_path = tmp_path / "chart_watches.json"
    panel._chart_watches = []

    holder = {"m5": golden_m5_series(list(h1_bars))}
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: list(holder["m5"]))
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])
    # No yfinance fallback: the cached M5 series above already clears the
    # 45-bar warm-up, and a test that could reach the network is a bug.
    monkeypatch.setattr(panel, "_h1_history_cache", lambda: None)

    phone = _Phone()
    panel.price_alert_service = phone

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        panel,
        "_record_review_event",
        lambda name, **kw: events.append((name, kw)),
    )
    return _Desk(panel, phone, events, holder)


# ---------------------------------------------------------------------------
# 1. The review's reproduction
# ---------------------------------------------------------------------------
def test_a_watch_armed_after_the_bounce_never_fires_and_stays_armed(
    monkeypatch, tmp_path
):
    """The review's own reproduction, inverted into the rule we want.

    Golden confirm bar 11:30, ending 12:30; the trader arms at 13:30; one poll.
    Today: `new_alerts 1  watches_left 0`.  The bounce the alert names finished
    a full hour before the trader pressed the button.
    """
    h1, _ = golden_long_h1_bars()
    desk = _desk(monkeypatch, tmp_path, h1)
    desk.arm(LATE_ARM, watch_id="rv-late-arm")

    desk.poll(now=LATE_ARM)

    assert desk.alert_count == 0
    assert desk.armed_ids == ["rv-late-arm"]
    assert desk.phone.calls == []
    assert "watch_fired" not in desk.event_names()
    assert "watch_invalidated" not in desk.event_names()


# ---------------------------------------------------------------------------
# 2. What must keep working: a bounce that completes after the arm
# ---------------------------------------------------------------------------
def test_a_confirmation_after_the_arm_still_fires_once_and_disarms(
    monkeypatch, tmp_path
):
    """The fence keeps every bar for warm-up: this is the same 55-bar series."""
    h1, _ = golden_long_h1_bars()
    desk = _desk(monkeypatch, tmp_path, h1)
    desk.arm(EARLY_ARM, watch_id="rv-early-arm")

    desk.poll(now=GOLDEN_CONFIRM_END)

    assert desk.alert_count == 1
    assert desk.armed_ids == []
    assert [call["watch_id"] for call in desk.phone.calls] == ["rv-early-arm"]
    assert "watch_fired" in desk.event_names()

    payload = dict(desk.panel._alerts[0].payload or {})
    assert str(payload.get("confirm_bar_dt") or "") == GOLDEN_CONFIRM_DT.isoformat()


# ---------------------------------------------------------------------------
# 3. The boundary, both edges
# ---------------------------------------------------------------------------
def test_the_arm_boundary_is_the_event_bars_end_and_is_inclusive_on_the_pre_arm_side(
    monkeypatch, tmp_path
):
    """`bar_end <= armed_at` is pre-arm; one minute earlier is post-arm.

    The existing armed-watch convention, copied from `_evaluate_extreme`, so
    the two kinds on this one store cannot disagree about what "already
    happened" means.
    """
    h1, _ = golden_long_h1_bars()

    on_the_edge = _desk(monkeypatch, tmp_path / "edge", h1)
    on_the_edge.arm(GOLDEN_CONFIRM_END, watch_id="rv-edge-exact")
    on_the_edge.poll(now=GOLDEN_CONFIRM_END)

    assert on_the_edge.alert_count == 0
    assert on_the_edge.armed_ids == ["rv-edge-exact"]
    assert on_the_edge.phone.calls == []

    a_minute_earlier = _desk(monkeypatch, tmp_path / "inside", h1)
    a_minute_earlier.arm(
        GOLDEN_CONFIRM_END - timedelta(minutes=1), watch_id="rv-edge-inside"
    )
    a_minute_earlier.poll(now=GOLDEN_CONFIRM_END)

    assert a_minute_earlier.alert_count == 1
    assert a_minute_earlier.armed_ids == []


def test_a_candle_that_was_forming_when_the_trader_armed_still_fires(
    monkeypatch, tmp_path
):
    """Armed 11:45, inside the 11:30-12:30 reclaim bar: post-arm once it closes.

    The same courtesy the M5 kinds already give - the trader armed while the
    bar the event lands on was still being written, so the event is theirs.
    """
    h1, _ = golden_long_h1_bars()
    desk = _desk(monkeypatch, tmp_path, h1)
    desk.arm(datetime(2026, 8, 26, 11, 45), watch_id="rv-forming")

    desk.poll(now=GOLDEN_CONFIRM_END)

    assert desk.alert_count == 1
    assert desk.armed_ids == []


# ---------------------------------------------------------------------------
# 4. Invalidation is an event too, and obeys the same fence
# ---------------------------------------------------------------------------
def test_a_pre_arm_invalidation_leaves_the_watch_armed_and_writes_no_row(
    monkeypatch, tmp_path
):
    """The level failed BEFORE the trader armed: that is not their episode.

    The frozen rule is not asked a different question - while that closing-through
    bar sits inside the age window the rule keeps saying `invalidated`, and the
    watch simply waits, exactly as it waits on `awaiting_reclaim`.
    """
    desk = _desk(monkeypatch, tmp_path, _invalidated_h1_bars())
    desk.arm(LATE_ARM, watch_id="rv-pre-arm-invalidation")

    desk.poll(now=LATE_ARM)

    assert desk.armed_ids == ["rv-pre-arm-invalidation"]
    assert "watch_invalidated" not in desk.event_names()
    assert desk.alert_count == 0
    assert desk.phone.calls == []


def test_a_post_arm_invalidation_still_disarms_and_records_its_row(
    monkeypatch, tmp_path
):
    """The guard on the other side: a failure the trader waited through ends it."""
    desk = _desk(monkeypatch, tmp_path, _invalidated_h1_bars())
    desk.arm(EARLY_ARM, watch_id="rv-post-arm-invalidation")

    desk.poll(now=GOLDEN_CONFIRM_END)

    assert desk.armed_ids == []
    assert "watch_invalidated" in desk.event_names()
    assert desk.alert_count == 0  # never an alert, never a buzz
    assert desk.phone.calls == []


# ---------------------------------------------------------------------------
# 5. The frozen rule's own refusal is untouched by the arm time
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("armed_at", [EARLY_ARM, LATE_ARM])
def test_same_candle_ambiguity_never_fires_whatever_the_arm_time(armed_at):
    from chart_watch import ChartWatch, evaluate_h1_bars

    watch = ChartWatch(
        symbol="AAPL",
        kind=WATCH_KIND,
        armed_at=armed_at,
        side="LONG",
        watch_id="rv-ambiguous",
        reason=ARM_REASON_LONG,
    )

    result = evaluate_h1_bars(
        watch, _ambiguous_h1_bars(), now=GOLDEN_CONFIRM_END
    )

    assert result is not None
    assert result.fired is False
    assert result.reason != "invalidated"


# ---------------------------------------------------------------------------
# 6. The seam itself: evaluate_h1_bars
# ---------------------------------------------------------------------------
def test_the_evaluation_seam_refuses_a_pre_arm_confirmation(monkeypatch):
    """`chart_watch.evaluate_h1_bars` is where the poll reads its verdict."""
    from chart_watch import ChartWatch, evaluate_h1_bars

    h1, _ = golden_long_h1_bars()

    def _watch(armed_at, watch_id):
        return ChartWatch(
            symbol="AAPL",
            kind=WATCH_KIND,
            armed_at=armed_at,
            side="LONG",
            watch_id=watch_id,
            reason=ARM_REASON_LONG,
        )

    late = evaluate_h1_bars(_watch(LATE_ARM, "rv-seam-late"), h1, now=LATE_ARM)
    assert late is not None
    assert late.fired is False

    early = evaluate_h1_bars(
        _watch(EARLY_ARM, "rv-seam-early"), h1, now=GOLDEN_CONFIRM_END
    )
    assert early is not None
    assert early.fired is True
    assert early.confirm_bar_dt == GOLDEN_CONFIRM_DT


def test_the_evaluation_seam_refuses_a_pre_arm_invalidation():
    from chart_watch import ChartWatch, evaluate_h1_bars

    watch = ChartWatch(
        symbol="AAPL",
        kind=WATCH_KIND,
        armed_at=LATE_ARM,
        side="LONG",
        watch_id="rv-seam-invalidation",
        reason=ARM_REASON_LONG,
    )

    result = evaluate_h1_bars(watch, _invalidated_h1_bars(), now=LATE_ARM)

    assert result is None or result.reason != "invalidated"


# ---------------------------------------------------------------------------
# 7. Attach, never strip
# ---------------------------------------------------------------------------
def _market_local(moment: datetime) -> datetime:
    from market_session import get_market_local_timezone

    tz, _name = get_market_local_timezone()
    return moment.replace(tzinfo=tz)


def _same_instant_three_hours_west(moment: datetime) -> datetime:
    """The same instant, written on a clock three hours behind the desk's.

    Strip the zone off this value and you get a wall time three hours EARLIER
    than the desk's, which turns a pre-arm arm into a post-arm one - the exact
    failure `_naive` would introduce, and why the fence attaches instead.
    """
    local = _market_local(moment)
    offset = local.utcoffset()
    assert offset is not None
    shifted = local.astimezone(timezone(offset - timedelta(hours=3)))
    assert shifted.hour == (moment.hour - 3) % 24
    return shifted


def test_an_aware_arm_time_is_the_instant_it_is_and_is_never_stripped(
    monkeypatch, tmp_path
):
    """One instant, three spellings, one verdict.

    `armed_at` is naive market-local in the store; a caller that hands the poll
    an aware value must not change the answer, and the conversion must go
    through the desk's zone (`autopilot_core._gate_moment`'s pattern), never
    `chart_watch._naive`, which strips.
    """
    h1, _ = golden_long_h1_bars()
    spellings = {
        "naive": GOLDEN_CONFIRM_END,
        "aware_market_local": _market_local(GOLDEN_CONFIRM_END),
        "aware_three_hours_west": _same_instant_three_hours_west(GOLDEN_CONFIRM_END),
    }

    verdicts = {}
    for index, (name, armed_at) in enumerate(sorted(spellings.items())):
        desk = _desk(monkeypatch, tmp_path / f"tz{index}", h1)
        desk.arm(armed_at, watch_id=f"rv-tz-{name}")
        desk.poll(now=GOLDEN_CONFIRM_END)
        verdicts[name] = (desk.alert_count, tuple(desk.armed_ids))

    assert verdicts["naive"] == (0, ("rv-tz-naive",))
    assert verdicts["aware_market_local"] == (0, ("rv-tz-aware_market_local",))
    assert verdicts["aware_three_hours_west"] == (
        0,
        ("rv-tz-aware_three_hours_west",),
    )


# ---------------------------------------------------------------------------
# 8. Restart, and disarm/re-arm
# ---------------------------------------------------------------------------
def test_the_same_armed_at_survives_a_restart_and_the_old_bounce_is_still_old(
    monkeypatch, tmp_path
):
    """`armed_at` round-trips through `chart_watches.json`; the fence reads the
    SAME value after a reload, so a desk restart is not a second chance at a
    bounce that already finished.  A genuinely new reclaim, printed the next
    session, still fires it."""
    from chart_watch import ChartWatch, load_chart_watches, save_chart_watches

    store = tmp_path / "chart_watches.json"
    armed = ChartWatch(
        symbol="AAPL",
        kind=WATCH_KIND,
        armed_at=LATE_ARM,
        side="LONG",
        source_text="",
        watch_id="rv-restart",
        reason=ARM_REASON_LONG,
    )
    save_chart_watches([armed], store, market_date=LATE_ARM.date())

    reloaded = load_chart_watches(store, market_date=date(2026, 8, 27))
    survivor = next(watch for watch in reloaded if watch.kind == WATCH_KIND)
    assert survivor.armed_at == LATE_ARM
    assert survivor.watch_id == "rv-restart"

    old_bounce, _ = golden_long_h1_bars()
    desk = _desk(monkeypatch, tmp_path, old_bounce)
    desk.panel._chart_watches = [survivor]

    desk.poll(now=LATE_ARM)
    assert desk.alert_count == 0
    assert desk.armed_ids == ["rv-restart"]

    # The next session prints a real one, after the arm.
    later, _ = _retest_h1_bars(62)
    desk.set_h1_bars(later)
    desk.poll(now=datetime(2026, 8, 27, 12, 30))

    assert desk.alert_count == 1
    assert desk.armed_ids == []
    payload = dict(desk.panel._alerts[0].payload or {})
    assert (
        str(payload.get("confirm_bar_dt") or "")
        == datetime(2026, 8, 27, 11, 30).isoformat()
    )


def test_re_arming_after_a_fire_does_not_fire_again_on_the_same_bounce(
    monkeypatch, tmp_path
):
    """Disarm + re-arm is a new `watch_id` with a new `armed_at`, and the
    bounce that already fired is pre-arm to the new watch too."""
    h1, _ = golden_long_h1_bars()
    desk = _desk(monkeypatch, tmp_path, h1)
    desk.arm(EARLY_ARM, watch_id="rv-first-arm")

    desk.poll(now=GOLDEN_CONFIRM_END)
    assert desk.alert_count == 1
    assert desk.armed_ids == []

    desk.arm(GOLDEN_CONFIRM_END + timedelta(minutes=5), watch_id="rv-second-arm")
    desk.poll(now=datetime(2026, 8, 26, 12, 45))

    assert desk.alert_count == 1  # still just the first fire
    assert desk.armed_ids == ["rv-second-arm"]
    assert [call["watch_id"] for call in desk.phone.calls] == ["rv-first-arm"]

    later, _ = _retest_h1_bars(62)
    desk.set_h1_bars(later)
    desk.poll(now=datetime(2026, 8, 27, 12, 30))

    assert desk.alert_count == 2
    assert desk.armed_ids == []
    assert [call["watch_id"] for call in desk.phone.calls] == [
        "rv-first-arm",
        "rv-second-arm",
    ]
