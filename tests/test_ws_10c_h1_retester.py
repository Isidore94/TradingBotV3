"""WS-10C - one opt-in H1 retester watch: the 15-EMA bounce, armed from the chart.

WISHLIST 10C step 1 only. These tests are the CONTRACT for the packet; they were
written and committed RED, before any of it existed.

What the packet asks for, and what each part of this file pins:

1. A pure rule sheet `h1_ema_bounce_v1` in ``scripts/indicators/h1_ema_bounce.py``
   (completed H1 bars in, an ``H1Bounce`` or ``None`` out; no clock, no store, no
   engine).  Constants ``EMA_LENGTH = 15``, ``WARMUP_BARS = 45``,
   ``TOUCH_TOLERANCE_ATR = 0.25``, ``REJECTION_CLOSE_ATR = 0.10``,
   ``MAX_TOUCH_AGE_BARS = 3``.
2. A watch kind ``h1_ema_bounce`` on the EXISTING chart-watch store, armed from a
   new "H1 retester" button on the arm bar, polled by ``_poll_d1_event_watches``,
   expiring in TRADING days, firing ONCE and then disarming.
3. One phone event per fire, de-duplicated by watch id, through the ONE armed
   price-alert sender.
4. Nothing new reaches the retired H1 emitter seam in ``bounce_bot_lib``.

=== The module contract these tests assert ===

``EMA_LENGTH`` is a close-based EMA seeded with the FIRST close and stepped with
``alpha = 2 / (EMA_LENGTH + 1)`` - the convention both existing copies in this
repo already use (``bounce_bot_lib.legacy._ema_series``, ``chart_watch._ema_last``).
The golden below recomputes it with an independent loop in this file and ALSO
pins the resulting floats as literals, so swapping to an SMA-seeded EMA fails.

``closed_h1_bars(m5_bars)`` aggregates dict M5 bars to session-aligned H1 dict
bars and drops the still-forming bucket.  It takes NO ``now``: completeness comes
from how far the M5 data itself reaches, which is the shipped rule
(``bounce_bot_lib.legacy._closed_h1_bars``).

``evaluate(h1_bars, side, *, atr, now=None) -> H1Bounce | None`` anchors the
CONFIRM bar at the LAST completed bar and looks back at most
``MAX_TOUCH_AGE_BARS`` bars for the touch (age = confirm index - touch index,
inclusive).  It returns ``None`` - not measured - for fewer than ``WARMUP_BARS``
bars and for a series that is stale relative to ``now``.

``H1Bounce`` carries ``fired``, ``reason``, ``touch_bar_dt``, ``confirm_bar_dt``,
``ema``, ``atr``, ``distance_atr``, ``rule_version`` and ``skipped_bars``.
Reasons asserted here: ``bounce_confirmed`` / ``ambiguous`` / ``no_touch`` /
``invalidated``.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# PCT-1 (2026-09-15) renamed the kind and widened it: the H1 retester is now
# the `h1_ema15_bounce` TRIGGER of the one "Pullback alert" watch. Only these
# three pinned strings move; every assertion below is the one WS-10C shipped,
# and the H1 rule sheet `h1_ema_bounce_v1` is untouched.
WATCH_KIND = "pullback"
BUTTON_LABEL = "Pullback alert"
ARM_REASON_LONG = (
    "waiting for a pullback entry (LONG): H1 15-EMA bounce, "
    "M15/M30 SMA reclaim + LRSI, SMA retest"
)

#: PCT-1 item 6 widened the armed-inventory health cell to ONE STATE PER
#: TIMEFRAME joined with `;`. Every H1 state below is the byte-identical
#: string WS-10C shipped - it is now the FIRST part of the cell, and these
#: fixtures give the two SMA timeframes no history at all, so this is what
#: follows it. Pinned once here, appended where it is asserted, so the H1
#: half of each pin stays exactly as strong as it was.
PULLBACK_TIMEFRAME_TAIL = (
    "; not measured (0 of 160 M15 bars); not measured (0 of 85 M30 bars)"
)

# ---------------------------------------------------------------------------
# The golden fixture: hand-computed here, never by the module under test.
# ---------------------------------------------------------------------------
#: The desk's regular session on the local (Pacific) clock, from
#: ``market_session.get_market_session_open_naive``: 06:30 -> 13:00, so the H1
#: buckets are 06:30, 07:30 ... 12:30 and the last one is a 30-minute hour.
SESSION_OPEN_HOUR = 6
SESSION_OPEN_MINUTE = 30
BUCKETS_PER_SESSION = 7

#: Ten consecutive exchange sessions (verified against ``market_calendar`` -
#: 2026-08-22/23 is a weekend and none of these days is a holiday). The golden
#: uses the first 55 buckets; the tail days only exist so the age-window tests
#: can hang extra bars off the end.
GOLDEN_SESSION_DAYS = (17, 18, 19, 20, 21, 24, 25, 26, 27, 28)

ALPHA = 2.0 / 16.0  # EMA_LENGTH 15


def _ema_series(closes: list[float]) -> list[float]:
    """EMA-15 the long way, in this file, so the golden is independent."""
    value = closes[0]
    out = [value]
    for close in closes[1:]:
        value = ALPHA * close + (1.0 - ALPHA) * value
        out.append(value)
    return out


def _close_for_ema_offset(previous_ema: float, offset: float) -> float:
    """The close that puts this bar exactly ``offset`` away from its own EMA.

    e = a*c + (1-a)*prev, and we want c - e = offset, so
    (1-a)*(c - prev) = offset  ->  c = prev + offset / (1 - a).
    """
    return previous_ema + offset / (1.0 - ALPHA)


def _bucket_dt(index: int) -> datetime:
    day = GOLDEN_SESSION_DAYS[index // BUCKETS_PER_SESSION]
    slot = index % BUCKETS_PER_SESSION
    return datetime(2026, 8, day, SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE) + timedelta(
        hours=slot
    )


def pin_armed_before_the_golden_bounce(panel, *, kind=None):
    """FIXTURE correction, RV-H1-ARM-TIME 2026-09-13 - not a weakened assertion.

    `arm_chart_watch_for` stamps `armed_at = datetime.now()`, i.e. today, which
    is AFTER the golden bars of 2026-08-26. Under the pre-arm fence (a watch
    never fires on a bounce that finished before it was armed) that fixture
    arms the watch an hour AFTER the reclaim it then expects to fire on - it
    encoded the very defect the fence repairs. Pinning `armed_at` an hour
    before the touch bar restores what these tests were always about: the
    trader armed, and THEN the bounce printed. Every assertion is untouched.
    """
    kind = kind or WATCH_KIND
    armed_at = GOLDEN_TOUCH_DT - timedelta(hours=1)
    panel._chart_watches = [
        dataclasses.replace(watch, armed_at=armed_at)
        if watch.kind == kind
        else watch
        for watch in panel._chart_watches
    ]
    return next(
        (watch for watch in panel._chart_watches if watch.kind == kind), None
    )


def _ramp_closes(count: int) -> list[float]:
    """A steadily rising close series: the EMA slope is unambiguously > 0."""
    return [100.0 + 0.25 * index for index in range(count)]


def _ramp_bar(index: int, close: float) -> dict:
    return {
        "dt": _bucket_dt(index),
        "open": close - 0.05,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
    }


def golden_long_h1_bars() -> tuple[list[dict], list[float]]:
    """55 completed H1 bars that hold ONE 15-EMA retest, and their EMA series.

    Bars 0-52 ramp: every low sits ~1.65 above its own EMA, so nothing there is
    a touch.  Bar 53 (2026-08-26 10:30) dips its LOW to 0.02 under the EMA and
    closes 0.01 under it - a touch with no reclaim and no invalidation.  Bar 54
    (2026-08-26 11:30) closes 1.9685 above its EMA - the reclaim.
    """
    closes = _ramp_closes(53)
    ema52 = _ema_series(closes)[-1]
    closes.append(_close_for_ema_offset(ema52, -0.01))
    ema53 = _ema_series(closes)[-1]
    closes.append(113.5)
    ema = _ema_series(closes)

    bars = []
    for index, close in enumerate(closes):
        if index == 53:
            bars.append(
                {
                    "dt": _bucket_dt(index),
                    "open": close + 0.05,
                    "high": close + 0.20,
                    "low": ema53 - 0.02,
                    "close": close,
                }
            )
        elif index == 54:
            bars.append(
                {
                    "dt": _bucket_dt(index),
                    "open": 112.5,
                    "high": close + 0.20,
                    "low": 112.0,
                    "close": close,
                }
            )
        else:
            bars.append(_ramp_bar(index, close))
    return bars, ema


#: Pinned by hand from the loop above (see the module docstring).  A different
#: EMA seeding convention moves these in the sixth decimal and fails.
GOLDEN_EMA_TOUCH_BAR = 111.250259756504
GOLDEN_EMA_CONFIRM_BAR = 111.531477286941
GOLDEN_TOUCH_DT = datetime(2026, 8, 26, 10, 30)
GOLDEN_CONFIRM_DT = datetime(2026, 8, 26, 11, 30)
#: ATR handed to the rule in the pure tests. The real Wilder ATR14 of this
#: series is 0.5949, and every threshold below holds at BOTH values.
GOLDEN_ATR = 1.0


def _mirror(bars: list[dict], axis: float = 200.0) -> list[dict]:
    """The same series reflected: a LONG retest becomes the SHORT one."""
    return [
        {
            "dt": bar["dt"],
            "open": axis - bar["open"],
            "high": axis - bar["low"],
            "low": axis - bar["high"],
            "close": axis - bar["close"],
        }
        for bar in bars
    ]


def _explode_to_m5(bar: dict, count: int) -> list[dict]:
    """M5 bars that aggregate back to exactly this H1 bar.

    Bar 0 carries the hour's open/high/low, the last bar carries its close.
    Verified against the shipped ``bounce_bot_lib.legacy._closed_h1_bars``.
    """
    mid = (bar["high"] + bar["low"]) / 2.0
    out = []
    for step in range(count):
        stamp = bar["dt"] + timedelta(minutes=5 * step)
        if step == 0:
            out.append(
                {
                    "dt": stamp,
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": mid,
                }
            )
        elif step == count - 1:
            out.append(
                {
                    "dt": stamp,
                    "open": mid,
                    "high": max(mid, bar["close"]),
                    "low": min(mid, bar["close"]),
                    "close": bar["close"],
                }
            )
        else:
            out.append({"dt": stamp, "open": mid, "high": mid, "low": mid, "close": mid})
    return out


def golden_m5_series(h1_bars: list[dict]) -> list[dict]:
    """The recorded 5-minute series behind the golden: 618 bars, 10 sessions."""
    m5: list[dict] = []
    for bar in h1_bars:
        # The 12:30 bucket is the session's short 30-minute hour: six M5 bars.
        m5.extend(_explode_to_m5(bar, 6 if bar["dt"].hour == 12 else 12))
    return m5


# ---------------------------------------------------------------------------
# 1. The pure rule
# ---------------------------------------------------------------------------
def test_the_rule_sheet_freezes_its_constants_and_its_version():
    from indicators import h1_ema_bounce as rule

    assert rule.EMA_LENGTH == 15
    assert rule.WARMUP_BARS == 45
    assert rule.TOUCH_TOLERANCE_ATR == 0.25
    assert rule.REJECTION_CLOSE_ATR == 0.10
    assert rule.MAX_TOUCH_AGE_BARS == 3
    assert rule.RULE_VERSION == "h1_ema_bounce_v1"


def test_a_hand_computed_retest_fires_on_the_exact_ema_and_bar_times():
    from indicators import h1_ema_bounce as rule

    bars, ema = golden_long_h1_bars()
    # The fixture agrees with the literals pinned above, so a reader can see
    # that the numbers below were computed, not copied out of the module.
    assert ema[53] == pytest.approx(GOLDEN_EMA_TOUCH_BAR, abs=1e-9)
    assert ema[54] == pytest.approx(GOLDEN_EMA_CONFIRM_BAR, abs=1e-9)

    result = rule.evaluate(bars, "long", atr=GOLDEN_ATR)

    assert result is not None
    assert result.fired is True
    assert result.reason == "bounce_confirmed"
    assert result.touch_bar_dt == GOLDEN_TOUCH_DT
    assert result.confirm_bar_dt == GOLDEN_CONFIRM_DT
    assert result.ema == pytest.approx(GOLDEN_EMA_CONFIRM_BAR, abs=1e-9)
    assert result.atr == pytest.approx(GOLDEN_ATR, abs=1e-9)
    # The touch sat 0.02 under an EMA of 111.2503 with ATR 1.0.
    assert result.distance_atr == pytest.approx(0.02, abs=1e-9)
    assert result.rule_version == "h1_ema_bounce_v1"
    assert result.skipped_bars == 0


def test_the_short_side_is_the_mirror_of_the_long_side():
    from indicators import h1_ema_bounce as rule

    bars, ema = golden_long_h1_bars()
    mirrored = _mirror(bars)

    long_hit = rule.evaluate(bars, "long", atr=GOLDEN_ATR)
    short_hit = rule.evaluate(mirrored, "short", atr=GOLDEN_ATR)

    assert short_hit is not None and short_hit.fired is True
    assert short_hit.touch_bar_dt == long_hit.touch_bar_dt
    assert short_hit.confirm_bar_dt == long_hit.confirm_bar_dt
    assert short_hit.ema == pytest.approx(200.0 - long_hit.ema, abs=1e-9)
    assert short_hit.distance_atr == pytest.approx(long_hit.distance_atr, abs=1e-9)
    # The same series read the WRONG way round is not a signal.
    assert rule.evaluate(bars, "short", atr=GOLDEN_ATR).fired is False
    assert rule.evaluate(mirrored, "long", atr=GOLDEN_ATR).fired is False


def test_the_side_is_read_case_insensitively():
    """The chart-watch store spells a side ``LONG``; the rule sheet spells it
    ``long``. One of those would silently never fire."""
    from indicators import h1_ema_bounce as rule

    bars, _ = golden_long_h1_bars()
    assert rule.evaluate(bars, "LONG", atr=GOLDEN_ATR).fired is True


def test_a_touch_and_a_reclaim_on_the_same_bar_is_ambiguous_and_never_fires():
    from indicators import h1_ema_bounce as rule

    closes = _ramp_closes(54)
    ema53 = _ema_series(closes)[-1]
    # One bar that both tags the EMA and closes half a point above it.
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

    result = rule.evaluate(bars, "long", atr=GOLDEN_ATR)

    assert result is not None
    assert result.fired is False
    assert result.reason == "ambiguous"
    assert result.touch_bar_dt == _bucket_dt(54)


def test_a_touch_older_than_the_age_window_is_not_confirmed():
    from indicators import h1_ema_bounce as rule

    bars, ema = golden_long_h1_bars()
    # Four ramp bars of daylight between the touch and the last bar: the touch
    # bar keeps its geometry, it has just aged past MAX_TOUCH_AGE_BARS.
    aged = bars[:54] + [
        _ramp_bar(index, 113.5 + 0.25 * step)
        for step, index in enumerate(range(54, 58))
    ]
    assert len(aged) - 1 - 53 == 4 > rule.MAX_TOUCH_AGE_BARS

    result = rule.evaluate(aged, "long", atr=GOLDEN_ATR)

    assert result is not None
    assert result.fired is False
    assert result.reason == "no_touch"


def test_a_touch_at_the_edge_of_the_age_window_still_confirms():
    from indicators import h1_ema_bounce as rule

    bars, _ = golden_long_h1_bars()
    aged = bars[:55] + [
        _ramp_bar(index, 113.5 + 0.25 * step)
        for step, index in enumerate(range(55, 57), start=1)
    ]
    assert len(aged) - 1 - 53 == rule.MAX_TOUCH_AGE_BARS

    result = rule.evaluate(aged, "long", atr=GOLDEN_ATR)

    assert result is not None and result.fired is True
    assert result.touch_bar_dt == GOLDEN_TOUCH_DT
    assert result.confirm_bar_dt == _bucket_dt(56)


def test_a_close_a_full_atr_the_wrong_side_of_the_ema_invalidates_the_retest():
    from indicators import h1_ema_bounce as rule

    bars, ema = golden_long_h1_bars()
    broken = list(bars[:54])
    close = _close_for_ema_offset(ema[53], -1.5)  # 1.5 ATR below its own EMA
    broken.append(
        {
            "dt": _bucket_dt(54),
            "open": close + 0.30,
            "high": close + 0.40,
            "low": close - 0.10,
            "close": close,
        }
    )

    result = rule.evaluate(broken, "long", atr=GOLDEN_ATR)

    assert result is not None
    assert result.fired is False
    assert result.reason == "invalidated"


def test_fewer_than_the_warm_up_bars_is_not_measured_not_a_no():
    from indicators import h1_ema_bounce as rule

    bars, _ = golden_long_h1_bars()
    short_series = bars[-(rule.WARMUP_BARS - 1) :]
    assert len(short_series) == rule.WARMUP_BARS - 1

    assert rule.evaluate(short_series, "long", atr=GOLDEN_ATR) is None
    # One more bar and it is measurable again.
    assert rule.evaluate(bars[-rule.WARMUP_BARS :], "long", atr=GOLDEN_ATR) is not None


def test_stale_bars_are_not_measured():
    """Cached bars from a session days ago answer nothing about right now."""
    from indicators import h1_ema_bounce as rule

    bars, _ = golden_long_h1_bars()
    fresh = GOLDEN_CONFIRM_DT + timedelta(hours=1)

    assert rule.evaluate(bars, "long", atr=GOLDEN_ATR, now=fresh) is not None
    assert rule.evaluate(bars, "long", atr=GOLDEN_ATR, now=fresh + timedelta(days=3)) is None


def test_an_invalid_candle_is_skipped_and_counted_never_priced():
    """low <= open, close <= high (plan.md sec 5). A bar that breaks it is not
    a price - averaging it into the EMA would move the level it is measuring."""
    from indicators import h1_ema_bounce as rule

    bars, _ = golden_long_h1_bars()
    poisoned = list(bars)
    broken = {
        "dt": _bucket_dt(20) + timedelta(minutes=1),
        "open": 105.0,
        "high": 104.0,  # high below the open: not a candle
        "low": 999.0,
        "close": 105.0,
    }
    poisoned.insert(21, broken)

    clean = rule.evaluate(bars, "long", atr=GOLDEN_ATR)
    result = rule.evaluate(poisoned, "long", atr=GOLDEN_ATR)

    assert result is not None
    assert result.skipped_bars == 1
    # Skipped means skipped: the surviving series is the clean one, to the bit.
    assert result.ema == pytest.approx(clean.ema, abs=1e-12)
    assert result.fired is True
    assert result.touch_bar_dt == GOLDEN_TOUCH_DT
    assert result.confirm_bar_dt == GOLDEN_CONFIRM_DT


# ---------------------------------------------------------------------------
# 2. The H1 aggregation from the recorded 5-minute series
# ---------------------------------------------------------------------------
def test_the_recorded_five_minute_series_aggregates_to_the_golden_h1_bars():
    from indicators import h1_ema_bounce as rule

    h1_bars, _ = golden_long_h1_bars()
    m5 = golden_m5_series(h1_bars)
    assert len(m5) == 618

    built = rule.closed_h1_bars(m5)

    assert len(built) == len(h1_bars)
    for got, want in zip(built, h1_bars):
        assert got["dt"] == want["dt"]
        for field in ("open", "high", "low", "close"):
            assert got[field] == pytest.approx(want[field], abs=1e-9), (
                f"{field} at {want['dt']}"
            )
    # Session-aligned, not clock-aligned: the first bucket of every session is
    # the 06:30 open and the last is the short 12:30 hour.
    assert built[0]["dt"] == datetime(2026, 8, 17, 6, 30)
    assert {bar["dt"].minute for bar in built} == {30}


def test_a_forming_h1_bar_is_a_preview_and_never_reaches_the_rule():
    from indicators import h1_ema_bounce as rule

    h1_bars, _ = golden_long_h1_bars()
    m5 = golden_m5_series(h1_bars)
    # Cut the tape mid-hour: the 11:30 bucket has printed 12:00 and 12:05 only.
    partial = [bar for bar in m5 if bar["dt"] <= datetime(2026, 8, 26, 12, 5)]

    built = rule.closed_h1_bars(partial)

    assert built[-1]["dt"] == GOLDEN_TOUCH_DT  # the 10:30 hour, not 11:30
    assert len(built) == len(h1_bars) - 1
    # And with the forming bar gone there is nothing to confirm the touch with.
    assert rule.evaluate(built, "long", atr=GOLDEN_ATR).fired is False


def test_a_missing_session_leaves_a_gap_and_invents_no_bar():
    from indicators import h1_ema_bounce as rule

    h1_bars, _ = golden_long_h1_bars()
    m5 = golden_m5_series(h1_bars)
    holiday = datetime(2026, 8, 19).date()
    with_gap = [bar for bar in m5 if bar["dt"].date() != holiday]

    built = rule.closed_h1_bars(with_gap)

    assert len(built) == len(h1_bars) - BUCKETS_PER_SESSION
    assert not any(bar["dt"].date() == holiday for bar in built)


def test_the_pure_rule_does_not_drag_the_bounce_engine_in():
    """``bounce_bot_lib.legacy`` pulls ibapi and ~1,050 modules (measured
    2026-09-13, 2.55 s).  A pure indicator that imports it is not pure."""
    source = (SCRIPTS_DIR / "indicators" / "h1_ema_bounce.py").read_text(encoding="utf-8")
    assert "bounce_bot_lib" not in source

    env = dict(os.environ)
    env["TRADINGBOTV3_DATA_DIR"] = str(Path(os.environ.get("TEMP", ".")) / "ws10c_probe")
    probe = (
        "import sys;"
        f"sys.path.insert(0, r'{SCRIPTS_DIR}');"
        "from indicators import h1_ema_bounce;"
        "print(int(any(name.startswith('ibapi') or name.startswith('bounce_bot_lib')"
        " for name in sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("0"), out.stdout


# ---------------------------------------------------------------------------
# 3. The desk: the arm bar, the chart-watch store, the poll
# ---------------------------------------------------------------------------
def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless without Qt
        pytest.skip("PySide6 not installed")
    return QApplication.instance() or QApplication([])


def _panel(monkeypatch, tmp_path):
    _qt_app()
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    panel._chart_watches_path = tmp_path / "chart_watches.json"
    return panel


def test_the_arm_bar_offers_an_h1_retester_button_that_arms_the_kind():
    _qt_app()
    from chart_watch import WATCH_KINDS
    from ui.widgets.arm_bar import ArmBar

    assert WATCH_KINDS[WATCH_KIND] == BUTTON_LABEL

    bar = ArmBar()
    # Lead fix 2026-09-13: a watch toggle is disabled until a symbol is charted.
    bar.set_enabled_for_symbol(True)
    button = bar.watch_buttons[WATCH_KIND]
    assert BUTTON_LABEL in button.text()

    emitted: list[str] = []
    bar.watchToggled.connect(emitted.append)
    button.click()
    assert emitted == [WATCH_KIND]


def test_arming_writes_a_watch_row_with_the_visible_reason_and_an_id(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])

    assert panel.arm_chart_watch_for("aapl", "LONG", WATCH_KIND) is True

    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)
    assert watch.symbol == "AAPL"
    assert watch.side == "LONG"
    assert getattr(watch, "reason", "") == ARM_REASON_LONG
    assert str(getattr(watch, "watch_id", "") or "").strip() != ""

    from chart_watch import chart_watch_to_dict

    row = chart_watch_to_dict(watch)
    assert row["reason"] == ARM_REASON_LONG
    assert row["watch_id"] == watch.watch_id
    assert row["kind"] == WATCH_KIND


def test_disarming_and_re_arming_is_a_new_watch_id(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])

    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    first = next(w for w in panel._chart_watches if w.kind == WATCH_KIND).watch_id

    assert panel.disarm_chart_watch_for("AAPL", WATCH_KIND) is True
    assert not [w for w in panel._chart_watches if w.kind == WATCH_KIND]

    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    second = next(w for w in panel._chart_watches if w.kind == WATCH_KIND).watch_id

    assert second != first


def test_a_restart_mid_watch_keeps_the_h1_watch(tmp_path):
    """The chart-watch store drops everything on a new market date
    (``chart_watch.load_chart_watches``: "armed watches never survive into a new
    session").  An H1 retester is armed for TEN TRADING DAYS - a desk restart,
    or simply tomorrow, must not silently retire it, while the session-scoped
    kinds beside it still go."""
    from chart_watch import (
        ChartWatch,
        load_chart_watches,
        save_chart_watches,
    )

    armed_day = datetime(2026, 8, 17, 7, 15)
    path = tmp_path / "chart_watches.json"
    h1_watch = ChartWatch(
        symbol="AAPL",
        kind=WATCH_KIND,
        armed_at=armed_day,
        side="LONG",
        source_text="",
        reason=ARM_REASON_LONG,
        watch_id="ws10c-restart",
    )
    session_watch = ChartWatch(
        symbol="AAPL", kind="new_hod", armed_at=armed_day, side="LONG", baseline=101.0
    )
    save_chart_watches([h1_watch, session_watch], path, market_date=armed_day.date())

    reloaded = load_chart_watches(path, market_date=date(2026, 8, 18))

    kinds = {watch.kind for watch in reloaded}
    assert WATCH_KIND in kinds
    assert "new_hod" not in kinds
    survivor = next(watch for watch in reloaded if watch.kind == WATCH_KIND)
    assert survivor.watch_id == "ws10c-restart"
    assert survivor.reason == ARM_REASON_LONG
    assert survivor.armed_at == armed_day


def test_the_existing_poll_fires_the_h1_retest_once_and_then_disarms(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    h1_bars, _ = golden_long_h1_bars()
    m5 = golden_m5_series(h1_bars)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: list(m5))
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])

    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    pin_armed_before_the_golden_bounce(panel)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)
    before = len(panel._alerts)
    moment = GOLDEN_CONFIRM_DT + timedelta(hours=1)

    panel._poll_d1_event_watches(now=moment)

    assert len(panel._alerts) == before + 1
    assert not [w for w in panel._chart_watches if w.kind == WATCH_KIND]

    fired = panel._alerts[0]
    assert fired.symbol == "AAPL"
    payload = dict(fired.payload or {})
    assert payload.get("chart_watch_kind") == WATCH_KIND
    assert str(payload.get("watch_id") or "") == watch.watch_id
    assert str(payload.get("touch_bar_dt") or "") == GOLDEN_TOUCH_DT.isoformat()
    assert str(payload.get("confirm_bar_dt") or "") == GOLDEN_CONFIRM_DT.isoformat()

    # One fire per watch: a second poll on the same bars adds nothing.
    panel._poll_d1_event_watches(now=moment)
    assert len(panel._alerts) == before + 1


def test_the_fired_watch_lands_on_the_d1_feed_as_an_armed_event(monkeypatch, tmp_path):
    """Never a detector alert: it carries the chart-watch tag and the D1
    timeframe, exactly like every other armed event."""
    from ui.models.bounce import CHART_WATCH_TAG, is_chart_watch_alert

    panel = _panel(monkeypatch, tmp_path)
    h1_bars, _ = golden_long_h1_bars()
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda symbol, **kw: golden_m5_series(h1_bars)
    )
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    pin_armed_before_the_golden_bounce(panel)

    panel._poll_d1_event_watches(now=GOLDEN_CONFIRM_DT + timedelta(hours=1))

    fired = panel._alerts[0]
    assert fired.tag == CHART_WATCH_TAG
    assert is_chart_watch_alert(fired) is True
    assert fired.timeframe == "D1"


def test_the_watch_expires_in_trading_days_not_calendar_days(monkeypatch, tmp_path):
    import armed_alert_expiry
    from market_calendar import trading_days_between

    armed_on = date(2026, 8, 17)
    still_open = date(2026, 8, 28)
    due = date(2026, 8, 31)
    # The premise, read from the calendar rather than counted off a wall chart:
    # eleven calendar days is nine sessions, fourteen is ten.
    assert trading_days_between(armed_on, still_open) < armed_alert_expiry.DEFAULT_EXPIRY_TRADING_DAYS
    assert trading_days_between(armed_on, due) == armed_alert_expiry.DEFAULT_EXPIRY_TRADING_DAYS
    assert (still_open - armed_on).days == 11 and (due - armed_on).days == 14

    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)
    panel._chart_watches = [
        dataclasses.replace(watch, armed_at=datetime(2026, 8, 17, 7, 15))
    ]

    panel._poll_d1_event_watches(now=datetime.combine(still_open, datetime.min.time()))
    assert [w.kind for w in panel._chart_watches] == [WATCH_KIND]

    panel._poll_d1_event_watches(now=datetime.combine(due, datetime.min.time()))
    assert not [w for w in panel._chart_watches if w.kind == WATCH_KIND]


# ---------------------------------------------------------------------------
# 4. Delivery: one phone event per fire, through the ONE armed sender
# ---------------------------------------------------------------------------
def test_the_armed_sender_pushes_once_per_watch_id_in_every_mode(monkeypatch):
    """The armed Research/Focus price-alert sender is the one door to the phone
    (CLAUDE.md: AWAY is the only mode that pushes routine output, and the armed
    price alerts are an exception that pushes in EVERY mode).  A watch id that
    has already been announced is never announced twice."""
    _qt_app()
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    sent: list[tuple] = []
    monkeypatch.setattr(
        push_notify, "send_push", lambda *a, **kw: sent.append((a, kw)) or {"ok": True}
    )
    import autopilot_core

    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda *a, **kw: "DESK")

    service = PriceAlertService()
    try:
        first = service.notify_armed_watch(
            watch_id="ws10c-1", title="H1 retester", message="AAPL LONG retest confirmed"
        )
        again = service.notify_armed_watch(
            watch_id="ws10c-1", title="H1 retester", message="AAPL LONG retest confirmed"
        )
        other = service.notify_armed_watch(
            watch_id="ws10c-2", title="H1 retester", message="MSFT SHORT retest confirmed"
        )
    finally:
        service.shutdown()

    assert bool(first.get("ok")) is True
    assert again.get("deduplicated") is True
    assert bool(other.get("ok")) is True
    assert len(sent) == 2


def test_one_phone_event_per_fire_routed_through_the_price_alert_service(
    monkeypatch, tmp_path
):
    panel = _panel(monkeypatch, tmp_path)
    h1_bars, _ = golden_long_h1_bars()
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda symbol, **kw: golden_m5_series(h1_bars)
    )
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])

    calls: list[dict] = []

    class _Recorder:
        def notify_armed_watch(self, *, watch_id, title, message):
            calls.append({"watch_id": watch_id, "title": title, "message": message})
            return {"ok": True}

    panel.price_alert_service = _Recorder()
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    pin_armed_before_the_golden_bounce(panel)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)

    moment = GOLDEN_CONFIRM_DT + timedelta(hours=1)
    panel._poll_d1_event_watches(now=moment)
    panel._poll_d1_event_watches(now=moment)

    assert len(calls) == 1
    assert calls[0]["watch_id"] == watch.watch_id
    assert "AAPL" in calls[0]["message"]


# ---------------------------------------------------------------------------
# 5. The retired H1 emitter stays retired
# ---------------------------------------------------------------------------
def test_nothing_new_reaches_the_retired_h1_emitter_seam():
    """``H1_ALERTS_RETIRED`` gates the EMIT seam only, and this packet adds a
    watch, not an emitter.  Its readers are frozen at the set recorded on
    2026-09-13; a new one means the retired path was reopened."""
    legacy = SCRIPTS_DIR / "bounce_bot_lib" / "legacy.py"
    text = legacy.read_text(encoding="utf-8")
    assert "H1_ALERTS_RETIRED = True" in text
    assert text.count("H1_ALERTS_RETIRED") == 4  # definition, docstring, two gates

    offenders = []
    for path in sorted(ROOT_DIR.rglob("*.py")):
        parts = set(path.relative_to(ROOT_DIR).parts)
        if (
            ".git" in parts
            or ".codex" in parts  # sibling worktrees checked out under the repo
            or ".venv" in parts
            or ".test_tmp" in parts
            or "build" in parts
            or "dist" in parts
        ):
            continue
        if path == legacy or path.name in (
            "test_bounce_learning.py",
            Path(__file__).name,
        ):
            continue
        if "H1_ALERTS_RETIRED" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(ROOT_DIR)))
    assert offenders == []


def test_the_bounce_engine_is_not_edited_by_this_packet():
    base = "origin/claude/wishlist-sweep-2026-09-12"
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", base],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:  # pragma: no cover - the base ref is fetched here
        pytest.skip(f"{base} is not fetched in this checkout")
    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "--", "scripts/bounce_bot_lib"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    assert changed.stdout.strip() == "", changed.stdout
