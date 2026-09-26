"""S6 - M5 structure facets on the alert-time M5 key (shadow only).

`m5_setup_key_stamp.structure_inputs` reads the bot's cached completed M5 bars up
to the alert bar (regular session only, never a later bar) and the facets in
`setup_permutations` bucket them: EMA 8/21 stack, previous day's range, the
first-30-minute range (only after 10:00 ET), the 12-bar squeeze and the side
against SPY's D1 environment. Missing data is unknown, never a default.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import d1_environment_store  # noqa: E402
import m5_setup_key_stamp as stamp  # noqa: E402
import project_paths  # noqa: E402
import setup_permutations as sp  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")
NEW = ("m5_ema_stack", "m5_pdh_pdl", "m5_open_range", "m5_compression", "m5_side_vs_d1_env")
#: The alert bar starts 07:30 PT (10:30 ET) and closes 07:35 PT.
ALERT_CLOSE = datetime(2026, 9, 25, 7, 35)


def _bar(when, close, high, low, volume=1000.0):
    return {"dt": when, "open": close, "high": high, "low": low, "close": close, "volume": volume}


def _previous_day():
    """A full 2026-09-24 session: 78 bars 06:30-12:55 PT, closes 100.00 up to 100.77, range 1.0."""
    start = datetime(2026, 9, 24, 6, 30)
    return [_bar(start + timedelta(minutes=5 * n), 100 + 0.01 * n, 100.5 + 0.01 * n, 99.5 + 0.01 * n)
            for n in range(78)]


def _today():
    """06:30-07:25 PT, then the alert bar at 07:30 PT closing 103.0 above everything."""
    start = datetime(2026, 9, 25, 6, 30)
    bars = []
    for n in range(6):  # the open range: 100.0 to 102.0
        high = 102.0 if n == 1 else 101.5
        low = 100.0 if n == 3 else 100.5
        bars.append(_bar(start + timedelta(minutes=5 * n), 101.0, high, low))
    for n in range(6, 12):  # a tight six bars
        bars.append(_bar(start + timedelta(minutes=5 * n), 101.0, 101.2, 100.8))
    bars.append(_bar(start + timedelta(minutes=60), 103.0, 103.2, 101.0))  # the alert bar
    return bars


def _bars():
    return [*_previous_day(), *_today()]


def _ema(values, length):
    ema = sum(values[:length]) / length
    for value in values[length:]:
        ema = ema + 2.0 / (length + 1) * (value - ema)
    return ema


def _m5(name, side="LONG", **inputs):
    return sp.m5_facets_for(inputs, side).get(name)


# ---------------------------------------------------------------------------
# structure_inputs: the fixture measured by hand
# ---------------------------------------------------------------------------
def test_structure_inputs_at_the_alert_bar():
    got = stamp.structure_inputs(_bars(), ALERT_CLOSE, PT)
    closes = [bar["close"] for bar in _bars()]
    assert got["alert_price"] == 103.0
    assert got["m5_ema8"] == pytest.approx(_ema(closes, 8))
    assert got["m5_ema21"] == pytest.approx(_ema(closes, 21))
    assert (got["prev_day_high"], got["prev_day_low"]) == pytest.approx((101.27, 99.5))
    assert (got["open_range_high"], got["open_range_low"]) == (102.0, 100.0)
    # Box 100-102 = 2.0; ATR20 over the 20 bars before the alert = (8x1.0 + 1+1.5+1+1.5+1+1 + 6x0.4) / 20.
    assert got["m5_range12_atr20"] == pytest.approx(2.0 / (17.4 / 20))
    assert got["m5_range12_break"] == "up"


def test_structure_inputs_never_read_a_bar_after_the_alert_bar():
    later = [*_bars(), _bar(ALERT_CLOSE, 50.0, 200.0, 1.0)]
    assert stamp.structure_inputs(later, ALERT_CLOSE, PT) == stamp.structure_inputs(_bars(), ALERT_CLOSE, PT)


def test_premarket_bars_do_not_count():
    early = [_bar(datetime(2026, 9, 25, 5, 0) + timedelta(minutes=5 * n), 500.0, 501.0, 499.0) for n in range(12)]
    assert stamp.structure_inputs([*early, *_bars()], ALERT_CLOSE, PT) == stamp.structure_inputs(
        _bars(), ALERT_CLOSE, PT)


def test_a_missing_alert_bar_is_all_unknown():
    assert stamp.structure_inputs(_bars()[:-1], ALERT_CLOSE, PT) == dict.fromkeys(stamp.STRUCTURE_INPUT_FIELDS)
    assert stamp.structure_inputs([], ALERT_CLOSE, PT) == dict.fromkeys(stamp.STRUCTURE_INPUT_FIELDS)


def test_a_premarket_alert_bar_is_all_unknown():
    bars = [*_previous_day(), _bar(datetime(2026, 9, 25, 5, 0), 101.0, 101.5, 100.5)]
    assert stamp.structure_inputs(bars, datetime(2026, 9, 25, 5, 5), PT) == dict.fromkeys(
        stamp.STRUCTURE_INPUT_FIELDS)


def test_a_holed_previous_day_has_no_range():
    previous = _previous_day()
    del previous[40]
    got = stamp.structure_inputs([*previous, *_today()], ALERT_CLOSE, PT)
    assert got["prev_day_high"] is None and got["prev_day_low"] is None
    assert got["open_range_high"] == 102.0  # the rest still speaks


def test_no_previous_day_has_no_range_and_too_few_bars_have_no_ema():
    got = stamp.structure_inputs(_today(), ALERT_CLOSE, PT)
    assert got["prev_day_high"] is None and got["prev_day_low"] is None
    assert got["m5_ema8"] is None and got["m5_ema21"] is None  # 13 bars, under 2 x 21
    assert got["m5_range12_atr20"] is None  # no 21 bars for ATR20


def test_the_open_range_waits_for_ten_o_clock():
    bars = [*_previous_day(), *_today()[:6]]  # alert bar = the 06:55 PT (09:55 ET) open-range bar
    got = stamp.structure_inputs(bars, datetime(2026, 9, 25, 7, 0), PT)
    assert got["open_range_high"] is None and got["open_range_low"] is None


def test_a_missing_open_range_bar_has_no_open_range():
    today = _today()
    del today[2]
    got = stamp.structure_inputs([*_previous_day(), *today], ALERT_CLOSE, PT)
    assert got["open_range_high"] is None


def test_the_squeeze_box_never_spans_the_overnight_gap():
    bars = [*_previous_day(), *_today()[:9]]  # alert = 07:10 PT, only 8 bars before it today
    got = stamp.structure_inputs(bars, datetime(2026, 9, 25, 7, 15), PT)
    assert got["m5_range12_atr20"] is None and got["m5_range12_break"] is None


# ---------------------------------------------------------------------------
# the facets: value and unknown, one table each
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("price", "fast", "slow", "expected"), [
    (103.0, 102.0, 101.0, "m5ema_8over21_above_both"),
    (101.5, 102.0, 101.0, "m5ema_8over21_between"),
    (100.0, 102.0, 101.0, "m5ema_8over21_below_both"),
    (103.0, 101.0, 102.0, "m5ema_8under21_above_both"),
    (101.5, 101.0, 102.0, "m5ema_8under21_between"),
    (100.0, 101.0, 102.0, "m5ema_8under21_below_both"),
    (None, 101.0, 102.0, "unknown"), (100.0, None, 102.0, "unknown"), (100.0, 101.0, "", "unknown"),
])
def test_m5_ema_stack(price, fast, slow, expected):
    assert _m5("m5_ema_stack", alert_price=price, m5_ema8=fast, m5_ema21=slow) == expected


@pytest.mark.parametrize(("price", "high", "low", "expected"), [
    (103.0, 102.0, 100.0, "above_pdh"), (101.0, 102.0, 100.0, "inside_pd_range"),
    (102.0, 102.0, 100.0, "inside_pd_range"), (99.0, 102.0, 100.0, "below_pdl"),
    (None, 102.0, 100.0, "unknown"), (101.0, None, 100.0, "unknown"), (101.0, 100.0, 102.0, "unknown"),
])
def test_m5_pdh_pdl(price, high, low, expected):
    assert _m5("m5_pdh_pdl", alert_price=price, prev_day_high=high, prev_day_low=low) == expected


@pytest.mark.parametrize(("close_et", "price", "expected"), [
    (datetime(2026, 9, 25, 10, 35, tzinfo=ET), 103.0, "above_or"),
    (datetime(2026, 9, 25, 10, 5, tzinfo=ET), 101.0, "inside_or"),
    (datetime(2026, 9, 25, 15, 0, tzinfo=ET), 99.0, "below_or"),
    (datetime(2026, 9, 25, 10, 0, tzinfo=ET), 103.0, "unknown"),  # the last open-range bar
    (datetime(2026, 9, 25, 9, 45, tzinfo=ET), 103.0, "unknown"),
    (datetime(2026, 9, 25, 10, 35), 103.0, "unknown"),  # naive: never guessed
    (datetime(2026, 9, 25, 10, 35, tzinfo=ZoneInfo("Asia/Tokyo")), 103.0, "unknown"),
    (None, 103.0, "unknown"),
    (datetime(2026, 9, 25, 10, 35, tzinfo=ET), None, "unknown"),
])
def test_m5_open_range_only_after_ten(close_et, price, expected):
    stamp_text = close_et.isoformat() if close_et is not None else ""
    assert _m5("m5_open_range", alert_bar_close=stamp_text, alert_price=price, open_range_high=102.0,
               open_range_low=100.0) == expected


def test_m5_open_range_without_the_range_is_unknown():
    assert _m5("m5_open_range", alert_bar_close="2026-09-25T10:35:00-04:00", alert_price=103.0) == sp.UNKNOWN


@pytest.mark.parametrize(("ratio", "broke", "expected"), [
    (2.0, "up", "squeeze_break_up"), (2.5, "down", "squeeze_break_down"), (1.0, "inside", "squeeze_inside"),
    (3.5, "up", "no_squeeze"), (3.5, "inside", "no_squeeze"),
    (None, "up", "unknown"), (2.0, None, "unknown"), (2.0, "sideways", "unknown"), (-1.0, "up", "unknown"),
])
def test_m5_compression(ratio, broke, expected):
    assert _m5("m5_compression", m5_range12_atr20=ratio, m5_range12_break=broke) == expected


@pytest.mark.parametrize(("side", "label", "expected"), [
    ("LONG", "trending_up", "side_with_d1_trend"), ("SHORT", "trending_down", "side_with_d1_trend"),
    ("LONG", "trending_down", "side_against_d1_trend"), ("SHORT", "trending_up", "side_against_d1_trend"),
    ("LONG", "compressed", "d1_env_compressed"), ("SHORT", "mixed", "d1_env_mixed"),
    ("LONG", "unknown", "unknown"), ("LONG", None, "unknown"), ("LONG", "junk", "unknown"),
    ("", "trending_up", "unknown"),
])
def test_m5_side_vs_d1_env(side, label, expected):
    assert _m5("m5_side_vs_d1_env", side=side, d1_environment=label) == expected


def test_the_new_facets_stay_out_of_the_label_and_old_records_key_as_before():
    for name in NEW:
        assert name in sp.M5_FACETS and not sp.M5_FACETS[name].in_label
    old = {"alert_bar_close": "2026-09-25T10:35:00-04:00", "session_rvol": 2.4, "bounce_type": "ema_15"}
    key = sp.m5_facets_for(old, "LONG")
    assert all(key.get(name) == sp.UNKNOWN for name in NEW)
    assert not any(name in key.compact_key for name in NEW)
    assert key.permutation_rule_version == "setup_permutations.m5.v1"


# ---------------------------------------------------------------------------
# the stamp: the live record carries the inputs and the facets
# ---------------------------------------------------------------------------
class _Bot:
    def m5_chart_bars(self, symbol, max_sessions=2):
        return _bars()


class _StubLookup:
    def stamp(self, symbol, side, trade_date):
        return {"d1_session": "2026-09-24", "status": stamp.STATUS_NO_SCAN_ROW}


def _registered():
    return {"event_id": "AAA_long_20260925_07_30_00_ema_15", "event_type": "registered",
            "trade_date": "2026-09-25", "symbol": "AAA", "direction": "long", "entry_time": "2026-09-25T07:30:00",
            "logged_at": "2026-09-25T07:35:30-07:00", "context_json": json.dumps({"session_rvol": 2.4})}


@pytest.fixture()
def live(tmp_path, monkeypatch):
    env = tmp_path / "d1_environment.jsonl"
    env.write_text(json.dumps({"session": "2026-09-24", "benchmark": "SPY", "label": "trending_up",
                               "rule_version": d1_environment_store.RULE_VERSION}) + "\n", encoding="utf-8")
    monkeypatch.setattr(project_paths, "D1_ENVIRONMENT_FILE", env)
    monkeypatch.setattr(stamp, "_local_tz", lambda: PT)
    monkeypatch.setattr(stamp, "_spy_log_path", lambda: tmp_path / "no_spy.jsonl")
    bot = _Bot()  # held here: the stamp keeps only a weak reference
    stamp.register_bar_source(bot)
    yield bot
    stamp.register_bar_source(None)
    stamp.reset_for_tests()


def test_the_record_carries_the_structure_facets(live):
    record = stamp.record_for(_registered(), _StubLookup())
    assert {name: record["m5_facets"][name] for name in NEW} == {
        "m5_ema_stack": "m5ema_8over21_above_both", "m5_pdh_pdl": "above_pdh", "m5_open_range": "above_or",
        "m5_compression": "squeeze_break_up", "m5_side_vs_d1_env": "side_with_d1_trend",
    }
    assert record["m5_inputs"]["d1_environment"] == "trending_up"
    assert record["m5_inputs"]["open_range_high"] == 102.0


def test_the_d1_environment_is_the_session_before_the_alert(live):
    assert stamp.d1_environment_before("2026-09-25") == "trending_up"
    assert stamp.d1_environment_before("2026-09-24") == "unknown"  # 09-23 was never labelled
    assert stamp.d1_environment_before("") == "unknown"


def test_a_bar_still_forming_when_logged_is_never_measured(live):
    row = {**_registered(), "logged_at": "2026-09-25T07:33:00-07:00"}
    facets = stamp.record_for(row, _StubLookup())["m5_facets"]
    for name in ("m5_ema_stack", "m5_pdh_pdl", "m5_open_range", "m5_compression"):
        assert facets[name] == sp.UNKNOWN
    assert facets["m5_side_vs_d1_env"] == "side_with_d1_trend"  # known before the open


def test_no_bar_source_leaves_the_bar_facets_unknown(live):
    stamp.register_bar_source(None)
    facets = stamp.record_for(_registered(), _StubLookup())["m5_facets"]
    for name in ("m5_ema_stack", "m5_pdh_pdl", "m5_open_range", "m5_compression"):
        assert facets[name] == sp.UNKNOWN
