"""Plan to 8 / P11 - more facets: M5-native facets on the alert, and D1 facets on the scan row.

Shadow only. Missing data is unknown, never a default. The M5 facets read only
what the alert carried then (no bar after the alert bar); the D1 columns are
appended to the scan row and change no detector or score output.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import m5_setup_key_stamp as stamp  # noqa: E402
import market_calendar  # noqa: E402
import project_paths  # noqa: E402
import setup_permutation_backfill as bf  # noqa: E402
import setup_permutation_search as search  # noqa: E402
import setup_permutations as sp  # noqa: E402

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
NAN = float("nan")


def _d1(name, **row):
    base = {"side": "LONG", "setup_family": "avwap_band_bounce"}
    base.update(row)
    return sp.facets_for_row(base).get(name)


def _m5(name, side="LONG", **inputs):
    return sp.m5_facets_for(inputs, side).get(name)


# ---------------------------------------------------------------------------
# D1 facets on the scan row: value and unknown
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("value", "expected"), [
    (0.05, "atr_pctile_0_20"), (0.3, "atr_pctile_20_50"), (0.6, "atr_pctile_50_80"), (1.0, "atr_pctile_80_100"),
    ("", "unknown"), (NAN, "unknown"), (1.5, "unknown"), (-0.1, "unknown"),
])
def test_atr_percentile_facet(value, expected):
    assert _d1("atr_percentile", perm_atr14_pctile_252=value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    (0.5, "off_52w_low_0_2atr"), (3.0, "off_52w_low_2_5atr"), (7.0, "off_52w_low_5_10atr"),
    (25.0, "off_52w_low_10atr_plus"), (None, "unknown"), (-1.0, "unknown"),
])
def test_52_week_low_distance_facet(value, expected):
    assert _d1("low_52w_distance", perm_low_52w_dist_atr=value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    (5, "closes_right_5of5"), (4.0, "closes_right_3_4of5"), (3, "closes_right_3_4of5"), (0, "closes_right_0_2of5"),
    ("", "unknown"), (6, "unknown"), (-1, "unknown"),
])
def test_closes_vs_level_facet(value, expected):
    assert _d1("closes_vs_level", perm_closes_right_of_level_5=value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    (0, "level_respect_0"), (1, "level_respect_1"), (3, "level_respect_2_3"), (9, "level_respect_4_plus"),
    ("", "unknown"), (-2, "unknown"),
])
def test_level_respect_facet(value, expected):
    assert _d1("level_respect", perm_level_respect_20=value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    ("LONG_z1", "zone_arm_long_z1"), ("SHORT_z3", "zone_arm_short_z3"), ("not_armed", "no_zone_arm"),
    ("", "unknown"), ("sideways_z9", "unknown"),
])
def test_d1_zone_arm_facet(value, expected):
    assert _d1("d1_zone_arm", perm_d1_zone_arm=value) == expected


def test_the_new_d1_facets_are_registered_as_d1_facets_not_m5():
    for name in ("atr_percentile", "low_52w_distance", "closes_vs_level", "level_respect", "d1_zone_arm"):
        assert name in sp.FACETS and name not in sp.M5_FACETS
    # A row without the new columns keys exactly as before: absent means unknown.
    key = sp.facets_for_row({"side": "LONG", "setup_family": "x"})
    for name in ("atr_percentile", "low_52w_distance", "closes_vs_level", "level_respect", "d1_zone_arm"):
        assert key.get(name) == sp.UNKNOWN


# ---------------------------------------------------------------------------
# the D1 column builder the scan calls (pure, point in time)
# ---------------------------------------------------------------------------
def _ohlc(count, *, tr=2.0, last_tr=None, last_n=0, low_dip_at=None):
    rows = []
    day = date(2025, 1, 2)
    for index in range(count):
        close = 100.0 + index * 0.01
        width = (last_tr if last_tr is not None and index >= count - last_n else tr) / 2.0
        low = close - width
        if low_dip_at is not None and index == low_dip_at:
            low = 50.0
        rows.append({"date": (day + timedelta(days=index)).isoformat(), "open": close,
                     "high": close + width, "low": low, "close": close, "volume": 1000.0})
    return rows


def test_atr_percentile_needs_252_sessions_of_atr14():
    short = sp.d1_history_columns(_ohlc(260), side="LONG", level=None, atr=2.0)
    assert short["perm_atr14_pctile_252"] is None  # 247 ATR14 values, not 252
    assert short["perm_low_52w_dist_atr"] is not None  # 260 sessions hold a year of lows
    assert sp.d1_history_columns(_ohlc(200), side="LONG", level=None, atr=2.0)["perm_low_52w_dist_atr"] is None
    flat = sp.d1_history_columns(_ohlc(300), side="LONG", level=None, atr=2.0)
    assert flat["perm_atr14_pctile_252"] is None  # no range to place it in
    high = sp.d1_history_columns(_ohlc(300, last_tr=4.0, last_n=20), side="LONG", level=None, atr=2.0)
    assert high["perm_atr14_pctile_252"] == pytest.approx(1.0)


def test_52_week_low_distance_in_atr():
    rows = _ohlc(300, low_dip_at=200)
    got = sp.d1_history_columns(rows, side="LONG", level=None, atr=2.0)
    assert got["perm_low_52w_dist_atr"] == pytest.approx((rows[-1]["close"] - 50.0) / 2.0)
    # A dip older than 252 sessions is outside the year.
    old_rows = _ohlc(300, low_dip_at=10)
    old = sp.d1_history_columns(old_rows, side="LONG", level=None, atr=2.0)
    assert old["perm_low_52w_dist_atr"] == pytest.approx((old_rows[-1]["close"] - old_rows[48]["low"]) / 2.0)
    assert sp.d1_history_columns(rows, side="LONG", level=None, atr=None)["perm_low_52w_dist_atr"] is None


def _bars(closes, lows=None, highs=None):
    day = date(2026, 3, 2)
    rows = []
    for index, close in enumerate(closes):
        rows.append({"date": (day + timedelta(days=index)).isoformat(), "open": close,
                     "high": (highs or {}).get(index, close + 0.5), "low": (lows or {}).get(index, close - 0.5),
                     "close": close, "volume": 1.0})
    return rows


def test_closes_vs_level_counts_the_last_five_closes_on_the_side_of_the_level():
    rows = _bars([100.0] * 10 + [101.0, 99.0, 102.0, 103.0, 98.0])
    assert sp.d1_history_columns(rows, side="LONG", level=100.0, atr=2.0)["perm_closes_right_of_level_5"] == 3
    assert sp.d1_history_columns(rows, side="SHORT", level=100.0, atr=2.0)["perm_closes_right_of_level_5"] == 2
    assert sp.d1_history_columns(rows, side="LONG", level=None, atr=2.0)["perm_closes_right_of_level_5"] is None
    assert sp.d1_history_columns(rows[:4], side="LONG", level=100.0, atr=2.0)["perm_closes_right_of_level_5"] is None
    assert sp.d1_history_columns(rows, side="", level=100.0, atr=2.0)["perm_closes_right_of_level_5"] is None


def test_level_respect_counts_held_touches_in_the_last_20_sessions():
    closes = [105.0] * 25
    lows = {22: 100.1, 23: 99.0, 24: 104.0}  # 22 and 23 touch; 23 closes above so both held
    lows[2] = 99.0  # outside the 20-session window
    rows = _bars(closes, lows=lows)
    got = sp.d1_history_columns(rows, side="LONG", level=100.0, atr=2.0)
    assert got["perm_level_respect_20"] == 2
    # A touch that closed through the level did not hold.
    rows[-1]["low"], rows[-1]["close"] = 99.5, 99.0
    assert sp.d1_history_columns(rows, side="LONG", level=100.0, atr=2.0)["perm_level_respect_20"] == 2
    # Mirrored for shorts: highs touch from below and the close stays below.
    short_rows = _bars([95.0] * 25, highs={23: 100.05})
    assert sp.d1_history_columns(short_rows, side="SHORT", level=100.0, atr=2.0)["perm_level_respect_20"] == 1
    assert sp.d1_history_columns(rows[:10], side="LONG", level=100.0, atr=2.0)["perm_level_respect_20"] is None


def test_the_zone_arm_column_is_unknown_when_the_scan_did_not_evaluate_it():
    rows = _bars([100.0] * 6)
    assert sp.d1_history_columns(rows, side="LONG", level=None, atr=2.0)["perm_d1_zone_arm"] is None
    evaluated = sp.d1_history_columns(rows, side="LONG", level=100.0, atr=2.0, zone_arm_evaluated=True)
    assert evaluated["perm_d1_zone_arm"] == "not_armed"
    armed = sp.d1_history_columns(rows, side="LONG", level=100.0, atr=2.0, zone_arm_evaluated=True,
                                  zone_arm={"side": "SHORT", "zone": 2})
    assert armed["perm_d1_zone_arm"] == "SHORT_z2"


def test_the_d1_columns_never_read_a_bar_after_the_scan_date():
    rows = _bars([100.0] * 10 + [101.0, 99.0, 102.0, 103.0, 98.0])
    later = [*rows, {"date": "2099-01-01", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}]
    as_of = rows[-1]["date"]
    assert (sp.d1_history_columns(later, side="LONG", level=100.0, atr=2.0, as_of=as_of)
            == sp.d1_history_columns(rows, side="LONG", level=100.0, atr=2.0))


def test_the_new_scan_columns_append_after_every_older_column():
    # P8b appends its setup-age column after the P11 set; S6 its trendline columns after that; then S15.
    assert sp.SCAN_ROW_COLUMNS[-len(sp.S15_COLUMNS):] == sp.S15_COLUMNS
    older = sp.SCAN_ROW_COLUMNS[:-len(sp.S15_COLUMNS)]
    assert older[-len(sp.TRENDLINE_COLUMNS):] == sp.TRENDLINE_COLUMNS
    assert older[-len(sp.TRENDLINE_COLUMNS) - 1] == sp.SETUP_AGE_COLUMN
    columns = older[:-len(sp.TRENDLINE_COLUMNS) - 1]
    assert tuple(columns[-len(sp.D1_HISTORY_COLUMNS):]) == sp.D1_HISTORY_COLUMNS
    assert columns[: -len(sp.D1_HISTORY_COLUMNS)][-3:] == sp.STAMP_COLUMNS


# ---------------------------------------------------------------------------
# M5-native facets: value and unknown
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("clock", "expected"), [
    ("09:35", "first30"), ("10:00", "first30"), ("10:05", "morning"), ("11:30", "morning"),
    ("11:35", "midday"), ("15:00", "midday"), ("15:05", "last60"), ("16:00", "last60"),
    ("08:00", "time_extended"), ("17:00", "time_extended"),
])
def test_m5_time_bucket_in_exchange_time(clock, expected):
    hour, minute = map(int, clock.split(":"))
    when = datetime(2026, 9, 25, hour, minute, tzinfo=ET)
    assert _m5("m5_time_bucket", alert_bar_close=when.isoformat()) == expected
    # The facet reads exchange time only; the stamp converts (see test_alert_inputs...).
    assert _m5("m5_time_bucket", alert_bar_close=when.astimezone(ZoneInfo("Asia/Tokyo")).isoformat()) == sp.UNKNOWN


def test_m5_time_bucket_without_a_timezone_is_unknown():
    assert _m5("m5_time_bucket", alert_bar_close="2026-09-25T10:00:00") == sp.UNKNOWN
    assert _m5("m5_time_bucket", alert_bar_close="") == sp.UNKNOWN


@pytest.mark.parametrize(("value", "expected"), [
    (0.4, "rvol_below_1"), (1.0, "rvol_1_2"), (2.5, "rvol_2_3"), (7.0, "rvol_3_plus"),
    ("", "unknown"), (None, "unknown"), (-1.0, "unknown"),
])
def test_m5_rvol_bucket(value, expected):
    assert _m5("m5_rvol_bucket", session_rvol=value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    (-2.5, "m5vwap_below_2atr"), (-0.5, "m5vwap_below_0to1atr"), (0.3, "m5vwap_above_0to1atr"),
    (2.1, "m5vwap_above_2atr"), ("", "unknown"),
])
def test_m5_vwap_distance_bucket(value, expected):
    assert _m5("m5_vwap_dist_atr", vwap_dist_atr=value) == expected


@pytest.mark.parametrize(("state", "sign", "expected"), [
    ("COUNTERMOVE_ACTIVE", 1, "spy_pullback"), ("COUNTERMOVE_ARMED", 1, "spy_pullback"),
    ("STABILIZING", -1, "spy_bounce"), ("COUNTERMOVE_ACTIVE", -1, "spy_bounce"),
    ("BULL_IMPULSE", 1, "spy_rally"), ("TREND_RESUMED", 1, "spy_rally"),
    ("BEAR_IMPULSE", -1, "spy_selloff"), ("TREND_RESUMED", -1, "spy_selloff"),
    ("RANGE", 0, "spy_none"), ("OPENING_DISCOVERY", 0, "spy_none"), ("REGIME_FAILED", 1, "spy_none"),
    ("COUNTERMOVE_ACTIVE", 0, "unknown"), ("", 1, "unknown"), ("MYSTERY", 1, "unknown"),
])
def test_m5_spy_state(state, sign, expected):
    assert _m5("m5_spy_state", spy_state=state, spy_side_sign=sign) == expected


def test_m5_bounce_type():
    assert _m5("m5_bounce_type", bounce_type="ema_15") == "bounce_ema_15"
    assert _m5("m5_bounce_type", bounce_type="EOD_VWAP-impulse") == "bounce_eod_vwap-impulse"
    assert _m5("m5_bounce_type", bounce_type="") == sp.UNKNOWN


def test_the_m5_key_is_its_own_versioned_key_and_its_label_shows_the_m5_part():
    key = sp.m5_facets_for({
        "alert_bar_close": datetime(2026, 9, 25, 9, 45, tzinfo=ET).isoformat(), "session_rvol": 2.4,
        "vwap_dist_atr": 0.4, "spy_state": "COUNTERMOVE_ACTIVE", "spy_side_sign": 1, "bounce_type": "ema_15",
    }, "LONG")
    assert key.permutation_rule_version == sp.M5_PERMUTATION_RULE_VERSION
    assert key.compact_key.startswith(f"{sp.M5_PERMUTATION_RULE_VERSION}|ema_15|LONG|")
    assert [name for name, _ in key.facets] == list(sp.M5_FACETS)
    for part in ("first30", "rvol_2_3", "m5vwap_above_0to1atr", "spy_pullback", "bounce_ema_15"):
        assert part in key.label.split("|")
    assert all(spec.group == "m5" for spec in sp.M5_FACETS.values())
    assert not set(sp.M5_FACETS) & set(sp.FACETS)
    empty = sp.m5_facets_for({}, "SHORT")
    assert set(empty.as_dict().values()) == {sp.UNKNOWN}
    assert empty.label == ""


# ---------------------------------------------------------------------------
# the stamp: inputs as the alert carried them, never a bar after the alert bar
# ---------------------------------------------------------------------------
def _m5_bar(when: datetime, close: float, *, high=None, low=None, volume=1000.0) -> dict:
    return {"dt": when, "open": close, "high": close + 0.5 if high is None else high,
            "low": close - 0.5 if low is None else low, "close": close, "volume": volume}


def _alert_bars() -> list[dict]:
    """Naive Pacific bars stamped at their START: 20 from the previous day, then 06:30-06:55 today.

    The alert bar starts 06:55 (the registered row's ``entry_time``) and closes 07:00.
    """
    bars = [_m5_bar(datetime(2026, 9, 24, 11, 20) + timedelta(minutes=5 * n), 100.0) for n in range(20)]
    bars += [_m5_bar(datetime(2026, 9, 25, 6, 30) + timedelta(minutes=5 * n), 100.0) for n in range(5)]
    bars.append(_m5_bar(datetime(2026, 9, 25, 6, 55), 103.0))  # the alert bar, closes 07:00
    return bars


def test_vwap_distance_is_measured_at_the_alert_bar():
    got = stamp.vwap_distance_atr(_alert_bars(), datetime(2026, 9, 25, 7, 0), PT)
    vwap = (5 * 100.0 + 103.0) / 6
    atr = (13 * 1.0 + 3.5) / 14
    assert got == pytest.approx((103.0 - vwap) / atr)


def test_vwap_distance_never_reads_a_bar_after_the_alert_bar():
    base = stamp.vwap_distance_atr(_alert_bars(), datetime(2026, 9, 25, 7, 0), PT)
    later = [*_alert_bars(), _m5_bar(datetime(2026, 9, 25, 7, 0), 50.0, volume=10_000_000.0)]
    assert stamp.vwap_distance_atr(later, datetime(2026, 9, 25, 7, 0), PT) == base


def test_vwap_distance_is_unknown_without_the_alert_bar_or_enough_bars():
    bars = _alert_bars()
    assert stamp.vwap_distance_atr(bars[:-1], datetime(2026, 9, 25, 7, 0), PT) is None  # alert bar missing
    assert stamp.vwap_distance_atr(bars[-6:], datetime(2026, 9, 25, 7, 0), PT) is None  # no ATR14
    assert stamp.vwap_distance_atr([], datetime(2026, 9, 25, 7, 0), PT) is None


def _spy_log(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _spy_row(bar: str, state: str, sign: int, *, usable=True, session="2026-09-25") -> dict:
    return {"schema": "spy_state_shadow_v4", "session_date": session, "bar_ts": bar, "state": state,
            "side_sign": sign, "usable": usable}


def test_spy_state_is_the_last_recorded_state_at_or_before_the_alert_bar(tmp_path):
    log = _spy_log(tmp_path / "spy.jsonl", [
        _spy_row("2026-09-25T06:40:00-07:00", "BULL_IMPULSE", 1),
        {"schema": "spy_episode_shadow_v1", "session_date": "2026-09-25", "direction": "BULL_PULLBACK"},
        _spy_row("2026-09-25T06:55:00-07:00", "COUNTERMOVE_ACTIVE", 1),
        _spy_row("2026-09-25T07:05:00-07:00", "RANGE", 0),  # after the alert: never read
    ])
    reader = stamp.SpyStateReader(log)
    alert = datetime(2026, 9, 25, 7, 0, tzinfo=PT)
    assert reader.state_at(alert) == ("COUNTERMOVE_ACTIVE", 1)
    assert reader.state_at(datetime(2026, 9, 25, 6, 45, tzinfo=PT)) == ("BULL_IMPULSE", 1)
    assert reader.state_at(datetime(2026, 9, 25, 6, 30, tzinfo=PT)) is None
    assert reader.state_at(datetime(2026, 9, 26, 7, 0, tzinfo=PT)) is None  # another session


def test_an_unusable_spy_state_is_unknown(tmp_path):
    log = _spy_log(tmp_path / "spy.jsonl", [_spy_row("2026-09-25T06:55:00-07:00", "BULL_IMPULSE", 1, usable=False)])
    assert stamp.SpyStateReader(log).state_at(datetime(2026, 9, 25, 7, 0, tzinfo=PT)) is None
    assert stamp.SpyStateReader(tmp_path / "missing.jsonl").state_at(datetime(2026, 9, 25, 7, 0, tzinfo=PT)) is None


def _registered(event_id="AAA_long_20260925_06_55_00_ema_15", *, rvol="2.4", entry="2026-09-25T06:55:00",
                logged="2026-09-25T07:00:40-07:00"):
    """A registered row as the bot writes it: ``entry_time`` is the M5 bar START, naive Pacific."""
    return {"event_id": event_id, "event_type": "registered", "trade_date": "2026-09-25", "symbol": "AAA",
            "direction": "long", "entry_time": entry, "logged_at": logged,
            "context_json": json.dumps({"session_rvol": rvol, "family": "ema_15", "atr": 0.4})}


def test_alert_inputs_come_from_the_registered_row():
    inputs = stamp.alert_inputs(_registered(), PT)
    # The bar started 06:55 Pacific, so it closed 07:00 Pacific = 10:00 exchange time.
    assert inputs["alert_bar_close"] == "2026-09-25T10:00:00-04:00"
    assert inputs["alert_bar_complete"] is True
    assert inputs["session_rvol"] == pytest.approx(2.4)
    assert inputs["bounce_type"] == "ema_15"
    assert inputs["alert_date"] == "2026-09-25"
    blank = stamp.alert_inputs({**_registered(rvol=""), "context_json": "not json"}, PT)
    assert blank["session_rvol"] is None
    assert blank["bounce_type"] == "ema_15"  # from the event id


class _StubLookup:
    def stamp(self, symbol, side, trade_date):
        return {"d1_session": "2026-09-24", "permutation_key": "", "permutation_label": "",
                "permutation_rule_version": "", "status": stamp.STATUS_NO_SCAN_ROW, "reason": "stub"}


class _Bot:
    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def m5_chart_bars(self, symbol, max_sessions=2):
        self.calls.append(symbol)
        return list(self.bars)


@pytest.fixture()
def m5_sources(tmp_path, monkeypatch):
    log = _spy_log(tmp_path / "spy.jsonl", [_spy_row("2026-09-25T06:55:00-07:00", "STABILIZING", -1)])
    monkeypatch.setattr(stamp, "_local_tz", lambda: PT)
    monkeypatch.setattr(stamp, "_spy_log_path", lambda: log)
    path = tmp_path / "m5_setup_key_stamps.jsonl"
    monkeypatch.setattr(project_paths, "M5_SETUP_KEY_STAMPS_FILE", path)
    monkeypatch.setattr(project_paths, "D1_ENVIRONMENT_FILE", tmp_path / "no_d1_environment.jsonl")
    monkeypatch.setattr(stamp, "_market_today", lambda: "2026-09-25")
    monkeypatch.delenv(stamp.ENABLED_ENV, raising=False)
    bot = _Bot(_alert_bars())
    stamp.register_bar_source(bot)
    yield {"sidecar": path, "bot": bot}
    stamp.register_bar_source(None)
    stamp.reset_for_tests()


def test_the_sidecar_row_carries_the_d1_key_and_an_m5_key(m5_sources):
    stamp.reset_for_tests(_StubLookup())
    row = _registered()
    before = dict(row)
    assert stamp.submit(row) is True
    assert stamp.drain()
    assert row == before  # the outcome row is never changed
    records = [json.loads(line) for line in m5_sources["sidecar"].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    # the D1 part is unchanged in shape
    for name in (*stamp.STAMP_FIELDS, "status", "d1_session", "event_id", "schema"):
        assert name in record
    assert record["status"] == stamp.STATUS_NO_SCAN_ROW
    # the M5 part stands on its own
    assert record["m5_rule_version"] == sp.M5_PERMUTATION_RULE_VERSION
    assert record["m5_key"].startswith(f"{sp.M5_PERMUTATION_RULE_VERSION}|ema_15|LONG|")
    assert record["m5_facets"] == {
        "m5_time_bucket": "first30", "m5_rvol_bucket": "rvol_2_3", "m5_vwap_dist_atr": "m5vwap_above_2atr",
        "m5_spy_state": "spy_bounce", "m5_bounce_type": "bounce_ema_15",
        # S6: 26 bars, a partial previous day, an alert inside the first 30 minutes, no D1 label.
        "m5_ema_stack": sp.UNKNOWN, "m5_pdh_pdl": sp.UNKNOWN, "m5_open_range": sp.UNKNOWN,
        "m5_compression": sp.UNKNOWN, "m5_side_vs_d1_env": sp.UNKNOWN,
    }
    assert "first30" in record["m5_label"]
    assert record["m5_inputs"]["vwap_dist_atr"] == pytest.approx(2.1213, abs=1e-3)
    assert m5_sources["bot"].calls == ["AAA"]


def test_with_no_bar_source_or_spy_log_those_facets_are_unknown(m5_sources, monkeypatch, tmp_path):
    stamp.register_bar_source(None)
    monkeypatch.setattr(stamp, "_spy_log_path", lambda: tmp_path / "none.jsonl")
    record = stamp.record_for(_registered(), _StubLookup())
    assert record["m5_facets"]["m5_vwap_dist_atr"] == sp.UNKNOWN
    assert record["m5_facets"]["m5_spy_state"] == sp.UNKNOWN
    assert record["m5_facets"]["m5_time_bucket"] == "first30"


def test_a_bar_source_that_raises_costs_only_the_facet(m5_sources):
    class Broken:
        def m5_chart_bars(self, symbol, max_sessions=2):
            raise RuntimeError("cache gone")

    stamp.register_bar_source(Broken())
    record = stamp.record_for(_registered(), _StubLookup())
    assert record["m5_facets"]["m5_vwap_dist_atr"] == sp.UNKNOWN
    assert record["m5_facets"]["m5_bounce_type"] == "bounce_ema_15"


# ---------------------------------------------------------------------------
# the backfill joins both keys; the search runs the m5 population on the union
# ---------------------------------------------------------------------------
def _session_after(day: date) -> date:
    day += timedelta(days=1)
    while not market_calendar.is_session(day):
        day += timedelta(days=1)
    return day


PREV = date(2026, 9, 24)
TODAY = _session_after(PREV)


def _csv(path: Path, rows: list[dict]) -> Path:
    import csv

    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _backfill_files(tmp_path):
    history = _csv(tmp_path / "hist.csv", [{
        "run_id": "r1", "run_timestamp": f"{PREV}T13:05:00", "run_date": PREV.isoformat(),
        "last_trade_date": PREV.isoformat(), "symbol": "AAA", "side": "LONG", "setup_family": "avwap_band_bounce",
        "last_close": 100.0, "atr20": 2.0, "perm_d1_zone_arm": "LONG_z1",
    }])
    rows = []
    for symbol, clock in (("AAA", "06:40:00"), ("BBB", "12:30:00")):
        event_id = f"{symbol}_long_{TODAY.strftime('%Y%m%d')}_{clock.replace(':', '_')}_ema_15"
        common = {"event_id": event_id, "trade_date": TODAY.isoformat(), "symbol": symbol, "direction": "long",
                  "entry_time": f"{TODAY}T{clock}"}
        rows.append({**common, "event_type": "registered", "bars_elapsed": 0, "minutes_elapsed": "", "mfe_r": "",
                     "stop_hit": "False", "context_json": json.dumps({"session_rvol": 1.5, "family": "ema_15"}),
                     "logged_at": f"{TODAY}T{clock}-07:00"})
        rows.append({**common, "event_type": "update", "bars_elapsed": 8, "minutes_elapsed": 40, "mfe_r": 1.2,
                     "stop_hit": "False", "context_json": "{}", "logged_at": f"{TODAY}T{clock}-07:00"})
    m5 = _csv(tmp_path / "m5.csv", rows)
    horizons = _csv(tmp_path / "h.csv", [{"scan_row_id": "x"}])
    return history, m5, horizons


def test_the_backfill_joins_the_d1_key_and_the_m5_key(tmp_path, monkeypatch):
    monkeypatch.setattr(stamp, "_local_tz", lambda: PT)
    history, m5, horizons = _backfill_files(tmp_path)
    sidecar = tmp_path / "stamps.jsonl"
    live_event = f"AAA_long_{TODAY.strftime('%Y%m%d')}_06_40_00_ema_15"
    live = {"m5_time_bucket": "first30", "m5_rvol_bucket": "rvol_1_2", "m5_vwap_dist_atr": "m5vwap_below_2atr",
            "m5_spy_state": "spy_rally", "m5_bounce_type": "bounce_ema_15",
            # S6: the structure facets as a live stamp would carry them.
            "m5_ema_stack": "m5ema_8over21_above_both", "m5_pdh_pdl": "above_pdh", "m5_open_range": "above_or",
            "m5_compression": "squeeze_break_up", "m5_side_vs_d1_env": "side_with_d1_trend"}
    stamp.append_record({"schema": stamp.SCHEMA, "event_id": live_event, "status": stamp.STATUS_NO_SCAN_ROW,
                         "m5_rule_version": sp.M5_PERMUTATION_RULE_VERSION, "m5_facets": live}, sidecar)
    result = bf.build_permutation_outcomes(history, horizons=horizons, m5_outcomes=m5, last_completed=TODAY,
                                           m5_stamps=sidecar)
    by_symbol = {row["symbol"]: row for row in result.rows if row["population"] == bf.POPULATION_M5}
    assert set(by_symbol) == {"AAA", "BBB"}
    # AAA: the live M5 stamp wins; the D1 key still comes from the previous session's scan row.
    assert {name: by_symbol["AAA"][bf.facet_column(name)] for name in sp.M5_FACETS} == live
    assert by_symbol["AAA"]["f_d1_zone_arm"] == "zone_arm_long_z1"
    # BBB: no stamp, so the M5 facets the registered row carried are recomputed; the rest are unknown.
    bbb = by_symbol["BBB"]
    assert bbb["f_m5_time_bucket"] == "last60"  # 12:30 PT = 15:30 ET
    assert bbb["f_m5_rvol_bucket"] == "rvol_1_2"
    assert bbb["f_m5_bounce_type"] == "bounce_ema_15"
    assert bbb["f_m5_vwap_dist_atr"] == sp.UNKNOWN and bbb["f_m5_spy_state"] == sp.UNKNOWN
    assert result.counts["m5_live_m5_stamped"] == 1
    for name in sp.M5_FACETS:
        assert bf.facet_column(name) in bf.output_columns()


def test_swing_rows_carry_unknown_m5_facets(tmp_path):
    keyed = bf.KeyedRow(symbol="AAA", side="LONG", session="2026-09-24", family="f", scan_row_id="AAA:2026-09-24:r",
                        last_close=100.0, atr20=2.0, run_id="r", run_timestamp="", last_trade_date="2026-09-24",
                        priority_bucket="", facets={name: sp.UNKNOWN for name in sp.FACETS},
                        rule_version=sp.PERMUTATION_RULE_VERSION)
    row = bf._swing_row(keyed, 1, True, 1.0, 100.0)
    assert all(row[bf.facet_column(name)] == sp.UNKNOWN for name in sp.M5_FACETS)


def test_the_search_runs_the_m5_population_on_d1_and_m5_facets_together(tmp_path):
    assert "f_m5_time_bucket" in bf.output_columns()
    facet_columns = [column for column in bf.output_columns() if column.startswith("f_")]
    rng = random.Random(11)
    sessions = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(60)]
    rows = []
    for day in sessions:
        for n in range(20):
            early = rng.random() < 0.3
            win = rng.random() < (0.85 if early else 0.4)
            row = {column: sp.UNKNOWN for column in facet_columns}
            row.update({"population": bf.POPULATION_M5, "family": "ema_15", "side": "LONG", "horizon": 0,
                        "session": day, "episode_id": f"{day}:{n}", "win": win, "r": 1.0 if win else -1.0,
                        "f_m5_time_bucket": "first30" if early else "midday",
                        "f_hv_level": rng.choice(["no_hv_level", "hv_level_strong_below"])})
            rows.append(row)
    report = search.build_report(rows, ledger_root=tmp_path)
    family = report["populations"]["m5"]["horizons"]["0"]["families"]["ema_15 LONG"]
    assert family["verdict"] == search.VERDICT_KEY
    assert family["keys"][0]["facets"] == {"m5_time_bucket": "first30"}
    assert "m5_time_bucket=first30" in family["keys"][0]["label"]


# ---------------------------------------------------------------------------
# the scan golden: output unchanged, new columns present
# ---------------------------------------------------------------------------
def _parity_module():
    path = Path(__file__).with_name("test_setup_permutation_scan_parity.py")
    spec = importlib.util.spec_from_file_location("_p11_scan_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan_runs(tmp_path_factory):
    parity = _parity_module()
    base = tmp_path_factory.mktemp("p11-scan")
    # 300 sessions: enough for ATR14 over 252 sessions and the 52-week low.
    return parity, parity._run(base, "on", 300), parity._run(base, "off", 300)


def test_the_scan_writes_the_p11_columns(scan_runs):
    _parity, stamped, _plain = scan_runs
    row = stamped["history"][-1]
    header = list(row)
    tail = len(sp.TRENDLINE_COLUMNS) + len(sp.S15_COLUMNS)  # S6, then S15, append after the setup age
    assert header[-len(sp.D1_HISTORY_COLUMNS) - 1 - tail:-1 - tail] == list(sp.D1_HISTORY_COLUMNS)
    assert header[-1 - tail] == sp.SETUP_AGE_COLUMN
    for column in ("perm_atr14_pctile_252", "perm_low_52w_dist_atr", "perm_closes_right_of_level_5",
                   "perm_level_respect_20", "perm_d1_zone_arm"):
        assert row[column] not in ("", None), column
    key = dict(part.split("=", 1) for part in row["permutation_key"].split("|")[3].split(";"))
    assert key["atr_percentile"].startswith("atr_pctile_")
    assert key["d1_zone_arm"] != sp.UNKNOWN


def test_the_scan_output_is_identical_with_and_without_the_p11_columns(scan_runs):
    parity, stamped, plain = scan_runs
    assert stamped["priority_row"] and plain["priority_row"]
    assert parity._strip(stamped["priority_row"]) == parity._strip(plain["priority_row"])
    assert parity._strip(stamped["ai_state_entry"]) == parity._strip(plain["ai_state_entry"])
    assert len(stamped["history"]) == len(plain["history"])
    for on_row, off_row in zip(stamped["history"], plain["history"], strict=True):
        assert parity._strip(on_row) == parity._strip(off_row)
    assert all(plain["history"][-1][column] == "" for column in sp.D1_HISTORY_COLUMNS)


# ---------------------------------------------------------------------------
# review fix: entry_time is the M5 bar START (only H1 families stamp the close)
# ---------------------------------------------------------------------------
def _facets_of(row):
    return sp.m5_facets_for(stamp.alert_inputs(row, PT), "LONG").as_dict()


def test_the_opening_bar_alert_is_first30_not_extended_hours():
    row = _registered("AAA_long_20260925_06_30_00_ema_15", entry="2026-09-25T06:30:00",
                      logged="2026-09-25T06:35:20-07:00")
    assert stamp.alert_inputs(row, PT)["alert_bar_close"] == "2026-09-25T09:35:00-04:00"
    assert _facets_of(row)["m5_time_bucket"] == "first30"


def test_the_last_bar_alert_is_last60():
    row = _registered("AAA_long_20260925_12_55_00_ema_15", entry="2026-09-25T12:55:00",
                      logged="2026-09-25T13:00:30-07:00")
    assert stamp.alert_inputs(row, PT)["alert_bar_close"] == "2026-09-25T16:00:00-04:00"
    assert _facets_of(row)["m5_time_bucket"] == "last60"


def test_an_h1_row_already_carries_its_bar_close():
    row = _registered("AAA_long_20260925_07_00_00_h1_ema10_bounce", entry="2026-09-25T07:00:00",
                      logged="2026-09-25T07:01:00-07:00")
    assert stamp.alert_inputs(row, PT)["alert_bar_close"] == "2026-09-25T10:00:00-04:00"


def test_the_opening_bar_alert_measures_vwap_on_its_own_bar(m5_sources):
    row = _registered("AAA_long_20260925_06_30_00_ema_15", entry="2026-09-25T06:30:00",
                      logged="2026-09-25T06:35:20-07:00")
    record = stamp.record_for(row, _StubLookup())
    # One regular-session bar (06:30, close 100): VWAP = its typical price = 100, close 100.
    assert record["m5_inputs"]["vwap_dist_atr"] == pytest.approx(0.0)
    assert record["m5_facets"]["m5_vwap_dist_atr"] == "m5vwap_above_0to1atr"


def test_a_bar_still_forming_when_the_alert_was_logged_is_never_measured(m5_sources):
    # regime_pause rows can be logged under 5 minutes after the bar start: the bar had not closed.
    row = _registered("AAA_long_20260925_06_55_00_regime_pause_rw", entry="2026-09-25T06:55:00",
                      logged="2026-09-25T06:57:00-07:00")
    inputs = stamp.alert_inputs(row, PT)
    assert inputs["alert_bar_complete"] is False
    record = stamp.record_for(row, _StubLookup())
    assert record["m5_facets"]["m5_vwap_dist_atr"] == sp.UNKNOWN
    assert m5_sources["bot"].calls == []  # the cache is not even read
    # Without a logged_at the bar's completeness is unknown, so VWAP is too.
    assert stamp.alert_inputs({**row, "logged_at": ""}, PT)["alert_bar_complete"] is None


def test_spy_state_falls_back_to_the_last_usable_row(tmp_path):
    log = _spy_log(tmp_path / "spy.jsonl", [
        _spy_row("2026-09-25T06:40:00-07:00", "BULL_IMPULSE", 1),
        _spy_row("2026-09-25T06:55:00-07:00", "COUNTERMOVE_ACTIVE", 1, usable=False),
    ])
    assert stamp.SpyStateReader(log).state_at(datetime(2026, 9, 25, 7, 0, tzinfo=PT)) == ("BULL_IMPULSE", 1)
