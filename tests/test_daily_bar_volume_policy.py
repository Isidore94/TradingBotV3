"""R10.V step 3 - the write seam takes shares, and a collision prefers them.

Two rules, and one deliberate exception to the first.

* **Only share-denominated volume is written to the durable store.** An
  IB-sourced row contributes its PRICES and `volume=NaN` with
  `volume_unit=lots_rth`. Never a rescaled number: the measured ratio is
  symbol-dependent (SPY 1.0x, TSLA 56x, AAPL 81x, A 162x, NVDA 188x), so a x100
  conversion would replace a visible error with an invisible one.
* **A date collision prefers `shares` over `unknown` over a blanked row**, so a
  Yahoo row is never overwritten by an IB row again. Among rows of equal
  standing the later one still wins, which is the previous behaviour.

The exception: **`unknown` legacy rows keep their volume.** Blanking them would
empty the volume column of the entire existing store between this step and the
step-4 backfill, and an AVWAP with no weights is not a safer answer than an
AVWAP with an old one - it is no answer at all, for every symbol, live. The
grandfathering ends when step 4's exit gate reads zero rows with
`volume_unit != shares`.

A blanked row must still be READABLE, so `NaN` volume survives normalization
(dropping it would delete the price bar too) and every weighting loop skips it
exactly as it skips a zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402


def _frame(rows: int = 5, *, start: str = "2026-08-03", volume: float = 1_000_000.0):
    days = pd.bdate_range(start, periods=rows)
    return pd.DataFrame(
        {
            "datetime": days,
            "open": [10.0 + i for i in range(rows)],
            "high": [10.5 + i for i in range(rows)],
            "low": [9.5 + i for i in range(rows)],
            "close": [10.2 + i for i in range(rows)],
            "volume": [volume + i for i in range(rows)],
        }
    )


def _stamped(frame, source):
    return master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(frame, source)
    )


# ---------------------------------------------------------------------------
# the write seam
# ---------------------------------------------------------------------------
def test_an_ib_row_is_written_with_its_prices_and_no_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    frame = _stamped(_frame(), master_avwap.DAILY_BAR_SOURCE_IBKR)
    master_avwap._persist_durable_daily_bars("IBW", frame)
    stored = pd.read_parquet(tmp_path / "IBW.parquet")
    assert stored["volume"].isna().all(), "IB round-lot volume must never be stored"
    assert set(stored["volume_unit"]) == {"lots_rth"}, "and the row must say why"
    assert stored["close"].notna().all(), "the prices are fine and are kept"


def test_a_yahoo_row_is_written_with_its_volume_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    frame = _stamped(_frame(), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    master_avwap._persist_durable_daily_bars("YHW", frame)
    stored = pd.read_parquet(tmp_path / "YHW.parquet")
    assert stored["volume"].notna().all()
    assert float(stored["volume"].iloc[0]) == 1_000_000.0


def test_legacy_unknown_rows_keep_their_volume_until_the_backfill(tmp_path, monkeypatch):
    """The deliberate exception. Blanking these would empty the whole store."""
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    frame = _stamped(_frame(), master_avwap.DAILY_BAR_SOURCE_CACHE)
    assert set(frame["volume_unit"]) == {"unknown"}
    master_avwap._persist_durable_daily_bars("UNK", frame)
    stored = pd.read_parquet(tmp_path / "UNK.parquet")
    assert stored["volume"].notna().all()


def test_the_blanking_is_not_a_rescale(tmp_path, monkeypatch):
    """No conversion factor appears anywhere: the value is absent, not adjusted."""
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    frame = _stamped(_frame(volume=5_000_000.0), master_avwap.DAILY_BAR_SOURCE_IBKR)
    master_avwap._persist_durable_daily_bars("NR", frame)
    stored = pd.read_parquet(tmp_path / "NR.parquet")
    assert stored["volume"].isna().all()
    assert not (stored["volume"].fillna(0) > 0).any()


# ---------------------------------------------------------------------------
# a blanked row stays readable
# ---------------------------------------------------------------------------
def test_a_nan_volume_row_survives_normalization_with_its_prices():
    frame = _frame(3)
    frame.loc[1, "volume"] = np.nan
    normalized = _stamped(frame, master_avwap.DAILY_BAR_SOURCE_YAHOO)
    assert len(normalized) == 3, "dropping the row would delete the price bar too"
    assert pd.isna(normalized.loc[1, "volume"])


def test_a_row_missing_a_price_is_still_dropped():
    frame = _frame(3)
    frame.loc[1, "close"] = np.nan
    normalized = _stamped(frame, master_avwap.DAILY_BAR_SOURCE_YAHOO)
    assert len(normalized) == 2


@pytest.mark.parametrize(
    "function",
    ["calc_anchored_vwap_bands", "calc_anchored_vwap_band_history"],
)
def test_a_blank_volume_bar_is_skipped_exactly_like_a_zero_volume_bar(function):
    """NaN is not <= 0, so without this guard one blank bar poisons the level.

    This is NOT a change to the sigma formula (plan.md sec 5) - the accumulation
    is untouched. It decides which bars enter it, on the same rule the function
    already applied to a zero.
    """
    blank = _frame(4)
    blank.loc[2, "volume"] = np.nan
    dropped = _frame(4).drop(index=2).reset_index(drop=True)

    if function == "calc_anchored_vwap_bands":
        with_blank = master_avwap.calc_anchored_vwap_bands(blank, 0)
        without = master_avwap.calc_anchored_vwap_bands(dropped, 0)
        assert with_blank[0] == pytest.approx(without[0], abs=1e-12)
        assert with_blank[1] == pytest.approx(without[1], abs=1e-12)
        assert not pd.isna(with_blank[0])
    else:
        anchor = blank["datetime"].iloc[0].date().isoformat()
        with_blank = master_avwap.calc_anchored_vwap_band_history(blank, anchor)
        without = master_avwap.calc_anchored_vwap_band_history(dropped, anchor)
        last = max(with_blank)
        assert with_blank[last]["vwap"] == pytest.approx(without[max(without)]["vwap"], abs=1e-12)


def test_an_all_blank_series_returns_the_unmeasurable_contract():
    """No weights at all is UNKNOWN, and the function already says so with NaN."""
    frame = _frame(3)
    frame["volume"] = np.nan
    vwap, stdev, bands = master_avwap.calc_anchored_vwap_bands(frame, 0)
    assert pd.isna(vwap) and pd.isna(stdev) and bands == {}


# ---------------------------------------------------------------------------
# the collision rule
# ---------------------------------------------------------------------------
def _collide(first_source, second_source):
    """Same session, two sources; `second` arrives later."""
    first = _stamped(_frame(1, volume=111.0), first_source)
    second = _stamped(_frame(1, volume=222.0), second_source)
    return master_avwap._merge_daily_bar_frames(first, second)


def test_a_yahoo_row_is_never_overwritten_by_an_ib_row():
    """The headline. Before this, `keep="last"` handed the day to whoever ran last."""
    merged = _collide(master_avwap.DAILY_BAR_SOURCE_YAHOO, master_avwap.DAILY_BAR_SOURCE_IBKR)
    assert len(merged) == 1
    assert merged["source"].iloc[0] == "yahoo"
    assert float(merged["volume"].iloc[0]) == 111.0


def test_shares_win_from_either_direction():
    merged = _collide(master_avwap.DAILY_BAR_SOURCE_IBKR, master_avwap.DAILY_BAR_SOURCE_YAHOO)
    assert merged["source"].iloc[0] == "yahoo"
    assert float(merged["volume"].iloc[0]) == 222.0


def test_unknown_beats_a_blanked_row_because_it_still_has_a_number():
    merged = _collide(master_avwap.DAILY_BAR_SOURCE_CACHE, master_avwap.DAILY_BAR_SOURCE_IBKR)
    assert merged["source"].iloc[0] == "unknown"
    assert float(merged["volume"].iloc[0]) == 111.0


def test_shares_beat_unknown_so_the_backfill_is_not_undone():
    merged = _collide(master_avwap.DAILY_BAR_SOURCE_YAHOO, master_avwap.DAILY_BAR_SOURCE_CACHE)
    assert merged["source"].iloc[0] == "yahoo"


def test_among_equals_the_later_row_still_wins():
    """The previous semantics, preserved: a fresh Yahoo bar replaces an old one."""
    merged = _collide(master_avwap.DAILY_BAR_SOURCE_YAHOO, master_avwap.DAILY_BAR_SOURCE_YAHOO)
    assert float(merged["volume"].iloc[0]) == 222.0


def test_the_rest_of_the_history_is_untouched_by_a_collision():
    older = _stamped(_frame(3, start="2026-08-03"), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    newer = _stamped(_frame(3, start="2026-08-05"), master_avwap.DAILY_BAR_SOURCE_IBKR)
    merged = master_avwap._merge_daily_bar_frames(older, newer)
    # 08-03..08-05 and 08-05..08-07: five sessions, one of them contested.
    assert len(merged) == 5
    assert merged["datetime"].is_monotonic_increasing
    by_day = dict(zip(merged["datetime"].dt.strftime("%Y-%m-%d"), merged["source"]))
    assert by_day["2026-08-03"] == "yahoo"
    assert by_day["2026-08-05"] == "yahoo", "the overlap keeps the share-denominated row"
    assert by_day["2026-08-07"] == "ibkr", "a day only IB has is still kept, prices only"


# ---------------------------------------------------------------------------
# what a reader sees
# ---------------------------------------------------------------------------
def test_the_chart_reads_a_blank_volume_as_no_bar_rather_than_as_nan(tmp_path, monkeypatch):
    import chart_snapshot
    import setup_playbook_study

    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    monkeypatch.setattr(setup_playbook_study, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    monkeypatch.setattr(setup_playbook_study, "MIN_BARS_REQUIRED", 1, raising=False)
    chart_snapshot._daily_bars_cache.clear()
    frame = _stamped(_frame(), master_avwap.DAILY_BAR_SOURCE_IBKR)
    master_avwap._persist_durable_daily_bars("CHT", frame)
    bars = chart_snapshot.load_d1_bars("CHT")
    assert len(bars) == 5
    assert all(bar["volume"] == 0.0 for bar in bars), "NaN must not reach the paint path"
    assert all(bar["close"] > 0 for bar in bars)


# ---------------------------------------------------------------------------
# the other volume readers in the scanner
# ---------------------------------------------------------------------------
def test_an_unmeasurable_average_volume_reads_zero_and_fails_the_liquidity_filter():
    """`int(nan)` raises, and 0 is the value that correctly REJECTS a candidate.

    Uncertainty is not confirmation: a name whose 20-day volume cannot be
    measured must not pass a minimum-liquidity gate by accident.
    """
    blanks = pd.Series([np.nan] * 20)
    assert (master_avwap._coerce_int(blanks.tail(20).mean()) or 0) == 0
    assert 0 < master_avwap.MIN_AVG_VOLUME_20D


def test_a_blank_last_volume_is_none_rather_than_zero():
    """`last_volume` is a bucketed liquidity factor; 0 would read as illiquid."""
    assert master_avwap._to_float(float("nan")) is None
    assert master_avwap._to_float(1_234.0) == 1_234.0
