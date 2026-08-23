"""R10.V step 4 - the volume backfill, and the cliff measure it is judged by.

No network: every test injects a downloader. What is under test is the policy,
not yfinance.

The rules that matter more than the mechanics:

* **Prices are never touched.** Only volume and the two provenance columns move.
* **A row Yahoo does not cover keeps what it had and is counted**, never guessed
  and never blanked.
* **Nothing is written without `--apply`**, and `--apply` freezes a verified copy
  of the whole directory first or refuses to run.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402
from ops import backfill_daily_bar_volume as backfill  # noqa: E402
from ops import daily_bar_cliff as cliff  # noqa: E402

NOW = datetime(2026, 8, 22, 23, 0, tzinfo=timezone.utc)


def _frame(days: int = 40, *, splice_at: int | None = None, base: float = 20_000_000.0):
    stamps = pd.bdate_range("2026-06-01", periods=days)
    volumes = []
    for index in range(days):
        volume = base + index * 1_000.0
        if splice_at is not None and index >= splice_at:
            volume = volume / 100.0
        volumes.append(volume)
    return pd.DataFrame(
        {
            "datetime": stamps,
            "open": [10.0 + i * 0.1 for i in range(days)],
            "high": [10.5 + i * 0.1 for i in range(days)],
            "low": [9.5 + i * 0.1 for i in range(days)],
            "close": [10.2 + i * 0.1 for i in range(days)],
            "volume": volumes,
        }
    )


def _store(tmp_path: Path, symbol: str, frame) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{symbol}.parquet"
    frame.to_parquet(path, index=False)
    return path


def _downloader_for(frames: dict[str, pd.DataFrame]):
    """A batched download that returns share volume for the named symbols."""

    def download(symbols, *, period):
        wanted = [symbol for symbol in symbols if symbol in frames]
        if not wanted:
            return pd.DataFrame()
        pieces = {}
        for symbol in wanted:
            frame = frames[symbol].set_index("datetime")
            pieces[symbol] = frame[["open", "high", "low", "close", "volume"]]
        return pd.concat(pieces, axis=1)

    return download


# ---------------------------------------------------------------------------
# the cliff measure
# ---------------------------------------------------------------------------
def test_a_spliced_series_is_flagged_with_its_date():
    reading = cliff.first_cliff(_frame(40, splice_at=20))
    assert reading.is_cliff
    assert reading.direction == "down"
    assert reading.ratio == pytest.approx(100.0, rel=0.05)
    assert reading.date == _frame(40)["datetime"].iloc[20].date().isoformat()


def test_a_single_source_series_is_clean():
    assert not cliff.first_cliff(_frame(40)).is_cliff


def test_a_short_series_is_unmeasurable_rather_than_clean():
    reading = cliff.first_cliff(_frame(8))
    assert reading.measurable is False
    assert not reading.is_cliff
    assert "fewer than" in reading.reason


def test_blank_volumes_are_not_read_as_zeros():
    frame = _frame(40)
    frame.loc[5, "volume"] = np.nan
    reading = cliff.first_cliff(frame)
    assert reading.measurable and not reading.is_cliff
    assert reading.bars == 39


def test_an_unreadable_file_counts_as_unmeasurable(tmp_path):
    (tmp_path / "BAD.parquet").write_bytes(b"nope")
    report = cliff.scan_store(tmp_path)
    assert report.files == 1
    assert report.unmeasurable == 1
    assert report.cliffed == 0


def test_the_store_scan_counts_each_outcome(tmp_path):
    _store(tmp_path, "GOOD", _frame(40))
    _store(tmp_path, "SPLICED", _frame(40, splice_at=20))
    _store(tmp_path, "SHORT", _frame(8))
    report = cliff.scan_store(tmp_path)
    assert (report.files, report.cliffed, report.unmeasurable) == (3, 1, 1)
    assert report.as_dict()["median_cliff_ratio"] == pytest.approx(100.0, rel=0.05)


# ---------------------------------------------------------------------------
# the rewrite
# ---------------------------------------------------------------------------
def test_only_volume_and_provenance_move():
    original = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(_frame(40, splice_at=20), master_avwap.DAILY_BAR_SOURCE_CACHE)
    )
    truth = _frame(40)
    volumes = backfill.yahoo_volume_by_date(truth.set_index("datetime"))
    rewritten, count, unknown = backfill.rewrite_frame(original, volumes)
    assert count == 40 and unknown == 0
    for column in ("datetime", "open", "high", "low", "close"):
        pd.testing.assert_series_equal(rewritten[column], original[column], check_names=False)
    assert set(rewritten["source"]) == {"yahoo"}
    assert set(rewritten["volume_unit"]) == {"shares"}
    assert cliff.first_cliff(rewritten).is_cliff is False


def test_a_row_yahoo_does_not_cover_keeps_its_value_and_is_counted():
    original = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(_frame(40, splice_at=20), master_avwap.DAILY_BAR_SOURCE_CACHE)
    )
    partial = _frame(40).iloc[:30]
    volumes = backfill.yahoo_volume_by_date(partial.set_index("datetime"))
    rewritten, count, unknown = backfill.rewrite_frame(original, volumes)
    assert count == 30
    assert unknown == 10, "the uncovered rows are counted, not silently kept"
    assert float(rewritten["volume"].iloc[35]) == float(original["volume"].iloc[35])
    assert rewritten["volume_unit"].iloc[35] == "unknown"
    assert rewritten["volume"].notna().all(), "an uncovered row is never blanked"


def test_a_row_already_in_shares_is_not_counted_as_left_unknown():
    original = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(_frame(40), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    )
    rewritten, count, unknown = backfill.rewrite_frame(original, {})
    assert (count, unknown) == (0, 0)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def test_a_dry_run_writes_nothing_and_still_reports(tmp_path):
    store = tmp_path / "store"
    _store(store, "AAA", _frame(40, splice_at=20))
    before = (store / "AAA.parquet").read_bytes()
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({"AAA": _frame(40)}),
        apply=False,
        now=NOW,
    )
    assert report.applied is False
    assert report.files_changed == 1
    assert report.rows_rewritten == 40
    assert report.cliffed_before == 1 and report.cliffed_after == 0
    assert (store / "AAA.parquet").read_bytes() == before, "a dry run must not write"
    assert not (tmp_path / "frozen").exists(), "and must not freeze a copy either"


def test_apply_freezes_the_whole_directory_before_writing(tmp_path):
    store = tmp_path / "store"
    _store(store, "AAA", _frame(40, splice_at=20))
    _store(store, "BBB", _frame(40, splice_at=25))
    frozen = tmp_path / "frozen"
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=frozen,
        downloader=_downloader_for({"AAA": _frame(40), "BBB": _frame(40)}),
        apply=True,
        now=NOW,
    )
    copy = Path(report.frozen_copy)
    assert copy.exists() and copy.name == "daily_bars_pre_backfill_2026-08-22"
    assert len(list(copy.glob("*.parquet"))) == 2
    # the frozen copy still carries the splice; the live store does not
    assert cliff.first_cliff(pd.read_parquet(copy / "AAA.parquet")).is_cliff
    assert not cliff.first_cliff(pd.read_parquet(store / "AAA.parquet")).is_cliff


def test_apply_refuses_to_overwrite_an_existing_frozen_copy(tmp_path):
    store = tmp_path / "store"
    _store(store, "AAA", _frame(40))
    frozen = tmp_path / "frozen"
    (frozen / "daily_bars_pre_backfill_2026-08-22").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        backfill.run_backfill(
            store_dir=store,
            frozen_dir=frozen,
            downloader=_downloader_for({"AAA": _frame(40)}),
            apply=True,
            now=NOW,
        )


def test_the_written_files_carry_the_v2_schema(tmp_path):
    store = tmp_path / "store"
    _store(store, "AAA", _frame(40, splice_at=20))
    backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({"AAA": _frame(40)}),
        apply=True,
        now=NOW,
    )
    assert master_avwap.daily_bars_schema_version(store / "AAA.parquet") == "v2"


def test_a_symbol_yahoo_has_no_data_for_is_named_not_silently_skipped(tmp_path):
    store = tmp_path / "store"
    _store(store, "AAA", _frame(40, splice_at=20))
    _store(store, "DEAD", _frame(40, splice_at=20))
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({"AAA": _frame(40)}),
        apply=True,
        now=NOW,
    )
    assert report.symbols_missing == ["DEAD"]
    outcomes = {item.symbol: item for item in report.outcomes}
    assert outcomes["DEAD"].status == "no_yahoo_data"
    assert outcomes["DEAD"].rows_left_unknown == 40
    assert cliff.first_cliff(pd.read_parquet(store / "DEAD.parquet")).is_cliff, \
        "an un-backfillable file is left exactly as it was"


def test_a_failed_batch_does_not_take_the_run_down(tmp_path):
    store = tmp_path / "store"
    _store(store, "AAA", _frame(40, splice_at=20))

    def angry(symbols, *, period):
        raise RuntimeError("yahoo said no")

    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=angry,
        apply=False,
        now=NOW,
    )
    assert report.symbols_downloaded == 0
    assert report.rows_rewritten == 0
    assert report.symbols_missing == ["AAA"]


def test_an_unreadable_file_is_counted_as_a_failure_not_a_success(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    (store / "BAD.parquet").write_bytes(b"nope")
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({}),
        apply=False,
        now=NOW,
    )
    assert report.files_failed == 1
    assert report.outcomes[0].status == "unreadable"


def test_the_manifest_records_before_and_after_per_file(tmp_path):
    import json

    store = tmp_path / "store"
    _store(store, "AAA", _frame(40, splice_at=20))
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({"AAA": _frame(40)}),
        apply=False,
        now=NOW,
    )
    path = backfill.write_manifest(report, tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["outcomes"][0]
    assert entry["symbol"] == "AAA"
    assert entry["cliff_before"] and entry["cliff_after"] is None
    assert entry["rows_rewritten"] == 40
    assert payload["applied"] is False
    assert payload["store_dir"].endswith("store")


def test_the_backfill_never_touches_ib():
    """Zero IB traffic is a property of the code, not a promise in a docstring."""
    source = Path(backfill.__file__).read_text(encoding="utf-8")
    for forbidden in ("ibapi", "IBApi", "reqHistoricalData", "connect_daily_data_client"):
        assert forbidden not in source
    assert "auto_adjust=False" in source


def test_the_reported_date_is_the_splice_bar_not_the_window_that_noticed_it():
    """A rolling median crosses about half a window early; the manifest is read
    by date, so the boundary is refined to the bar the step happened on."""
    for splice_at in (15, 20, 25, 30):
        reading = cliff.first_cliff(_frame(45, splice_at=splice_at))
        expected = _frame(45)["datetime"].iloc[splice_at].date().isoformat()
        assert reading.date == expected, f"splice at {splice_at} dated {reading.date}"


def test_an_upward_step_is_reported_with_its_direction():
    """A partially-applied backfill would look like this, and must be visible."""
    frame = _frame(40, splice_at=20)
    frame["volume"] = frame["volume"].iloc[::-1].to_numpy()
    reading = cliff.first_cliff(frame)
    assert reading.is_cliff and reading.direction == "up"


# ---------------------------------------------------------------------------
# the two refusals (both learned from the live dry run)
# ---------------------------------------------------------------------------
def test_a_near_empty_download_leaves_the_file_alone(tmp_path):
    """Rewriting 2 of 787 rows manufactures a boundary; it does not remove one.

    Measured: EA, TMHC, JHG, SATS and AVNS all came back with a near-empty
    history on the live dry run.
    """
    store = tmp_path / "store"
    _store(store, "EAX", _frame(40, splice_at=20))
    thin = _frame(40).iloc[-2:]
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({"EAX": thin}),
        apply=True,
        now=NOW,
    )
    outcome = report.outcomes[0]
    assert outcome.status == "insufficient_coverage"
    assert outcome.rows_rewritten == 0
    assert report.files_skipped_low_coverage == 1
    assert report.files_changed == 0
    assert "covered" in outcome.note
    stored = pd.read_parquet(store / "EAX.parquet")
    assert float(stored["volume"].iloc[-1]) == float(_frame(40, splice_at=20)["volume"].iloc[-1])


def test_a_file_the_run_would_make_worse_is_left_alone(tmp_path):
    """A repair that can make a file worse is not a repair."""
    store = tmp_path / "store"
    _store(store, "WRS", _frame(40))          # clean today
    poisoned = _frame(40)
    poisoned.loc[20:, "volume"] = poisoned.loc[20:, "volume"] / 500.0   # a worse cliff
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({"WRS": poisoned}),
        apply=True,
        now=NOW,
    )
    outcome = report.outcomes[0]
    assert outcome.status == "would_worsen"
    assert report.files_skipped_would_worsen == 1
    assert report.files_changed == 0
    assert not cliff.first_cliff(pd.read_parquet(store / "WRS.parquet")).is_cliff


def test_a_windows_reserved_stem_is_downloaded_under_its_real_symbol(tmp_path):
    """`CON_.parquet` holds CON. Asking Yahoo for "CON_" returns nothing."""
    assert backfill.symbol_for_stem("CON_") == "CON"
    assert backfill.symbol_for_stem("AAPL") == "AAPL"
    store = tmp_path / "store"
    _store(store, "CON_", _frame(40, splice_at=20))
    asked: list[list[str]] = []

    def download(symbols, *, period):
        asked.append(list(symbols))
        return _downloader_for({"CON": _frame(40)})(symbols, period=period)

    report = backfill.run_backfill(
        store_dir=store, frozen_dir=tmp_path / "frozen", downloader=download,
        apply=False, now=NOW,
    )
    assert asked[0] == ["CON"]
    assert report.rows_rewritten == 40


def test_a_symbol_a_batch_dropped_is_retried_on_its_own(tmp_path):
    """A batched download drops the odd ticker; one retry each before giving up."""
    store = tmp_path / "store"
    _store(store, "AAA", _frame(40, splice_at=20))
    _store(store, "BBB", _frame(40, splice_at=20))
    calls: list[list[str]] = []

    def flaky(symbols, *, period):
        calls.append(list(symbols))
        if len(symbols) > 1:
            return _downloader_for({"AAA": _frame(40)})(symbols, period=period)
        return _downloader_for({"AAA": _frame(40), "BBB": _frame(40)})(symbols, period=period)

    report = backfill.run_backfill(
        store_dir=store, frozen_dir=tmp_path / "frozen", downloader=flaky,
        apply=False, now=NOW,
    )
    assert calls[-1] == ["BBB"], "the dropped symbol is asked for individually"
    assert report.symbols_missing == []
    assert report.symbols_downloaded == 2


def test_the_manifest_after_count_reconciles_with_an_independent_scan(tmp_path):
    """A summary that disagrees with the store it summarises is worse than none.

    The first live run reported 44 cliffed-after where a scan of the same store
    found 53: the nine files Yahoo had no data for kept their cliffs and were
    never counted.
    """
    store = tmp_path / "store"
    _store(store, "FIXED", _frame(40, splice_at=20))
    _store(store, "DEAD", _frame(40, splice_at=20))     # yahoo has nothing
    _store(store, "CLEAN", _frame(40))
    report = backfill.run_backfill(
        store_dir=store,
        frozen_dir=tmp_path / "frozen",
        downloader=_downloader_for({"FIXED": _frame(40)}),
        apply=True,
        now=NOW,
    )
    live = cliff.scan_store(store)
    assert report.cliffed_after == live.cliffed == 1
    assert report.unmeasurable_after == live.unmeasurable == 0
