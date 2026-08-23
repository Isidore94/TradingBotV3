"""R10.V step 2 - provenance travels with the row.

The durable daily-bar store held no record of where a row came from or what its
`volume` column meant, which is why the unit splice took a field-level diff of
60,519 mark-days to find. Every row now carries `source` and `volume_unit`, and
the file carries `daily_bars_schema=v2` in its Arrow metadata.

Two rules the tests below pin, because both are easy to get subtly wrong:

* **A row whose origin is unrecorded reads `unknown`, never `cache`.** A v1 file
  read through the cache path knows only that it came off disk; writing `cache`
  as its source would be recording something nobody measured.
* **Every consumer must read v1 AND v2**, so the store can be upgraded file by
  file rather than in one flag day. There is a test per consumer from the cliff
  report's consumer table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402


def _bars(rows: int = 5, *, start: str = "2026-08-03") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=rows)
    return pd.DataFrame(
        {
            "datetime": days,
            "open": [10.0 + i for i in range(rows)],
            "high": [10.5 + i for i in range(rows)],
            "low": [9.5 + i for i in range(rows)],
            "close": [10.2 + i for i in range(rows)],
            "volume": [1_000_000.0 + i for i in range(rows)],
        }
    )


def _write_v1(path: Path, rows: int = 5) -> Path:
    """A file exactly as the store holds it today: six columns, no metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _bars(rows).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------
def test_a_yahoo_frame_is_share_denominated():
    assert master_avwap.daily_bar_provenance_for_source(master_avwap.DAILY_BAR_SOURCE_YAHOO) == (
        "yahoo",
        master_avwap.DAILY_BAR_UNIT_SHARES,
    )


def test_an_ib_frame_is_round_lots_of_the_regular_session():
    """`useRTH=1`, `whatToShow="TRADES"` - the unit that caused all of this."""
    assert master_avwap.daily_bar_provenance_for_source(master_avwap.DAILY_BAR_SOURCE_IBKR) == (
        "ibkr",
        master_avwap.DAILY_BAR_UNIT_LOTS_RTH,
    )


@pytest.mark.parametrize("source", ["cache", "", None, "something-else"])
def test_an_unrecorded_origin_is_unknown_and_not_invented(source):
    """A cache read knows the row came off disk, not what wrote it."""
    assert master_avwap.daily_bar_provenance_for_source(source) == (
        master_avwap.DAILY_BAR_SOURCE_UNKNOWN,
        master_avwap.DAILY_BAR_UNIT_UNKNOWN,
    )


# ---------------------------------------------------------------------------
# normalization keeps it
# ---------------------------------------------------------------------------
def test_normalize_stamps_the_frames_own_source_onto_every_row():
    frame = master_avwap._set_daily_bar_source(_bars(), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    normalized = master_avwap._normalize_daily_bar_frame(frame)
    assert list(normalized.columns) == master_avwap.DAILY_BAR_STORE_COLUMNS
    assert set(normalized["source"]) == {"yahoo"}
    assert set(normalized["volume_unit"]) == {"shares"}


def test_normalize_never_relabels_rows_that_already_carry_provenance():
    """A merged frame holds rows from two sources; normalize must not flatten it."""
    frame = _bars(4)
    frame["source"] = ["yahoo", "yahoo", "ibkr", "ibkr"]
    frame["volume_unit"] = ["shares", "shares", "lots_rth", "lots_rth"]
    frame = master_avwap._set_daily_bar_source(frame, master_avwap.DAILY_BAR_SOURCE_CACHE)
    normalized = master_avwap._normalize_daily_bar_frame(frame)
    assert list(normalized["source"]) == ["yahoo", "yahoo", "ibkr", "ibkr"]
    assert list(normalized["volume_unit"]) == ["shares", "shares", "lots_rth", "lots_rth"]


def test_a_v1_row_normalizes_to_unknown_rather_than_to_the_reading_path():
    frame = master_avwap._set_daily_bar_source(_bars(), master_avwap.DAILY_BAR_SOURCE_CACHE)
    normalized = master_avwap._normalize_daily_bar_frame(frame)
    assert set(normalized["source"]) == {"unknown"}
    assert set(normalized["volume_unit"]) == {"unknown"}


def test_a_blank_provenance_cell_is_filled_not_left_as_nan():
    frame = _bars(3)
    frame["source"] = ["yahoo", None, ""]
    frame["volume_unit"] = ["shares", None, ""]
    frame = master_avwap._set_daily_bar_source(frame, master_avwap.DAILY_BAR_SOURCE_IBKR)
    normalized = master_avwap._normalize_daily_bar_frame(frame)
    assert list(normalized["source"]) == ["yahoo", "ibkr", "ibkr"]
    assert list(normalized["volume_unit"]) == ["shares", "lots_rth", "lots_rth"]


def test_an_empty_frame_still_declares_the_v2_columns():
    empty = master_avwap._empty_daily_bar_frame(source=master_avwap.DAILY_BAR_SOURCE_YAHOO)
    assert list(empty.columns) == master_avwap.DAILY_BAR_STORE_COLUMNS


def test_provenance_survives_a_merge_of_two_sources():
    older = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(
            _bars(3, start="2026-08-03"), master_avwap.DAILY_BAR_SOURCE_YAHOO
        )
    )
    newer = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(
            _bars(2, start="2026-08-06"), master_avwap.DAILY_BAR_SOURCE_IBKR
        )
    )
    merged = master_avwap._merge_daily_bar_frames(older, newer)
    by_source = dict(zip(merged["datetime"].dt.strftime("%Y-%m-%d"), merged["source"]))
    assert by_source["2026-08-03"] == "yahoo"
    assert by_source["2026-08-06"] == "ibkr"


# ---------------------------------------------------------------------------
# the file says which schema it is
# ---------------------------------------------------------------------------
def test_a_written_file_declares_v2_in_its_arrow_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    frame = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(_bars(), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    )
    master_avwap._persist_durable_daily_bars("ZZZ", frame)
    path = tmp_path / "ZZZ.parquet"
    assert path.exists()
    assert master_avwap.daily_bars_schema_version(path) == "v2"


def test_an_existing_file_reads_as_v1(tmp_path):
    path = _write_v1(tmp_path / "OLD.parquet")
    assert master_avwap.daily_bars_schema_version(path) == "v1"


def test_an_unreadable_file_is_unknown_rather_than_assumed_current(tmp_path):
    path = tmp_path / "BROKEN.parquet"
    path.write_bytes(b"not a parquet file")
    assert master_avwap.daily_bars_schema_version(path) == "unknown"


def test_the_written_file_round_trips_its_columns_and_values(tmp_path, monkeypatch):
    """Fidelity is asserted on a Yahoo frame; an IB frame's volume is
    deliberately blanked on the way out (R10.V step 3,
    `test_daily_bar_volume_policy.py`), so it is the wrong subject for a
    round-trip test."""
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    frame = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(_bars(), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    )
    master_avwap._persist_durable_daily_bars("RT", frame)
    read_back = pd.read_parquet(tmp_path / "RT.parquet")
    assert list(read_back.columns) == master_avwap.DAILY_BAR_STORE_COLUMNS
    assert set(read_back["volume_unit"]) == {"shares"}
    pd.testing.assert_series_equal(
        read_back["volume"].reset_index(drop=True),
        frame["volume"].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# every consumer reads v1 and v2 (cliff report section 4 table)
# ---------------------------------------------------------------------------
def _write_v2(tmp_path: Path, stem: str, rows: int = 5) -> Path:
    frame = master_avwap._normalize_daily_bar_frame(
        master_avwap._set_daily_bar_source(_bars(rows), master_avwap.DAILY_BAR_SOURCE_YAHOO)
    )
    path = tmp_path / f"{stem}.parquet"
    master_avwap._write_daily_bar_parquet(path, frame)
    return path


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_consumer_master_avwap_durable_loader(tmp_path, monkeypatch, schema):
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    if schema == "v1":
        _write_v1(tmp_path / "CON1.parquet")
        stem = "CON1"
    else:
        _write_v2(tmp_path, "CON2")
        stem = "CON2"
    frame = master_avwap._load_durable_daily_bar_frame(stem)
    assert len(frame) == 5
    assert list(frame.columns) == master_avwap.DAILY_BAR_STORE_COLUMNS
    expected = "unknown" if schema == "v1" else "yahoo"
    assert set(frame["source"]) == {expected}


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_consumer_chart_snapshot_load_d1_bars(tmp_path, monkeypatch, schema):
    import chart_snapshot
    import setup_playbook_study

    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    monkeypatch.setattr(setup_playbook_study, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    monkeypatch.setattr(setup_playbook_study, "MIN_BARS_REQUIRED", 1, raising=False)
    chart_snapshot._daily_bars_cache.clear()
    stem = "CS1" if schema == "v1" else "CS2"
    if schema == "v1":
        _write_v1(tmp_path / f"{stem}.parquet")
    else:
        _write_v2(tmp_path, stem)
    bars = chart_snapshot.load_d1_bars(stem)
    assert len(bars) == 5
    assert bars[0]["close"] > 0
    assert bars[0]["volume"] > 0


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_consumer_human_focus_tracking(tmp_path, schema):
    import human_focus_tracking

    stem = "HF1" if schema == "v1" else "HF2"
    if schema == "v1":
        _write_v1(tmp_path / f"{stem}.parquet")
    else:
        _write_v2(tmp_path, stem)
    frame = human_focus_tracking._normalize_daily_frame(
        human_focus_tracking._load_durable_daily_frame(stem, tmp_path)
    )
    assert list(frame.columns) == ["datetime", "close"]
    assert len(frame) == 5


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_consumer_setup_playbook_study(tmp_path, monkeypatch, schema):
    import setup_playbook_study

    monkeypatch.setattr(setup_playbook_study, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    monkeypatch.setattr(setup_playbook_study, "MIN_BARS_REQUIRED", 1, raising=False)
    stem = "PB1" if schema == "v1" else "PB2"
    if schema == "v1":
        _write_v1(tmp_path / f"{stem}.parquet")
    else:
        _write_v2(tmp_path, stem)
    frame = setup_playbook_study._load_daily_frame(stem)
    assert frame is not None and len(frame) == 5


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_consumer_veto_cohort_grading_reads_both(tmp_path, schema):
    """`ai_jobs/cohorts.py` grades forward off this store, through human_focus."""
    import human_focus_tracking

    stem = "VC1" if schema == "v1" else "VC2"
    if schema == "v1":
        _write_v1(tmp_path / f"{stem}.parquet")
    else:
        _write_v2(tmp_path, stem)
    frame = human_focus_tracking._frame_for_symbol(stem, None, tmp_path)
    assert len(frame) == 5
    assert list(frame.columns) == ["datetime", "close"]


# ---------------------------------------------------------------------------
# the fetch paths stamp the rows, not just the frame
# ---------------------------------------------------------------------------
def test_the_yahoo_fetch_stamps_every_row_it_returns(monkeypatch):
    """A frame-level attr set AFTER normalization leaves the rows saying unknown.

    That was the first version of this change, and the store would have gained
    two columns of `unknown` on every fresh Yahoo fetch - provenance in name
    only. The source is declared before normalization instead.
    """
    raw = _bars()
    raw = raw.rename(columns={"datetime": "Date"}).set_index("Date")
    raw.columns = ["Open", "High", "Low", "Close", "Volume"]
    monkeypatch.setattr(master_avwap.yf, "download", lambda *a, **k: raw)
    result = master_avwap.fetch_daily_bars_from_yahoo("ANY", 30)
    assert set(result["source"]) == {"yahoo"}
    assert set(result["volume_unit"]) == {"shares"}
    assert master_avwap._get_daily_bar_source(result) == "yahoo"


def test_reading_a_v2_file_through_the_cache_path_does_not_relabel_it(tmp_path, monkeypatch):
    """`_set_daily_bar_source(..., cache)` must not overwrite a known origin."""
    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    _write_v2(tmp_path, "KEEP")
    frame = master_avwap._load_durable_daily_bar_frame("KEEP")
    assert set(frame["source"]) == {"yahoo"}, "a cache read must not claim authorship"


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_consumer_ui_bar_cache(tmp_path, monkeypatch, schema):
    """The desk's chart series builder ignores the extra columns, both ways."""
    import chart_snapshot
    from ui.services.bar_cache import BarSeries

    monkeypatch.setattr(master_avwap, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    stem = "UI1" if schema == "v1" else "UI2"
    if schema == "v1":
        path = _write_v1(tmp_path / f"{stem}.parquet")
    else:
        path = _write_v2(tmp_path, stem)
    assert chart_snapshot._daily_store_candidates(stem)[0][1] == path
    series = BarSeries.from_frame(stem, pd.read_parquet(path))
    assert len(series.dt) == 5
    assert float(series.close[-1]) > 0


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_consumer_research_warehouse_ingestion(tmp_path, schema):
    """The warehouse reads the same store; extra columns must not break it.

    Its `provider="UNKNOWN"` docstring is now understated - v2 rows DO carry a
    source - but wiring that through is a warehouse change this packet does not
    authorize, so it is recorded as owed rather than done here.
    """
    from research_warehouse.ingest_existing import read_durable_daily_bars

    stem = "RW1" if schema == "v1" else "RW2"
    if schema == "v1":
        _write_v1(tmp_path / f"{stem}.parquet")
    else:
        _write_v2(tmp_path, stem)
    frame = read_durable_daily_bars(stem, tmp_path)
    assert frame is not None and len(frame) == 5
    assert {"datetime", "open", "high", "low", "close", "volume"} <= set(frame.columns)
