"""The D1 scan's "output/scan-factors" and "output/tier-tracker" steps use far
less memory, and every file they write is BYTE-IDENTICAL to before.

The golden `scan_factor_export_memory_v1` was frozen from the code BEFORE the
memory change (main at 3168c0b6). Its input is 2,684 raw lines copied verbatim
from a copy of the live `d1_features_history.csv` (all 300 columns): 20 whole
symbols, both sides, 532 symbol-days scanned more than once, and old rows with
the newer columns empty. The expected section is the sha256 of every file the
two exports write - the scan-factor observations and leaderboard, the four tier
files and the session-horizon file - and the counts they return to runner.py.

Nothing is masked: the wall clock (`generated_at`), the last completed session,
the cached SPY closes and the caller's daily closes are all FROZEN here, so a
byte that moves is a byte the code moved.
"""

from __future__ import annotations

import gzip
import hashlib
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
from master_avwap_lib import legacy  # noqa: E402

GOLDEN = "scan_factor_export_memory_v1"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / GOLDEN
HISTORY_GZ = FIXTURE_DIR / "history.csv.gz"

_REAL_DATETIME = datetime
FROZEN_NOW = _REAL_DATETIME(2026, 10, 2, 16, 30, 0)
FROZEN_LAST_COMPLETED_SESSION = date(2026, 10, 2)

#: Every file the two exports write, by the name the golden records it under.
OUTPUT_FILES = (
    "scan_factor_observations",
    "scan_factor_leaderboard",
    "tier_list",
    "tier_outcomes",
    "tier_performance",
    "tier_catch_rate",
    "session_horizon_outcomes",
)


class _FrozenMeta(type):
    def __instancecheck__(cls, obj):  # a real datetime is still a datetime
        return isinstance(obj, _REAL_DATETIME)


class _FrozenDatetime(_REAL_DATETIME, metaclass=_FrozenMeta):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW if tz is None else FROZEN_NOW.replace(tzinfo=tz)


def _history_csv(tmp_path: Path) -> Path:
    path = tmp_path / "d1_features_history.csv"
    path.write_bytes(gzip.decompress(HISTORY_GZ.read_bytes()))
    return path


def _frozen_spy_closes(history_path: Path) -> dict[str, float]:
    dates = sorted(
        pd.read_csv(history_path, usecols=["last_trade_date"])["last_trade_date"].dropna().unique()
    )
    return {str(day): round(400.0 + (index % 7) * 1.25 - index * 0.1, 2) for index, day in enumerate(dates)}


def _frozen_closes_for(history_path: Path):
    """The caller's completed daily bars: half the symbols have them, half do not."""
    frame = pd.read_csv(history_path, usecols=["symbol", "last_trade_date", "last_close"])
    symbols = sorted(frame["symbol"].dropna().astype(str).unique())
    with_bars = set(symbols[::2])
    closes: dict[str, dict[date, float]] = {}
    for symbol, day, close in zip(frame["symbol"], frame["last_trade_date"], frame["last_close"], strict=False):
        if str(symbol) in with_bars and pd.notna(close):
            closes.setdefault(str(symbol), {})[date.fromisoformat(str(day)[:10])] = float(close)

    def closes_for(symbol):
        return closes.get(str(symbol))

    return closes_for


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    history_path = _history_csv(tmp_path)
    spy = _frozen_spy_closes(history_path)
    monkeypatch.setattr(legacy, "datetime", _FrozenDatetime)
    monkeypatch.setattr(legacy, "_cached_spy_closes", lambda: dict(spy))
    monkeypatch.setattr(
        market_calendar, "last_completed_session", lambda now: FROZEN_LAST_COMPLETED_SESSION
    )
    return history_path


def _paths(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {name: out_dir / f"{name}.csv" for name in OUTPUT_FILES}


def _counts(result: dict) -> dict:
    return {key: value for key, value in sorted(result.items()) if not key.endswith("_path")}


def run_runner_sequence(history_path: Path, out_dir: Path) -> tuple[dict, dict, dict]:
    """The exact call sequence of `runner.py` (output/scan-factors, then output/tier-tracker)."""
    paths = _paths(out_dir)
    scan_result = legacy.export_scan_factor_views(
        history_path,
        paths["scan_factor_observations"],
        paths["scan_factor_leaderboard"],
        include_data=True,
    )
    shared_history_df = scan_result.pop("_history_df", None)
    shared_observation_rows = scan_result.pop("_observation_rows", None)
    shared_leaderboard_rows = scan_result.pop("_leaderboard_rows", None)
    tier_result = legacy.export_bot_tier_tracker_views(
        history_path,
        paths["tier_list"],
        paths["tier_outcomes"],
        paths["tier_performance"],
        paths["tier_catch_rate"],
        history_df=shared_history_df,
        observation_rows=shared_observation_rows,
        leaderboard_rows=shared_leaderboard_rows,
        closes_for=_frozen_closes_for(history_path),
        session_horizon_path=paths["session_horizon_outcomes"],
    )
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    return hashes, _counts(scan_result), _counts(tier_result)


def run_standalone_tier_export(history_path: Path, out_dir: Path) -> tuple[dict, dict]:
    """`export_bot_tier_tracker_views` reading the history file itself."""
    paths = _paths(out_dir)
    tier_result = legacy.export_bot_tier_tracker_views(
        history_path,
        paths["tier_list"],
        paths["tier_outcomes"],
        paths["tier_performance"],
        paths["tier_catch_rate"],
        closes_for=_frozen_closes_for(history_path),
        session_horizon_path=paths["session_horizon_outcomes"],
    )
    names = ("tier_list", "tier_outcomes", "tier_performance", "tier_catch_rate", "session_horizon_outcomes")
    hashes = {name: hashlib.sha256(paths[name].read_bytes()).hexdigest() for name in names}
    return hashes, _counts(tier_result)


def test_fixture_input_is_the_declared_file():
    from conftest import load_fixture_contract

    contract = load_fixture_contract(GOLDEN)
    declared = contract["history_input"]
    assert hashlib.sha256(HISTORY_GZ.read_bytes()).hexdigest() == declared["history_csv_gz_sha256"]
    raw = gzip.decompress(HISTORY_GZ.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == declared["history_csv_sha256"]
    assert raw.count(b"\n") == declared["line_count"]


def test_runner_sequence_outputs_are_byte_identical_to_the_pre_change_golden(frozen, tmp_path):
    from conftest import load_fixture_contract

    contract = load_fixture_contract(GOLDEN)
    expected = contract["expected_outputs"]
    hashes, scan_counts, tier_counts = run_runner_sequence(frozen, tmp_path / "runner")
    assert scan_counts == expected["scan_factor_counts"]
    assert tier_counts == expected["tier_tracker_counts"]
    for name in OUTPUT_FILES:
        assert hashes[name] == expected["sha256"][name], f"{name} changed"


def test_standalone_tier_export_is_byte_identical_to_the_pre_change_golden(frozen, tmp_path):
    from conftest import load_fixture_contract

    contract = load_fixture_contract(GOLDEN)
    expected = contract["expected_outputs"]
    hashes, tier_counts = run_standalone_tier_export(frozen, tmp_path / "standalone")
    assert tier_counts == expected["tier_tracker_counts"]
    for name, digest in hashes.items():
        assert digest == expected["sha256"][name], f"{name} changed"


def test_exports_match_the_builders_fed_the_full_width_history(frozen, tmp_path):
    """Whatever the exports read, they must agree with the builders given ALL 300 columns.

    This is the guard on any column narrowing: a reader that starts using a
    field the export no longer loads makes the two disagree here.
    """
    hashes, _, _ = run_runner_sequence(frozen, tmp_path / "runner")
    full = pd.read_csv(frozen, low_memory=False)
    observations = legacy.build_scan_factor_observation_rows(full)
    leaderboard = legacy.build_scan_factor_leaderboard_rows(full, observations)
    picks = legacy.build_bot_tier_pick_rows(full, leaderboard)
    outcomes = legacy.build_bot_tier_outcome_rows(full, observations, leaderboard)
    catch_rate = legacy.build_bot_tier_catch_rate_rows(full, observations, leaderboard)
    from master_avwap_lib.session_horizon_outcomes import (
        SESSION_HORIZON_OUTCOME_COLUMNS,
        build_session_horizon_observation_rows,
    )

    session = build_session_horizon_observation_rows(
        full, _frozen_closes_for(frozen), last_completed_session=FROZEN_LAST_COMPLETED_SESSION
    ).rows
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    written = {
        "scan_factor_observations": (observations, legacy.SCAN_FACTOR_OBSERVATION_COLUMNS),
        "scan_factor_leaderboard": (leaderboard, legacy.SCAN_FACTOR_LEADERBOARD_COLUMNS),
        "tier_list": (picks, legacy.TIER_LIST_COLUMNS),
        "tier_outcomes": (outcomes, legacy.TIER_OUTCOME_COLUMNS),
        "tier_catch_rate": (catch_rate, legacy.TIER_CATCH_RATE_COLUMNS),
        "session_horizon_outcomes": (session, SESSION_HORIZON_OUTCOME_COLUMNS),
    }
    for name, (rows, columns) in written.items():
        path = direct_dir / f"{name}.csv"
        legacy._write_scan_factor_csv(path, rows, columns)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name], name


# ---------------------------------------------------------------------------
# The improvement, pinned without a megabyte number.
# ---------------------------------------------------------------------------


def test_leaderboard_builds_no_frame_from_one_dict_per_observation_factor_pair(frozen, monkeypatch):
    """The spike's largest part: one merged dict per (observation, factor) pair.

    On the live history that is millions of 30-key dicts. The builder may box
    the observations themselves, never a dict per pair.
    """
    history_df = pd.read_csv(frozen, low_memory=False)
    observations = legacy.build_scan_factor_observation_rows(history_df)
    list_of_dict_sizes: list[int] = []
    real_init = pd.DataFrame.__init__

    def recording_init(self, data=None, *args, **kwargs):
        if isinstance(data, list) and data and isinstance(data[0], dict):
            list_of_dict_sizes.append(len(data))
        real_init(self, data, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "__init__", recording_init)
    rows = legacy.build_scan_factor_leaderboard_rows(history_df, observations)
    assert rows
    assert max(list_of_dict_sizes) <= len(observations), (
        f"a frame was built from {max(list_of_dict_sizes)} dicts for {len(observations)} observations"
    )


def _spy_on_exports(monkeypatch):
    calls = {"prepare": 0, "to_dict_records": []}
    real_prepare = legacy._prepare_scan_factor_history_frame

    def counting_prepare(*args, **kwargs):
        calls["prepare"] += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(legacy, "_prepare_scan_factor_history_frame", counting_prepare)
    real_to_dict = pd.DataFrame.to_dict

    def recording_to_dict(self, orient="dict", *args, **kwargs):
        if orient == "records":
            calls["to_dict_records"].append(self.shape)
        return real_to_dict(self, orient, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_dict", recording_to_dict)
    return calls


def test_each_export_prepares_the_history_once(frozen, tmp_path, monkeypatch):
    calls = _spy_on_exports(monkeypatch)
    paths = _paths(tmp_path / "once")
    legacy.export_scan_factor_views(
        frozen, paths["scan_factor_observations"], paths["scan_factor_leaderboard"]
    )
    assert calls["prepare"] == 1, "export_scan_factor_views prepared the history more than once"
    calls["prepare"] = 0
    legacy.export_bot_tier_tracker_views(
        frozen,
        paths["tier_list"],
        paths["tier_outcomes"],
        paths["tier_performance"],
        paths["tier_catch_rate"],
        session_horizon_path=paths["session_horizon_outcomes"],
    )
    assert calls["prepare"] == 1, "export_bot_tier_tracker_views prepared the history more than once"


def test_no_full_width_history_is_boxed_or_kept(frozen, tmp_path, monkeypatch):
    width = len(pd.read_csv(frozen, nrows=0).columns)
    assert width == 300
    calls = _spy_on_exports(monkeypatch)
    hashes, _, _ = run_runner_sequence(frozen, tmp_path / "runner")
    assert hashes  # the sequence ran
    wide = [shape for shape in calls["to_dict_records"] if shape[1] >= width]
    assert not wide, f"to_dict('records') boxed a full-width history frame: {wide}"

    result = legacy.export_scan_factor_views(
        frozen,
        tmp_path / "again_obs.csv",
        tmp_path / "again_lb.csv",
        include_data=True,
    )
    kept = result["_history_df"]
    assert kept.shape[1] < width / 3, (
        f"the history handed to the tier step keeps {kept.shape[1]} of {width} columns"
    )
