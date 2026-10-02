"""Reviewer advisories on the D1 history archive (2026-10-02).

Each test here was written to kill a mutation the first test file let live:
the trim's cell-for-cell proof, its byte-copy check, the keep-days boundary,
a stale trim temp left by a hard kill, and the key check against the writer's
own widening rewrite of a ticker literally named ``NA``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import d1_feature_history_archive as arc  # noqa: E402

TODAY = date(2026, 10, 2)


def _frame(run_date: str, run_id: str, symbols: list[str], note: str = "plain") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_history_schema_version": "1",
                "run_id": run_id,
                "run_timestamp": f"{run_date}T15:00:00",
                "run_date": run_date,
                "watchlist_label": "home",
                "scoring_config_hash": "abc",
                "scoring_config_updated_at": "2026-07-01T00:00:00",
                "symbol": symbol,
                "side": "LONG",
                "last_close": "10.25",
                "note": note,
            }
            for symbol in symbols
        ]
    )


def _write(path: Path, frames: list[pd.DataFrame]) -> None:
    for frame in frames:
        if not path.exists():
            frame.to_csv(path, index=False)
        else:
            frame.to_csv(path, mode="a", header=False, index=False)


def _ref(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


@pytest.fixture
def store(tmp_path):
    csv_path = tmp_path / "d1_features_history.csv"
    _write(
        csv_path,
        [
            _frame("2026-08-31", "r0831", ["AAA", "BBB"]),
            _frame("2026-09-01", "r0901", ["AAA", "BBB"]),
            _frame("2026-09-02", "r0902", ["AAA", "BBB"]),
            _frame("2026-09-03", "r0903", ["AAA", "BBB"]),
        ],
    )
    return csv_path, tmp_path / "archive"


def _kw(store):
    return {"csv_path": store[0], "archive_dir": store[1]}


def test_thirty_days_on_10_02_drops_09_01_and_keeps_09_02(store):
    arc.archive(**_kw(store))
    result = arc.trim(**_kw(store), today=TODAY, keep_days=30, apply=True)
    assert result["removed"] == 4
    assert sorted(set(_ref(store[0])["run_date"])) == ["2026-09-02", "2026-09-03"]
    assert result["first_kept_run_date"] == "2026-09-02"


def test_trim_refuses_a_live_cell_that_differs_from_the_archive(store):
    """Same keys, different text in a non-key cell: not proven, so not removed."""
    csv_path = store[0]
    arc.archive(**_kw(store))
    frame = _ref(csv_path)
    frame.loc[1, "note"] = "edited after packing"
    frame.to_csv(csv_path, index=False)
    before = csv_path.read_bytes()
    with pytest.raises(arc.VerifyFailed):
        arc.trim(**_kw(store), today=TODAY, apply=True)
    assert csv_path.read_bytes() == before


def test_trim_refuses_when_the_byte_copy_does_not_equal_the_kept_rows(store, monkeypatch):
    csv_path = store[0]
    arc.archive(**_kw(store))
    before = csv_path.read_bytes()
    real = arc._record_offset

    def one_record_short(path, records):
        return real(path, records - 1)

    monkeypatch.setattr(arc, "_record_offset", one_record_short)
    with pytest.raises(arc.VerifyFailed):
        arc.trim(**_kw(store), today=TODAY, apply=True)
    assert csv_path.read_bytes() == before
    assert not [p for p in csv_path.parent.iterdir() if ".trim-tmp-" in p.name]


def test_a_stale_trim_temp_from_a_hard_kill_is_removed_on_the_next_run(store):
    csv_path = store[0]
    arc.archive(**_kw(store))
    stale = csv_path.with_name(csv_path.name + ".trim-tmp-999999")
    stale.write_bytes(b"x" * 1000)
    unrelated = csv_path.with_name(csv_path.name + ".trim-tmp-notapid")
    unrelated.write_bytes(b"keep me")
    other = csv_path.with_name("other.csv.trim-tmp-123")
    other.write_bytes(b"keep me too")
    arc.trim(**_kw(store), today=TODAY)  # a dry run also sweeps, under the lock
    assert not stale.exists()
    assert unrelated.exists() and other.exists()


def test_a_stale_trim_temp_is_also_swept_by_the_nightly_archive(store):
    csv_path = store[0]
    stale = csv_path.with_name(csv_path.name + ".trim-tmp-424242")
    stale.write_bytes(b"x")
    arc.archive(**_kw(store))
    assert not stale.exists()


def test_a_ticker_named_na_survives_the_writers_widening_rewrite(tmp_path, monkeypatch):
    """The writer's widening rewrite turns the ticker `NA` into an empty cell.

    The key check must accept exactly that rewrite (the same equivalence the
    trim proof uses) - otherwise archive and trim refuse every night after it.
    """
    from master_avwap_lib import legacy

    csv_path = tmp_path / "d1_features_history.csv"
    _write(csv_path, [_frame("2026-08-03", "r0803", ["NA", "None", "AAA"])])
    kw = {"csv_path": csv_path, "archive_dir": tmp_path / "archive"}
    original = _ref(csv_path)
    arc.archive(**kw)
    monkeypatch.setattr(legacy, "D1_FEATURE_HISTORY_FILE", csv_path)
    widened = _frame("2026-08-04", "r0804", ["ZZZ"]).drop(
        columns=[
            "feature_history_schema_version", "run_id", "run_timestamp", "run_date",
            "watchlist_label", "scoring_config_hash", "scoring_config_updated_at",
        ]
    ).assign(new_col="x")
    legacy.append_d1_feature_history(widened, {"run_id": "r0804", "run_date": "2026-08-04"})
    live = _ref(csv_path)
    assert live.loc[0, "symbol"] == "" and live.loc[1, "symbol"] == ""  # the rewrite's loss
    assert arc.verify(**kw)["ok"] is True
    assert arc.archive(**kw)["archived_rows"] == 1
    arc.trim(**kw, today=TODAY, apply=True)
    back = arc.read_history(**kw)
    assert back.loc[:2, "symbol"].tolist() == ["NA", "None", "AAA"]  # the archive kept the text
    assert back.loc[:2, "note"].tolist() == original["note"].tolist()


def test_the_key_check_still_refuses_a_different_ticker(store):
    csv_path = store[0]
    arc.archive(**_kw(store))
    frame = _ref(csv_path)
    frame.loc[2, "symbol"] = "CCC"
    frame.to_csv(csv_path, index=False)
    assert arc.verify(**_kw(store))["ok"] is False
    with pytest.raises(arc.VerifyFailed):
        arc.archive(**_kw(store))
