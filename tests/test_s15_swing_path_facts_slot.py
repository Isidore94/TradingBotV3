"""S15: the deterministic `swing_path_facts` night slot - writes, keeps the last file, refuses 0 rows."""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
import swing_path_facts as spf  # noqa: E402

SCAN = date(2026, 9, 1)


def _inputs(tmp_path: Path, *, rows: int = 1):
    import pandas as pd

    bars_dir = tmp_path / "daily_bars"
    bars_dir.mkdir()
    days, cursor = [SCAN], SCAN
    for _ in range(25):
        cursor = market_calendar.next_session(cursor)
        days.append(cursor)
    pd.DataFrame({"datetime": pd.to_datetime(days), "open": 100.0, "high": 101.0, "low": 99.0,
                  "close": 100.0}).to_parquet(bars_dir / "AAA.parquet")
    horizons = tmp_path / "h.csv"
    fields = ["outcome_kind", "scan_row_id", "symbol", "side", "scan_date", "entry_close",
              "setup_family", "horizon_sessions"]
    with horizons.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for _ in range(rows):
            writer.writerow({"outcome_kind": spf.SOURCE_OUTCOME_KIND, "scan_row_id": "AAA:2026-09-01:r1",
                             "symbol": "AAA", "side": "LONG", "scan_date": SCAN.isoformat(),
                             "entry_close": "100", "setup_family": "avwap_breakout",
                             "horizon_sessions": "5"})
    features = tmp_path / "f.csv"
    pd.DataFrame([{"run_id": "r1", "run_date": SCAN.isoformat(), "last_trade_date": SCAN.isoformat(),
                   "symbol": "AAA", "atr20": "2.0", "pct_from_current_vwap": "1", "sector": "Energy"}]
                 ).to_csv(features, index=False)
    return {"horizon_path": horizons, "features_path": features, "bars_dir": bars_dir,
            "out_path": tmp_path / "out" / "swing_path_facts.csv", "session_date": "2026-10-09"}


def test_slot_writes_every_horizon_including_20(tmp_path):
    from ai_jobs import swing_path_facts_night as slot

    kwargs = _inputs(tmp_path)
    result = slot.run_swing_path_facts(**kwargs)
    assert result["status"] == "ok", result
    assert result["outputs"] == [str(kwargs["out_path"])]
    with kwargs["out_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["horizon_sessions"] for r in rows] == ["1", "3", "5", "10", "20"]
    assert all(r["measured"] == "True" for r in rows)
    assert float(rows[-1]["mfe_atr"]) == pytest.approx(0.5)
    assert not list(kwargs["out_path"].parent.glob("*.tmp"))


def test_a_failed_build_keeps_the_last_good_file(tmp_path):
    from ai_jobs import swing_path_facts_night as slot

    kwargs = _inputs(tmp_path)
    kwargs["out_path"].parent.mkdir(parents=True)
    kwargs["out_path"].write_text("last good\n", encoding="utf-8")
    kwargs["features_path"] = tmp_path / "missing.csv"
    result = slot.run_swing_path_facts(**kwargs)
    assert result["status"] == "failed" and result["outputs"] == []
    assert kwargs["out_path"].read_text(encoding="utf-8") == "last good\n"


def test_a_failed_write_keeps_the_last_good_file(tmp_path, monkeypatch):
    from ai_jobs import swing_path_facts_night as slot

    kwargs = _inputs(tmp_path)
    kwargs["out_path"].parent.mkdir(parents=True)
    kwargs["out_path"].write_text("last good\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(spf.os, "replace", boom)
    result = slot.run_swing_path_facts(**kwargs)
    assert result["status"] == "failed"
    assert kwargs["out_path"].read_text(encoding="utf-8") == "last good\n"
    assert not list(kwargs["out_path"].parent.glob("*.tmp"))


def test_zero_rows_writes_nothing(tmp_path):
    from ai_jobs import swing_path_facts_night as slot

    kwargs = _inputs(tmp_path, rows=0)
    kwargs["out_path"].parent.mkdir(parents=True)
    kwargs["out_path"].write_text("last good\n", encoding="utf-8")
    result = slot.run_swing_path_facts(**kwargs)
    assert result["status"] == "ok" and result["outputs"] == []
    assert kwargs["out_path"].read_text(encoding="utf-8") == "last good\n"


def test_slot_closes_stage_one_after_family_side_evidence(tmp_path):
    from ai_jobs import runner

    for kind in ("weeknight", "saturday", "sunday"):
        slate = runner.slots_for(kind, session_date="2026-09-25", ledger_path=tmp_path / "ledger.jsonl")
        assert "swing_path_facts" in [s.name for s in slate], kind

    slots = runner.default_slots()
    names = [slot.name for slot in slots]
    slot = next(s for s in slots if s.name == "swing_path_facts")
    assert names.index("swing_path_facts") == names.index("family_side_evidence") + 1
    assert runner._STAGE_ONE_LAST_SLOT == "swing_path_facts"
    assert slot.goal == "setup_quality" and slot.max_attempts == 3 and slot.uses_model is False


def test_path_constant_sits_beside_the_horizons_file():
    import project_paths as pp

    assert pp.SWING_PATH_FACTS_FILE.parent == pp.MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE.parent
    assert pp.SWING_PATH_FACTS_FILE.name == "swing_path_facts.csv"
