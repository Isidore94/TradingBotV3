"""The deterministic `history_pack` night slot (trader 2026-10-02).

Packs the D1 feature history every night (archive + verify), loads no model,
writes one small JSON report with store sizes now and at the previous report,
keeps the last good report on failure, and NEVER trims by default: seven
readers still read the CSV directly, so trimming stays off until they move to
`read_history`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SLOT = "history_pack"


def _history(path: Path, dates: list[str]) -> None:
    rows = []
    for day in dates:
        for i in range(4):
            rows.append(
                {
                    "feature_history_schema_version": "1",
                    "run_id": f"{day}-150000",
                    "run_date": day,
                    "symbol": f"S{i}",
                    "side": "LONG",
                    "last_close": "1.50",
                    "atr20": "",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def paths(tmp_path):
    csv_path = tmp_path / "runtime" / "d1_features_history.csv"
    _history(csv_path, ["2026-06-01", "2026-06-02", "2026-07-15", "2026-08-20"])
    big = tmp_path / "runtime" / "intraday_bounce_outcomes.csv"
    big.write_text("a,b\n1,2\n", encoding="utf-8")
    ledger_dir = tmp_path / "runtime" / "evidence_ledgers"
    (ledger_dir / "x").mkdir(parents=True)
    (ledger_dir / "x" / "2026-09.jsonl").write_bytes(b'{"a": 1}\n')
    return {
        "csv_path": csv_path,
        "archive_dir": tmp_path / "runtime" / "d1_features_history_archive",
        "report_path": tmp_path / "runtime" / "history_pack_report.json",
        "stores": {
            "d1_features_history.csv": csv_path,
            "intraday_bounce_outcomes.csv": big,
            "evidence_ledgers": ledger_dir,
            "missing.json": tmp_path / "runtime" / "missing.json",
        },
    }


def _run(paths, **kwargs):
    from ai_jobs import history_pack

    return history_pack.run_history_pack(session_date="2026-10-02", **paths, **kwargs)


def test_the_slot_packs_and_writes_one_report(paths):
    result = _run(paths)
    assert result["status"] == "ok", result
    assert result["model"] == ""
    assert result["outputs"] == [str(paths["report_path"])]
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert report["archive"]["archived_rows"] == 16
    assert report["verify"]["ok"] is True
    assert report["trim"]["enabled"] is False
    sizes = {row["name"]: row for row in report["stores"]}
    assert sizes["d1_features_history.csv"]["bytes"] == paths["csv_path"].stat().st_size
    assert sizes["d1_features_history.csv"]["previous_bytes"] is None
    assert sizes["evidence_ledgers"]["bytes"] == len('{"a": 1}\n'.encode())
    assert sizes["missing.json"]["exists"] is False
    assert sizes["d1_features_history_archive"]["bytes"] > 0


def test_the_next_report_carries_the_previous_sizes(paths):
    _run(paths)
    first = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    big = paths["stores"]["intraday_bounce_outcomes.csv"]
    big.write_bytes(big.read_bytes() + b"3,4\n")
    assert _run(paths)["status"] == "ok"
    second = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    before = {row["name"]: row["bytes"] for row in first["stores"]}
    for row in second["stores"]:
        assert row["previous_bytes"] == before[row["name"]], row["name"]
    grown = next(r for r in second["stores"] if r["name"] == "intraday_bounce_outcomes.csv")
    assert grown["delta_bytes"] == 4
    assert second["archive"]["archived_rows"] == 0  # idempotent second night


def test_the_default_night_never_shortens_the_live_file(paths):
    """Every row is months old, so a trim would remove all of them - it must not run."""
    before = paths["csv_path"].read_bytes()
    assert _run(paths)["status"] == "ok"
    assert _run(paths)["status"] == "ok"
    assert paths["csv_path"].read_bytes() == before


def test_the_trim_setting_defaults_off(monkeypatch):
    import project_paths as pp
    from ai_jobs import history_pack

    monkeypatch.setattr(pp, "get_local_setting", lambda key, default=None: default)
    assert history_pack.TRIM_SETTING == "d1_history_trim_enabled"
    assert history_pack.trim_enabled() is False


def test_trimming_runs_only_when_switched_on(paths, monkeypatch):
    import project_paths as pp

    real = pp.get_local_setting
    monkeypatch.setattr(
        pp,
        "get_local_setting",
        lambda key, default=None: True if key == "d1_history_trim_enabled" else real(key, default),
    )
    result = _run(paths)
    assert result["status"] == "ok", result
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert report["trim"]["enabled"] is True and report["trim"]["removed"] == 16
    import d1_feature_history_archive as arc

    back = arc.read_history(csv_path=paths["csv_path"], archive_dir=paths["archive_dir"])
    assert len(back) == 16


def test_a_failed_pack_keeps_the_last_good_report(paths):
    assert _run(paths)["status"] == "ok"
    good = paths["report_path"].read_bytes()
    manifest = json.loads((paths["archive_dir"] / "manifest.json").read_text(encoding="utf-8"))
    target = paths["archive_dir"] / next(iter(manifest["files"].values()))["file"]
    target.write_bytes(target.read_bytes() + b"tamper")
    result = _run(paths)
    assert result["status"] == "failed"
    assert "verify" in result["reason"] or "sha256" in result["reason"]
    assert paths["report_path"].read_bytes() == good


def test_the_slot_sits_in_stage_one_after_the_permutation_report(tmp_path):
    from ai_jobs import runner

    slots = runner.default_slots()
    names = [s.name for s in slots]
    slot = next(s for s in slots if s.name == SLOT)
    assert names.index(SLOT) == names.index("permutation_report") + 1
    assert names.index(SLOT) == names.index("market_story_rollups") - 1
    assert SLOT in [s.name for s in runner._deterministic_stage(slots)]
    assert slot.uses_model is False and slot.goal == "ops" and slot.max_attempts == 3
    for kind in ("weeknight", "saturday", "sunday"):
        slate = runner.slots_for(kind, session_date="2026-09-25", ledger_path=tmp_path / "ledger.jsonl")
        assert SLOT in [s.name for s in slate], kind


def test_the_default_store_list_names_the_big_live_stores():
    import project_paths as pp
    from ai_jobs import history_pack

    stores = history_pack.default_stores()
    assert stores["d1_features_history.csv"] == pp.D1_FEATURES_HISTORY_FILE
    assert stores["master_avwap_setup_tracker.json"] == pp.MASTER_AVWAP_SETUP_TRACKER_FILE
    assert stores["master_avwap_setup_tracker.json.bak"].name == "master_avwap_setup_tracker.json.bak"
    assert stores["master_avwap_setup_tracker.sqlite"] == pp.MASTER_AVWAP_SETUP_TRACKER_DB
    assert stores["intraday_bounce_outcomes.csv"] == pp.INTRADAY_BOUNCE_OUTCOMES_FILE
    assert stores["intraday_bounce_candidates.csv"] == pp.INTRADAY_BOUNCE_CANDIDATES_FILE
    assert stores["evidence_ledgers"].name == "evidence_ledgers"
    assert pp.HISTORY_PACK_REPORT_FILE.parent == pp.RUNTIME_DATA_DIR
