"""The `history_pack` night slot over every registered store (2026-10-02).

One result block per store; one store failing never stops another packing;
a no-lock store is never trimmed whatever its setting says.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import d1_feature_history_archive as arc  # noqa: E402


@pytest.fixture
def specs(tmp_path):
    d1_csv = tmp_path / "d1.csv"
    pd.DataFrame(
        [{"run_id": f"r{i}", "run_date": "2026-06-01", "symbol": f"S{i}", "side": "LONG"} for i in range(5)]
    ).to_csv(d1_csv, index=False)
    out_csv = tmp_path / "outcomes.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "event_type", "logged_at", "trade_date", "x"])
        writer.writeheader()
        for i in range(7):
            writer.writerow({"event_id": f"E{i}", "event_type": "final", "logged_at": f"t{i}",
                             "trade_date": "2026-06-02", "x": '{"a": 1}'})
    stores = arc.registered_stores()
    return [
        stores["d1_features_history"].with_paths(d1_csv, tmp_path / "d1_arc"),
        stores["intraday_bounce_outcomes"].with_paths(out_csv, tmp_path / "out_arc"),
    ]


def _run(specs, tmp_path, **kwargs):
    from ai_jobs import history_pack

    return history_pack.run_history_pack(
        session_date="2026-10-02", specs=specs, report_path=tmp_path / "report.json", stores={}, **kwargs
    )


def test_one_block_per_store(specs, tmp_path):
    result = _run(specs, tmp_path)
    assert result["status"] == "ok", result
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    blocks = report["stores_packed"]
    assert list(blocks) == ["d1_features_history", "intraday_bounce_outcomes"]
    assert blocks["d1_features_history"]["archive"]["archived_rows"] == 5
    assert blocks["intraday_bounce_outcomes"]["archive"]["archived_rows"] == 7
    assert all(block["verify"]["ok"] for block in blocks.values())
    assert blocks["intraday_bounce_outcomes"]["trim"]["enabled"] is False
    assert "no lock" in blocks["intraday_bounce_outcomes"]["trim"]["reason"]
    sizes = {row["name"] for row in report["stores"]}
    assert {"d1_features_history_archive", "intraday_bounce_outcomes_archive"} <= sizes


def test_a_no_lock_store_is_never_trimmed_even_when_its_setting_is_on(specs, tmp_path, monkeypatch):
    import project_paths as pp

    monkeypatch.setattr(pp, "get_local_setting", lambda key, default=None: True)
    bounce = arc.StoreSpec(**{**specs[1].__dict__, "trim_setting": "anything_on"})
    before = bounce.csv_path.read_bytes()
    result = _run([specs[0], bounce], tmp_path)
    assert result["status"] == "ok", result
    assert bounce.csv_path.read_bytes() == before
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["stores_packed"]["intraday_bounce_outcomes"]["trim"]["enabled"] is False
    assert report["stores_packed"]["d1_features_history"]["trim"]["removed"] == 5


def test_one_failing_store_does_not_stop_the_other_and_keeps_the_report(specs, tmp_path):
    assert _run(specs, tmp_path)["status"] == "ok"
    good = (tmp_path / "report.json").read_bytes()
    manifest = json.loads((specs[0].archive_dir / "manifest.json").read_text(encoding="utf-8"))
    target = specs[0].archive_dir / next(iter(manifest["files"].values()))["file"]
    target.write_bytes(target.read_bytes() + b"tamper")
    with specs[1].csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["E9", "final", "t9", "2026-06-03", "{}"])
    result = _run(specs, tmp_path)
    assert result["status"] == "failed"
    assert "d1_features_history" in result["reason"]
    assert (tmp_path / "report.json").read_bytes() == good
    assert arc.load_manifest(specs[1].archive_dir)["archived_rows"] == 8  # the other store still packed


def test_the_default_slate_packs_every_registered_store(monkeypatch, tmp_path):
    from ai_jobs import history_pack

    seen = []
    monkeypatch.setattr(history_pack, "_pack_one", lambda spec, **k: seen.append(spec.name) or {
        "archive": {}, "verify": {"ok": True, "problems": []}, "trim": {"enabled": False}})
    history_pack.run_history_pack(report_path=tmp_path / "r.json", stores={})
    assert seen == list(arc.registered_stores())
