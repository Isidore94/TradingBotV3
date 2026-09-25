"""P2-8 8c - the sector and industry maps carry a sidecar review stamp; Health
warns past 90 days and shows "age unknown" when there is no stamp."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import map_freshness  # noqa: E402
import project_paths  # noqa: E402

NOW = datetime(2026, 9, 25, 9, 0)


def _maps(tmp_path, monkeypatch):
    sector = tmp_path / "sector_etf_map.json"
    industry = tmp_path / "industry_etf_map.json"
    sector.write_text(json.dumps({"technology": "XLK"}), encoding="utf-8")
    industry.write_text(json.dumps({"updated_utc": "x", "yahoo_industryKey_to_ref": {}}), encoding="utf-8")
    monkeypatch.setattr(project_paths, "SECTOR_ETF_MAP_FILE", sector)
    monkeypatch.setattr(project_paths, "INDUSTRY_ETF_MAP_FILE", industry)
    return sector, industry


def test_no_stamp_is_age_unknown(tmp_path, monkeypatch):
    _maps(tmp_path, monkeypatch)
    rows = {row["id"]: row for row in map_freshness.health_checks(NOW)}
    assert rows["map_freshness_sector"]["status"] == "unknown"
    assert "age unknown" in rows["map_freshness_industry"]["summary"]


def test_past_90_days_warns_and_inside_is_healthy(tmp_path, monkeypatch):
    sector, _industry = _maps(tmp_path, monkeypatch)
    before = sector.read_bytes()
    assert map_freshness.main(["stamp", "sector", "--date", "2026-06-01", "--apply"]) == 0
    assert map_freshness.main(["stamp", "industry", "--date", "2026-09-01", "--apply"]) == 0
    rows = {row["id"]: row for row in map_freshness.health_checks(NOW)}
    assert rows["map_freshness_sector"]["status"] == "degraded"
    assert "116 days old" in rows["map_freshness_sector"]["summary"]
    assert rows["map_freshness_industry"]["status"] == "healthy"
    assert sector.read_bytes() == before  # the map itself is never rewritten
    assert (tmp_path / "sector_etf_map.reviewed.json").is_file()


def test_the_stamp_cli_is_dry_by_default(tmp_path, monkeypatch):
    _maps(tmp_path, monkeypatch)
    assert map_freshness.main(["stamp", "sector"]) == 0
    assert not (tmp_path / "sector_etf_map.reviewed.json").exists()


def test_health_shows_the_rows_and_a_stale_map_degrades_the_page(tmp_path, monkeypatch):
    from ui.panels import health_panel

    _maps(tmp_path, monkeypatch)
    map_freshness.main(["stamp", "sector", "--date", "2026-01-01", "--apply"])
    payload = {"status": "healthy", "checks": [], "summary": {"healthy": 0, "total": 0}}

    merged = health_panel._with_map_freshness_checks(payload)

    ids = {row["id"]: row["status"] for row in merged["checks"]}
    assert ids == {"map_freshness_sector": "degraded", "map_freshness_industry": "unknown"}
    assert merged["status"] == "degraded"
    assert merged["summary"]["total"] == 2


def test_an_unknown_age_never_turns_the_page_green_or_red(tmp_path, monkeypatch):
    from ui.panels import health_panel

    _maps(tmp_path, monkeypatch)
    merged = health_panel._with_map_freshness_checks({"status": "healthy", "checks": [], "summary": {}})
    assert merged["status"] == "healthy"
    assert merged["summary"]["unknown"] == 2
