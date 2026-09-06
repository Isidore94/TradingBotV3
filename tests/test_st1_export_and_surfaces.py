"""Packet ST1, builder's half: the export seam and the three describe() lines.

The tester's `tests/test_st1_outcome_clock.py` pins the BUILDERS and the reader.
These pin what the packet asks for around them:

* the v2 file is written in the SAME export pass that writes the tier outcomes,
  from the caller's own completed bars, and a failure there never costs a v1
  export (ST1 item 2);
* every surface that shows the family rate also says WHAT it is - the Master
  AVWAP setups panel, the setup docs and the AWAY digest (ST1 item 3).

Nothing here opens a live store, fetches, or writes outside `tmp_path`.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from master_avwap_lib import legacy  # noqa: E402

JUNE_SESSIONS = (
    "2026-06-01",
    "2026-06-02",
    "2026-06-03",
    "2026-06-04",
    "2026-06-05",
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
)


def _scan_row(symbol: str, scan_date: str, close: float) -> dict:
    return {
        "run_id": f"run-{scan_date}",
        "run_timestamp": f"{scan_date}T13:00:00",
        "run_date": scan_date,
        "watchlist_label": "swing_longs",
        "symbol": symbol,
        "side": "LONG",
        "last_trade_date": scan_date,
        "last_close": float(close),
        "priority_bucket": "favorite_setup",
        "priority_score": 120.0,
        "setup_family": "avwap_breakout",
        "favorite_zone": "AVWAPE to UPPER_1",
        "current_band_zone": "AVWAPE to UPPER_1",
        "trend_20d": "UP",
        "assigned_tier": "S",
    }


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _scan_row("AAA", day, 100.0 + index)
            for index, day in enumerate(JUNE_SESSIONS)
        ]
    )


def _export_paths(tmp_path: Path) -> dict:
    return {
        "tier_list_path": tmp_path / "tier_list.csv",
        "tier_outcomes_path": tmp_path / "tier_outcomes.csv",
        "tier_performance_path": tmp_path / "tier_performance.csv",
        "tier_catch_rate_path": tmp_path / "tier_catch.csv",
        "session_horizon_path": tmp_path / "session_horizon.csv",
    }


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_the_export_writes_the_v2_file_beside_the_tier_outcomes(tmp_path, monkeypatch):
    """One pass, five files, and the v2 rows carry their own clock.

    `sessions_spanned == horizon_sessions` on every measured row is the whole
    point of the file: v1's column is whatever the symbol's scan membership made
    it, and v2's is the declared horizon by construction.
    """
    import market_calendar

    paths = _export_paths(tmp_path)
    closes = {
        date.fromisoformat(day): 100.0 + index * 2.0
        for index, day in enumerate(JUNE_SESSIONS)
    }
    monkeypatch.setattr(market_calendar, "last_completed_session", lambda now: date(2026, 6, 30))

    result = legacy.export_bot_tier_tracker_views(
        history_df=_history(),
        closes_for=lambda symbol: closes,
        **paths,
    )

    assert paths["session_horizon_path"].exists()
    rows = _read(paths["session_horizon_path"])
    assert rows, "the v2 export wrote no rows"
    assert result["session_horizon_outcome_count"] == len(rows)
    assert result["session_horizon_measured_count"] >= 1

    for row in rows:
        assert row["outcome_kind"] == "favorable_direction_session_v2"
        assert row["knowledge_basis"] == "entry_session_close_to_target_session_close"
        if str(row["measured"]).strip().lower() == "true":
            assert row["sessions_spanned"] == row["horizon_sessions"]
            assert row["unmeasured_reason"] == ""
            assert row["target_close"] not in ("", None)
        else:
            assert row["unmeasured_reason"], "an unmeasured row must say why"

    # The v1 file is untouched by any of this and still carries its own column.
    v1_rows = _read(paths["tier_outcomes_path"])
    assert v1_rows
    assert {row["outcome_kind"] for row in v1_rows} == {"favorable_direction_scanrow_v1"}

    # The two files join on `observation_id`.
    v1_ids = {row["observation_id"] for row in v1_rows}
    v2_ids = {row["observation_id"] for row in rows}
    assert v1_ids and v1_ids.issubset(v2_ids)


def test_the_export_never_fetches_and_survives_a_closes_lookup_that_raises(tmp_path, monkeypatch):
    """A broken `closes_for` costs the v2 rows their measurement, never the v1 files.

    The evidence-store rule, applied to an export: a shadow file may never cost
    the thing it describes.
    """
    import market_calendar

    paths = _export_paths(tmp_path)
    monkeypatch.setattr(market_calendar, "last_completed_session", lambda now: date(2026, 6, 30))

    def exploding_closes(symbol):
        raise RuntimeError("no bars, and certainly no network")

    result = legacy.export_bot_tier_tracker_views(
        history_df=_history(), closes_for=exploding_closes, **paths
    )

    assert result["tier_outcome_count"] > 0
    assert paths["tier_outcomes_path"].exists()
    rows = _read(paths["session_horizon_path"])
    assert rows
    assert all(str(row["measured"]).strip().lower() == "false" for row in rows)
    assert {row["unmeasured_reason"] for row in rows} == {"no_bar_for_target_session"}


def test_a_failed_v2_write_never_costs_the_v1_exports(tmp_path, monkeypatch):
    """The guard, proven by breaking the builder the export calls."""
    import market_calendar

    from master_avwap_lib import session_horizon_outcomes

    paths = _export_paths(tmp_path)
    monkeypatch.setattr(market_calendar, "last_completed_session", lambda now: date(2026, 6, 30))
    monkeypatch.setattr(
        session_horizon_outcomes,
        "build_session_horizon_observation_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = legacy.export_bot_tier_tracker_views(
        history_df=_history(), closes_for=lambda symbol: None, **paths
    )

    assert result["tier_outcome_count"] > 0
    assert result["session_horizon_outcome_count"] == 0
    assert paths["tier_outcomes_path"].exists()


def test_the_daily_frame_lookup_reads_closes_and_never_fetches():
    """`closes_from_daily_frames` is the desk's `closes_for`: frames in, dict out."""
    from master_avwap_lib.session_horizon_outcomes import closes_from_daily_frames

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "close": [100.0, 101.5],
        }
    )
    closes_for = closes_from_daily_frames({"AAA": frame})
    assert closes_for("aaa") == {date(2026, 6, 1): 100.0, date(2026, 6, 2): 101.5}
    # A symbol with no frame is "we did not look", not "there was no bar".
    assert closes_for("ZZZ") is None


# ---------------------------------------------------------------------------
# The describe() line, on all three surfaces
# ---------------------------------------------------------------------------
def _write_v1_csv(path: Path, *, stale_rows: int = 1) -> None:
    columns = list(legacy.TIER_OUTCOME_COLUMNS)
    rows = []
    for index in range(6):
        stale = index < stale_rows
        rows.append(
            {
                "observation_id": f"OBS{index}:5",
                "scan_row_id": f"OBS{index}",
                "scan_date": "2026-06-03",
                "future_scan_date": "2026-06-10",
                "horizon_sessions": 5,
                "tier": "S",
                "tier_source": "assigned",
                "symbol": f"SYM{index}",
                "side": "LONG",
                "setup_family": "avwap_breakout",
                "entry_close": 100.0,
                "future_close": 105.0,
                "side_return_pct": 5.0,
                "win": "True",
                "sessions_spanned": 18 if stale else 5,
                "stale_horizon": "True" if stale else "False",
                "outcome_kind": "favorable_direction_scanrow_v1",
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def test_the_setup_docs_surface_says_what_the_rate_is(tmp_path, monkeypatch):
    import setup_docs

    path = tmp_path / "master_avwap_tier_outcomes.csv"
    _write_v1_csv(path)
    setup_docs.clear_family_outcome_cache()
    monkeypatch.setattr(setup_docs, "_family_outcomes_path", lambda: path)
    monkeypatch.setattr(setup_docs, "_family_outcomes_window", lambda: ("2026-06-01", "2026-06-30"))
    try:
        rows = setup_docs.family_headline_rows()
        line = setup_docs.family_record_coverage_line()
    finally:
        setup_docs.clear_family_outcome_cache()

    assert rows["avwap_breakout"]["n"] == 5
    assert "favorable_direction_scanrow_v1" in line
    # The horizon is stated IN ITS OWN UNIT - the packet's whole point.
    assert "5 scan rows" in line
    assert "5 eligible" in line
    assert "stale_horizon 1" in line


def test_the_away_digest_says_what_it_ranked_on():
    import autopilot_core

    payload = {
        "swing_picks": [{"symbol": "AAA", "side": "LONG", "family": "avwap_breakout"}],
        "swing_family_records": {"avwap_breakout": {"wins": 5, "losses": 1}},
        "swing_family_record_line": (
            "favorable_direction_scanrow_v1 over 5 scan rows, last 20 sessions "
            "(2026-06-02..2026-06-30): 5 eligible / 0 pending / 1 excluded (stale_horizon 1)"
        ),
    }
    report = autopilot_core.render_away_report(payload)
    assert "Ranked on: favorable_direction_scanrow_v1 over 5 scan rows" in report

    # No picks, no claim: the line describes a ranking that happened.
    empty = autopilot_core.render_away_report(
        {"swing_picks": [], "swing_family_record_line": payload["swing_family_record_line"]}
    )
    assert "Ranked on:" not in empty


def test_the_setups_panel_shows_the_coverage_line():
    pytest.importorskip("PySide6", reason="the setups table is a Qt panel")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from ui.panels.master_avwap_panel import MasterAvwapPanel

    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = MasterAvwapPanel()
    try:
        assert panel.family_record_label.text() == ""
        line = (
            "favorable_direction_scanrow_v1 over 5 scan rows, last 20 sessions "
            "(2026-06-02..2026-06-30): 5 eligible / 0 pending / 1 excluded (stale_horizon 1)"
        )
        panel._on_family_records_ready(({"avwap_breakout": {"win_rate": 0.6}}, line))
        assert panel.family_record_label.text() == line
        assert "5 scan rows" in panel.family_record_label.toolTip()
    finally:
        panel.deleteLater()


def test_the_three_readers_apply_ONE_policy_object():
    """The rules live in `swing_evidence`, not written out three times."""
    import autopilot_core
    import setup_docs

    docs_source = (ROOT / "scripts" / "setup_docs.py").read_text(encoding="utf-8")
    core_source = (ROOT / "scripts" / "autopilot_core.py").read_text(encoding="utf-8")
    for source in (docs_source, core_source):
        assert "read_eligible_rows" in source
        # The hand-written stale filter each of them used to carry is gone.
        assert 'row.get("stale_horizon") or ""' not in source

    assert callable(setup_docs.family_record_coverage_line)
    assert callable(autopilot_core.swing_family_read)
