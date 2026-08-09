from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


NY = ZoneInfo("America/New_York")
DAY = "2026-07-30"


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base(event_type, event_id):
    return {
        "code_version": "regime_infrastructure_phase1_v1",
        "event_type": event_type,
        "event_id": event_id,
        "session_date": DAY,
        "as_of": f"{DAY}T10:30:00-04:00",
        "written_at": f"{DAY}T10:30:01-04:00",
    }


def test_healthy_after_close_session_passes_acceptance_audit(tmp_path):
    from regime_collection_audit import audit_regime_collection

    technical = tmp_path / "technical.jsonl"
    breadth = tmp_path / "vold.jsonl"
    rows = [
        {
            **_base("level_resolved", "r1"),
            "followup_tracking_version": "regime_infrastructure_phase1_v1",
        },
        {
            **_base("post_resolution_tracking_started", "r1|followup"),
            "source_resolution_id": "r1",
        },
    ]
    for horizon in (30, 60, 90):
        rows.append(
            {
                **_base("post_resolution_followup", f"r1|followup|{horizon}"),
                "source_resolution_id": "r1",
                "horizon_minutes": horizon,
                "truncated": horizon > 30,
                "data_gap": False,
            }
        )
    rows.extend(
        [
            {
                **_base("frozen_intraday_snapshot", "s1030"),
                "snapshot_key": f"{DAY}|10:30",
                "target_market_time": "10:30",
            },
            {
                **_base("frozen_intraday_snapshot", "s1200"),
                "snapshot_key": f"{DAY}|12:00",
                "target_market_time": "12:00",
            },
            {
                **_base("opening_range_baseline", "opening"),
                "snapshot_key": f"{DAY}|opening_range",
                "data_gap": False,
            },
        ]
    )
    _write(technical, rows)
    contract = {
        "symbol": "TICK-NYSE",
        "proxy_kind": "nyse_tick_proxy",
        "is_exact_vold": False,
    }
    breadth_rows = [
        {
            **_base("contract_verified", "contract"),
            "contract": contract,
        }
    ]
    open_time = datetime(2026, 7, 30, 9, 30, tzinfo=NY)
    for index in range(78):
        end = open_time + timedelta(minutes=5 * (index + 1))
        breadth_rows.append(
            {
                **_base("breadth_bar", f"b{index}"),
                "contract": contract,
                "bar_end": end.isoformat(timespec="seconds"),
                "as_of": end.isoformat(timespec="seconds"),
            }
        )
    _write(breadth, breadth_rows)

    report = audit_regime_collection(
        session_date=DAY,
        technical_events_path=technical,
        breadth_events_path=breadth,
        now=datetime(2026, 7, 31, 10, 0, tzinfo=NY),
    )

    assert report["status"] == "HEALTHY"
    assert report["promotion_status"] == "EXPLORATORY / NON-PROMOTABLE"
    assert report["technical_followups"]["horizon_counts"] == {
        "30": 1,
        "60": 1,
        "90": 1,
    }
    assert report["breadth_recorder"]["semantic_status"] == "PROXY:nyse_tick_proxy"


def test_missing_snapshot_duplicate_bar_and_incomplete_chain_are_blockers(tmp_path):
    from regime_collection_audit import audit_regime_collection

    technical = tmp_path / "technical.jsonl"
    breadth = tmp_path / "vold.jsonl"
    _write(
        technical,
        [
            {
                **_base("level_resolved", "r1"),
                "followup_tracking_version": "regime_infrastructure_phase1_v1",
            },
            {
                **_base("post_resolution_tracking_started", "r1|followup"),
                "source_resolution_id": "r1",
            },
            {
                **_base("post_resolution_followup", "f30"),
                "source_resolution_id": "r1",
                "horizon_minutes": 30,
            },
        ],
    )
    duplicate = {
        **_base("breadth_bar", "b1"),
        "bar_end": f"{DAY}T09:35:00-04:00",
    }
    _write(breadth, [duplicate, duplicate])

    report = audit_regime_collection(
        session_date=DAY,
        technical_events_path=technical,
        breadth_events_path=breadth,
        now=datetime(2026, 7, 30, 16, 5, tzinfo=NY),
    )

    assert report["status"] == "UNHEALTHY"
    assert report["frozen_snapshots"]["missing_labels"] == ["10:30", "12:00"]
    assert report["technical_followups"]["incomplete_chains"]["r1"] == [60, 90]
    assert report["breadth_recorder"]["duplicate_bar_ends"] == [
        f"{DAY}T09:35:00-04:00"
    ]
    assert "breadth ledger is partial without enough explicit data-gap coverage" in report[
        "blockers"
    ]
    assert "breadth bars have no contract-verification event" in report["blockers"]


# --- follow-up gaps and outcome coverage, reported on their own -----------
#
# A session can be HEALTHY -- every chain closed, every gap explicitly marked
# -- and still hand the promotion study almost no usable outcomes, because an
# explicit data_gap row satisfies the completeness check while carrying no
# displacement/MFE/MAE at all. The audit reports both figures so that state is
# visible; the verdict logic is deliberately unchanged (checkpoint review
# 2026-08-08 second review).


def _healthy_session_rows(*, gap_horizons=()):
    """One resolved level with a full +30/60/90 chain, some windows empty."""
    rows = [
        {
            **_base("level_resolved", "r1"),
            "followup_tracking_version": "regime_infrastructure_phase1_v1",
        },
        {
            **_base("post_resolution_tracking_started", "r1|followup"),
            "source_resolution_id": "r1",
            "resolution_bar_close": f"{DAY}T10:00:00-04:00",
        },
    ]
    for horizon in (30, 60, 90):
        is_gap = horizon in gap_horizons
        rows.append(
            {
                **_base("post_resolution_followup", f"r1|followup|{horizon}"),
                "source_resolution_id": "r1",
                "horizon_minutes": horizon,
                "truncated": False,
                "data_gap": is_gap,
                "data_gap_reason": (
                    "chain sweeper: no completed M5 bars available for MU "
                    f"on {DAY} after 3 attempt(s)"
                    if is_gap
                    else ""
                ),
            }
        )
    rows.extend(
        [
            {
                **_base("frozen_intraday_snapshot", "s1030"),
                "target_market_time": "10:30",
            },
            {
                **_base("frozen_intraday_snapshot", "s1200"),
                "target_market_time": "12:00",
            },
            {**_base("opening_range_baseline", "opening"), "data_gap": False},
        ]
    )
    return rows


def _audit_with(tmp_path, rows):
    from regime_collection_audit import audit_regime_collection

    technical = tmp_path / "technical.jsonl"
    _write(technical, rows)
    (tmp_path / "vold.jsonl").write_text("", encoding="utf-8")
    return audit_regime_collection(
        session_date=DAY,
        technical_events_path=technical,
        breadth_events_path=tmp_path / "vold.jsonl",
        now=datetime(2026, 7, 30, 16, 30, tzinfo=NY),
    )


def test_full_outcome_coverage_is_reported_when_every_window_has_bars(tmp_path):
    report = _audit_with(tmp_path, _healthy_session_rows())

    followups = report["technical_followups"]
    assert followups["matured_window_count"] == 3
    assert followups["outcome_count"] == 3
    assert followups["outcome_coverage"] == 1.0
    assert followups["data_gap_by_horizon"] == {"30": 0, "60": 0, "90": 0}


def test_healthy_with_gaps_shows_the_coverage_shortfall(tmp_path):
    from regime_collection_audit import format_audit

    report = _audit_with(tmp_path, _healthy_session_rows(gap_horizons=(60, 90)))

    # The verdict logic is untouched: the chain is complete, so the empty
    # windows raise no follow-up blocker at all. (The one blocker here is the
    # deliberately empty breadth ledger this fixture does not populate.)
    assert not report["technical_followups"]["incomplete_chains"]
    assert report["blockers"] == ["breadth ledger has no completed-M5 rows"]

    # ...but two thirds of its matured windows carry no outcome at all, and
    # that is now stated rather than buried in a single "gaps=2" token.
    followups = report["technical_followups"]
    assert followups["matured_window_count"] == 3
    assert followups["outcome_count"] == 1
    assert followups["outcome_coverage"] == round(1 / 3, 4)
    assert followups["data_gap_by_horizon"] == {"30": 0, "60": 1, "90": 1}
    assert any("after 3 attempt(s)" in reason for reason in followups["data_gap_reasons"])

    text = format_audit(report)
    gap_line = next(line for line in text.splitlines() if line.startswith("Follow-up data gaps:"))
    coverage_line = next(line for line in text.splitlines() if line.startswith("Outcome coverage:"))
    assert "2 of 3 window(s)" in gap_line
    assert "1/3 matured window(s) carry metrics (33%)" in coverage_line


def test_nothing_matured_yet_is_not_reported_as_zero_coverage(tmp_path):
    # Missing data is uncertainty, never confirmation: before any window has
    # run out there is no shortfall to report, only nothing to report.
    from regime_collection_audit import audit_regime_collection, format_audit

    technical = tmp_path / "technical.jsonl"
    _write(
        technical,
        [
            {
                **_base("level_resolved", "r1"),
                "followup_tracking_version": "regime_infrastructure_phase1_v1",
            },
            {
                **_base("post_resolution_tracking_started", "r1|followup"),
                "source_resolution_id": "r1",
                "resolution_bar_close": f"{DAY}T10:00:00-04:00",
            },
        ],
    )
    (tmp_path / "vold.jsonl").write_text("", encoding="utf-8")
    report = audit_regime_collection(
        session_date=DAY,
        technical_events_path=technical,
        breadth_events_path=tmp_path / "vold.jsonl",
        now=datetime(2026, 7, 30, 10, 15, tzinfo=NY),
    )

    followups = report["technical_followups"]
    assert followups["matured_window_count"] == 0
    assert followups["outcome_coverage"] is None
    assert "nothing matured yet" in format_audit(report)
