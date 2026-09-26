"""B9: a repeat Working-lately build reuses its reads and publishes the same bytes.

The three reads (and the grades / recent M5 built from the same outcome window)
are cached on their input files' mtime and size plus today's windows. A cached
build must equal a cold one byte for byte; a changed input must be read again.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

OUTCOME_FIELDS = (
    "event_id", "event_type", "trade_date", "symbol", "direction", "entry_time",
    "bars_elapsed", "minutes_elapsed", "target_1r_hit", "stop_hit", "mfe_r", "context_json",
)
TIER_FIELDS = (
    "observation_id", "scan_date", "future_scan_date", "horizon_sessions", "tier", "symbol",
    "side", "priority_bucket", "setup_family", "side_return_pct", "win", "stale_horizon",
)


def _sessions() -> list[str]:
    """Weekdays inside today's lately window, newest last."""
    import held_run_score

    first, last = held_run_score.window_bounds()
    day, end = date.fromisoformat(first), date.fromisoformat(last)
    out = []
    while day <= end:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out[-9:]


def _outcome_rows(sessions: list[str]) -> list[dict]:
    rows = []
    for day_index, session in enumerate(sessions):
        stamp = session.replace("-", "")
        for n in range(6):
            bounce = ("vwap", "ema_15", "eod_vwap-vwap")[n % 3]
            direction = "long" if n % 2 == 0 else "short"
            event_id = f"S{n}_{direction}_{stamp}_10_{n:02d}_00_{bounce}"
            won = (n + day_index) % 3 != 0
            for bars in (1, 2, 3):
                rows.append({
                    "event_id": event_id,
                    "event_type": "final" if bars == 3 else "update",
                    "trade_date": session,
                    "symbol": f"S{n}",
                    "direction": direction,
                    "entry_time": f"{session}T10:{n:02d}:00",
                    "bars_elapsed": str(bars),
                    "minutes_elapsed": str(bars * 5 + 30),
                    "target_1r_hit": "True" if (won and bars >= 2) else "False",
                    "stop_hit": "True" if (not won and bars >= 2) else "False",
                    "mfe_r": "2.0" if won else "0.3",
                    "context_json": json.dumps({"market_environment": "bullish_strong"}),
                })
    return rows


def _tier_rows(sessions: list[str]) -> list[dict]:
    rows = []
    for day_index, session in enumerate(sessions[:6]):
        for n in range(8):
            move = 1.5 if (n + day_index) % 3 else -0.8
            rows.append({
                "observation_id": f"F{n}:{session}:5",
                "scan_date": session,
                "future_scan_date": sessions[min(day_index + 3, len(sessions) - 1)],
                "horizon_sessions": "5",
                "tier": "S",
                "symbol": f"F{n}",
                "side": "LONG" if n % 2 else "SHORT",
                "priority_bucket": "favorite_setup",
                "setup_family": ("avwap_breakout", "general")[n % 2],
                "side_return_pct": str(move),
                "win": "True" if move > 0 else "False",
                "stale_horizon": "",
            })
    return rows


def _recent_rows(sessions: list[str]) -> list[dict]:
    return [
        {
            "namespace": "live", "side": side, "priority_bucket": "favorite_setup",
            "setup_family": family, "closed_setups": "40", "tracked_setups": "43",
            "avg_closed_r": "0.21", "representative_closed_r": rep, "n_wins": wins,
            "n_losses": losses, "n_flats": "0", "n_unmeasured": "1", "n_pending": "2",
            "n_symbols": "12", "n_entry_sessions": "14",
            "outcome_kind": "representative_closed_r", "outcome_version": "recent_types_v2",
            "latest_measured_session": sessions[-1],
            "tracker_saved_at": f"{sessions[-1]}T16:20:00-04:00",
        }
        for side, family, rep, wins, losses in (
            ("LONG", "avwape_to_1stdev", "0.38", "34", "6"),
            ("SHORT", "avwap_breakout", "0.12", "22", "18"),
        )
    ]


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """The service pointed at a small fixture home, cache empty, clock frozen."""
    import held_run_score
    import project_paths
    import swing_evidence
    import working_lately
    from ui.services import working_lately_service as svc

    sessions = _sessions()
    log = tmp_path / "intraday_bounce_outcomes.csv"
    _write_csv(log, OUTCOME_FIELDS, _outcome_rows(sessions))
    tier = tmp_path / "master_avwap_tier_outcomes.csv"
    _write_csv(tier, TIER_FIELDS, _tier_rows(sessions))
    recent = tmp_path / "master_avwap_setup_type_recent_stats.csv"
    rows = _recent_rows(sessions)
    _write_csv(recent, list(rows[0]), rows)

    monkeypatch.setattr(svc, "_outcome_log_path", lambda: log)
    monkeypatch.setattr(svc, "_recent_stats_path", lambda: recent)
    monkeypatch.setattr(svc, "_scoring_snapshot_path", lambda: tmp_path / "absent_snapshot.json")
    monkeypatch.setattr(svc, "_horizon_outcomes_path", lambda: tmp_path / "absent_horizon.csv")
    monkeypatch.setattr(svc, "_spy_bars_path", lambda: tmp_path / "absent_spy.parquet")
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_TIER_OUTCOMES_FILE", tier)
    monkeypatch.setattr(project_paths, "INTRADAY_BOUNCE_OUTCOMES_FILE", log)
    monkeypatch.setattr(
        project_paths, "MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE", tmp_path / "absent_snapshot.json"
    )
    monkeypatch.setattr(working_lately, "market_local_now", lambda: "2026-09-26T09:00:00-04:00")
    monkeypatch.setattr(svc, "_LOOKING_BACK_CACHE", {})

    reads = {"outcome": 0, "favorable": 0}
    real_outcome = held_run_score.read_outcome_rows
    real_eligible = swing_evidence.read_eligible_rows

    def counted_outcome(*args, **kwargs):
        reads["outcome"] += 1
        return real_outcome(*args, **kwargs)

    def counted_eligible(*args, **kwargs):
        reads["favorable"] += 1
        return real_eligible(*args, **kwargs)

    monkeypatch.setattr(held_run_score, "read_outcome_rows", counted_outcome)
    monkeypatch.setattr(swing_evidence, "read_eligible_rows", counted_eligible)

    counter = iter(range(1000))

    def build() -> str:
        # A fresh store each time, so no build sees another's persisted snapshot.
        service = svc.WorkingLatelyService(store_dir=tmp_path / f"wl{next(counter)}")
        return json.dumps(service.build_payload(), sort_keys=True, default=str)

    def cold() -> str:
        svc._LOOKING_BACK_CACHE.clear()
        return build()

    return svc, build, cold, reads, log, tier


def test_a_repeat_build_reads_nothing_and_equals_a_cold_build(staged):
    svc, build, cold, reads, _log, _tier = staged
    first = build()
    payload = json.loads(first)
    kinds = {cell.get("kind") for cell in payload["cells"]}
    assert {"daytrade_held_run", "swing_favorable", "swing_trade_r"} <= kinds
    assert payload["setup_grades"]["daytrade"] and payload["looking_back"]
    reads.update(outcome=0, favorable=0)

    cached = build()
    assert reads == {"outcome": 0, "favorable": 0}, "a repeat build re-read an unchanged input"
    assert svc._OUTCOME_ROWS_THIS_BUILD is None and svc._OUTCOME_BUILD_ACTIVE is False

    assert cached == cold() == first


def test_a_changed_outcome_log_is_read_again_and_matches_a_cold_build(staged):
    _svc, build, cold, reads, log, _tier = staged
    before = build()
    rows = list(csv.DictReader(log.open(newline="", encoding="utf-8")))
    extra = dict(rows[-1], event_id="EXTRA_" + rows[-1]["event_id"], mfe_r="0.1",
                 target_1r_hit="False", stop_hit="True")
    with log.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=list(OUTCOME_FIELDS)).writerow(extra)
    _bump_mtime(log)
    reads.update(outcome=0, favorable=0)

    after = build()
    assert reads["outcome"] == 1, "the changed log was read exactly once for the build"
    assert reads["favorable"] == 0
    assert after != before
    assert after == cold()


def test_a_changed_tier_file_invalidates_the_favorable_read(staged):
    _svc, build, cold, reads, _log, tier = staged
    build()
    text = tier.read_text(encoding="utf-8").replace(",-0.8,False,", ",2.5,True,", 1)
    tier.write_text(text, encoding="utf-8")
    _bump_mtime(tier)
    reads.update(outcome=0, favorable=0)

    after = build()
    assert reads["favorable"] >= 1 and reads["outcome"] == 0
    assert after == cold()


def test_a_caller_editing_the_payload_cannot_change_the_next_build(staged):
    svc, build, cold, _reads, _log, _tier = staged
    reference = cold()
    service = svc.WorkingLatelyService(store_dir=svc.default_store_dir() / "b9_edit")
    payload = service.build_payload()
    payload["setup_grades"]["daytrade"].clear()
    payload["looking_back"].clear()
    held = svc.read_held_run_summaries()
    assert held
    held.clear()
    svc.read_favorable_read().rows.clear()
    assert build() == reference
