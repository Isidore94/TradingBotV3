"""P2-9 review fixes (NO-GO @ 05c5c146).

* The hold-out keys a cell by (kind, namespace, SIDE, bucket, family): rows are
  the cells 1:1, and the prior side pairs only with the same population.
* The prior M5 cache follows the outcome log's mtime/size and never keeps a
  result read from a missing log.
* The prior stream runs after the recent window's rows are freed.
* A swing pick with no representative scenario is "unmeasurable".
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _row(namespace, bucket, wins, losses, sessions, latest, *, family="mid_earnings_ema15_retest"):
    """One `master_avwap_setup_type_recent_stats.csv` row (strings, as read)."""
    return {
        "namespace": namespace,
        "side": "LONG",
        "priority_bucket": bucket,
        "setup_family": family,
        "closed_setups": str(wins + losses),
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "0",
        "n_unmeasured": "0",
        "n_pending": "0",
        "n_symbols": "5",
        "n_entry_sessions": str(sessions),
        "latest_measured_session": latest,
    }


#: The reviewer's live-copy case (2026-09-24 export): LONG mid_earnings_ema15_retest
#: is SIX populations - two live buckets (n=41, n=37) and four study buckets.
LIVE_COPY_ROWS = [
    _row("live", "near_favorite_zone", 31, 10, 19, "2026-09-21"),
    _row("live", "favorite_setup", 23, 14, 13, "2026-09-21"),
    _row("study", "study_hv_level", 3, 2, 3, "2026-09-17"),
    _row("study", "study_relative_avwap", 1, 0, 1, "2026-09-16"),
    _row("study", "study_weekly_ema8_hold", 1, 0, 2, "2026-09-11"),
    _row("study", "study_htf_trend", 0, 0, 1, ""),
]


def test_the_live_copy_case_keeps_every_population_and_pairs_its_own_prior():
    import working_lately as wl

    recent = wl.bucketed_trade_r_cells(LIVE_COPY_ROWS)
    prior = wl.bucketed_trade_r_cells([_row("live", "near_favorite_zone", 20, 12, 15, "2026-08-20")])
    rows = wl.holdout_view(recent, prior, prior_sources={"swing_trade_r": ("live",)})
    assert len(rows) == len(LIVE_COPY_ROWS), "one row per cell, none merged"
    by = {(r["namespace"], r["bucket"]): r for r in rows}
    assert len(by) == 6
    near = by[("live", "near_favorite_zone")]
    assert near["recent_text"].endswith("n=41")
    assert near["prior_text"] == "0.62 (>= 0.45) n=32"
    fav = by[("live", "favorite_setup")]
    assert fav["recent_text"].endswith("n=37")
    assert fav["prior_text"] == wl.HOLDOUT_NO_PRIOR, "never the other bucket's prior"
    for bucket in ("study_hv_level", "study_relative_avwap", "study_weekly_ema8_hold", "study_htf_trend"):
        assert by[("study", bucket)]["prior_text"] == "no prior source (study)"
    assert by[("study", "study_htf_trend")]["recent_text"] == "n<30 (n=0)"
    # The live cells lead.
    assert [r["namespace"] for r in rows[:2]] == ["live", "live"]


def test_duplicate_side_family_across_buckets_and_namespaces_never_collapse():
    import working_lately as wl

    recent = wl.bucketed_trade_r_cells(
        [
            _row("live", "a", 40, 5, 12, "2026-09-21", family="f"),
            _row("live", "b", 10, 30, 12, "2026-09-21", family="f"),
            _row("study", "a", 2, 1, 2, "2026-09-21", family="f"),
        ]
    )
    prior = wl.bucketed_trade_r_cells(
        [
            _row("live", "b", 35, 5, 12, "2026-08-21", family="f"),
            _row("live", "c", 35, 5, 12, "2026-08-21", family="f"),
        ]
    )
    rows = wl.holdout_view(recent, prior)
    keyed = {(r["namespace"], r["bucket"]): r for r in rows}
    assert len(rows) == 4 and len(keyed) == 4
    assert keyed[("live", "a")]["prior_text"] == wl.HOLDOUT_NO_PRIOR
    assert keyed[("live", "b")]["recent_text"].endswith("n=40")
    assert keyed[("live", "b")]["prior_text"].endswith("n=40")
    assert keyed[("live", "b")]["prior_text"].startswith("0.88")
    assert keyed[("study", "a")]["prior_text"] == wl.HOLDOUT_NO_PRIOR
    assert keyed[("live", "c")]["recent_text"] == wl.HOLDOUT_NOT_IN_WINDOW


def test_the_prior_m5_cache_follows_the_log_and_never_keeps_a_missing_file(tmp_path, monkeypatch):
    import os

    from ui.services import working_lately_service as svc

    log = tmp_path / "intraday_bounce_outcomes.csv"
    calls = []
    monkeypatch.setattr(svc, "_outcome_log_path", lambda: log)
    monkeypatch.setattr(svc, "_stream_outcome_rows", lambda window: calls.append(window) or [])
    svc._LOOKING_BACK_CACHE.clear()
    window = ("2026-08-01", "2026-08-28")

    svc._prior_m5(window)
    svc._prior_m5(window)
    assert len(calls) == 2, "a missing log is never cached"

    log.write_text("event_id,trade_date\n", encoding="utf-8")
    svc._prior_m5(window)
    svc._prior_m5(window)
    assert len(calls) == 3, "cached while the log is unchanged"

    log.write_text("event_id,trade_date\nx,2026-08-02\n", encoding="utf-8")
    stat = log.stat()
    os.utime(log, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    svc._prior_m5(window)
    assert len(calls) == 4, "a changed log is read again"


def test_the_prior_stream_runs_after_the_recent_rows_are_freed(tmp_path, monkeypatch):
    import looking_back as lb
    from ui.services import working_lately_service as svc

    seen = []

    def fake_stream(window):
        seen.append(svc._OUTCOME_ROWS_THIS_BUILD)
        return []

    svc._LOOKING_BACK_CACHE.clear()
    monkeypatch.setattr(svc, "read_recent_rows", lambda: [])
    monkeypatch.setattr(svc, "read_favorable_read", lambda: None)
    monkeypatch.setattr(svc, "read_held_run_summaries", lambda: None)
    monkeypatch.setattr(svc, "_outcome_rows", lambda: svc._OUTCOME_ROWS_THIS_BUILD or [])
    monkeypatch.setattr(svc, "_stream_outcome_rows", fake_stream)
    monkeypatch.setattr(svc, "_scoring_snapshot_path", lambda: tmp_path / "absent.json")
    log = tmp_path / "log.csv"
    log.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(svc, "_outcome_log_path", lambda: log)
    payload = svc.WorkingLatelyService(store_dir=tmp_path / "wl").build_payload()
    assert seen == [None], "the recent window's rows were still held during the prior stream"
    assert payload["looking_back"]["schema"] == lb.SCHEMA


def test_a_swing_pick_with_no_representative_is_unmeasurable_not_pending():
    import looking_back as lb

    def setup(symbol, status, r):
        return {
            "symbol": symbol,
            "side": "LONG",
            "scan_date": "2026-09-01",
            "anchor_date": "2026-08-01",
            "priority_bucket": "favorite_setup",
            "setup_family": "general",
            "_scoring_outcome_summary": {
                "tradeable_scenario_count": 3,
                "closed_tradeable_scenario_count": 3 if status == "closed" else 0,
                "representative_closed_r": r,
                "representative_status": status,
            },
        }

    results = lb.swing_pick_results(
        {"a": setup("AAA", "closed", 1.0), "b": setup("BBB", "pending", None), "c": setup("CCC", "", None)},
        context=lambda s: (s["side"], s["priority_bucket"], s["setup_family"]),
    )
    status = {r["symbol"]: r["status"] for r in results}
    assert status == {"AAA": "closed", "BBB": "pending", "CCC": lb.UNMEASURABLE}
    curve = lb.equity_curve(results, population=lb.SWING)
    assert (curve["n"], curve["not_graded"], curve["unmeasurable"]) == (1, 1, 1)
    assert lb.curve_line(curve).endswith("1 not graded yet; 1 unmeasurable")
