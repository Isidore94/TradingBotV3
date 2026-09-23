"""scripts/research_pack.py - the read-only research pack for a frontier model.

Pinned here: it refuses to write under a live store, a missing source is
reported as unknown (never zero), journal trades join only to setups that were
known BEFORE the trade opened, cells under the sample floor are never ranked,
and an export leaves every source file untouched.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import research_pack as rp  # noqa: E402
from scripts.research_warehouse import exchange_calendar as xcal  # noqa: E402
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
TRIGGER_DAY = date(2026, 8, 3)  # Monday
NEXT_DAY = date(2026, 8, 4)
TRIGGER_AT = xcal.trading_session(TRIGGER_DAY).rth_close_at


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _occurrence(occ_id, symbol="AAPL", setup="AVWAPE_TO_FIRST_DEV", side="LONG", trigger_at=TRIGGER_AT, cluster=None):
    return {
        "occurrence_id": occ_id,
        "symbol": symbol,
        "canonical_setup_id": setup,
        "side": side,
        "structural_timeframe": "D1",
        "trigger_timeframe": "D1",
        "dependency_cluster_id": cluster or f"cl-{occ_id}",
        "status": "CLOSED",
        "trigger_at": trigger_at,
        "entry_price_ref": 100.0,
        "stop_price_ref": 95.0,
        "detector_version": "fixture",
        "tags": json.dumps({"priority_bucket": "favorite_setup", "anchor_date": "2026-07-01"}),
        "event_at": trigger_at,
        "observed_at": trigger_at,
        "computed_at": trigger_at,
        "revision_id": "rev-1",
    }


def _outcome(occ_id, net_r, recipe="m5close_current_anchor1_2r_v1", state="TARGETED"):
    entry_at = xcal.trading_session(NEXT_DAY).rth_open_at + timedelta(minutes=5)
    return {
        "occurrence_id": occ_id,
        "recipe_id": recipe,
        "outcome_definition_id": rp.OUTCOME_DEFINITION_ID,
        "analysis_unit": "OPPORTUNITY",
        "entry_at": entry_at,
        "entry_price": 100.0,
        "stop_price": 95.0,
        "stop_distance": 5.0,
        "r_at_s5": net_r,
        "net_r": net_r,
        "gross_r": net_r,
        "mfe_r": abs(net_r) + 0.5,
        "mae_r": -0.5,
        "result_state": state,
        "maturity_at": entry_at + timedelta(days=3),
        "computed_at": entry_at + timedelta(days=4),
    }


def _feature(symbol, session_date, computed_at, spy_state):
    return {
        "symbol": symbol,
        "session_date": session_date,
        "feature_set_version": "tier1_v2",
        "close": 101.0,
        "atr14": 2.0,
        "dist_sma50_atr": 0.5,
        "dist_sma200_atr": 3.0,
        "spy_regime_state": spy_state,
        "computed_at": computed_at,
        "event_at": computed_at,
    }


@pytest.fixture()
def lake(tmp_path):
    store = ResearchStore.open(tmp_path / "lake")
    store.publish(
        "setup_occurrence",
        [
            _occurrence("o1", "AAPL", side="LONG"),
            _occurrence("o2", "MSFT", side="SHORT"),
            _occurrence("o3", "AA", side="LONG"),
        ],
        git_commit="fixture",
    )
    store.publish(
        "outcome_path",
        [_outcome("o1", 2.0), _outcome("o2", -1.0, state="STOPPED"), _outcome("o3", 1.0)],
        git_commit="fixture",
    )
    close = TRIGGER_AT
    store.publish(
        "feature_snapshot_daily",
        [
            # Known at the trigger: the one the pack must use.
            _feature("AAPL", TRIGGER_DAY, close, "bull"),
            # The NEXT session knows how the setup went: never used.
            _feature("AAPL", NEXT_DAY, close + timedelta(days=1), "LEAK"),
        ],
        git_commit="fixture",
    )
    return tmp_path / "lake"


def _write_bounce_csv(path: Path, events):
    fields = [
        "schema_version", "event_id", "event_type", "logged_at", "trade_date", "symbol", "direction",
        "entry_time", "entry_price", "stop_price", "risk_per_share", "bars_elapsed", "minutes_elapsed",
        "close_r", "mfe_r", "mae_r", "best_price", "worst_price", "target_1r_hit", "target_2r_hit",
        "stop_hit", "status", "milestone_bar", "context_json", "outcome_mode", "eod_close",
        "eod_move_pct", "mfe_pct", "mae_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in events:
            writer.writerow({key: row.get(key, "") for key in fields})


def _bounce_rows(symbol, direction, hhmm, close_r, bounce="ema_15", day="2026-08-04"):
    hh, mm = hhmm
    event_id = f"{symbol}_{direction}_{day.replace('-', '')}_{hh:02d}_{mm:02d}_00_{bounce}"
    context = json.dumps({"market_environment": "bullish_strong", "rrs_spy_signal": "RS", "sector": "Tech"})
    # Pacific wall clock, like the live file (entry_time naive, logged_at with offset).
    registered_at = f"{day}T{hh:02d}:{mm + 3:02d}:00-07:00"
    base = {
        "schema_version": "1", "event_id": event_id, "trade_date": day, "symbol": symbol,
        "direction": direction, "entry_time": f"{day}T{hh:02d}:{mm:02d}:00", "entry_price": "10",
        "stop_price": "9.5", "risk_per_share": "0.5", "context_json": context, "outcome_mode": "eod_hold",
    }
    return [
        {**base, "event_type": "registered", "logged_at": registered_at, "status": "open"},
        {**base, "event_type": "12_bar", "logged_at": f"{day}T{hh + 1:02d}:{mm:02d}:00-07:00", "close_r": "0.5", "status": "open"},
        {**base, "event_type": "update", "logged_at": f"{day}T{hh + 1:02d}:{mm:02d}:30-07:00", "close_r": "0.7", "status": "open"},
        {**base, "event_type": "final", "logged_at": f"{day}T13:10:00-07:00", "close_r": str(close_r),
         "mfe_r": "2.0", "mae_r": "-0.4", "status": "eod_complete"},
    ]


def _write_journal(path: Path, trades):
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE trades (trade_id TEXT PRIMARY KEY, broker TEXT, account_number TEXT, account_label TEXT,
        symbol TEXT, security_type TEXT, currency TEXT, direction TEXT, status TEXT, opened_at TEXT,
        closed_at TEXT, trade_date TEXT, quantity_opened REAL, quantity_closed REAL, average_entry_price REAL,
        average_exit_price REAL, gross_pnl REAL, commission REAL, fees REAL, net_pnl REAL, pnl_usd REAL,
        auto_tag_summary TEXT, tag_confidence REAL, updated_at TEXT, net_pnl_usd REAL)"""
    )
    connection.execute(
        """CREATE TABLE trade_annotations (trade_id TEXT PRIMARY KEY, setup_tags TEXT, notes TEXT,
        updated_at TEXT, planned_entry REAL, planned_stop REAL, planned_risk REAL, risk_source TEXT,
        tag_status TEXT, label_provenance TEXT)"""
    )
    for trade in trades:
        row = {
            "broker": "ibkr", "account_number": "U999", "account_label": "secret", "currency": "USD",
            "status": "CLOSED", "closed_at": "", "quantity_opened": 1, "quantity_closed": 1,
            "average_entry_price": 1, "average_exit_price": 1, "gross_pnl": 0, "commission": 0, "fees": 0,
            "auto_tag_summary": "", "tag_confidence": None, "updated_at": "2026-08-05T00:00:00+00:00",
            **trade,
        }
        row.setdefault("net_pnl_usd", row.get("net_pnl"))
        row.setdefault("pnl_usd", row.get("net_pnl"))
        columns = ",".join(row)
        connection.execute(f"INSERT INTO trades ({columns}) VALUES ({','.join('?' * len(row))})", list(row.values()))
    connection.commit()
    connection.close()


@pytest.fixture()
def sources(tmp_path, lake):
    bounce_csv = tmp_path / "intraday_bounce_outcomes.csv"
    _write_bounce_csv(
        bounce_csv,
        _bounce_rows("AAPL", "long", (7, 0), 1.5) + _bounce_rows("NVDA", "short", (8, 0), -0.5, bounce="vwap-eod_vwap"),
    )
    bounces = tmp_path / "intraday_bounces.csv"
    bounces.write_text(
        "time_local,trade_date,symbol,direction,bounce_types,tier,composite_r\n"
        "07:03:30,2026-08-04,AAPL,long,ema_15,A,0.2\n",
        encoding="utf-8",
    )
    journal = tmp_path / "trade_journal.sqlite3"
    _write_journal(
        journal,
        [
            # Next session after the AAPL LONG trigger, 12 min after the AAPL M5 alert (07:03 PT = 10:03 ET).
            {"trade_id": "t1", "symbol": "AAPL", "security_type": "STK", "direction": "LONG",
             "opened_at": "2026-08-04T10:15:00-04:00", "closed_at": "2026-08-04T11:00:00-04:00",
             "trade_date": "2026-08-04", "net_pnl": 50.0},
            # Opposite side to the MSFT SHORT setup: stays unmatched.
            {"trade_id": "t2", "symbol": "MSFT", "security_type": "STK", "direction": "LONG",
             "opened_at": "2026-08-04T10:15:00-04:00", "closed_at": "2026-08-04T11:00:00-04:00",
             "trade_date": "2026-08-04", "net_pnl": -20.0},
            # Sold put = bullish: matches the AA LONG setup through its underlying.
            {"trade_id": "t3", "symbol": "AA260918P00030000", "security_type": "OPT", "direction": "SHORT",
             "opened_at": "2026-08-05T10:00:00-04:00", "closed_at": "2026-08-06T10:00:00-04:00",
             "trade_date": "2026-08-06", "net_pnl": 30.0},
            # Same session as the trigger: the setup was not known yet.
            {"trade_id": "t4", "symbol": "AAPL", "security_type": "STK", "direction": "LONG",
             "opened_at": "2026-08-03T11:00:00-04:00", "closed_at": "2026-08-03T12:00:00-04:00",
             "trade_date": "2026-08-03", "net_pnl": 5.0},
        ],
    )
    return rp.Sources(
        lake_root=lake,
        journal_db=journal,
        bounce_outcomes_csv=bounce_csv,
        bounces_csv=bounces,
        extra_files={},
        ai_store_root=None,
        protected=(tmp_path / "live_home",),
    )


# ---------------------------------------------------------------------------
# refuse-live-write
# ---------------------------------------------------------------------------
def test_export_refuses_an_output_dir_inside_a_live_root(sources, tmp_path):
    target = tmp_path / "live_home" / "pack"
    with pytest.raises(rp.LiveStoreWriteRefused):
        rp.export_pack(sources, target)
    assert not target.exists()


def test_export_refuses_the_lake_itself(sources):
    with pytest.raises(rp.LiveStoreWriteRefused):
        rp.export_pack(sources, sources.lake_root / "gold" / "pack")


def test_protected_roots_include_the_project_path_stores(sources):
    from scripts import project_paths

    roots = {os.path.normcase(str(Path(root))) for root in rp.protected_roots(sources)}
    for live in (project_paths.PERSISTENT_DATA_DIR, project_paths.LOCAL_SETTINGS_DIR, Path(r"C:\TradingBotData")):
        assert os.path.normcase(str(Path(live))) in roots


def test_cli_export_into_a_live_root_fails_and_writes_nothing(tmp_path, capsys):
    from scripts import project_paths

    target = Path(project_paths.PERSISTENT_DATA_DIR) / "research_pack_should_not_exist"
    code = rp.main(["export", "--out", str(target)])
    assert code != 0
    assert not target.exists()


# ---------------------------------------------------------------------------
# missing source = unknown
# ---------------------------------------------------------------------------
def test_status_reports_missing_sources_as_unknown(tmp_path):
    empty = rp.Sources(
        lake_root=tmp_path / "no_lake",
        journal_db=tmp_path / "no.sqlite3",
        bounce_outcomes_csv=tmp_path / "no.csv",
        bounces_csv=None,
        extra_files={"setup_points_log": (tmp_path / "no.jsonl", "scan_date")},
        ai_store_root=None,
        protected=(),
    )
    rows = {row["source"]: row for row in rp.collect_status(empty)}
    for name in ("research_lake", "trade_journal", "m5_bounce_outcomes", "setup_points_log"):
        assert rows[name]["reachable"] is False
        assert rows[name]["rows"] is None
    assert rows["bounces_csv"]["reachable"] is False
    assert rows["ai_store"]["reachable"] is False


def test_export_with_no_sources_writes_a_manifest_that_says_missing(tmp_path):
    empty = rp.Sources(
        lake_root=None, journal_db=None, bounce_outcomes_csv=None, bounces_csv=None,
        extra_files={}, ai_store_root=None, protected=(),
    )
    manifest = rp.export_pack(empty, tmp_path / "out")
    assert manifest["tables"]["d1_occurrences"]["status"] == "missing"
    assert manifest["tables"]["journal_trades"]["status"] == "missing"
    assert (tmp_path / "out" / "manifest.json").exists()


def test_status_counts_a_fixture_lake(sources):
    rows = {row["source"]: row for row in rp.collect_status(sources)}
    assert rows["research_lake"]["reachable"] is True
    assert rows["lake:setup_occurrence"]["rows"] == 3
    assert rows["trade_journal"]["rows"] == 4
    assert rows["m5_bounce_outcomes"]["rows"] == 8
    assert rows["m5_bounce_outcomes"]["date_max"] == "2026-08-04"


# ---------------------------------------------------------------------------
# D1 table: point-in-time features
# ---------------------------------------------------------------------------
def test_d1_features_never_use_a_later_session(sources):
    store = ResearchStore(sources.lake_root)
    occurrences, outcomes = rp.build_d1_tables(store, as_of=datetime(2026, 9, 1, tzinfo=UTC))
    by_id = {row["occurrence_id"]: row for row in occurrences}
    assert by_id["o1"]["feat_spy_regime_state"] == "bull"
    assert by_id["o1"]["feat_basis"] == "as_observed"
    assert by_id["o1"]["priority_bucket"] == "favorite_setup"
    assert by_id["o1"]["hl_net_r"] == 2.0
    assert by_id["o1"]["session_date"] == "2026-08-03"
    # No snapshot at all is unknown, not zero.
    assert by_id["o2"]["feat_basis"] == "missing"
    assert by_id["o2"]["feat_dist_sma50_atr"] is None
    assert len(outcomes) == 3


def test_as_of_hides_occurrences_and_outcomes_from_the_future(sources):
    store = ResearchStore(sources.lake_root)
    occurrences, outcomes = rp.build_d1_tables(store, as_of=datetime(2026, 8, 5, tzinfo=UTC))
    by_id = {row["occurrence_id"]: row for row in occurrences}
    # Occurrence known, but its outcome was computed on 2026-08-08: not visible yet.
    assert by_id["o1"]["hl_net_r"] is None
    assert outcomes == []
    early, _ = rp.build_d1_tables(store, as_of=datetime(2026, 8, 1, tzinfo=UTC))
    assert early == []


# ---------------------------------------------------------------------------
# M5 alerts
# ---------------------------------------------------------------------------
def test_m5_alerts_pivot_milestones_and_join_the_tier(sources):
    alerts = rp.build_m5_alerts(sources.bounce_outcomes_csv, sources.bounces_csv, as_of=None)
    by_symbol = {row["symbol"]: row for row in alerts}
    aapl = by_symbol["AAPL"]
    assert aapl["family"] == "ema_15"
    assert aapl["r_12bar"] == 0.5
    assert aapl["close_r_final"] == 1.5
    assert aapl["market_environment"] == "bullish_strong"
    assert aapl["tier"] == "A"
    assert aapl["side"] == "LONG"
    assert aapl["known_at"].startswith("2026-08-04T14:03:00")  # 07:03 PT in UTC
    nvda = by_symbol["NVDA"]
    assert nvda["bounce_types"] == "vwap|eod_vwap"
    assert nvda["tier"] is None
    assert nvda["tier_match"] == "unmatched"


def test_m5_as_of_hides_the_final_row(sources):
    as_of = datetime(2026, 8, 4, 17, 0, tzinfo=UTC)  # 10:00 PT, before the 13:10 PT final
    alerts = rp.build_m5_alerts(sources.bounce_outcomes_csv, None, as_of=as_of)
    aapl = next(row for row in alerts if row["symbol"] == "AAPL")
    assert aapl["close_r_final"] is None
    assert aapl["final_status"] is None


# ---------------------------------------------------------------------------
# journal join
# ---------------------------------------------------------------------------
def _matched(sources):
    store = ResearchStore(sources.lake_root)
    occurrences, _ = rp.build_d1_tables(store, as_of=datetime(2026, 9, 1, tzinfo=UTC))
    alerts = rp.build_m5_alerts(sources.bounce_outcomes_csv, sources.bounces_csv, as_of=None)
    trades = rp.load_journal_trades(sources.journal_db, as_of=None)
    return {row["trade_id"]: row for row in rp.match_journal_trades(trades, occurrences, alerts)}


def test_journal_joins_the_next_session_setup_and_the_preceding_alert(sources):
    trades = _matched(sources)
    t1 = trades["t1"]
    assert t1["d1_occurrence_id"] == "o1"
    assert t1["d1_match"] == "next_session"
    assert t1["d1_family"] == "AVWAPE_TO_FIRST_DEV"
    assert t1["m5_event_id"].startswith("AAPL_long_20260804_07_00_00")
    assert t1["m5_match"] == "within_15m"
    assert t1["win"] is True
    assert "account_number" not in t1 and "account_label" not in t1


def test_opposite_side_stays_unmatched(sources):
    t2 = _matched(sources)["t2"]
    assert t2["d1_occurrence_id"] is None
    assert t2["d1_match"] == "unmatched_opposite_side"
    assert t2["m5_match"] == "unmatched"


def test_sold_put_matches_the_long_setup_on_its_underlying(sources):
    t3 = _matched(sources)["t3"]
    assert t3["underlying"] == "AA"
    assert t3["setup_side"] == "LONG"
    assert t3["d1_occurrence_id"] == "o3"
    assert t3["d1_match"] == "within_2_sessions"


def test_a_trade_on_the_trigger_session_is_not_matched(sources):
    t4 = _matched(sources)["t4"]
    assert t4["d1_occurrence_id"] is None
    assert t4["d1_match"] == "unmatched"


# ---------------------------------------------------------------------------
# floors
# ---------------------------------------------------------------------------
def test_cells_below_the_floor_are_never_ranked():
    cells = [
        {"source": "d1_setups", "recipe_id": "r", "trait": "all", "n_episodes": 50, "mean": 0.1},
        {"source": "d1_setups", "recipe_id": "r", "trait": "all", "n_episodes": 5, "mean": 9.9},
        {"source": "d1_setups", "recipe_id": "r", "trait": "all", "n_episodes": 40, "mean": 0.4},
    ]
    ranked = rp.rank_cells(cells, floor=30)
    assert [cell["rank"] for cell in ranked] == [2, None, 1]
    assert [cell["below_floor"] for cell in ranked] == [False, True, False]


def test_summary_has_counts_and_honours_the_floor(sources):
    store = ResearchStore(sources.lake_root)
    occurrences, _ = rp.build_d1_tables(store, as_of=datetime(2026, 9, 1, tzinfo=UTC))
    alerts = rp.build_m5_alerts(sources.bounce_outcomes_csv, sources.bounces_csv, as_of=None)
    cells = rp.summarize(occurrences, alerts, [], floor=1, journal_floor=1)
    d1_all = [c for c in cells if c["source"] == "d1_setups" and c["trait"] == "all" and c["family"] == "*"]
    assert d1_all and d1_all[0]["n"] == 3
    assert d1_all[0]["win_rate"] == pytest.approx(2 / 3)
    assert d1_all[0]["median"] == 1.0
    high_floor = rp.summarize(occurrences, alerts, [], floor=1000, journal_floor=1000)
    assert all(cell["rank"] is None for cell in high_floor)


# ---------------------------------------------------------------------------
# end to end, read-only
# ---------------------------------------------------------------------------
def _mtimes(paths):
    result = {}
    for path in paths:
        for item in ([path] if path.is_file() else sorted(path.rglob("*"))):
            if item.is_file():
                result[str(item)] = (item.stat().st_mtime_ns, item.stat().st_size)
    return result


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_export_writes_the_pack_and_touches_no_source(sources, tmp_path, fmt):
    watched = [sources.lake_root, sources.journal_db, sources.bounce_outcomes_csv, sources.bounces_csv]
    before = _mtimes(watched)
    out = tmp_path / f"pack_{fmt}"
    manifest = rp.export_pack(sources, out, fmt=fmt, floor=1, journal_floor=1)
    assert _mtimes(watched) == before
    assert not (sources.journal_db.parent / "trade_journal.sqlite3-journal").exists()
    for name in ("d1_occurrences", "d1_outcomes", "m5_alerts", "journal_trades"):
        assert manifest["tables"][name]["status"] == "ok"
        assert (out / manifest["tables"][name]["file"]).exists()
    assert (out / "setup_summary.csv").exists()
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved["tables"]["journal_trades"]["rows"] == 4
    assert saved["join"]["d1_matched"] == 2
    assert saved["join"]["m5_matched"] == 1
    assert "point_in_time" in saved
