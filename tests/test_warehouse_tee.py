"""M5 tee archive, scan coverage, and collection gaps (plan Phase 3).

Exit criteria pinned here:

* the tee adds **zero** provider requests - it reads BounceBot's in-memory
  ``latest_bars`` cache and contains no provider client at all (risk R3; the
  live request-count assertion is pilot item 1, which needs IB);
* coverage rows reconcile against the run manifest that produced them;
* absence is recorded with the right reason - ``NOT_COLLECTED_BY_POLICY`` is
  never conflated with ``MISSING``;
* only completed bars are archived, and re-teeing the same cache is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import bar_archive as tee
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
# 2026-08-03 is a Monday; 13:30 UTC is the 09:30 ET open.
OPEN_UTC = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)


@dataclass(frozen=True)
class FakeIbBar:
    """Shaped like the champion's IbBar: naive market-local dt."""

    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


@pytest.fixture()
def session():
    return tee.SessionContext(
        session_id="XNYS-2026-08-03",
        session_date=OPEN_UTC.date(),
        rth_open_at=OPEN_UTC,
        rth_close_at=OPEN_UTC + timedelta(hours=6, minutes=30),
        market_timezone="America/New_York",
    )


def _bars(count, *, start_offset=0, symbol_base=100.0):
    return [
        FakeIbBar(
            dt=(OPEN_UTC + timedelta(minutes=5 * (index + start_offset))).replace(tzinfo=None),
            open=symbol_base + index,
            high=symbol_base + index + 1,
            low=symbol_base + index - 1,
            close=symbol_base + index + 0.5,
            volume=1000 + index,
        )
        for index in range(count)
    ]


def _capture(store, session, cohort, *, now, **kwargs):
    return tee.capture_m5_tee(
        store,
        cohort,
        now=now,
        session=session,
        market_tz=UTC,  # the fixture bars are UTC-naive
        **kwargs,
    )


def test_tee_has_no_provider_client_at_all():
    """The zero-added-requests property is structural, not a runtime promise.

    Read from the parsed module, not the text, so prose about the providers is
    allowed while an actual import or request call is not.
    """
    import ast

    tree = ast.parse(open(tee.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"ibapi", "yfinance", "requests", "urllib", "http", "socket", "curl_cffi"}

    called = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    }
    assert not called & {"reqHistoricalData", "reqRealTimeBars", "reqMktData", "download", "history"}


def test_tee_reads_the_champion_cache_without_touching_provider_counters(store, session, monkeypatch):
    from scripts.diagnostics import provider_counters

    provider_counters.reset()
    provider_counters.begin_run()
    latest_bars = {
        "AAPL|5 D|5 mins": _bars(3),
        "MSFT|5 D|5 mins": _bars(3, symbol_base=400.0),
        "AAPL": _bars(3),  # the champion's alias key, not an M5 cohort key
        "SPY|5 D|1 hour": _bars(3),  # a different timeframe cache
    }
    before = dict(latest_bars)

    report = _capture(store, session, latest_bars, now=OPEN_UTC + timedelta(minutes=20))

    assert report.symbols == 2 and report.rows_published == 6
    # No provider boundary was crossed: the counters bucket stays empty.
    assert provider_counters.snapshot() == {}
    # The champion's cache is untouched.
    assert latest_bars == before
    provider_counters.reset()


def test_extract_tee_bars_selects_only_the_m5_cohort():
    cohort = tee.extract_tee_bars(
        {
            "AAPL|5 D|5 mins": [1],
            "spy|5 D|5 mins": [1],
            "AAPL|5 D|1 hour": [1],
            "AAPL": [1],
            "TSLA|5 D|5 mins": [],
        }
    )
    assert sorted(cohort) == ["AAPL", "SPY"]


def test_only_completed_bars_are_archived(store, session):
    now = OPEN_UTC + timedelta(minutes=12)  # 09:42 ET: the 09:40 bar is forming
    report = _capture(store, session, {"AAPL": _bars(4)}, now=now)

    assert report.rows_published == 2 and report.forming_skipped == 2
    rows = store.read_table("bar_m5").to_pylist()
    assert [row["interval_start"] for row in rows] == [OPEN_UTC, OPEN_UTC + timedelta(minutes=5)]
    assert {row["is_complete"] for row in rows} == {True}
    assert {row["interval_end"] - row["interval_start"] for row in rows} == {timedelta(minutes=5)}


def test_re_teeing_the_same_cache_is_a_no_op(store, session):
    cohort = {"AAPL": _bars(3)}
    now = OPEN_UTC + timedelta(minutes=20)
    first = _capture(store, session, cohort, now=now)
    manifest_before = store.manifest.path.read_bytes()

    second = _capture(store, session, cohort, now=now)

    assert first.rows_published == 3
    assert second.rows_published == 0 and second.duplicates_skipped == 3
    assert second.status == "NOTHING_TO_CAPTURE"
    assert store.manifest.path.read_bytes() == manifest_before
    assert store.read_table("bar_m5").num_rows == 3

    # A later cycle archives only the bars that have since completed.
    third = _capture(store, session, {"AAPL": _bars(5)}, now=OPEN_UTC + timedelta(minutes=30))
    assert third.rows_published == 2 and third.duplicates_skipped == 3
    assert store.read_table("bar_m5").num_rows == 5


def test_captured_rows_carry_session_phase_and_provenance(store, session):
    cohort = {"AAPL": _bars(1)}
    _capture(store, session, cohort, now=OPEN_UTC + timedelta(minutes=10), run_id="cycle-1")
    row = store.read_table("bar_m5").to_pylist()[0]

    assert row["session_id"] == "XNYS-2026-08-03"
    assert row["session_phase"] == "RTH"
    assert row["provider"] == "IBKR" and row["capture_mode"] == "LIVE"
    assert row["quality"] == "COMPLETE"
    assert row["event_at"] == row["interval_end"]
    assert row["observed_at"] == OPEN_UTC + timedelta(minutes=10)
    assert len(row["source_hash"]) == 64
    assert row["run_id"] == "cycle-1"
    # Volume is stored exactly as the provider gave it (the IB round-lot
    # difference is a sentinel check against `provider`, never a rewrite).
    assert row["volume"] == 1000


def test_premarket_and_postmarket_bars_are_phase_tagged(store, session):
    pre = FakeIbBar(dt=(OPEN_UTC - timedelta(minutes=30)).replace(tzinfo=None), open=1, high=2, low=0.5, close=1.5)
    post = FakeIbBar(
        dt=(session.rth_close_at + timedelta(minutes=5)).replace(tzinfo=None), open=1, high=2, low=0.5, close=1.5
    )
    _capture(store, session, {"AAPL": [pre, post]}, now=session.rth_close_at + timedelta(hours=1))

    phases = {row["session_phase"] for row in store.read_table("bar_m5").to_pylist()}
    assert phases == {"PRE", "POST"}


def test_unreadable_bars_are_skipped_not_guessed(store, session):
    good = _bars(1)[0]
    broken = FakeIbBar(dt=None, open=1.0, high=2.0, low=0.5, close=1.5)
    no_price = {"dt": OPEN_UTC.replace(tzinfo=None), "open": None, "high": 2.0, "low": 1.0, "close": 1.5}

    report = _capture(store, session, {"AAPL": [good, broken, no_price]}, now=OPEN_UTC + timedelta(minutes=20))

    assert report.rows_published == 1 and report.unparsable_skipped == 2


def test_tee_is_inert_when_the_warehouse_is_disabled(session):
    report = tee.capture_m5_tee(None, {"AAPL": _bars(3)}, now=OPEN_UTC, session=session, market_tz=UTC)
    assert report.status == "DISABLED" and report.rows_published == 0


def test_live_capture_spools_and_the_build_job_seals(store, session, tmp_path):
    """The live path never writes the lake: GUI spools, CLI seals (sec 8.4)."""
    from scripts.research_warehouse.spool import ResearchSpoolWriter, seal_spool

    writer = ResearchSpoolWriter(tmp_path / "spool")
    seen: set = set()

    first = _capture(
        store, session, {"AAPL": _bars(3)}, now=OPEN_UTC + timedelta(minutes=20), spool=writer, seen=seen
    )
    assert first.status == "SPOOLED" and first.rows_published == 3
    assert store.read_table("bar_m5").num_rows == 0  # nothing in the lake yet

    # A second cycle over the same in-memory cache spools nothing new, even
    # though the lake cannot be consulted for de-duplication.
    second = _capture(
        store, session, {"AAPL": _bars(3)}, now=OPEN_UTC + timedelta(minutes=25), spool=writer, seen=seen
    )
    assert second.rows_published == 0 and second.duplicates_skipped == 3

    writer.roll()
    result = seal_spool(store, writer.dir)
    assert result.rows_published == 3
    assert store.read_table("bar_m5").num_rows == 3


def test_dict_shaped_bars_are_accepted(store, session):
    rows = [
        {"time": "20260803  09:30:00", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10},
        {"dt": OPEN_UTC + timedelta(minutes=5), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 11},
    ]
    report = _capture(store, session, {"AAPL": rows}, now=OPEN_UTC + timedelta(minutes=20))
    assert report.rows_published == 2


# --- scan_coverage ---------------------------------------------------------
def _manifest(run_id="master_scan-20260803T134500Z-abc123", symbols=3):
    return {
        "schema": "run_manifest_v1",
        "run_id": run_id,
        "job_type": "master_scan",
        "started_at": "2026-08-03T13:45:00+00:00",
        "status": "ok",
        "counters": {
            "symbols_processed": symbols,
            "provider.daily_bars.lookup": 12,
            "provider.intraday_bars.lookup": 4,
        },
    }


def test_coverage_rows_reconcile_against_the_run_manifest(store):
    manifest = _manifest()
    context = tee.coverage_context_from_manifest(manifest)
    assert context["risk_set_id"] == manifest["run_id"]
    assert context["run_kind"] == "master_scan"

    report = tee.record_scan_coverage(
        store,
        {
            "AAPL": {"scan_status": "EVALUATED_ELIGIBLE", "family_status_map": '{"AVWAPE_TO_FIRST_DEV": "TRIGGERED"}'},
            "MSFT": {"scan_status": "EVALUATED_INELIGIBLE"},
            "NVDA": {"scan_status": "NO_RESPONSE"},
        },
        risk_set_id=context["risk_set_id"],
        run_kind=context["run_kind"],
        scheduled_at=context["scheduled_at"],
        provider="IBKR",
        bar_source="ibkr_hist",
    )
    assert report.rows == 3

    outcome = tee.reconcile_scan_coverage(store, manifest)
    assert outcome["matched"] is True
    assert outcome["coverage_rows"] == 3 and outcome["manifest_symbols"] == 3
    assert outcome["provider_lookups"] == 16 and outcome["reason"] == ""


def test_reconciliation_reports_a_mismatch_instead_of_repairing_it(store):
    manifest = _manifest(symbols=5)
    context = tee.coverage_context_from_manifest(manifest)
    tee.record_scan_coverage(
        store,
        {"AAPL": "EVALUATED_ELIGIBLE"},
        risk_set_id=context["risk_set_id"],
        run_kind=context["run_kind"],
        scheduled_at=context["scheduled_at"],
    )
    outcome = tee.reconcile_scan_coverage(store, manifest)

    assert outcome["matched"] is False
    assert "1 coverage rows vs 5" in outcome["reason"]
    assert store.read_table("scan_coverage").num_rows == 1  # nothing invented


def test_unevaluated_symbols_keep_their_own_status(store):
    context = tee.coverage_context_from_manifest(_manifest())
    tee.record_scan_coverage(
        store,
        {"AAPL": "NOT_ASSIGNED", "MSFT": "TIMED_OUT", "NVDA": "PARTIAL_DATA"},
        risk_set_id=context["risk_set_id"],
        run_kind=context["run_kind"],
        scheduled_at=context["scheduled_at"],
    )
    statuses = {row["symbol"]: row["scan_status"] for row in store.read_table("scan_coverage").to_pylist()}
    assert statuses == {"AAPL": "NOT_ASSIGNED", "MSFT": "TIMED_OUT", "NVDA": "PARTIAL_DATA"}


def test_coverage_rerun_does_not_duplicate_a_risk_set(store):
    context = tee.coverage_context_from_manifest(_manifest())
    kwargs = dict(
        risk_set_id=context["risk_set_id"],
        run_kind=context["run_kind"],
        scheduled_at=context["scheduled_at"],
    )
    tee.record_scan_coverage(store, {"AAPL": "EVALUATED_ELIGIBLE"}, **kwargs)
    again = tee.record_scan_coverage(store, {"AAPL": "EVALUATED_ELIGIBLE"}, **kwargs)
    assert again.status == "ALREADY_RECORDED" and store.read_table("scan_coverage").num_rows == 1


# --- collection_gap --------------------------------------------------------
def test_policy_absence_is_never_recorded_as_missing(store, session):
    report = tee.record_collection_gaps(
        store,
        session=session,
        captured_counts={"AAPL": 78, "MSFT": 40, "NVDA": 0},
        policy_symbols=["AAPL", "TSLA", "AMD"],
        detected_at=session.rth_close_at,
    )
    assert report.rows == 4
    rows = {row["symbol"]: row for row in store.read_table("collection_gap").to_pylist()}

    assert "AAPL" not in rows  # complete coverage: no gap row at all
    assert rows["MSFT"]["reason"] == "PARTIAL" and rows["MSFT"]["expected_bars"] == 38
    assert rows["NVDA"]["reason"] == "MISSING" and rows["NVDA"]["expected_bars"] == 78
    for symbol in ("TSLA", "AMD"):
        assert rows[symbol]["reason"] == "NOT_COLLECTED_BY_POLICY"
        assert rows[symbol]["expected_bars"] == 78
    assert report.by_reason == {"PARTIAL": 1, "MISSING": 1, "NOT_COLLECTED_BY_POLICY": 2}
    # Gaps span the session, and are unresolved until a backfill closes them.
    assert {row["gap_start"] for row in rows.values()} == {session.rth_open_at}
    assert {row["resolution"] for row in rows.values()} == {None}


def test_gap_recording_is_idempotent_per_session(store, session):
    kwargs = dict(session=session, captured_counts={"NVDA": 0}, policy_symbols=["TSLA"])
    tee.record_collection_gaps(store, **kwargs)
    again = tee.record_collection_gaps(store, **kwargs)
    assert again.status == "NO_GAPS" and store.read_table("collection_gap").num_rows == 2


def test_captured_bar_counts_reconcile_the_session(store, session):
    _capture(store, session, {"AAPL": _bars(6), "MSFT": _bars(2, symbol_base=50.0)}, now=OPEN_UTC + timedelta(hours=1))
    counts = tee.captured_bar_counts(store, session, symbols=["AAPL", "MSFT", "NEVER"])

    assert counts == {"AAPL": 6, "MSFT": 2, "NEVER": 0}
    gaps = tee.record_collection_gaps(store, session=session, captured_counts=counts, expected_bars=6)
    reasons = {row["symbol"]: row["reason"] for row in store.read_table("collection_gap").to_pylist()}
    assert gaps.rows == 2 and reasons == {"MSFT": "PARTIAL", "NEVER": "MISSING"}


def test_session_context_wraps_the_champion_session_helper():
    context = tee.session_context(datetime(2026, 8, 3, 17, 0, tzinfo=UTC))
    assert context.session_id == "XNYS-2026-08-03"
    assert context.rth_close_at - context.rth_open_at == timedelta(hours=6, minutes=30)
    assert context.phase_of(context.rth_open_at - timedelta(minutes=1)) == "PRE"
    assert context.phase_of(context.rth_open_at) == "RTH"
    assert context.phase_of(context.rth_close_at) == "POST"
