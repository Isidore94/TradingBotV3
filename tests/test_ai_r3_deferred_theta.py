"""AI-R3: theta evidence is recorded after the deferred quote attempt."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_deferred_theta_enrichment_records_enriched_rows_even_when_a_newer_report_wins(monkeypatch):
    """The latest-report guard cannot erase the scan's observed pick evidence."""
    from master_avwap_lib import runner

    recorded: list[tuple[list[dict], list[dict], object, object]] = []
    observed_at = datetime(2026, 9, 22, 16, 7, 3)
    monkeypatch.setattr(runner, "datetime", type("Clock", (), {"now": staticmethod(lambda: observed_at)}))
    monkeypatch.setattr(runner, "ensure_theta_option_data_client", lambda _client: (object(), False))
    monkeypatch.setattr(runner, "disconnect_daily_data_client", lambda *_args: None)
    monkeypatch.setattr(runner, "_theta_enrichment_run_is_latest", lambda _run_id: False)
    monkeypatch.setattr(runner, "write_theta_put_report", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale run wrote report")))
    monkeypatch.setattr(runner, "save_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale run wrote AI state")))

    def enrich(_client, sold_puts, spreads, _reference_date):
        sold_puts[0]["best_option"] = {"strike": 95.0, "credit": 1.15, "expiration": "20261016"}
        spreads[0]["best_option"] = {"short_strike": 90.0, "long_strike": 85.0, "credit": 0.72, "expiration": "20261016"}

    monkeypatch.setattr(runner, "enrich_theta_rows_with_ib_option_premiums", enrich)
    monkeypatch.setattr(
        runner,
        "record_theta_picks",
        lambda puts, spreads, scan_date, now: recorded.append((deepcopy(puts), deepcopy(spreads), scan_date, now)),
    )

    runner._run_deferred_theta_enrichment(
        run_id="older-scan",
        theta_put_rows=[{"symbol": "PUT", "last_close": 101.0}],
        theta_pcs_rows=[{"symbol": "PCS", "last_close": 101.0}],
        priority_rows=[],
        ai_state={"symbols": {}},
        sorted_events=[],
        range_buckets={},
        reference_date=date(2026, 9, 21),
        favorite_watchlist_context={"allowed": False},
    )

    assert len(recorded) == 1
    puts, spreads, scan_date, now = recorded[0]
    assert scan_date == date(2026, 9, 21)
    assert now == observed_at
    assert puts[0]["best_option"]["strike"] == 95.0
    assert spreads[0]["best_option"] == {
        "short_strike": 90.0,
        "long_strike": 85.0,
        "credit": 0.72,
        "expiration": "20261016",
    }


def test_deferred_theta_quote_failure_still_records_the_unmeasured_pick(monkeypatch):
    """A quote outage loses a quote, never the observed scan candidate."""
    from master_avwap_lib import runner

    recorded: list[tuple[list[dict], list[dict], object, object]] = []
    observed_at = datetime(2026, 9, 22, 16, 9, 4)
    monkeypatch.setattr(runner, "datetime", type("Clock", (), {"now": staticmethod(lambda: observed_at)}))
    monkeypatch.setattr(runner, "ensure_theta_option_data_client", lambda _client: (object(), False))
    monkeypatch.setattr(runner, "disconnect_daily_data_client", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "enrich_theta_rows_with_ib_option_premiums",
        lambda *_args: (_ for _ in ()).throw(ConnectionError("quote endpoint unavailable")),
    )
    monkeypatch.setattr(
        runner,
        "record_theta_picks",
        lambda puts, spreads, scan_date, now: recorded.append((deepcopy(puts), deepcopy(spreads), scan_date, now)),
    )
    monkeypatch.setattr(runner, "write_theta_put_report", lambda *_args, **_kwargs: pytest.fail("quote failure wrote report"))

    runner._run_deferred_theta_enrichment(
        run_id="quote-failure",
        theta_put_rows=[{"symbol": "PUT", "last_close": 101.0}],
        theta_pcs_rows=[],
        priority_rows=[],
        ai_state={"symbols": {}},
        sorted_events=[],
        range_buckets={},
        reference_date=date(2026, 9, 21),
        favorite_watchlist_context={"allowed": False},
    )

    assert len(recorded) == 1
    puts, spreads, scan_date, now = recorded[0]
    assert puts == [{"symbol": "PUT", "last_close": 101.0}]
    assert spreads == []
    assert scan_date == date(2026, 9, 21)
    assert now == observed_at
