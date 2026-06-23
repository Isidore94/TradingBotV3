import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_collect_earnings_dates_skips_sleep_and_network_on_warm_cache(monkeypatch):
    from master_avwap_lib import legacy

    today = datetime.now().date()
    lookback = 6
    warm_cache = {}
    for delta in range(lookback):
        ds = (today - timedelta(days=delta)).isoformat()
        warm_cache[ds] = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "row_count": 1,
            "rows": [{"symbol": "XYZ"}],
        }
    monkeypatch.setattr(legacy, "_EARNINGS_CALENDAR_ROWS_CACHE", warm_cache)

    sleeps: list[float] = []
    monkeypatch.setattr(legacy.time, "sleep", lambda seconds: sleeps.append(seconds))

    def _no_network(*args, **kwargs):
        raise AssertionError("network fetch attempted despite a warm cache")

    monkeypatch.setattr(legacy.requests, "get", _no_network)
    monkeypatch.setattr(legacy, "_record_shared_nasdaq_rows_safely", lambda *a, **k: None)
    monkeypatch.setattr(legacy, "_save_earnings_calendar_rows_cache", lambda: None)

    result = legacy.collect_earnings_dates(["XYZ"], lookback_days=lookback, base_sleep=0.15)

    assert result["XYZ"]  # dates recovered straight from cache
    assert sleeps == []  # the key fix: a warm walk no longer idle-sleeps per day


def test_collect_earnings_dates_still_rate_limits_real_fetches(monkeypatch):
    from master_avwap_lib import legacy

    monkeypatch.setattr(legacy, "_EARNINGS_CALENDAR_ROWS_CACHE", {})
    sleeps: list[float] = []
    monkeypatch.setattr(legacy.time, "sleep", lambda seconds: sleeps.append(seconds))

    calls: list[str] = []

    def fake_fetch(date_str):
        calls.append(date_str)
        return [{"symbol": "XYZ"}]

    result = legacy.collect_earnings_dates(
        ["XYZ"], fetch_fn=fake_fetch, lookback_days=4, base_sleep=0.01
    )

    assert result["XYZ"]
    assert len(calls) == 4
    assert len(sleeps) == 4  # uncached fetches still sleep to stay polite to the API


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": {"rows": [{"symbol": "XYZ"}]}}


def test_collect_earnings_dates_batches_cache_writes_on_cold_walk(monkeypatch):
    from master_avwap_lib import legacy

    lookback = 120
    monkeypatch.setattr(legacy, "_EARNINGS_CALENDAR_ROWS_CACHE", {})  # cold cache
    monkeypatch.setattr(legacy, "_EARNINGS_CALENDAR_SAVE_DEFERRED", False)
    monkeypatch.setattr(legacy, "_EARNINGS_CALENDAR_PENDING_WRITES", 0)
    monkeypatch.setattr(legacy.time, "sleep", lambda *_: None)
    monkeypatch.setattr(legacy, "_record_shared_nasdaq_rows_safely", lambda *a, **k: None)
    monkeypatch.setattr(legacy.requests, "get", lambda *a, **k: _FakeResponse())

    writes = {"count": 0}
    monkeypatch.setattr(legacy, "save_json", lambda *a, **k: writes.__setitem__("count", writes["count"] + 1))

    result = legacy.collect_earnings_dates(["XYZ"], lookback_days=lookback, base_sleep=0)

    flush_every = legacy._EARNINGS_CALENDAR_FLUSH_EVERY
    expected_writes = lookback // flush_every + (1 if lookback % flush_every else 0)
    assert result["XYZ"]  # every cold date fetched and recorded
    assert writes["count"] == expected_writes  # batched, not one write per date
    assert writes["count"] < lookback  # the win: far fewer full-file rewrites
    assert len(legacy._EARNINGS_CALENDAR_ROWS_CACHE) == lookback  # all dates persisted in memory
