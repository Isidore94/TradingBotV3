"""Mentor P8: the read-only Questrade book. No token = no request; a 401 is the importer's own
refresh (once, under its lock); only accounts / positions / balances GETs; a failure is None
with a reason and a 1-hour backoff; a happy path with two accounts. No network."""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_importers as ji  # noqa: E402
import questrade_positions as qp  # noqa: E402

NOW = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)
API = "https://api01.iq.questrade.com/"


class _Resp:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


def _pos(symbol, qty, avg, price, value):
    return {"symbol": symbol, "openQuantity": qty, "averageEntryPrice": avg, "currentPrice": price,
            "currentMarketValue": value, "securityType": "Stock"}


BOOK = {
    "v1/accounts": {"accounts": [{"type": "TFSA", "number": "111", "status": "Active"},
                                 {"type": "Margin", "number": "222", "status": "Active"}]},
    "v1/accounts/111/positions": {"positions": [_pos("NVDA", 50, 120.0, 125.0, 6250.0),
                                                _pos("OLD", 0, 10.0, 11.0, 0.0)]},
    "v1/accounts/222/positions": {"positions": [_pos("AMD", -100, 150.0, 148.0, -14800.0)]},
    "v1/accounts/111/balances": {"perCurrencyBalances": [{"currency": "CAD", "cash": 2500.0},
                                                        {"currency": "USD", "cash": 1200.0}]},
    "v1/accounts/222/balances": {"perCurrencyBalances": [{"currency": "USD", "cash": 900.5}]},
}


class _Session:
    """Answers the book paths; the first ``fail_first`` API calls return 401; records every call."""

    def __init__(self, fail_first=0, raise_on=None):
        self.calls: list[tuple[str, dict]] = []
        self.fail_first = fail_first
        self.raise_on = raise_on

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url == ji.QUESTRADE_LOGIN_URL:
            return _Resp({"access_token": "A2", "refresh_token": "R2", "api_server": API, "expires_in": 1800})
        path = url[len(API):]
        if self.raise_on and self.raise_on in path:
            raise ConnectionError("network down")
        if self.fail_first > 0:
            self.fail_first -= 1
            return _Resp(status=401)
        return _Resp(BOOK[path])

    def post(self, *a, **k):  # pragma: no cover - a POST is an order-shaped call: never
        raise AssertionError("the book never POSTs")

    put = delete = post


@contextlib.contextmanager
def _importer(monkeypatch, saved):
    monkeypatch.setattr(ji, "get_local_setting", lambda key, default="": saved.get(key, default))
    monkeypatch.setattr(ji, "save_local_setting", lambda key, value: saved.__setitem__(key, value))
    monkeypatch.setattr(ji, "save_local_settings", lambda values: saved.update(values))
    locks: list[str] = []

    @contextlib.contextmanager
    def lock(key, **kwargs):
        locks.append(key)
        yield

    monkeypatch.setattr(ji, "local_writer_lock", lock, raising=False)
    monkeypatch.delenv("QUESTRADE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("QUESTRADE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("QUESTRADE_API_SERVER", raising=False)
    yield locks


def _allowed(url: str) -> bool:
    if url == ji.QUESTRADE_LOGIN_URL:
        return True
    path = url[len(API):]
    parts = path.split("/")
    return path == "v1/accounts" or (len(parts) == 4 and parts[:2] == ["v1", "accounts"]
                                     and parts[3] in ("positions", "balances"))


def test_no_token_sends_nothing(monkeypatch):
    session = _Session()
    with _importer(monkeypatch, {}):
        snap, reason = qp.fetch_book(NOW, ji.QuestradeImporter(session=session))
    assert snap is None and reason == "no token" and session.calls == []


def test_happy_path_two_accounts(monkeypatch):
    saved = {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "R1", ji.QUESTRADE_ACCESS_TOKEN_SETTING: "A1",
             ji.QUESTRADE_API_SERVER_SETTING: API}
    session = _Session()
    with _importer(monkeypatch, saved) as locks:
        snap, reason = qp.fetch_book(NOW, ji.QuestradeImporter(session=session))
    assert reason == "" and snap is not None
    assert locks == [] and saved[ji.QUESTRADE_REFRESH_TOKEN_SETTING] == "R1", "a valid access token spends nothing"
    assert snap.fetched_utc == "2026-09-29T14:00:00+00:00" and datetime.fromisoformat(snap.fetched_utc).tzinfo
    accounts = {a["account_number"]: a for a in snap.accounts}
    assert accounts["111"]["account_type"] == "TFSA" and accounts["111"]["cash"] == {"CAD": 2500.0, "USD": 1200.0}
    assert accounts["222"]["cash"] == {"USD": 900.5} and accounts["222"]["cash_known"]
    by_sym = {p["symbol"]: p for p in snap.positions}
    assert set(by_sym) == {"NVDA", "AMD"}, "a flat (0 qty) row is not a position"
    assert by_sym["NVDA"] == {**by_sym["NVDA"], "side": "LONG", "open_qty": 50.0, "avg_price": 120.0,
                              "current_price": 125.0, "market_value": 6250.0, "account_type": "TFSA"}
    assert by_sym["AMD"]["side"] == "SHORT" and by_sym["AMD"]["open_qty"] == 100.0
    assert all(_allowed(url) for url, _ in session.calls) and ji.QUESTRADE_LOGIN_URL not in [u for u, _ in session.calls]
    assert qp.BookSnapshot.from_json(snap.as_json()) == snap


def test_a_401_is_the_importers_own_single_refresh_under_its_lock(monkeypatch):
    saved = {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "R1", ji.QUESTRADE_ACCESS_TOKEN_SETTING: "A1",
             ji.QUESTRADE_API_SERVER_SETTING: API}
    session = _Session(fail_first=1)
    with _importer(monkeypatch, saved) as locks:
        snap, reason = qp.fetch_book(NOW, ji.QuestradeImporter(session=session))
    assert snap is not None and reason == ""
    refreshes = [url for url, _ in session.calls if url == ji.QUESTRADE_LOGIN_URL]
    assert len(refreshes) == 1 and locks == [ji.QUESTRADE_REFRESH_LOCK_KEY], "one refresh, under the importer's lock"
    assert saved[ji.QUESTRADE_REFRESH_TOKEN_SETTING] == "R2" and saved[ji.QUESTRADE_ACCESS_TOKEN_SETTING] == "A2"
    assert all(_allowed(url) for url, _ in session.calls)


def test_a_failure_is_none_with_a_reason_and_a_one_hour_backoff(monkeypatch):
    saved = {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "R1", ji.QUESTRADE_ACCESS_TOKEN_SETTING: "A1",
             ji.QUESTRADE_API_SERVER_SETTING: API}
    session = _Session(raise_on="222/positions")
    with _importer(monkeypatch, saved):
        snap, reason = qp.fetch_book(NOW, ji.QuestradeImporter(session=session))
    assert snap is None and "network down" in reason
    failed = NOW.isoformat()
    assert qp.backoff_left(failed, NOW + timedelta(minutes=59)) is not None
    assert qp.backoff_left(failed, NOW + timedelta(hours=1)) is None
    assert qp.backoff_left(None, NOW) is None


def test_a_balances_failure_is_cash_unknown_not_a_failed_book(monkeypatch):
    saved = {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "R1", ji.QUESTRADE_ACCESS_TOKEN_SETTING: "A1",
             ji.QUESTRADE_API_SERVER_SETTING: API}
    session = _Session(raise_on="111/balances")
    with _importer(monkeypatch, saved):
        snap, _ = qp.fetch_book(NOW, ji.QuestradeImporter(session=session))
    tfsa = {a["account_number"]: a for a in snap.accounts}["111"]
    assert tfsa["cash"] is None and not tfsa["cash_known"] and snap.errors


def test_freshness_is_15_minutes():
    snap = qp.fixture_snapshot(NOW.isoformat())
    assert snap.is_fresh(NOW + timedelta(minutes=15)) and not snap.is_fresh(NOW + timedelta(minutes=16))


def test_the_module_never_names_an_order_endpoint_or_refreshes_itself():
    source = (SCRIPTS_DIR / "questrade_positions.py").read_text(encoding="utf-8")
    for word in ("orders", "refresh_access_token", ".post(", "QUESTRADE_LOGIN_URL", "save_local_setting"):
        assert word not in source, word
    assert json.dumps(qp.ALLOWED_PATHS)
