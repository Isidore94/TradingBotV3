"""The Questrade refresh chain has ONE owner (2026-08-25, trader-directed).

Questrade rotates on every refresh: a successful refresh invalidates both the
access token it replaces AND the refresh token it consumed. So two consumers -
"Pull today now", the gap backfill, the nightly slot - are enough to break the
chain, and the desk's own log shows exactly that: a Questrade import OK at
20:54:59, a year-wide backfill at 20:59, then `400 Bad Request` on the refresh
endpoint at 21:06:51. The trader had pasted a fresh token eleven minutes
earlier.

What the tests below pin is that a refresh is SERIALIZED, that it re-reads the
token inside the lock, and that a 401 caused by somebody else's rotation is
answered by picking up their new access token rather than by spending another
one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_importers as ji  # noqa: E402


class _Response:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


def _settings(monkeypatch, saved):
    monkeypatch.setattr(ji, "get_local_setting", lambda key, default="": saved.get(key, default))
    monkeypatch.setattr(ji, "save_local_setting", lambda key, value: saved.__setitem__(key, value))
    if hasattr(ji, "save_local_settings"):
        monkeypatch.setattr(ji, "save_local_settings", lambda values: saved.update(values))
    return saved


def test_a_refresh_is_serialized_by_the_machine_local_writer_lock(monkeypatch):
    """Two consumers refreshing at once is the whole defect. The lock is the
    same primitive the outcome finalizer already uses."""
    taken: list[str] = []
    saved = _settings(monkeypatch, {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "T1"})

    import contextlib

    @contextlib.contextmanager
    def _fake_lock(key, **kwargs):
        taken.append(key)
        yield

    monkeypatch.setattr(ji, "local_writer_lock", _fake_lock, raising=False)

    class _Session:
        def get(self, *a, **k):
            return _Response({"access_token": "A2", "refresh_token": "T2",
                              "api_server": "https://api.example/", "expires_in": 1800})

    ji.QuestradeImporter(session=_Session()).refresh_access_token()

    assert taken, "the refresh must hold the machine-local writer lock"
    assert saved[ji.QUESTRADE_REFRESH_TOKEN_SETTING] == "T2"


def test_the_token_is_re_read_inside_the_lock(monkeypatch):
    """A consumer that waited on the lock must spend the token the winner
    LEFT, not the one it read before waiting - spending a consumed token is
    exactly the 400 the trader kept hitting."""
    saved = _settings(monkeypatch, {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "STALE"})
    sent: list[str] = []

    import contextlib

    @contextlib.contextmanager
    def _rotating_lock(key, **kwargs):
        # While we waited, the other consumer rotated the chain.
        saved[ji.QUESTRADE_REFRESH_TOKEN_SETTING] = "FRESH"
        yield

    monkeypatch.setattr(ji, "local_writer_lock", _rotating_lock, raising=False)

    class _Session:
        def get(self, url, params=None, **k):
            sent.append((params or {}).get("refresh_token"))
            return _Response({"access_token": "A9", "refresh_token": "T9",
                              "api_server": "https://api.example/", "expires_in": 1800})

    ji.QuestradeImporter(session=_Session()).refresh_access_token()

    assert sent == ["FRESH"], f"the refresh spent {sent}, not the token left inside the lock"


def test_a_401_from_someone_elses_rotation_reuses_their_token(monkeypatch):
    """The rotation cascade. Another consumer refreshed, which killed OUR access
    token; the 401 that follows must be answered with THEIR new access token,
    not by burning another refresh - burning one is what snaps the chain."""
    saved = _settings(monkeypatch, {
        ji.QUESTRADE_REFRESH_TOKEN_SETTING: "T1",
        ji.QUESTRADE_ACCESS_TOKEN_SETTING: "A1",
        ji.QUESTRADE_API_SERVER_SETTING: "https://api.example/",
    })
    refreshes: list[int] = []

    class _Session:
        def __init__(self):
            self.calls = 0

        def get(self, url, headers=None, params=None, **k):
            if "oauth2/token" in url:
                refreshes.append(1)
                return _Response({"access_token": "A3", "refresh_token": "T3",
                                  "api_server": "https://api.example/", "expires_in": 1800})
            self.calls += 1
            token = (headers or {}).get("Authorization", "")
            if self.calls == 1:
                # Our A1 was invalidated by the other consumer's refresh, and
                # by now the settings already hold their A2.
                saved[ji.QUESTRADE_ACCESS_TOKEN_SETTING] = "A2"
                return _Response(status=401)
            assert token == "Bearer A2", token
            return _Response({"accounts": []})

    importer = ji.QuestradeImporter(session=_Session())
    importer.get_accounts()

    assert refreshes == [], "a 401 explained by someone else's rotation must not spend a refresh"


def test_the_rotation_is_written_in_one_save_not_four(monkeypatch):
    """Four read-modify-write cycles over one JSON file is four chances for a
    concurrent writer to drop the new refresh token on the floor."""
    saved = {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "T1"}
    writes: list[object] = []
    monkeypatch.setattr(ji, "get_local_setting", lambda key, default="": saved.get(key, default))
    monkeypatch.setattr(ji, "save_local_setting",
                        lambda key, value: (writes.append(key), saved.__setitem__(key, value)))
    monkeypatch.setattr(ji, "save_local_settings",
                        lambda values: (writes.append("BATCH"), saved.update(values)), raising=False)

    class _Session:
        def get(self, *a, **k):
            return _Response({"access_token": "A2", "refresh_token": "T2",
                              "api_server": "https://api.example/", "expires_in": 1800})

    ji.QuestradeImporter(session=_Session()).refresh_access_token()

    assert writes == ["BATCH"], f"the rotation wrote {writes}"


def test_a_failed_refresh_leaves_the_stored_token_alone(monkeypatch):
    """Missing data is uncertainty: a rejected refresh must not clear the token
    the trader pasted, or the repair action becomes 'paste it again'."""
    saved = _settings(monkeypatch, {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "T1"})

    class _Session:
        def get(self, *a, **k):
            return _Response(status=400)

    with pytest.raises(Exception):
        ji.QuestradeImporter(session=_Session()).refresh_access_token()

    assert saved[ji.QUESTRADE_REFRESH_TOKEN_SETTING] == "T1"
