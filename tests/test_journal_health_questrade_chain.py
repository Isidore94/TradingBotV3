"""AI-P4 - the Questrade credential chain needs a surface, not a log line.

Review 2026-08-24 §5. On the day this was written the chain had been dead since
2026-08-19: 0 of 142 Questrade session days covered, 56 identical
``oauth2/token`` failures, one whole broker (including a TFSA) absent from the
journal - and nothing on the desk said so. It was found by opening the SQLite
database by hand.

Every test here is offline: an injected settings reader and a temporary
database. Nothing touches the network, the live store, or Qt.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_health  # noqa: E402

NOW = datetime(2026, 8, 24, 10, 0, 0)


def _settings(**values):
    def read(key, default=""):
        return values.get(key, default)

    return read


def _db(tmp_path: Path, failures=()) -> Path:
    path = tmp_path / "trade_journal.sqlite3"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS import_coverage (broker TEXT, "
            "account_number TEXT, day TEXT, status TEXT, message TEXT, "
            "updated_at TEXT)"
        )
        for day, message in failures:
            conn.execute(
                "INSERT INTO import_coverage VALUES ('QUESTRADE', '51830546', "
                "?, 'FAILED', ?, ?)",
                (day, message, f"{day}T00:30:00"),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def _health(tmp_path, *, failures=(), **settings):
    return journal_health.questrade_chain_health(
        now=NOW,
        db_path=_db(tmp_path, failures),
        settings_reader=_settings(**settings),
    )


# --- the three states the review demanded be kept apart --------------------


def test_a_dead_chain_is_visible_and_says_how_to_fix_it(tmp_path):
    """The live 2026-08-24 shape, reproduced."""
    verdict = _health(
        tmp_path,
        failures=[
            (
                "2026-08-21",
                "500 Server Error: Internal Server Error for url: "
                "https://login.questrade.com/oauth2/token?grant_type=refresh_token"
                "&refresh_token=va8Ope",
            )
        ],
        journal_questrade_refresh_token="token",
        journal_questrade_expires_at="2026-08-22T00:30:04",
    )

    assert verdict["state"] == journal_health.STATE_DEAD
    assert verdict["state"] in journal_health.ALERTING_STATES
    # The headline names when the ATTEMPT failed, not the trading day it left
    # uncovered. One broken chain marks a year of days FAILED, so naming the
    # coverage day reports a 2025 outage for a chain that broke last week -
    # which is exactly how this read on its first run against the live store.
    assert "2026-08-21T00:30:00" in verdict["headline"]
    assert verdict["last_failure_at"] == "2026-08-21T00:30:00"
    assert verdict["last_failure_day"] == "2026-08-21"
    # The trader must not have to work out what to do about it.
    assert "Journal > Health" in verdict["action"]
    assert "single-use" in verdict["action"]


def test_an_absent_setting_is_not_configured_and_never_healthy(tmp_path):
    """"Nobody set this up" and "this is working" must never render the same.

    A machine with no Questrade token has been MEASURED, and the answer is that
    the broker was never asked for - which is why this is not an alert either.
    """
    verdict = _health(tmp_path, journal_questrade_refresh_token="")

    assert verdict["state"] == journal_health.STATE_NOT_CONFIGURED
    assert verdict["state"] not in journal_health.ALERTING_STATES
    assert verdict["state"] != journal_health.STATE_OK
    assert not verdict["action"]


def test_an_unreadable_database_is_unknown_not_green(tmp_path, monkeypatch):
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""

    def _explode(_path):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(journal_health, "_last_oauth_failure", _explode)
    verdict = _health(
        tmp_path,
        journal_questrade_refresh_token="token",
        journal_questrade_expires_at="2026-08-24T09:00:00",
    )

    assert verdict["state"] == journal_health.STATE_UNKNOWN
    assert verdict["state"] != journal_health.STATE_OK
    assert "UNKNOWN" in verdict["headline"]
    assert "not therefore fine" in verdict["headline"]


def test_a_live_chain_reads_ok(tmp_path):
    verdict = _health(
        tmp_path,
        journal_questrade_refresh_token="token",
        journal_questrade_expires_at="2026-08-24T09:30:00",
    )

    assert verdict["state"] == journal_health.STATE_OK
    assert verdict["days_since_refresh"] == pytest.approx(0.02, abs=0.01)


def test_silence_for_three_days_is_presumed_dead(tmp_path):
    """Questrade refresh tokens expire after three days of disuse, so a chain
    nothing has tried is not therefore healthy."""
    stale = (NOW - timedelta(days=5)).isoformat(timespec="seconds")
    verdict = _health(
        tmp_path,
        journal_questrade_refresh_token="token",
        journal_questrade_expires_at=stale,
    )

    assert verdict["state"] == journal_health.STATE_STALE
    assert verdict["state"] in journal_health.ALERTING_STATES
    assert str(journal_health.QUESTRADE_STALE_AFTER_DAYS) in verdict["headline"]
    assert verdict["action"]


# --- the reasoning that has to hold ----------------------------------------


def test_the_coverage_day_is_never_reported_as_the_failure_date(tmp_path):
    """A year-spanning backfill marks every session day FAILED from one broken
    chain. The oldest of those days is not when anything broke."""
    verdict = _health(
        tmp_path,
        failures=[
            ("2025-09-30", "500 Server Error ... oauth2/token"),
            ("2026-08-21", "500 Server Error ... oauth2/token"),
        ],
        journal_questrade_refresh_token="token",
        journal_questrade_expires_at="2026-08-22T00:30:04",
    )

    assert verdict["last_failure_at"] == "2026-08-21T00:30:00"
    assert "2025-09-30" not in verdict["headline"]


def test_a_recorded_auth_failure_outranks_a_fresh_looking_stamp(tmp_path):
    """`journal_questrade_expires_at` records the last refresh that WORKED.

    A chain that broke an hour ago still carries this morning's stamp, so a
    freshness check alone would call a dead chain healthy - which is exactly
    what happened on the desk between 2026-08-19 and 2026-08-24.
    """
    verdict = _health(
        tmp_path,
        failures=[("2026-08-24", "400 Client Error ... oauth2/token")],
        journal_questrade_refresh_token="token",
        journal_questrade_expires_at=NOW.isoformat(timespec="seconds"),
    )

    assert verdict["state"] == journal_health.STATE_DEAD


def test_a_failure_that_is_not_an_auth_failure_does_not_read_as_a_dead_chain(tmp_path):
    """A broker outage and a broken credential chain need different actions;
    telling the trader to paste a token would waste the one thing this surface
    is spending - their attention."""
    verdict = _health(
        tmp_path,
        failures=[("2026-08-24", "503 Service Unavailable for url: .../accounts")],
        journal_questrade_refresh_token="token",
        journal_questrade_expires_at="2026-08-24T09:30:00",
    )

    assert verdict["state"] == journal_health.STATE_OK


def test_a_token_that_never_refreshed_is_unknown_not_zero_days_old(tmp_path):
    """No path may state a number it did not measure (Phase 0.7 ground rule 6)."""
    verdict = _health(tmp_path, journal_questrade_refresh_token="token")

    assert verdict["state"] == journal_health.STATE_UNKNOWN
    assert verdict["days_since_refresh"] is None
    assert verdict["last_refresh_at"] is None


def test_every_verdict_carries_the_same_keys(tmp_path):
    """The renderers read these blind; a missing key would be a crash on the
    one path that only appears when something is already wrong."""
    expected = {
        "state", "headline", "action", "stale_after_days",
        "last_refresh_at", "days_since_refresh",
        "last_failure_day", "last_failure_at", "last_failure",
    }
    cases = [
        {"journal_questrade_refresh_token": ""},
        {"journal_questrade_refresh_token": "token"},
        {
            "journal_questrade_refresh_token": "token",
            "journal_questrade_expires_at": "2026-08-24T09:30:00",
        },
    ]
    for settings in cases:
        assert set(_health(tmp_path, **settings)) == expected


def test_the_module_imports_no_qt_and_no_ui():
    """It renders in System Health (frozen) and in the Journal tab; it must be
    reachable from both without dragging either one's world in."""
    import ast

    tree = ast.parse((SCRIPTS_DIR / "journal_health.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert not {"PySide6", "ui", "qtawesome", "pyqtgraph"} & imported


def test_an_aware_caller_and_a_naive_stamp_do_not_raise(tmp_path):
    """The Journal tab passes a naive `datetime.now()`; `operations_audit`
    passes a timezone-AWARE moment. The stamp on disk is naive machine-local,
    written by a bare `datetime.now()` in journal_importers.

    Subtracting those raises, which is how this first ran against the live
    store - the same naive/aware seam that cost a whole session on 2026-08-19.
    Normalizing ATTACHES the caller's zone to the naive side; stripping the
    aware side would end the crash and keep the wrong answer.
    """
    from datetime import timezone

    aware_now = NOW.replace(tzinfo=timezone.utc)
    verdict = journal_health.questrade_chain_health(
        now=aware_now,
        db_path=_db(tmp_path),
        settings_reader=_settings(
            journal_questrade_refresh_token="token",
            journal_questrade_expires_at="2026-08-24T09:00:00",
        ),
    )

    assert verdict["state"] == journal_health.STATE_OK
    assert verdict["days_since_refresh"] == pytest.approx(0.04, abs=0.01)


def test_an_aware_stamp_against_a_naive_caller_is_normalized_the_same_way(tmp_path):
    from datetime import timezone

    verdict = journal_health.questrade_chain_health(
        now=NOW,
        db_path=_db(tmp_path),
        settings_reader=_settings(
            journal_questrade_refresh_token="token",
            journal_questrade_expires_at=NOW.replace(
                hour=9, tzinfo=timezone.utc
            ).isoformat(),
        ),
    )

    assert verdict["state"] == journal_health.STATE_OK
    assert verdict["days_since_refresh"] == pytest.approx(0.04, abs=0.01)
