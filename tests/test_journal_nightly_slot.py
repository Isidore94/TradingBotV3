"""R7 §9 step 10 - the nightly journal pull (spec §6, invariant I8).

The promotion of the queued P3.3 slice. I8 is the whole shape of it: no new
timer, no new thread owner, no new ntfy sender - the work runs inside the
existing ``ai_jobs`` runner slot, and its only outputs are database rows plus the
ledger entry the runner already writes.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_coverage as jc  # noqa: E402
import journal_runner  # noqa: E402
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


class _QuietBroker:
    """A broker that authenticates, holds nothing, and traded nothing."""

    refresh_token = "token"
    access_token = ""
    api_server = ""

    def __init__(self):
        self.quarantined = []

    def get_accounts(self):
        return [{"number": "51830546", "type": "TFSA"}]

    def iter_execution_chunks(self, start_date, end_date):
        yield {
            "account": {"number": "51830546", "type": "TFSA"},
            "account_number": "51830546",
            "start": start_date,
            "end": end_date,
            "executions": [],
        }

    def get_activities(self, account_number, start, end):
        return []

    def get_positions(self, account_number):
        return []


@pytest.fixture
def quiet_night(monkeypatch):
    """No network anywhere: both brokers stubbed, FX stubbed."""
    monkeypatch.setattr(journal_runner, "QuestradeImporter", _QuietBroker)
    monkeypatch.setattr(
        journal_runner, "import_ibkr_flex_executions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("IBKR Flex is not configured.")),
    )
    monkeypatch.setattr(
        journal_runner.journal_fx, "ensure_rates",
        lambda store, pairs, **kwargs: {"booked": 0, "carried_back": 0, "unavailable": [], "errors": []},
    )


def test_a_night_with_no_trades_is_ok(quiet_night, store):
    """A quiet market is not a failure.

    A job that reported one would teach the trader to ignore it, which is the
    opposite of what a self-healing ledger is for.
    """
    result = journal_runner.run_nightly_journal_import(store=store)
    assert result["ok"] is True and result["status"] == "OK"
    assert result["trade_count"] == 0


def test_the_night_runs_its_steps_in_the_order_that_makes_them_true(quiet_night, store, monkeypatch):
    """Import, heal, convert, rebuild, reconcile - and the order is load-bearing.

    A rebuild before the self-heal assembles a journal with known holes in it,
    and a reconciliation before the rebuild compares against trades the night's
    imports already invalidated.
    """
    order: list[str] = []
    monkeypatch.setattr(
        journal_runner.journal_coverage, "self_heal",
        lambda *a, **k: order.append("heal") or {"repaired": [], "failed": [], "exhausted": []},
    )
    monkeypatch.setattr(
        journal_runner.journal_fx, "ensure_rates",
        lambda *a, **k: order.append("fx") or {"booked": 0, "carried_back": 0, "unavailable": [], "errors": []},
    )
    real_rebuild = store.rebuild_trades
    monkeypatch.setattr(store, "rebuild_trades", lambda **k: order.append("rebuild") or real_rebuild(**k))
    monkeypatch.setattr(
        journal_runner.journal_reconcile, "reconcile",
        lambda *a, **k: order.append("reconcile") or {"positions_checked": 0, "mismatched": []},
    )
    journal_runner.run_nightly_journal_import(store=store)
    assert order == ["heal", "fx", "rebuild", "reconcile"]


def test_the_nightly_pull_re_reads_a_week_not_a_day(quiet_night, store):
    """A broker can amend or late-report a fill for days afterwards.

    A nightly job that only ever looked at yesterday would never see the
    amendment, which is A3 wearing a different hat.
    """
    result = journal_runner.run_nightly_journal_import(store=store)
    span = (date.fromisoformat(result["end_date"]) - date.fromisoformat(result["start_date"])).days
    assert span == journal_runner.NIGHTLY_LOOKBACK_DAYS == 7


def test_a_quiet_night_still_records_coverage(quiet_night, store):
    """Which is the point: "we looked, there was nothing" is a fact worth storing."""
    journal_runner.run_nightly_journal_import(store=store)
    rows = jc.coverage_rows(store, broker="QUESTRADE")
    assert rows, "a night that imported nothing still says which days it read"
    assert {row["status"] for row in rows} <= {jc.COVERED, jc.NO_SESSION}


def test_an_unreachable_broker_is_reported_and_does_not_fail_the_night(quiet_night, store):
    """IBKR Flex is unconfigured in this fixture. That is a message, not a crash."""
    result = journal_runner.run_nightly_journal_import(store=store)
    assert result["ok"] is True
    assert any("IBKR" in message for message in result["messages"])


def test_a_failed_rebuild_fails_the_night(quiet_night, store, monkeypatch):
    monkeypatch.setattr(
        store, "rebuild_trades", lambda **k: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    result = journal_runner.run_nightly_journal_import(store=store)
    assert result["ok"] is False
    assert any("rebuild failed" in message for message in result["messages"])


def test_a_failed_fx_booking_does_not_fail_the_night(quiet_night, store, monkeypatch):
    """Unconverted is an honest state (I5); a BoC outage is not a broken journal."""
    monkeypatch.setattr(
        journal_runner.journal_fx, "ensure_rates",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("valet down")),
    )
    result = journal_runner.run_nightly_journal_import(store=store)
    assert result["ok"] is True
    assert any("fx booking failed" in message for message in result["messages"])


def test_the_night_reports_a_re_key_that_needs_a_human(quiet_night, store, monkeypatch):
    monkeypatch.setattr(
        store, "rebuild_trades",
        lambda **k: setattr(store, "last_rekey", {"remapped": [], "ambiguous": [{"old_trade_id": "x"}], "orphaned": []}) or 0,
    )
    result = journal_runner.run_nightly_journal_import(store=store)
    assert any("re-key needs review" in message for message in result["messages"])


def test_the_self_heals_per_day_fetch_refuses_ibkr_honestly(store):
    """A gap it cannot fill must say so, not report a repair that did not happen.

    The socket sees only the current TWS session and Flex is a whole-statement
    operation; there is no per-day IBKR fetch to make. Raising marks the day
    FAILED with a readable reason, which is the truth.
    """
    with pytest.raises(RuntimeError, match="no per-day fetch"):
        journal_runner._fetch_one_day(store, "IBKR", "U4867396", date(2026, 8, 5))


# ---------------------------------------------------------------------------
# I8 - it runs where it was told to run, and nowhere else
# ---------------------------------------------------------------------------


def test_the_slot_is_registered_first_and_cheaply():
    from ai_jobs.runner import default_slots

    slots = {slot.name: slot for slot in default_slots()}
    assert list(slots)[0] == "journal_import"
    assert slots["journal_import"].reserve_minutes == 5.0
    assert slots["journal_import"].max_attempts == 3
    assert slots["journal_import"].enabled is True


def test_the_nightly_path_adds_no_timer_thread_or_notifier():
    """I8, checked against the source rather than asserted in prose.

    The failure this prevents is a well-meant future edit adding a push "so the
    trader knows it ran" - which would quietly break the standing phone-push
    policy that AWAY is the only mode that pushes routine output.
    """
    import ast

    tree = ast.parse((SCRIPTS_DIR / "journal_runner.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    # Parsed, not grepped: this module's own docstring says the words "no new
    # ntfy sender", and a substring search would fail on the sentence promising
    # the thing it is checking for.
    for forbidden in {"QTimer", "Thread", "Timer", "ntfy", "send_push", "push_alert", "notify"}:
        assert forbidden not in names, f"{forbidden} has no business in the nightly journal path"
