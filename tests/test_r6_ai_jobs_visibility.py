"""R6(a): the overnight AI batch layer becomes visible from the desk.

The layer runs entirely outside the GUI - a scheduled task against the repo
checkout - so until now a night where it never ran and a night where it had
nothing to do looked identical from System Health. That is the same failure
`run_ai_jobs.ps1` was written to end, reproduced one level up.

The load-bearing constraint is that `ai_jobs` is in PACKAGES_NOT_IN_THE_BUNDLE:
the frozen desk does not contain it, and System Health IS frozen. So the audit
resolves the store by path alone, and the pin below keeps that duplicate rule
honest against the real one.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import operations_audit as audit  # noqa: E402

NOW = datetime(2026, 8, 18, 7, 0, tzinfo=timezone.utc)


def write_ledger(store: Path, rows):
    logs = store / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / audit.AI_JOB_LEDGER_NAME
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def row(job, status, *, ts=None):
    return {
        "job": job,
        "status": status,
        "ts": (ts or NOW - timedelta(hours=1)).isoformat(),
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    target = tmp_path / "ai_store"
    target.mkdir()
    monkeypatch.setenv(audit.AI_STORE_DIR_ENV, str(target))
    return target


class TestTheRow:
    def test_an_unconfigured_layer_is_measured_not_unknown(self, monkeypatch):
        """"Deliberately off" is an answer. UNKNOWN means we could not look."""
        monkeypatch.delenv(audit.AI_STORE_DIR_ENV, raising=False)
        monkeypatch.setattr(audit, "get_local_setting", lambda key, default=None: None)

        check = audit._ai_jobs_check(NOW, timezone.utc)

        assert check["id"] == "ai_jobs"
        assert check["status"] == audit.STATUS_HEALTHY
        assert check["details"]["configured"] is False
        # It must still name where the answer came from.
        assert check["source"].endswith("operations_audit.py")

    def test_a_configured_layer_with_no_ledger_is_unknown(self, store):
        check = audit._ai_jobs_check(NOW, timezone.utc)

        assert check["status"] == audit.STATUS_UNKNOWN
        assert check["details"]["configured"] is True

    def test_an_empty_ledger_is_unknown_not_healthy(self, store):
        (store / "logs").mkdir(parents=True, exist_ok=True)
        (store / "logs" / audit.AI_JOB_LEDGER_NAME).write_text("", encoding="utf-8")

        check = audit._ai_jobs_check(NOW, timezone.utc)

        assert check["status"] == audit.STATUS_UNKNOWN
        assert check["details"]["row_count"] == 0

    def test_a_recent_clean_run_is_healthy(self, store):
        write_ledger(store, [row("ai_summary", "ok"), row("digest", "ok")])

        check = audit._ai_jobs_check(NOW, timezone.utc)

        assert check["status"] == audit.STATUS_HEALTHY
        assert check["details"]["row_count"] == 2
        assert check["details"]["last_job"] == "digest"

    def test_a_failed_job_is_unhealthy(self, store):
        write_ledger(store, [row("ai_summary", "ok"), row("digest", "failed")])

        check = audit._ai_jobs_check(NOW, timezone.utc)

        assert check["status"] == audit.STATUS_UNHEALTHY
        assert "1 AI job(s) failed" in check["summary"]

    def test_a_degraded_job_is_degraded_not_failed(self, store):
        write_ledger(store, [row("ai_summary", "degraded")])

        check = audit._ai_jobs_check(NOW, timezone.utc)

        assert check["status"] == audit.STATUS_DEGRADED

    def test_failure_outranks_degradation(self, store):
        write_ledger(store, [row("a", "degraded"), row("b", "failed")])

        assert audit._ai_jobs_check(NOW, timezone.utc)["status"] == audit.STATUS_UNHEALTHY

    def test_a_stale_schedule_stops_being_healthy(self, store):
        """Nightly, so hours are normal and days are not."""
        write_ledger(store, [row("ai_summary", "ok", ts=NOW - timedelta(days=4))])

        assert audit._ai_jobs_check(NOW, timezone.utc)["status"] != audit.STATUS_HEALTHY

    def test_a_corrupt_tail_line_does_not_hide_the_whole_ledger(self, store):
        path = write_ledger(store, [row("ai_summary", "ok")])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"job": "digest", "stat')  # killed mid-flush

        check = audit._ai_jobs_check(NOW, timezone.utc)

        assert check["details"]["row_count"] == 1
        assert check["status"] == audit.STATUS_HEALTHY


class TestTheDuplicateRuleStaysHonest:
    def test_ai_jobs_store_resolution_matches_the_batch_layer(self, tmp_path, monkeypatch):
        """The audit cannot import ai_jobs, so it must agree with it instead."""
        from ai_jobs import store as batch_store

        assert audit.AI_STORE_DIR_ENV == batch_store.AI_STORE_DIR_ENV
        assert audit.AI_STORE_DIR_SETTING == batch_store.AI_STORE_DIR_SETTING

        target = tmp_path / "store"
        target.mkdir()
        monkeypatch.setenv(audit.AI_STORE_DIR_ENV, str(target))
        assert audit._ai_store_dir() == batch_store.get_ai_store_dir()

    def test_the_setting_is_read_when_the_env_is_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv(audit.AI_STORE_DIR_ENV, raising=False)
        monkeypatch.setattr(
            audit,
            "get_local_setting",
            lambda key, default=None: str(tmp_path) if key == audit.AI_STORE_DIR_SETTING else None,
        )

        assert audit._ai_store_dir() == tmp_path

    def test_the_ledger_filename_matches_the_batch_layer(self):
        from ai_jobs import ledger as batch_ledger

        assert audit.AI_JOB_LEDGER_NAME == batch_ledger.LEDGER_NAME

    def test_the_audit_never_imports_the_unbundled_package(self):
        """A frozen System Health cannot reach `ai_jobs` at all."""
        source = (SCRIPTS_DIR / "operations_audit.py").read_text(encoding="utf-8")
        for forbidden in ("import ai_jobs", "from ai_jobs"):
            assert forbidden not in source


class TestTheWrapperLogLine:
    def test_the_routine_run_no_longer_reads_as_a_mistake(self):
        source = (SCRIPTS_DIR / "run_ai_jobs.ps1").read_text(encoding="utf-8")
        # Comments still discuss the old wording on purpose - they record WHY
        # it changed. Only what the wrapper actually logs is under test.
        code = [
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        ]
        emitted = "\n".join(code)

        assert "(no arguments)" not in emitted
        assert "scheduled run: every due slot" in emitted
