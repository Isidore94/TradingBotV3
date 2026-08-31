"""Three System Health handlers said "must never take the audit down" and did.

`operations_audit` guards three reads with `except Exception:` and then calls
`logging.exception(...)` - but the module never imported `logging`, so the
handler raised `NameError` out of the audit at exactly the moment its guard was
supposed to hold. All three carry `# pragma: no cover`, which is why nothing
noticed: the failure path had no test, and the linter that catches an undefined
name was declared in `requirements-dev.txt` but not installed, so it had never
been run against this tree.

These are the fail-before-fix proofs. Each drives the guarded call into an
exception and asserts the audit degrades - UNKNOWN, or an empty note - instead
of raising.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import operations_audit  # noqa: E402


def test_the_module_can_actually_log(caplog):
    """`logging` has to be a name the module holds, not one it hopes for."""
    assert hasattr(operations_audit, "logging"), (
        "three except-handlers call logging.exception; without the import each "
        "one raises NameError instead of logging"
    )


def test_unreadable_daily_bar_history_returns_an_empty_note(tmp_path, monkeypatch, caplog):
    import evidence_rules

    def _boom(_manifest_dir):
        raise OSError("manifest directory went away")

    monkeypatch.setattr(evidence_rules, "daily_volume_session_verdicts", _boom)
    note, details = operations_audit._daily_bar_history_note(tmp_path)
    assert (note, details) == ("", {})


def test_an_unreadable_source_pin_reports_unknown(tmp_path, monkeypatch):
    from master_avwap_lib import legacy

    def _boom():
        raise OSError("local settings unreadable")

    monkeypatch.setattr(legacy, "daily_bars_source_pin", _boom)
    check = operations_audit._daily_bar_source_check(
        datetime.now(timezone.utc), timezone.utc, tmp_path
    )
    assert check["status"] == operations_audit.STATUS_UNKNOWN
    assert "could not be read" in check["summary"]


def test_unreadable_snapshot_health_reports_unknown(tmp_path, monkeypatch):
    from ops import evidence_snapshot

    def _boom(_staging, das_root=None):
        raise OSError("the DAS is not reachable")

    monkeypatch.setattr(evidence_snapshot, "health", _boom)
    check = operations_audit._evidence_snapshot_check(
        datetime.now(timezone.utc), timezone.utc, staging=tmp_path
    )
    assert check["status"] == operations_audit.STATUS_UNKNOWN
    assert "could not be read" in check["summary"]


@pytest.mark.parametrize(
    "name", ["_daily_bar_history_note", "_daily_bar_source_check", "_evidence_snapshot_check"]
)
def test_each_guarded_reader_still_exists(name):
    """If one of these is renamed, the proof above must be renamed with it
    rather than passing vacuously against a function that no longer runs."""
    assert callable(getattr(operations_audit, name))
