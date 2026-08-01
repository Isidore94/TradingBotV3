"""Contract tests for the macOS Keychain credential backend.

The backend shells out to the ``security`` CLI, so these tests inject a fake
runner and verify the storage contract on every platform: missing item reads
as "", writes upsert, an empty write deletes, and unexpected CLI failures
surface as OSError instead of being swallowed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_credentials import AiCredentialVault, MacKeychainCredentialBackend


class FakeSecurityCli:
    """In-memory stand-in for the ``security`` CLI's generic-password verbs."""

    ERR_ITEM_NOT_FOUND = 44
    ERR_DUPLICATE_ITEM = 45

    def __init__(self) -> None:
        self.items: dict[str, str] = {}
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        verb, rest = args[0], args[1:]
        opts: dict[str, str | bool] = {}
        index = 0
        while index < len(rest):
            flag = rest[index]
            # -U (upsert) never takes a value; -w is valueless in
            # find-generic-password (print secret) but takes one in add.
            if flag == "-U" or index + 1 >= len(rest):
                opts[flag] = True
                index += 1
            else:
                opts[flag] = rest[index + 1]
                index += 2
        service = str(opts.get("-s", ""))

        if verb == "find-generic-password":
            if service not in self.items:
                return subprocess.CompletedProcess(args, self.ERR_ITEM_NOT_FOUND, "", "not found")
            return subprocess.CompletedProcess(args, 0, self.items[service] + "\n", "")
        if verb == "add-generic-password":
            if service in self.items and "-U" not in opts:
                return subprocess.CompletedProcess(args, self.ERR_DUPLICATE_ITEM, "", "duplicate")
            self.items[service] = str(opts["-w"])
            return subprocess.CompletedProcess(args, 0, "", "")
        if verb == "delete-generic-password":
            if service not in self.items:
                return subprocess.CompletedProcess(args, self.ERR_ITEM_NOT_FOUND, "", "not found")
            del self.items[service]
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", f"unknown verb {verb}")


def test_read_missing_item_is_empty_not_error():
    backend = MacKeychainCredentialBackend(runner=FakeSecurityCli())
    assert backend.read("TradingBotV3/ai-summary/openai") == ""


def test_write_read_roundtrip_and_upsert():
    fake = FakeSecurityCli()
    backend = MacKeychainCredentialBackend(runner=fake)
    backend.write("svc", "first")
    assert backend.read("svc") == "first"
    # Second write must replace, not fail on the duplicate item (-U upsert).
    backend.write("svc", "second")
    assert backend.read("svc") == "second"


def test_empty_write_deletes_and_delete_tolerates_missing():
    fake = FakeSecurityCli()
    backend = MacKeychainCredentialBackend(runner=fake)
    backend.write("svc", "value")
    backend.write("svc", "")
    assert fake.items == {}
    backend.delete("svc")  # already gone: must not raise


def test_unexpected_cli_failure_raises():
    def broken_runner(args):
        return subprocess.CompletedProcess(args, 1, "", "keychain locked")

    backend = MacKeychainCredentialBackend(runner=broken_runner)
    with pytest.raises(OSError):
        backend.read("svc")
    with pytest.raises(OSError):
        backend.write("svc", "value")
    with pytest.raises(OSError):
        backend.delete("svc")


def test_vault_reports_keychain_as_source_without_exposing_value():
    backend = MacKeychainCredentialBackend(runner=FakeSecurityCli())
    vault = AiCredentialVault(backend, environ={})
    vault.save("openai", "kc-secret")
    assert vault.resolve("openai") == ("kc-secret", "macOS Keychain")
    assert "kc-secret" not in vault.status("openai")


def test_default_construction_requires_macos():
    if sys.platform == "darwin":
        pytest.skip("guard only applies off macOS")
    with pytest.raises(RuntimeError):
        MacKeychainCredentialBackend()
