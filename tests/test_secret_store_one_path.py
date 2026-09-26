"""secret_store stores through ai_credentials (one Credential Manager path).

In-memory backend only. The targets are keyring's WinVault layout, so a secret
saved while secret_store used the ``keyring`` package still reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_credentials  # noqa: E402
import project_paths  # noqa: E402
import secret_store  # noqa: E402

SERVICE = secret_store.KEYRING_SERVICE
OPENAI = secret_store.MARKET_PREP_OPENAI_KEY
NTFY = secret_store.PUSH_NTFY_TOKEN


@pytest.fixture
def backend(tmp_path, monkeypatch):
    path = tmp_path / "local_settings.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", path)
    project_paths.invalidate_local_settings_cache()
    store = ai_credentials.MemoryCredentialBackend()
    monkeypatch.setattr(ai_credentials, "default_backend", lambda: store)
    yield store
    project_paths.invalidate_local_settings_cache()


def test_secrets_saved_by_keyring_still_read(backend):
    # keyring's layout: the newest secret under the plain service target, an
    # older one moved to "{name}@{service}".
    backend.write(SERVICE, "ntfy-token", username=NTFY)
    backend.write(f"{OPENAI}@{SERVICE}", "sk-old-key", username=OPENAI)
    assert secret_store.read_secret_setting(NTFY) == "ntfy-token"
    assert secret_store.read_secret_setting(OPENAI) == "sk-old-key"


def test_a_save_writes_keyring_layout_and_keeps_the_other_secret(backend):
    assert secret_store.save_secret_setting(OPENAI, "sk-one") == "keyring"
    assert secret_store.save_secret_setting(NTFY, "tok-two") == "keyring"
    assert backend.read_entry(SERVICE) == ("tok-two", NTFY)
    assert backend.read_entry(f"{OPENAI}@{SERVICE}") == ("sk-one", OPENAI)
    assert secret_store.read_secret_setting(OPENAI) == "sk-one"
    assert secret_store.read_secret_setting(NTFY) == "tok-two"


def test_a_delete_removes_only_that_secret(backend):
    secret_store.save_secret_setting(OPENAI, "sk-one")
    secret_store.save_secret_setting(NTFY, "tok-two")
    secret_store.delete_keyring_secret(NTFY)
    assert secret_store.read_keyring_secret(NTFY) == ""
    assert secret_store.read_keyring_secret(OPENAI) == "sk-one"


def test_the_store_is_the_ctypes_path_not_the_keyring_package(backend):
    store = secret_store._keyring()
    assert isinstance(store, ai_credentials.KeyringLayoutStore)
    assert store.backend is backend
