"""P2-11d: the OpenAI key and ntfy token move to Windows Credential Manager.

A fake keyring backend only; the settings file is a temp copy. Migration may
never lose a secret: it blanks the JSON field only after an exact read-back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402
import secret_store  # noqa: E402

OPENAI = secret_store.MARKET_PREP_OPENAI_KEY
NTFY = secret_store.PUSH_NTFY_TOKEN


class FakeKeyring:
    def __init__(self, *, fail_set=False, mangle=False, fail_get=False):
        self.values: dict[tuple[str, str], str] = {}
        self.fail_set = fail_set
        self.mangle = mangle
        self.fail_get = fail_get

    def get_password(self, service, name):
        if self.fail_get:
            raise RuntimeError("vault locked")
        return self.values.get((service, name))

    def set_password(self, service, name, value):
        if self.fail_set:
            raise RuntimeError("vault locked")
        self.values[(service, name)] = value + "x" if self.mangle else value

    def delete_password(self, service, name):
        self.values.pop((service, name), None)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    path = tmp_path / "local_settings.json"
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", path)
    project_paths.invalidate_local_settings_cache()

    def write(payload):
        path.write_text(json.dumps(payload), encoding="utf-8")
        project_paths.invalidate_local_settings_cache()

    def read():
        return json.loads(path.read_text(encoding="utf-8"))

    yield write, read
    project_paths.invalidate_local_settings_cache()


def _use(monkeypatch, backend):
    monkeypatch.setattr(secret_store, "_keyring", lambda: backend)
    return backend


def test_migration_moves_both_secrets_and_blanks_json_only_after_readback(settings, monkeypatch):
    write, read = settings
    write({OPENAI: "sk-123", NTFY: "tk-9", "push_ntfy_topic": "desk"})
    backend = _use(monkeypatch, FakeKeyring())
    results = secret_store.migrate_secrets_to_keyring()
    assert results == {OPENAI: "migrated", NTFY: "migrated"}
    assert backend.values[(secret_store.KEYRING_SERVICE, OPENAI)] == "sk-123"
    assert backend.values[(secret_store.KEYRING_SERVICE, NTFY)] == "tk-9"
    after = read()
    assert after[OPENAI] == "" and after[NTFY] == ""
    assert after["push_ntfy_topic"] == "desk"


@pytest.mark.parametrize(
    "backend",
    [FakeKeyring(fail_set=True), FakeKeyring(mangle=True), None],
    ids=["write-fails", "readback-mismatch", "no-keyring"],
)
def test_a_failed_migration_leaves_the_json_untouched(settings, monkeypatch, backend):
    write, read = settings
    write({OPENAI: "sk-123", NTFY: ""})
    _use(monkeypatch, backend)
    results = secret_store.migrate_secrets_to_keyring()
    assert results == {OPENAI: "kept_in_json", NTFY: "empty"}
    assert read()[OPENAI] == "sk-123"


def test_a_failed_blank_keeps_the_verified_copy(settings, monkeypatch):
    write, read = settings
    write({NTFY: "tk-9"})
    backend = _use(monkeypatch, FakeKeyring())

    def refuse(*_args, **_kwargs):
        raise OSError("settings locked")

    # The blank is the locked compare-and-blank now (review advisory 2).
    monkeypatch.setattr(project_paths, "save_local_setting", refuse)
    monkeypatch.setattr(project_paths, "blank_local_setting_if_equal", refuse)
    assert secret_store.migrate_secrets_to_keyring((NTFY,)) == {NTFY: "blank_failed"}
    assert backend.values[(secret_store.KEYRING_SERVICE, NTFY)] == "tk-9"
    assert read()[NTFY] == "tk-9"


def test_readers_try_the_store_first_then_the_json(settings, monkeypatch):
    write, _read = settings
    write({NTFY: "json-token"})
    backend = _use(monkeypatch, FakeKeyring())
    assert secret_store.read_secret_setting(NTFY) == "json-token"
    backend.values[(secret_store.KEYRING_SERVICE, NTFY)] = "vault-token"
    assert secret_store.read_secret_setting(NTFY) == "vault-token"
    backend.fail_get = True
    assert secret_store.read_secret_setting(NTFY) == "json-token"


def test_push_config_and_openai_key_read_the_store(settings, monkeypatch):
    write, _read = settings
    write({"push_ntfy_topic": "desk", NTFY: "", OPENAI: ""})
    backend = _use(monkeypatch, FakeKeyring())
    backend.values[(secret_store.KEYRING_SERVICE, NTFY)] = "vault-token"
    backend.values[(secret_store.KEYRING_SERVICE, OPENAI)] = "sk-vault"
    import push_notify
    from market_prep.services import ai_service

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert push_notify.load_push_config()["token"] == "vault-token"
    assert ai_service._local_openai_api_key() == "sk-vault"


def test_saving_a_typed_token_goes_to_the_store_or_falls_back_to_json(settings, monkeypatch):
    write, read = settings
    write({NTFY: "old"})
    backend = _use(monkeypatch, FakeKeyring())
    assert secret_store.save_secret_setting(NTFY, "new") == "keyring"
    assert backend.values[(secret_store.KEYRING_SERVICE, NTFY)] == "new"
    assert read()[NTFY] == ""

    _use(monkeypatch, FakeKeyring(fail_set=True))
    assert secret_store.save_secret_setting(NTFY, "newer") == "json"
    assert read()[NTFY] == "newer"


def test_a_failed_verification_leaves_no_store_entry_to_shadow_the_json(settings, monkeypatch):
    write, read = settings
    write({OPENAI: "sk-123"})
    backend = _use(monkeypatch, FakeKeyring(mangle=True))
    assert secret_store.migrate_secrets_to_keyring((OPENAI,)) == {OPENAI: "kept_in_json"}
    assert (secret_store.KEYRING_SERVICE, OPENAI) not in backend.values
    assert secret_store.read_secret_setting(OPENAI) == "sk-123"


def test_a_json_fallback_save_drops_the_older_stored_token(settings, monkeypatch):
    write, read = settings
    write({NTFY: ""})
    backend = _use(monkeypatch, FakeKeyring())
    backend.values[(secret_store.KEYRING_SERVICE, NTFY)] = "old-token"
    backend.fail_set = True
    assert secret_store.save_secret_setting(NTFY, "new-token") == "json"
    assert (secret_store.KEYRING_SERVICE, NTFY) not in backend.values
    assert secret_store.read_secret_setting(NTFY) == "new-token"


def test_migration_never_blanks_a_value_saved_after_the_copy(settings, monkeypatch):
    write, read = settings
    write({NTFY: "tk-old"})

    class SaveDuringCopy(FakeKeyring):
        def set_password(self, service, name, value):
            super().set_password(service, name, value)
            write({NTFY: "tk-new"})  # the trader saves while the copy is in flight

    _use(monkeypatch, SaveDuringCopy())
    assert secret_store.migrate_secrets_to_keyring((NTFY,)) == {NTFY: "changed_meanwhile"}
    assert read()[NTFY] == "tk-new"
