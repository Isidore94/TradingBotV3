"""Machine-local secrets in Windows Credential Manager (via ``keyring``), P2-11d.

Two secrets moved out of ``local_settings.json``: the market-prep OpenAI key
and the ntfy push token. Readers ask the credential store first and fall back
to the JSON value, so a machine that has not migrated yet keeps working.

Migration copies a JSON value into the store, reads it back, and only after an
exact match blanks the JSON field. Any failure leaves the JSON untouched and
logs why: a secret is never lost.
"""

from __future__ import annotations

import logging
from typing import Any

_LOG = logging.getLogger(__name__)

#: Credential Manager "service" name; each secret is stored under its settings key.
KEYRING_SERVICE = "TradingBotV3"
MARKET_PREP_OPENAI_KEY = "market_prep_openai_api_key"
PUSH_NTFY_TOKEN = "push_ntfy_token"
#: The settings keys that live in the credential store.
SECRET_SETTING_KEYS = (MARKET_PREP_OPENAI_KEY, PUSH_NTFY_TOKEN)


def _keyring():
    """The ``keyring`` module, or None when it is not installed."""
    try:
        import keyring
    except Exception:  # an absent or broken keyring means "JSON only"
        return None
    return keyring


def read_keyring_secret(name: str) -> str:
    """The stored secret, or "" when missing or the store cannot be read."""
    backend = _keyring()
    if backend is None:
        return ""
    try:
        return str(backend.get_password(KEYRING_SERVICE, name) or "")
    except Exception as exc:
        _LOG.warning("Credential store read failed for %s: %s", name, type(exc).__name__)
        return ""


def write_keyring_secret(name: str, value: str) -> bool:
    """Store ``value`` and confirm it reads back exactly. Never raises."""
    backend = _keyring()
    if backend is None:
        return False
    try:
        backend.set_password(KEYRING_SERVICE, name, value)
        return str(backend.get_password(KEYRING_SERVICE, name) or "") == value
    except Exception as exc:
        _LOG.warning("Credential store write failed for %s: %s", name, type(exc).__name__)
        return False


def read_secret_setting(name: str, default: str = "") -> str:
    """The credential store first, then the JSON setting, then ``default``."""
    stored = read_keyring_secret(name).strip()
    if stored:
        return stored
    try:
        from project_paths import get_local_setting

        return str(get_local_setting(name, default) or default).strip()
    except Exception:
        return default


def save_secret_setting(name: str, value: str) -> str:
    """Save a secret the trader typed: credential store when it verifies, else JSON.

    Returns where it went: "keyring" or "json". The JSON copy is blanked only
    after the store holds the exact value.
    """
    from project_paths import get_local_setting, save_local_setting

    value = str(value or "").strip()
    if value and write_keyring_secret(name, value):
        if str(get_local_setting(name, "") or ""):
            save_local_setting(name, "")
        return "keyring"
    if not value:
        # Clearing: blank both so neither store resurrects an old secret.
        backend = _keyring()
        if backend is not None:
            try:
                backend.delete_password(KEYRING_SERVICE, name)
            except Exception:  # nothing stored is the same as deleted
                _LOG.debug("No stored %s to delete.", name)
    save_local_setting(name, value)
    return "json"


def migrate_secrets_to_keyring(names: tuple[str, ...] = SECRET_SETTING_KEYS) -> dict[str, str]:
    """Move each non-empty JSON secret into the credential store; never loses one.

    Per key: "migrated", "empty" (nothing in JSON), "kept_in_json" (store
    unavailable or read-back mismatch; JSON untouched), or "blank_failed"
    (stored and verified, but the JSON field could not be blanked).
    """
    from project_paths import get_local_setting, save_local_setting

    results: dict[str, str] = {}
    for name in names:
        try:
            value = str(get_local_setting(name, "") or "")
        except Exception as exc:
            _LOG.warning("Secret migration: settings unreadable for %s: %s", name, exc)
            results[name] = "kept_in_json"
            continue
        if not value.strip():
            results[name] = "empty"
            continue
        if not write_keyring_secret(name, value):
            _LOG.warning("Secret migration: %s stays in local_settings.json (store unavailable or read-back mismatch).", name)
            results[name] = "kept_in_json"
            continue
        try:
            save_local_setting(name, "")
        except Exception as exc:
            _LOG.warning("Secret migration: %s is stored, but the JSON copy was not blanked: %s", name, exc)
            results[name] = "blank_failed"
            continue
        _LOG.info("Secret migration: %s moved to the credential store.", name)
        results[name] = "migrated"
    return results


def _warm_and_migrate() -> None:
    # One read loads keyring and its backend here, not later on the Qt thread.
    read_keyring_secret(PUSH_NTFY_TOKEN)
    migrate_secrets_to_keyring()


def migrate_in_background() -> Any:
    """Run the migration on a daemon thread (desk start). Returns the thread."""
    import threading

    thread = threading.Thread(target=_warm_and_migrate, name="secret-migration", daemon=True)
    thread.start()
    return thread
