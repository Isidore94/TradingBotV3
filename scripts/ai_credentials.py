"""Provider API-key storage for the optional A.I. Summary workspace.

Environment variables always win. Saved keys use the OS credential store —
Credential Manager (generic credentials) on Windows, the login Keychain on
macOS — and never enter local_settings.json, logs, evidence packages,
prompts, or exports.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes
from typing import Mapping, Protocol


PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_TARGET_PREFIX = "TradingBotV3/ai-summary/"


class CredentialBackend(Protocol):
    def read(self, target: str) -> str: ...
    def write(self, target: str, secret: str) -> None: ...
    def delete(self, target: str) -> None: ...


class MemoryCredentialBackend:
    """Small injectable backend used by contract tests."""

    LABEL = "saved key store"

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def read(self, target: str) -> str:
        return self.values.get(target, "")

    def write(self, target: str, secret: str) -> None:
        self.values[target] = str(secret)

    def delete(self, target: str) -> None:
        self.values.pop(target, None)


if sys.platform == "win32":
    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]


class WindowsCredentialBackend:
    LABEL = "Windows Credential Manager"

    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    ERROR_NOT_FOUND = 1168

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Windows Credential Manager is unavailable on this platform")
        self._api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        self._api.CredReadW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
        ]
        self._api.CredReadW.restype = wintypes.BOOL
        self._api.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
        self._api.CredWriteW.restype = wintypes.BOOL
        self._api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self._api.CredDeleteW.restype = wintypes.BOOL
        self._api.CredFree.argtypes = [ctypes.c_void_p]

    def read(self, target: str) -> str:
        pointer = ctypes.POINTER(_CREDENTIALW)()
        if not self._api.CredReadW(target, self.CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
            error = ctypes.get_last_error()
            if error == self.ERROR_NOT_FOUND:
                return ""
            raise OSError(error, "CredReadW failed")
        try:
            credential = pointer.contents
            raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return raw.decode("utf-16-le")
        finally:
            self._api.CredFree(pointer)

    def write(self, target: str, secret: str) -> None:
        raw = str(secret).encode("utf-16-le")
        if not raw:
            self.delete(target)
            return
        blob = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        credential = _CREDENTIALW()
        credential.Type = self.CRED_TYPE_GENERIC
        credential.TargetName = target
        credential.CredentialBlobSize = len(raw)
        credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
        credential.Persist = self.CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = "TradingBotV3"
        credential.Comment = "TradingBotV3 optional A.I. Summary provider key"
        if not self._api.CredWriteW(ctypes.byref(credential), 0):
            error = ctypes.get_last_error()
            raise OSError(error, "CredWriteW failed")

    def delete(self, target: str) -> None:
        if self._api.CredDeleteW(target, self.CRED_TYPE_GENERIC, 0):
            return
        error = ctypes.get_last_error()
        if error != self.ERROR_NOT_FOUND:
            raise OSError(error, "CredDeleteW failed")


class MacKeychainCredentialBackend:
    """macOS login-Keychain storage via the ``security`` CLI (generic passwords).

    Mirrors the Windows backend's contract: ``read`` returns ``""`` for a
    missing item, ``delete`` tolerates a missing item, ``write`` upserts and
    treats an empty secret as delete. The secret does pass through the
    ``security`` argv for the duration of the call — acceptable on a
    single-user desktop, and it keeps the key out of every file the app writes.
    """

    LABEL = "macOS Keychain"

    _ACCOUNT = "TradingBotV3"
    _COMMENT = "TradingBotV3 optional A.I. Summary provider key"
    # errSecItemNotFound; the security CLI exits with the OSStatus low byte.
    _ERR_ITEM_NOT_FOUND = 44

    def __init__(self, runner=None) -> None:
        if runner is None and sys.platform != "darwin":
            raise RuntimeError("macOS Keychain is unavailable on this platform")
        self._runner = runner if runner is not None else self._run_security

    @staticmethod
    def _run_security(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["security", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def read(self, target: str) -> str:
        result = self._runner(
            ["find-generic-password", "-s", target, "-a", self._ACCOUNT, "-w"]
        )
        if result.returncode == self._ERR_ITEM_NOT_FOUND:
            return ""
        if result.returncode != 0:
            raise OSError(result.returncode, "security find-generic-password failed")
        return result.stdout.rstrip("\n")

    def write(self, target: str, secret: str) -> None:
        raw = str(secret)
        if not raw:
            self.delete(target)
            return
        result = self._runner(
            [
                "add-generic-password",
                "-s",
                target,
                "-a",
                self._ACCOUNT,
                "-w",
                raw,
                "-j",
                self._COMMENT,
                "-U",
            ]
        )
        if result.returncode != 0:
            raise OSError(result.returncode, "security add-generic-password failed")

    def delete(self, target: str) -> None:
        result = self._runner(
            ["delete-generic-password", "-s", target, "-a", self._ACCOUNT]
        )
        if result.returncode not in (0, self._ERR_ITEM_NOT_FOUND):
            raise OSError(result.returncode, "security delete-generic-password failed")


class AiCredentialVault:
    def __init__(
        self,
        backend: CredentialBackend | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.environ = environ if environ is not None else os.environ
        if backend is not None:
            self.backend = backend
        elif sys.platform == "win32":
            self.backend = WindowsCredentialBackend()
        elif sys.platform == "darwin":
            self.backend = MacKeychainCredentialBackend()
        else:
            self.backend = None

    @staticmethod
    def _provider(provider: str) -> str:
        normalized = str(provider or "").strip().lower()
        if normalized not in PROVIDER_ENV_KEYS:
            raise ValueError(f"unsupported AI provider: {provider}")
        return normalized

    def resolve(self, provider: str) -> tuple[str, str]:
        normalized = self._provider(provider)
        env_name = PROVIDER_ENV_KEYS[normalized]
        env_value = str(self.environ.get(env_name) or "").strip()
        if env_value:
            return env_value, f"environment ({env_name})"
        if self.backend is None:
            return "", "not configured"
        value = str(self.backend.read(_TARGET_PREFIX + normalized) or "").strip()
        label = getattr(self.backend, "LABEL", "saved key store")
        return (value, label) if value else ("", "not configured")

    def save(self, provider: str, secret: str) -> None:
        normalized = self._provider(provider)
        value = str(secret or "").strip()
        if not value:
            raise ValueError("API key cannot be blank")
        if self.backend is None:
            raise RuntimeError("No secure credential backend is available; use the provider environment variable")
        self.backend.write(_TARGET_PREFIX + normalized, value)

    def delete(self, provider: str) -> None:
        normalized = self._provider(provider)
        if self.backend is not None:
            self.backend.delete(_TARGET_PREFIX + normalized)

    def status(self, provider: str) -> str:
        value, source = self.resolve(provider)
        return f"Key ready · {source}" if value else "Key not configured"
