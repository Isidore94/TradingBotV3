"""Layer 2: genuine machine-local, cross-process exclusion for the publisher.

WHAT THIS IS, AND WHAT IT IS NOT (plan.md sec 4)
------------------------------------------------
The Drive-synchronized lease in :mod:`writer_lease` is *cross-machine writer
protection*: a shared file that names the current writer. It is **not** a
compare-and-swap lock, and two machines can genuinely race before Google Drive
converges. Nothing in this module changes that.

What this module *does* fix is the other half of the problem, which a
read/replace lease on a synced file cannot solve at all: **two processes on the
designated Windows host**. The GUI, a second GUI started by accident, and a
scheduled scan all run on one machine with one Drive mount, so they see the
same lease bytes with no sync delay whatsoever and can interleave a
read-check-write. That is a local concurrency problem and it needs a local
kernel primitive.

THE PRIMITIVE, AND WHY
----------------------
Two layers are taken together, always, in a fixed order:

1. an in-process :class:`threading.RLock`, because neither OS primitive below
   arbitrates two *threads* of one process usefully (a Win32 mutex is owned by
   a thread and is recursive for that thread; a byte-range file lock is owned by
   the *handle* and a second lock attempt from the same process succeeds);
2. a **Win32 named mutex** (``CreateMutexW`` / ``WaitForSingleObject`` /
   ``ReleaseMutex``) *and* an exclusive byte-range lock on a machine-local lock
   file (``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere).

Both OS layers are attempted on every acquisition rather than one being a
fallback for the other. A fallback would be worse than either alone: if process
A took the mutex and process B silently fell back to the file lock, the two
would not exclude each other at all. Taking both means a layer that is
unavailable in one process cannot create a hole in the other, and the state of
each layer is reported in :class:`LocalLockInfo` so Health telemetry can say
which protections were actually in force.

If **neither** OS layer is really in force, acquisition fails closed rather than
reporting ``held=True`` on the strength of an in-process ``RLock`` that
arbitrates nothing between processes.

WHAT HAPPENS WHEN THE HOLDER IS KILLED WITHOUT RELEASING
--------------------------------------------------------
This is the deciding property, and it is why a plain "lock file exists" marker
was rejected: a marker file left behind by a killed process wedges the writer
until somebody deletes it by hand.

* Named mutex: mutex ownership is a kernel object tied to the process's handle.
  When the process dies - ``TerminateProcess``, a power cut, a native crash -
  the kernel closes the handle and the mutex is released. The next waiter
  returns ``WAIT_ABANDONED`` instead of ``WAIT_OBJECT_0``: it *gets* the lock,
  and it is told the previous owner died mid-transaction.

  ``WAIT_ABANDONED`` alone is not a sufficient detector, and it would be
  dishonest to present it as one. If the dead process held the only handle to
  the name, the kernel destroyed the object with it and the next
  ``CreateMutexW`` builds a *new*, unowned mutex - so the next waiter sees a
  clean ``WAIT_OBJECT_0`` and learns nothing. That is exactly the lone-GUI
  native-crash case this repo has a history of. A small owner marker file,
  written under the lock and deleted only on a clean release, covers it: it
  survives the kill, and the next acquirer reports
  ``abandoned_by_previous_owner=True``.
* Byte-range file lock: also owned by the open handle, also released by the
  kernel on process death. The lock *file* may survive as an empty file; that is
  inert, because the file's existence means nothing - only a held range lock
  does.

So a hard kill never wedges publishing. It also never lets a killed process's
Drive lease block forever: that lease still expires on its TTL
(see :mod:`writer_lease`).

SCOPE, HONESTLY
---------------
The mutex is created in the ``Local\\`` (per-logon-session) namespace and the
lock file lives under the machine's temp directory, so the guarantee is
*same machine, same user session*. Two processes run by different users, or one
in session 0 as a Windows service, are not arbitrated by the mutex layer; the
lock file still arbitrates them when both can open it. Deliberately no
``Global\\`` namespace: creating a Global object needs a privilege a standard
user account does not have, and a name that succeeds in one process and fails in
another is precisely the hole described above.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LocalLockInfo",
    "LocalLockUnavailable",
    "lock_key_for_path",
    "local_writer_lock",
]

DEFAULT_TIMEOUT_SECONDS = 20.0
_MUTEX_PREFIX = "Local\\TradingBotV3-writer-"
_POLL_SECONDS = 0.005


class LocalLockUnavailable(RuntimeError):
    """The machine-local exclusion could not be taken within the timeout.

    Raised, never swallowed: a publisher that cannot prove it is the only
    process in the transaction must fail closed.
    """


@dataclass
class LocalLockInfo:
    """Observable state of the local exclusion, for Health telemetry."""

    key: str
    name: str
    held: bool = False
    depth: int = 0
    mutex: str = "unavailable"          # "held" | "unavailable" | "unsupported"
    file_lock: str = "unavailable"      # "held" | "unavailable" | "unsupported"
    abandoned_by_previous_owner: bool = False
    lock_file: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "held": bool(self.held),
            "depth": int(self.depth),
            "mutex": self.mutex,
            "file_lock": self.file_lock,
            "abandoned_by_previous_owner": bool(self.abandoned_by_previous_owner),
            "lock_file": self.lock_file,
        }


def lock_key_for_path(path: Path | str) -> str:
    """A stable key for ``path`` that every process on this machine agrees on.

    ``normcase`` + ``abspath`` so ``C:\\Drive\\x.txt`` and ``c:/drive/x.txt``
    resolve to one lock; a digest so the key is legal in a kernel object name.
    """
    resolved = os.path.normcase(os.path.abspath(str(path)))
    return hashlib.sha256(resolved.encode("utf-8", errors="replace")).hexdigest()[:32]


def _lock_directory() -> Path:
    directory = Path(tempfile.gettempdir()) / "tradingbotv3-writer-locks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Win32 named mutex
# ---------------------------------------------------------------------------
_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102
_WAIT_FAILED = 0xFFFFFFFF


def _kernel32():
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        return kernel32
    except Exception:  # pragma: no cover - ctypes unavailable
        return None


class _MutexLayer:
    """Named-mutex half of the local lock. Unsupported layers report so."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handle = None
        self.state = "unsupported"
        self.abandoned = False

    def acquire(self, deadline: float) -> None:
        kernel32 = _kernel32()
        if kernel32 is None:
            self.state = "unsupported"
            return
        handle = kernel32.CreateMutexW(None, False, self.name)
        if not handle:
            self.state = "unsupported"
            return
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        result = kernel32.WaitForSingleObject(handle, remaining_ms)
        if result == _WAIT_OBJECT_0:
            self.handle, self.state = handle, "held"
            return
        if result == _WAIT_ABANDONED:
            # The previous owner died holding it. We own it now; say so loudly
            # rather than pretending the transaction before ours completed.
            self.handle, self.state, self.abandoned = handle, "held", True
            return
        kernel32.CloseHandle(handle)
        if result == _WAIT_TIMEOUT:
            raise LocalLockUnavailable(
                f"another process on this machine has held the writer mutex {self.name} "
                "for longer than the timeout; refusing to publish concurrently"
            )
        raise LocalLockUnavailable(
            f"waiting on the writer mutex {self.name} failed (code {result:#x})"
        )

    def release(self) -> None:
        if self.handle is None:
            return
        kernel32 = _kernel32()
        try:
            if kernel32 is not None:
                kernel32.ReleaseMutex(self.handle)
                kernel32.CloseHandle(self.handle)
        finally:
            self.handle, self.state = None, "unavailable"


class _FileLockLayer:
    """Exclusive byte-range lock on a machine-local file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None
        self.state = "unsupported"

    def acquire(self, deadline: float) -> None:
        try:
            handle = open(self.path, "a+b")  # noqa: SIM115 - lifetime is the lock
        except OSError as exc:
            raise LocalLockUnavailable(
                f"the machine-local writer lock file {self.path} could not be opened: {exc}"
            ) from exc
        try:
            while True:
                outcome = self._try_lock(handle)
                if outcome is None:
                    self.state = "unsupported"
                    handle.close()
                    return
                if outcome:
                    self.handle, self.state = handle, "held"
                    return
                if time.monotonic() >= deadline:
                    raise LocalLockUnavailable(
                        f"another process on this machine holds the writer lock file "
                        f"{self.path}; refusing to publish concurrently"
                    )
                time.sleep(_POLL_SECONDS)
        except BaseException:
            try:
                handle.close()
            except OSError:
                pass
            raise

    @staticmethod
    def _try_lock(handle) -> bool | None:
        """``True`` locked, ``False`` somebody else holds it, ``None`` unsupported.

        The three answers must stay distinct. Reporting "locked" when no locking
        primitive exists is the worst of the three: the caller marks the layer
        held, telemetry says ``local_lock.held`` is true, and two processes
        exclude each other via nothing at all.
        """
        if sys.platform == "win32":
            try:
                import msvcrt
            except ImportError:  # pragma: no cover - msvcrt is stdlib on Windows
                return None
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        try:
            import fcntl
        except ImportError:  # pragma: no cover - no advisory locking available
            return None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def release(self) -> None:
        handle, self.handle = self.handle, None
        self.state = "unavailable"
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ImportError):
            pass
        finally:
            try:
                handle.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# per-key guard (threads + both OS layers)
# ---------------------------------------------------------------------------
@dataclass
class _Guard:
    key: str
    rlock: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0
    mutex: _MutexLayer | None = None
    file_lock: _FileLockLayer | None = None
    info: LocalLockInfo | None = None


_GUARDS: dict[str, _Guard] = {}
_GUARDS_LOCK = threading.Lock()

_OWNER_INSTANCE = os.urandom(8).hex()


def _owner_marker_path(key: str) -> Path:
    return _lock_directory() / f"{key}.owner.json"


def _claim_owner_marker(key: str) -> bool:
    """Record that this process holds the lock; report a previous owner's death.

    ``WAIT_ABANDONED`` alone cannot detect the case that actually happens here.
    When the killed process held the *only* handle to the named mutex, the last
    handle closes with it, the kernel destroys the object, and the next
    ``CreateMutexW`` creates a brand-new unowned mutex - so the next waiter gets
    ``WAIT_OBJECT_0`` and is told nothing. That is precisely the lone-GUI native
    crash this repo has a history of. A marker written under the lock and
    removed only on a *clean* release survives the kill and makes the previous
    owner's death observable.

    Returns ``True`` when a previous owner left without releasing.
    """
    marker = _owner_marker_path(key)
    abandoned = False
    try:
        raw = marker.read_text(encoding="utf-8")
    except OSError:
        raw = ""
    if raw.strip():
        try:
            previous = json.loads(raw)
            abandoned = isinstance(previous, dict) and previous.get("instance") != _OWNER_INSTANCE
        except ValueError:
            abandoned = True
    try:
        marker.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "instance": _OWNER_INSTANCE,
                    "since": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    return abandoned


def _clear_owner_marker(key: str) -> None:
    """Clean release. Anything that skips this leaves the marker as evidence."""
    try:
        _owner_marker_path(key).unlink(missing_ok=True)
    except OSError:
        pass


def _guard_for(key: str) -> _Guard:
    with _GUARDS_LOCK:
        guard = _GUARDS.get(key)
        if guard is None:
            guard = _Guard(key=key)
            _GUARDS[key] = guard
        return guard


@contextmanager
def local_writer_lock(key: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
    """Hold the machine-local writer exclusion for ``key``.

    Re-entrant for the same thread, so the publish transaction can wrap the
    lease acquisition that also takes it. The OS layers are taken once, at the
    outermost entry, and released when the outermost block exits - including on
    an exception, which is what keeps a failed publish from leaving the machine
    locked out of its own writer slot.
    """
    key = str(key)
    guard = _guard_for(key)
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))

    if not guard.rlock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise LocalLockUnavailable(
            f"another thread in this process has held the writer lock {key} for longer "
            "than the timeout; refusing to publish concurrently"
        )
    try:
        if guard.depth == 0:
            name = _MUTEX_PREFIX + key
            lock_file = _lock_directory() / f"{key}.lock"
            mutex = _MutexLayer(name)
            file_lock = _FileLockLayer(lock_file)
            mutex.acquire(deadline)
            try:
                file_lock.acquire(deadline)
                if mutex.state != "held" and file_lock.state != "held":
                    # Neither OS layer is real on this box, so nothing but a
                    # per-process RLock stands between two publishers. Failing
                    # closed here is the whole point: reporting held=True with
                    # no exclusion in force is worse than refusing to publish.
                    raise LocalLockUnavailable(
                        "no machine-local exclusion primitive is available (named mutex: "
                        f"{mutex.state}, file lock: {file_lock.state}); refusing to publish "
                        "shared output that a second process on this machine could be "
                        "publishing at the same time"
                    )
            except BaseException:
                mutex.release()
                raise
            abandoned = _claim_owner_marker(key) or mutex.abandoned
            guard.mutex, guard.file_lock = mutex, file_lock
            guard.info = LocalLockInfo(
                key=key,
                name=name,
                held=True,
                mutex=mutex.state,
                file_lock=file_lock.state,
                abandoned_by_previous_owner=abandoned,
                lock_file=str(lock_file),
            )
        guard.depth += 1
        assert guard.info is not None
        guard.info.depth = guard.depth
        info = guard.info
    except BaseException:
        guard.rlock.release()
        raise

    try:
        yield info
    finally:
        try:
            guard.depth -= 1
            if guard.depth == 0:
                _clear_owner_marker(key)
                if guard.file_lock is not None:
                    guard.file_lock.release()
                if guard.mutex is not None:
                    guard.mutex.release()
                guard.mutex = guard.file_lock = None
                if guard.info is not None:
                    guard.info.held = False
                    guard.info.depth = 0
                    guard.info.mutex = "unavailable"
                    guard.info.file_lock = "unavailable"
        finally:
            guard.rlock.release()
