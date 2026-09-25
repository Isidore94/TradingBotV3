"""Own BounceBot in one below-normal child process (SN1).

The detector stays untouched.  Commands cross one pipe and every GUI callback
crosses one bounded queue.  The proxy deliberately looks like the small bot
surface the desk already reads.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import multiprocessing
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable
from swallowed import note_swallowed

MESSAGE_QUEUE_LIMIT = 8192
START_TIMEOUT_SECONDS = 60.0
RPC_TIMEOUT_SECONDS = 180.0
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _set_below_normal_priority() -> bool:
    """Best effort on Windows; other platforms keep their inherited priority."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        return bool(
            kernel32.SetPriorityClass(
                kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS
            )
        )
    except Exception:
        return False


def _resolve_launcher(spec: str) -> Callable[..., Any]:
    module_name, separator, function_name = str(spec).rpartition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("launcher spec must be module-or-file:function")
    if module_name.endswith(".py") or Path(module_name).exists():
        path = Path(module_name).resolve()
        dynamic_name = f"_sn1_launcher_{abs(hash(str(path)))}"
        module_spec = importlib.util.spec_from_file_location(dynamic_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load launcher from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    launcher = getattr(module, function_name)
    if not callable(launcher):
        raise TypeError(f"launcher {spec!r} is not callable")
    return launcher


def _state(bot: Any) -> dict[str, Any]:
    return {
        "connection_status": bool(getattr(bot, "connection_status", False)),
        "rrs_threshold": float(getattr(bot, "rrs_threshold", 2.0)),
        "rrs_timeframe_key": str(getattr(bot, "rrs_timeframe_key", "5m")),
        "market_environment_user_override": bool(
            getattr(bot, "market_environment_user_override", False)
        ),
    }


def _register_setup_key_bar_source(bot: Any) -> None:
    """P11: the shadow M5 setup-key stamp reads this bot's cached bars (cache only); never fails the child."""
    try:
        import m5_setup_key_stamp

        m5_setup_key_stamp.register_bar_source(bot)
    except Exception as exc:  # noqa: BLE001 - a shadow stamp never costs the scanner
        note_swallowed("M5 setup key bar source not registered", exc, quiet=True)


def _child_main(
    connection,
    events,
    launcher_spec: str,
    start_scanning_enabled: bool,
) -> None:
    """Child entry point. It must remain top-level for Windows spawn."""
    priority_below_normal = _set_below_normal_priority()
    bot = None

    def callback(message: Any, tag: str) -> None:
        # Full means the GUI has stopped draining. Block rather than lose an
        # alert or silently change detector evidence.
        events.put(("callback", message, str(tag or "")), block=True)

    try:
        launcher = _resolve_launcher(launcher_spec)
        bot = launcher(callback, start_scanning_enabled=bool(start_scanning_enabled))
        _register_setup_key_bar_source(bot)
        connection.send(
            {
                "type": "ready",
                "pid": os.getpid(),
                "state": _state(bot),
                "priority_below_normal": priority_below_normal,
            }
        )
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            request_id = request.get("id")
            operation = str(request.get("op") or "")
            name = str(request.get("name") or "")
            try:
                if operation == "get":
                    result = getattr(bot, name)
                elif operation == "call":
                    result = getattr(bot, name)(
                        *(request.get("args") or ()), **(request.get("kwargs") or {})
                    )
                elif operation == "stop":
                    stopper = getattr(bot, "stop", None)
                    if callable(stopper):
                        stop_timeout = (request.get("kwargs") or {}).get("timeout", 5.0)
                        stopper(timeout=float(stop_timeout))
                    else:
                        bot.disconnect()
                    connection.send({"id": request_id, "ok": True, "result": None})
                    bot = None
                    break
                else:
                    raise ValueError(f"unknown child operation {operation!r}")
                connection.send({"id": request_id, "ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001 - returned to the desk
                connection.send(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except Exception as exc:  # noqa: BLE001 - startup must report, not vanish
        try:
            connection.send({"type": "failed", "error": f"{type(exc).__name__}: {exc}"})
        except Exception as swallowed_exc:
            note_swallowed("bot child could not report its startup failure", swallowed_exc)
    finally:
        if bot is not None:
            try:
                stopper = getattr(bot, "stop", None)
                if callable(stopper):
                    stopper(timeout=5.0)
                else:
                    bot.disconnect()
            except Exception:
                try:
                    bot.disconnect()
                except Exception as swallowed_exc:
                    note_swallowed("bot child disconnect failed", swallowed_exc)
        try:
            events.put(("closed", None, ""), timeout=0.5)
        except Exception as swallowed_exc:
            note_swallowed("bot child could not post its closed event", swallowed_exc, quiet=True)
        try:
            connection.close()
        except Exception as swallowed_exc:
            note_swallowed("bot child connection close failed", swallowed_exc, quiet=True)


class BounceProcessProxy:
    """Synchronous command proxy plus one callback-drain thread."""

    is_process_proxy = True

    def __init__(
        self,
        callback: Callable[[Any, str], None],
        *,
        start_scanning_enabled: bool = False,
        launcher_spec: str = "bounce_bot:run_bot_with_gui",
        context=None,
    ) -> None:
        self._context = context or multiprocessing.get_context("spawn")
        parent, child = self._context.Pipe(duplex=True)
        self._connection = parent
        self._events = self._context.Queue(maxsize=MESSAGE_QUEUE_LIMIT)
        self._callback = callback
        self._rpc_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._closed = threading.Event()
        self._process = self._context.Process(
            target=_child_main,
            args=(child, self._events, launcher_spec, bool(start_scanning_enabled)),
            name="bouncebot-scanner",
            daemon=True,
        )
        self._process.start()
        child.close()
        if not parent.poll(START_TIMEOUT_SECONDS):
            self._terminate()
            raise TimeoutError("BounceBot child did not start in time")
        hello = parent.recv()
        if hello.get("type") != "ready":
            self._terminate()
            raise RuntimeError(str(hello.get("error") or "BounceBot child failed to start"))
        self.pid = int(hello.get("pid") or self._process.pid or 0)
        self.priority_below_normal = bool(hello.get("priority_below_normal"))
        self._initial_state = dict(hello.get("state") or {})
        self._event_thread = threading.Thread(
            target=self._drain_events,
            name="bouncebot-callbacks",
            daemon=True,
        )
        self._event_thread.start()

    @property
    def process(self):
        return self._process

    @property
    def connection_status(self) -> bool:
        try:
            return bool(self._get("connection_status"))
        except Exception:
            return False

    @property
    def rrs_threshold(self) -> float:
        return float(self._get("rrs_threshold"))

    @property
    def rrs_timeframe_key(self) -> str:
        return str(self._get("rrs_timeframe_key"))

    @property
    def market_environment_user_override(self) -> bool:
        return bool(self._get("market_environment_user_override"))

    @property
    def latest_bars(self) -> dict:
        # Warehouse capture is the only production caller after the chart path
        # switches to m5_chart_bars. One full snapshot per minute.
        value = self._get("latest_bars")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def d1_zone_arms(self) -> dict:
        value = self._get("d1_zone_arms")
        return dict(value) if isinstance(value, dict) else {}

    def _drain_events(self) -> None:
        while not self._closed.is_set():
            try:
                kind, message, tag = self._events.get(timeout=0.5)
            except queue.Empty:
                if not self._is_alive():
                    break
                continue
            except (EOFError, OSError):
                break
            if kind == "closed":
                break
            if kind == "callback":
                self._callback(message, tag)

    def _rpc(self, operation: str, name: str = "", *args, **kwargs):
        if self._closed.is_set() or not self._is_alive():
            raise RuntimeError("BounceBot child is not running")
        request_id = next(self._ids)
        request = {
            "id": request_id,
            "op": operation,
            "name": name,
            "args": args,
            "kwargs": kwargs,
        }
        with self._rpc_lock:
            self._connection.send(request)
            if not self._connection.poll(RPC_TIMEOUT_SECONDS):
                raise TimeoutError(f"BounceBot child command timed out: {name or operation}")
            response = self._connection.recv()
        if response.get("id") != request_id:
            raise RuntimeError("BounceBot child reply identity mismatch")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "BounceBot child command failed"))
        return response.get("result")

    def _get(self, name: str):
        return self._rpc("get", name)

    def __getattr__(self, name: str):
        if name.startswith("_") and name not in {"_spy_session_bars"}:
            raise AttributeError(name)

        def remote(*args, **kwargs):
            return self._rpc("call", name, *args, **kwargs)

        return remote

    def stop(self, timeout: float = 5.0) -> None:
        if self._closed.is_set():
            return
        try:
            self._rpc("stop", timeout=float(timeout))
        except Exception as swallowed_exc:
            note_swallowed("bot child stop RPC failed; terminating", swallowed_exc, quiet=True)
        self._closed.set()
        self._process.join(max(0.0, float(timeout)))
        if self._is_alive():
            self._terminate()
        if self._event_thread is not threading.current_thread():
            self._event_thread.join(1.0)
        try:
            self._events.close()
            self._events.join_thread()
        except Exception as swallowed_exc:
            note_swallowed("bot child event queue close failed", swallowed_exc, quiet=True)
        try:
            self._connection.close()
        except Exception as exc:
            note_swallowed("bot child connection close failed at stop", exc, quiet=True)

    def disconnect(self) -> None:
        self.stop()

    def _terminate(self) -> None:
        self._closed.set()
        if self._is_alive():
            self._process.terminate()
            self._process.join(2.0)

    def _is_alive(self) -> bool:
        """Read child liveness without leaking a diagnostic PID shim.

        A process-wide test or diagnostic may temporarily replace ``os.getpid``.
        ``multiprocessing.Process.is_alive`` rejects that as a foreign parent;
        the child's exit code is still a safe fallback.
        """
        try:
            return bool(self._process.is_alive())
        except AssertionError:
            return self._process.exitcode is None
