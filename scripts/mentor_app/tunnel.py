"""The app's own ssh tunnel to the RTX 5080's Ollama (a Python port of the night's preflight).

Same steps as ``run_ai_jobs.ps1`` ``Invoke-RemoteGpuPreflight`` (which is not touched):
``ssh -G`` resolves the alias -> Wake-on-LAN via ``~/bin/host-on.ps1`` when port 22 is
shut -> ``scripts/remote_gpu/ollama_up.sh`` piped over ssh (with
``OLLAMA_NUM_PARALLEL=2`` spliced into its serve line) -> ``ssh -N -L`` on the app's
port, restarted by a watchdog thread -> ``/api/version`` probe trusted 60 s.
Blocking calls throughout: run it on a worker thread, never the Qt thread.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from mentor_app import settings

UP_SCRIPT = Path(__file__).resolve().parents[1] / "remote_gpu" / "ollama_up.sh"
#: The serve line in ollama_up.sh the parallel setting is spliced into.
SERVE_ANCHOR = 'tmux new -d -s ollama "'
NUM_PARALLEL = 2
HEALTH_TTL_SECONDS = 60.0
TUNNEL_OPEN_WAIT_SECONDS = 30.0
WAKE_TIMEOUT_SECONDS = 330
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class TunnelStatus:
    ok: bool
    reason: str
    host: str = ""
    woke: bool = False


def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def up_script_text(parallel: int = NUM_PARALLEL, *, source: Path = UP_SCRIPT) -> str:
    """ollama_up.sh with OLLAMA_NUM_PARALLEL added to its serve line (the file is not changed)."""
    text = source.read_text(encoding="utf-8").replace("\r", "")
    if SERVE_ANCHOR not in text:
        logging.warning("Trade Mentor: ollama_up.sh has no serve line to add parallel slots to")
        return text
    return text.replace(SERVE_ANCHOR, f"{SERVE_ANCHOR}OLLAMA_NUM_PARALLEL={int(parallel)} ", 1)


class Tunnel:
    """One ssh forward 127.0.0.1:<port> -> host:11434, kept open by a watchdog thread."""

    def __init__(
        self,
        alias: str,
        port: int,
        *,
        run: Callable[..., Any] = subprocess.run,
        popen: Callable[..., Any] = subprocess.Popen,
        get: Callable[..., Any] = requests.get,
        port_open: Callable[[str, int], bool] = tcp_open,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wake_script: Path | None = None,
    ) -> None:
        self.alias = str(alias)
        self.port = int(port)
        self._run = run
        self._popen = popen
        self._get = get
        self._port_open = port_open
        self._sleep = sleep
        self._clock = clock
        self._wake_script = wake_script or (Path.home() / "bin" / "host-on.ps1")
        self._stop = threading.Event()
        self._proc: Any = None
        self._watchdog: threading.Thread | None = None
        self._checked_at: float | None = None
        self.restarts = 0

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------------ steps
    def resolve_host(self) -> str:
        try:
            done = self._run(
                ["ssh", "-G", self.alias], capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Trade Mentor: ssh -G failed: %s", exc)
            return ""
        for line in str(getattr(done, "stdout", "") or "").splitlines():
            if line.lower().startswith("hostname "):
                return line[9:].strip()
        return ""

    def ensure_host_up(self, host: str) -> tuple[bool, bool]:
        """(up, woke). Sends Wake-on-LAN through host-on.ps1 when port 22 is shut."""
        if self._port_open(host, 22):
            return True, False
        if not self._wake_script.exists():
            return False, False
        try:
            self._run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self._wake_script)],
                capture_output=True, text=True, timeout=WAKE_TIMEOUT_SECONDS, creationflags=_NO_WINDOW,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Trade Mentor: Wake-on-LAN failed: %s", exc)
        return self._port_open(host, 22), True

    def start_ollama(self) -> tuple[bool, str]:
        try:
            done = self._run(
                ["ssh", *SSH_OPTIONS, self.alias, "sed '1s/^\\xEF\\xBB\\xBF//' | bash -s"],
                # Bytes, not text: text mode on Windows would turn every \n into \r\n for bash.
                input=up_script_text().encode("utf-8"), capture_output=True, timeout=90, creationflags=_NO_WINDOW,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"the host start script did not answer ({type(exc).__name__})"
        raw = getattr(done, "stdout", b"") or b""
        out = " ".join((raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)).split())
        return int(getattr(done, "returncode", 1) or 0) == 0, out

    def tunnel_command(self) -> list[str]:
        return [
            "ssh", "-N", "-L", f"127.0.0.1:{self.port}:127.0.0.1:{settings.REMOTE_OLLAMA_PORT}",
            "-o", "ExitOnForwardFailure=yes", *SSH_OPTIONS, self.alias,
        ]

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                self._proc = self._popen(
                    self.tunnel_command(),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=_NO_WINDOW,
                )
                self._proc.wait()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Trade Mentor: tunnel ssh failed: %s", exc)
            if self._stop.is_set():
                break
            self.restarts += 1
            self._checked_at = None
            self._sleep(2.0)

    def open(self) -> bool:
        """Start the watchdog (once) and wait for the local port to answer."""
        if self._port_open("127.0.0.1", self.port):
            return True
        if self._watchdog is None or not self._watchdog.is_alive():
            self._stop.clear()
            self._watchdog = threading.Thread(target=self._watch, name="mentor-tunnel-watchdog", daemon=True)
            self._watchdog.start()
        deadline = self._clock() + TUNNEL_OPEN_WAIT_SECONDS
        while self._clock() < deadline:
            if self._port_open("127.0.0.1", self.port):
                return True
            self._sleep(1.0)
        return self._port_open("127.0.0.1", self.port)

    def alive(self) -> bool:
        """``/api/version`` answers; a pass is trusted for HEALTH_TTL_SECONDS."""
        now = self._clock()
        if self._checked_at is not None and now - self._checked_at < HEALTH_TTL_SECONDS:
            return True
        try:
            healthy = self._get(f"{self.endpoint}/api/version", timeout=5).status_code == 200
        except Exception:  # noqa: BLE001 - any failure means not alive
            healthy = False
        self._checked_at = now if healthy else None
        return healthy

    def preflight(self) -> TunnelStatus:
        if not self.alias:
            return TunnelStatus(False, "no GPU host is set (ai_remote_gpu_ssh_alias)")
        host = self.resolve_host()
        if not host:
            return TunnelStatus(False, f"cannot resolve ssh alias {self.alias!r}")
        up, woke = self.ensure_host_up(host)
        if not up:
            return TunnelStatus(False, f"{host} is off and did not wake", host, woke)
        started, note = self.start_ollama()
        if not started:
            return TunnelStatus(False, f"Ollama did not start on {host}: {note}", host, woke)
        if not self.open():
            return TunnelStatus(False, f"the tunnel to {self.alias} did not open on {self.port}", host, woke)
        if not self.alive():
            return TunnelStatus(False, "the tunnel is open but Ollama does not answer", host, woke)
        return TunnelStatus(True, note or "ready", host, woke)

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass


def from_settings(**kwargs: Any) -> Tunnel:
    return Tunnel(settings.ssh_alias(), settings.tunnel_port(), **kwargs)
