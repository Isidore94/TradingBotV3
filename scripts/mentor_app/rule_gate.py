"""The plan-rule gate: a small CPU decision model (Kev-4B on llama.cpp) answers one yes/no
question before the plan-inference call: does this trader turn state a standing rule? Qt-free.

The gate can only skip the plan-inference call; it never adds, changes or words a plan line.
Any failure is "no answer", and no answer means plan inference runs as before (fail open).
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from mentor_app import brain, settings

#: The offline cut (18/20 rules caught, 1/53 false alarms on 73 messages); not to be tuned from one session.
CUT = 0.5
TIMEOUT_SECONDS = 10.0
HEALTH_TIMEOUT_SECONDS = 2.0
STOP_WAIT_SECONDS = 5.0
THREADS = 4
CONTEXT_TOKENS = 4096

MODE_KEY = "mentor_rule_gate"
MODES = ("off", "shadow", "on")
DEFAULT_MODE = "shadow"
SERVER_KEY = "mentor_rule_gate_server"
MODEL_KEY = "mentor_rule_gate_model"
PORT_KEY = "mentor_rule_gate_port"
DEFAULT_PORT = 11438
#: One shadow record per inference, keyed by the last new turn id (app_state, JSON).
RECORD_KEY = "rule_gate:{turn_id}"

QUESTION = "rule"
RULE_INSTRUCTIONS = (
    "Does the trader clearly state a standing rule he follows or has decided to follow from now on "
    "(a limit, a time, a goal, a setup he trades, a risk rule, or something he is testing)? A question, "
    "a hypothetical, a maybe, a feeling, a market observation, or a plan for one single trade is NOT a rule."
)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def mode() -> str:
    """``off`` | ``shadow`` | ``on``; default shadow, anything unknown is off."""
    raw = settings._setting(MODE_KEY, DEFAULT_MODE)
    value = str(raw).strip().lower() if isinstance(raw, str) else ""
    return value if value in MODES else "off"


def default_server() -> Path:
    return Path.home() / "llama-decision" / "bin2" / "llama-server.exe"


def default_model() -> Path:
    return Path.home() / "llama-decision" / "models" / "Kev-4B-Q4_K_M.gguf"


def server_path() -> Path:
    return Path(str(settings._setting(SERVER_KEY, "") or "").strip() or default_server())


def model_path() -> Path:
    return Path(str(settings._setting(MODEL_KEY, "") or "").strip() or default_model())


def port() -> int:
    """The gate's local port; never the app tunnel's, the night tunnel's or the fallback port."""
    taken = {settings.tunnel_port(), settings.night_tunnel_port(), settings.FALLBACK_TUNNEL_PORT}
    chosen = settings._as_port(settings._setting(PORT_KEY, DEFAULT_PORT), DEFAULT_PORT)
    if chosen in taken:
        chosen = DEFAULT_PORT
        while chosen in taken:
            chosen += 1
    return chosen


def endpoint(at_port: int | None = None) -> str:
    return f"http://127.0.0.1:{at_port or port()}"


def payload(text: str) -> dict[str, Any]:
    return {
        "state": f"Trader message: {text}",
        "questions": {QUESTION: {"type": "noul", "instructions": RULE_INSTRUCTIONS}},
    }


def score(text: str, *, endpoint: str, post: brain.Post = brain.default_post,
          timeout: float = TIMEOUT_SECONDS) -> float | None:
    """P(the turn states a standing rule), 0..1; None on any failure (down, timeout, bad shape)."""
    try:
        reply = post(f"{endpoint.rstrip('/')}/v1/systemone", payload(text), timeout)
        value = reply["answers"][QUESTION]["noul"]
    except Exception as exc:  # noqa: BLE001 - any failure is "no answer"
        logging.debug("Trade Mentor: rule gate gave no answer (%s)", type(exc).__name__)
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


class GateServer:
    """Owns one local ``llama-server.exe`` child (127.0.0.1 only, CPU, 4 threads, 4k context)."""

    def __init__(
        self,
        exe: Path,
        model: Path,
        port: int,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        get: Callable[[str, float], Any] = brain.default_get,
        exists: Callable[[Path], bool] = Path.is_file,
    ) -> None:
        self.exe = Path(exe)
        self.model = Path(model)
        self.port = int(port)
        self._popen = popen
        self._get = get
        self._exists = exists
        self._lock = threading.Lock()
        self._proc: Any = None
        self._missing_logged = False

    @property
    def endpoint(self) -> str:
        return endpoint(self.port)

    def command(self) -> list[str]:
        return [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(CONTEXT_TOKENS), "-np", "1", "-t", str(THREADS)]

    def running(self) -> bool:
        proc = self._proc
        try:
            return proc is not None and proc.poll() is None
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> bool:
        """Launch the server once (no console window, no wait for load); False when it cannot."""
        with self._lock:
            if self.running():
                return True
            missing = [str(path) for path in (self.exe, self.model) if not self._exists(path)]
            if missing:
                if not self._missing_logged:
                    self._missing_logged = True
                    logging.info("Trade Mentor: rule gate off, file missing: %s", ", ".join(missing))
                return False
            try:
                self._proc = self._popen(
                    self.command(),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=_NO_WINDOW,
                )
            except Exception as exc:  # noqa: BLE001 - a failed launch is "no answer"
                logging.info("Trade Mentor: rule gate server did not start (%s: %s)", type(exc).__name__, exc)
                self._proc = None
                return False
            logging.info("Trade Mentor: rule gate server started on 127.0.0.1:%d", self.port)
            return True

    def ready(self) -> bool:
        """``GET /health`` says ok (503 while the model loads)."""
        try:
            return str((self._get(f"{self.endpoint}/health", HEALTH_TIMEOUT_SECONDS) or {}).get("status")) == "ok"
        except Exception:  # noqa: BLE001
            return False

    def stop(self) -> None:
        """Terminate the child (kill when it does not exit within STOP_WAIT_SECONDS)."""
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                proc.terminate()
                proc.wait(timeout=STOP_WAIT_SECONDS)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            logging.info("Trade Mentor: rule gate server stopped")

    def scorer(self, post: brain.Post = brain.default_post) -> Callable[[str], float | None]:
        return lambda text: score(text, endpoint=self.endpoint, post=post)


def from_settings(**kwargs: Any) -> GateServer:
    return GateServer(server_path(), model_path(), port(), **kwargs)
