"""yfinance.download is not thread-safe in one process; every caller shares one lock."""

from __future__ import annotations

import ast
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import yahoo_download  # noqa: E402


def test_two_threads_never_overlap_inside_yf_download(monkeypatch):
    import yfinance

    state = {"inside": 0, "peak": 0, "calls": []}
    guard = threading.Lock()

    def fake_download(*args, **kwargs):
        with guard:
            state["inside"] += 1
            state["peak"] = max(state["peak"], state["inside"])
            state["calls"].append((args, kwargs))
        time.sleep(0.05)
        with guard:
            state["inside"] -= 1
        return kwargs.get("tickers")

    monkeypatch.setattr(yfinance, "download", fake_download)
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda n=n: results.append(
            yahoo_download.download(tickers=f"T{n}", period="1d", threads=True)))
        for n in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert state["peak"] == 1
    assert sorted(results) == [f"T{n}" for n in range(6)]
    assert all(kw["period"] == "1d" and kw["threads"] is True for _a, kw in state["calls"])


class _FakeYf:
    """A stand-in yfinance: `download` is whatever the test sets; `multi._time` is time."""

    def __init__(self, download):
        import types

        self.download = download
        self.multi = types.SimpleNamespace(_time=time)


def test_the_process_guard_serialises_direct_yf_download_calls():
    state = {"inside": 0, "peak": 0}
    guard = threading.Lock()

    def fake_download(*args, **kwargs):
        with guard:
            state["inside"] += 1
            state["peak"] = max(state["peak"], state["inside"])
        time.sleep(0.03)
        with guard:
            state["inside"] -= 1
        return "ok"

    fake = _FakeYf(fake_download)
    assert yahoo_download.install_process_guard(fake) is True
    first = fake.download
    assert yahoo_download.install_process_guard(fake) is True
    assert fake.download is first  # idempotent: wrapped once
    # Direct callers (the way legacy.py calls it) and the module's own door share one lock.
    threads = [threading.Thread(target=fake.download, args=(f"T{n}",)) for n in range(4)]
    threads += [threading.Thread(target=yahoo_download.download_frames, args=(["X"],),
                                 kwargs={"yf_module": fake}) for _ in range(2)]
    fake.shared = type("S", (), {"_DFS": {}, "_ERRORS": {}})
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert state["peak"] == 1
    assert isinstance(fake.multi._time, yahoo_download._BoundedTime)


def test_a_download_stuck_in_the_yfinance_wait_times_out_and_frees_the_lock(monkeypatch):
    monkeypatch.setattr(yahoo_download, "CALL_TIMEOUT_SECONDS", 0.2)
    fake = _FakeYf(None)

    def stuck_download(*args, **kwargs):
        # yfinance.multi's wait: a result table that never fills.
        while True:
            fake.multi._time.sleep(0.01)

    fake.download = stuck_download
    yahoo_download.install_process_guard(fake)
    started = time.monotonic()
    try:
        fake.download("SPY")
    except yahoo_download.YahooDownloadTimeout:
        pass
    else:
        raise AssertionError("the stuck download never gave up")
    assert time.monotonic() - started < 3
    # The lock is free again and the deadline is cleared for the next call.
    assert yahoo_download._LOCK.acquire(timeout=1)
    yahoo_download._LOCK.release()
    assert getattr(yahoo_download._DEADLINE, "at", None) is None
    # Outside a guarded call the bounded sleep is plain time.sleep.
    fake.multi._time.sleep(0)


def test_positional_arguments_pass_through(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: (a, k))
    assert yahoo_download.download(["SPY"], interval="5m") == ((["SPY"],), {"interval": "5m"})


# Ask-first detector files run in the scan_worker subprocess; they are left alone.
_EXEMPT = {
    SCRIPTS_DIR / "yahoo_download.py",
    SCRIPTS_DIR / "master_avwap_lib" / "legacy.py",
}


def _direct_download_uses(path: Path) -> list[int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    lines = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == "download"
                and isinstance(node.value, ast.Name) and node.value.id in {"yf", "yfinance"}):
            lines.append(node.lineno)
    return lines


def test_no_module_calls_yf_download_directly():
    offenders = []
    for base in (SCRIPTS_DIR, ROOT_DIR / "market_prep"):
        for path in base.rglob("*.py"):
            if path in _EXEMPT or "bounce_bot_lib" in path.parts:
                continue
            for line in _direct_download_uses(path):
                offenders.append(f"{path.relative_to(ROOT_DIR)}:{line}")
    assert offenders == [], "route these through yahoo_download.download: " + ", ".join(offenders)
