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
