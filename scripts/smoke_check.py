"""Deterministic smoke checks (plan.md Phase 0.5): no network, no market hours.

Run: python scripts/smoke_check.py
Exit code 0 = every check passed. Each check is independent and reports its
own failure so a broken area is named instead of a generic crash.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def check_qt_shell() -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    import ui.app  # noqa: F401  (import cost + syntax/deps)

    return f"Qt {app.platformName()} shell imports"


def check_scanner_imports() -> str:
    from master_avwap_lib import runner

    assert callable(runner.run_master)
    assert callable(runner._run_master_impl)
    return "master scanner imports (manifest wrapper present)"


def check_scheduler_tick() -> str:
    import autopilot_core as core

    now = datetime(2026, 7, 10, 10, 0)
    slots = core.get_autopilot_swing_slots(now)
    assert slots, "no swing slots for a weekday"
    return f"scheduler produced {len(slots)} slots"


def check_report_render() -> str:
    import autopilot_core as core

    payload = {
        "generated_at": "2026-07-10 13:00:00",
        "enabled": True,
        "auto_mode": "DESK",
        "ib_status": "connected",
        "regime": "bullish_weak",
        "longs": ["NVDA"],
        "shorts": [],
        "swing_picks": [],
        "alerts": [],
        "slots_done": [],
        "next_slot": "11:00",
        "log_lines": [],
        "auto_longs": [],
        "auto_shorts": [],
    }
    text = core.render_away_report(payload)
    assert "TRADINGBOT AUTO PILOT" in text and "Freshness:" in text
    with tempfile.TemporaryDirectory() as tmp:
        result = core.publish_away_report(payload, Path(tmp) / "report.txt")
        assert result["ok"] and result["verified"], result
    return "away report renders + verified publish"


def check_engines() -> str:
    from market_state import M5Bar, MarketStateEngine
    from relative_strength import CandidateInput, RelativeStrengthEngine

    start = datetime(2026, 7, 10, 9, 35)
    closes = [500 + i * 0.4 for i in range(12)]
    bars = []
    prev = 500.0
    for i, close in enumerate(closes):
        bars.append(
            M5Bar(
                ts=start + timedelta(minutes=5 * i),
                open=prev,
                high=max(prev, close) + 0.4,
                low=min(prev, close) - 0.4,
                close=close,
                volume=1_000_000,
            )
        )
        prev = close
    engine = MarketStateEngine(500.0)
    for bar in bars:
        snapshot = engine.on_bar(bar)
    assert snapshot.state.name in ("BULL_IMPULSE", "COUNTERMOVE_ARMED"), snapshot.state

    rs = RelativeStrengthEngine()
    ranks = rs.rank(bars, [CandidateInput(symbol="TEST", side_sign=1, stock_bars=bars)])
    assert ranks and ranks[0].symbol == "TEST"
    return "market-state + relative-strength engines run"


def check_candidate_registry() -> str:
    from candidate_registry import CandidateRegistry

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "registry.json"
        registry = CandidateRegistry()
        registry.add("AAPL", "LONG", "focus")
        registry.save(path)
        assert CandidateRegistry.load(path).get("AAPL", "LONG") is not None
    return "candidate registry round-trips"


def check_diagnostics() -> str:
    from diagnostics import ManifestRecorder

    with tempfile.TemporaryDirectory() as tmp:
        recorder = ManifestRecorder(job_type="smoke")
        recorder.record_phase("noop", 0.01)
        recorder.finalize("ok")
        path = recorder.save(tmp)
        assert path.exists()
    return "run manifest writes"


CHECKS = (
    check_qt_shell,
    check_scanner_imports,
    check_scheduler_tick,
    check_report_render,
    check_engines,
    check_candidate_registry,
    check_diagnostics,
)


def run_smoke() -> list[tuple[str, bool, str]]:
    results = []
    for check in CHECKS:
        name = check.__name__.removeprefix("check_")
        try:
            detail = check()
            results.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001 - each check reports independently
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> int:
    results = run_smoke()
    width = max(len(name) for name, _, _ in results)
    failures = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
        failures += 0 if ok else 1
    print(f"\n{len(results) - failures}/{len(results)} smoke checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
