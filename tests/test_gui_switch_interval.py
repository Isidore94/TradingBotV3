"""The desk shortens CPython's switch interval at launch (GIL waits on Qt paint).

2026-09-30: two thirds of the day's GUI blocked time sat in the bare event loop
with only Setup Tracker paint frames inside and 5 CPU-s/min on the Qt thread -
the main thread waiting for the interpreter lock behind pure-Python workers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui import interpreter_tuning as tuning  # noqa: E402


def test_default_is_one_millisecond():
    assert tuning.resolve_switch_interval_ms({}) == 1.0


def test_env_override_is_honoured_and_clamped():
    key = tuning.ENV_SWITCH_INTERVAL_MS
    assert tuning.resolve_switch_interval_ms({key: "2.5"}) == 2.5
    assert tuning.resolve_switch_interval_ms({key: "0"}) == tuning.MIN_SWITCH_INTERVAL_MS
    assert tuning.resolve_switch_interval_ms({key: "9999"}) == tuning.MAX_SWITCH_INTERVAL_MS
    assert tuning.resolve_switch_interval_ms({key: "nope"}) == tuning.DEFAULT_SWITCH_INTERVAL_MS
    assert tuning.resolve_switch_interval_ms({key: "nan"}) == tuning.DEFAULT_SWITCH_INTERVAL_MS


def test_configure_passes_seconds_to_the_setter():
    seen: list[float] = []
    applied = tuning.configure_switch_interval({tuning.ENV_SWITCH_INTERVAL_MS: "2"}, setter=seen.append)
    assert applied == 2.0
    assert seen == [0.002]


def test_the_desk_configures_it_before_the_application_is_built():
    text = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    call = text.index("configure_switch_interval()")
    assert call < text.index("app = QApplication(sys.argv[:1])")
