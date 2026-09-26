"""S7: the bounce service owns one shadow-setups timer and worker, and the slot is memory only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from test_s7_shadow_parity_golden import seeded_tapes  # noqa: E402


class _FakeBot:
    def __init__(self):
        self.latest_bars = {f"{name}|5 D|5 mins": bars for name, bars in seeded_tapes().items()}


def test_the_service_owns_one_shadow_timer_armed_only_once_started():
    from ui.services.bounce_service import BounceService

    service = BounceService()
    try:
        assert service._shadow_setups_timer.interval() == 300_000
        assert not service._shadow_setups_timer.isActive()
        assert service._shadow_setups is None
        service.stop()
        assert not service._shadow_setups_timer.isActive()
    finally:
        service.shutdown()


def test_the_slot_is_inert_without_a_bot():
    from ui.services.bounce_service import BounceService

    service = BounceService()
    try:
        service.capture_shadow_setups()
        assert service._shadow_setups is None
    finally:
        service.shutdown()


def test_the_slot_hands_the_cache_to_the_worker_and_shutdown_retires_it(tmp_path, monkeypatch):
    import project_paths
    from ui.services.bounce_service import BounceService

    path = tmp_path / "m5_shadow_setups.jsonl"
    monkeypatch.setattr(project_paths, "M5_SHADOW_SETUPS_FILE", path)
    service = BounceService()
    service._bot = _FakeBot()
    service.capture_shadow_setups()
    capture = service._shadow_setups
    assert capture is not None
    assert capture.wait_idle(20.0)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["shadow_only"] is True for row in rows)
    service._bot = None
    service.shutdown()
    worker = capture._worker
    assert worker is None or not worker.is_alive()
