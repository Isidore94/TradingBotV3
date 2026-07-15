import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_environment_annotation_keeps_user_choice_and_auto_context(tmp_path):
    from market_environment_annotations import (
        load_market_environment_annotations,
        record_market_environment_annotation,
    )

    path = tmp_path / "environment.jsonl"
    row = record_market_environment_annotation(
        selected_environment="bearish_weak",
        auto_reading={"env_key": "bullish_weak", "day_pct": 0.31, "possibilities": ["bullish"]},
        session_id="session-1",
        now=datetime(2026, 7, 14, 17, 5, tzinfo=timezone.utc),
        path=path,
    )

    assert row is not None
    assert row["user_mode"] == "bearish_weak"
    assert row["auto_environment"] == "bullish_weak"
    assert row["manual_override_active"] is True
    assert load_market_environment_annotations(path) == [row]

    # Bad/partial cloud-sync lines never hide valid observations.
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{partial\n")
    assert load_market_environment_annotations(path) == [row]


def test_return_to_na_is_an_explicit_logged_event(tmp_path):
    from market_environment_annotations import record_market_environment_annotation

    path = tmp_path / "environment.jsonl"
    row = record_market_environment_annotation(
        selected_environment=None,
        auto_reading={"env_key": "neutral_chop"},
        session_id="session-2",
        event="returned_to_na",
        path=path,
    )

    assert row is not None
    assert row["user_mode"] == "n/a"
    assert row["event"] == "returned_to_na"
    assert row["manual_override_active"] is False
    assert json.loads(path.read_text(encoding="utf-8"))["auto_environment"] == "neutral_chop"


def test_bounce_service_defaults_to_na_and_logs_session_override(tmp_path):
    try:
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.services.bounce_service import BounceService
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    path = tmp_path / "environment.jsonl"
    service = BounceService(environment_annotations_path=path)
    assert service.market_environment is None

    calls = []

    class Bot:
        market_environment_user_override = False

        def set_market_environment(self, env):
            calls.append(("set", env))

        def clear_market_environment_override(self):
            calls.append(("clear", None))

        def get_auto_regime_reading(self):
            return {"env_key": "bullish_weak", "day_pct": 0.2}

    with service._lock:
        service._bot = Bot()

    service.set_market_environment("bearish_weak")
    assert service.market_environment == "bearish_weak"
    service.clear_market_environment_override()
    assert service.market_environment is None
    assert calls == [("set", "bearish_weak"), ("clear", None)]

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["user_mode"] for row in rows] == ["bearish_weak", "n/a"]
    assert all(row["auto_environment"] == "bullish_weak" for row in rows)
