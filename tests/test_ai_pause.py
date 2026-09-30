"""Pause AI (trader 2026-09-30): one switch, `ai_paused_until`, stops every local-AI use.

The helper, the one choke point in `ai_summary` every desk caller inherits, and the
Trade Mentor card: answered while paused it writes exactly what it writes with the
brain down (the questions still work; only the AI fill is skipped).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import ai_pause  # noqa: E402
import project_paths  # noqa: E402

PT = ai_pause.PT
#: A Wednesday, 15:00 PT.
NOW = datetime(2026, 9, 30, 15, 0, tzinfo=PT)


@pytest.fixture(autouse=True)
def scratch_settings(tmp_path, monkeypatch):
    """Every test gets its own local settings file."""
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", tmp_path / "local_settings.json")
    project_paths.invalidate_local_settings_cache()
    yield tmp_path / "local_settings.json"
    project_paths.invalidate_local_settings_cache()


# ---------------------------------------------------------------- the helper
def test_nothing_saved_means_not_paused():
    assert ai_pause.paused_until(NOW) is None
    assert ai_pause.is_paused(NOW) is False
    assert ai_pause.reason(NOW) == ""


def test_pause_for_two_hours_is_saved_with_an_offset_and_read_back(scratch_settings):
    until = ai_pause.pause_for("2h", NOW)
    assert until == NOW + timedelta(hours=2)
    saved = json.loads(scratch_settings.read_text(encoding="utf-8"))[ai_pause.PAUSE_KEY]
    assert saved == "2026-09-30T17:00:00-07:00"
    assert ai_pause.is_paused(NOW)
    assert ai_pause.paused_until(NOW) == until
    assert ai_pause.reason(NOW) == "AI paused until 17:00"


def test_four_hours_and_a_timedelta():
    assert ai_pause.pause_for("4h", NOW) == NOW + timedelta(hours=4)
    assert ai_pause.pause_for(timedelta(minutes=30), NOW) == NOW + timedelta(minutes=30)


def test_tonight_is_the_next_0600_pt():
    assert ai_pause.pause_for("tonight", NOW) == datetime(2026, 10, 1, 6, 0, tzinfo=PT)
    early = datetime(2026, 9, 30, 3, 0, tzinfo=PT)
    assert ai_pause.resume_time("tonight", early) == datetime(2026, 9, 30, 6, 0, tzinfo=PT)
    assert ai_pause.reason(NOW) == "AI paused until Thu 06:00"


def test_until_resumed_is_a_far_sentinel_that_only_resume_ends():
    ai_pause.pause_for("until_resumed", NOW)
    assert ai_pause.is_paused(NOW + timedelta(days=3650))
    assert ai_pause.reason(NOW) == "AI paused until you resume"
    ai_pause.resume()
    assert not ai_pause.is_paused(NOW)


def test_an_expired_pause_is_not_a_pause():
    ai_pause.pause_for("2h", NOW)
    assert ai_pause.is_paused(NOW + timedelta(hours=1, minutes=59))
    assert not ai_pause.is_paused(NOW + timedelta(hours=2))
    assert ai_pause.paused_until(NOW + timedelta(hours=3)) is None


@pytest.mark.parametrize("raw", ["junk", "2026-09-30T17:00:00", 5, None])
def test_an_unreadable_or_naive_value_is_not_a_pause(raw):
    project_paths.save_local_setting(ai_pause.PAUSE_KEY, raw)
    assert not ai_pause.is_paused(NOW)


def test_a_malformed_value_is_logged_once_and_an_offset_without_a_colon_is_refused(caplog):
    project_paths.save_local_setting(ai_pause.PAUSE_KEY, "2999-01-01T00:00:00+0700")
    with caplog.at_level("WARNING"):
        assert not ai_pause.is_paused(NOW)
        assert not ai_pause.is_paused(NOW)
    warnings = [r for r in caplog.records if "ai_paused_until" in r.getMessage()]
    assert len(warnings) == 1


def test_an_unknown_length_is_refused_and_writes_nothing(scratch_settings):
    with pytest.raises(ValueError):
        ai_pause.pause_for("forever-ish", NOW)
    with pytest.raises(ValueError):
        ai_pause.pause_for(timedelta(0), NOW)
    assert not scratch_settings.exists()


def test_naive_now_is_read_as_local_time():
    ai_pause.pause_for("2h", datetime.now(timezone.utc))
    assert ai_pause.is_paused(datetime.now())


# ---------------------------------------------------------------- ai_summary: one choke point
ENDPOINT = "http://127.0.0.1:11434/v1"


def test_the_local_provider_is_off_while_paused():
    import ai_summary

    project_paths.save_local_setting(ai_summary.LOCAL_ENDPOINT_SETTING_KEY, ENDPOINT)
    with mock.patch.object(ai_summary, "get_local_setting", project_paths.get_local_setting):
        assert ai_summary.local_provider_enabled() is True
        ai_pause.pause_for("2h")
        assert ai_summary.local_endpoint_url() == ""
        assert ai_summary.local_provider_enabled() is False
        ai_pause.resume()
        assert ai_summary.local_endpoint_url() == ENDPOINT


def test_no_local_request_leaves_while_paused_even_with_an_explicit_endpoint():
    import ai_summary

    calls = []
    ai_pause.pause_for("2h")
    with pytest.raises(RuntimeError, match="AI paused"):
        ai_summary.request_ai_summary(
            provider="local", model="m", api_key="", evidence={}, endpoint="http://127.0.0.1:11436/v1",
            post=lambda *a, **k: calls.append(a),
        )
    assert calls == []


def test_trade_mentor_ai_fill_returns_the_model_down_result_while_paused():
    import ai_summary
    import trade_mentor_ai

    project_paths.save_local_setting(ai_summary.LOCAL_ENDPOINT_SETTING_KEY, ENDPOINT)
    ai_pause.pause_for("4h")
    with mock.patch.object(ai_summary, "get_local_setting", project_paths.get_local_setting):
        with pytest.raises(RuntimeError, match="local AI is not ready"):
            trade_mentor_ai.extract_draft("my words", ("thesis",), {"trade_id": "t1"})


# ---------------------------------------------------------------- the Trade Mentor card
class _Journal:
    def write_entry(self, **_kwargs):
        return {"ok": True, "entry": {}}


class _InlinePool:
    """QThreadPool stand-in: runs the AI-fill worker at once, on this thread."""

    @staticmethod
    def globalInstance():
        return _InlinePool()

    def start(self, worker):
        worker.run()


def _answer_a_card(root: Path, monkeypatch) -> list[tuple[str, str]]:
    """Answer one trade on a card (thesis typed, the rest blank) and return its stored rows."""
    pytest.importorskip("PySide6", reason="the Mentor card is Qt")
    from PySide6.QtWidgets import QApplication
    from tj14b_support import REVIEWED, SESSION, add_round_trip, mark_covered, new_store, pacific, slot_at

    import trade_mentor_trade_check as check
    from ui.widgets import trade_mentor_card
    from ui.widgets.trade_mentor_card import TradeMentorCard

    QApplication.instance() or QApplication([])
    root.mkdir()
    store = new_store(root)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "DRAM")
    card = TradeMentorCard(drafts_path=root / "drafts.json", journal=_Journal())
    card._clock = lambda: pacific(SESSION, 9, 5)
    monkeypatch.setattr(trade_mentor_card, "QThreadPool", _InlinePool)
    card.show_slot(slot_at(SESSION, 9))
    card.set_trade_check(check.build_task(store, SESSION), store=store)
    card._answer_inputs[trade_id]["thesis"][1].setText("Bought the bounce off VWAP.")
    card.show_slot(slot_at(SESSION, 10))
    rows = store.list_opportunity_events(trade_id=trade_id)
    card.close()
    card.deleteLater()
    return sorted(
        (str(row["event_type"]), json.dumps({k: v for k, v in dict(row["payload"]).items()
                                            if not k.endswith(("_at", "_utc", "at"))}, sort_keys=True))
        for row in rows
    )


def test_a_card_answered_while_paused_writes_the_same_rows_as_with_the_brain_down(tmp_path, monkeypatch):
    import ai_summary

    pytest.importorskip("PySide6", reason="the Mentor card is Qt")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(ai_summary, "get_local_setting", project_paths.get_local_setting)
    calls = []

    def model(*_args, **kwargs):
        # A live model that would fill the thesis: it must never be reached while paused.
        calls.append(kwargs)
        draft = {"answers": [{"field": "target", "state": "not_supplied", "text": "no target", "value": None,
                              "unit": "", "source_span": "Bought the bounce off VWAP."}], "follow_up": ""}
        return mock.Mock(status_code=200, json=lambda: {
            "id": "x", "choices": [{"message": {"role": "assistant", "content": json.dumps(draft)}}]})

    monkeypatch.setitem(ai_summary.request_ai_summary.__kwdefaults__, "post", model)

    # The brain down: no local endpoint at all.
    down = _answer_a_card(tmp_path / "down", monkeypatch)
    # Paused: the endpoint is set and a model would answer, but AI is paused.
    project_paths.save_local_setting(ai_summary.LOCAL_ENDPOINT_SETTING_KEY, ENDPOINT)
    ai_pause.pause_for("2h")
    paused = _answer_a_card(tmp_path / "paused", monkeypatch)

    assert calls == [], "no model call while AI is paused"
    assert down, "the answer and its marker are written"
    assert any(kind == "MENTOR_ASKED" for kind, _ in down)
    assert paused == down
