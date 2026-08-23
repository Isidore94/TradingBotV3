"""R10.A / MAJOR-7 - the corrected finals reach the learning state in SHADOW first.

No detector reads the ledger, but the outcome CSV does feed the learning chain:
finals -> `build_intraday_bounce_performance_rows` -> `refresh_bounce_learning_state`
-> `_evaluate_bounce_alert_quality` -> `muted` / tier -> whether an alert is
suppressed at all. The D2 and MAJOR-3 corrections move segment averages - a
stop-out that used to score 0R now does not - so the first refresh after them can
flip a segment's mute or proven verdict.

That is live alert behaviour changing from a **data correction**. Legitimate, and
still a change plan.md sec 5 says to measure before taking. So the refresh writes
a state file beside the live one and a diff of every segment whose mute or proven
verdict would move; the live state is untouched until the trader flips one
switch. **Default shadow.**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _Host:
    pass


def _host():
    from bounce_bot_lib.legacy import BounceBot

    host = _Host.__new__(_Host)
    host.LEARNING_REFRESH_MODE_SETTING = BounceBot.LEARNING_REFRESH_MODE_SETTING
    host._learning_refresh_is_live = BounceBot._learning_refresh_is_live.__get__(host, _Host)
    host._learning_state_diff = BounceBot._learning_state_diff
    host.refresh_learning_state_with_shadow = (
        BounceBot.refresh_learning_state_with_shadow.__get__(host, _Host)
    )
    return host


def _state(segments):
    return {"segments": segments}


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------
def test_shadow_is_the_default():
    from unittest import mock

    host = _host()
    for value in ("", "shadow", "nonsense", None):
        with mock.patch("project_paths.get_local_setting", return_value=value):
            assert host._learning_refresh_is_live() is False


def test_only_live_flips_it():
    from unittest import mock

    host = _host()
    with mock.patch("project_paths.get_local_setting", return_value="live"):
        assert host._learning_refresh_is_live() is True
    with mock.patch("project_paths.get_local_setting", side_effect=RuntimeError):
        assert host._learning_refresh_is_live() is False, "an unreadable setting stays shadow"


# ---------------------------------------------------------------------------
# the diff
# ---------------------------------------------------------------------------
def test_a_segment_that_would_be_muted_is_named():
    from bounce_bot_lib.legacy import BounceBot

    live = _state({"bounce_type": {"vwap_reclaim": {"muted": False, "proven": True,
                                                    "sample_count": 40, "avg_close_r": 0.31}}})
    shadow = _state({"bounce_type": {"vwap_reclaim": {"muted": True, "proven": False,
                                                      "sample_count": 40, "avg_close_r": -0.22}}})
    diff = BounceBot._learning_state_diff(live, shadow)
    assert diff["changed_segments"] == 1
    assert diff["would_mute"] == 1 and diff["would_unprove"] == 1
    change = diff["changes"][0]
    assert change["segment"] == "vwap_reclaim"
    assert change["avg_close_r"] == [0.31, -0.22]
    assert change["sample_count"] == [40, 40]


def test_a_segment_whose_average_moves_without_changing_a_verdict_is_not_listed():
    """Listing it would bury the ones that actually reach an alert."""
    from bounce_bot_lib.legacy import BounceBot

    live = _state({"bounce_type": {"a": {"muted": False, "proven": False, "avg_close_r": 0.10}}})
    shadow = _state({"bounce_type": {"a": {"muted": False, "proven": False, "avg_close_r": 0.40}}})
    assert BounceBot._learning_state_diff(live, shadow)["changed_segments"] == 0


def test_a_segment_that_appears_or_disappears_is_flagged():
    from bounce_bot_lib.legacy import BounceBot

    live = _state({"bounce_type": {"a": {"muted": True, "proven": False}}})
    shadow = _state({"bounce_type": {"b": {"muted": False, "proven": True}}})
    diff = BounceBot._learning_state_diff(live, shadow)
    by_segment = {item["segment"]: item for item in diff["changes"]}
    assert by_segment["a"]["disappeared"] is True
    assert by_segment["b"]["appeared"] is True


def test_two_empty_states_diff_to_nothing():
    from bounce_bot_lib.legacy import BounceBot

    assert BounceBot._learning_state_diff({}, {})["changed_segments"] == 0


# ---------------------------------------------------------------------------
# the refresh itself
# ---------------------------------------------------------------------------
def test_the_shadow_refresh_leaves_the_live_state_untouched(tmp_path, monkeypatch):
    import bounce_bot_lib.learning as learning

    live_path = tmp_path / "intraday_bounce_learning_state.json"
    live_path.write_text(json.dumps(_state({"bounce_type": {"a": {"muted": False, "proven": False}}})),
                         encoding="utf-8")
    before = live_path.read_bytes()
    monkeypatch.setattr(learning, "BOUNCE_LEARNING_STATE_FILE", live_path)
    monkeypatch.setattr(
        learning, "refresh_bounce_learning_state",
        lambda **kwargs: (
            Path(kwargs["path"]).write_text(
                json.dumps(_state({"bounce_type": {"a": {"muted": True, "proven": False}}})),
                encoding="utf-8",
            ),
            _state({"bounce_type": {"a": {"muted": True, "proven": False}}}),
        )[1],
    )
    monkeypatch.setattr(
        learning, "load_bounce_learning_state",
        lambda path=None: json.loads(live_path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr("project_paths.get_local_setting", lambda *a, **k: "shadow")
    monkeypatch.setattr("project_paths.get_diagnostics_dir", lambda: tmp_path)

    host = _host()
    summary = host.refresh_learning_state_with_shadow()

    assert summary["mode"] == "shadow"
    assert summary["would_mute"] == 1
    assert live_path.read_bytes() == before, "the live state is frozen"
    assert (tmp_path / "intraday_bounce_learning_state.shadow.json").exists()
    report = json.loads((tmp_path / "bounce_learning_shadow_diff.json").read_text(encoding="utf-8"))
    assert report["changed_segments"] == 1
    assert report["setting"] == "bounce_learning_refresh_mode"


def test_live_mode_writes_the_live_state_and_no_diff(tmp_path, monkeypatch):
    import bounce_bot_lib.learning as learning

    written: list[dict] = []
    monkeypatch.setattr(
        learning, "refresh_bounce_learning_state",
        lambda **kwargs: written.append(kwargs) or _state({"bounce_type": {"a": {}}}),
    )
    monkeypatch.setattr(learning, "load_bounce_learning_state", lambda path=None: {})
    monkeypatch.setattr("project_paths.get_local_setting", lambda *a, **k: "live")

    host = _host()
    summary = host.refresh_learning_state_with_shadow()
    assert summary["mode"] == "live"
    assert written == [{}], "no path override: it writes the live state"


def test_the_after_close_worker_uses_the_shadow_aware_refresh():
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    assert "refresh_learning_state_with_shadow()" in source
    assert "refresh_bounce_learning_state()" not in source


def test_the_refresh_date_is_stamped_only_after_the_work_ran():
    """A raising sweep or refresh used to wait a whole weekday for its retry."""
    import inspect

    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot._maybe_refresh_learning_after_close)
    stamp_at = source.index("self._learning_refresh_date = today")
    worker_at = source.index("def worker():")
    assert worker_at < stamp_at, "the stamp is inside the worker, after the work"
