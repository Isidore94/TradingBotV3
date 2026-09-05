"""Phase 2 review guidance: policy contract, scoring, and queue ordering."""

import json
import sys

import pytest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _queue_mechanics_only(monkeypatch):
    """Routing off: these tests are about what the QUEUE does with a row.

    Since 2026-08-27 an ordinary intraday alert lists in the M5 alert bar
    instead of queueing a chart (trader rule; `test_qt_m5_alert_bar.py` owns
    that routing and its exemptions). The mechanics below are the same for
    any row the queue holds, so they are exercised with the routing switched
    off rather than rewritten around D1 fixtures.
    """
    from ui.panels.alert_center_panel import AlertCenterPanel

    monkeypatch.setattr(
        AlertCenterPanel, "_is_m5_review_alert", staticmethod(lambda alert: False)
    )

from review_guidance import (
    AlertGuidance,
    ReviewGuide,
    alert_segments,
    build_guidance,
)
from review_policy import (
    PolicyRule,
    draft_policy_from_state,
    load_review_policy,
    save_review_policy,
)


def _fields(**overrides):
    fields = dict(
        symbol="NVDA",
        side="LONG",
        tier="A",
        tag="green",
        timeframe="M5",
        is_d1=False,
        bounce_types="dynamic_vwap_upper_band",
        market_environment="BULLISH_WEAK",
        session_rvol=2.1,
        rrs_spy=1.4,
    )
    fields.update(overrides)
    return fields


def _state(**overrides):
    state = {
        "schema": "review_learning_v1",
        "overall_take_rate": 0.3,
        "dimensions": {
            "tier": {
                "A": {
                    "shown": 20,
                    "take_rate_shrunk": 0.55,
                    "taken": {"r_n": 9, "r_avg": 0.42, "fwd_n": 0, "fwd_avg_pct": None},
                    "passed": {"r_n": 6, "r_avg": 0.10, "fwd_n": 0, "fwd_avg_pct": None},
                },
                "B": {
                    "shown": 15,
                    "take_rate_shrunk": 0.10,
                    "taken": {"r_n": 0, "r_avg": None, "fwd_n": 0, "fwd_avg_pct": None},
                    "passed": {"r_n": 12, "r_avg": 0.55, "fwd_n": 0, "fwd_avg_pct": None},
                },
                # Too thin to contribute anything.
                "S": {
                    "shown": 2,
                    "take_rate_shrunk": 0.90,
                    "taken": {"r_n": 1, "r_avg": 3.0, "fwd_n": 0, "fwd_avg_pct": None},
                    "passed": {"r_n": 0, "r_avg": None, "fwd_n": 0, "fwd_avg_pct": None},
                },
            }
        },
        "blind_spots": [
            {
                "dimension": "tier",
                "segment": "B",
                "shown": 15,
                "take_rate": 0.0,
                "take_rate_shrunk": 0.10,
                "passed_r_avg": 0.55,
                "passed_r_n": 12,
            }
        ],
        "leaks": [],
    }
    state.update(overrides)
    return state


def test_alert_segments_match_scoreboard_dimensions():
    segments = dict(alert_segments(_fields(), now=datetime(2026, 7, 28, 10, 0)))
    assert segments["tier"] == "A"
    assert segments["side"] == "LONG"
    assert segments["alert_kind"] == "m5"
    assert segments["rvol_bucket"] == "elevated(1.5-2.5)"
    assert segments["rrs_alignment"] == "aligned"
    assert ("bounce_type", "dynamic_vwap_upper_band") in alert_segments(_fields())


def test_build_guidance_scores_from_state_and_gates_thin_segments():
    guidance = build_guidance(_fields(), _state(), [], now=datetime(2026, 7, 28, 10, 0))
    # tier A contributes 0.55; the thin S segment must not.
    assert guidance.take_prob == 0.55
    assert guidance.edge_r == 0.42  # taken preferred over passed
    assert guidance.score == round(0.55 * 100 + 0.42 * 20, 2)
    assert guidance.notes == []

    # A B-tier alert picks up the blind-spot callout.
    b_alert = build_guidance(
        _fields(tier="B"), _state(), [], now=datetime(2026, 7, 28, 10, 0)
    )
    assert any("Blind spot" in note for note in b_alert.notes)
    assert "take 0%" in b_alert.notes[0]


def test_build_guidance_applies_policy_rules():
    rules = [
        PolicyRule(
            dimension="bounce_type",
            segment="dynamic_vwap_upper_band",
            priority_delta=3,
            annotation="Your best segment - front of the queue.",
            watch_kind="band_bounce",
            fill_source="upper_1",
        ),
        PolicyRule(dimension="tier", segment="D", priority_delta=-4),  # no match
    ]
    guidance = build_guidance(_fields(), None, rules)
    assert guidance.policy_delta == 3
    assert guidance.score == 30.0  # no state: policy only
    assert guidance.watch_kind == "band_bounce"
    assert "your usual arm: band_bounce @ upper_1" in guidance.summary_text()
    assert "Your best segment" in guidance.summary_text()


def test_guidance_is_neutral_with_no_documents():
    guidance = build_guidance(_fields(), None, [])
    assert guidance.score == 0.0
    assert guidance.take_prob is None
    assert guidance.summary_text() == ""
    disabled = ReviewGuide(None, None)
    assert not disabled.enabled
    assert disabled.guidance_for(SimpleNamespace(symbol="NVDA", side="LONG")).score == 0.0


def test_review_guide_reloads_documents_on_mtime_change(tmp_path):
    state_path = tmp_path / "state.json"
    policy_path = tmp_path / "policy.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    guide = ReviewGuide(state_path, policy_path)
    alert = SimpleNamespace(
        symbol="NVDA",
        side="LONG",
        raw_text="[A-TIER] NVDA: Bounce confirmed (long)",
        tag="green",
        timeframe="M5",
        is_d1=False,
        trigger="",
        payload={"feedback": {"context_json": json.dumps({"session_rvol": 2.1, "rrs_spy": 1.4, "market_environment": "BULLISH_WEAK"}), "bounce_types": "dynamic_vwap_upper_band"}},
    )
    first = guide.guidance_for(alert)
    assert first.take_prob == 0.55

    # Promote a policy afterwards - picked up without a restart.
    save_review_policy(
        [PolicyRule(dimension="tier", segment="A", priority_delta=5)], policy_path
    )
    import os
    import time

    later = time.time() + 5
    os.utime(policy_path, (later, later))
    second = guide.guidance_for(alert)
    assert second.policy_delta == 5
    assert second.score > first.score


def test_policy_round_trip_and_draft_generation(tmp_path):
    path = tmp_path / "policy.json"
    rules = [
        PolicyRule("tier", "B", priority_delta=99, annotation="clamped"),
        PolicyRule("side", "SHORT", priority_delta=-2),
    ]
    save_review_policy(rules, path, author="fable", notes="test")
    loaded = load_review_policy(path)
    assert [rule.key() for rule in loaded] == [("tier", "B"), ("side", "SHORT")]
    assert loaded[0].priority_delta == 5  # clamped to MAX_PRIORITY_DELTA

    draft = draft_policy_from_state(
        _state(
            leaks=[
                {
                    "dimension": "time_bucket",
                    "segment": "midday",
                    "shown": 10,
                    "take_rate": 0.6,
                    "taken_r_avg": -0.4,
                    "taken_r_n": 9,
                }
            ]
        )
    )
    by_key = {rule.key(): rule for rule in draft}
    assert by_key[("tier", "B")].priority_delta > 0
    assert "Blind spot" in by_key[("tier", "B")].annotation
    assert by_key[("time_bucket", "midday")].priority_delta < 0
    assert "Leak" in by_key[("time_bucket", "midday")].annotation


# ---------------------------------------------------------------------------
# Alert Center integration (offscreen Qt; skipped when PySide6 is unavailable)
# ---------------------------------------------------------------------------
def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _bounce_alert(symbol, tier, **overrides):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="10:15:00",
        symbol=symbol,
        side=overrides.pop("side", "LONG"),
        trigger="Bounce confirmed",
        timeframe="M5",
        tag=overrides.pop("tag", "green"),
        raw_text=f"[{tier}-TIER] {symbol}: Bounce confirmed (long)",
        payload=overrides.pop("payload", {}),
        **overrides,
    )


def test_panel_orders_review_queue_by_guidance_score(tmp_path):
    """Characterization replay of the pre-gate preference ordering.

    Phase 0 task 6 of docs/archive/GUI_TRADE_DISCOVERY_LEARNING_PLAN.md requires this
    behavior to stay reproducible behind the switch while the active queue
    runs FIFO, so the champion can be restored without a code revert.
    """
    if _qt_app() is None:
        return
    from ui.panels.alert_center_panel import AlertCenterPanel

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    panel = AlertCenterPanel(
        review_events_path=tmp_path / "events.jsonl",
        review_guide=ReviewGuide(
            state_path, tmp_path / "policy.json", ordering_mode="preference"
        ),
    )
    # First alert occupies the pane; the queue then receives B (low score)
    # and A (high score) - A must jump ahead of B.
    panel._enqueue_review_alert(_bounce_alert("FIRST", "A"))
    panel._enqueue_review_alert(_bounce_alert("LOWB", "B"))
    panel._enqueue_review_alert(_bounce_alert("HIGHA", "A"))
    assert [alert.symbol for alert in panel._review_queue] == ["HIGHA", "LOWB"]

    # A chart-watch hit still beats every score to the front.
    watch_hit = _bounce_alert("WATCHED", "B", tag="chart_watch")
    panel._enqueue_review_alert(watch_hit)
    assert panel._review_queue[0].symbol == "WATCHED"

    # And the guidance line reaches the review pane for the shown alert.
    assert panel.chart_review.guidance_label.isVisibleTo(panel.chart_review)
    assert "take-prob" in panel.chart_review.guidance_label.text()

    # The shown impression recorded what the guidance claimed.
    from review_events import load_review_events

    rows = load_review_events(tmp_path / "events.jsonl")
    shown = [row for row in rows if row["action"] == "shown"]
    assert shown and shown[0]["detail"]["guidance_score"] > 0
    assert shown[0]["detail"]["queue_ordering"] == "preference"


def test_default_ordering_mode_is_annotation_only(monkeypatch):
    from review_guidance import (
        ORDERING_ANNOTATION_ONLY,
        ORDERING_MODE_ENV_VAR,
        ORDERING_PREFERENCE,
        resolve_ordering_mode,
    )

    monkeypatch.delenv(ORDERING_MODE_ENV_VAR, raising=False)
    assert resolve_ordering_mode() == ORDERING_ANNOTATION_ONLY
    assert resolve_ordering_mode("preference") == ORDERING_PREFERENCE
    # A typo must fail closed, never re-enable ungated ordering.
    assert resolve_ordering_mode("prefrence") == ORDERING_ANNOTATION_ONLY

    gated = ReviewGuide(None, None)
    assert not gated.orders_queue
    assert gated.queue_score(AlertGuidance(score=93.4)) == 0.0
    assert ReviewGuide(None, None, ordering_mode="preference").queue_score(
        AlertGuidance(score=93.4)
    ) == 93.4

    monkeypatch.setenv(ORDERING_MODE_ENV_VAR, "preference")
    assert resolve_ordering_mode() == ORDERING_PREFERENCE
    # An explicit argument still wins over the environment.
    assert resolve_ordering_mode("annotation_only") == ORDERING_ANNOTATION_ONLY


def test_gated_guidance_annotates_without_reordering_the_queue(tmp_path, monkeypatch):
    """Phase 0 gate: a populated scoreboard annotates but cannot reorder."""
    if _qt_app() is None:
        return
    from review_guidance import ORDERING_MODE_ENV_VAR
    from ui.panels.alert_center_panel import AlertCenterPanel

    monkeypatch.delenv(ORDERING_MODE_ENV_VAR, raising=False)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    panel = AlertCenterPanel(
        review_events_path=tmp_path / "events.jsonl",
        review_guide=ReviewGuide(state_path, tmp_path / "policy.json"),
    )
    panel._enqueue_review_alert(_bounce_alert("FIRST", "A"))
    panel._enqueue_review_alert(_bounce_alert("LOWB", "B"))
    panel._enqueue_review_alert(_bounce_alert("HIGHA", "A"))
    # Same documents as the characterization test above, but arrival order wins.
    assert [alert.symbol for alert in panel._review_queue] == ["LOWB", "HIGHA"]

    # An armed chart-watch hit is a trader instruction, not preference - it
    # still goes to the front under the gate.
    panel._enqueue_review_alert(_bounce_alert("WATCHED", "B", tag="chart_watch"))
    assert panel._review_queue[0].symbol == "WATCHED"

    # The annotation still reaches the chart and the impression still records
    # the score - only the queue position is withheld.
    assert "take-prob" in panel.chart_review.guidance_label.text()
    from review_events import load_review_events

    shown = [
        row
        for row in load_review_events(tmp_path / "events.jsonl")
        if row["action"] == "shown"
    ]
    assert shown and shown[0]["detail"]["guidance_score"] > 0
    assert shown[0]["detail"]["queue_ordering"] == "annotation_only"


def test_panel_queue_stays_fifo_without_documents(tmp_path):
    if _qt_app() is None:
        return
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    panel._enqueue_review_alert(_bounce_alert("FIRST", "A"))
    for symbol, tier in (("ONE", "B"), ("TWO", "S"), ("THREE", "D")):
        panel._enqueue_review_alert(_bounce_alert(symbol, tier))
    assert [alert.symbol for alert in panel._review_queue] == ["ONE", "TWO", "THREE"]
