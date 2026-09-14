"""Packet D1C-A items 2 and 3 - where a claimed like GOES, and what it moves.

Trader, 2026-09-14:

    "A successful D1 'Like and claim' must add the pick to Master AVWAP Setups
    immediately ... Keep quick likes unchanged. Determine the trade horizon
    explicitly; do not rely on a stale chart timeframe.

    Save and confirm the pick before removing its D1 item from Visual Chart
    Review. If saving fails, keep the chart and show the failure."

Every test here drives the REAL gesture: `CaptureRail.commit_like()` (what a
double-click on a setup calls) inside a live `AlertCenterPanel`, so the whole
chain runs - `_record_like` -> `verdicts.record_like` -> the annotation row ->
`captured(EVENT_LIKE_CLAIM, row)` -> `AlertChartReview._on_captured` -> the
route. Nothing is hand-built.

Seams this file requires of the builder (both named by the packet):

* ``AlertCenterPanel(..., claimed_picks_path=...)`` - the store this desk
  writes and the file ``_active_claim_keys`` reads, so a test never touches
  ``C:\\TradingBotData``;
* ``AlertChartReview(..., claim_writer=...)`` kept on the widget, the
  injection point the packet names.

P5 holds throughout: a claim writes its own row and is never combined with a
veto, a pass or a "Not today".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _neuter_cohort_merges(rail) -> None:
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True, "added": 0}


def d1_alert(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="08:25:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) zone1 reject at AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{side.lower()}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) zone1 reject",
        is_d1=True,
        payload={
            "priority_score": 72.5,
            "expected_r": 1.4,
            "setup_family": "avwap_band_bounce",
            "priority_bucket": "favorite_setup",
            "key_level": "AVWAPE 188.40",
        },
    )


def m5_alert(symbol: str, side: str = "LONG"):
    """An ordinary intraday bounce alert - the kind the M5 bar lists."""
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="10:05:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) 5m reclaim of VWAP",
        timeframe="M5",
        tag="",
        raw_text=f"BOUNCE: {symbol} ({side.lower()}) 5m reclaim of VWAP",
        is_d1=False,
    )


def manual_alert(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert
    from ui.models.bounce import MANUAL_CHART_TAG

    return BounceAlert(
        time_text="09:00:00",
        symbol=symbol,
        side=side,
        trigger="",
        timeframe="",
        tag=MANUAL_CHART_TAG,
        raw_text=symbol,
        is_d1=False,
    )


def build_panel(tmp_path, monkeypatch, *, claims_path=None):
    """AAPL on the chart, NVDA waiting behind it, every store in tmp."""
    import pick_feedback
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    pick_feedback.clear_reviewed_today_cache()
    made = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
        claimed_picks_path=(
            claims_path if claims_path is not None else tmp_path / "claimed_picks.jsonl"
        ),
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(made, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(made, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(made.chart_review, "_reviewed_symbols", lambda: set())
    monkeypatch.setattr(
        made.chart_review.capture_rail,
        "_annotations_path",
        tmp_path / "trader_annotations.jsonl",
    )
    _neuter_cohort_merges(made.chart_review.capture_rail)
    return made


@pytest.fixture
def panel(tmp_path, monkeypatch):
    made = build_panel(tmp_path, monkeypatch)
    made.add_alert(d1_alert("AAPL"))
    made.add_alert(d1_alert("NVDA", "SHORT"))
    assert made._current_review_alert.symbol == "AAPL"
    assert [a.symbol for a in made._review_queue] == ["NVDA"]
    yield made
    made.close()
    made.deleteLater()


def claim_rows(tmp_path) -> list[dict]:
    path = tmp_path / "claimed_picks.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def do_claim(rail, why: str = "reclaimed the band") -> None:
    rail.setup_list.setCurrentRow(0)
    rail.like_note_input.setText(why)


# ---------------------------------------------------------------------------
# item 3 - a CLAIMED like on a D1 chart places, confirms, then retires
# ---------------------------------------------------------------------------
def test_a_claimed_d1_like_writes_one_claim_row_and_moves_the_chart_on(
    panel, tmp_path
):
    """The trader's gesture end to end: Alt+K, a setup, and the pick exists."""
    pane = panel.chart_review
    rail = pane.capture_rail
    placed: list = []
    pane.claimPlaced.connect(lambda _alert, row: placed.append(row))

    do_claim(rail)
    assert rail.commit_like() is not None

    rows = claim_rows(tmp_path)
    assert len(rows) == 1, f"one claim row, got {rows}"
    row = rows[0]
    assert row["action"] == "claim"
    assert row["symbol"] == "AAPL"
    assert row["side"] == "LONG"
    assert row["horizon"] == "d1"
    assert row["claimed_setup_id"], "the claim names the setup"
    assert row["claim_at"], "the claim time is kept"
    assert "chart_review" in str(row["source"]), (
        f"the source records the surface and the alert kind, got {row['source']!r}"
    )
    assert row["annotation_ref"], "the like row and the claim row must join"
    assert placed and placed[0]["symbol"] == "AAPL", "claimPlaced carries the row"
    # ...and the chart is done with.
    assert panel._current_review_alert.symbol == "NVDA"
    assert [a.symbol for a in panel._review_queue] == []


def test_the_claim_carries_what_the_desk_already_knew_about_the_name(panel, tmp_path):
    """`known_at_claim` is built from the ALERT'S OWN payload - honest, and
    possibly empty. It is what a claimed-only row shows instead of inventing a
    score."""
    do_claim(panel.chart_review.capture_rail)
    assert panel.chart_review.capture_rail.commit_like() is not None

    known = claim_rows(tmp_path)[0]["known_at_claim"]
    assert known.get("priority_score") == 72.5
    assert known.get("expected_r") == 1.4
    assert known.get("setup_family") == "avwap_band_bounce"
    assert known.get("key_level") == "AVWAPE 188.40"


def test_a_payload_free_alert_claims_with_an_empty_measurement_dict(
    tmp_path, monkeypatch
):
    """Missing measurements are shown honestly, never as zeros."""
    made = build_panel(tmp_path, monkeypatch)
    try:
        bare = d1_alert("ZZZZ")
        bare.payload = {}
        made.add_alert(bare)
        assert made._current_review_alert.symbol == "ZZZZ"
        do_claim(made.chart_review.capture_rail)
        assert made.chart_review.capture_rail.commit_like() is not None

        assert claim_rows(tmp_path)[0]["known_at_claim"] == {}
    finally:
        made.close()
        made.deleteLater()


def test_a_claim_records_like_advance_once_and_parks_nothing(panel, monkeypatch):
    """P5 and R9.2 together: the claim's forward record is the like's own
    review event, and the retirement verb is NOT borrowed."""
    recorded: list[str] = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: recorded.append(action)
    )
    retired: list = []
    monkeypatch.setattr(
        panel, "_retire_review_alert", lambda *a, **k: retired.append(a)
    )

    do_claim(panel.chart_review.capture_rail)
    assert panel.chart_review.capture_rail.commit_like() is not None

    assert recorded == ["like_advance"], (
        f"exactly one review event, under its historical name; got {recorded}"
    )
    assert retired == [], (
        "`_retire_review_alert` is the PARKING verb - the claim route must have "
        "its own `_retire_claimed_review`"
    )
    assert "AAPL" not in panel._parked_symbols
    assert "AAPL" not in panel._ignored_symbols


def test_the_panel_retires_a_claimed_chart_through_its_own_method(panel):
    """`_place_claimed_d1` / `_retire_claimed_review` exist and are separate."""
    assert callable(getattr(panel, "_place_claimed_d1", None))
    assert callable(getattr(panel, "_retire_claimed_review", None))
    assert panel._retire_claimed_review is not panel._retire_review_alert


def test_the_status_line_says_the_pick_was_placed(panel):
    do_claim(panel.chart_review.capture_rail)
    statuses: list[str] = []
    panel.statusChanged.connect(statuses.append)

    assert panel.chart_review.capture_rail.commit_like() is not None

    joined = " | ".join(statuses).lower()
    assert "claimed" in joined and "setups" in joined, statuses


# ---------------------------------------------------------------------------
# item 3 - a failed save keeps the chart and shows the failure
# ---------------------------------------------------------------------------
def test_a_store_that_cannot_be_written_keeps_the_chart_and_names_the_failure(
    tmp_path, monkeypatch
):
    """A REAL failure: the claims path is a directory, so every append fails.

    The like still stands - its annotation row is written first and is never
    conditional on the placement - but nothing is retired and the trader is
    told, rather than losing the chart to a pick that does not exist.
    """
    blocked = tmp_path / "claimed_picks.jsonl"
    blocked.mkdir()
    made = build_panel(tmp_path, monkeypatch, claims_path=blocked)
    try:
        made.add_alert(d1_alert("AAPL"))
        made.add_alert(d1_alert("NVDA", "SHORT"))
        pane = made.chart_review
        rail = pane.capture_rail
        fired: list[str] = []
        pane.claimPlaced.connect(lambda *_a: fired.append("placed"))
        pane.likeRecorded.connect(lambda *_a: fired.append("recorded"))
        pane.likeAdvanceRequested.connect(lambda *_a: fired.append("advance"))

        do_claim(rail)
        assert rail.commit_like() is not None, "the LIKE itself still succeeds"

        annotations = [
            json.loads(line)
            for line in (tmp_path / "trader_annotations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert [row["event_type"] for row in annotations] == ["like_claim"]
        assert fired == ["recorded"], (
            f"a failed placement reports the like and moves nothing; got {fired}"
        )
        assert made._current_review_alert.symbol == "AAPL", "the chart is kept"
        assert [a.symbol for a in made._review_queue] == ["NVDA"], "queue unchanged"
        status = rail.status_text().upper()
        assert "NOT PLACED" in status, rail.status_text()
        assert "CLAIMED_PICKS" in status.replace(" ", ""), rail.status_text()
    finally:
        made.close()
        made.deleteLater()


def test_the_pane_takes_an_injected_claim_writer(tmp_path, monkeypatch):
    """The packet's test seam: the writer is a constructor parameter."""
    import inspect

    import claimed_picks
    from ui.widgets.alert_chart_review import AlertChartReview

    parameters = inspect.signature(AlertChartReview.__init__).parameters
    assert "claim_writer" in parameters, sorted(parameters)
    assert parameters["claim_writer"].default in (None, claimed_picks.record_claim)

    calls: list[dict] = []
    pane = AlertChartReview(claim_writer=lambda **kwargs: calls.append(kwargs) or None)
    try:
        assert pane._claim_writer is not None
    finally:
        pane.deleteLater()


# ---------------------------------------------------------------------------
# item 2 - the horizon, resolved from the alert and never from the rail
# ---------------------------------------------------------------------------
def test_a_stale_rail_timeframe_never_decides_the_horizon(panel, tmp_path):
    """The rail is "D1" from construction and `set_alert` never re-pointed it.

    Forced to the wrong value here, the claim still places as a D1 pick,
    because the horizon comes from the alert.
    """
    rail = panel.chart_review.capture_rail
    rail._timeframe = "M5"

    do_claim(rail)
    assert rail.commit_like() is not None

    rows = claim_rows(tmp_path)
    assert [row["horizon"] for row in rows] == ["d1"]


def test_set_alert_hands_the_rail_the_charts_own_timeframe(tmp_path, monkeypatch):
    """The correctness fix behind the trader's sentence: the M5 sidecar attach
    in `_record_like` reads `_timeframe`, so it has to be the chart's."""
    made = build_panel(tmp_path, monkeypatch)
    try:
        rail = made.chart_review.capture_rail
        made.add_alert(d1_alert("AAPL"))
        assert rail._timeframe == "D1"

        made.chart_alert(m5_alert("TSLA"))

        assert made._current_review_alert.symbol == "TSLA"
        assert rail._timeframe == "M5", (
            "an M5 chart must not leave the rail claiming D1"
        )
    finally:
        made.close()
        made.deleteLater()


def test_a_claimed_like_on_a_genuine_m5_chart_takes_the_old_route_and_places_nothing(
    tmp_path, monkeypatch
):
    """The left side of the desk is for M5 trades: nothing lands in Setups."""
    made = build_panel(tmp_path, monkeypatch)
    try:
        made.add_alert(d1_alert("AAPL"))
        made.chart_alert(m5_alert("TSLA"))
        assert made._current_review_alert.symbol == "TSLA"
        pane = made.chart_review
        fired: list[str] = []
        pane.claimPlaced.connect(lambda *_a: fired.append("placed"))
        pane.likeAdvanceRequested.connect(lambda *_a: fired.append("advance"))

        do_claim(pane.capture_rail)
        assert pane.capture_rail.commit_like() is not None

        assert fired == ["advance"], fired
        assert claim_rows(tmp_path) == [], "an M5 claim writes no claimed pick"
    finally:
        made.close()
        made.deleteLater()


def test_a_manual_look_claimed_as_a_swing_setup_places(tmp_path, monkeypatch):
    made = build_panel(tmp_path, monkeypatch)
    try:
        made.chart_symbol("CRM")
        assert made._current_review_alert is not None
        assert made._current_review_alert.symbol == "CRM"
        rail = made.chart_review.capture_rail
        do_claim(rail)
        assert rail.commit_like() is not None

        rows = claim_rows(tmp_path)
        assert [(row["symbol"], row["horizon"]) for row in rows] == [("CRM", "d1")]
    finally:
        made.close()
        made.deleteLater()


def test_a_claim_the_registry_cannot_name_places_nothing_and_says_so(
    tmp_path, monkeypatch
):
    """`none_of_these` is an honest answer with no horizon behind it.

    Driven through `_record_like`, the rail's ONE like writer (`commit_like`
    calls it): the rail's picklist does not offer `none_of_these`, but the
    annotation store and the whole route below it are the real ones.
    """
    from ui.annotations.store import LIKE_MODE_CLAIMED

    made = build_panel(tmp_path, monkeypatch)
    try:
        made.chart_symbol("CRM")
        pane = made.chart_review
        fired: list[str] = []
        pane.claimPlaced.connect(lambda *_a: fired.append("placed"))
        pane.likeAdvanceRequested.connect(lambda *_a: fired.append("advance"))

        row = pane.capture_rail._record_like(
            claimed_setup_id="none_of_these",
            note="",
            like_mode=LIKE_MODE_CLAIMED,
        )
        assert row is not None

        assert claim_rows(tmp_path) == []
        assert fired == ["advance"], fired
        status = pane.capture_rail.status_text().lower()
        assert "horizon unknown" in status, pane.capture_rail.status_text()
    finally:
        made.close()
        made.deleteLater()


# ---------------------------------------------------------------------------
# the quick like is untouched, and a repeat claim invents nothing
# ---------------------------------------------------------------------------
def test_a_quick_like_on_a_d1_chart_writes_no_claim_and_keeps_the_chart(
    panel, tmp_path
):
    """Alt+L names no setup, so there is nothing to place (T1/P9 unchanged)."""
    pane = panel.chart_review
    fired: list[str] = []
    pane.claimPlaced.connect(lambda *_a: fired.append("placed"))
    pane.likeRecorded.connect(lambda *_a: fired.append("recorded"))

    assert pane.capture_rail.commit_quick_like() is not None

    assert claim_rows(tmp_path) == []
    assert fired == ["recorded"], fired
    assert panel._current_review_alert.symbol == "AAPL"
    assert [a.symbol for a in panel._review_queue] == ["NVDA"]


def test_claiming_the_same_setup_twice_appends_no_second_row_and_still_advances(
    panel, tmp_path
):
    """A duplicate claim is a SUCCESS - the pick exists - and takes the same
    route, so the chart is still done with."""
    pane = panel.chart_review
    rail = pane.capture_rail
    placed: list = []
    pane.claimPlaced.connect(lambda _a, row: placed.append(row))

    do_claim(rail)
    assert rail.commit_like() is not None
    assert panel._current_review_alert.symbol == "NVDA"

    # The trader charts AAPL again and claims the same setup.
    panel.chart_alert(d1_alert("AAPL"))
    assert panel._current_review_alert.symbol == "AAPL"
    do_claim(rail, "same read")
    assert rail.commit_like() is not None

    assert len(claim_rows(tmp_path)) == 1, "the duplicate appends nothing"
    assert len(placed) == 2, "but it still reports a placed pick"
    assert placed[1].get("duplicate") is True
    assert panel._current_review_alert is None or (
        panel._current_review_alert.symbol != "AAPL"
    ), "the duplicate still finishes with the chart"


# ---------------------------------------------------------------------------
# ADDED BY THE BUILDER (2026-09-14) - the invariant the test above is reaching
# for, stated in a form the desk can satisfy.
#
# `test_a_claim_records_like_advance_once_and_parks_nothing` asserts
# `recorded == ["like_advance"]`. The claim route writes exactly one review
# event, but retiring the chart ADVANCES the queue (packet item 3: "remove the
# alert from `_current_review_alert` / `_review_queue` / `_hidden_inside_range`
# and advance"), and putting the next chart up records its IMPRESSION -
# `_render_current_review` writes `shown` for NVDA. That is not a second
# verdict, it is the decision log saying a chart was seen, and the pre-existing
# claimed-like route writes it too (which is why
# `tests/test_t1_capture_and_board_like.py` uses `in` rather than `==` at the
# same seam). The assertion is left as the tester wrote it and reported.
# ---------------------------------------------------------------------------
def test_a_claim_writes_exactly_one_verdict_and_no_rejection_of_any_kind(
    panel, monkeypatch
):
    """P5 and R9.2: one forward record, and none of the rejection verbs."""
    recorded: list[str] = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: recorded.append(action)
    )
    retired: list = []
    monkeypatch.setattr(
        panel, "_retire_review_alert", lambda *a, **k: retired.append(a)
    )

    do_claim(panel.chart_review.capture_rail)
    assert panel.chart_review.capture_rail.commit_like() is not None

    assert recorded.count("like_advance") == 1, recorded
    for rejection in ("remove_today", "skip", "focus_review_remove", "unfavorite"):
        assert rejection not in recorded, f"{rejection} is not a claim; got {recorded}"
    assert retired == [], (
        "`_retire_review_alert` is the PARKING verb - the claim route must have "
        "its own `_retire_claimed_review`"
    )
    assert "AAPL" not in panel._parked_symbols
    assert "AAPL" not in panel._ignored_symbols


def test_the_unchanged_advance_route_records_the_next_charts_impression_too(
    panel, monkeypatch
):
    """Proof that `shown` is the ADVANCE's, not the claim's.

    ADDED BY THE BUILDER. This drives `_advance_after_like` - the route packet
    D1C-A leaves exactly as it found it, reached here by claiming a setup the
    registry cannot name - and it records the same two events. So
    `recorded == ["like_advance"]` in the test above is asking the claim route
    to be quieter than the route it is modelled on, which would mean not
    recording that the NEXT chart was put in front of the trader.
    """
    from ui.annotations.store import LIKE_MODE_CLAIMED

    recorded: list[str] = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: recorded.append(action)
    )

    row = panel.chart_review.capture_rail._record_like(
        claimed_setup_id="none_of_these", note="", like_mode=LIKE_MODE_CLAIMED
    )

    assert row is not None
    assert recorded == ["like_advance", "shown"], recorded
    assert panel._current_review_alert.symbol == "NVDA"
