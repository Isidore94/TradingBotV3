"""Packet D1C-A item 4 - a claimed D1 setup stops coming back for review.

Trader, 2026-09-14:

    "I should not have to keep reviewing the same D1 chart. ... Once claimed,
    keep that same D1 setup out of repeat review while the claim remains
    active. Its row remains available in Master AVWAP Setups. Preserve M5 entry
    review for that symbol. This is a D1 review-queue change, not symbol-wide
    alert suppression. Keep detection, alert evidence and outcome recording
    intact."

The gate lives at the ONE door into the queue, `_enqueue_review_alert`, after
the parked check and BEFORE `_is_m5_review_alert` is consulted - so every M5
alert still reaches the M5 bar exactly as today. Everything upstream (the feed,
History, the D1 badge, the evidence streams, the AWAY recap, the phone push) is
written before that call and is untouched: this is a DISPLAY decision that
withholds nothing, which is the repetition-control precedent.

`review_policy.json` gets no suppression field, ever (plan.md sec 5).
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
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

from test_d1c_claimed_picks_route import (  # noqa: E402
    build_panel,
    claim_rows,
    d1_alert,
    do_claim,
    m5_alert,
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _chart_watch_alert(symbol: str, side: str = "LONG"):
    from ui.models.bounce import CHART_WATCH_TAG, BounceAlert

    return BounceAlert(
        time_text="11:15:00",
        symbol=symbol,
        side=side,
        trigger="New HOD",
        timeframe="D1",
        tag=CHART_WATCH_TAG,
        raw_text=f"MASTER_AVWAP_D1_WATCH: {symbol} new high of day",
        is_d1=True,
    )


@pytest.fixture
def panel(tmp_path, monkeypatch):
    made = build_panel(tmp_path, monkeypatch)
    yield made
    made.close()
    made.deleteLater()


def _claim_aapl(panel, tmp_path) -> None:
    """The real gesture, so the gate is proven against a real claim row."""
    panel.add_alert(d1_alert("AAPL"))
    assert panel._current_review_alert.symbol == "AAPL"
    rail = panel.chart_review.capture_rail
    do_claim(rail)
    assert rail.commit_like() is not None
    assert len(claim_rows(tmp_path)) == 1
    assert panel._current_review_alert is None
    assert panel._review_queue == []


def test_a_claimed_d1_setup_does_not_come_back_to_the_review_queue(panel, tmp_path):
    _claim_aapl(panel, tmp_path)

    panel.add_alert(d1_alert("AAPL"))

    assert panel._current_review_alert is None, (
        "the same D1 flag firing again must not put the chart back up"
    )
    assert [a.symbol for a in panel._review_queue] == []
    assert panel._claimed_d1_skipped.get("AAPL") == 1, (
        "the skip is COUNTED, the way the movers-only hidden count is"
    )


def test_the_gate_keys_on_symbol_and_side_not_on_the_symbol_alone(panel, tmp_path):
    """A claimed LONG says nothing about a SHORT thesis on the same name."""
    _claim_aapl(panel, tmp_path)

    panel.add_alert(d1_alert("AAPL", "SHORT"))

    assert panel._current_review_alert is not None, (
        "the SHORT side was never claimed and still deserves its chart"
    )
    assert panel._current_review_alert.symbol == "AAPL"
    assert panel._current_review_alert.side == "SHORT"


def test_an_armed_chart_watch_on_a_claimed_symbol_is_still_queued(panel, tmp_path):
    """The trader armed this exact condition and is waiting on it."""
    _claim_aapl(panel, tmp_path)

    panel.add_alert(_chart_watch_alert("AAPL"))

    assert panel._current_review_alert is not None
    assert panel._current_review_alert.tag == "chart_watch"


def test_an_m5_alert_for_a_claimed_symbol_still_reaches_the_m5_bar(panel, tmp_path):
    """"Preserve M5 entry review for that symbol." The gate sits BEFORE the M5
    routing, so the left side of the desk is untouched."""
    _claim_aapl(panel, tmp_path)
    posted: list = []
    panel.m5AlertPosted.connect(posted.append)

    panel.add_alert(m5_alert("AAPL"))

    assert [a.symbol for a in posted] == ["AAPL"], (
        "an intraday alert for a claimed name must still list in the M5 bar"
    )
    assert panel._claimed_d1_skipped.get("AAPL", 0) == 0, (
        "an M5 alert is not a repeat D1 review and must not be counted as one"
    )


def test_a_claim_parks_nothing_and_the_other_symbols_are_unaffected(panel, tmp_path):
    _claim_aapl(panel, tmp_path)

    panel.add_alert(d1_alert("NVDA", "SHORT"))

    assert panel._current_review_alert.symbol == "NVDA"
    assert "AAPL" not in panel._parked_symbols
    assert "AAPL" not in panel._ignored_symbols


def test_dropping_the_claim_lets_the_next_d1_alert_queue_again(panel, tmp_path):
    import claimed_picks

    _claim_aapl(panel, tmp_path)
    panel.add_alert(d1_alert("AAPL"))
    assert panel._current_review_alert is None

    claimed_picks.record_drop(
        "AAPL",
        "LONG",
        claim_rows(tmp_path)[0]["claimed_setup_id"],
        path=tmp_path / "claimed_picks.jsonl",
    )
    panel.claimsChanged.emit()
    panel.add_alert(d1_alert("AAPL"))

    assert panel._current_review_alert is not None, (
        "only a drop or an expiry ends a claim - and a drop must take effect"
    )
    assert panel._current_review_alert.symbol == "AAPL"


def test_an_expired_claim_does_not_gate_the_queue(tmp_path, monkeypatch):
    """A claim more than `focus_picks.FADE_TRADING_DAYS` sessions old is over.

    The session date is computed on the exchange calendar, not by subtracting
    eleven calendar days.
    """
    import claimed_picks
    import market_calendar

    today = date.today()
    day = today
    stale = None
    for _ in range(90):
        day = day - timedelta(days=1)
        try:
            if market_calendar.trading_days_between(day, today) == 11:
                stale = day
                break
        except market_calendar.SessionCalendarError:
            pytest.skip("today is outside the validated NYSE calendar range")
    assert stale is not None

    path = tmp_path / "claimed_picks.jsonl"
    claimed_picks.append_row(
        claimed_picks.build_claim_row(
            symbol="AAPL",
            side="LONG",
            horizon="d1",
            claimed_setup_id="avwap_band_bounce",
            source="chart_review:d1_flag_long",
            annotation_ref="",
            known_at_claim={},
            note="",
            session_date=stale.isoformat(),
        ),
        path=path,
    )

    made = build_panel(tmp_path, monkeypatch)
    try:
        made.add_alert(d1_alert("AAPL"))

        assert made._current_review_alert is not None, (
            "a faded claim stops gating - the chart comes back"
        )
        assert made._current_review_alert.symbol == "AAPL"
    finally:
        made.close()
        made.deleteLater()


def test_the_claim_keys_are_reread_only_when_the_claims_file_changes(panel, tmp_path):
    """One small file read when the file changed, never one per alert.

    Nothing expensive belongs on the Qt thread, and an alert burst is exactly
    where a per-alert read would be paid for. `os.utime` moves the stamp two
    seconds so a coarse filesystem clock cannot make this flaky.
    """
    import os as _os

    import claimed_picks

    _claim_aapl(panel, tmp_path)
    path = tmp_path / "claimed_picks.jsonl"

    assert panel._active_claim_keys() == {("AAPL", "LONG")}
    for _ in range(25):
        panel.add_alert(d1_alert("AAPL"))
    assert panel._claimed_d1_skipped.get("AAPL") == 25

    claimed_picks.record_drop(
        "AAPL", "LONG", claim_rows(tmp_path)[0]["claimed_setup_id"], path=path
    )
    stamp = path.stat().st_mtime + 2
    _os.utime(path, (stamp, stamp))

    assert panel._active_claim_keys() == set(), (
        "the cache is keyed on the file, so a change to the file is seen"
    )


def test_the_claim_row_carries_no_suppression_field_of_any_kind(panel, tmp_path):
    """`review_policy.json` ranks and annotates only (plan.md sec 5), and this
    store is not a back door into one."""
    _claim_aapl(panel, tmp_path)
    panel.add_alert(d1_alert("AAPL"))

    row = claim_rows(tmp_path)[0]
    assert set(row) == {
        "schema",
        "action",
        "symbol",
        "side",
        "horizon",
        "claimed_setup_id",
        "claim_at",
        "claim_at_utc",
        "session_date",
        "source",
        "annotation_ref",
        "known_at_claim",
        "note",
    }, sorted(row)
    assert not (tmp_path / "review_policy.json").exists()


# ---------------------------------------------------------------------------
# ADDED BY THE BUILDER (2026-09-14, lead ruling 4): the packet says the skip
# count "shows in the review pane's status the way the movers-only hidden count
# does", and left the display to the builder.
# ---------------------------------------------------------------------------
def test_the_skip_count_is_stated_on_the_review_pane(panel, tmp_path):
    """A queue that goes quiet always says why. Nothing is withheld here - the
    thesis was answered and the pick is in the setups table - so it is a muted
    LABEL beside the movers-only button rather than a second reveal action."""
    pane = panel.chart_review
    assert not pane.claimed_skipped_label.isVisible() or not pane.claimed_skipped_label.text()

    _claim_aapl(panel, tmp_path)
    panel.add_alert(d1_alert("AAPL"))
    panel.add_alert(d1_alert("AAPL"))

    assert panel._claimed_d1_skipped.get("AAPL") == 2
    text = pane.claimed_skipped_label.text()
    assert "2" in text and "claimed" in text.lower(), text
    assert "hidden" not in text.lower(), (
        "a claimed chart is ANSWERED, not withheld - the movers-only line is "
        "the one that says hidden"
    )
    # ...and it is not the reveal button: there is nothing to reveal.
    assert pane.claimed_skipped_label is not pane.hidden_button
