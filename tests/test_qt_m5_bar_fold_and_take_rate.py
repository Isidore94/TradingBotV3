"""P2 items 5-6: the M5 alert bar shows the take rate and folds repeats.

Two display changes over information the desk already had.

* The Alert Center measures P(take | shown) per segment and used it only for
  queue ordering and the review pane. The bar - the surface the trader actually
  reads during the session - showed none of it.
* The bar drew one line per alert. The main feed has folded repeats since
  2026-08-16 ("less spam and more quality"); the bar, which is narrower and
  read faster, did not.

The invariant both changes had to keep: **the bar deletes nothing, mutes
nothing, records nothing and withholds nothing.** Every event reaches the review
queue door, the outcome CSV and the review-event store BEFORE the bar sees it.
The fold draws N events on one line; it does not drop them, and
`test_the_fold_is_presentation_only` is the test that says so.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.models.bounce import BounceAlert  # noqa: E402
from ui.widgets import m5_alert_bar  # noqa: E402
from ui.widgets.m5_alert_bar import M5AlertBar  # noqa: E402


def row_text(*args, **kwargs):
    return m5_alert_bar.row_text(*args, **kwargs)


def take_probability(*args, **kwargs):
    """Resolved at CALL time so the un-fixed module still imports and each
    test fails on its own merit rather than at collection."""
    return m5_alert_bar.take_probability(*args, **kwargs)


@pytest.fixture(scope="module", autouse=True)
def _app():
    return QApplication.instance() or QApplication([])


def _alert(symbol="AAA", side="LONG", trigger="new HOD", at="07:09:00", **extra):
    return BounceAlert(
        time_text=datetime.strptime(at, "%H:%M:%S").strftime("%H:%M:%S"),
        symbol=symbol,
        side=side,
        trigger=trigger,
        timeframe="M5",
        raw_text=f"[B-TIER] {symbol}: {trigger}",
        **extra,
    )


# ==========================================================================
# item 5 - the take-rate suffix
# ==========================================================================
def test_the_row_carries_the_take_rate_when_the_desk_measured_one():
    """Fail-before-fix: `take_probability` does not exist and `row_text` has no
    suffix."""
    assert row_text(_alert(review_take_prob=0.283)) == "07:09  ▲ AAA  new HOD  take 28%"


def test_no_guidance_is_SILENCE_not_a_zero():
    """A missing suffix says nothing. A 0% would be a claim about a segment
    nobody has measured."""
    assert row_text(_alert()) == "07:09  ▲ AAA  new HOD"
    assert take_probability(_alert()) is None


def test_a_nonsense_probability_is_refused_rather_than_rendered():
    for value in ("not a number", -0.2, 1.5, None, float("nan")):
        assert take_probability(_alert(review_take_prob=value)) is None
    assert row_text(_alert(review_take_prob=2.0)) == "07:09  ▲ AAA  new HOD"


def test_the_bar_never_computes_the_take_rate_itself():
    """READ ONLY. The host attaches the value the Alert Center already had;
    the bar must not look one up, because it draws on the Qt thread."""
    import inspect

    source = inspect.getsource(m5_alert_bar)
    for forbidden in ("guidance_for", "ReviewGuide", "review_learning", "load_review"):
        assert forbidden not in source, f"the bar must not reach for {forbidden}"


def test_the_alert_center_uses_the_CACHE_and_never_recomputes(monkeypatch):
    """`_guidance_for` on a miss stats two files and can re-read a 34 KB JSON,
    per alert, on the Qt thread. The suffix is not worth that."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel.__new__(AlertCenterPanel)
    panel._review_guidance = {}

    def must_not_run(self, alert):  # pragma: no cover - the point is it is unused
        raise AssertionError("the emit path must not compute guidance")

    monkeypatch.setattr(AlertCenterPanel, "_guidance_for", must_not_run)

    alert = _alert()
    AlertCenterPanel._attach_cached_take_prob(panel, alert)
    assert alert.review_take_prob is None

    from review_guidance import AlertGuidance

    panel._review_guidance["AAA"] = AlertGuidance(take_prob=0.283)
    AlertCenterPanel._attach_cached_take_prob(panel, alert)
    assert alert.review_take_prob == pytest.approx(0.283)


# ==========================================================================
# item 6 - the repetition fold
# ==========================================================================
def test_a_repeat_folds_and_badges_instead_of_inserting_a_line():
    bar = M5AlertBar()
    bar.post(_alert(at="07:00:00"))
    bar.post(_alert(at="07:05:00", trigger="VWAP reclaim"))
    bar.post(_alert(at="07:10:00", trigger="new HOD"))

    assert bar.count() == 1
    assert bar.list.item(0).text() == "07:10  ▲ AAA  new HOD  ×3"
    # The row carries the NEWEST alert, so clicking it charts what just fired.
    assert bar.alerts()[0].time_text.startswith("07:10")


def test_a_repeat_returns_the_row_to_the_top():
    bar = M5AlertBar()
    bar.post(_alert("AAA", at="07:00:00"))
    bar.post(_alert("BBB", at="07:05:00"))
    assert [a.symbol for a in bar.alerts()] == ["BBB", "AAA"]

    bar.post(_alert("AAA", at="07:10:00"))
    assert [a.symbol for a in bar.alerts()] == ["AAA", "BBB"]
    assert bar.count() == 2


def test_the_other_side_of_the_same_name_is_a_different_row():
    """A name that flips direction is a different claim, and folding the two
    would hide the flip - the one thing on this bar most worth seeing."""
    bar = M5AlertBar()
    bar.post(_alert("AAA", side="LONG"))
    bar.post(_alert("AAA", side="SHORT"))

    assert bar.count() == 2
    assert {a.side for a in bar.alerts()} == {"LONG", "SHORT"}


def test_the_fold_is_presentation_only():
    """The bar's contract, unchanged: it draws N events on one line and drops
    none of them. The folded row's tooltip says so in words, and every event
    still reached the evidence path before the bar was called at all."""
    bar = M5AlertBar()
    bar.post(_alert(at="07:00:00"))
    bar.post(_alert(at="07:05:00"))

    tooltip = bar.list.item(0).toolTip()
    assert "2 alerts on this name" in tooltip
    assert "it does not drop them" in tooltip
    # The newest alert's own text is still there to read.
    assert "AAA: new HOD" in tooltip


def test_a_single_alert_carries_no_badge_and_its_own_tooltip():
    bar = M5AlertBar()
    bar.post(_alert())
    assert "×" not in bar.list.item(0).text()
    assert bar.list.item(0).toolTip() == "[B-TIER] AAA: new HOD"


def test_copy_all_still_lists_one_symbol_per_row():
    bar = M5AlertBar()
    bar.post(_alert("AAA", at="07:00:00"))
    bar.post(_alert("BBB", at="07:05:00"))
    bar.post(_alert("AAA", at="07:10:00"))

    assert bar.copy_all() == "AAA\nBBB"


def test_clicking_a_folded_row_charts_the_newest_alert_and_clears_the_line():
    bar = M5AlertBar()
    charted = []
    bar.alertActivated.connect(charted.append)
    bar.post(_alert(at="07:00:00", trigger="VWAP reclaim"))
    bar.post(_alert(at="07:10:00", trigger="new HOD"))

    bar._on_item_clicked(bar.list.item(0))

    assert bar.count() == 0
    assert len(charted) == 1
    assert charted[0].trigger == "new HOD"


def test_a_tier_upgrade_rewrites_the_row_with_the_stronger_alert():
    """Escalation: what changed is exactly what the trader wanted to see."""
    bar = M5AlertBar()
    bar.post(_alert(at="07:00:00", trigger="[C-TIER] weak thing"))
    bar.post(_alert(at="07:10:00", trigger="[S-TIER] PROVEN thing"))

    assert bar.count() == 1
    assert "PROVEN thing" in bar.list.item(0).text()
    assert bar.alerts()[0].raw_text.endswith("[S-TIER] PROVEN thing")


def test_the_fold_survives_the_row_cap():
    from ui.widgets.m5_alert_bar import MAX_ROWS

    bar = M5AlertBar()
    for index in range(MAX_ROWS + 5):
        bar.post(_alert(f"S{index}", at="07:00:00"))
    assert bar.count() == MAX_ROWS

    # The newest name is present and still folds rather than growing the list.
    bar.post(_alert(f"S{MAX_ROWS + 4}", at="07:30:00"))
    assert bar.count() == MAX_ROWS


def test_a_folded_row_shows_the_take_rate_of_the_newest_alert():
    bar = M5AlertBar()
    bar.post(_alert(at="07:00:00"))
    bar.post(_alert(at="07:10:00", review_take_prob=0.5))

    assert bar.list.item(0).text() == "07:10  ▲ AAA  new HOD  ×2  take 50%"
