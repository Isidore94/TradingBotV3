"""V2 item 2 - one Refresh, and a verdict card that says what the week was.

Decision 0016 answer 10, in the trader's words: Weekend Prep's *"first screen is a
wall of text whose three CALLOUT lines are the only part that matters, its tables
show three rows at a time, and each table has its own refresh. Wanted: one Refresh
for the whole tab, a short verdict card on top, tables not prose, ten visible
rows."*

The card is a PURE builder (`scripts/weekend_verdict.py`) so it is testable
without a journal, a lake or an event loop; the panel reads the stores on its
worker and hands the rows over.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def test_the_card_is_five_to_eight_lines():
    """The trader asked for a card, not a second wall of text."""
    import weekend_verdict

    lines = weekend_verdict.build_verdict().rendered()
    assert 5 <= len(lines) <= 8, lines


def test_every_measured_line_carries_its_n():
    """A verdict with no sample size is an opinion."""
    import weekend_verdict

    verdict = weekend_verdict.build_verdict(
        # R4 A13: the REAL key shape. `build_review_learning_state` publishes
        # `shown` / `takes` / `overall_take_rate` and has never published
        # `skips` or `rejects` - the hand-written dict this used to carry is
        # what let the card's denominator be wrong.
        learning_state={"shown": 17, "takes": 4, "overall_take_rate": 0.235},
        like_rows=[{"source": "like_avwap_breakout", "avg_r_h3": 0.42, "n_h3": 9}],
        veto_rows=[{"source": "veto_v2_compressed", "avg_r_h3": 0.31, "n_h3": 7}],
        week_trades=[
            {"tag_status": "confirmed", "setup_tags": "x", "net_pnl": 120.0},
            {"tag_status": "confirmed", "setup_tags": "x", "net_pnl": -40.0},
        ],
        awaiting_review=12,
    )

    for line in verdict.lines:
        if line.measured:
            assert line.n is not None, line.key
            assert f"(n={line.n})" in line.rendered()


def test_a_missing_input_says_so_rather_than_printing_a_zero():
    """"No graded likes yet" and "your likes averaged 0.00R" are different facts."""
    import weekend_verdict

    verdict = weekend_verdict.build_verdict()
    text = "\n".join(verdict.rendered())

    assert "nothing was shown for review" in text
    assert "nothing with enough behind it yet" in text
    assert "none with a confirmed tag yet" in text
    assert "0.00R" not in text
    assert "0%" not in text


def test_the_callouts_are_named_not_counted():
    """"Blind Spots: 3" is a number a reader cannot act on."""
    import weekend_verdict

    verdict = weekend_verdict.build_verdict(
        learning_state={
            "shown": 2,
            "takes": 1,
            "overall_take_rate": 0.5,
            "blind_spots": [{"segment": "ema_15 morning"}, {"segment": "vwap_reclaim"}],
            "leaks": [{"segment": "gap_fill power_hour"}],
        }
    )
    text = "\n".join(verdict.rendered())

    assert "ema_15 morning" in text
    assert "vwap_reclaim" in text
    assert "gap_fill power_hour" in text


def test_a_thin_cohort_is_never_the_headline():
    """A top row resting on two observations is worse than no row."""
    import weekend_verdict

    verdict = weekend_verdict.build_verdict(
        like_rows=[
            {"source": "like_lucky", "avg_r_h3": 9.9, "n_h3": 2},
            {"source": "like_real", "avg_r_h3": 0.31, "n_h3": 40},
        ]
    )
    line = next(item for item in verdict.lines if item.key == "best_like")

    assert "like_real" in line.text
    assert "like_lucky" not in line.text


def test_only_a_confirmed_tag_counts_toward_the_weeks_record():
    """"My setups" counts confirmed tags; a provisional guess is the machine's."""
    import weekend_verdict

    line = weekend_verdict.journal_week_line(
        [
            {"tag_status": "confirmed", "setup_tags": "mine", "net_pnl": 100.0},
            {"tag_status": "provisional", "setup_tags": "guessed", "net_pnl": -500.0},
            {"tag_status": "needs_review", "setup_tags": "", "net_pnl": -500.0},
        ]
    )

    assert line.n == 1
    assert "+100.00" in line.text


def test_the_card_reaches_nothing_live():
    source = (ROOT / "scripts" / "weekend_verdict.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]
    for forbidden in ("review_policy", "focus", "alert", "watchlist"):
        assert forbidden not in body.lower().replace("# ", ""), forbidden


# ---------------------------------------------------------------------------
# One Refresh
# ---------------------------------------------------------------------------


def test_one_click_starts_every_step_and_returns_immediately(qapp, monkeypatch):
    """The reads behind this button were once 8.45 s of frozen GUI on one page."""
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    started: list[str] = []
    for step, page in panel._pages.items():
        monkeypatch.setattr(
            page, "reload", lambda step=step: started.append(step), raising=False
        )
    monkeypatch.setattr(panel, "_start_verdict", lambda: None)

    began = time.perf_counter()
    panel.refresh_everything()
    elapsed_ms = (time.perf_counter() - began) * 1000.0

    assert sorted(started) == sorted(panel._pages), started
    assert elapsed_ms < 50.0, f"the click itself took {elapsed_ms:.1f} ms"
    assert "Building:" in panel.building_note.text()


def test_the_per_page_refresh_buttons_are_out_of_the_layout(qapp):
    """Five buttons for one routine is what the trader complained about."""
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    for page in panel._pages.values():
        button = getattr(page, "refresh_button", None)
        if button is None:
            continue
        # Still an object - `reload()` uses it as its own single-flight guard -
        # but no longer parented into the page's layout.
        assert page.layout().indexOf(button) == -1, page.step_id


def test_one_page_that_will_not_start_does_not_stop_the_others(qapp, monkeypatch):
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    started: list[str] = []
    for index, (step, page) in enumerate(panel._pages.items()):
        if index == 0:
            monkeypatch.setattr(
                page,
                "reload",
                lambda: (_ for _ in ()).throw(RuntimeError("store gone")),
                raising=False,
            )
            continue
        monkeypatch.setattr(
            page, "reload", lambda step=step: started.append(step), raising=False
        )
    monkeypatch.setattr(panel, "_start_verdict", lambda: None)

    panel.refresh_everything()

    assert len(started) == len(panel._pages) - 1


def test_the_verdict_is_read_off_the_qt_thread(qapp):
    """It reads four stores and a journal."""
    source = (ROOT / "scripts" / "ui" / "panels" / "weekend_prep_panel.py").read_text(
        encoding="utf-8"
    )
    assert "_ReadWorker(self._read_verdict, self)" in source
    build = source.split("def _read_verdict", 1)[1].split("def _on_verdict_ready", 1)[0]
    assert "setText" not in build, "the worker must touch no widget"


def test_one_unreadable_store_still_leaves_a_card(qapp, monkeypatch):
    """The screen the trader opens to find out how the week went."""
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    monkeypatch.setattr(
        panel_module,
        "_read_like_cohort",
        lambda: (_ for _ in ()).throw(OSError("csv gone")),
    )
    monkeypatch.setattr(panel_module, "_read_veto_cohort", lambda: [])
    monkeypatch.setattr(panel_module, "_read_week_trades", lambda bounds: [])
    monkeypatch.setattr(panel_module, "_read_awaiting_review", lambda: 0)
    monkeypatch.setattr(
        "review_learning.build_review_learning_state", lambda **_kwargs: {}
    )

    lines = panel._read_verdict()

    assert 5 <= len(lines) <= 8
    assert any("nothing with enough behind it yet" in line for line in lines)
