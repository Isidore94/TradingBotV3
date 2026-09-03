"""R4 A13 - the verdict card's take rate reads the state instead of inventing it.

`take_rate_line` computed `shown = takes + skips + rejects`, and
`build_review_learning_state` publishes NO `skips` key and NO `rejects` key -
`aggregate_dimensions` returns `episodes`, `shown`, `takes`, `overall_take_rate`
and `dimensions`, and that is the whole top level. So the denominator was
`takes + 0 + 0`, the card printed "100% of 94 shown" on a week whose real answer
was 30% of 318, and the first number the trader reads said the opposite of the
truth.

**Fixtured on a REAL state.** Every test here builds one through
`build_review_learning_state` from event rows. A hand-written dict is precisely
what let this ship: it can carry any key the code happens to ask for.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import review_learning  # noqa: E402
import weekend_verdict  # noqa: E402


def _row(action, symbol, *, trade_date="2026-08-28"):
    return {
        "schema": "review_events_v1",
        "ts": f"{trade_date}T10:15:00",
        "trade_date": trade_date,
        "action": action,
        "symbol": symbol,
        "side": "LONG",
        "tier": "A",
        "bounce_types": "dynamic_vwap_upper_band",
        "market_environment": "BULLISH_WEAK",
        "event_id": f"evt-{symbol}-{trade_date}",
    }


def _state(tmp_path, *, takes: int, skips: int):
    """A REAL state: `takes` charts taken and `skips` charts shown and skipped."""
    events = tmp_path / "events.jsonl"
    rows = []
    for index in range(takes):
        rows.append(_row("shown", f"T{index}"))
        rows.append(_row("add_focus", f"T{index}"))
        rows[-1]["detail"] = {"category": "m5", "added": True}
    for index in range(skips):
        rows.append(_row("shown", f"S{index}"))
        rows.append(_row("skip", f"S{index}"))
    with events.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return review_learning.build_review_learning_state(
        events_path=events,
        outcomes_path=tmp_path / "missing_outcomes.csv",
        annotations_path=tmp_path / "missing_annotations.jsonl",
        window_days=90,
        now=datetime(2026, 8, 29, 18, 0),
    )


def test_the_state_has_no_skips_or_rejects_key_to_read():
    """The premise, asserted against the builder's own contract.

    Not against a fixture: this is a statement about the SHAPE
    `aggregate_dimensions` publishes, and a fixture can carry any key the code
    happens to ask for - which is exactly how the defect survived its tests.
    """
    published = review_learning.aggregate_dimensions([])

    assert set(published) == {
        "episodes",
        "shown",
        "takes",
        "overall_take_rate",
        "dimensions",
    }
    assert "skips" not in published and "rejects" not in published


def test_the_denominator_is_every_chart_shown_and_not_just_the_takes(tmp_path):
    """3 of 10, not 100% of 3."""
    state = _state(tmp_path, takes=3, skips=7)

    assert state["shown"] == 10
    assert state["takes"] == 3

    line = weekend_verdict.take_rate_line(state)

    assert line.n == 10
    assert "30% of 10 shown (3 taken)" in line.text
    assert "100%" not in line.text


def test_the_rate_is_the_states_own_and_not_a_second_division(tmp_path):
    state = _state(tmp_path, takes=1, skips=2)

    line = weekend_verdict.take_rate_line(state)

    assert state["overall_take_rate"] == 0.333
    assert "33% of 3 shown" in line.text


def test_a_week_with_nothing_shown_says_so_rather_than_printing_a_zero(tmp_path):
    state = _state(tmp_path, takes=0, skips=0)

    line = weekend_verdict.take_rate_line(state)

    assert line.measured is False
    assert "nothing was shown for review" in line.text
    assert "0%" not in line.text


def test_the_card_carries_the_same_number_the_line_does(tmp_path):
    """The card is what the trader actually reads."""
    state = _state(tmp_path, takes=3, skips=7)

    text = "\n".join(weekend_verdict.build_verdict(learning_state=state).rendered())

    assert "30% of 10 shown (3 taken)" in text
