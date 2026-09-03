"""R4 Part B item B6 - one "lately", counted in sessions, with no literals left.

CLAUDE.md, from decision 0016 answer 6: *"'Lately' is ONE number and it is
counted in trading sessions. `evidence_stats.LATELY_SESSIONS` (20) is the home;
`lately_window()` walks the exchange calendar. Twenty calendar days is fourteen
sessions in a normal month and twelve across a holiday week, so a calendar window
silently shortens the sample exactly when the market was closed."*

V3 item 3 built the constant and converted most callers. Two named paths kept
their own calendar-day literals: `review_learning.DEFAULT_WINDOW_DAYS = 90`,
which is the window the blind-spot and leak callouts are cut on - one of the
surfaces CLAUDE.md explicitly lists as reading `LATELY_SESSIONS` - and Weekend
Prep's two `build_review_learning_state(window_days=7)` calls.

The literal scan is the part that keeps this true. A converted caller can be
un-converted by one line, and the failure is invisible: the board still renders,
with a differently sized sample.

Offline and pure: no store, no Qt widget, no network.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


def test_the_review_board_counts_sessions_and_not_calendar_days():
    import evidence_stats
    import review_learning

    assert not hasattr(review_learning, "DEFAULT_WINDOW_DAYS"), (
        "a calendar-day default is a second definition of the same window"
    )
    assert review_learning.DEFAULT_WINDOW_SESSIONS == evidence_stats.LATELY_SESSIONS


def test_the_cutoff_walks_the_exchange_calendar():
    """A holiday must LENGTHEN the window in calendar days, never shorten it.

    That is the whole reason the desk counts sessions: a fixed 20 calendar days
    silently drops sessions exactly when the market was closed.
    """
    from datetime import date

    import evidence_stats

    start, end = evidence_stats.lately_window(date(2026, 9, 3), sessions=20)
    span = (date.fromisoformat(end) - date.fromisoformat(start)).days
    assert span > 20, f"20 sessions spanned only {span} calendar days"


def test_the_state_reports_the_window_in_the_unit_it_measured():
    """A report that says "days" over a session count is a lie in the header."""
    from review_learning import build_review_learning_state, render_report

    state = build_review_learning_state(
        events_path=ROOT / "tests" / "does-not-exist.jsonl",
        outcomes_path=ROOT / "tests" / "does-not-exist.csv",
        annotations_path=ROOT / "tests" / "does-not-exist.jsonl",
        window_sessions=5,
    )
    assert state["window_sessions"] == 5
    assert "window_days" not in state
    assert "last 5 sessions" in render_report(state)


# ---------------------------------------------------------------------------
# The literal scan
# ---------------------------------------------------------------------------

#: Every module the packet named, plus the two that RENDER the window - a
#: renderer that still says "days" is the same defect one layer out.
SCANNED = (
    "scripts/review_learning.py",
    "scripts/ui/panels/weekend_prep_panel.py",
    "scripts/ui/panels/daytrade_tracker_panel.py",
    "scripts/review_capture_audit.py",
)


def test_no_named_path_still_carries_a_calendar_day_window_literal():
    offenders: dict[str, list[str]] = {}
    for name in SCANNED:
        text = (ROOT / name).read_text(encoding="utf-8")
        hits = [
            line.strip()
            for line in text.splitlines()
            # The identifier, anywhere it is still LIVE code. A comment saying
            # what it used to be is the record and stays.
            if re.search(r"\bwindow_days\b", line)
            and not line.strip().startswith("#")
        ]
        if hits:
            offenders[name] = hits
    assert not offenders, offenders


def test_lately_has_exactly_one_home():
    """No module may define its own 20 under a "lately" name."""
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "scripts").rglob("*.py"):
        if path.name == "evidence_stats.py":
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        hits = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if re.match(r"^\s*[A-Z_]*LATELY[A-Z_]*\s*=", line)
        ]
        if hits:
            offenders[rel] = hits
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# A flat is neither
# ---------------------------------------------------------------------------


def test_a_flat_outcome_is_neither_a_win_nor_a_loss():
    """`close_r == 0.0` was counted as a loss, which understates every rate.

    A scratch is not a loss. Counting it as one moves the headline the packet
    made the headline: three flats in ten graded rows dropped a 5-2 record from
    71% to 50%.
    """
    from swing_headline import headline_from_outcomes

    rows = [{"close_r": r} for r in (1.0, 2.0, -1.0, 0.0, 0.0)]
    headline = headline_from_outcomes("FAM", rows)

    assert headline.wins == 2
    assert headline.losses == 1
    assert headline.flats == 2
    assert headline.n == 3, "a flat must not sit in the win-rate denominator"
    assert headline.win_rate == 2 / 3


def test_a_flat_still_counts_in_the_mean():
    """It is a MEASURED outcome; only the win/loss question has no answer."""
    from swing_headline import headline_from_outcomes

    headline = headline_from_outcomes("FAM", [{"close_r": 3.0}, {"close_r": 0.0}])
    assert headline.avg_r == 1.5
    assert headline.n == 1


def test_an_unreadable_row_is_still_counted_nowhere_at_all():
    """Unmeasured and flat are different facts and must not merge."""
    from swing_headline import headline_from_outcomes

    headline = headline_from_outcomes(
        "FAM", [{"close_r": None}, {"close_r": "n/a"}, {}, {"close_r": 0.0}]
    )
    assert (headline.wins, headline.losses, headline.flats) == (0, 0, 1)
    assert headline.avg_r == 0.0


def test_the_row_a_surface_renders_says_how_many_were_flat():
    from swing_headline import headline_from_outcomes

    row = headline_from_outcomes("FAM", [{"close_r": 1.0}, {"close_r": 0.0}]).as_row()
    assert row["flats"] == 1
    assert row["n"] == 1
