"""Three verified AWAY Recap defects, each pinned by the test that failed first.

Found in the 2026-08-25 GUI review and confirmed at source level before any
line was changed. Two of them share a shape worth naming: a `try/except` that
exists to keep a recap from crashing had been quietly absorbing a **programming
error** - a bad call signature - and reporting it to the trader as though the
data were missing. Fail-quiet is right for a store that might not be there. It
is not right for a call that can never work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


def _recap_panel():
    from ui.panels.away_recap_panel import AwayRecapPanel

    QApplication.instance() or QApplication([])
    return AwayRecapPanel()


# ==========================================================================
# (a) the Focus lists were NEVER read - the call could not have worked
# ==========================================================================
def test_the_recap_actually_reads_the_focus_lists():
    """`load_focus_map(side)` against a keyword-only signature.

    `focus_picks.load_focus_map` takes no positional argument at all: it is
    `def load_focus_map(*, focus_longs_path=None, focus_shorts_path=None)`.
    The recap called it as `load_focus_map(side)`, which raises TypeError on
    every run, and the surrounding fail-quiet `except` turned that into
    `unavailable["Focus lists"]`. So the page reported the trader's Focus
    lists as unreadable every single time it opened, and no amount of the
    files being present could have changed that.

    The union map is the right accessor here - the recap asks "what did the
    day's Focus hold", which is the swing and m5 lists together, exactly what
    the no-argument form returns.
    """
    import focus_picks

    from ui.panels.away_recap_panel import _RecapWorker

    calls: list[tuple] = []

    def recording_load_focus_map(*args, **kwargs):
        calls.append((args, kwargs))
        return {"long": {"AAA", "BBB"}, "short": {"CCC"}}

    original = focus_picks.load_focus_map
    focus_picks.load_focus_map = recording_load_focus_map
    try:
        worker = _RecapWorker("2026-08-25", [])
        seen: list[dict] = []
        worker.loaded.connect(seen.append)
        worker.run()
    finally:
        focus_picks.load_focus_map = original

    assert seen, "the worker emitted nothing"
    unavailable = seen[0].get("unavailable") or {}
    assert "Focus lists" not in unavailable, (
        "the Focus lists are reported unavailable: " f"{unavailable.get('Focus lists')!r}"
    )

    # No positional argument may be passed - that is what could never work.
    assert calls, "load_focus_map was never called"
    assert all(args == () for args, _kwargs in calls), calls
    # And it is read once for the page, not once per side.
    assert len(calls) == 1, f"the store was read {len(calls)} times, expected 1"

    # The names must actually reach the recap.
    rows = seen[0].get("focus_to_manage") or []
    assert {(row["symbol"], row["side"]) for row in rows} == {
        ("AAA", "long"),
        ("BBB", "long"),
        ("CCC", "short"),
    }, rows


def test_the_focus_read_still_fails_quiet_when_the_store_is_genuinely_broken():
    """The repair must not trade a wrong call for a crashing page.

    A real store failure - unreadable file, bad permissions - still belongs in
    `unavailable`, because that is a fact about the day rather than a bug.
    """
    import focus_picks

    from ui.panels.away_recap_panel import _RecapWorker

    def boom(*args, **kwargs):
        raise OSError("the focus files are unreadable")

    original = focus_picks.load_focus_map
    focus_picks.load_focus_map = boom
    try:
        worker = _RecapWorker("2026-08-25", [])
        seen: list[dict] = []
        worker.loaded.connect(seen.append)
        worker.run()
    finally:
        focus_picks.load_focus_map = original

    assert seen, "the worker emitted nothing"
    assert "the focus files are unreadable" in (seen[0].get("unavailable") or {}).get(
        "Focus lists", ""
    )


# ==========================================================================
# (b) the adoption-gate line was decorative - it measured nothing
# ==========================================================================
def test_the_gate_line_never_dresses_an_unmeasured_gate_as_a_verdict():
    """`mover_state(side, None, None, None)` cannot return anything but UNKNOWN.

    The signature is `(side, price, prev_high, prev_low)`. With no price and no
    previous-day extremes, `prev_day_break_state` has nothing to compare, so
    the call returned UNKNOWN on every symbol, every time - and the page
    rendered it as "R2 adoption gate for AAA: UNKNOWN (...)", which reads like
    a measurement that came back inconclusive rather than a measurement that
    was never taken.

    UNKNOWN stays UNKNOWN - this is not a licence to invent a pass. What
    changes is honesty about *why*: the recap has no bar source on this page,
    so it must say the gate was not measured here and point at the surfaces
    that do measure it.
    """
    panel = _recap_panel()
    text = panel._gate_text("AAA", "long")

    lowered = text.lower()
    assert "not measured" in lowered, text
    # It must not present itself as a gate verdict for this symbol.
    assert "adoption gate for AAA:" not in text, text
    # And it must never claim the symbol passed.
    for forbidden in ("open", "passes", "qualified", "adopted"):
        assert forbidden not in lowered.split(), text


def test_the_gate_line_says_where_the_gate_IS_measured():
    """A line that says "not here" and stops is only half an answer."""
    panel = _recap_panel()
    text = panel._gate_text("AAA", "long")
    assert "AAA" in text
    assert "your action is unaffected" in text.lower(), text
