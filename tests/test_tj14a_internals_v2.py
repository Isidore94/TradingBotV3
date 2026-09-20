"""TJ-14A items 4 and 6 - `trade_mentor_context_v2` and the rebuild.

RED BEFORE THE FIX. On `claude/tj14a-mentor-card`'s base (`8077a758`) the
builder emits `trade_mentor_context_v1`, seventeen symbols with four facts each,
no `derived` block, no `XLRE`, and there is no `internals_at`.

WHAT THESE PIN, AND WHY EACH IS A NUMBER
----------------------------------------
Every expectation is computed in THIS file from `tj14a_support.DAY`, a
hand-written table of `(close 30 minutes ago, last completed close)` pairs -
never from the code under test. The prior close is pinned at 100.0 for every
symbol, so a day change of `+2.00` is arithmetic a reader can check by eye, and
a builder that measures the day from the session OPEN, or from a forming bar,
prints a different number and fails.

The table is built so the two rankings DISAGREE: `XLF` leads the day at +2.00
while it is DOWN 0.39% over 30 minutes, and `XLRE` leads the 30 minutes while
sitting eighth on the day. One list reused for both fails. `XLE` and `XLF` are
exactly tied at the top of the day, so the tie-break by symbol is exercised.

`sectors_above_vwap` is decidable by hand because every bar carries the same
volume and `high + low + close) / 3 == close`, so the session VWAP is the mean
of the closes: nine of the eleven sectors sit above their own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj14a_support as fixture  # noqa: E402

SCHEMA_V2 = "trade_mentor_context_v2"


def _derived(snapshot, name: str) -> dict:
    block = snapshot["derived"]
    assert name in block, f"the derived block is missing {name}"
    return block[name]


def _measured(line: dict, *inputs: str) -> None:
    assert line["status"] == "measured"
    assert set(line["inputs"]) == set(inputs), (
        "a derived line names the readings it rests on"
    )


# ---------------------------------------------------------------------------
# The symbol list
# ---------------------------------------------------------------------------


def test_xlre_joins_the_internals_the_desk_already_watches():
    """The desk's own sector map has eleven SPDRs; the Mentor context had ten.

    `group_rrs.SECTOR_ETFS` is the desk's list and it carries `XLRE`, so a
    trader reading the strip saw a hole where real estate should be.
    """
    from group_rrs import SECTOR_ETFS
    from trade_mentor_context import SYMBOLS

    assert "XLRE" in SYMBOLS
    assert set(SECTOR_ETFS.values()) <= set(SYMBOLS)
    assert len(SYMBOLS) == 18
    assert len(set(SYMBOLS)) == len(SYMBOLS)


# ---------------------------------------------------------------------------
# Item 4 - the per-symbol facts
# ---------------------------------------------------------------------------


def test_a_symbol_carries_the_days_change_its_place_in_the_range_and_both_prior_day_sides():
    """SPY closes its last completed bar at 100.50 over a 100.00 prior close.

    The day is +0.50%, the place in the day's range is a fraction of ONE, and
    the two prior-day sides are read against Friday's 103.00 high and 97.00 low
    - so 100.50 is BELOW the high and ABOVE the low at the same time.
    """
    snapshot = fixture.context()
    assert snapshot["schema"] == SCHEMA_V2
    spy = fixture.reading(snapshot, "SPY")

    assert spy["day_change_pct"] == pytest.approx(0.50, abs=1e-6)
    assert spy["day_change_pct"] == pytest.approx(
        fixture.expected_day_change("SPY"), abs=1e-6
    )
    assert spy["day_range_place"] == pytest.approx(
        fixture.expected_range_place("SPY"), abs=1e-9
    )
    assert 0.0 <= spy["day_range_place"] <= 1.0
    assert spy["vs_prior_high"] == "below"
    assert spy["vs_prior_low"] == "above"

    # The existing v1 facts are untouched by the widening.
    assert spy["m5_change_30m_pct"] == pytest.approx(
        fixture.expected_thirty_minute_change("SPY"), abs=1e-6
    )
    assert spy["m5_status"] == "measured"
    assert spy["d1_status"] == "measured"

    # A symbol that closed ABOVE Friday's high, on the other side of both.
    xle = fixture.reading(snapshot, "XLE")
    assert xle["day_change_pct"] == pytest.approx(2.0, abs=1e-6)
    assert xle["vs_prior_low"] == "above"


def test_a_forming_bar_never_moves_a_v2_reading():
    """The 10:30 bar is not complete at 10:30; its 500.00 close must not show.

    Completed bars only (plan.md §5). A builder that reads `bars[-1]` prints a
    day change of +400% here.
    """
    settled = fixture.context()
    with_forming = fixture.context(forming=True)

    for symbol in ("SPY", "XLE", "VXX"):
        before = fixture.reading(settled, symbol)
        after = fixture.reading(with_forming, symbol)
        assert after["day_change_pct"] == pytest.approx(before["day_change_pct"], abs=1e-9)
        assert after["day_range_place"] == pytest.approx(before["day_range_place"], abs=1e-9)
        assert after["vs_prior_high"] == before["vs_prior_high"]
    assert with_forming["derived"] == settled["derived"]


# ---------------------------------------------------------------------------
# Item 4 - the derived block
# ---------------------------------------------------------------------------


def test_breadth_rates_and_oil_are_measured_from_the_readings_they_name():
    """RSP +1.00 against SPY +0.50 is breadth of +0.50, and it says so."""
    snapshot = fixture.context()

    breadth = _derived(snapshot, "breadth")
    _measured(breadth, "RSP", "SPY")
    assert breadth["value"] == pytest.approx(
        fixture.expected_day_change("RSP") - fixture.expected_day_change("SPY"), abs=1e-6
    )
    assert breadth["value"] == pytest.approx(0.50, abs=1e-6)

    rates = _derived(snapshot, "rates")
    _measured(rates, "TLT")
    assert rates["value"] == pytest.approx(0.80, abs=1e-6)

    oil = _derived(snapshot, "oil")
    _measured(oil, "USO")
    assert oil["value"] == pytest.approx(-2.10, abs=1e-6)


def test_fear_reads_vxx_against_spy_and_flags_the_divergence():
    """VXX down while SPY is up is the ordinary day. Both UP is the divergence.

    The flag is the whole point of the line: it is the day the trader would
    otherwise have typed out by hand.
    """
    ordinary = _derived(fixture.context(), "fear")
    _measured(ordinary, "VXX", "SPY")
    assert ordinary["vxx_direction"] == "down"
    assert ordinary["spy_direction"] == "up"
    assert ordinary["divergence"] is False

    from trade_mentor_context import build_context

    payload = fixture.bars()
    payload["m5"]["VXX"] = fixture.m5_bars_at(100.0, 101.0)
    both_up = build_context(
        now=fixture.NOW, m5_bars=payload["m5"], d1_bars=payload["d1"]
    )
    fear = _derived(both_up, "fear")
    assert fear["vxx_direction"] == "up"
    assert fear["spy_direction"] == "up"
    assert fear["divergence"] is True


def test_sector_leaders_and_laggards_rank_the_day_and_the_thirty_minutes_apart():
    """Two rankings, never one list twice, and a tie broken by symbol.

    XLE and XLF both finish the day at +2.00, so the top of the day is
    ["XLE", "XLF", ...] by name. XLF is DOWN over the same 30 minutes and XLRE
    leads it, so the two lists must differ.
    """
    snapshot = fixture.context()

    by_day = sorted(
        fixture.SECTORS, key=lambda name: (-fixture.expected_day_change(name), name)
    )
    by_m30 = sorted(
        fixture.SECTORS,
        key=lambda name: (-fixture.expected_thirty_minute_change(name), name),
    )

    leaders = _derived(snapshot, "sector_leaders")
    _measured(leaders, *fixture.SECTORS)
    assert leaders["day"] == by_day[:3] == ["XLE", "XLF", "XLK"]
    assert leaders["m30"] == by_m30[:3] == ["XLRE", "XLK", "XLY"]
    assert leaders["day"] != leaders["m30"]

    laggards = _derived(snapshot, "sector_laggards")
    _measured(laggards, *fixture.SECTORS)
    # Worst first, so a reader can stop after one name.
    assert laggards["day"] == list(reversed(by_day))[:3] == ["XLP", "XLU", "XLV"]
    assert laggards["m30"] == list(reversed(by_m30))[:3] == ["XLP", "XLU", "XLF"]


def test_offense_against_defense_and_the_count_of_sectors_above_their_own_vwap():
    """XLK/XLY/XLC average +1.20; XLP/XLU/XLV average -0.80. The read is +2.00.

    `sectors_above_vwap` is a COUNT WITH ITS DENOMINATOR (nine of eleven here),
    never a bare number and never a percentage of an unstated total.
    """
    snapshot = fixture.context()

    battle = _derived(snapshot, "offense_vs_defense")
    _measured(battle, *(fixture.OFFENSE + fixture.DEFENSE))
    offense = sum(fixture.expected_day_change(name) for name in fixture.OFFENSE) / 3
    defense = sum(fixture.expected_day_change(name) for name in fixture.DEFENSE) / 3
    assert battle["value"] == pytest.approx(offense - defense, abs=1e-6)
    assert battle["value"] == pytest.approx(2.00, abs=1e-6)

    above = _derived(snapshot, "sectors_above_vwap")
    _measured(above, *fixture.SECTORS)
    expected = [
        name
        for name in fixture.SECTORS
        if fixture.DAY[name][1] > fixture.expected_session_vwap(name)
    ]
    assert above["count"] == len(expected) == 9
    assert above["denominator"] == 11


def test_a_missing_reading_makes_only_its_own_derived_line_unmeasured():
    """No TLT bars at all: `rates` is `unmeasured` and says why; breadth is fine.

    Missing data is uncertainty, never zero (plan.md §5), and it never poisons
    a line that rests on other readings.
    """
    snapshot = fixture.context(drop=("TLT",))

    rates = _derived(snapshot, "rates")
    assert rates["status"] == "unmeasured"
    assert set(rates["inputs"]) == {"TLT"}
    assert str(rates.get("reason") or "").strip(), "an unmeasured line says why"
    assert rates.get("value") is None

    assert _derived(snapshot, "breadth")["status"] == "measured"
    assert _derived(snapshot, "fear")["status"] == "measured"
    assert _derived(snapshot, "sector_leaders")["status"] == "measured"


def test_an_unavailable_snapshot_is_v2_and_every_derived_line_is_unmeasured():
    """The card stores this while a read is still loading. It may not be blank."""
    from trade_mentor_context import unavailable_context

    snapshot = unavailable_context(now=fixture.NOW, reason="context pending")
    assert snapshot["schema"] == SCHEMA_V2
    assert [row["symbol"] for row in snapshot["readings"]][-1:] != []
    assert "XLRE" in {row["symbol"] for row in snapshot["readings"]}
    for name in (
        "breadth",
        "fear",
        "rates",
        "oil",
        "sector_leaders",
        "sector_laggards",
        "offense_vs_defense",
        "sectors_above_vwap",
    ):
        assert snapshot["derived"][name]["status"] == "unmeasured"


# ---------------------------------------------------------------------------
# Item 6 - the rebuild from the durable tape
# ---------------------------------------------------------------------------


def test_internals_at_rebuilds_exactly_what_the_live_card_would_have_shown():
    """One builder serves the live card and the rebuild - never two.

    Proven by equality on the SAME bars: every reading and every derived line
    matches the live snapshot. A second implementation drifts on the first
    rounding decision.
    """
    from trade_mentor_context import internals_at

    payload = fixture.bars()
    live = fixture.context()
    rebuilt = internals_at(fixture.SESSION.isoformat(), fixture.NOW, payload)

    assert rebuilt["schema"] == SCHEMA_V2
    assert rebuilt["readings"] == live["readings"]
    assert rebuilt["derived"] == live["derived"]


def test_the_session_tape_download_covers_every_internals_symbol(monkeypatch):
    """TJ-2A's ONE batched post-close download has to carry these names.

    An hour the trader never answered still gets its internals, and that is
    only possible if the durable tape holds the symbols. The decision and trade
    readers are stubbed so this asserts the FIXED part of the union only.
    """
    import day_review_bars
    import trade_mentor_context

    class _Journal:
        def list_trades(self, *, trade_date):
            return [{"symbol": "NVDA"}]

    monkeypatch.setattr(
        day_review_bars.daily_recap_reader, "_decisions", lambda *_a, **_k: ()
    )
    monkeypatch.setattr(day_review_bars, "shared_journal_service", lambda: _Journal())

    names = day_review_bars.decided_symbols(fixture.SESSION.isoformat(), object())
    missing = [name for name in trade_mentor_context.SYMBOLS if name not in names]
    assert missing == [], f"the durable tape would never hold {missing}"
    assert {"NVDA", "SPY"} <= names, "the existing union is untouched"


# ---------------------------------------------------------------------------
# The AI projection reads BOTH vintages
# ---------------------------------------------------------------------------


def test_the_compact_projection_still_reads_a_v1_row(monkeypatch):
    """GUARD, green before the fix: twenty-two live rows carry v1 and stay readable.

    Measured on a copy of `market_journal-202609.jsonl`, 2026-09-19: 22 of 28
    Trade Mentor rows hold a `trade_mentor_context_v1` snapshot. If v2 makes
    `compact_for_ai` blind to them, the AI silently loses every read before
    this packet.
    """
    from trade_mentor_context import compact_for_ai

    v1 = {
        "schema": "trade_mentor_context_v1",
        "captured_at": "2026-09-14T09:11:55-07:00",
        "availability": "available",
        "reason": "",
        "rules": {"m5": "…", "d1": "…"},
        "sources": {"m5": "yahoo", "d1": "cached"},
        "readings": [
            {
                "symbol": name,
                "m5_status": "measured",
                "m5_reason": "",
                "m5_as_of": "2026-09-14T09:10:00-07:00",
                "m5_change_30m_pct": 0.1,
                "m5_direction": "up",
                "m5_vs_session_vwap": "above",
                "d1_status": "measured",
                "d1_reason": "",
                "d1_as_of": "2026-09-11",
                "d1_change_5d_pct": 0.2,
                "d1_vs_sma20": "above",
            }
            for name in ("VXX", "RSP", "SPY")
        ],
    }
    compact = compact_for_ai(v1)
    assert compact is not v1, "a v1 row must still be projected, not passed through"
    assert "rows" in compact and "columns" in compact
    assert len(compact["rows"]) == 3


def test_the_compact_projection_carries_a_v2_rows_derived_block():
    """The derived lines are the reads the trader used to type. They must reach
    the AI, and they are identical for every symbol, so they belong in `common`."""
    from trade_mentor_context import compact_for_ai

    snapshot = fixture.context()
    compact = compact_for_ai(snapshot)

    assert compact is not snapshot
    assert len(compact["rows"]) == 18
    assert compact["common"]["schema"] == SCHEMA_V2
    assert compact["common"]["derived"] == snapshot["derived"]
