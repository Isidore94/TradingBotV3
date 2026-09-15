"""Packet D1C-B, item 1 - the ONE pure reader, `scripts/claimed_pick_evidence.py`.

Trader, 2026-09-14:

    *"Reuse the existing outcome and evidence services. ... Let me compare FAV,
    HC and My liked trades, and compare the setup types I claimed. Show sample
    sizes, pending and unmeasured results, and the existing uncertainty
    measures. Handle overlapping buckets explicitly. Repeated clicks must not
    create extra independent trades."*

Everything here is asserted as a NUMBER, on a fixture whose true answer is that
number, so a formula that prints a plausible-looking rate fails:

* the like cohort measures EXCHANGE SESSIONS from the claim day's close
  (`human_focus_tracking.HORIZONS`); the tracker measures SCAN ROWS
  (`swing_evidence.POLICY_SCANROW_V1`). Two clocks, side by side, never pooled.
* the ONE statistic is `swing_headline.wilson_lower_bound` (z 1.96) and a count.
  No bootstrap, no average of two cells (ground rule 10).
* HC has no forward record on the live tracker - `priority_bucket` is
  {favorite_setup, near_favorite_zone, blank} and `high_conviction` is an
  overlay computed at feed-write time - so an HC cell is UNMEASURED and says so
  in words, never 0%.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tests import d1c_claim_grading_fixtures as fx  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reader():
    import claimed_pick_evidence

    return claimed_pick_evidence


def _built(tmp_path, *, window="all", tiers=None, picks=None, outcomes=None, claim_rows=None):
    """Load the four stores off disk, then build - the real path the CLI takes."""
    module = _reader()
    paths = fx.write_store(
        tmp_path, tiers=tiers, picks=picks, outcomes=outcomes, claim_rows=claim_rows
    )
    inputs = module.load_inputs(**paths)
    return module.build_comparison(
        **inputs, as_of=date.fromisoformat(fx.AS_OF), window=window
    )


def _bound(wins, n):
    from swing_headline import wilson_lower_bound

    return wilson_lower_bound(wins, n)


def _numbers(comparison):
    """Every integer a population cell carries, for the never-summed check."""
    seen = []
    for population in comparison.populations.values():
        for name in (
            "n",
            "wins",
            "pending",
            "unmeasured",
            "also_fav",
            "also_near",
            "dropped_duplicates",
        ):
            value = getattr(population, name, None)
            if isinstance(value, int):
                seen.append(value)
    return seen


# ---------------------------------------------------------------------------
# The module, and the shape of its inputs
# ---------------------------------------------------------------------------


def test_build_comparison_opens_no_file_and_takes_every_store_as_rows(tmp_path, monkeypatch):
    """`build_comparison` is PURE. A file read belongs to `load_inputs`.

    Proved by taking the filesystem away: with `builtins.open` and
    `Path.open` raising, the build still produces the same four numbers.
    """
    import builtins
    import inspect

    module = _reader()
    parameters = inspect.signature(module.build_comparison).parameters
    assert set(parameters) >= {
        "like_picks",
        "like_outcomes",
        "tier_rows",
        "claims",
        "as_of",
        "window",
    }
    for name, parameter in parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name

    inputs = module.load_inputs(**fx.write_store(tmp_path))

    def _refuse(*args, **kwargs):
        raise AssertionError("build_comparison opened a file")

    monkeypatch.setattr(builtins, "open", _refuse)
    monkeypatch.setattr(Path, "open", _refuse)
    comparison = module.build_comparison(
        **inputs, as_of=date.fromisoformat(fx.AS_OF), window="all"
    )
    assert comparison.populations["liked"].n == 4
    assert comparison.populations["fav"].n == 5


# ---------------------------------------------------------------------------
# My liked trades - n, wins, pending, unmeasured, duplicates, quick likes
# ---------------------------------------------------------------------------


def test_my_liked_trades_count_four_graded_three_won_and_one_duplicate_dropped(tmp_path):
    """Nine pick rows; exactly four are graded trades and three of them won.

    AAA (win), BBB (loss), CCC (win), OLD (win) are measured; DDD's fifth
    session has not closed; EEE has no outcome row; FFF is `like_unclaimed`;
    GGG is a QUICK like; the second AAA row is the same click again.
    """
    comparison = _built(tmp_path, window="all")
    liked = comparison.populations["liked"]

    assert liked.n == 4
    assert liked.wins == 3
    assert liked.win_rate == pytest.approx(0.75)
    assert liked.bound == pytest.approx(_bound(3, 4))
    assert liked.pending == 1
    assert liked.unmeasured == 1
    assert liked.dropped_duplicates == 1


def test_a_quick_like_is_excluded_and_counted_once_in_a_footnote(tmp_path):
    """P9: only a CLAIMED like names a setup. GGG is the one quick like."""
    comparison = _built(tmp_path, window="all")
    assert any("quick likes excluded: 1" in str(note) for note in comparison.footnotes), (
        comparison.footnotes
    )
    symbols = {row.symbol for row in comparison.liked_rows}
    assert "GGG" not in symbols
    assert "FFF" not in symbols, "an unclaimed like is not one of MY claimed trades"


def test_an_unmatured_horizon_is_pending_and_a_missing_row_is_unmeasured(tmp_path):
    """The two kinds of "no answer yet" are different facts and stay apart."""
    comparison = _built(tmp_path, window="all")
    by_symbol = {row.symbol: row for row in comparison.liked_rows}

    assert by_symbol["DDD"].status == "pending"
    assert by_symbol["DDD"].win is None
    assert by_symbol["EEE"].status == "unmeasured"
    assert by_symbol["EEE"].unmeasured_reason == "no outcome row"
    assert by_symbol["AAA"].status == "measured"
    assert by_symbol["AAA"].win is True
    assert by_symbol["BBB"].win is False


def test_a_claimed_row_carries_what_was_known_then_and_an_old_one_says_so(tmp_path):
    """The claim's own record travels with the row and is never graded."""
    comparison = _built(tmp_path, window="all")
    by_symbol = {row.symbol: row for row in comparison.liked_rows}

    assert by_symbol["AAA"].claimed_setup_id == "avwap_breakout"
    assert by_symbol["AAA"].claim_at == "2026-09-02T09:41:00-07:00"
    assert by_symbol["AAA"].known_at_claim == {"close": 24.58, "rrs": 1.4}
    assert by_symbol["BBB"].provenance == "annotation only (pre-D1C)"


# ---------------------------------------------------------------------------
# FAV / Near / HC - the tracker's own clock
# ---------------------------------------------------------------------------


def test_fav_grades_five_scan_rows_and_four_won(tmp_path):
    comparison = _built(tmp_path, window="all")
    fav = comparison.populations["fav"]

    assert fav.n == 5
    assert fav.wins == 4
    assert fav.win_rate == pytest.approx(0.80)
    assert fav.bound == pytest.approx(_bound(4, 5))
    assert fav.pending == 0


def test_a_stale_horizon_row_and_a_wrong_horizon_row_are_never_graded(tmp_path):
    """FV5 is `stale_horizon` True and FV6 carries horizon 1. Seven favorite
    rows go in and five come out - the ONE policy, `POLICY_SCANROW_V1`."""
    comparison = _built(tmp_path, window="all")
    assert comparison.populations["fav"].n == 5, "the policy's exclusions were not applied"


def test_near_is_a_labelled_extra_and_grades_its_own_two_rows(tmp_path):
    comparison = _built(tmp_path, window="all")
    near = comparison.populations["near"]

    assert near.n == 2
    assert near.wins == 1
    assert near.win_rate == pytest.approx(0.5)
    assert near.bound == pytest.approx(_bound(1, 2))


def test_the_two_clocks_are_named_and_never_the_same_unit(tmp_path):
    comparison = _built(tmp_path, window="all")
    liked_clock = comparison.populations["liked"].clock
    fav_clock = comparison.populations["fav"].clock

    assert "session" in liked_clock.lower()
    assert "scan row" in fav_clock.lower()
    assert liked_clock != fav_clock


# ---------------------------------------------------------------------------
# HC - the honesty line
# ---------------------------------------------------------------------------


def test_hc_with_no_rows_is_unmeasured_in_words_and_never_a_percent(tmp_path):
    comparison = _built(tmp_path, window="all")
    hc = comparison.populations["hc"]

    assert hc.n == 0
    assert hc.win_rate is None
    assert hc.bound is None
    assert hc.note == (
        "unmeasured: the tracker records favorite_setup / near_favorite_zone "
        "only (0 HC rows)"
    )
    assert "%" not in hc.note


def test_hc_is_graded_when_the_tracker_ever_stamps_it(tmp_path):
    """The same reader, on a tracker that DOES carry `high_conviction`."""
    comparison = _built(
        tmp_path, window="all", tiers=fx.tier_rows() + fx.high_conviction_rows()
    )
    hc = comparison.populations["hc"]

    assert hc.n == 2
    assert hc.wins == 1
    assert hc.win_rate == pytest.approx(0.5)
    assert hc.bound == pytest.approx(_bound(1, 2))
    assert hc.note == ""


# ---------------------------------------------------------------------------
# Overlap is NAMED, never summed
# ---------------------------------------------------------------------------


def test_an_overlapping_pick_is_counted_once_on_each_side_and_named_once(tmp_path):
    """AAA is liked on 2026-09-02 and a FAV scan row that same day; CCC is
    liked on 2026-09-03 and a Near row that same day."""
    comparison = _built(tmp_path, window="all")
    liked = comparison.populations["liked"]

    assert liked.also_fav == 1
    assert liked.also_near == 1
    assert liked.n == 4, "the overlap was removed from My liked trades"
    assert comparison.populations["fav"].n == 5, "the overlap was removed from FAV"


def test_no_rendered_cell_is_the_sum_of_two_populations(tmp_path):
    """A union of two clocks is not a sample. 4 + 5 must appear nowhere."""
    comparison = _built(tmp_path, window="all")
    liked_n = comparison.populations["liked"].n
    fav_n = comparison.populations["fav"].n
    union = liked_n + fav_n

    assert union == 9
    assert union not in _numbers(comparison), "a cell pooled the two clocks"


# ---------------------------------------------------------------------------
# By claimed setup, and the leader floor
# ---------------------------------------------------------------------------


def test_by_claimed_setup_is_one_row_per_source_sorted_by_the_bound(tmp_path):
    comparison = _built(tmp_path, window="all")
    rows = {row.source: row for row in comparison.by_setup}

    assert set(rows) == {fx.BREAKOUT, fx.ONE_STDEV}
    assert rows[fx.BREAKOUT].n == 4
    assert rows[fx.BREAKOUT].wins == 3
    assert rows[fx.BREAKOUT].bound == pytest.approx(_bound(3, 4))
    assert rows[fx.ONE_STDEV].n == 0
    assert rows[fx.ONE_STDEV].pending == 1
    assert rows[fx.ONE_STDEV].unmeasured == 1
    assert rows[fx.ONE_STDEV].bound is None
    # Sorted by the BOUND, with the ungraded family last.
    assert [row.source for row in comparison.by_setup] == [fx.BREAKOUT, fx.ONE_STDEV]


def test_below_the_floor_the_leader_names_no_setup_and_says_the_best_n(tmp_path):
    from evidence_stats import MIN_REPORTABLE_N

    assert MIN_REPORTABLE_N == 30
    comparison = _built(tmp_path, window="all")

    assert comparison.leader == "no setup has n >= 30 yet (best n was 4)"
    assert fx.BREAKOUT not in comparison.leader


def test_at_thirty_the_leader_names_the_setup_with_a_real_bound(tmp_path):
    """Thirty graded claims of one setup, twenty-one of them won: 70%,
    Wilson lower bound 0.5212421254128504."""
    source = "like_post_earnings_52w_break"
    picks = [
        fx.like_pick("2026-09-02", f"S{index:02d}", "LONG", source) for index in range(30)
    ]
    outcomes = [
        fx.like_outcome(
            "2026-09-02",
            f"S{index:02d}",
            "LONG",
            source,
            h5_return="0.021000" if index < 21 else "-0.019000",
            matured_horizons="1,3,5,10",
        )
        for index in range(30)
    ]
    comparison = _built(tmp_path, window="all", picks=picks, outcomes=outcomes, claim_rows=[])
    row = comparison.by_setup[0]

    assert row.source == source
    assert row.n == 30
    assert row.wins == 21
    assert row.win_rate == pytest.approx(0.70)
    assert row.bound == pytest.approx(0.5212421254128504)
    assert "post_earnings_52w_break" in comparison.leader
    assert "no setup has n >= 30 yet" not in comparison.leader


# ---------------------------------------------------------------------------
# The two windows
# ---------------------------------------------------------------------------


def test_the_reader_returns_both_windows_and_lately_is_twenty_sessions(tmp_path):
    """`lately` is 2026-08-14 .. 2026-09-11 - 20 EXCHANGE SESSIONS, and the
    2024-01-03 rows fall outside it on both sides."""
    from evidence_stats import LATELY_SESSIONS

    assert LATELY_SESSIONS == 20
    comparison = _built(tmp_path, window="lately")

    assert set(comparison.by_window) == {"lately", "all"}
    assert comparison.window == "lately"
    assert comparison.populations is comparison.by_window["lately"].populations
    assert comparison.by_window["lately"].window_dates == (fx.LATELY_FIRST, fx.AS_OF)

    lately = comparison.by_window["lately"].populations
    assert lately["liked"].n == 3
    assert lately["liked"].wins == 2
    assert lately["liked"].win_rate == pytest.approx(2 / 3)
    assert lately["liked"].bound == pytest.approx(_bound(2, 3))
    assert lately["fav"].n == 4
    assert lately["fav"].wins == 3

    every = comparison.by_window["all"].populations
    assert every["liked"].n == 4
    assert every["fav"].n == 5


def test_the_lately_leader_counts_only_the_lately_rows(tmp_path):
    comparison = _built(tmp_path, window="lately")
    assert comparison.leader == "no setup has n >= 30 yet (best n was 3)"


def test_an_iso_string_as_of_reads_the_same_as_a_date(tmp_path):
    """`--as-of YYYY-MM-DD` reaches the reader as text."""
    module = _reader()
    paths = fx.write_store(tmp_path)
    inputs = module.load_inputs(**paths)
    from_text = module.build_comparison(**inputs, as_of=fx.AS_OF, window="lately")
    from_date = module.build_comparison(
        **inputs, as_of=date.fromisoformat(fx.AS_OF), window="lately"
    )
    assert from_text.populations["liked"].n == from_date.populations["liked"].n == 3


# ---------------------------------------------------------------------------
# Repeated clicks, driven through the REAL like-cohort path
# ---------------------------------------------------------------------------


def test_three_clicks_on_one_chart_are_one_liked_row_and_one_claim(tmp_path):
    """Drive `like_cohort.like_pick_rows` (the writer) into the reader.

    Three `like_claim` annotations for one session and three `claim` rows for
    one key must produce ONE graded trade - not three, and not a trade that
    counts its own repeats.
    """
    from ui.annotations.like_cohort import like_pick_rows
    from ui.annotations.store import EVENT_LIKE_CLAIM

    annotations = [
        {
            "event_type": EVENT_LIKE_CLAIM,
            "symbol": "AAA",
            "side": "LONG",
            "session_date": "2026-09-02",
            "claimed_setup_id": "avwap_breakout",
            "like_mode": "claimed",
            "surface": "chart_review",
        }
        for _ in range(3)
    ]
    picks, skipped_no_side = like_pick_rows(annotations)
    assert len(picks) == 1 and skipped_no_side == 0

    module = _reader()
    repeated_claims = [
        fx.claim_row("AAA", "LONG", "avwap_breakout", "2026-09-02") for _ in range(3)
    ]
    paths = fx.write_store(
        tmp_path,
        picks=picks,
        outcomes=[
            fx.like_outcome(
                "2026-09-02", "AAA", "LONG", fx.BREAKOUT,
                h5_return="0.052631", matured_horizons="1,3,5,10",
            )
        ],
        tiers=[],
        claim_rows=repeated_claims,
    )
    comparison = module.build_comparison(
        **module.load_inputs(**paths), as_of=date.fromisoformat(fx.AS_OF), window="all"
    )
    liked = comparison.populations["liked"]

    assert liked.n == 1
    assert liked.wins == 1
    assert len(comparison.liked_rows) == 1


def test_a_second_pick_row_for_one_key_is_a_duplicate_not_a_second_trade(tmp_path):
    """The same three picks written to the file twice is still one trade."""
    module = _reader()
    row = fx.like_pick("2026-09-02", "AAA", "LONG", fx.BREAKOUT)
    paths = fx.write_store(
        tmp_path,
        picks=[row, dict(row), dict(row)],
        outcomes=[
            fx.like_outcome(
                "2026-09-02", "AAA", "LONG", fx.BREAKOUT,
                h5_return="0.052631", matured_horizons="1,3,5,10",
            )
        ],
        tiers=[],
        claim_rows=[],
    )
    comparison = module.build_comparison(
        **module.load_inputs(**paths), as_of=date.fromisoformat(fx.AS_OF), window="all"
    )

    assert comparison.populations["liked"].n == 1
    assert comparison.populations["liked"].dropped_duplicates == 2


# ---------------------------------------------------------------------------
# Nothing moves, and the journal is a different surface
# ---------------------------------------------------------------------------


def test_the_reader_builds_with_the_journal_modules_unimportable(tmp_path, monkeypatch):
    """Item 4 / the trader's "keep opportunity results separate from actual
    journal trade results": this packet imports nothing from the journal."""
    import importlib.abc
    import importlib.machinery

    forbidden = {"journal_store", "preference_trade_outcomes", "journal_analytics"}

    class _Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                raise ImportError(f"{fullname} is a different surface")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in forbidden:
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Refuse(), *sys.meta_path])

    comparison = _built(tmp_path, window="all")
    assert comparison.populations["liked"].n == 4


def test_the_reader_writes_nothing(tmp_path):
    """Item 4: no weight, policy, detector, tracker or cohort file is written."""
    module = _reader()
    paths = fx.write_store(tmp_path)
    before = {
        path: Path(path).stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()
    }
    listing_before = sorted(str(path) for path in tmp_path.rglob("*"))

    module.build_comparison(
        **module.load_inputs(**paths), as_of=date.fromisoformat(fx.AS_OF), window="all"
    )

    assert sorted(str(path) for path in tmp_path.rglob("*")) == listing_before
    assert {
        path: Path(path).stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()
    } == before


def test_a_missing_store_is_zero_rows_and_never_a_raised_reader(tmp_path):
    """The trader opens this tab mid-session; an absent file renders empty."""
    module = _reader()
    inputs = module.load_inputs(
        picks_path=tmp_path / "nope-picks.csv",
        outcomes_path=tmp_path / "nope-outcomes.csv",
        tier_path=tmp_path / "nope-tier.csv",
        claims_path=tmp_path / "nope-claims.jsonl",
    )
    comparison = module.build_comparison(
        **inputs, as_of=date.fromisoformat(fx.AS_OF), window="all"
    )

    assert comparison.populations["liked"].n == 0
    assert comparison.populations["liked"].win_rate is None
    assert len(comparison.by_setup) == 0
    assert comparison.leader == "no setup has n >= 30 yet (best n was 0)"
