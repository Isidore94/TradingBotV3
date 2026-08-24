"""R10.B - outcome semantics and path capture.

Two things the outcome store could not previously say, and one it said wrongly:

* **What kind of claim is this?** Every registered row was measured as a trade.
  `regime_pause_rw` never claimed to be one and carries an all-time mean of
  -1.82R across n=934 (audit D7); the three retired H1 engines are 82% of every
  registered row and mark a bar that already CLOSED (D6a/D6b). Averaging those
  as trades is not a rounding error, it is a category error.
* **What did price actually do?** Only the one configured exit rule's answer was
  kept, so "would a different exit have done better" needed a refetch of bars
  that were in memory at the time.
* **LRSI produced nothing at all** (D5a): the registration bar was synthetic
  and flat, so risk was zero and registration returned early - 0 outcome rows
  for both levels.

The fixture is 78 real M5 bars per symbol for 2026-08-21, frozen in the repo.
The tests over it assert the HONEST calculation, never a desired sign: none of
them says EAT or CAKE should have worked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import outcome_path  # noqa: E402
import outcome_semantics as sem  # noqa: E402

FIXTURE = ROOT_DIR / "tests" / "fixtures" / "outcome_path_eat_cake_v1.json"


@pytest.fixture(scope="module")
def frozen():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _bars(frozen, symbol):
    return outcome_path.to_bars(frozen["sessions"][symbol]["bars"])


# ==========================================================================
# The registry
# ==========================================================================
def test_an_unknown_family_is_unconfigured_and_never_a_trade():
    """The whole point. A family nobody classified is UNMEASURED, and the one
    thing it must never become is a trade with an R."""
    spec = sem.spec_for("some_engine_nobody_registered")

    assert spec.claim_kind == sem.CLAIM_UNCONFIGURED
    assert sem.is_trade_bearing("some_engine_nobody_registered") is False
    assert "not therefore a trade" in spec.why


def test_a_family_is_never_guessed_from_a_similar_name():
    """Prefix matching would quietly enrol every future family into whatever
    its neighbour claimed - which is the mechanism this registry exists to
    replace, not a shortcut it can afford."""
    assert sem.claim_kind("lrsi_cross_20") == sem.CLAIM_ENTRY
    assert sem.claim_kind("lrsi_cross_80") == sem.CLAIM_UNCONFIGURED


def test_the_pause_observation_is_information_not_a_trade():
    """Audit D7: `regime_pause_rw` all-time n=934, mean -1.82R. It never
    claimed an entry; the -1.82 is a statement about a category error."""
    assert sem.claim_kind("regime_pause_rw") == sem.CLAIM_INFORMATION
    assert sem.is_trade_bearing("regime_pause_rw") is False


def test_the_retired_h1_engines_are_annotations():
    """Audit D6a/D6b: 6,439 of 6,439 H1 rows stamped on the bar START, and they
    are 82% of every registered row. A mark on a closed bar is not an entry.

    The three names are the ones the STORE contains. An earlier draft of the
    registry was written from the audit's prose and invented two of them
    (`h1_red_after_blue`, `h1_reversal`) while missing `h1_ema10_bounce` -
    which is the single largest family in the store at 92,477 rows. Reading the
    27 distinct level names is what corrected it.
    """
    for family in ("h1_ema10_bounce", "h1_blue_after_red", "h1_green_to_yellow"):
        assert sem.claim_kind(family) == sem.CLAIM_ANNOTATION
        assert sem.is_trade_bearing(family) is False


def test_an_h1_derived_level_is_an_entry_claim_despite_the_shared_prefix():
    """The trap that makes prefix matching unusable here: `h1_ema_15` is an H1
    average used as a bounce LEVEL on an M5 bar - an ordinary entry claim -
    while `h1_ema10_bounce` is a colour annotation. They share a prefix and
    mean different things."""
    assert sem.claim_kind("h1_ema_15") == sem.CLAIM_ENTRY
    assert sem.claim_kind("h1_sma_20") == sem.CLAIM_ENTRY
    assert sem.claim_kind("h1_ema10_bounce") == sem.CLAIM_ANNOTATION


def test_a_compound_family_is_decided_by_its_parts():
    """`_make_bounce_event_id` builds the family as the sorted level names
    joined by `-`, so splitting on it recovers the exact parts. That is
    construction, not similarity - and without it 158,053 live rows read as
    unconfigured, which is how this was found."""
    spec = sem.spec_for("10_candle_high-vwap_lower_band")

    assert spec.claim_kind == sem.CLAIM_ENTRY
    assert "compound of 2 level(s)" in spec.why


def test_a_compound_whose_parts_disagree_is_unconfigured():
    """A row whose pieces claim different things has not been classified,
    whatever each piece says on its own."""
    spec = sem.spec_for("vwap-regime_pause_rw")

    assert spec.claim_kind == sem.CLAIM_UNCONFIGURED
    assert "disagree about what they claim" in spec.why


def test_a_compound_with_one_unknown_part_is_unconfigured_and_names_it():
    spec = sem.spec_for("vwap-some_level_nobody_declared")

    assert spec.claim_kind == sem.CLAIM_UNCONFIGURED
    assert "some_level_nobody_declared" in spec.why


def test_a_row_with_no_family_at_all_is_unconfigured():
    assert sem.spec_for("").claim_kind == sem.CLAIM_UNCONFIGURED
    assert sem.spec_for(None).claim_kind == sem.CLAIM_UNCONFIGURED


def test_lrsi_is_an_entry_claim_even_though_it_produced_no_rows():
    """D5a is a defect in the WRITER, not a statement about the family. LRSI
    claims an entry; it was never able to record one."""
    assert sem.claim_kind("lrsi_cross_20") == sem.CLAIM_ENTRY
    assert sem.is_trade_bearing("lrsi_cross_50") is True


def test_coverage_names_the_unconfigured_families_loudly():
    """A count nobody can act on is not a coverage report. The NAMES are the
    to-do list."""
    result = sem.coverage(
        ["vwap", "regime_pause_rw", "mystery_one", "mystery_two", "mystery_one"]
    )

    assert result["counts"][sem.CLAIM_UNCONFIGURED] == 3
    assert result["unconfigured_families"] == ["mystery_one", "mystery_two"]
    assert "mystery_one" in result["note"] and "mystery_two" in result["note"]
    assert "excluded from every trade statistic" in result["note"]


def test_the_registry_is_named_never_numbered():
    """Ground rule 5: a changed meaning is a new NAME."""
    assert sem.REGISTRY_NAME == "outcome_claim_kinds_v1"
    assert sem.coverage([])["registry"] == sem.REGISTRY_NAME


# ==========================================================================
# Path capture over the frozen bars
# ==========================================================================
def test_the_fixture_is_two_real_sessions(frozen):
    """And it carries the Milestone 3 contract, like every shipped fixture."""
    assert frozen["schema"] == "outcome_path_eat_cake_v1"
    assert frozen["intentional_difference"].startswith("None -")
    assert "Zero IB traffic" in frozen["provider_assumptions"]
    for symbol in ("EAT", "CAKE"):
        session = frozen["sessions"][symbol]
        assert session["session_date"] == "2026-08-21"
        assert len(session["bars"]) == 78  # a full regular M5 session
        assert session["bars"][0]["time"].endswith("-04:00")  # tz-aware, market-local


def test_a_long_path_measures_excursion_not_just_the_close(frozen):
    """The number a close-only record cannot produce: a trade that ran and gave
    it back is not the same trade as one that never moved."""
    bars = _bars(frozen, "EAT")
    entry = bars[0].close
    stop = entry - 1.0
    result = outcome_path.capture_path(
        entry_price=entry, stop_price=stop, side="long", bars=bars
    )

    assert result["measurable"] is True
    assert result["bars_measured"] == 78
    assert result["mfe_r"] >= result["close_r"]
    assert result["mae_r"] <= result["close_r"]
    assert result["giveback_r"] == pytest.approx(
        max(0.0, result["mfe_r"] - result["close_r"]), abs=1e-3
    )
    assert len(result["excursion_r"]) == 78
    assert set(result["at_marks"]) == {"1", "3", "6", "12", "24", "36"}


def test_giveback_is_never_negative(frozen):
    """A trade whose close IS its high gave nothing back. A negative giveback
    would read as "it gained after the end", which is not a thing."""
    bars = _bars(frozen, "CAKE")
    entry = bars[0].close
    result = outcome_path.capture_path(
        entry_price=entry, stop_price=entry - 0.5, side="long", bars=bars
    )
    assert result["giveback_r"] >= 0.0


def test_a_short_is_mirrored_not_re_derived(frozen):
    """Same bars, same distance, opposite side: the favourable direction flips.
    Writing the short case by hand is how the two drift apart."""
    bars = _bars(frozen, "EAT")
    entry = bars[0].close
    long_side = outcome_path.capture_path(
        entry_price=entry, stop_price=entry - 1.0, side="long", bars=bars
    )
    short_side = outcome_path.capture_path(
        entry_price=entry, stop_price=entry + 1.0, side="short", bars=bars
    )

    assert long_side["close_r"] == pytest.approx(-short_side["close_r"], abs=1e-3)
    assert long_side["mfe_r"] == pytest.approx(-short_side["mae_r"], abs=1e-3)


def test_the_stop_wins_a_bar_that_contains_both(frozen):
    """A bar's OHLC carries no intrabar sequence. Taking the favourable order
    would manufacture a profit out of an unknown, on every such bar, in one
    direction only - so the STOP is taken first and the row says it did."""
    wide = [{"high": 120.0, "low": 80.0, "close": 100.0}]
    result = outcome_path.capture_path(
        entry_price=100.0, stop_price=90.0, side="long", bars=wide
    )

    assert result["first_stop_bar"] == 1
    assert result["stop_first_intrabar"] is True
    assert result["first_target_bar"] is None
    assert result["exit_policies"]["eod_hold"]["r"] == -1.0


def test_zero_risk_is_unmeasurable_never_a_zero_result():
    """D5a's shape: a flat bar makes entry == stop. That is not a trade that
    risked nothing, it is a row R has no meaning for."""
    flat = {"high": 50.0, "low": 50.0, "close": 50.0}
    result = outcome_path.capture_path(
        entry_price=50.0, stop_price=50.0, side="long", bars=[flat]
    )

    assert result["measurable"] is False
    assert result["mfe_r"] is None and result["close_r"] is None
    assert "no meaning" in result["reason"]


def test_no_bars_yet_is_unmeasurable_not_flat(frozen):
    result = outcome_path.capture_path(
        entry_price=100.0, stop_price=99.0, side="long", bars=[]
    )
    assert result["measurable"] is False
    assert "nothing has been measured yet" in result["reason"]


# ==========================================================================
# The frozen exit policies
# ==========================================================================
def test_every_frozen_policy_reports_on_its_own(frozen):
    bars = _bars(frozen, "CAKE")
    entry = bars[0].close
    result = outcome_path.capture_path(
        entry_price=entry, stop_price=entry - 0.5, side="long", bars=bars, atr=1.0
    )

    assert set(result["exit_policies"]) == set(outcome_path.FROZEN_EXIT_POLICIES)
    for name, value in result["exit_policies"].items():
        assert value["reason"], f"{name} must say why it stopped where it did"


def test_a_policy_missing_its_input_is_unmeasured_not_zero(frozen):
    """`vwap_close_after_1r` without VWAP, and `atr_1p5_trail` without an ATR.

    A policy that silently degrades into a different policy when its input is
    missing publishes a number under the wrong name.
    """
    bars = _bars(frozen, "EAT")  # the fixture carries no VWAP column
    entry = bars[0].close
    result = outcome_path.capture_path(
        entry_price=entry, stop_price=entry - 1.0, side="long", bars=bars, atr=None
    )

    vwap = result["exit_policies"]["vwap_close_after_1r"]
    atr = result["exit_policies"]["atr_1p5_trail"]
    assert vwap["r"] is None and "unmeasured (not zero)" in vwap["reason"]
    assert atr["r"] is None and "unmeasured (not zero)" in atr["reason"]


def test_a_pre_target_fallback_is_reported_under_its_own_policy_name(frozen):
    """Two policies hold when +1R never arrives. The result has to carry THEIR
    name; `eod_hold` appearing inside the trail column mis-attributes it."""
    bars = [{"high": 100.2, "low": 99.9, "close": 100.0}] * 5
    result = outcome_path.capture_path(
        entry_price=100.0, stop_price=90.0, side="long", bars=bars, atr=1.0
    )
    policies = result["exit_policies"]

    assert policies["trail_2bar_after_1r"]["r"] is not None
    assert result["first_target_bar"] is None


def test_the_oracle_is_labelled_an_upper_bound_and_beats_no_policy(frozen):
    """Ground rule 12. It is the best of the frozen policies chosen with
    hindsight - never a result, never attributable to a policy."""
    bars = _bars(frozen, "EAT")
    entry = bars[0].close
    result = outcome_path.capture_path(
        entry_price=entry, stop_price=entry - 1.0, side="long", bars=bars, atr=1.0
    )

    scored = [v["r"] for v in result["exit_policies"].values() if v["r"] is not None]
    assert result[outcome_path.ORACLE_KEY] == pytest.approx(max(scored), abs=1e-6)
    assert "upper bound" in result["oracle_note"]
    assert "never a result" in result["oracle_note"]
    # And it is not itself a policy anyone can select.
    assert outcome_path.ORACLE_KEY not in result["exit_policies"]


def test_realizable_r_is_not_a_field_name_anywhere(frozen):
    """Ground rule 12: "realizable R" is not a term this repo uses.

    Checked against the emitted PAYLOAD rather than the source text, because
    the module's own docstring names the term in order to rule it out - and a
    test that cannot tell those apart teaches you to delete the explanation.
    """
    bars = _bars(frozen, "EAT")
    entry = bars[0].close
    result = outcome_path.capture_path(
        entry_price=entry, stop_price=entry - 1.0, side="long", bars=bars, atr=1.0
    )
    emitted = json.dumps(result).lower()
    assert "realizable" not in emitted
    assert all("realizable" not in key.lower() for key in result)


def test_the_path_payload_is_json_safe(frozen):
    """It is written to an append-only ledger line; anything unserializable
    turns a row into a torn line."""
    bars = _bars(frozen, "CAKE")
    entry = bars[0].close
    result = outcome_path.capture_path(
        entry_price=entry, stop_price=entry - 0.5, side="long", bars=bars, atr=1.0
    )
    assert json.loads(json.dumps(result))["schema"] == outcome_path.PATH_SCHEMA


# ==========================================================================
# The wiring: which rows actually get a path
# ==========================================================================
def _final_context(family, bars, *, finalize=True):
    import pandas as pd
    from bounce_bot_lib import legacy

    frame = pd.DataFrame(bars) if bars else pd.DataFrame()
    state = {
        "entry_price": 100.0,
        "stop_price": 99.0,
        "direction": "long",
        "context": {"family": family, "atr": 1.0},
    }
    return legacy._context_with_path({}, state, frame, finalize_eod=finalize)


def test_only_a_final_row_carries_the_path():
    """A milestone row would store the same growing array four times per trade
    and answer nothing extra."""
    bars = [{"high": 101.0, "low": 99.5, "close": 100.5, "open": 100.0}]
    assert _final_context("vwap", bars, finalize=False) == {}


def test_an_entry_claim_gets_a_path_on_its_final_row():
    bars = [{"high": 102.0, "low": 99.5, "close": 101.5, "open": 100.0}]
    context = _final_context("vwap", bars)

    assert context["path"]["schema"] == outcome_path.PATH_SCHEMA
    assert context["path"]["measurable"] is True
    assert "path_absent" not in context


def test_an_annotation_gets_no_path_and_the_row_says_why():
    """An MFE "in R" over a family with no entry is a number about a
    denominator that does not exist. Recorded as absent, not omitted: a reader
    must tell "not a trade" from "something failed"."""
    bars = [{"high": 102.0, "low": 99.5, "close": 101.5, "open": 100.0}]
    context = _final_context("h1_ema10_bounce", bars)

    assert "path" not in context
    assert context["path_absent"]["claim_kind"] == sem.CLAIM_ANNOTATION
    assert context["path_absent"]["reason"] == "family does not claim an entry"


def test_an_observation_gets_no_path_either():
    bars = [{"high": 102.0, "low": 99.5, "close": 101.5, "open": 100.0}]
    context = _final_context("regime_pause_rw", bars)
    assert context["path_absent"]["claim_kind"] == sem.CLAIM_INFORMATION


def test_no_bars_records_an_absent_path_rather_than_an_empty_one():
    context = _final_context("vwap", [])
    assert context["path_absent"]["reason"] == "no bars after entry were measured"
