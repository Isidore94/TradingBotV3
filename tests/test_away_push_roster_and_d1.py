"""The phone push carries the full favorite/high-conviction roster, and D1
level events get their own hourly push (trader ask 2026-08-11).

Three behaviours matter more than the formatting:

- the roster is a MEMBERSHIP answer, so it must be complete or say it is not;
- the D1 push carries only what is NEW since the last one, because a
  cumulative hourly resend teaches the trader to swipe the channel away;
- neither push may ever run in a mode other than AWAY. The Research-tab price
  alerts are the only always-on phone channel.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autopilot_core as core  # noqa: E402


def _row(symbol, bucket="favorite_setup", side="LONG"):
    return {"symbol": symbol, "side": side, "bucket": bucket}


# ---------------------------------------------------------------------------
# The roster itself
# ---------------------------------------------------------------------------


def test_the_roster_splits_by_bucket_and_side():
    roster = core.build_bucket_roster(
        [
            _row("NVDA", "high_conviction"),
            _row("COIN", "high_conviction", side="SHORT"),
            _row("AMD"),
            _row("SMCI", side="SHORT"),
        ]
    )
    assert roster["high_conviction"] == {"LONG": ["NVDA"], "SHORT": ["COIN"]}
    assert roster["favorite"] == {"LONG": ["AMD"], "SHORT": ["SMCI"]}


def test_raw_keys_and_display_labels_land_in_the_same_bucket():
    """The scan CSV says high_conviction; SetupRow.bucket_label says
    'High Conviction'. Two spellings must not become two rosters."""
    roster = core.build_bucket_roster(
        [_row("NVDA", "high_conviction"), _row("AMD", "High Conviction")]
    )
    assert roster["high_conviction"]["LONG"] == ["NVDA", "AMD"]
    assert core.normalize_bucket_key("Favorite") == "favorite"
    assert core.normalize_bucket_key("favorite_setup") == "favorite"
    assert core.normalize_bucket_key("near_favorite_zone") == "near"


def test_near_and_unrelated_buckets_stay_out_of_the_roster():
    roster = core.build_bucket_roster(
        [_row("AAA", "near_favorite_zone"), _row("BBB", "study"), _row("CCC")]
    )
    assert roster["favorite"]["LONG"] == ["CCC"]
    assert all("AAA" not in side for side in roster["favorite"].values())


def test_a_symbol_on_both_sides_keeps_both_entries():
    """Long and short disagreeing about one name is information, not noise."""
    roster = core.build_bucket_roster([_row("NVDA"), _row("NVDA", side="SHORT")])
    assert roster["favorite"] == {"LONG": ["NVDA"], "SHORT": ["NVDA"]}


def test_duplicates_and_junk_rows_never_reach_the_phone():
    roster = core.build_bucket_roster(
        [_row("NVDA"), _row("NVDA"), _row(""), "garbage", _row("AMD", side="")]
    )
    assert roster["favorite"]["LONG"] == ["NVDA"]


# ---------------------------------------------------------------------------
# The roster on the swing push
# ---------------------------------------------------------------------------


def _pick(symbol, **kwargs):
    base = {
        "symbol": symbol,
        "side": "LONG",
        "bucket": "favorite_setup",
        "expected_r": 1.8,
        "key_level": "AVWAPE 101.25",
    }
    base.update(kwargs)
    return base


def test_the_swing_push_carries_the_whole_roster_not_just_the_ranked_picks():
    payload = {
        "swing_picks": [_pick("NVDA")],
        "swing_data_current": True,
        "bucket_roster": core.build_bucket_roster(
            [_row(f"SYM{index}") for index in range(12)]
            + [_row("TSLA", "high_conviction")]
        ),
    }
    _title, message = core.build_swing_push(payload)
    assert "1. NVDA LONG 1.8R" in message  # the ranked list is unchanged
    assert "HC L (1): TSLA" in message
    assert "FAV L (12): SYM0,SYM1" in message
    assert "SYM11" in message


def test_a_roster_with_no_qualifying_picks_still_pushes():
    """The membership answer is worth sending on its own."""
    payload = {
        "swing_picks": [],
        "swing_data_current": True,
        "bucket_roster": core.build_bucket_roster([_row("NVDA")]),
    }
    title, message = core.build_swing_push(payload)
    assert title == "Favorites / high conviction"
    assert "FAV L (1): NVDA" in message
    assert title.encode("ascii", "replace").decode("ascii") == title


def test_nothing_at_all_still_sends_nothing():
    assert core.build_swing_push({"swing_picks": [], "bucket_roster": {}}) is None
    assert core.build_swing_push({}) is None


def test_an_oversized_push_says_it_was_trimmed():
    """A silently shortened list reads as a complete one - the one failure
    this channel cannot have (plan.md sec 5)."""
    rows = [_row(f"SYMBOL{index}") for index in range(400)]
    payload = {
        "swing_picks": [_pick("NVDA")],
        "swing_data_current": True,
        "bucket_roster": core.build_bucket_roster(rows),
    }
    _title, message = core.build_swing_push(payload)
    assert len(message) <= core.PUSH_MESSAGE_MAX_CHARS
    assert "did not fit" in message


def test_a_message_that_fits_is_never_marked_as_trimmed():
    message = core.fit_push_message(["one", "two", "three"])
    assert message == "one\ntwo\nthree"


# ---------------------------------------------------------------------------
# The D1 events push
# ---------------------------------------------------------------------------


def _event(symbol, label, time_text="10:31:00"):
    return {"symbol": symbol, "label": label, "time_text": time_text}


def test_nothing_fired_stays_silent():
    assert core.build_d1_events_push([]) is None
    assert core.build_d1_events_push(["garbage", {}]) is None


def test_one_line_per_symbol_with_every_label_it_fired():
    built = core.build_d1_events_push(
        [_event("NVDA", "5d high"), _event("NVDA", "AVWAPE bounce", "11:02:00"),
         _event("AMD", "D1 break above", "10:45:00")]
    )
    title, message = built
    assert title == "D1 events (2)"
    assert "1. NVDA 5d high, AVWAPE bounce (11:02)" in message
    assert "2. AMD D1 break above (10:45)" in message
    assert "TV: NVDA,AMD" in message


def test_a_repeated_label_is_listed_once():
    _title, message = core.build_d1_events_push(
        [_event("NVDA", "5d high"), _event("NVDA", "5d high")]
    )
    assert message.count("5d high") == 1


def test_the_symbol_count_is_honest_when_the_list_is_capped():
    events = [_event(f"SYM{index}", "5d high") for index in range(40)]
    title, message = core.build_d1_events_push(events, limit=25)
    assert title == "D1 events (40)"  # the count never lies about the total
    numbered = [line for line in message.splitlines() if line[:1].isdigit()]
    assert len(numbered) == 25
    assert "15 more symbol(s) not shown" in message


def test_an_event_without_a_label_still_names_the_symbol():
    _title, message = core.build_d1_events_push([_event("NVDA", "")])
    assert "1. NVDA (10:31)" in message


def test_the_title_survives_the_ascii_only_push_header():
    title, _message = core.build_d1_events_push([_event("NVDA", "5d high")])
    assert title.encode("ascii", "replace").decode("ascii") == title
