"""The best Master AVWAP swings, delivered as a phone push.

The Away report already publishes hourly to Drive; this is the same picks
arriving in the notification itself so the trader does not have to open the
digest to see them.

Two behaviours matter more than the formatting: a stale pick must never read
as a current one, and "nothing qualified" must stay silent rather than push an
empty list seven times a session.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autopilot_core as core  # noqa: E402


def _payload(picks, *, current=True):
    return {"swing_picks": picks, "swing_data_current": current}


def _pick(symbol, **kwargs):
    base = {
        "symbol": symbol,
        "side": "LONG",
        "bucket": "favorite_setup",
        "expected_r": 1.8,
        "family": "avwap_reclaim",
        "key_level": "AVWAPE 101.25",
    }
    base.update(kwargs)
    return base


def test_picks_render_symbol_side_r_and_level():
    built = core.build_swing_push(_payload([_pick("NVDA")]))
    assert built is not None
    title, message = built
    assert title == "Best swings (1)"
    assert "1. NVDA LONG 1.8R @ AVWAPE 101.25" in message
    assert "TV: NVDA" in message


def test_no_picks_sends_nothing():
    """An hourly 'no setups' push trains the trader to ignore the channel."""
    assert core.build_swing_push(_payload([])) is None
    assert core.build_swing_push({}) is None
    assert core.build_swing_push({"swing_picks": None}) is None


def test_near_rows_are_dropped():
    """The report already caps them at -0.18R; a push has less room, not more."""
    built = core.build_swing_push(
        _payload([_pick("AAA", bucket="near_favorite"), _pick("BBB")])
    )
    assert built is not None
    _title, message = built
    assert "AAA" not in message
    assert "1. BBB" in message


def test_only_near_rows_sends_nothing():
    assert core.build_swing_push(_payload([_pick("AAA", bucket="near_favorite")])) is None


def test_stale_data_is_labelled_not_hidden():
    """plan.md sec 5: missing data is uncertainty, never confirmation."""
    built = core.build_swing_push(_payload([_pick("NVDA")], current=False))
    assert built is not None
    _title, message = built
    assert "not from the current session" in message


def test_current_data_carries_no_stale_warning():
    built = core.build_swing_push(_payload([_pick("NVDA")], current=True))
    assert "not from the current session" not in built[1]


def test_missing_currency_flag_is_treated_as_stale():
    """Absence of the flag is not proof the data is current."""
    built = core.build_swing_push({"swing_picks": [_pick("NVDA")]})
    assert "not from the current session" in built[1]


def test_the_push_is_capped_so_the_os_does_not_truncate_it():
    picks = [_pick(f"SYM{index}") for index in range(20)]
    built = core.build_swing_push(_payload(picks), limit=5)
    _title, message = built
    numbered = [line for line in message.splitlines() if line[:1].isdigit()]
    assert len(numbered) == 5
    assert built[0] == "Best swings (5)"


def test_a_missing_expected_r_or_level_still_renders():
    built = core.build_swing_push(
        _payload([_pick("NVDA", expected_r=None, key_level="")])
    )
    _title, message = built
    assert "1. NVDA LONG" in message
    assert "None" not in message
    assert "@" not in message


def test_a_junk_expected_r_never_breaks_the_push():
    built = core.build_swing_push(_payload([_pick("NVDA", expected_r="n/a")]))
    assert "1. NVDA LONG" in built[1]


def test_rows_without_a_symbol_are_skipped():
    built = core.build_swing_push(_payload([_pick(""), _pick("NVDA")]))
    assert built[0] == "Best swings (1)"
    assert "1. NVDA" in built[1]


def test_non_mapping_rows_are_ignored():
    built = core.build_swing_push(_payload(["garbage", _pick("NVDA")]))
    assert built[0] == "Best swings (1)"


def test_the_title_survives_the_ascii_only_push_header():
    """ntfy puts the title in an HTTP header, which is latin-1 at best."""
    built = core.build_swing_push(_payload([_pick("NVDA")]))
    title = built[0]
    assert title.encode("ascii", "replace").decode("ascii") == title
