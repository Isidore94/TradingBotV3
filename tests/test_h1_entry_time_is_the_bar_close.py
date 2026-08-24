"""R10.B / audit D6a - an H1 mark's `entry_time` was the bar START.

**6,439 of 6,439** H1 registered rows carry an `entry_time` whose minute is 30,
because `_emit_h1_color_alert` stamps the outcome row with `signal_bar.dt` -
the hour bar's opening timestamp. Non-H1 rows spread across minutes (55, 40,
35, 0), so this is not a coincidence of the schedule.

Why it matters more than a label: those rows are **82% of every registered row
in the store** (D6b), and an entry stamped an hour before the signal existed
makes every entry-timing statistic over them measure the wrong instant. The
row claims the desk could have acted at 06:30 on information that did not exist
until 07:30.

Two halves, and only one of them is a code change:

* **Forward**: the outcome row's `entry_time` is the bar CLOSE, which is the
  first moment the signal was knowable.
* **Backward**: the 6,439 existing rows are NOT rewritten. History is never
  rewritten (R10 ground rule 5); `evidence_rules.h1_bar_start_v1` already tags
  them for every reader.

The alert row keeps the bar's own start in its text, because "on the 06:30 H1
candle" is a correct description of which candle fired.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bounce_bot_lib import legacy  # noqa: E402
import evidence_rules  # noqa: E402

BAR_START = datetime(2026, 8, 21, 6, 30)


@dataclass
class StubBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float


def _bot():
    bot = legacy.BounceBot.__new__(legacy.BounceBot)
    bot.market_environment_lock = threading.Lock()
    bot.registered = []
    bot.logged_rows = []
    bot._log_bounce_candidate_event = lambda *a, **k: (
        bot.logged_rows.append({"bar": a[4], "reason": k.get("reason", "")})
        or {"event_id": "evt-h1", "symbol": a[1], "direction": a[2]}
    )
    bot._register_bounce_outcome = lambda *a, **k: bot.registered.append(a)
    bot._evaluate_bounce_alert_quality = lambda *a, **k: {"tier": "B"}
    bot.record_alert_tier = lambda *a, **k: None
    bot._measured_exit_suffix = lambda *a, **k: ""
    bot.log_symbol = lambda *a, **k: None
    bot.log_bounce_to_file = lambda **k: None
    bot.gui_callback = None
    return bot


def _hit(side="long"):
    return {
        "symbol": "AAA",
        "side": side,
        "type": legacy.H1_BLUE_AFTER_RED_TYPE,
        "level": 100.0,
        "signal_bar": StubBar(BAR_START, 99.0, 101.0, 98.5, 100.5),
        "color": "blue",
        "prev_color": "red",
        "detail": "blue reclaim candle after a red H1",
    }


def test_the_outcome_row_is_stamped_at_the_bar_close_not_the_bar_start():
    """Fail-before-fix. The signal is not knowable until the hour bar closes."""
    bot = _bot()

    bot._emit_h1_color_alert(_hit())

    assert bot.registered, "the H1 mark still records its evidence row"
    current_candle = bot.registered[0][4]
    stamped = datetime.strptime(current_candle["time"], "%Y%m%d  %H:%M:%S")
    assert stamped == BAR_START + timedelta(hours=1)
    assert stamped.minute == 30 or stamped.minute == 0  # whichever the close lands on


def test_v1_cannot_see_the_fix_which_is_why_v2_exists():
    """An H1 bar in PT starts at :30 and therefore CLOSES at :30.

    So `h1_bar_start_v1`'s family-AND-minute heuristic reports `mixed` on a
    correctly stamped forward row - a false positive on every row written from
    now on, which is worse than the defect the rule was written to describe.
    That is not a bug in v1; it is the limit of the only evidence the store
    carried when v1 was written. Rules are never edited in place (ground rule
    5), so the answer is a new NAME, not a new definition under the old one.
    """
    bot = _bot()
    bot._emit_h1_color_alert(_hit())
    stamp = bot.registered[0][4]["time"]
    iso = datetime.strptime(stamp, "%Y%m%d  %H:%M:%S").isoformat()

    assert evidence_rules.h1_bar_start_v1(
        legacy.H1_BLUE_AFTER_RED_TYPE, iso
    ).verdict == evidence_rules.VERDICT_MIXED


def test_v2_reads_the_recorded_basis_and_calls_the_new_stamp_clean():
    """The forward row records where its stamp came from, so v2 does not have
    to guess - and a row that records nothing still falls back to v1."""
    clean = evidence_rules.h1_bar_start_v2(
        legacy.H1_BLUE_AFTER_RED_TYPE,
        "2026-08-21T07:30:00",
        entry_time_basis=evidence_rules.BASIS_BAR_CLOSE,
    )
    assert clean.verdict == evidence_rules.VERDICT_SHARES
    assert "bar close" in clean.reason

    broken = evidence_rules.h1_bar_start_v2(
        legacy.H1_BLUE_AFTER_RED_TYPE,
        "2026-08-21T06:30:00",
        entry_time_basis=evidence_rules.BASIS_BAR_START,
    )
    assert broken.verdict == evidence_rules.VERDICT_MIXED

    legacy_row = evidence_rules.h1_bar_start_v2(
        legacy.H1_BLUE_AFTER_RED_TYPE, "2026-08-21T06:30:00"
    )
    assert legacy_row.verdict == evidence_rules.VERDICT_MIXED
    assert "fell back to h1_bar_start_v1" in legacy_row.reason


def test_the_forward_row_records_its_basis_and_its_claim_kind():
    """Both are recorded at registration rather than inferred by a later
    reader - which is the only way v2 can be exact instead of heuristic."""
    bot = _bot()
    bot.atr_cache = {}
    bot.get_market_environment = lambda: "neutral_chop"
    fields = legacy.BounceBot._registration_context_fields(
        bot,
        "AAA",
        "AAA_long_20260821_07_30_00_h1_blue_after_red",
        {"entry_price": 100.0, "risk_per_share": 1.0},
        BAR_START + timedelta(hours=1),
    )

    assert fields["entry_time_basis"] == evidence_rules.BASIS_BAR_CLOSE
    # An H1 mark is an annotation, not a trade - so it is never averaged as one.
    assert fields["claim_kind"] == "annotation"


def test_a_legacy_bar_start_stamp_is_still_tagged_not_rewritten():
    """History is never rewritten (ground rule 5). The old rows keep their
    stamp and the reader-side rule keeps naming what is wrong with them."""
    tag = evidence_rules.h1_bar_start_v1(
        legacy.H1_BLUE_AFTER_RED_TYPE, BAR_START.isoformat()
    )
    assert tag.verdict == evidence_rules.VERDICT_MIXED
    assert "bar start, not the signal time" in tag.reason


def test_the_alert_row_still_describes_the_candle_that_fired():
    """"on the 06:30 H1 candle" is a correct description of WHICH candle. Only
    the outcome row's entry instant moved."""
    bot = _bot()
    bot._emit_h1_color_alert(_hit())

    logged = bot.logged_rows[0]
    assert "06:30" in logged["reason"]
    assert logged["bar"]["close"] == 100.5


def test_the_stop_still_comes_from_the_signal_bars_own_range():
    """Moving the stamp must not move the trade. The bar handed to registration
    keeps the signal bar's OHLC; only its `time` differs."""
    bot = _bot()
    bot._emit_h1_color_alert(_hit())

    bounce_candle = bot.registered[0][3]
    assert bounce_candle["high"] == 101.0
    assert bounce_candle["low"] == 98.5
    assert bounce_candle["open"] == 99.0
    assert bounce_candle["close"] == 100.5
