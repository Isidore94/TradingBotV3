"""P13 journal pack: today's (or a day's/week's) trades from the journal, read-only, ids per trade."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import journal_pack  # noqa: E402
from mentor_packs.journal_read import ET  # noqa: E402

NOW = journal_pack.FIXTURE_NOW


@pytest.fixture
def journal(tmp_path):
    return journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def test_today_lists_each_trade_with_r_or_dollars_hold_and_tax_class(journal):
    rows = _rows(journal_pack.build("today", now=NOW, journal=journal))
    nvda = rows["jrn:2026-09-30:W1"]["text"]
    assert "LONG NVDA (day) size 100" in nvda and "+1.00R (+100.00 $)" in nvda and "held 45 min" in nvda
    assert "opened Wed 09:35 ET" in nvda and "account M1 (margin)" in nvda
    amd = rows["jrn:2026-09-30:W2"]["text"]
    assert "-120.00 $ (R unknown: no planned stop)" in amd, "no stop: dollars, R unknown, never a guess"


def test_option_legs_opened_together_are_one_spread_in_a_tax_free_account(journal):
    rows = _rows(journal_pack.build("today", now=NOW, journal=journal))
    spread = rows["jrn:2026-09-30:W3+W4"]["text"]
    assert spread.startswith("SPREAD ALL (spread)") and "+50.00 $" in spread and "registered, tax-free" in spread


def test_totals_and_open_positions(journal):
    rows = _rows(journal_pack.build("today", now=NOW, journal=journal))
    totals = rows["jrn:2026-09-30:totals"]
    assert (totals["count"], totals["wins"], totals["net"], totals["open"]) == (3, 2, 30.0, 1)
    assert "largest loss AMD -120.00 $" in totals["text"] and "+1.00R over 1 of 3" in totals["text"]
    assert "Open LONG MSFT 10 @ 400.00, stop 395.00" in rows["jrn:2026-09-30:open:W5"]["text"]


def test_a_short_closed_lower_is_a_win_with_its_side_and_points(journal):
    """2026-10-01: RIOT short 18.83 -> 18.80 was called "a small loss" by the model."""
    import sqlite3

    conn = sqlite3.connect(journal)
    conn.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 ("R1", "M1", "Margin", "RIOT", "STK", "SHORT", "CLOSED", "2026-09-30T12:10:00-04:00",
                  "2026-09-30T12:40:00-04:00", 100, 100, 18.83, 18.80, None, None))
    conn.commit()
    conn.close()
    rows = _rows(journal_pack.build("today", now=NOW, journal=journal))
    riot = rows["jrn:2026-09-30:R1"]["text"]
    assert riot.startswith("SHORT RIOT")
    assert "WIN, short 18.83 -> 18.80 = +0.03 pts in your favour (short: exit below entry is a win)" in riot
    amd = rows["jrn:2026-09-30:W2"]["text"]
    assert "LOSS, short 150.00 -> 152.40 = -2.40 pts in your favour" in amd
    assert "WIN, long 120.00 -> 121.00 = +1.00 pts" in rows["jrn:2026-09-30:W1"]["text"]
    assert rows["jrn:2026-09-30:W3+W4"]["text"].count("WIN") == 1, "a spread's word comes from its $ alone"
    assert "Worst trade: SHORT AMD LOSS" in rows["jrn:2026-09-30:worst"]["text"]
    assert journal_pack.result_word(None, None) == "result unknown" and journal_pack.result_word(0.0) == "FLAT"


def test_the_prompt_says_a_short_exit_below_entry_is_a_win():
    from mentor_app.chat_model import SYSTEM_PROMPT

    assert "For a SHORT, an exit below the entry is a win" in SYSTEM_PROMPT


def test_yesterday_weekday_week_and_last_week(journal):
    assert "jrn:2026-09-29:Y1" in _rows(journal_pack.build("yesterday", now=NOW, journal=journal))
    assert "jrn:2026-09-29:Y1" in _rows(journal_pack.build("tuesday", now=NOW, journal=journal))
    assert "jrn:2026-09-29:Y1" in _rows(journal_pack.build("2026-09-29", now=NOW, journal=journal))
    week = _rows(journal_pack.build("week", now=NOW, journal=journal))
    assert {"jrn:wk2026-09-28:Y1", "jrn:wk2026-09-28:W1", "jrn:wk2026-09-28:totals"} <= set(week)
    assert week["jrn:wk2026-09-28:totals"]["count"] == 4
    last = _rows(journal_pack.build("last_week", now=NOW, journal=journal))
    assert last["jrn:wk2026-09-21:totals"]["count"] == 0 and "no trades" in last["jrn:wk2026-09-21:totals"]["text"]


def test_a_quiet_day_is_a_citable_none_and_a_bad_day_word_is_empty(journal):
    quiet = journal_pack.build("2026-09-28", now=NOW, journal=journal)
    assert quiet.ids[0] == "jrn:2026-09-28:totals" and "no trades" in quiet.rows[0]["text"]
    assert journal_pack.build("someday", now=NOW, journal=journal).ids == ()


def test_a_missing_journal_is_no_trades_not_a_raise(tmp_path):
    pack = journal_pack.build("today", now=NOW, journal=tmp_path / "none.sqlite3")
    assert "jrn:2026-09-30:totals" in pack.ids


def test_the_journal_is_opened_read_only(journal):
    before = journal.read_bytes()
    journal_pack.build("week", now=NOW, journal=journal)
    assert journal.read_bytes() == before
    with sqlite3.connect(journal) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 6


def test_today_summary_and_recent_symbols(journal):
    assert journal_pack.today_summary(journal, now=NOW) == "Today so far: 3 closed trade(s), 2 win(s), net +30.00 $, 1 open"
    assert journal_pack.recent_symbols(journal, now=NOW) == ["TSLA", "NVDA", "AMD", "ALL", "MSFT"]
    later = datetime(2026, 12, 30, 11, 0, tzinfo=ET)
    assert journal_pack.recent_symbols(journal, now=later) == [], "only the last 60 days"


def test_fixture_ids_are_unique():
    pack = journal_pack.fixture()
    assert pack.ids and len(pack.ids) == len(set(pack.ids))


def test_month_puts_its_totals_first_with_the_win_rate(journal):
    """P14: "what's my win rate this month" has a month window; its totals survive a tight attach budget."""
    pack = journal_pack.build("month", now=NOW, journal=journal)
    first = pack.rows[0]
    assert first["id"] == "jrn:mo2026-09:totals" and (first["count"], first["wins"]) == (4, 2)
    assert "2026-09-01 to 2026-09-30" in first["text"] and "win rate 50% (2 of 4 with a PnL)" in first["text"]
    assert {"jrn:mo2026-09:Y1", "jrn:mo2026-09:W1"} <= set(_rows(pack))
    assert journal_pack.resolve("last_month", NOW.date())[1:] == (datetime(2026, 8, 1).date(), datetime(2026, 8, 31).date())
    assert "win rate" not in _rows(journal_pack.build("week", now=NOW, journal=journal))["jrn:wk2026-09-28:totals"]["text"]


def test_a_day_puts_its_totals_first_so_a_wrong_premise_shows_at_once(journal):
    """15:27 retest: "why did I lose money tuesday" on a green day; the day's net must lead the pack."""
    for day, label in (("today", "2026-09-30"), ("tuesday", "2026-09-29"), ("yesterday", "2026-09-29")):
        assert journal_pack.build(day, now=NOW, journal=journal).rows[0]["id"] == f"jrn:{label}:totals", day
    week = journal_pack.build("week", now=NOW, journal=journal).rows
    assert week[0]["id"] != "jrn:wk2026-09-28:totals", "a week keeps its trades first"
