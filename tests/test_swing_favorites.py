"""The trader's hand-vetted swing picks - store, replay, and the taken join.

Trader, 2026-08-31: *"at the end of the day I have a list of my top swing
targets. I want a place to put them in so the bot knows my personal favourite
picks... the bot should scan the journal to know which ones I actually took."*

What these defend: the store is append-only and a removal is a RETRACTION row
rather than an edit; the live list is a replay of one session in file order;
the store is addressed by its `project_paths` named constant; and the "taken"
mark is a display-only join that says nothing when it cannot measure.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import swing_favorites  # noqa: E402


SESSION = "2026-08-31"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestTheStorePath:
    def test_the_store_is_addressed_by_its_named_constant(self):
        """Resolving by name under the wrong root shipped a blank page for six
        days (CLAUDE.md). Every reader here goes through the constant."""
        import project_paths

        assert swing_favorites.SWING_FAVORITES_FILE is project_paths.SWING_FAVORITES_FILE
        assert project_paths.SWING_FAVORITES_FILE.name == "swing_favorites.jsonl"
        assert (
            project_paths.SWING_FAVORITES_FILE.parent == project_paths.PERSISTENT_DATA_DIR
        ), "shared home folder, same storage class as the other trader logs"


class TestAddingAndRetracting:
    def test_one_add_writes_one_row_with_a_tz_aware_time(self, tmp_path):
        path = tmp_path / "swing_favorites.jsonl"
        row = swing_favorites.record_favorite(
            symbol="nvda", side="Long", session_date=SESSION, path=path
        )
        assert row is not None
        stored = _rows(path)
        assert len(stored) == 1
        assert stored[0]["symbol"] == "NVDA"
        assert stored[0]["side"] == "long"
        assert stored[0]["action"] == "add"
        assert stored[0]["session_date"] == SESSION
        assert stored[0]["origin"] == "trader"
        assert stored[0]["schema"] == swing_favorites.SCHEMA_SWING_FAVORITE
        assert datetime.fromisoformat(stored[0]["event_at"]).tzinfo is not None

    def test_a_removal_appends_and_never_rewrites(self, tmp_path):
        """The record of "added AMD then thought better of it" must survive."""
        path = tmp_path / "swing_favorites.jsonl"
        swing_favorites.record_favorite(symbol="AMD", side="long", session_date=SESSION, path=path)
        first = _rows(path)[0]
        swing_favorites.record_favorite(
            symbol="AMD", side="long", action="remove", session_date=SESSION, path=path
        )
        stored = _rows(path)
        assert len(stored) == 2
        assert stored[0] == first, "the original row is untouched"
        assert stored[1]["action"] == "remove"
        assert swing_favorites.favorites_for_session(SESSION, path=path) == []

    def test_a_failed_append_loses_the_event_and_never_raises(self, tmp_path):
        """An evidence store is never allowed to cost the thing it records."""
        unwritable = tmp_path / "blocked"
        unwritable.write_text("not a directory", encoding="utf-8")
        assert (
            swing_favorites.record_favorite(
                symbol="NVDA", side="long", session_date=SESSION,
                path=unwritable / "swing_favorites.jsonl",
            )
            is None
        )

    def test_a_blank_or_sideless_pick_writes_nothing(self, tmp_path):
        path = tmp_path / "swing_favorites.jsonl"
        assert swing_favorites.record_favorite(symbol="", side="long", path=path) is None
        assert swing_favorites.record_favorite(symbol="NVDA", side="", path=path) is None
        assert not path.exists()


class TestTheSessionList:
    def test_the_list_is_a_replay_in_the_order_the_trader_typed(self, tmp_path):
        path = tmp_path / "swing_favorites.jsonl"
        for symbol in ("NVDA", "AMD", "TSLA"):
            swing_favorites.record_favorite(
                symbol=symbol, side="long", session_date=SESSION, path=path
            )
        live = swing_favorites.favorites_for_session(SESSION, path=path)
        assert [row["symbol"] for row in live] == ["NVDA", "AMD", "TSLA"]

    def test_a_re_add_returns_to_the_end_where_the_trader_just_put_it(self, tmp_path):
        path = tmp_path / "swing_favorites.jsonl"
        for symbol in ("NVDA", "AMD"):
            swing_favorites.record_favorite(
                symbol=symbol, side="long", session_date=SESSION, path=path
            )
        swing_favorites.record_favorite(
            symbol="NVDA", side="long", action="remove", session_date=SESSION, path=path
        )
        swing_favorites.record_favorite(
            symbol="NVDA", side="long", session_date=SESSION, path=path
        )
        live = swing_favorites.favorites_for_session(SESSION, path=path)
        assert [row["symbol"] for row in live] == ["AMD", "NVDA"]

    def test_prior_sessions_stay_in_the_store_and_out_of_todays_list(self, tmp_path):
        path = tmp_path / "swing_favorites.jsonl"
        swing_favorites.record_favorite(
            symbol="OLD", side="long", session_date="2026-08-28", path=path
        )
        swing_favorites.record_favorite(
            symbol="NEW", side="short", session_date=SESSION, path=path
        )
        assert [row["symbol"] for row in swing_favorites.favorites_for_session(SESSION, path=path)] == ["NEW"]
        assert len(_rows(path)) == 2, "yesterday's row is untouched"

    def test_the_two_sides_are_separate_picks(self, tmp_path):
        path = tmp_path / "swing_favorites.jsonl"
        swing_favorites.record_favorite(symbol="NVDA", side="long", session_date=SESSION, path=path)
        swing_favorites.record_favorite(symbol="NVDA", side="short", session_date=SESSION, path=path)
        swing_favorites.record_favorite(
            symbol="NVDA", side="long", action="remove", session_date=SESSION, path=path
        )
        live = swing_favorites.favorites_for_session(SESSION, path=path)
        assert [(row["symbol"], row["side"]) for row in live] == [("NVDA", "short")]

    def test_a_torn_line_is_skipped_not_raised(self, tmp_path):
        path = tmp_path / "swing_favorites.jsonl"
        swing_favorites.record_favorite(symbol="NVDA", side="long", session_date=SESSION, path=path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema": "swing_favor')
        assert [row["symbol"] for row in swing_favorites.favorites_for_session(SESSION, path=path)] == ["NVDA"]

    def test_a_missing_store_is_an_empty_list(self, tmp_path):
        assert swing_favorites.favorites_for_session(SESSION, path=tmp_path / "nope.jsonl") == []


class TestPastedInput:
    def test_a_paste_yields_order_preserving_unique_symbols(self):
        assert swing_favorites.parse_symbols("nvda\namd, NVDA  tsla") == ["NVDA", "AMD", "TSLA"]

    def test_blank_input_yields_nothing(self):
        assert swing_favorites.parse_symbols("   ") == []


class TestTheTakenMark:
    def _favorite(self, symbol, side="long", session_date=SESSION):
        return {"symbol": symbol, "side": side, "session_date": session_date}

    def test_a_trade_opened_the_same_day_marks_the_pick(self):
        marks = swing_favorites.taken_keys(
            [self._favorite("NVDA")],
            [{"symbol": "NVDA", "opened_at": f"{SESSION}T09:41:00-04:00"}],
        )
        assert marks == {("NVDA", "long")}

    def test_a_trade_opened_after_the_pick_still_marks_it(self):
        marks = swing_favorites.taken_keys(
            [self._favorite("NVDA")],
            [{"symbol": "NVDA", "opened_at": "2026-09-02T09:41:00-04:00"}],
        )
        assert marks == {("NVDA", "long")}

    def test_a_trade_before_the_pick_marks_nothing(self):
        """Yesterday's trade is not evidence that today's pick was taken."""
        marks = swing_favorites.taken_keys(
            [self._favorite("NVDA")],
            [{"symbol": "NVDA", "opened_at": "2026-08-28T09:41:00-04:00"}],
        )
        assert marks == set()

    def test_an_unmeasurable_trade_shows_nothing(self):
        """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
        marks = swing_favorites.taken_keys(
            [self._favorite("NVDA")],
            [{"symbol": "NVDA", "opened_at": "", "trade_date": ""}],
        )
        assert marks == set()

    def test_an_empty_journal_marks_nothing(self):
        assert swing_favorites.taken_keys([self._favorite("NVDA")], []) == set()

    def test_only_the_matching_symbol_is_marked(self):
        marks = swing_favorites.taken_keys(
            [self._favorite("NVDA"), self._favorite("AMD", side="short")],
            [{"symbol": "AMD", "trade_date": SESSION}],
        )
        assert marks == {("AMD", "short")}

    def test_the_lookback_is_bounded(self):
        """An unbounded query grows without limit against a year of fills."""
        start = swing_favorites.taken_lookback_start(SESSION, days=10)
        assert start.isoformat() == "2026-08-21"

    def test_the_lookback_default_is_ten_days(self):
        assert swing_favorites.TAKEN_LOOKBACK_DAYS == 10


class TestTheSessionStamp:
    def test_a_naive_moment_gains_an_offset_rather_than_losing_one(self):
        """`astimezone` on a naive datetime attaches; it never strips."""
        row = swing_favorites.build_row(
            symbol="NVDA", side="long", now=datetime(2026, 8, 31, 16, 5, 0)
        )
        assert datetime.fromisoformat(row["event_at"]).utcoffset() is not None

    def test_an_aware_moment_keeps_its_own_offset(self):
        moment = datetime(2026, 8, 31, 20, 5, 0, tzinfo=timezone(timedelta(hours=-4)))
        row = swing_favorites.build_row(symbol="NVDA", side="long", now=moment)
        assert datetime.fromisoformat(row["event_at"]).utcoffset() == timedelta(hours=-4)


@pytest.mark.parametrize(
    "text,expected",
    [("long", "long"), ("LONGS", "long"), ("buy", "long"), ("short", "short"),
     ("Shorts", "short"), ("sell", "short"), ("", ""), ("sideways", "")],
)
def test_side_spellings(text, expected):
    assert swing_favorites.normalize_side(text) == expected
