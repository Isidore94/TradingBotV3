"""Econ times from the trader's pasted brief - a fixed parser, never a model.

Golden input: the brief the trader pasted on 2026-09-24,
``tests/fixtures/day_review/forecast_2026-09-24.md``. A misread time is a
wrong alarm, so every event, date and clock time below is pinned by hand from
that file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day_review"
FIXTURE_0924 = FIXTURES / "forecast_2026-09-24.md"
FIXTURE_0917 = FIXTURES / "forecast_2026-09-17.md"


def _text(path: Path = FIXTURE_0924) -> str:
    return path.read_text(encoding="utf-8")


def _events(text: str | None = None):
    import econ_events

    return econ_events.parse_events(_text() if text is None else text)


def _find(events, date: str, needle: str):
    return [
        event
        for event in events
        if event.date == date and needle.casefold() in event.label.casefold()
    ]


def test_the_golden_fixture_is_the_brief_the_trader_pasted():
    text = _text()
    assert text.startswith("# Market Morning Brief")
    assert "Thursday, September 24, 2026" in text
    assert "**As of ~8:50 a.m. ET**" in text


def test_brief_date_and_as_of_come_from_the_document():
    import econ_events

    parsed = econ_events.parse(_text())
    assert parsed.brief_date == "2026-09-24"
    assert parsed.as_of == "08:50"


def test_the_golden_events_have_the_right_dates_and_times():
    events = _events()
    expected = [
        ("2026-09-24", "10:00", "new-home sales"),
        ("2026-09-24", "13:00", "7-year Treasury auction"),
        ("2026-09-25", "08:30", "durable goods"),
        ("2026-09-25", "10:00", "Michigan"),
        ("2026-09-29", "10:00", "JOLTS"),
        ("2026-09-30", "", "ADP"),
        ("2026-09-30", "", "PCE"),
        ("2026-09-30", "", "GDP"),
        ("2026-10-02", "08:30", "employment report"),
        ("2026-09-24", "", "Costco"),
    ]
    for date, time_et, needle in expected:
        found = _find(events, date, needle)
        assert len(found) == 1, (date, needle, [e.label for e in events])
        assert found[0].time_et == time_et, (needle, found[0])


def test_costco_is_after_the_close_and_carries_no_clock_time():
    (costco,) = _find(_events(), "2026-09-24", "Costco")
    assert costco.time_et == ""
    assert "after the close" in costco.label


def test_the_same_event_said_twice_is_one_event_with_its_time():
    events = _events()
    assert len([e for e in events if "jolts" in e.label.casefold()]) == 1
    auctions = [e for e in events if "auction" in e.label.casefold()]
    assert len(auctions) == 1, auctions
    assert auctions[0].time_et == "13:00"
    assert len([e for e in events if "michigan" in e.label.casefold()]) == 1


def test_past_prints_and_undated_mentions_are_not_events():
    events = _events()
    labels = " | ".join(e.label.casefold() for e in events)
    # "## Fresh 8:30 data" and "claims just printed" are a past print.
    assert "claims" not in labels
    # "Yesterday's flash U.S. composite PMI" is yesterday.
    assert "pmi" not in labels
    # "The most recent August CPI" has no date at all.
    assert "cpi" not in labels
    # Nothing lands before the brief's own date.
    assert all(e.date >= "2026-09-24" for e in events)


def test_every_event_keeps_the_line_it_came_from():
    for event in _events():
        assert event.source_line.strip(), event
        assert "**" not in event.source_line


def test_events_come_out_in_time_order():
    events = _events()
    keys = [(e.date, e.time_et or "99:99") for e in events]
    assert keys == sorted(keys)


def test_a_bare_time_inherits_the_meridiem_of_the_earlier_time_in_its_sentence():
    events = _events(
        "# Brief - Monday, October 5, 2026\n\n"
        "Tomorrow brings ISM services at 9:45 a.m. ET and factory orders at 10:00.\n"
    )
    (ism,) = _find(events, "2026-10-06", "ISM")
    assert ism.time_et == "09:45"
    (orders,) = _find(events, "2026-10-06", "factory orders")
    assert orders.time_et == "10:00"


def test_a_bare_time_with_no_earlier_meridiem_is_unknown():
    events = _events(
        "# Brief - Monday, October 5, 2026\n\nTomorrow brings factory orders at 10:00.\n"
    )
    (orders,) = _find(events, "2026-10-06", "factory orders")
    assert orders.time_et == ""


def test_an_inherited_time_that_would_run_backwards_is_unknown():
    events = _events(
        "# Brief - Monday, October 5, 2026\n\n"
        "Tomorrow brings ISM services at 11:30 a.m. ET and factory orders at 1:00.\n"
    )
    (orders,) = _find(events, "2026-10-06", "factory orders")
    assert orders.time_et == ""


def test_tomorrow_on_a_friday_is_the_next_trading_day():
    events = _events(
        "# Brief - Friday, October 2, 2026\n\nTomorrow brings the ISM report at 10 a.m. ET.\n"
    )
    (ism,) = _find(events, "2026-10-05", "ISM")
    assert ism.time_et == "10:00"


def test_a_january_date_in_a_december_brief_rolls_to_next_year():
    events = _events(
        "# Brief - Tuesday, December 29, 2026\n\n**Friday, Jan. 8:** December jobs report.\n"
    )
    assert _find(events, "2027-01-08", "employment report")


def test_a_time_on_the_brief_date_at_or_before_as_of_is_past():
    events = _events(
        "# Brief - Monday, October 5, 2026\n**As of ~10:15 a.m. ET**\n\n"
        "Today brings ISM services at 10 a.m. ET and a 3-year Treasury auction at 1 p.m. ET.\n"
    )
    assert not _find(events, "2026-10-05", "ISM")
    assert _find(events, "2026-10-05", "auction")[0].time_et == "13:00"


def test_a_time_in_another_zone_is_unknown_rather_than_guessed():
    events = _events(
        "# Brief - Monday, October 5, 2026\n\nTomorrow brings ISM services at 9:45 a.m. CT.\n"
    )
    (ism,) = _find(events, "2026-10-06", "ISM")
    assert ism.time_et == ""


def test_the_older_golden_parses_without_error():
    import econ_events

    parsed = econ_events.parse(_text(FIXTURE_0917))
    assert parsed.brief_date == "2026-09-17"
    assert all(e.date >= "2026-09-17" for e in parsed.events)


def test_an_empty_text_has_no_events():
    import econ_events

    parsed = econ_events.parse("")
    assert parsed.brief_date == ""
    assert parsed.events == ()


def test_the_parser_opens_no_socket(monkeypatch):
    import socket

    import econ_events

    def _refuse(*_args, **_kwargs):  # pragma: no cover - the point is it is not hit
        raise AssertionError("the econ parser is deterministic; it fetches nothing")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    assert econ_events.parse(_text()).events


# ---------------------------------------------------------------------------
# forecast_brief side fix: either word for either side of the playbook
# ---------------------------------------------------------------------------
def test_todays_playbook_reads_bullish_reversal_and_bearish_continuation():
    import forecast_brief

    brief = forecast_brief.parse(_text())
    assert brief.playbook_bullish.startswith("**Bullish reversal:**")
    assert "NVDA/SMH reclaim VWAP" in brief.playbook_bullish
    assert brief.playbook_bearish.startswith("**Bearish continuation:**")
    assert "7-year auction is weak" in brief.playbook_bearish


# ---------------------------------------------------------------------------
# the machine-readable NEXT 7 DAYS block (trader, 2026-09-24)
# ---------------------------------------------------------------------------
FIXTURE_CALENDAR = FIXTURES / "forecast_2026-09-24_calendar.md"


def _calendar():
    import econ_events

    return econ_events.parse(_text(FIXTURE_CALENDAR))


def _row(parsed, date: str, needle: str):
    found = [
        e for e in parsed.events if e.date == date and needle.casefold() in e.label.casefold()
    ]
    assert len(found) == 1, (date, needle, [e.label for e in parsed.events])
    return found[0]


def test_the_calendar_block_is_the_only_source_when_present():
    parsed = _calendar()
    assert parsed.source == "calendar"
    # Costco and the prose-only employment report are not in the block.
    labels = " | ".join(e.label for e in parsed.events)
    assert "Costco" not in labels
    assert not [e for e in parsed.events if e.date == "2026-10-02"]
    # Labels are the block's own words.
    assert _row(parsed, "2026-09-25", "Durable goods orders (Aug)").time_et == "08:30"


def test_calendar_rows_use_the_et_time():
    parsed = _calendar()
    assert _row(parsed, "2026-09-24", "New home sales").time_et == "10:00"
    assert _row(parsed, "2026-09-24", "Treasury note auction").time_et == "13:00"
    assert _row(parsed, "2026-09-25", "Michigan").time_et == "10:00"
    assert _row(parsed, "2026-09-29", "JOLTS").time_et == "10:00"
    assert _row(parsed, "2026-09-28", "Early test release").time_et == "06:00"


def test_time_tbd_is_listed_with_no_time():
    assert _row(_calendar(), "2026-09-30", "ADP").time_et == ""


def test_a_pt_et_mismatch_keeps_the_event_with_no_time():
    # 05:30 PT + 3 h is 08:30, not the 09:30 ET the line says.
    assert _row(_calendar(), "2026-09-30", "PCE").time_et == ""


def test_a_malformed_line_is_skipped_whole_and_counted():
    parsed = _calendar()
    assert parsed.unread_lines == 1
    assert not [e for e in parsed.events if "ISM" in e.label]


def test_a_calendar_row_at_or_before_as_of_on_the_brief_date_is_past():
    parsed = _calendar()
    assert not [e for e in parsed.events if "claims" in e.label.casefold()]


def test_alarms_are_only_for_timed_events_at_or_after_seven_et():
    import econ_events

    parsed = _calendar()
    assert [e.label for e in econ_events.alarm_events(parsed.events, day="2026-09-28")] == []
    today = econ_events.alarm_events(parsed.events, day="2026-09-24")
    assert [e.time_et for e in today] == ["10:00", "13:00"]
    assert econ_events.alarm_events(parsed.events, day="2026-09-30") == ()


def test_the_calendar_heading_matches_any_level_bold_or_hyphen():
    import econ_events

    for heading in (
        "NEXT 7 DAYS — ECONOMIC CALENDAR",
        "### Next 7 days - economic calendar",
        "**NEXT 7 DAYS – ECONOMIC CALENDAR**",
    ):
        text = (
            "# Brief - Monday, October 5, 2026\n\nProse says ISM at 9 a.m. ET tomorrow.\n\n"
            f"{heading}\n2026-10-06 | 07:00 PT / 10:00 ET | ISM services\n"
        )
        parsed = econ_events.parse(text)
        assert parsed.source == "calendar", heading
        assert [(e.date, e.time_et, e.label) for e in parsed.events] == [
            ("2026-10-06", "10:00", "ISM services")
        ], heading


def test_a_brief_without_the_block_falls_back_to_prose():
    import econ_events

    parsed = econ_events.parse(_text())
    assert parsed.source == "prose"
    assert parsed.unread_lines == 0
