"""TJ-13A item 4: the morning brief lists only names with session evidence.

Measured on the live file ``C:\\TradingBotData\\ai_morning_brief.txt``
(2026-09-19): 1,028 lines, header ``Analyzed 53 of 312. Membership-only 259.
Failed 0.`` - so 259 of the 312 names in the file say nothing except::

    membership only - no session evidence beyond membership in focus_longs

A membership-only name already costs no model call (``briefs.run_ticker_briefs``
answers it from the projection). What it still costs is the brief itself: 259 of
312 sections carry a sentence that says the symbol is on a list, which is the
one thing the reader of a watchlist already knows, and they push the 53 real
briefs past the file's 48 KB ceiling.

So the name leaves the BODY and stays in the HEADER - counted, never silently
dropped. Missing data is stated, never hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import briefs  # noqa: E402

SESSION = "2026-09-17"


def _membership_only(symbol, lists=("focus_longs",)):
    """A real manifest entry, built by the writer's own entry builder."""
    return briefs._manifest_entry(
        symbol=symbol,
        memberships=[{"list": name} for name in lists],
        session_date=SESSION,
        evidence_hash="deadbeef",
        resume_key="r-" + symbol,
        status=briefs.BRIEF_STATUS_MEMBERSHIP_ONLY,
        reason="no session evidence beyond membership in " + ", ".join(lists),
    )


def _briefed(symbol, statement):
    return briefs._manifest_entry(
        symbol=symbol,
        memberships=[{"list": "focus_longs"}],
        session_date=SESSION,
        evidence_hash="cafebabe",
        resume_key="r-" + symbol,
        status=briefs.BRIEF_STATUS_BRIEFED,
        result={
            "model": "gemma3:12b",
            "summary": {
                "executive_summary": f"{symbol} held its level into the close.",
                "what_is_working": [
                    {
                        "statement": statement,
                        "evidence_refs": ["daily.auto_report"],
                        "confidence": "medium",
                    }
                ],
            },
        },
    )


def test_a_membership_only_name_is_counted_in_the_header_and_absent_from_the_body():
    """The live proportions, in miniature: 2 real briefs, 5 membership-only."""
    entries = [
        _briefed("NVDA", "NVDA reclaimed the anchored band."),
        _membership_only("TAK"),
        _membership_only("ABCL"),
        _membership_only("ERAS"),
        _briefed("AMD", "AMD held above yesterday's high."),
        _membership_only("CRBG"),
        _membership_only("TH"),
    ]

    text = briefs.render_morning_file(SESSION, entries, total=len(entries))

    # counted, in the header, before any brief
    assert "Membership-only 5." in text
    assert "Analyzed 2 of 7." in text
    # and absent from the body
    assert briefs.MEMBERSHIP_ONLY_PREFIX not in text
    for symbol in ("TAK", "ABCL", "ERAS", "CRBG", "TH"):
        assert f"## {symbol}" not in text, f"{symbol} has no session evidence to print"
    # the names that DO have evidence are still there, whole
    assert "## NVDA" in text
    assert "## AMD" in text
    assert "NVDA reclaimed the anchored band." in text
    assert "AMD held above yesterday's high." in text


def test_a_brief_of_nothing_but_membership_names_still_publishes_its_counts():
    """The honest empty night: no body, and the header says why.

    Publishing an empty file with no explanation would read as "the model found
    nothing worth saying", which is a different night from "no name carried any
    session evidence at all".
    """
    entries = [_membership_only(f"SYM{index}") for index in range(12)]

    text = briefs.render_morning_file(SESSION, entries, total=12)

    assert "Analyzed 0 of 12." in text
    assert "Membership-only 12." in text
    assert briefs.MEMBERSHIP_ONLY_PREFIX not in text
    assert "## SYM0" not in text


def test_dropping_the_membership_names_leaves_room_for_the_real_briefs():
    """Why the change is worth making: the file has a 48 KB ceiling.

    259 membership-only sections ahead of the briefs is how a real one gets
    "omitted from this small summary file". With them out of the body, none of
    the analysed names is omitted.
    """
    # The live 2026-09-19 shape: 312 names, 259 of them membership-only. The
    # analysed sections carry real prose, so the two together overrun the
    # ceiling and the tail of the file is dropped.
    prose = "It reclaimed the anchored band and held it into the close. " * 6
    entries: list[dict] = []
    for index in range(259):
        entries.append(_membership_only(f"NOI{index}", lists=("swinglongs", "longs")))
    for index in range(52):
        entries.append(_briefed(f"REAL{index}", prose))
    entries.append(_briefed("LAST", "LAST is the brief the ceiling used to eat."))

    text = briefs.render_morning_file(SESSION, entries, total=len(entries))

    assert "omitted from this small summary file" not in text
    assert "## LAST" in text
    assert "LAST is the brief the ceiling used to eat." in text


def test_a_membership_only_name_still_costs_no_model_call():
    """Guard on what is already true (TB-2), so item 4 cannot regress it."""
    evidence = {"sources": [{"source_id": briefs.MEMBERSHIP_SOURCE_ID}]}
    assert briefs.is_membership_only(evidence) is True

    with_evidence = {
        "sources": [
            {"source_id": briefs.MEMBERSHIP_SOURCE_ID},
            {"source_id": "daily.auto_report"},
        ]
    }
    assert briefs.is_membership_only(with_evidence) is False
