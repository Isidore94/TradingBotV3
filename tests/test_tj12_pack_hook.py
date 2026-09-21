"""TJ-12 fills TJ-4's `report_card` hook, and the pack's hash still behaves.

`plan.md` TJ-4 change 1, AMENDED: *"The pack also gains `internals` ..., `skill`
... and `report_card` (TJ-12), each with `source_id`s."* The hook is already
there and empty (`scripts/day_review_pack.py:466-468`,
`tests/test_tj4_day_pack.py:106`).

Two properties the hook must not break (`day_review_pack.build_pack`'s own
docstring): `inputs_hash` is over the SECTIONS and never over the clock, and
`built_at` is added AFTER the hash. So the same session built by the post-close
tick and again by the nightly slot hours apart still skips the model call - and
a card line that MOVED still moves the hash, because a narration written about
yesterday's card is not a narration about today's.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tj12_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
EVENING = datetime(2026, 9, 18, 13, 20, tzinfo=PACIFIC)


def _card(tmp_path):
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    return day_report_card.build(fx.day_inputs(tmp_path))


def _pack(tmp_path, *, card=None, now=EVENING):
    import day_review_pack

    return day_review_pack.build_pack(
        fx.SESSION, report_card=card if card is not None else _card(tmp_path), now=now
    )


def test_the_hook_is_still_empty_when_nobody_hands_a_card_in():
    """TJ-4's contract does not change for a caller that has no card."""
    import day_review_pack

    pack = day_review_pack.build_pack(fx.SESSION, now=EVENING)
    assert pack["report_card"] == {}


def test_the_pack_carries_the_six_lines_in_order(tmp_path):
    import day_report_card

    section = _pack(tmp_path)["report_card"]
    assert tuple(line["key"] for line in section["lines"]) == tuple(
        day_report_card.LINE_KEYS
    )


def test_every_card_line_names_its_own_source_and_is_citable(tmp_path):
    """A narration may cite a card line, so a line needs a `source_id`."""
    import day_review_pack

    pack = _pack(tmp_path)
    ids = [line["source_id"] for line in pack["report_card"]["lines"]]
    assert all(ids), "a line with no source_id cannot be cited"
    assert len(set(ids)) == len(ids), "two lines shared a source_id"
    allowed = day_review_pack.allowed_source_ids(pack)
    for source_id in ids:
        assert source_id in allowed, source_id


def test_the_inputs_hash_still_ignores_the_clock(tmp_path):
    card = _card(tmp_path)
    first = _pack(tmp_path, card=card, now=EVENING)
    later = _pack(tmp_path, card=card, now=EVENING + timedelta(hours=9))

    assert first["inputs_hash"] == later["inputs_hash"]
    assert first["built_at"] != later["built_at"]


def test_the_inputs_hash_moves_when_a_card_line_moves(tmp_path):
    import day_report_card

    card = _card(tmp_path)
    before = _pack(tmp_path, card=card)

    lines = [dict(line) for line in card.lines]
    lines[0]["text"] = lines[0]["text"] + " and one more real miss"
    moved = day_report_card.ReportCard(session=card.session, lines=tuple(lines))
    after = _pack(tmp_path, card=moved)

    assert before["inputs_hash"] != after["inputs_hash"]


def test_a_pack_with_no_card_and_one_with_a_card_are_not_the_same_pack(tmp_path):
    import day_review_pack

    bare = day_review_pack.build_pack(fx.SESSION, now=EVENING)
    carded = _pack(tmp_path)
    assert bare["inputs_hash"] != carded["inputs_hash"]
