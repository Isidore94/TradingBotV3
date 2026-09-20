r"""TJ-4 item 3 - the rolling D1 view. RED before the build.

Packet `.claude/packets/TJ-4.md` item 3; `plan.md` §12.4 "TJ-4" change 3.
**NO MODEL IS CALLED HERE**: every test hands the module its own `request`.

The packet AMENDS the plan on one point and this file follows the packet: the
`open_theses` are built from **the trader's D1 PREDICTION clicks and D1 notes of
the last `LATELY_SESSIONS`** - *"the thesis store is empty and stays so"*. So
the rolling view reads the day packs already on disk, not `market_thesis`, and
an empty thesis store never makes the view empty.

The contract these tests pin (the builder may ADD keys, never remove one)::

    D1_VIEW_SCHEMA = D1_VIEW_PROMPT_VERSION = "d1_view_narration_v1"
    D1_VIEW_JSON_SCHEMA: dict                    # closed, like the day story's

    d1_view_path(*, root=None) -> Path           # <root>/d1_view.json
    read_d1_view(*, root=None) -> dict | None

`run_day_review_narration` writes BOTH artifacts, so its `outputs` list carries
the day story and the D1 view, and the two calls are told apart by their
`prompt_version`. The written file mirrors the day story's shape and its
`narration` is ``{"belief_now", "open_theses": [{"claim", "since",
"still_true", "evidence_id"}], "sources"}``.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj4_support as fx  # noqa: E402

SESSION = fx.SESSION


def _sessions_back(count: int) -> list[str]:
    """`count` exchange sessions ending at (and including) SESSION, oldest first."""
    import market_calendar

    cursor = date.fromisoformat(SESSION)
    out = [cursor.isoformat()]
    while len(out) < count:
        cursor = market_calendar.previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


@pytest.fixture
def rolling_root(tmp_path, monkeypatch):
    """Day packs across the window and one session OUTSIDE it.

    `evidence_stats.LATELY_SESSIONS` is 20, so the window is SESSION plus the 19
    exchange sessions before it. The 20th session back is one session too old
    and its D1 thesis may not be cited.
    """
    import day_review_pack
    import evidence_stats
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)

    window = _sessions_back(evidence_stats.LATELY_SESSIONS + 1)
    too_old, inside = window[0], window[1:]
    assert len(inside) == evidence_stats.LATELY_SESSIONS

    day_review_pack.write_pack(
        day_review_pack.build_pack(
            too_old,
            entries=[fx.d1_click_entry(too_old, direction="down", because="stale")],
            now=fx.AFTER_THE_CLOSE,
        ),
        root=root,
    )
    # The oldest session still inside the window carries the standing thesis.
    oldest = inside[0]
    day_review_pack.write_pack(
        day_review_pack.build_pack(
            oldest,
            entries=[
                fx.d1_click_entry(oldest, direction="up", because="higher lows"),
                fx.m5_note_entry(oldest, text="Chop until 11; I sat out."),
            ],
            now=fx.AFTER_THE_CLOSE,
        ),
        root=root,
    )
    # The session itself: TJ-4's full pack, plus a D1 note in words.
    entries = fx.pack_inputs()["entries"] + [
        fx.d1_note_entry(SESSION, text="I still think the index grinds higher into month end.")
    ]
    reads, _grades = fx.graded_reads(entries)
    day_review_pack.write_pack(
        fx.build(entries=entries, story=fx.daily_story(entries), reads=reads,
                 congruence=fx.congruence(reads)),
        root=root,
    )
    return {"root": root, "too_old": too_old, "oldest": oldest, "inside": inside}


def _d1_items(root, session):
    import day_review_pack

    pack = day_review_pack.read_pack(session, root=root)
    assert pack is not None, session
    return [
        item for item in pack["trader_said"]
        if str(item.get("timeframe") or "").upper() == "D1"
    ]


def _run(root, *, day_reply, d1_reply):
    """Run the slot with a fake that answers each prompt version in turn."""
    from ai_jobs import day_review_narration as module

    calls: list[dict] = []

    def request(**kwargs):
        calls.append(kwargs)
        if kwargs.get("prompt_version") == module.D1_VIEW_PROMPT_VERSION:
            if d1_reply is None:
                raise AssertionError("the D1 view was rebuilt when nothing moved")
            return d1_reply(**kwargs) if callable(d1_reply) else d1_reply
        return day_reply(**kwargs) if callable(day_reply) else day_reply

    outcome = module.run_day_review_narration(
        session_date=SESSION, now=fx.OVERNIGHT, root=root, request=request
    )
    return outcome, calls


def _day_reply(root):
    import day_review_pack

    pack = day_review_pack.read_pack(SESSION, root=root)
    clicked = fx.observing_and_predicting_entry()
    call = next(
        item for item in pack["trader_said"]
        if item.get("entry_id") == clicked["entry_id"] and item["kind"] == "prediction"
    )
    read = next(row for row in pack["reads"] if row.get("entry_id") == clicked["entry_id"])
    return {
        "model": "local-test-medium",
        "summary": {
            "headline": "You called the afternoon up and it went up.",
            "what_happened": "SPY closed +2.00%.",
            "what_you_thought": "You called it up at 07:02.",
            "were_you_right": [{
                "claim": "Rest of day: up", "source_id": call["source_id"],
                "verdict": read["verdict"], "evidence_id": read["source_id"],
            }],
            "chased_against_news": {"verdict": "unknown", "evidence_id": read["source_id"]},
            "process": "One call.",
            "sources": [call["source_id"], read["source_id"]],
        },
    }


def _d1_reply(evidence_id):
    return {
        "model": "local-test-medium",
        "summary": {
            "belief_now": "You have been leaning long the index since the gap held.",
            "open_theses": [{
                "claim": "The index grinds higher into month end",
                "since": fx.SESSION,
                "still_true": "unknown",
                "evidence_id": evidence_id,
            }],
            "sources": [evidence_id],
        },
    }


# ---------------------------------------------------------------------------
# what the model is allowed to see
# ---------------------------------------------------------------------------


def test_the_rolling_view_sees_d1_clicks_and_d1_notes_inside_the_window_only(
    rolling_root,
):
    """`LATELY_SESSIONS` (20) exchange sessions, and D1 only.

    An M5 note is a read about the rest of the day, not a belief about the
    bigger picture; a D1 thesis from 21 sessions ago is outside "lately". Both
    are excluded, and the window is walked on the exchange calendar.
    """
    from ai_jobs import day_review_narration as module

    root = rolling_root["root"]
    _outcome, calls = _run(
        root,
        day_reply=_day_reply(root),
        d1_reply=_d1_reply(_d1_items(root, SESSION)[0]["source_id"]),
    )

    d1_calls = [
        call for call in calls
        if call.get("prompt_version") == module.D1_VIEW_PROMPT_VERSION
    ]
    assert len(d1_calls) == 1, [call.get("prompt_version") for call in calls]
    allowed = set(d1_calls[0]["evidence"]["allowed_source_ids"])

    for item in _d1_items(root, rolling_root["oldest"]):
        assert item["source_id"] in allowed, item
    for item in _d1_items(root, SESSION):
        assert item["source_id"] in allowed, item
    for item in _d1_items(root, rolling_root["too_old"]):
        assert item["source_id"] not in allowed, (
            "a thesis from outside LATELY_SESSIONS was offered as current"
        )

    import day_review_pack

    oldest_pack = day_review_pack.read_pack(rolling_root["oldest"], root=root)
    m5_items = [
        item for item in oldest_pack["trader_said"]
        if str(item.get("timeframe") or "").upper() == "M5"
    ]
    assert m5_items, "fixture drift: the window needs an M5 row to exclude"
    for item in m5_items:
        assert item["source_id"] not in allowed, item


def test_an_empty_thesis_store_never_empties_the_rolling_view(rolling_root, monkeypatch):
    """Packet TJ-4 item 3: "the thesis store is empty and stays so".

    The live `market_thesis` store holds nothing and nothing here writes to it,
    so the view is built from the trader's own D1 rows or it is built from
    nothing at all.
    """
    import market_thesis
    from ai_jobs import day_review_narration as module

    root = rolling_root["root"]
    monkeypatch.setattr(market_thesis, "read_rows", lambda *_a, **_k: [])

    _outcome, calls = _run(
        root,
        day_reply=_day_reply(root),
        d1_reply=_d1_reply(_d1_items(root, SESSION)[0]["source_id"]),
    )

    d1_call = next(
        call for call in calls
        if call.get("prompt_version") == module.D1_VIEW_PROMPT_VERSION
    )
    assert d1_call["evidence"]["allowed_source_ids"], (
        "with an empty thesis store the D1 clicks and notes are the whole input"
    )


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------


def test_the_rolling_view_is_written_beside_the_day_stories(rolling_root):
    """`DAY_REVIEW_DIR / "d1_view.json"` - ONE rolling file, not one per day."""
    from ai_jobs.day_review_narration import (
        D1_VIEW_SCHEMA,
        d1_view_path,
        read_d1_view,
    )

    root = rolling_root["root"]
    evidence_id = _d1_items(root, SESSION)[0]["source_id"]
    outcome, _calls = _run(
        root, day_reply=_day_reply(root), d1_reply=_d1_reply(evidence_id)
    )

    assert outcome["status"] == "ok", outcome
    path = d1_view_path(root=root)
    assert path == root / "d1_view.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema"] == D1_VIEW_SCHEMA
    assert saved["narration"]["belief_now"].startswith("You have been leaning")
    assert saved["narration"]["open_theses"][0]["still_true"] == "unknown"
    assert read_d1_view(root=root)["inputs_hash"] == saved["inputs_hash"]
    assert str(path) in [str(value) for value in outcome["outputs"]]


def test_the_rolling_view_schema_is_closed_and_bounded(rolling_root):
    from ai_jobs.day_review_narration import D1_VIEW_JSON_SCHEMA

    assert D1_VIEW_JSON_SCHEMA["additionalProperties"] is False
    properties = D1_VIEW_JSON_SCHEMA["properties"]
    assert properties["belief_now"]["maxLength"] == 600
    assert "open_theses" in properties and "sources" in properties

    def _lengths(node):
        if isinstance(node, dict):
            if "maxLength" in node:
                yield node["maxLength"]
            for value in node.values():
                yield from _lengths(value)
        elif isinstance(node, list):
            for value in node:
                yield from _lengths(value)

    assert 2000 not in set(_lengths(D1_VIEW_JSON_SCHEMA))


def test_a_thesis_grounded_in_an_m5_read_is_rejected_and_the_prior_view_stands(
    rolling_root,
):
    """The rolling view is about the bigger picture. An M5 read is not evidence
    for a standing D1 belief, and a cited id outside the allowed list is the
    same rejection the day story makes.
    """
    from ai_jobs.day_review_narration import d1_view_path

    root = rolling_root["root"]
    path = d1_view_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"last week\'s rolling view"}\n', encoding="utf-8")
    before = path.read_bytes()

    import day_review_pack

    pack = day_review_pack.read_pack(SESSION, root=root)
    m5_read = next(
        row for row in pack["reads"]
        if str(row.get("timeframe") or "").upper() == "M5"
    )

    _outcome, _calls = _run(
        root, day_reply=_day_reply(root), d1_reply=_d1_reply(m5_read["source_id"])
    )

    assert path.read_bytes() == before, "the prior rolling view was overwritten"


def test_a_rejected_day_story_never_costs_the_rolling_view_its_prior_file(
    rolling_root,
):
    """Two artifacts, two verdicts. Neither may destroy the other's last good
    file: `market_story_narration`'s rule, applied twice.
    """
    from ai_jobs.day_review_narration import d1_view_path, narration_path

    root = rolling_root["root"]
    story = narration_path(SESSION, root=root)
    story.parent.mkdir(parents=True, exist_ok=True)
    story.write_text('{"verified":"last night\'s story"}\n', encoding="utf-8")
    story_before = story.read_bytes()

    bad = _day_reply(root)
    bad["summary"] = {**bad["summary"], "sources": ["journal:invented"]}
    evidence_id = _d1_items(root, SESSION)[0]["source_id"]

    _outcome, _calls = _run(root, day_reply=bad, d1_reply=_d1_reply(evidence_id))

    assert story.read_bytes() == story_before
    assert d1_view_path(root=root).exists(), (
        "a rejected day story must not stop the rolling view being written"
    )


def test_the_rolling_view_is_rebuilt_only_when_a_d1_row_moved(rolling_root):
    """plan TJ-4 change 3: "rebuilt only when a D1 note or thesis row changed".

    The second run is on the same D1 rows, so the fake REFUSES the D1 prompt -
    a call at all is the failure. The day story's own hash skip is separate and
    is pinned in `test_tj4_narration_slot.py`.
    """
    root = rolling_root["root"]
    evidence_id = _d1_items(root, SESSION)[0]["source_id"]
    first, _calls = _run(
        root, day_reply=_day_reply(root), d1_reply=_d1_reply(evidence_id)
    )
    assert first["status"] == "ok", first

    second, calls = _run(root, day_reply=_day_reply(root), d1_reply=None)

    assert second["status"] == "ok", second
    from ai_jobs import day_review_narration as module

    assert not [
        call for call in calls
        if call.get("prompt_version") == module.D1_VIEW_PROMPT_VERSION
    ]
