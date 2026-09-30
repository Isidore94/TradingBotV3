"""Mentor app P4: the full /scorecard - challenge hit rate by kind with n, and the app's own service stats."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import challenge  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402


def _facts(day: str, values, offline: float, numbers: int, uncited: int) -> dict:
    return {"session_date": day, "first_token_ms": {"values": list(values)}, "brain_offline_min": offline,
            "numbers": numbers, "uncited_numbers": uncited}


def test_the_scorecard_lists_each_kind_with_n_and_the_service_stats(tmp_path):
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    for index in range(30):
        store.add_challenge(f"veto:{index}", kind="veto", symbol="X", claim="c", outcome={"hit": index < 18})
    for index in range(4):
        store.add_challenge(f"pick:{index}", kind="pick", symbol="Y", claim="c", outcome={"hit": True})
    store.update_challenge("pick:0", outcome={"hit": True}, graded_utc="2026-10-01T00:00:00+00:00")
    facts = [_facts(f"2026-09-{day:02d}", [1000, 3000], 2, 10, 1) for day in range(20, 30)]
    text = challenge.scorecard(store, facts=facts)
    assert "**Scorecard: veto challenges**" in text and "hit rate 60% (n=30, floor 30)" in text
    assert "**Scorecard: pick challenges**" in text and "issued 4, fully graded 1" in text
    assert "hit rate: too few (n=4, floor 30)" in text
    assert "(last 7 nights of facts)" in text, "only the newest seven nights count"
    assert "first token p50 2000 ms (n=14)" in text
    assert "brain offline 14 min" in text and "uncited numbers 7 of 70 (10%)" in text


def test_the_scorecard_without_night_facts_says_so(tmp_path):
    text = challenge.scorecard(MentorChatStore(tmp_path / "chat.sqlite3"))
    assert "too few (n=0, floor 30)" in text and "no night facts yet" in text
