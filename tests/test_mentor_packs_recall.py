"""Trade Mentor recall pack: off without a brain, cosine ranking with one."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import recall  # noqa: E402


def test_without_a_searcher_recall_is_a_no_op():
    recall.set_searcher(None)
    pack = recall.build("NVDA")
    assert pack.ids == () and recall.OFF_TEXT in pack.as_text()


def test_fixture_ranks_the_closer_memory_first():
    pack = recall.fixture()
    assert pack.ids[0] == "mem:turn:1"
    assert len(pack.ids) == len(set(pack.ids))


def test_top_k_orders_by_cosine_and_caps():
    rows = [{"ref_id": i, "vector": vec} for i, vec in enumerate(([0, 1], [1, 0], [1, 1]))]
    ranked = recall.top_k([1.0, 0.0], rows, k=2)
    assert [row["ref_id"] for row in ranked] == [1, 2]


def test_a_dropped_brain_falls_back_to_substring_search_and_logs_once(caplog):
    def dropped(_texts):
        raise ConnectionError("the tunnel is down")

    searcher = recall.make_searcher(dropped, lambda: [])
    recall.set_searcher(searcher)
    recall.set_fallback(lambda query, k: [{"kind": "note", "ref_id": 7, "text": f"said {query}"}])
    try:
        with caplog.at_level("WARNING", logger=recall.__name__):
            first = recall.build("NVDA")
            second = recall.build("AMD")
        assert first.ids == ("mem:note:7",) and "said NVDA" in first.as_text()
        assert second.ids == ("mem:note:7",)
        warned = [rec for rec in caplog.records if rec.name == recall.__name__]
        assert len(warned) == 1, "one log line per drop, not one per call"
        recall.set_fallback(None)
        assert recall.build("NVDA").ids == (), "no fallback: an empty pack, never a raise"
    finally:
        recall.set_searcher(None)
        recall.set_fallback(None)
