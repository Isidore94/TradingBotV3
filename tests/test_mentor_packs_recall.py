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
