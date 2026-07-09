import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_record_and_load_pick_feedback(tmp_path):
    from pick_feedback import load_pick_feedback, record_pick_feedback

    path = tmp_path / "pick_feedback.jsonl"
    like = record_pick_feedback(
        symbol="nvda",
        side="long",
        verdict="like",
        category="swing",
        origin="h1",
        context="[A-TIER] NVDA: Bounce confirmed (long) H1",
        now=datetime(2026, 7, 8, 10, 15),
        path=path,
    )
    dislike = record_pick_feedback(
        symbol="AAOI",
        side="SHORT",
        verdict="dislike",
        origin="m5",
        reason="chasing a spike, no level behind it",
        path=path,
    )
    assert record_pick_feedback(symbol="", verdict="like", path=path) is None

    rows = load_pick_feedback(path)
    assert [row["symbol"] for row in rows] == ["NVDA", "AAOI"]
    assert like["side"] == "LONG" and like["origin"] == "h1" and like["category"] == "swing"
    assert dislike["verdict"] == "dislike"
    assert rows[1]["reason"] == "chasing a spike, no level behind it"
    assert rows[0]["ts"].startswith("2026-07-08T10:15")


def test_latest_like_origins_keeps_most_recent_like(tmp_path):
    from pick_feedback import latest_like_origins, record_pick_feedback

    path = tmp_path / "pick_feedback.jsonl"
    record_pick_feedback(symbol="NVDA", side="LONG", verdict="like", category="swing", origin="d1", path=path)
    record_pick_feedback(symbol="NVDA", side="LONG", verdict="like", category="swing", origin="h1", path=path)
    record_pick_feedback(symbol="NVDA", side="LONG", verdict="dislike", origin="m5", reason="x", path=path)
    record_pick_feedback(symbol="AAPL", side="SHORT", verdict="like", category="m5", origin="m5", path=path)
    record_pick_feedback(symbol="MSFT", side="LONG", verdict="like", category="swing", path=path)  # no origin

    origins = latest_like_origins(path=path)
    assert origins == {
        ("NVDA", "LONG", "swing"): "h1",  # latest like wins; dislike doesn't clear it
        ("AAPL", "SHORT", "m5"): "m5",
    }


def test_load_pick_feedback_skips_bad_lines_and_missing_file(tmp_path):
    from pick_feedback import load_pick_feedback

    assert load_pick_feedback(tmp_path / "missing.jsonl") == []
    path = tmp_path / "pick_feedback.jsonl"
    path.write_text('{"symbol": "NVDA", "verdict": "like"}\nnot json\n\n[1,2]\n', encoding="utf-8")
    rows = load_pick_feedback(path)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "NVDA"
