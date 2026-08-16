import json
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


def test_reviewed_today_unions_decisions_from_all_three_ledgers(tmp_path):
    from pick_feedback import clear_reviewed_today_cache, reviewed_symbols_today

    pick_path = tmp_path / "pick_feedback.jsonl"
    events_path = tmp_path / "review_events.jsonl"
    annotations_path = tmp_path / "trader_annotations.jsonl"
    pick_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"trade_date": "2026-08-14", "symbol": "NVDA", "verdict": "like"},
                {"trade_date": "2026-08-13", "symbol": "OLD", "verdict": "dislike"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    events_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"trade_date": "2026-08-14", "symbol": "AMD", "action": "dislike"},
                {"trade_date": "2026-08-14", "symbol": "SHOWN", "action": "shown"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"session_date": "2026-08-14", "symbol": "TSLA", "event_type": "veto"},
                {"session_date": "2026-08-14", "symbol": "META", "event_type": "note"},
                {"session_date": "2026-08-14", "symbol": "STOP", "event_type": "hypo_stop"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    clear_reviewed_today_cache()
    reviewed = reviewed_symbols_today(
        market_date="2026-08-14",
        pick_feedback_path=pick_path,
        review_events_path=events_path,
        annotations_path=annotations_path,
    )
    assert reviewed == {"NVDA", "AMD", "TSLA", "META"}

    # Signature-keyed caching invalidates when a ledger grows.
    with pick_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"trade_date": "2026-08-14", "symbol": "AAPL", "verdict": "dislike"}) + "\n")
    assert reviewed_symbols_today(
        market_date="2026-08-14",
        pick_feedback_path=pick_path,
        review_events_path=events_path,
        annotations_path=annotations_path,
    ) == {"NVDA", "AMD", "TSLA", "META", "AAPL"}
