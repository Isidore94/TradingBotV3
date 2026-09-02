"""P9 item 4 - the sidecar is finished after the close, so the grade is reachable.

`pass_cohort`'s intraday grade returns blank on every live pass with the reason
`sidecar_ends_before_the_entry_bar`. That is the shape of the evidence, not a bug
in the grade: the sidecar holds the bars the desk was ALREADY HOLDING at the
click, so the first completed close AFTER the click is never inside it.

Completing it overnight makes that bar real - and answers gate 34's open
definition question without changing the definition.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

OPEN = datetime(2026, 9, 1, 9, 30)


def _bar(minute_offset: int, close: float = 10.0) -> dict:
    moment = OPEN + timedelta(minutes=minute_offset)
    return {
        "dt": moment.isoformat(),
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": 1000,
    }


def _row_with_sidecar(tmp_path: Path, *, last_minute: int = 35) -> dict:
    """A capture whose sidecar ends mid-session, exactly like a live one."""
    from ui.annotations.store import LIKE_MODE_QUICK, record_annotation_with_bars

    log = tmp_path / "trader_annotations.jsonl"
    bars = [
        {
            "dt": OPEN + timedelta(minutes=index * 5),
            "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0 + index * 0.1,
            "volume": 1000,
        }
        for index in range(last_minute // 5 + 1)
    ]
    row = record_annotation_with_bars(
        "like_claim",
        symbol="NVDA",
        side="LONG",
        like_mode=LIKE_MODE_QUICK,
        m5_bars=bars,
        path=log,
    )
    assert row is not None and "m5_bars_ref" in row
    return row


def _lake(bars):
    def _reader(symbol, start, end):
        return [b for b in bars], ""
    return _reader


def test_a_sidecar_ending_midsession_is_completed_to_the_close(tmp_path):
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"
    # The rest of the session, from 10:05 to the close.
    rest = [_bar(offset, 11.0) for offset in range(40, 390, 5)]

    result = sc.complete_sidecar(
        row, annotations_path=log, lake_reader=_lake(rest)
    )

    assert result["completed"] is True
    assert result[sc.COMPLETED_SOURCE_FIELD] == sc.SOURCE_LAKE
    assert result["added_bars"] > 60

    completed = json.loads(
        (log.parent / result[sc.COMPLETED_REF_FIELD]).read_text(encoding="utf-8")
    )
    last = datetime.fromisoformat(completed["bars"][-1]["dt"])
    assert last.hour == 15 and last.minute == 55, "it runs to the session close"
    assert completed["completes_ref"] == row["m5_bars_ref"]


def test_the_original_snapshot_is_byte_identical_afterwards(tmp_path):
    """It records what the desk held AT THE CLICK. That is not ours to edit."""
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"
    original = (log.parent / row["m5_bars_ref"]).read_bytes()

    sc.complete_sidecar(
        row, annotations_path=log,
        lake_reader=_lake([_bar(o, 11.0) for o in range(40, 390, 5)]),
    )

    assert (log.parent / row["m5_bars_ref"]).read_bytes() == original


def test_no_lake_and_no_cache_leaves_it_uncompleted_with_a_reason(tmp_path):
    """An unfinished sidecar is a gap; one padded from nowhere is worse."""
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"

    result = sc.complete_sidecar(
        row, annotations_path=log, lake_reader=lambda *a: ([], sc.REASON_NO_STORE)
    )

    assert result["completed"] is False
    assert result["reason"] == sc.REASON_NO_STORE
    assert sc.COMPLETED_REF_FIELD not in result


def test_the_cache_answers_when_the_lake_has_not_ingested_yet(tmp_path):
    """The NORMAL case the morning after: the warehouse runs on its own cadence."""
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"
    rest = [_bar(offset, 11.0) for offset in range(40, 390, 5)]

    result = sc.complete_sidecar(
        row,
        annotations_path=log,
        lake_reader=lambda *a: ([], ""),
        cache_reader=lambda symbol, start, end: rest,
    )

    assert result["completed"] is True
    assert result[sc.COMPLETED_SOURCE_FIELD] == sc.SOURCE_CACHE


def test_a_completed_sidecar_is_never_completed_twice(tmp_path):
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"
    rest = [_bar(offset, 11.0) for offset in range(40, 390, 5)]

    first = sc.complete_sidecar(row, annotations_path=log, lake_reader=_lake(rest))
    merged = {**row, **{k: v for k, v in first.items() if k.startswith(("m5_", "sidecar_"))}}

    second = sc.complete_sidecar(merged, annotations_path=log, lake_reader=_lake(rest))
    assert second["completed"] is False
    assert second["reason"] == sc.REASON_ALREADY_COMPLETED


def test_a_sidecar_that_already_reaches_the_close_is_left_alone(tmp_path):
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=385)
    log = tmp_path / "trader_annotations.jsonl"

    result = sc.complete_sidecar(row, annotations_path=log, lake_reader=_lake([]))
    assert result["completed"] is False
    assert result["reason"] == sc.REASON_ALREADY_COMPLETE


def test_the_reader_prefers_the_completed_file_and_falls_back(tmp_path):
    """ONE reader for both, so no grader has to remember which file to open."""
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"
    rest = [_bar(offset, 11.0) for offset in range(40, 390, 5)]
    fields = sc.complete_sidecar(row, annotations_path=log, lake_reader=_lake(rest))
    merged = {**row, sc.COMPLETED_REF_FIELD: fields[sc.COMPLETED_REF_FIELD]}

    assert len(sc.read_completed_bars(merged, annotations_path=log)["bars"]) > 60
    assert len(sc.read_completed_bars(row, annotations_path=log)["bars"]) == 8

    # An unreadable completed file falls back to the snapshot that certainly works.
    (log.parent / fields[sc.COMPLETED_REF_FIELD]).write_text("{", encoding="utf-8")
    assert len(sc.read_completed_bars(merged, annotations_path=log)["bars"]) == 8


def test_every_refusal_is_counted_and_one_bad_row_never_stops_the_night(tmp_path):
    from ui.annotations import sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"

    report = sc.complete_sidecars(
        [row, {"event_id": "nothing"}, {}],
        annotations_path=log,
        lake_reader=lambda *a: ([], sc.REASON_STORE_UNREACHABLE),
    )
    assert report["completed"] == {}
    assert report["reasons"][sc.REASON_STORE_UNREACHABLE] == 1
    assert report["reasons"][sc.REASON_NO_SIDECAR] == 2


# ---------------------------------------------------------------------------
# The additive field: SCHEMA_VERSION stays 1, and this is why
# ---------------------------------------------------------------------------


def test_every_reader_tolerates_the_new_field(tmp_path, monkeypatch):
    """`like_mode` is additive, so the schema version does not move.

    Proven rather than asserted: each reader below is handed a row carrying the
    new key and has to produce its normal answer. A version bump would force all
    of them to learn a number that changes nothing about how they read.
    """
    import json as _json

    import project_paths
    from ui.annotations.store import LIKE_MODE_QUICK, SCHEMA_VERSION, build_annotation

    assert SCHEMA_VERSION == 1

    log = tmp_path / "trader_annotations.jsonl"
    quick = build_annotation(
        "like_claim", symbol="NVDA", side="LONG", like_mode=LIKE_MODE_QUICK
    )
    claimed = build_annotation(
        "like_claim", symbol="AMD", side="LONG",
        claimed_setup_id="avwap_breakout", note="held it",
    )
    # A row written BEFORE P9 - no `like_mode` at all.
    legacy = {k: v for k, v in claimed.items() if k != "like_mode"}
    legacy["symbol"] = "OLD"
    log.write_text(
        "\n".join(_json.dumps(row) for row in (quick, claimed, legacy)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", log)

    # 1. The generic loader.
    from ui.annotations.store import EVENT_LIKE_CLAIM, load_annotations, like_mode_of

    loaded = load_annotations(log, event_types=(EVENT_LIKE_CLAIM,))
    assert len(loaded) == 3
    assert [like_mode_of(row) for row in loaded] == ["quick", "claimed", "claimed"], (
        "a row written before P9 reads as claimed - a claim was REQUIRED until then"
    )

    # 2. The like cohort.
    from ui.annotations import like_cohort

    rows, _skipped = like_cohort.like_pick_rows(loaded)
    assert {r["symbol"] for r in rows} == {"NVDA", "AMD", "OLD"}
    assert {r["source"] for r in rows} == {"like_unclaimed", "like_avwap_breakout"}

    # 3. The auto-tagger's capture lane.
    from journal_analytics import AutoTagger

    capture = AutoTagger()._load_annotation_capture_rows()
    by_symbol = {row["symbol"]: row for row in capture}
    assert by_symbol["NVDA"]["link_only"] is True, "a quick like names no setup"
    assert by_symbol["NVDA"]["tag"] == ""
    assert by_symbol["AMD"]["tag"] == "avwap_breakout"

    # 4. The pass cohort, which reads the same log for a different event type.
    from ui.annotations import pass_cohort

    pass_rows, _ = pass_cohort.pass_pick_rows(loaded)
    assert pass_rows == [], "no passes in this log, and no crash reading past them"


def test_the_completed_sidecar_makes_the_intraday_grade_a_number(tmp_path):
    """Gate 34's open question, answered without changing the definition.

    Entry stays "the first completed M5 close AFTER the click". It was
    unreachable because the snapshot ended at the click; completing the sidecar
    after the close makes that bar real.
    """
    from ui.annotations import pass_cohort, sidecar_completion as sc

    row = _row_with_sidecar(tmp_path, last_minute=35)
    log = tmp_path / "trader_annotations.jsonl"
    snapshot = sc.read_completed_bars(row, annotations_path=log)

    # BEFORE: the grade cannot find an entry bar and says exactly why.
    # The click happens AFTER the last bar the desk was holding completed,
    # which is the live shape: the snapshot is what was in hand at that moment.
    passed_at = datetime.fromisoformat(snapshot["bars"][-1]["dt"]) + timedelta(minutes=5)
    before = pass_cohort.intraday_pass_outcome(
        snapshot["bars"], side="LONG", passed_at=passed_at
    )
    assert before["intraday_unmeasured_reason"] == "sidecar_ends_before_the_entry_bar"
    assert before["intraday_close_r"] in ("", None)
    assert before["intraday_entry_at"] in ("", None)

    # AFTER: the rest of the session is on disk, and it is a number.
    rest = []
    price = 10.7
    for offset in range(40, 390, 5):
        price += 0.05
        rest.append(
            {
                "dt": (OPEN + timedelta(minutes=offset)).isoformat(),
                "open": price, "high": price + 0.3, "low": price - 0.05,
                "close": price, "volume": 1000,
            }
        )
    fields = sc.complete_sidecar(row, annotations_path=log, lake_reader=_lake(rest))
    merged = {**row, sc.COMPLETED_REF_FIELD: fields[sc.COMPLETED_REF_FIELD]}
    completed = sc.read_completed_bars(merged, annotations_path=log)

    after = pass_cohort.intraday_pass_outcome(
        completed["bars"], side="LONG", passed_at=passed_at
    )
    assert after["intraday_unmeasured_reason"] == ""
    assert after["intraday_entry_at"], after
    assert after["intraday_close_r"] not in ("", None), after
    assert after["intraday_first_hit"], after
