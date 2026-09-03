"""R4 fix round 1, blocker 3 - the link dataset republished at every month roll.

`bronze_like_occurrence_link` is month-partitioned on `partition_ts`, and
`link_rows_for_bronze` set `partition_ts` (and `event_at`) to the RUN STAMP. The
nightly pass looks back 30 days, so a like from late September was written into
September's partition on the 26th and into OCTOBER'S partition on the 1st - and
the caller's dedup reads the row's own partition, as BD-74 requires, so it could
not see the earlier copy.

Reproduced before the fix, one like dated 2026-09-25 over three nightly passes::

    night1 (2026-09-26) link_rows 1  total 1
    night2 (2026-10-01) link_rows 1  total 2   <- same record_hash twice
    night3 (2026-10-02) link_rows 0  total 2

Any count over the dataset, and the BD-92 join that is the ONLY route from an
after-like outcome row back to its setup family, over-counted by however many
month boundaries the lookback had crossed. The A4 test could not see it: it
passed the same stamp twice.

The fix partitions by the LIKE'S OWN DATE, which is also what `event_at` is
supposed to mean - the market fact, not the run. `observed_at` still means when
this installation received the row.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.research_warehouse import like_links  # noqa: E402
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _link(event_id="e1", *, like_date="2026-09-25"):
    return like_links.LikeLink(
        event_id=event_id,
        symbol="NVDA",
        side="LONG",
        like_date=like_date,
        occurrence_id="",
        canonical_setup_id="",
        trigger_at="",
        match_basis=like_links.BASIS_NONE,
        candidates_in_window=0,
    )


def test_the_row_is_partitioned_by_the_like_and_not_by_the_run():
    september = datetime(2026, 9, 26, 4, 0, tzinfo=UTC)
    october = datetime(2026, 10, 1, 4, 0, tzinfo=UTC)

    first = like_links.link_rows_for_bronze([_link()], observed_at=september)[0]
    second = like_links.link_rows_for_bronze([_link()], observed_at=october)[0]

    assert first["partition_ts"] == second["partition_ts"]
    assert first["partition_ts"].date().isoformat() == "2026-09-25"
    assert first["event_at"] == first["partition_ts"], "event_at is the market fact"
    # `observed_at` still says when THIS installation saw it, and it differs.
    assert first["observed_at"] == september
    assert second["observed_at"] == october


def test_three_nightly_passes_across_a_month_boundary_write_one_row(store, tmp_path, monkeypatch):
    """The reproduction, run through the real nightly pass."""
    import project_paths
    from scripts.research_warehouse import cli

    log = tmp_path / "trader_annotations.jsonl"
    log.write_text(
        json.dumps(
            {
                "schema": "trader_annotation_v1",
                "event_id": "e1",
                "event_type": "like_claim",
                "symbol": "NVDA",
                "side": "LONG",
                "session_date": "2026-09-25",
                "ts": "2026-09-25T20:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", log, raising=False)

    written = []
    for stamp in (
        datetime(2026, 9, 26, 4, 0, tzinfo=UTC),
        datetime(2026, 10, 1, 4, 0, tzinfo=UTC),  # the month roll
        datetime(2026, 10, 2, 4, 0, tzinfo=UTC),
    ):
        step = cli._run_after_like_pass(store, {}, stamp=stamp, run_id="r4fix")
        assert step["status"] == "ok", step
        written.append(step["link_rows"])

    rows = store.read_table("bronze_like_occurrence_link").to_pylist()

    assert written == [1, 0, 0], written
    assert len(rows) == 1, "the month roll republished the like"
    assert len({row["record_hash"] for row in rows}) == 1


def test_a_like_with_no_readable_date_still_writes(store):
    """A row filed under the run's month beats a row that cannot be written."""
    stamp = datetime(2026, 10, 1, 4, 0, tzinfo=UTC)

    row = like_links.link_rows_for_bronze([_link(like_date="")], observed_at=stamp)[0]

    assert row["partition_ts"] == stamp
    assert row["event_at"] == stamp


def test_an_unparseable_date_falls_back_rather_than_raising():
    stamp = datetime(2026, 10, 1, 4, 0, tzinfo=UTC)

    assert like_links.like_event_at("not-a-date", stamp) == stamp
    assert like_links.like_event_at("2026-09-25T13:00:00", stamp).date().isoformat() == "2026-09-25"


def test_the_payload_and_the_hash_are_unchanged_by_the_partition_fix():
    """The schema is frozen; only which month the row lands in moved."""
    stamp = datetime(2026, 9, 26, 4, 0, tzinfo=UTC)

    row = like_links.link_rows_for_bronze([_link()], observed_at=stamp)[0]
    payload = json.loads(row["payload"])

    assert set(payload) == {
        "event_id",
        "symbol",
        "side",
        "like_date",
        "occurrence_id",
        "canonical_setup_id",
        "trigger_at",
        "match_basis",
        "candidates_in_window",
    }
    assert row["legacy_id"] == "e1"
    assert row["capture_mode"] == "derived"
    assert row["source_artifact"] == "like_occurrence_link"
