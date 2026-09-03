"""V3 item 4 - one stream, and no sixth writer.

Decision 0016 goal 2: *"teach the bot what the trader likes with ONE CLICK, from
any screen."* P9 and P10 built that; this asserts the seams stayed closed.

**What is measured rather than assumed.** P10 defined five `surface` values and
wired three. R4 A5 wired the last two, so all five now have a writer:

* `master_avwap_setups` - the setups table's star and cross;
* `chart_review` - the review pane's "Not today", and the capture rail hosted on
  a chart-review screen, which now calls the `surface` override that
  `set_scan_context` has carried since P10 B1 and that no host ever used;
* `rail` - the rail standing on its own;
* `focus_panel` - the Focus chip's right-click Like / Not today (R4 A5);
* `m5_alert_bar` - the bar row's right-click quick like (R4 A5).

The M5 bar's click-AWAY is still a review event and deliberately not an
annotation: a click away is a pass and `review_learning` keys on
`clicked_away_from_m5_alert`. What R4 added is a separate, explicit verb.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: The surfaces that actually write today. Not the vocabulary - the callers.
WIRED_SURFACES = (
    "master_avwap_setups",
    "chart_review",
    "rail",
    "focus_panel",
    "m5_alert_bar",
)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# One writer
# ---------------------------------------------------------------------------


def test_every_like_and_dislike_goes_through_the_one_module():
    """A second writer is how two screens end up with two row shapes."""
    import re

    callers: dict[str, list[str]] = {}
    for path in (ROOT / "scripts").rglob("*.py"):
        if path.name in {"store.py", "verdicts.py"} and "annotations" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        hits = re.findall(r"record_annotation(?:_with_bars)?\(", text)
        if hits:
            callers[str(path.relative_to(ROOT)).replace("\\", "/")] = hits

    # Exactly ONE module outside the store may call the raw writer, and it is the
    # capture rail - which now stamps the surface on every row it writes.
    assert set(callers) == {"scripts/ui/widgets/capture_rail.py"}, callers


def test_the_rails_veto_carries_a_surface_like_its_like_does(tmp_path):
    """The seam V3 closed: a veto and a like from one rail had different shapes.

    Any rollup by screen silently omitted every veto, which is exactly the kind
    of gap that reads as "the trader does not veto from the rail".
    """
    source = (ROOT / "scripts" / "ui" / "widgets" / "capture_rail.py").read_text(
        encoding="utf-8"
    )
    record = source.split("def _record(self", 1)[1].split("def commit_veto", 1)[0]
    assert 'common.setdefault("surface", self._surface)' in record
    assert 'common.setdefault("scan_context"' in record


def test_every_declared_surface_has_a_gesture_that_writes_it():
    """Measured, not assumed. R4 A5 closed the last two.

    A surface in the vocabulary with no writer is a column that can never be
    populated, and a rollup over it reads as "the trader never judges from that
    screen" rather than as "nobody wired it".
    """
    from ui.annotations import verdicts

    assert set(WIRED_SURFACES) == set(verdicts.SURFACES)

    wired = set()
    for path, surface in (
        ("scripts/ui/panels/master_avwap_panel.py", "SURFACE_MASTER_AVWAP"),
        ("scripts/ui/panels/alert_center_panel.py", "SURFACE_CHART_REVIEW"),
        ("scripts/ui/widgets/capture_rail.py", "SURFACE_RAIL"),
        ("scripts/ui/panels/focus_picks_panel.py", "SURFACE_FOCUS_PANEL"),
        ("scripts/ui/widgets/m5_alert_bar.py", "SURFACE_M5_ALERT_BAR"),
    ):
        text = (ROOT / path).read_text(encoding="utf-8")
        if surface in text or "record_not_today" in text or "self._surface" in text:
            wired.add(surface)
    assert len(wired) == 5, wired

    # And the chart-review OVERRIDE is called by the hosts that are that screen.
    for host in (
        "scripts/ui/widgets/alert_chart_review.py",
        "scripts/ui/panels/chart_review_panel.py",
    ):
        text = (ROOT / host).read_text(encoding="utf-8")
        assert "set_scan_context(surface=SURFACE_CHART_REVIEW)" in text, host


# ---------------------------------------------------------------------------
# One stream, read by everything downstream
# ---------------------------------------------------------------------------


def test_the_like_cohort_the_after_like_grid_and_the_capture_lane_read_one_file():
    """Three consumers, one stream. A second stream is a second answer."""
    from project_paths import TRADER_ANNOTATIONS_FILE

    for module, needle in (
        ("scripts/ui/annotations/like_cohort.py", "TRADER_ANNOTATIONS_FILE"),
        ("scripts/research_warehouse/cli.py", "TRADER_ANNOTATIONS_FILE"),
    ):
        text = (ROOT / module).read_text(encoding="utf-8")
        assert needle in text, module

    assert TRADER_ANNOTATIONS_FILE.name == "trader_annotations.jsonl"


def test_a_row_from_every_wired_surface_grades_in_one_cohort(tmp_path):
    from ui.annotations import like_cohort, verdicts
    from ui.annotations.store import load_annotations

    path = tmp_path / "trader_annotations.jsonl"
    for index, surface in enumerate(WIRED_SURFACES):
        verdicts.record_like(
            symbol=f"SYM{index}",
            side="LONG",
            surface=surface,
            session_date="2026-09-02",
            path=path,
        )

    rows, skipped = like_cohort.like_pick_rows(load_annotations(path))

    assert skipped == 0
    assert len(rows) == len(WIRED_SURFACES)
    assert {row["source"] for row in rows} == {"like_unclaimed"}, "one bucket"
    assert sorted(row["surface"] for row in rows) == sorted(WIRED_SURFACES)


def test_an_annotation_written_without_a_surface_is_still_readable(tmp_path):
    """Every row written before P10 has none, and they are never rewritten."""
    from ui.annotations import like_cohort
    from ui.annotations.store import load_annotations, record_annotation

    path = tmp_path / "trader_annotations.jsonl"
    record_annotation(
        "like_claim",
        symbol="OLD",
        side="LONG",
        session_date="2026-08-01",
        like_mode="quick",
        path=path,
    )
    rows, _skipped = like_cohort.like_pick_rows(load_annotations(path))

    assert len(rows) == 1
    assert rows[0]["surface"] == "", "absent, not invented"
