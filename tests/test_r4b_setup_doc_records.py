"""R4 Part B item B2 - one horizon per setup doc, one read, and a caller.

`setup_docs._read_family_outcomes` pooled EVERY horizon. The tracker grades each
scan row at 1, 3, 5 and 10 sessions, so `master_avwap_tier_outcomes.csv` carries
one row per `(scan_row_id, horizon)` and reading it whole counts one decision up
to four times. n is inflated, the Wilson lower bound is therefore NARROWER than
the truth, and because the inflation is uneven across families it changes the
ORDER - which is the whole output.

It also re-read the file once per family: 24 documented families, ~0.1 s a read
on the live store, ~2.3 s to render the overview page.

And nothing called `family_record_sentence` at all, so none of it reached a
screen. This file asserts all three.

Offline: every CSV here is written into `tmp_path`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


FIELDS = (
    "scan_row_id",
    "scan_date",
    "setup_family",
    "horizon_sessions",
    "win",
    "side_return_pct",
    "stale_horizon",
)


def _write_tracker(path: Path, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    return path


def _four_horizons(scan_row_id: str, day: str, family: str, wins: dict[int, str]):
    """One decision, graded at all four horizons - exactly what the file holds."""
    return [
        {
            "scan_row_id": scan_row_id,
            "scan_date": day,
            "setup_family": family,
            "horizon_sessions": str(horizon),
            "win": verdict,
            "side_return_pct": "1.5" if verdict == "1" else "-1.0",
            "stale_horizon": "",
        }
        for horizon, verdict in wins.items()
    ]


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    """Ten decisions in one family, each graded at four horizons.

    At horizon 5 the record is 6-4. Pooled over all four it is 24-16 - the same
    RATE, four times the n, and a lower bound that claims a precision the ten
    decisions do not support.
    """
    rows = []
    for index in range(10):
        won = index < 6
        rows += _four_horizons(
            f"row-{index}",
            "2026-09-01",
            "avwap_band_bounce",
            {h: ("1" if won else "0") for h in (1, 3, 5, 10)},
        )
    path = _write_tracker(tmp_path / "outcomes.csv", rows)

    import setup_docs

    monkeypatch.setattr(setup_docs, "_family_outcomes_path", lambda: path)
    monkeypatch.setattr(
        setup_docs, "_family_outcomes_window", lambda: ("2026-08-01", "2026-09-30")
    )
    setup_docs.clear_family_outcome_cache()
    yield path
    setup_docs.clear_family_outcome_cache()


def test_only_one_declared_horizon_reaches_the_record(tracker):
    import setup_docs

    rows = setup_docs._read_family_outcomes("avwap_band_bounce")
    assert len(rows) == 10, "all four horizons were pooled; n is 4x the decisions"
    assert {row["horizon_sessions"] for row in rows} == {"5"}


def test_the_declared_horizon_is_the_desks_one_horizon(tracker):
    """B3's failure is two constants; the docs and the digest read one."""
    import autopilot_core
    import evidence_stats
    import setup_docs

    assert setup_docs.RECORD_HORIZON_SESSIONS == evidence_stats.SWING_HORIZON_SESSIONS
    assert autopilot_core.SWING_DIGEST_HORIZON_SESSIONS == evidence_stats.SWING_HORIZON_SESSIONS


def test_the_bound_widens_once_the_correlated_looks_are_dropped(tracker):
    """The bound is the point: pooling makes it too TIGHT, not too loose."""
    from swing_headline import headline_from_tracker_rows

    import setup_docs

    honest = headline_from_tracker_rows(
        "avwap_band_bounce", setup_docs._read_family_outcomes("avwap_band_bounce")
    )
    pooled = headline_from_tracker_rows(
        "avwap_band_bounce",
        [
            row
            for row in setup_docs._all_family_outcomes().get("avwap_band_bounce", [])
            for row in (row,)
        ],
    )
    assert honest.n == 10
    assert honest.win_rate == pytest.approx(0.6)
    assert honest.win_rate_lb < pooled.win_rate_lb or pooled.n == honest.n


def test_a_stale_horizon_row_is_dropped_and_an_unmeasured_one_is_not(tmp_path, monkeypatch):
    """The same rule the digest and the scan-factor leaderboard apply.

    Only an explicit True drops. `None` means the drift could not be measured,
    and uncertainty is not grounds for deletion.
    """
    import setup_docs

    rows = [
        {
            "scan_row_id": "a", "scan_date": "2026-09-01", "setup_family": "fam",
            "horizon_sessions": "5", "win": "1", "side_return_pct": "2",
            "stale_horizon": "True",
        },
        {
            "scan_row_id": "b", "scan_date": "2026-09-01", "setup_family": "fam",
            "horizon_sessions": "5", "win": "1", "side_return_pct": "2",
            "stale_horizon": "",
        },
        {
            "scan_row_id": "c", "scan_date": "2026-09-01", "setup_family": "fam",
            "horizon_sessions": "5", "win": "0", "side_return_pct": "-1",
            "stale_horizon": "False",
        },
    ]
    path = _write_tracker(tmp_path / "outcomes.csv", rows)
    monkeypatch.setattr(setup_docs, "_family_outcomes_path", lambda: path)
    monkeypatch.setattr(
        setup_docs, "_family_outcomes_window", lambda: ("2026-08-01", "2026-09-30")
    )
    setup_docs.clear_family_outcome_cache()

    kept = setup_docs._read_family_outcomes("fam")
    assert [row["scan_row_id"] for row in kept] == ["b", "c"]


def test_the_file_is_read_once_for_every_family_not_once_per_family(tracker, monkeypatch):
    """24 documented families used to mean 24 full passes over the CSV."""
    import setup_docs

    opens: list[str] = []
    real_open = open

    def counting_open(file, *args, **kwargs):
        opens.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    setup_docs.clear_family_outcome_cache()
    sentences = setup_docs.family_record_sentences(
        ["avwap_band_bounce", "avwap_breakout", "extreme_move_retest"]
    )
    assert len(sentences) == 3
    assert sum(1 for name in opens if name == str(tracker)) == 1, opens


# ---------------------------------------------------------------------------
# It reaches a screen
# ---------------------------------------------------------------------------


def test_the_setup_doc_renderer_shows_the_record():
    pytest.importorskip("PySide6", reason="the docs panel is a Qt widget")
    from ui.panels.setup_docs_panel import render_doc_html

    from setup_docs import SETUP_DOCS

    key = "avwap_band_bounce"
    html = render_doc_html(
        key, SETUP_DOCS[key], record_sentence="avwap_band_bounce: 60% win rate"
    )
    assert "60% win rate" in html


def test_the_overview_renderer_shows_every_familys_record():
    pytest.importorskip("PySide6", reason="the docs panel is a Qt widget")
    from ui.panels.setup_docs_panel import render_all_docs_html

    html = render_all_docs_html(records={"avwap_band_bounce": "MARKER-SENTENCE"})
    assert "MARKER-SENTENCE" in html


def test_the_panel_reads_the_tracker_on_a_worker_and_never_on_the_click():
    """A file read inside `_on_family_selected` is a read on the Qt thread."""
    source = (
        ROOT / "scripts" / "ui" / "panels" / "setup_docs_panel.py"
    ).read_text(encoding="utf-8")
    handler = source.split("def _on_family_selected", 1)[1].split("\ndef ", 1)[0]
    for forbidden in ("family_record_sentence", "_read_family_outcomes", "open("):
        assert forbidden not in handler, (
            f"{forbidden} in the selection handler is a store read on the Qt thread"
        )
    assert "QThread" in source, "the record has to be built somewhere off the click"
