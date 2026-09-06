"""ST2 fix round - the reviewer's three blockers and eight advisories.

Written by the builder AFTER the reviewer's NO-GO on `1145d3b7`, each test
proven to fail on that tip before its fix existed.

**Blocker 1** - the Summary card's plain-English block sat THREE LINES ABOVE the
fixed banner and still crowned `max(avg_closed_r)` over any family with three
closes across BOTH namespaces (`research_explanations.py`). Live, it read *"LONG
top_pattern leads at +0.99R ... 3 closes"* while the banner underneath said
*"SHORT general"*, and 10 of its 17 candidates were STUDIES.

**Blocker 2** - the "Best Type Edge" tile read `setup_type_rows[0]`, so ST2.2's
new bound-first sort silently moved it from `SHORT +23` to `LONG +14`.

**Blocker 3** - the Summary's "Setup types working" block shows `rows[:8]` of a
list ST2.2 made side-first, so it showed eight LONG rows and no SHORT one (the
first SHORT row sat at index 68 of 117 on the live export).

**Advisories** - the new columns landed mid-header; `last_reliable_reading` was
unreachable from the panel; `min_n` did not bind the stale/undated discovery
pools; the population sentence named no window and no freshness rule and read an
exported integer 0 as "not exported"; `flats` was not passed to
`headline_from_counts`; and an `n_expired_unmeasured` of 0 rendered as nothing.

**The ask answered** (lead, 2026-09-06): `build_tracker_short_horizon_rows` may
gain additive `n_wins` / `n_losses` / `latest_measured_session`,
default-preserving and golden-pinned - `tests/fixtures/st2_short_horizon_golden.csv`
was pinned from the code as it stood BEFORE those columns existed.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SHIPPED_HEADERS_GOLDEN = FIXTURES / "st2_shipped_headers_golden.json"
SHORT_HORIZON_GOLDEN = FIXTURES / "st2_short_horizon_golden.csv"

SHORT_REFERENCE_DATE = date(2026, 4, 22)


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is a gui extra
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel_module():
    if _qt_app() is None:  # pragma: no cover - PySide6 is a gui extra
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    yield setup_tracker_panel
    setup_tracker_panel.clear_setup_tracker_csv_cache()


def _legacy():
    from master_avwap_lib import legacy

    return legacy


def _last_completed_session() -> date:
    import market_calendar

    return market_calendar.last_completed_session(datetime.now(market_calendar.MARKET_TZ))


def _setup(
    symbol,
    scan_date,
    total_r,
    *,
    side="LONG",
    family="post_earnings_52w_break",
    bucket="favorite_setup",
    scenario_status="TARGET_HIT",
    favorite_zone="",
    retest_label="",
    compression_label="N",
):
    setup_status = "CLOSED" if scenario_status in {"TARGET_HIT", "STOPPED"} else "OPEN"
    return {
        "setup_id": f"{scan_date}:{symbol}:{family}:{retest_label}:{compression_label}",
        "symbol": symbol,
        "side": side,
        "scan_date": scan_date,
        "priority_bucket": bucket,
        "setup_family": family,
        "favorite_zone": favorite_zone,
        "retest_label": retest_label,
        "compression_label": compression_label,
        "setup_tags": [],
        "favorite_signals": [],
        "setup_status": setup_status,
        "scenarios": {
            "baseline": {
                "experimental": False,
                "tradeable": True,
                "status": scenario_status,
                "total_r": total_r,
                "stop_reference_label": "LOWER_1" if side.upper() == "LONG" else "UPPER_1",
            }
        },
    }


def _evidence_row(
    family: str,
    *,
    namespace: str = "live",
    wins: int,
    losses: int,
    side: str = "LONG",
    avg_closed_r: float = 0.5,
    session: str | None = None,
    flats: int = 0,
) -> dict:
    n = wins + losses
    return {
        "namespace": namespace,
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "status": "",
        "closed_setups": str(n),
        "tracked_setups": str(n),
        "avg_closed_r": str(avg_closed_r),
        "target_hit_rate": "0.5",
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": str(flats),
        "n_unmeasured": "0",
        "n_pending": "0",
        "outcome_kind": "trade_r_representative_exit",
        "horizon_basis": "30d lookback, representative exit",
        "latest_measured_session": session or _last_completed_session().isoformat(),
    }


# ===========================================================================
# Blocker 1 - one leader on the card, and it is the banner's
# ===========================================================================


def test_the_plain_english_block_reads_the_same_verdict_as_the_banner(
    panel_module, tmp_path, monkeypatch
):
    """`fat_but_wide` has the biggest mean R, `tight_and_hot` the biggest bound,
    and `huge_r_study` beats both on every visible number. Before the fix the
    card crowned the max-R row and the banner named the bound leader; now both
    name the bound leader and neither names the study."""
    from research_explanations import build_plain_english_whats_working

    session = _last_completed_session().isoformat()
    rows = [
        _evidence_row("huge_r_study", namespace="study", wins=40, losses=2, avg_closed_r=3.1, session=session),
        _evidence_row("fat_but_wide", wins=54, losses=36, avg_closed_r=2.50, session=session),
        _evidence_row("tight_and_hot", wins=24, losses=6, avg_closed_r=0.40, session=session, side="SHORT"),
    ]

    plain = build_plain_english_whats_working(recent_rows=rows)
    swing_bullets = [text for text in plain["bullets"] if "recently closed swings" in text]
    assert len(swing_bullets) == 1, plain["bullets"]
    bullet = swing_bullets[0]
    assert "tight_and_hot" in bullet, bullet
    assert "fat_but_wide" not in bullet, bullet
    assert "huge_r_study" not in bullet, bullet

    csv_path = tmp_path / "recent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", csv_path)
    panel = panel_module.SetupTrackerPanel()
    try:
        banner = panel_module._best_now_banner_html(panel)
        summary = panel_module._summary_html(panel)
    finally:
        panel.deleteLater()

    # The card and the banner are on the SAME page and now name the SAME family.
    assert "tight_and_hot" in banner, banner
    assert "fat_but_wide" not in summary, summary
    assert "huge_r_study" not in summary, summary


def test_a_study_only_card_says_no_clear_leader_rather_than_naming_the_study():
    from research_explanations import build_plain_english_whats_working

    session = _last_completed_session().isoformat()
    rows = [
        _evidence_row("only_a_study", namespace="study", wins=40, losses=2, avg_closed_r=3.1, session=session)
    ]
    plain = build_plain_english_whats_working(recent_rows=rows)
    bullet = next(text for text in plain["bullets"] if "recently closed swings" in text)
    assert "no clear leader" in bullet.lower(), bullet
    assert "only_a_study" not in bullet, bullet


# ===========================================================================
# Blocker 2 - the edge tile reads its own meaning
# ===========================================================================


def test_the_best_type_edge_tile_picks_the_biggest_delta_not_the_tables_first_row(
    panel_module,
):
    """The tab sorts by the bound inside each side, so its first row is a LONG
    row with a small delta. The tile is headed EDGE and must say +23."""
    rows = [
        {"side": "LONG", "setup_family": "bound_leader", "score_delta": "14", "win_rate_lb": 0.62},
        {"side": "SHORT", "setup_family": "edge_leader", "score_delta": "23", "win_rate_lb": 0.41},
    ]
    assert panel_module._best_type_label(rows) == "SHORT +23"
    # ...and reversing the list cannot change the answer.
    assert panel_module._best_type_label(list(reversed(rows))) == "SHORT +23"
    assert panel_module._best_type_label([]) == "-"


# ===========================================================================
# Blocker 3 - the Summary card shows both books
# ===========================================================================


def test_the_summary_setup_type_block_shows_both_sides(panel_module):
    """Ten LONG rows with better bounds than the two SHORT ones. The tab keeps
    its side-first order; the CARD takes its eight across both sides, so a
    reader is never told the short book has nothing."""
    rows = [
        {
            "side": "LONG",
            "setup_family": f"long_{index}",
            "win_rate_lb": 0.60 - index * 0.01,
            "score_delta": "1",
            "closed_setups": "40",
        }
        for index in range(10)
    ] + [
        {
            "side": "SHORT",
            "setup_family": "short_best",
            "win_rate_lb": 0.55,
            "score_delta": "1",
            "closed_setups": "40",
        },
        {
            "side": "SHORT",
            "setup_family": "short_thin",
            "win_rate_lb": 0.10,
            "score_delta": "1",
            "closed_setups": "40",
        },
    ]
    ranked = panel_module._rank_setup_types(rows, min_closed=0)
    assert {row["side"] for row in ranked[:8]} == {"LONG"}, "the TAB stays side-first"

    card = panel_module._summary_setup_type_rows(ranked, limit=8)
    sides = {row["side"] for row in card}
    assert sides == {"LONG", "SHORT"}, [row["setup_family"] for row in card]
    assert "short_best" in [row["setup_family"] for row in card]
    # Ordered by the bound across both books.
    bounds = [row["win_rate_lb"] for row in card]
    assert bounds == sorted(bounds, reverse=True), bounds


# ===========================================================================
# Advisory 1 - the new columns are at the END of the SHIPPED header
# ===========================================================================


def _shipped_payload() -> dict:
    today = date.today()
    return {
        "setups": {
            "a": _setup("AAA", (today - timedelta(days=1)).isoformat(), 1.8),
            "b": _setup("BBB", (today - timedelta(days=3)).isoformat(), -1.0, scenario_status="STOPPED"),
            "c": _setup("CCC", (today - timedelta(days=5)).isoformat(), 0.5, scenario_status="OPEN"),
        },
        "study_setups": {
            "s": _setup("SSS", (today - timedelta(days=2)).isoformat(), 2.4, family="second_dev_breakout"),
        },
    }


def test_the_count_columns_land_at_the_end_of_both_shipped_headers():
    """The golden is the SHIPPED header as `main` at `84ee24d6` wrote it - the
    header of the file the trader opens, after `namespace` / `status` and after
    the rank columns, not the inner builder's."""
    legacy = _legacy()
    golden = json.loads(SHIPPED_HEADERS_GOLDEN.read_text(encoding="utf-8"))
    payload = _shipped_payload()

    recent = legacy.build_recent_setup_type_stat_rows(payload)
    assert recent, "the fixture must produce rows"
    recent_columns = list(recent[0].keys())
    base = golden["recent_shipped"]
    assert recent_columns[: len(base)] == base, recent_columns
    assert recent_columns[len(base) :] == list(legacy.TRACKER_RECENT_COUNT_COLUMNS)

    types = legacy.build_tracker_setup_type_rows(payload["setups"])
    type_columns = list(types[0].keys())
    type_base = golden["setup_type_shipped"]
    assert type_columns[: len(type_base)] == type_base, type_columns
    assert type_columns[len(type_base) :] == list(legacy.TRACKER_SETUP_TYPE_COUNT_COLUMNS)

    # Every row carries the same header - a ragged export is a broken CSV.
    for row in recent:
        assert list(row.keys()) == recent_columns
    for row in types:
        assert list(row.keys()) == type_columns


# ===========================================================================
# The answered ask - the 2-session export counts its own wins
# ===========================================================================


def _short_horizon_setups(*, with_marks: bool = True) -> dict:
    """The golden's five episodes.

    ``daily_marks`` carry the trade date of every bar, and the 2-session R is
    read from ``post_marks[1]`` - so the MEASURED session is that bar's date,
    not the entry (re-review advisory 4). They are additive to the fixture: no
    old column of this export reads them, which the golden test proves.
    """

    def with_bars(record, r1, r2, mfe, mae, marks):
        record["short_horizon"] = {
            "r_close_1d": r1,
            "r_close_2d": r2,
            "mfe_r_2d": mfe,
            "mae_r_2d": mae,
            "complete": True,
        }
        if with_marks:
            record["daily_marks"] = [
                {"trade_date": marks[0], "is_entry_day": True, "close": 10.0},
                {"trade_date": marks[1], "is_entry_day": False, "close": 10.5},
                {"trade_date": marks[2], "is_entry_day": False, "close": 11.0},
            ]
        return record

    return {
        "h1": with_bars(
            _setup("AAA", "2026-04-21", 1.8), 0.6, 1.2, 1.5, -0.3,
            ("2026-04-21", "2026-04-22", "2026-04-23"),
        ),
        "h2": with_bars(
            _setup("BBB", "2026-04-20", -1.0, scenario_status="STOPPED"), -0.4, -0.9, 0.2, -1.1,
            ("2026-04-20", "2026-04-21", "2026-04-22"),
        ),
        "h3": with_bars(
            _setup("CCC", "2026-04-18", 0.0), 0.1, 0.0, 0.4, -0.2,
            ("2026-04-18", "2026-04-21", "2026-04-22"),
        ),
        "h4": with_bars(
            _setup("DDD", "2026-04-15", 2.2, side="SHORT"), 0.9, 1.7, 2.0, -0.1,
            ("2026-04-15", "2026-04-16", "2026-04-17"),
        ),
        "h5": with_bars(
            _setup("EEE", "2026-04-10", -0.5, side="SHORT", scenario_status="STOPPED"), -0.2, -0.6, 0.1, -0.8,
            ("2026-04-10", "2026-04-13", "2026-04-14"),
        ),
    }


def test_the_short_horizon_export_keeps_every_old_column_byte_identical():
    """Golden pinned from the code BEFORE the seven columns were added."""
    import pandas as pd

    legacy = _legacy()
    rows = legacy.build_tracker_short_horizon_rows(
        _short_horizon_setups(), reference_date=SHORT_REFERENCE_DATE
    )
    with SHORT_HORIZON_GOLDEN.open("r", newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
        handle.seek(0)
        golden_text = handle.read()

    produced = pd.DataFrame(rows)[columns].to_csv(index=False)
    assert produced == golden_text

    # ...and the new ones are AFTER them, never wedged in.
    produced_columns = list(rows[0].keys())
    assert produced_columns[: len(columns)] == columns, produced_columns
    assert produced_columns[len(columns) :] == list(legacy.TRACKER_SHORT_HORIZON_COUNT_COLUMNS)


def test_the_short_horizon_export_counts_its_own_wins_and_dates_them():
    legacy = _legacy()
    rows = legacy.build_tracker_short_horizon_rows(
        _short_horizon_setups(), reference_date=SHORT_REFERENCE_DATE
    )
    by_side = {row["side"]: row for row in rows}

    # LONG: +1.20, -0.90, 0.00 -> one win, one loss, one FLAT.
    long_row = by_side["LONG"]
    assert (long_row["n_wins"], long_row["n_losses"], long_row["n_flats"]) == (1, 1, 1)
    assert long_row["n_unmeasured"] == 0
    # The pre-existing weighted-free rate still counts the flat as a zero flag,
    # unchanged - it is an existing column and moving it would be a scoring
    # change this packet is not allowed to make.
    assert long_row["win_rate_2d"] == pytest.approx(1 / 3, abs=1e-9)
    # The session it was MEASURED on - the second bar after AAA's 2026-04-21
    # entry - not the entry itself (advisory 4).
    assert long_row["latest_measured_session"] == "2026-04-23"

    short_row = by_side["SHORT"]
    assert (short_row["n_wins"], short_row["n_losses"], short_row["n_flats"]) == (1, 1, 0)
    assert short_row["latest_measured_session"] == "2026-04-17"
    assert short_row["outcome_kind"] == "trade_r_close_2d"

    # The identity the export holds, stated so a reader never sums it wrong
    # (advisory 5): wins + losses + flats == samples_2d, and
    # samples_2d + unmeasured == tracked_setups.
    for row in rows:
        assert row["n_wins"] + row["n_losses"] + row["n_flats"] == row["samples_2d"]
        assert row["samples_2d"] + row["n_unmeasured"] == row["tracked_setups"]


def test_a_two_session_row_whose_marks_cannot_date_it_is_undated_not_guessed():
    """No `daily_marks`, so the measured session is unknown. It is EMPTY -
    which `select_leader` reads as not fresh - never back-filled from the entry
    date, which is what made a current file read as 58 sessions stale."""
    legacy = _legacy()
    rows = legacy.build_tracker_short_horizon_rows(
        _short_horizon_setups(with_marks=False), reference_date=SHORT_REFERENCE_DATE
    )
    assert rows
    for row in rows:
        assert row["latest_measured_session"] == "", row


def _current_short_horizon_setups() -> dict:
    """Seven episodes MEASURED on the last completed session.

    Entered eight weeks ago and measured two sessions later would still read as
    stale under entry dating; these are entered long ago and measured NOW, which
    is the case advisory 4 exists for.
    """
    import market_calendar

    measured = _last_completed_session()
    day_before = market_calendar.previous_session(measured)
    entry = day_before
    for _ in range(40):
        entry = market_calendar.previous_session(entry)

    setups = {}
    for index, (r2, status) in enumerate(
        [(1.2, "TARGET_HIT"), (0.9, "TARGET_HIT"), (1.4, "TARGET_HIT"), (0.7, "TARGET_HIT"),
         (1.1, "TARGET_HIT"), (-0.8, "STOPPED"), (-0.5, "STOPPED")]
    ):
        record = _setup(f"S{index:02d}", entry.isoformat(), r2, scenario_status=status)
        record["short_horizon"] = {
            "r_close_1d": r2 / 2.0,
            "r_close_2d": r2,
            "mfe_r_2d": abs(r2),
            "mae_r_2d": -abs(r2) / 3.0,
            "complete": True,
        }
        record["daily_marks"] = [
            {"trade_date": entry.isoformat(), "is_entry_day": True, "close": 10.0},
            {"trade_date": day_before.isoformat(), "is_entry_day": False, "close": 10.5},
            {"trade_date": measured.isoformat(), "is_entry_day": False, "close": 11.0},
        ]
        setups[f"s{index}"] = record
    return setups


def test_a_current_two_session_export_renders_a_leader_and_never_says_discovery(
    panel_module, tmp_path, monkeypatch
):
    """The whole chain, from the real export to the rendered banner.

    Blocker 2 and advisory 4 together: a file measured on the last completed
    session must render a LEADER, its label must say so, and the words
    "discovery" and the no-session sentence must be absent. Before the fix the
    export dated itself by the ENTRY (here 41 sessions back), so this block
    could only ever be discovery, and its label hardcoded the word.
    """
    import pandas as pd

    from working_lately import select_leader, short_term_evidence_rows

    legacy = _legacy()
    last_session = _last_completed_session()
    rows = legacy.build_tracker_short_horizon_rows(
        _current_short_horizon_setups(), reference_date=last_session
    )
    assert rows and rows[0]["latest_measured_session"] == last_session.isoformat(), rows

    verdict = select_leader(
        short_term_evidence_rows(rows),
        kind="swing_short_term",
        last_completed_session=last_session,
        min_n=6,
    )
    assert verdict.state == "leader", verdict.reason
    assert "trade_r_close_2d" in verdict.policy_line

    short_path = tmp_path / "short.csv"
    pd.DataFrame(rows).to_csv(short_path, index=False)
    monkeypatch.setattr(panel_module, "SHORT_HORIZON_FILE", short_path)
    monkeypatch.setattr(
        panel_module, "RECENT_SETUP_TYPE_STATS_FILE", tmp_path / "absent_recent.csv"
    )
    panel = panel_module.SetupTrackerPanel()
    try:
        html = panel_module._best_now_banner_html(panel)
    finally:
        panel.deleteLater()

    assert "Short-term (1-2d), leader" in html, html
    assert "discovery" not in html.lower(), html
    assert "carries no session" not in html, html


# ===========================================================================
# Advisory 2 - last_reliable_reading is reachable from the panel
# ===========================================================================


def test_the_panel_carries_its_last_fresh_verdict_into_a_stale_refresh(
    panel_module, tmp_path, monkeypatch
):
    import market_calendar

    last_session = _last_completed_session()
    stale_day = last_session
    for _ in range(5):
        stale_day = market_calendar.previous_session(stale_day)

    fresh_rows = [_evidence_row("post_earnings_52w_break", wins=24, losses=16)]
    stale_rows = [
        _evidence_row(
            "post_earnings_52w_break", wins=24, losses=16, session=stale_day.isoformat()
        )
    ]

    fresh_path = tmp_path / "fresh.csv"
    stale_path = tmp_path / "stale.csv"
    for path, rows in ((fresh_path, fresh_rows), (stale_path, stale_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", fresh_path)
    panel = panel_module.SetupTrackerPanel()
    try:
        first = panel_module._best_now_banner_html(panel)
        assert "post_earnings_52w_break" in first
        assert panel._last_fresh_verdicts["swing"].state == "leader"

        # The tracker stops writing: the same family, five sessions stale.
        panel_module.clear_setup_tracker_csv_cache()
        monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", stale_path)
        panel.refresh()
        second = panel_module._best_now_banner_html(panel)
    finally:
        panel.deleteLater()

    assert "last reliable reading" in second.lower(), second
    assert "post_earnings_52w_break" in second, second


def test_the_card_and_the_banner_agree_on_the_stale_path_too(
    panel_module, tmp_path, monkeypatch
):
    """Re-review blocker 1, the fresh-then-stale sequence on BOTH surfaces.

    The card computed its own verdict with no `previous`, so after a stale
    refresh it printed "no clear leader ... discovery only" three lines above
    the banner's "last reliable reading" for the SAME family. The page now
    computes each horizon ONCE and hands the same objects to both renderers.
    """
    import market_calendar

    last_session = _last_completed_session()
    stale_day = last_session
    for _ in range(5):
        stale_day = market_calendar.previous_session(stale_day)

    fresh_rows = [_evidence_row("post_earnings_52w_break", wins=24, losses=16)]
    stale_rows = [
        _evidence_row(
            "post_earnings_52w_break", wins=24, losses=16, session=stale_day.isoformat()
        )
    ]
    fresh_path = tmp_path / "fresh.csv"
    stale_path = tmp_path / "stale.csv"
    for path, rows in ((fresh_path, fresh_rows), (stale_path, stale_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", fresh_path)
    panel = panel_module.SetupTrackerPanel()
    try:
        first = panel_module._summary_html(panel)
        assert "post_earnings_52w_break" in first

        panel_module.clear_setup_tracker_csv_cache()
        monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", stale_path)
        panel.refresh()
        summary = panel_module._summary_html(panel)
    finally:
        panel.deleteLater()

    # ONE page, ONE verdict: the card cannot say "discovery only" while the
    # banner says "last reliable reading" about the same family.
    assert "last reliable reading" in summary.lower(), summary
    assert "discovery only" not in summary, summary
    assert summary.count("post_earnings_52w_break") >= 2, summary


# ===========================================================================
# Advisory 3 - min_n binds the stale and undated discovery pools too
# ===========================================================================


def test_min_n_binds_the_stale_pool_so_a_thin_row_is_never_a_stale_discovery():
    """`thin_and_stale` (2-1) has the better Wilson bound of the two stale rows
    at 2/3 vs 8/12, and it is under the floor of six. The row shown as the stale
    discovery must be `fat_and_stale` - a floor that only holds when the clock
    is right is not a floor."""
    import market_calendar

    from working_lately import select_leader

    last_session = _last_completed_session()
    stale_day = last_session
    for _ in range(6):
        stale_day = market_calendar.previous_session(stale_day)

    rows = [
        _evidence_row("thin_and_stale", wins=2, losses=1, session=stale_day.isoformat()),
        _evidence_row("fat_and_stale", wins=8, losses=4, session=stale_day.isoformat()),
    ]
    verdict = select_leader(
        rows, kind="swing_short_term", last_completed_session=last_session, min_n=6
    )
    assert verdict.state == "no_evidence"
    assert verdict.coverage["discovery_reason"] == "not_fresh"
    assert verdict.coverage["discovery_leader"]["setup_family"] == "fat_and_stale"
    assert verdict.coverage["under_floor"] == 1
    assert verdict.coverage["not_fresh"] == 1


def test_the_floor_is_judged_before_the_clock():
    """A family with three samples is under the floor whether or not its
    evidence is fresh. Telling the reader "not fresh" about three samples
    answers a question they did not ask.

    The row is built EXPLICITLY undated rather than through `_evidence_row`,
    whose `session or <last completed>` default silently dated it and made this
    test unable to fail on the old code (re-review advisory 2).
    """
    from working_lately import select_leader

    rows = [dict(_evidence_row("three_samples", wins=2, losses=1), latest_measured_session="")]
    assert rows[0]["latest_measured_session"] == "", "the fixture must be genuinely undated"
    verdict = select_leader(
        rows, kind="swing_short_term", last_completed_session=_last_completed_session(), min_n=6
    )
    assert verdict.state == "no_evidence"
    assert verdict.coverage["discovery_reason"] == "floor", verdict.reason
    assert "floor" in verdict.reason.lower(), verdict.reason
    assert "not fresh" not in verdict.reason.lower(), verdict.reason
    assert verdict.coverage["discovery_leader"]["setup_family"] == "three_samples"


# ===========================================================================
# Advisories 4, 5, 6, 8 - the sentence says what it measured, over what
# ===========================================================================


def _type_row(family: str, *, wins: int, losses: int, expired: int = 0, side: str = "LONG") -> dict:
    n = wins + losses
    return {
        "side": side,
        "setup_family": family,
        "closed_setups": str(n),
        "tradeable_setups": str(n),
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "1",
        "n_unmeasured": "0",
        "n_pending": "2",
        "outcome_kind": "trade_r_representative_exit",
        "n_expired_unmeasured": str(expired),
    }


def test_the_population_sentence_names_its_window_and_its_freshness_rule(panel_module):
    sentence = panel_module.setup_type_population_sentence(
        [_type_row("a", wins=5, losses=3), _type_row("b", wins=2, losses=2)]
    )
    assert "ALL HISTORY" in sentence, sentence
    assert "Last 30 Days" in sentence, sentence
    assert "trade_r_representative_exit" in sentence, sentence
    # Advisory 8: an empty clause is not a statement.
    assert "0 expired unmeasured" in sentence, sentence


def test_the_banner_states_the_freshness_rule_in_words(panel_module, tmp_path, monkeypatch):
    """...and states it PER KIND, because the two surfaces are dated differently.

    Re-check advisory 1: one sentence for both made whichever surface it did not
    describe say something false. The swing rows are ENTRY-dated until ST4's
    `representative_exit_date` lands; the 2-session rows carry the measured bar.
    """
    from working_lately import freshness_sentence

    assert "entry-dated inside" in freshness_sentence("swing")
    assert "measured inside" in freshness_sentence("swing_short_term")
    # An unknown kind takes the conservative reading, never the flattering one.
    assert "entry-dated inside" in freshness_sentence("something_new")

    rows = [_evidence_row("supported", wins=24, losses=16)]
    csv_path = tmp_path / "recent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", csv_path)
    panel = panel_module.SetupTrackerPanel()
    try:
        html = panel_module._best_now_banner_html(panel)
    finally:
        panel.deleteLater()

    # The SWING line is the one on screen here, so it must say entry-dated -
    # and must not claim its evidence was measured two sessions ago.
    assert freshness_sentence("swing") in html, html
    assert freshness_sentence("swing_short_term") not in html, html


def test_an_exported_zero_count_is_a_count_and_not_a_missing_column(panel_module):
    """Advisory 6. A family that went 0-0 exports integer zeros, and a
    truthiness check on the raw cell reads them as "never exported"."""
    zeros = [
        {
            "side": "LONG",
            "setup_family": "graded_nothing",
            "closed_setups": "0",
            "n_wins": 0,
            "n_losses": 0,
            "n_flats": 0,
            "n_unmeasured": 3,
            "n_pending": 0,
            "outcome_kind": "trade_r_representative_exit",
        }
    ]
    sentence = panel_module.setup_type_population_sentence(zeros)
    assert "no win/loss counts in this export yet" not in sentence, sentence
    assert "3 closed-unmeasured" in sentence, sentence

    # ...and a row that truly has no columns still refuses.
    missing = [{"side": "LONG", "setup_family": "old_export", "closed_setups": "5"}]
    assert "no win/loss counts in this export yet" in panel_module.setup_type_population_sentence(missing)

    blank = [dict(missing[0], n_wins="", n_losses="")]
    assert "no win/loss counts in this export yet" in panel_module.setup_type_population_sentence(blank)


# ===========================================================================
# Advisory 7 - a flat is measured, and it is not in n
# ===========================================================================


def test_a_short_term_no_clear_leader_is_rendered_and_not_called_empty(
    panel_module, tmp_path, monkeypatch
):
    """Re-check blocker: twelve eligible 2-session families whose top two bounds
    are inside the declared margin.

    The banner used to guard its short-term block on "a discovery row OR a
    leader" and otherwise print "not enough 2-session samples yet (accrues
    automatically each scan)" - which on today's live state (12 eligible, top
    two 0.001 apart) was FALSE and contradicted the card three lines above
    saying "no clear leader". A renderer that has a verdict renders the verdict.
    """
    import pandas as pd

    from research_explanations import build_plain_english_whats_working
    from working_lately import select_leader, short_term_evidence_rows

    session = _last_completed_session().isoformat()
    rows = []
    for index, (wins, losses) in enumerate([(30, 20), (33, 22)]):
        rows.append(
            {
                "side": "LONG" if index == 0 else "SHORT",
                "namespace": "live",
                "setup_family": f"neck_and_neck_{index}",
                "tracked_setups": wins + losses,
                "samples_2d": wins + losses,
                "avg_r_2d": 0.4 + index * 0.05,
                "win_rate_2d": wins / (wins + losses),
                "n_wins": wins,
                "n_losses": losses,
                "n_flats": 0,
                "n_unmeasured": 0,
                "outcome_kind": "trade_r_close_2d",
                "horizon_basis": "2 sessions after entry, close to close",
                "latest_measured_session": session,
            }
        )

    verdict = select_leader(
        short_term_evidence_rows(rows),
        kind="swing_short_term",
        last_completed_session=_last_completed_session(),
        min_n=panel_module.SHORT_TERM_MIN_SAMPLES,
    )
    assert verdict.state == "no_clear_leader", verdict.reason
    assert verdict.coverage["discovery_leader"] is None
    assert verdict.leader is None

    short_path = tmp_path / "short.csv"
    pd.DataFrame(rows).to_csv(short_path, index=False)
    monkeypatch.setattr(panel_module, "SHORT_HORIZON_FILE", short_path)
    monkeypatch.setattr(
        panel_module, "RECENT_SETUP_TYPE_STATS_FILE", tmp_path / "absent_recent.csv"
    )
    panel = panel_module.SetupTrackerPanel()
    try:
        html = panel_module._best_now_banner_html(panel)
        summary = panel_module._summary_html(panel)
    finally:
        panel.deleteLater()

    assert "Short-term (1-2d), no clear leader" in html, html
    assert "not enough 2-session samples yet" not in html, html
    assert "neck_and_neck_0" in html and "neck_and_neck_1" in html, html

    # ...and the card says the same thing, on the same page.
    assert "no clear leader" in summary.lower(), summary
    assert "not enough 2-session samples yet" not in summary, summary

    card = build_plain_english_whats_working(short_term_rows=rows, verdicts={"swing_short_term": verdict})
    bullet = next(text for text in card["bullets"] if "first two sessions" in text)
    assert "no clear leader" in bullet.lower(), bullet


def test_an_old_discovery_row_is_called_old_and_not_thin(panel_module, tmp_path, monkeypatch):
    """Advisory 1. A row kept out for being OLD has the evidence - it is simply
    not current - and "thin" names the wrong gate. Both renderers say it the
    same way, from the same helper."""
    import market_calendar

    from working_lately import discovery_basis_phrase
    from research_explanations import build_plain_english_whats_working

    assert discovery_basis_phrase("not_fresh") == "leading on older evidence"
    assert discovery_basis_phrase("no_session") == "leading on undated evidence"
    assert discovery_basis_phrase("floor") == "leading on thin evidence"

    last_session = _last_completed_session()
    stale_day = last_session
    for _ in range(5):
        stale_day = market_calendar.previous_session(stale_day)
    rows = [
        _evidence_row("old_but_measured", wins=24, losses=16, session=stale_day.isoformat())
    ]

    # The card (no previous verdict to carry, so it shows the discovery row).
    plain = build_plain_english_whats_working(recent_rows=rows)
    bullet = next(text for text in plain["bullets"] if "recently closed swings" in text)
    assert "older evidence" in bullet, bullet
    assert "thin evidence" not in bullet, bullet

    # ...and the banner.
    csv_path = tmp_path / "recent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", csv_path)
    panel = panel_module.SetupTrackerPanel()
    try:
        html = panel_module._best_now_banner_html(panel)
    finally:
        panel.deleteLater()
    assert "older evidence" in html, html
    assert "thin evidence" not in html, html
    # The no-session sentence belongs only to a row that has no session.
    assert "carries no session" not in html, html
    assert "Swing (30d realized), discovery" in html, html


def test_a_flat_reaches_the_headline_record_and_stays_out_of_n(panel_module):
    rows = panel_module._recent_type_headline_rows(
        [_evidence_row("with_flats", wins=6, losses=4, flats=5)]
    )
    headline = rows[0]["win_rate_headline"]
    assert "n=10" in headline, headline
    assert "60%" in headline, headline

    from swing_headline import headline_from_counts

    record = headline_from_counts("x", wins=6, losses=4, flats=5)
    assert record.n == 10
    assert record.flats == 5
    assert record.as_row()["flats"] == 5
    # The default is unchanged for every caller that does not count flats.
    assert headline_from_counts("x", wins=6, losses=4).flats == 0
