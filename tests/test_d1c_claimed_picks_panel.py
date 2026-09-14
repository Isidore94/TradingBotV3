"""Packet D1C-A items 5-8 - the claimed pick in Master AVWAP Setups.

Trader, 2026-09-14:

    "Provide independently selectable FAV, HC, and My liked trades filters. Let
    me select any combination. Rank manually claimed picks using the existing
    setup-ranking system. A like alone must not invent a score or grant FAV/HC
    status. Show missing measurements honestly and keep the pick visible while
    they are unavailable. A setup belonging to several buckets appears once
    with all its labels. Preserve distinct setups and directions for the same
    symbol. Claimed picks must survive refreshes, rescans and restarts."

The panel tests drive `MasterAvwapPanel.refresh_from_reports` over a temp focus
feed and a temp claims store; the merge tests drive the pure
`ui.services.claimed_setup_rows.merge_claims`. No hand-built model rows stand
in for either.

Seams this file requires of the builder (both named by the packet):

* ``MasterAvwapPanel(..., claimed_picks_path=...)`` - the same shape as the
  existing ``review_events_path``, so "survives a restart" can be proven by
  building a SECOND panel on the same temp file;
* the five chips: ``panel.bucket_chips`` (a dict keyed by bucket key plus
  ``"all"``), ``panel.set_bucket_chips(keys)`` and
  ``panel.active_bucket_chip_keys()``, persisted under
  ``qt_setups_bucket_chips``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


CHIP_FAV = "favorite_setup"
CHIP_HC = "high_conviction"
CHIP_NEAR = "near_favorite_zone"
CHIP_LIKED = "claimed_like"
CHIP_ALL = "all"


# ---------------------------------------------------------------------------
# the feed the "scan" produced, built through the REAL focus-payload reader so
# the HC+FAV fold under one `opportunity_identity` is the real one
# ---------------------------------------------------------------------------
def _scan_payload() -> dict:
    def entry(symbol, side, bucket, family, score, **extra):
        row = {
            "symbol": symbol,
            "side": side,
            "priority_bucket": bucket,
            "setup_family": family,
            "priority_score": score,
            "anchor_date": "2026-07-18",
            "last_trade_date": "2026-09-11",
        }
        row.update(extra)
        return row

    return {
        "data_date": "2026-09-11",
        # BOTH entries are the same opportunity - symbol, side, family, anchor -
        # so the feed folds them into ONE row carrying both labels.
        "high_conviction": [
            entry("MSFT", "LONG", "high_conviction", "avwap_band_bounce", 88.0)
        ],
        "favorites": [
            entry("MSFT", "LONG", "favorite_setup", "avwap_band_bounce", 88.0),
            entry("TSLA", "SHORT", "favorite_setup", "avwap_breakout", 81.0),
        ],
        "near_favorite_zones": [
            entry("AMD", "LONG", "near_favorite_zone", "sma_breakout", 64.0)
        ],
    }


def _scan_rows():
    from ui.services.data_feed import _rows_from_focus_payload

    return _rows_from_focus_payload(_scan_payload())


def _claim(
    symbol="ZZZZ",
    side="LONG",
    setup_id="avwap_band_bounce",
    *,
    known=None,
    claim_at="2026-09-14T08:25:00-07:00",
):
    return {
        "schema": "claimed_pick_v1",
        "action": "claim",
        "symbol": symbol,
        "side": side,
        "horizon": "d1",
        "claimed_setup_id": setup_id,
        "claim_at": claim_at,
        "claim_at_utc": "2026-09-14T15:25:00+00:00",
        "session_date": "2026-09-14",
        "source": "chart_review:d1_flag_long",
        "annotation_ref": "a1",
        "known_at_claim": {} if known is None else dict(known),
        "note": "",
    }


def _write_claims(path: Path, claims) -> None:
    import json

    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in claims),
        encoding="utf-8",
    )


def _build_panel(tmp_path, monkeypatch, *, rows=None, claims_path=None, settings=None):
    """A panel over a fixed report and a temp claims store."""
    from ui.panels import master_avwap_panel as panel_module

    rows = _scan_rows() if rows is None else rows
    meta = {
        "rows": list(rows),
        "data_date": "2026-09-11",
        "source": "focus",
        "is_stale": False,
    }
    monkeypatch.setattr(
        panel_module, "load_latest_setup_rows_with_meta", lambda: dict(meta, rows=list(rows))
    )
    store = {} if settings is None else settings
    monkeypatch.setattr(
        panel_module,
        "get_local_setting",
        lambda key, default=None: store.get(key, default),
    )
    monkeypatch.setattr(
        panel_module, "save_local_setting", lambda key, value: store.__setitem__(key, value)
    )
    made = panel_module.MasterAvwapPanel(
        None,
        claimed_picks_path=(
            claims_path if claims_path is not None else tmp_path / "claimed_picks.jsonl"
        ),
    )
    monkeypatch.setattr(made, "_start_family_record_read", lambda *a, **k: None)
    monkeypatch.setattr(made, "_start_scan_freshness_read", lambda *a, **k: None)
    return made, store


def _symbols(rows) -> list[str]:
    return [row.symbol for row in rows]


def _row_named(panel, symbol: str, side: str = ""):
    for row in panel.model.rows():
        if row.symbol == symbol and (not side or row.side == side):
            return row
    raise AssertionError(f"{symbol} {side} is not in the table: {_symbols(panel.model.rows())}")


# ---------------------------------------------------------------------------
# item 5 - the row model
# ---------------------------------------------------------------------------
def test_a_claimed_row_is_labelled_my_liked_trade():
    from ui.models.setup import SETUP_BUCKET_LABELS, SetupRow

    assert SETUP_BUCKET_LABELS["claimed_like"] == "My liked trade"
    assert SetupRow(symbol="ZZZZ", bucket="claimed_like").bucket_label == "My liked trade"


def test_bucket_keys_unions_the_bucket_with_the_keys_the_fold_recorded():
    """One row can belong to several buckets, so the KEY is a set."""
    from ui.models.setup import SetupRow

    row = SetupRow(
        symbol="MSFT",
        bucket="High_Conviction",
        raw={"bucket_keys": ["favorite_setup", "", "  CLAIMED_LIKE "]},
    )

    assert row.bucket_keys == {"high_conviction", "favorite_setup", "claimed_like"}
    assert SetupRow(symbol="X", bucket="").bucket_keys == set()


def test_the_feeds_fold_records_both_buckets_not_just_both_labels():
    """`_merge_classification_badges` merged LABELS only; nothing recorded that
    an HC row is also a FAV, so no filter could ever ask."""
    rows = {(row.symbol, row.side): row for row in _scan_rows()}

    msft = rows[("MSFT", "LONG")]
    assert msft.bucket_keys == {"high_conviction", "favorite_setup"}
    assert "High Conviction" in msft.bucket_display
    assert "Favorite" in msft.bucket_display
    assert len([r for r in _scan_rows() if r.symbol == "MSFT"]) == 1, "one row, both labels"


def test_merge_claims_adds_a_label_to_the_scan_row_it_matches():
    from ui.services.claimed_setup_rows import merge_claims

    rows = _scan_rows()
    before = len(rows)

    merged = merge_claims(rows, [_claim("MSFT", "LONG", "avwap_band_bounce")])

    assert len(merged) == before, "a matched claim adds no second row"
    msft = next(row for row in merged if row.symbol == "MSFT")
    assert msft.bucket_keys == {"high_conviction", "favorite_setup", "claimed_like"}
    assert "My liked trade" in msft.bucket_display
    assert "High Conviction" in msft.bucket_display and "Favorite" in msft.bucket_display
    assert msft.score == 88.0, "the scan's measurement is untouched"
    assert msft.bucket == "high_conviction", (
        "a like grants no FAV/HC status and takes none away"
    )


def test_merge_claims_builds_a_new_row_for_a_pick_the_scan_does_not_carry():
    from ui.services.claimed_setup_rows import merge_claims

    claim = _claim("ZZZZ", "LONG", "avwap_band_bounce")
    merged = merge_claims(_scan_rows(), [claim])

    row = next(r for r in merged if r.symbol == "ZZZZ")
    assert row.bucket == "claimed_like"
    assert row.bucket_keys == {"claimed_like"}
    assert row.side == "LONG"
    assert row.score is None, "a like must not invent a score"
    assert row.expected_r is None, "and must not invent an expected R"
    assert row.source == "claim"
    assert row.setup_tags, "the row names the setup the trader claimed"
    assert row.raw["claimed_setup_id"] == "avwap_band_bounce"
    assert row.raw["claim_at"] == claim["claim_at"]
    assert row.raw["source"] == claim["source"]
    assert row.raw["known_at_claim"] == {}
    assert row.raw["bucket_keys"] == ["claimed_like"]


def test_a_claimed_only_row_carries_forward_what_the_desk_knew(monkeypatch):
    """`known_at_claim` is what the pick shows instead of a hole - both as the
    honest record and as the input the point system reads."""
    from ui.services.claimed_setup_rows import merge_claims

    known = {"expected_r": 1.4, "setup_family": "avwap_band_bounce", "priority_score": 72.5}
    merged = merge_claims([], [_claim("ZZZZ", "LONG", "avwap_band_bounce", known=known)])

    row = merged[0]
    assert row.expected_r == 1.4
    assert row.score is None, (
        "the SCAN never scored this name; `priority_score` at claim time is not a scan score"
    )
    assert row.raw["known_at_claim"] == known
    assert row.raw.get("setup_family") == "avwap_band_bounce", (
        "the measurements the claim carried are readable by the point system"
    )
    assert row.raw.get("expected_r") == 1.4


def test_merge_claims_never_folds_a_claim_the_registry_cannot_name():
    """`none_of_these` maps to no family, so it matches no scan row."""
    from ui.services.claimed_setup_rows import merge_claims

    merged = merge_claims(_scan_rows(), [_claim("MSFT", "LONG", "none_of_these")])

    msft_rows = [row for row in merged if row.symbol == "MSFT"]
    assert len(msft_rows) == 2, "it becomes its own row rather than labelling MSFT's"
    scan_row = next(row for row in msft_rows if row.source != "claim")
    assert "claimed_like" not in scan_row.bucket_keys


def test_merge_claims_keeps_two_sides_and_two_setups_apart():
    from ui.services.claimed_setup_rows import merge_claims

    merged = merge_claims(
        [],
        [
            _claim("ZZZZ", "LONG", "avwap_band_bounce"),
            _claim("ZZZZ", "SHORT", "avwap_band_bounce"),
            _claim("ZZZZ", "LONG", "avwap_breakout"),
        ],
    )

    assert len(merged) == 3, [(r.symbol, r.side, r.raw.get("claimed_setup_id")) for r in merged]
    assert {(r.side, r.raw["claimed_setup_id"]) for r in merged} == {
        ("LONG", "avwap_band_bounce"),
        ("SHORT", "avwap_band_bounce"),
        ("LONG", "avwap_breakout"),
    }


def test_merge_claims_does_not_mutate_the_rows_it_was_given():
    from ui.services.claimed_setup_rows import merge_claims

    rows = _scan_rows()
    snapshot = [(row.symbol, row.bucket, sorted(row.bucket_keys)) for row in rows]

    merge_claims(rows, [_claim("MSFT", "LONG", "avwap_band_bounce")])

    assert [(r.symbol, r.bucket, sorted(r.bucket_keys)) for r in rows] == snapshot, (
        "the merge is pure - the caller's list is the scan's, unchanged"
    )


# ---------------------------------------------------------------------------
# item 5 - the panel: placement, refresh and restart
# ---------------------------------------------------------------------------
def test_a_claimed_pick_absent_from_the_scan_shows_in_the_setups_table(
    tmp_path, monkeypatch
):
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(claims, [_claim("ZZZZ", "LONG", "avwap_band_bounce")])
    panel, _settings = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()

        assert "ZZZZ" in _symbols(panel.model.rows()), _symbols(panel.model.rows())
        row = _row_named(panel, "ZZZZ")
        assert row.bucket == "claimed_like"
        assert row.side == "LONG"
        assert row.raw["claim_at"] == "2026-09-14T08:25:00-07:00"
        assert row.raw["source"] == "chart_review:d1_flag_long"
        # ...and a second refresh over the SAME unchanged report keeps it.
        panel.refresh_from_reports()
        assert "ZZZZ" in _symbols(panel.model.rows())
    finally:
        panel.deleteLater()


def test_the_claimed_pick_is_still_there_after_a_restart(tmp_path, monkeypatch):
    """The FILE is the persistence: a second panel on the same store sees it."""
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(claims, [_claim("ZZZZ", "LONG", "avwap_band_bounce")])
    first, _ = _build_panel(tmp_path, monkeypatch)
    first.refresh_from_reports()
    assert "ZZZZ" in _symbols(first.model.rows())
    first.deleteLater()

    second, _ = _build_panel(tmp_path, monkeypatch)
    try:
        second.refresh_from_reports()
        assert "ZZZZ" in _symbols(second.model.rows())
    finally:
        second.deleteLater()


def test_a_dropped_claim_leaves_the_table_on_the_next_refresh(tmp_path, monkeypatch):
    import claimed_picks

    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(claims, [_claim("ZZZZ", "LONG", "avwap_band_bounce")])
    panel, _ = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()
        assert "ZZZZ" in _symbols(panel.model.rows())

        claimed_picks.record_drop("ZZZZ", "LONG", "avwap_band_bounce", path=claims)
        panel.refresh_from_reports()

        assert "ZZZZ" not in _symbols(panel.model.rows())
    finally:
        panel.deleteLater()


def test_the_claimed_rows_score_is_blank_and_its_points_say_what_was_not_measured(
    tmp_path, monkeypatch
):
    """"Show missing measurements honestly and keep the pick visible."

    A claimed name the scan never measured scores +10 - clean path and nothing
    else - and the tooltip names every part that was not measured. A formula
    that filled the holes with zeros and printed a confident total would fail
    this.
    """
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(claims, [_claim("ZZZZ", "LONG", "avwap_band_bounce")])
    panel, _ = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()
        keys = [key for key, _label in panel.model.COLUMNS]
        index = panel.model.rows().index(_row_named(panel, "ZZZZ"))

        score_cell = panel.model.data(panel.model.index(index, keys.index("score")))
        points_cell = panel.model.data(panel.model.index(index, keys.index("points")))
        bucket_cell = panel.model.data(panel.model.index(index, keys.index("bucket")))
        tooltip = panel.model.data(
            panel.model.index(index, keys.index("points")),
            Qt.ItemDataRole.ToolTipRole,
        )

        assert str(score_cell).strip() in ("", "-"), (
            f"a like must not invent a score; the cell read {score_cell!r}"
        )
        assert points_cell == "+10", points_cell
        assert "family ungraded" in tooltip
        assert "unmeasured" in tooltip
        assert bucket_cell == "My liked trade"
    finally:
        panel.deleteLater()


def test_a_claimed_scan_row_keeps_its_own_labels_and_gains_one(tmp_path, monkeypatch):
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(claims, [_claim("MSFT", "LONG", "avwap_band_bounce")])
    panel, _ = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()

        assert _symbols(panel.model.rows()).count("MSFT") == 1
        row = _row_named(panel, "MSFT")
        assert row.bucket_keys == {"high_conviction", "favorite_setup", "claimed_like"}
        for label in ("High Conviction", "Favorite", "My liked trade"):
            assert label in row.bucket_display, row.bucket_display
    finally:
        panel.deleteLater()


def test_the_same_symbol_claimed_short_under_another_setup_is_a_second_row(
    tmp_path, monkeypatch
):
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(
        claims,
        [
            _claim("MSFT", "LONG", "avwap_band_bounce"),
            _claim("MSFT", "SHORT", "avwap_breakout"),
        ],
    )
    panel, _ = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()

        msft = [row for row in panel.model.rows() if row.symbol == "MSFT"]
        assert len(msft) == 2, [(r.side, r.bucket) for r in msft]
        sides = {row.side: row for row in msft}
        assert sides["LONG"].bucket == "high_conviction"
        assert sides["SHORT"].bucket == "claimed_like"
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# item 7 - five chips, any combination
# ---------------------------------------------------------------------------
@pytest.fixture
def chip_panel(tmp_path, monkeypatch):
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(
        claims,
        [
            _claim("ZZZZ", "LONG", "avwap_band_bounce"),
            _claim("MSFT", "LONG", "avwap_band_bounce"),
        ],
    )
    panel, settings = _build_panel(tmp_path, monkeypatch)
    panel.refresh_from_reports()
    yield panel, settings
    panel.deleteLater()


def test_the_strip_offers_five_independently_checkable_chips(chip_panel):
    panel, _ = chip_panel

    assert set(panel.bucket_chips) == {CHIP_FAV, CHIP_HC, CHIP_NEAR, CHIP_LIKED, CHIP_ALL}
    for key, chip in panel.bucket_chips.items():
        assert chip.isCheckable(), key


def test_fav_and_liked_together_show_the_union_not_the_intersection(chip_panel):
    """MSFT is HC+FAV+claimed, TSLA is FAV only, ZZZZ is claimed only, AMD is Near."""
    panel, _ = chip_panel

    panel.set_bucket_chips({CHIP_FAV, CHIP_LIKED})

    assert set(_symbols(panel.filtered_rows())) == {"MSFT", "TSLA", "ZZZZ"}


def test_hc_alone_hides_the_claimed_only_row(chip_panel):
    panel, _ = chip_panel

    panel.set_bucket_chips({CHIP_HC})

    assert set(_symbols(panel.filtered_rows())) == {"MSFT"}


def test_liked_alone_shows_every_claimed_row_including_the_scan_one(chip_panel):
    panel, _ = chip_panel

    panel.set_bucket_chips({CHIP_LIKED})

    assert set(_symbols(panel.filtered_rows())) == {"MSFT", "ZZZZ"}


def test_all_shows_everything_and_unchecks_the_others(chip_panel):
    panel, _ = chip_panel
    panel.set_bucket_chips({CHIP_FAV, CHIP_LIKED})

    panel.bucket_chips[CHIP_ALL].click()

    assert set(_symbols(panel.filtered_rows())) == {"MSFT", "TSLA", "AMD", "ZZZZ"}
    for key in (CHIP_FAV, CHIP_HC, CHIP_NEAR, CHIP_LIKED):
        assert not panel.bucket_chips[key].isChecked(), key


def test_checking_any_bucket_chip_unchecks_all(chip_panel):
    panel, _ = chip_panel
    panel.bucket_chips[CHIP_ALL].click()

    panel.bucket_chips[CHIP_NEAR].click()

    assert panel.bucket_chips[CHIP_NEAR].isChecked()
    assert not panel.bucket_chips[CHIP_ALL].isChecked()
    assert set(_symbols(panel.filtered_rows())) == {"AMD"}


def test_no_chip_checked_reads_as_all(chip_panel):
    panel, _ = chip_panel

    panel.set_bucket_chips(set())

    assert set(_symbols(panel.filtered_rows())) == {"MSFT", "TSLA", "AMD", "ZZZZ"}


@pytest.mark.parametrize(
    "old_value, expected",
    [
        ("fav_hc_near", {CHIP_FAV, CHIP_HC, CHIP_NEAR, CHIP_LIKED}),
        ("fav_hc", {CHIP_FAV, CHIP_HC, CHIP_LIKED}),
        ("all", set()),
    ],
)
def test_the_old_exclusive_setting_migrates_once(tmp_path, monkeypatch, old_value, expected):
    panel, settings = _build_panel(
        tmp_path, monkeypatch, settings={"qt_setups_bucket_filter": old_value}
    )
    try:
        assert panel.active_bucket_chip_keys() == expected, old_value
        assert sorted(settings.get("qt_setups_bucket_chips") or []) == sorted(expected), (
            "the migration is written once, under the new key"
        )
    finally:
        panel.deleteLater()


def test_with_no_setting_at_all_the_default_is_fav_hc_near_and_liked(tmp_path, monkeypatch):
    panel, _ = _build_panel(tmp_path, monkeypatch, settings={})
    try:
        assert panel.active_bucket_chip_keys() == {CHIP_FAV, CHIP_HC, CHIP_NEAR, CHIP_LIKED}
    finally:
        panel.deleteLater()


def test_the_chip_selection_round_trips_through_a_panel_rebuild(tmp_path, monkeypatch):
    settings: dict = {}
    first, _ = _build_panel(tmp_path, monkeypatch, settings=settings)
    first.set_bucket_chips({CHIP_HC, CHIP_LIKED})
    assert sorted(settings["qt_setups_bucket_chips"]) == sorted([CHIP_HC, CHIP_LIKED])
    first.deleteLater()

    second, _ = _build_panel(tmp_path, monkeypatch, settings=settings)
    try:
        assert second.active_bucket_chip_keys() == {CHIP_HC, CHIP_LIKED}
    finally:
        second.deleteLater()


def test_a_row_passes_when_any_of_its_buckets_is_selected():
    """The proxy's row test is a set intersection, not a single-key compare."""
    from ui.models.setup import SetupRow
    from ui.models.setup_table_model import SetupFilterProxyModel, SetupTableModel

    model = SetupTableModel(
        [
            SetupRow(
                symbol="MSFT",
                side="LONG",
                bucket="high_conviction",
                raw={"bucket_keys": ["favorite_setup", "claimed_like"]},
            ),
            SetupRow(symbol="AMD", side="LONG", bucket="near_favorite_zone", raw={}),
        ]
    )
    proxy = SetupFilterProxyModel()
    proxy.setSourceModel(model)

    proxy.set_filters(buckets={"claimed_like"})

    shown = [proxy.data(proxy.index(r, 2)) for r in range(proxy.rowCount())]
    assert shown == ["MSFT"], (
        "the row test must be `row.bucket_keys & selected`, not `row.bucket in selected`"
    )


# ---------------------------------------------------------------------------
# item 6 - ranking
# ---------------------------------------------------------------------------
def _set_points_switch(on: bool) -> None:
    import project_paths
    import setup_points

    project_paths.save_local_setting(setup_points.SETTING_KEY, bool(on))
    project_paths.invalidate_local_settings_cache()


def test_the_claimed_bucket_is_one_the_point_system_ranks():
    import setup_points

    assert "claimed_like" in setup_points.RANKED_BUCKETS


def test_the_points_switch_lifts_a_claimed_pick_above_a_weaker_favourite(
    tmp_path, monkeypatch
):
    """The SAME points as FAV/HC/Near - the existing ranking system, reused.

    FAVR: family bound 0.6 x 40 = 24, five blocking levels take S/R to -10 -> +14.
    The claimed pick: 24 + expected R 1.4 (capped at +10) + clean path 10 -> +44.
    So with the switch ON the claimed row sorts FIRST; with it OFF the arrival
    order stands and the favourite is first.
    """
    from ui.models.setup import SetupRow

    scan = [
        SetupRow(
            symbol="FAVR",
            side="LONG",
            score=90.0,
            bucket="favorite_setup",
            raw={"setup_family": "alpha", "hv_level_blocking_count": 5},
        )
    ]
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(
        claims,
        [
            _claim(
                "ZZZZ",
                "LONG",
                "avwap_band_bounce",
                known={"setup_family": "alpha", "expected_r": 1.4},
            )
        ],
    )

    def _order(switch_on: bool) -> list[str]:
        _set_points_switch(switch_on)
        panel, _ = _build_panel(tmp_path, monkeypatch, rows=scan)
        try:
            panel.model.set_family_records({"alpha": {"win_rate_lb": 0.6}})
            panel.refresh_from_reports()
            return _symbols(panel.model.rows())
        finally:
            panel.deleteLater()

    try:
        off = _order(False)
        on = _order(True)
    finally:
        _set_points_switch(False)

    assert sorted(off) == sorted(on) == ["FAVR", "ZZZZ"], (off, on)
    assert on == ["ZZZZ", "FAVR"], on
    assert off == ["FAVR", "ZZZZ"], off


def test_with_the_switch_off_claimed_rows_follow_the_scan_newest_first(
    tmp_path, monkeypatch
):
    _set_points_switch(False)
    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(
        claims,
        [
            _claim("OLDER", "LONG", "avwap_band_bounce", claim_at="2026-09-14T07:00:00-07:00"),
            _claim("NEWER", "LONG", "avwap_breakout", claim_at="2026-09-14T11:30:00-07:00"),
        ],
    )
    panel, _ = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()
        shown = _symbols(panel.model.rows())

        claimed_only = [s for s in shown if s in {"OLDER", "NEWER"}]
        assert claimed_only == ["NEWER", "OLDER"], shown
        assert shown.index("NEWER") > shown.index("MSFT"), (
            "claimed-only rows follow the scan rows"
        )
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# item 8 - drop my claim
# ---------------------------------------------------------------------------
def test_the_claimed_rows_menu_offers_drop_my_claim(tmp_path, monkeypatch):
    """Ending a claim is a claim-store action, never a Focus one.

    This panel is built with no focus service, exactly as the other tests here
    are, so the verb must NOT live inside the `focus_service is not None`
    block that gates "Add to Swing Focus Picks": a claim writes nothing to
    Focus (D1C0 decision 1) and dropping one must not need Focus to exist.
    """
    import claimed_picks

    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(claims, [_claim("ZZZZ", "LONG", "avwap_band_bounce")])
    panel, _ = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()
        actions = {label: callback for label, callback in panel.table._row_actions}
        assert "Drop my claim" in actions, sorted(actions)

        index = panel.model.rows().index(_row_named(panel, "ZZZZ"))
        proxy_index = panel.proxy.mapFromSource(panel.model.index(index, 0))
        actions["Drop my claim"](proxy_index)

        rows = claimed_picks.load_rows(claims)
        assert [row["action"] for row in rows] == ["claim", "drop"]
        assert "ZZZZ" not in _symbols(panel.model.rows()), (
            "the row leaves the table on the same click"
        )
    finally:
        panel.deleteLater()


def test_dropping_the_claim_on_a_scan_row_keeps_the_row_and_takes_the_label_off(
    tmp_path, monkeypatch
):
    """ADDED BY THE BUILDER: a claim on a name the SCAN also carries.

    The packet's own drop test uses a claimed-only row, which leaves the other
    half untested - and it is the half that can go wrong, because a labelled
    scan row has to carry the claimed setup id for the drop to know which pick
    to end. The scan's row is the scan's: dropping the claim takes the label
    off and leaves the row, its bucket and its score exactly where they were.
    """
    import claimed_picks

    claims = tmp_path / "claimed_picks.jsonl"
    _write_claims(claims, [_claim("MSFT", "LONG", "avwap_band_bounce")])
    panel, _ = _build_panel(tmp_path, monkeypatch)
    try:
        panel.refresh_from_reports()
        row = _row_named(panel, "MSFT")
        assert "claimed_like" in row.bucket_keys
        assert row.raw.get("claimed_setup_id") == "avwap_band_bounce"

        actions = {label: callback for label, callback in panel.table._row_actions}
        index = panel.model.rows().index(row)
        actions["Drop my claim"](panel.proxy.mapFromSource(panel.model.index(index, 0)))

        assert [r["action"] for r in claimed_picks.load_rows(claims)] == ["claim", "drop"]
        after = _row_named(panel, "MSFT")
        assert "claimed_like" not in after.bucket_keys
        assert after.bucket == "high_conviction", "the scan's row is the scan's"
        assert after.score == 88.0
        assert "My liked trade" not in after.bucket_display
    finally:
        panel.deleteLater()
