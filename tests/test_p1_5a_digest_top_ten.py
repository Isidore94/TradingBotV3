"""P1-5 5a: the digest's top ten spread across setups, and the setup-key label.

Display and ranking only: at most three per (family, side) first, then the
rest in the same rank order. The points order stays a switch. A row shows its
setup key's short label only when the scan stamped one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import autopilot_core as core  # noqa: E402
import setup_key_labels  # noqa: E402

#: family A has the best record, so a plain Wilson order puts every A pick first.
RECORDS = {
    "fam_a": {"wins": 60, "losses": 10},
    "fam_b": {"wins": 40, "losses": 20},
    "fam_c": {"wins": 30, "losses": 25},
}


def _pick(symbol, family, side="LONG", expected_r=1.0, raw=None):
    return {
        "symbol": symbol,
        "side": side,
        "bucket": "Favorite",
        "bucket_key": "favorite_setup",
        "family": family,
        "expected_r": expected_r,
        "raw": dict(raw or {}),
    }


def _fixture_picks():
    """Twelve A longs, three B longs, two C shorts, one A short."""
    picks = [_pick(f"A{i:02d}", "fam_a", expected_r=3.0 - i * 0.1) for i in range(12)]
    picks += [_pick(f"B{i}", "fam_b") for i in range(3)]
    picks += [_pick(f"C{i}", "fam_c", side="SHORT") for i in range(2)]
    picks += [_pick("S0", "fam_a", side="SHORT", expected_r=0.5)]
    return picks


@pytest.fixture(autouse=True)
def _points_switch_off(monkeypatch):
    import setup_points

    monkeypatch.setattr(setup_points, "rank_enabled", lambda: False)
    setup_key_labels.reset_cache_for_tests()
    yield
    setup_key_labels.reset_cache_for_tests()


# --- the pure cap -----------------------------------------------------------


def test_cap_takes_three_per_group_then_fills_in_rank_order():
    items = [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 5), ("a", 6), ("c", 7)]
    out = core.cap_ranked_items(items, lambda item: item[0], limit=6, per_group=3)
    assert out == [("a", 1), ("a", 2), ("a", 3), ("b", 5), ("c", 7), ("a", 4)]


def test_cap_without_a_limit_keeps_every_item_and_drops_nothing():
    items = [("a", n) for n in range(5)] + [("b", 9)]
    out = core.cap_ranked_items(items, lambda item: item[0], limit=None, per_group=3)
    assert sorted(out) == sorted(items)
    assert out[3] == ("b", 9)


def test_cap_with_fewer_items_than_the_limit_returns_them_all():
    items = [("a", 1), ("b", 2)]
    assert core.cap_ranked_items(items, lambda item: item[0], limit=10) == items


def test_the_group_is_family_and_side():
    assert core.digest_pick_group(_pick("X", "Fam A", side="long")) == ("fam_a", "LONG")
    assert core.digest_pick_group(_pick("Y", "fam_a", side="SHORT")) == ("fam_a", "SHORT")


# --- the digest fixture ----------------------------------------------------


def test_the_digest_top_ten_is_ten_names_across_at_least_three_families():
    top, label = core.rank_digest_picks(_fixture_picks(), RECORDS)
    assert label == core.SWING_ORDER_WILSON
    symbols = [pick["symbol"] for pick in top]
    assert len(symbols) == 10 and len(set(symbols)) == 10
    groups = [core.digest_pick_group(pick) for pick in top]
    assert len({family for family, _side in groups}) >= 3
    # The capped pass: three A longs (best first), then the other groups by rank.
    assert symbols[:3] == ["A00", "A01", "A02"]
    assert groups[:9].count(("fam_a", "LONG")) == 3
    # The fill pass: the next A long in rank order takes the last slot.
    assert symbols[9] == "A03"


def test_the_points_switch_still_decides_the_order(monkeypatch):
    import setup_points

    calls = []

    def fake_order(indexed, records):
        calls.append(len(indexed))
        return list(reversed(indexed)), core.SWING_ORDER_POINTS

    monkeypatch.setattr(core, "order_swing_picks", fake_order)
    top, label = core.rank_digest_picks(_fixture_picks(), RECORDS)
    assert label == core.SWING_ORDER_POINTS and calls
    groups = [core.digest_pick_group(pick) for pick in top]
    assert max(groups[:9].count(group) for group in set(groups[:9])) <= 3  # the cap still applies
    assert setup_points  # the real switch is untouched


def test_the_rendered_digest_states_the_rule_and_leads_with_three_per_group():
    top, _label = core.rank_digest_picks(_fixture_picks(), RECORDS)
    text = core.render_away_report(
        {
            "auto_mode": "AWAY",
            "swing_picks": top,
            "swing_family_records": RECORDS,
            "swing_data_current": True,
        }
    )
    body = text.split("== BEST SWING TRADES ==", 1)[1]
    ranked = [line for line in body.splitlines() if line.startswith("Ranked on:")]
    assert ranked and core.DIGEST_TOP_RULE in ranked[0]
    assert "max 3 per family+side" in ranked[0]
    numbered = [line for line in body.splitlines() if re.match(r"^[0-9]+[.] ", line)]
    assert len(numbered) == 10
    assert sum("fam a" in line and "(LONG)" in line for line in numbered[:9]) == 3


def test_a_pick_with_a_stamped_label_shows_it_and_one_without_shows_nothing():
    picks = [
        _pick("KEYD", "fam_a", raw={"permutation_label": "sma100_support|weekly_ema15_hold"}),
        _pick("BARE", "fam_b"),
    ]
    text = core.render_away_report(
        {"auto_mode": "AWAY", "swing_picks": picks, "swing_family_records": RECORDS}
    )
    keyd = next(line for line in text.splitlines() if "KEYD" in line and "(LONG)" in line)
    bare = next(line for line in text.splitlines() if "BARE" in line and "(LONG)" in line)
    assert keyd.endswith("| key sma100_support|weekly_ema15_hold")
    assert "| key" not in bare


# --- the service picks the top ten by rank, not by feed position -----------


def test_the_service_chooses_the_top_ten_by_rank_with_the_cap(monkeypatch):
    from collections import deque

    from ui.services import autopilot_service as svc_mod
    from ui.services.autopilot_service import AUTO_PROFILE_AWAY, AutopilotService

    service = AutopilotService.__new__(AutopilotService)
    service._enabled = True
    service._profile = AUTO_PROFILE_AWAY
    service._state = {}
    service._alerts_today = deque(maxlen=60)
    service._alert_symbols_today = deque(maxlen=60)
    service._log_lines = deque(maxlen=60)
    service._scorecard_line = ""
    service._outcome_coverage_line = ""
    service._evening_briefing_lines = []
    service._log = lambda *_a, **_k: None
    service._read_watchlists = lambda: ([], [])
    service.status_snapshot = lambda: {
        "ib_status": "connected", "regime": "x", "slots_done": [], "next_slot": ""
    }
    today = svc_mod.datetime.now().date().isoformat()
    picks = _fixture_picks()
    rows = [SimpleNamespace(**{k: v for k, v in p.items() if k != "raw"}, raw={}) for p in picks]
    service._load_swing_feed = lambda: {"data_date": today, "rows": rows, "source": "test"}
    service._read_auto_watchlist = lambda _path: []
    service._staged_pick_summary = lambda: {"long": [], "short": []}
    by_symbol = {p["symbol"]: p for p in picks}
    monkeypatch.setattr(core, "swing_pick_projection", lambda row: dict(by_symbol[row.symbol]))
    monkeypatch.setattr(core, "swing_family_read", lambda: (RECORDS, None))
    monkeypatch.setattr(core, "_sector_hidden_predicate", lambda _is_hidden=None: None)
    monkeypatch.setattr(svc_mod, "_working_lately_report_line", lambda: "")
    published: list[dict] = []
    monkeypatch.setattr(
        core, "publish_away_report", lambda payload: published.append(payload) or {"ok": False, "error": "t"}
    )
    service._write_report_locked()
    symbols = [pick["symbol"] for pick in published[0]["swing_picks"]]
    assert len(symbols) == 10
    assert sum(symbol.startswith("A") for symbol in symbols) == 4  # 3 capped + 1 fill
    assert {"B0", "C0", "S0"} <= set(symbols)


# --- setup_key_labels -------------------------------------------------------


def _write_features(path: Path, rows, with_label=True):
    header = ["symbol", "side", "last_trade_date"] + (["permutation_label"] if with_label else [])
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_labels_are_read_by_symbol_and_side_and_absent_column_means_none(tmp_path):
    target = tmp_path / "d1_features.csv"
    _write_features(
        target,
        [
            {"symbol": "aapl", "side": "LONG", "last_trade_date": "2026-09-24", "permutation_label": "slot_1000|m5_vwap"},
            {"symbol": "MSFT", "side": "SHORT", "last_trade_date": "2026-09-24", "permutation_label": "unknown"},
        ],
    )
    labels = setup_key_labels.read_labels(target)
    assert labels == {("AAPL", "LONG"): {"label": "slot_1000|m5_vwap", "date": "2026-09-24"}}
    bare = tmp_path / "bare.csv"
    _write_features(bare, [{"symbol": "AAPL", "side": "LONG", "last_trade_date": "2026-09-24"}], with_label=False)
    assert setup_key_labels.read_labels(bare) == {}


def test_attach_fills_matching_rows_and_skips_another_session(tmp_path):
    target = tmp_path / "d1_features.csv"
    _write_features(
        target,
        [
            {"symbol": "AAPL", "side": "LONG", "last_trade_date": "2026-09-24", "permutation_label": "k1"},
            {"symbol": "NVDA", "side": "LONG", "last_trade_date": "2026-09-23", "permutation_label": "k2"},
        ],
    )
    rows = [
        SimpleNamespace(symbol="AAPL", side="LONG", last_trade_date="2026-09-24", raw={}),
        SimpleNamespace(symbol="AAPL", side="SHORT", last_trade_date="2026-09-24", raw={}),
        SimpleNamespace(symbol="NVDA", side="LONG", last_trade_date="2026-09-24", raw={}),
    ]
    assert setup_key_labels.attach_labels(rows, allow_read=True, path=target) == 1
    assert rows[0].raw["permutation_label"] == "k1"
    assert "permutation_label" not in rows[1].raw and "permutation_label" not in rows[2].raw


def test_attach_without_allow_read_never_opens_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_key_labels, "read_labels", lambda *_a, **_k: pytest.fail("read on Qt"))
    rows = [SimpleNamespace(symbol="AAPL", side="LONG", last_trade_date="", raw={})]
    assert setup_key_labels.attach_labels(rows) == 0


def test_short_label_caps_the_length():
    long = "a" * 60
    out = setup_key_labels.short_label(long)
    assert len(out) <= setup_key_labels.SHORT_LABEL_MAX and out.endswith("..")
    assert setup_key_labels.short_label(float("nan")) == ""


# --- the setups table -------------------------------------------------------


def test_the_setups_table_shows_the_label_and_knows_when_none_exist():
    from ui.models.setup import SetupRow
    from ui.models.setup_table_model import SetupTableModel

    column = [key for key, _label in SetupTableModel.COLUMNS].index("setup_key")
    model = SetupTableModel(
        [
            SetupRow(symbol="AAPL", side="LONG", raw={"permutation_label": "slot_1000"}),
            SetupRow(symbol="MSFT", side="LONG", raw={}),
        ]
    )
    assert model.has_setup_keys()
    assert model.data(model.index(0, column)) == "slot_1000"
    assert model.data(model.index(1, column)) == ""
    assert not SetupTableModel([SetupRow(symbol="MSFT", side="LONG")]).has_setup_keys()


def test_the_tags_tooltip_carries_the_key_for_the_compact_profile():
    from PySide6.QtCore import Qt

    from ui.models.setup import SetupRow
    from ui.models.setup_table_model import SetupTableModel

    column = [key for key, _label in SetupTableModel.COLUMNS].index("setup_tags")
    model = SetupTableModel([SetupRow(symbol="AAPL", side="LONG", raw={"permutation_label": "slot_1000"})])
    tip = model.data(model.index(0, column), Qt.ItemDataRole.ToolTipRole)
    assert tip.endswith("Setup key: slot_1000")
