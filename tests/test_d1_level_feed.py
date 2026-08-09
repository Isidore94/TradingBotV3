import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


NOW = datetime(2026, 7, 15, 10, 0)


def _write_ai_state(path: Path, *, last_trade_date="2026-07-15", trendline=104.2):
    payload = {
        "symbols": {
            "MU": {
                "last_trade_date": last_trade_date,
                "priority_sma_levels": {
                    "SMA_20": 101.0,  # not a major level; must be ignored
                    "SMA_50": 102.5,
                    "SMA_100": 98.0,
                    "SMA_200": 90.0,
                },
                "priority_trendline_candidate": {"current_line_price": trendline},
                "priority_trendline_break_candidate": None,
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_store(levels_dir: Path):
    levels_dir.mkdir(parents=True, exist_ok=True)
    store = {
        "schema_version": 1,
        "symbol": "MU",
        "levels": [
            {"kind": "hv_horizontal", "price": 99.4, "strength": 1.4},
            {"kind": "hv_horizontal", "price": 99.9, "strength": 0.35},  # too weak
            {"kind": "hv_horizontal", "price": 260.0, "strength": 2.0},  # too far
            {
                "kind": "cloud_flat",
                "price": 101.7,
                "strength": 1.4,
                "effective_range": ["2026-01-01", "2026-12-31"],
            },
            {
                "kind": "cloud_flat",
                "price": 100.9,
                "strength": 1.4,
                "effective_range": ["2025-01-01", "2025-06-30"],  # expired
            },
        ],
    }
    (levels_dir / "MU.json").write_text(json.dumps(store), encoding="utf-8")


def test_feed_shapes_major_levels_with_filters(tmp_path):
    from d1_level_feed import get_d1_extra_levels

    ai_state = tmp_path / "ai_state.json"
    levels_dir = tmp_path / "levels"
    _write_ai_state(ai_state)
    _write_store(levels_dir)

    out = get_d1_extra_levels(
        "MU",
        reference_price=100.0,
        atr20=2.0,
        now=NOW,
        ai_state_path=ai_state,
        levels_dir=levels_dir,
    )
    by_family = {}
    for level in out:
        by_family.setdefault(level["family"], []).append(level["value"])
    assert by_family["d1_sma_50"] == [102.5]
    assert by_family["d1_sma_100"] == [98.0]
    assert "d1_sma_200" not in by_family  # 90.0 is >3.5 ATR away
    assert by_family["d1_trendline"] == [104.2]
    assert by_family["d1_horizontal"] == [99.4]  # weak + far levels filtered
    assert by_family["d1_cloud_flat"] == [101.7]  # expired cloud filtered
    weights = {level["family"]: level["weight"] for level in out}
    assert weights["d1_sma_50"] == 1.6
    assert weights["d1_horizontal"] == 1.5
    assert all(level["weight"] > 1.25 for level in out)  # outrank every M5 family


def test_feed_staleness_drops_trendline_before_smas(tmp_path):
    from d1_level_feed import get_d1_extra_levels

    ai_state = tmp_path / "ai_state.json"
    _write_ai_state(ai_state, last_trade_date="2026-07-08")  # 7 days stale
    out = get_d1_extra_levels(
        "MU",
        now=NOW,
        ai_state_path=ai_state,
        levels_dir=tmp_path / "levels",
    )
    families = {level["family"] for level in out}
    assert "d1_sma_50" in families  # SMAs tolerate up to 10 days
    assert "d1_trendline" not in families  # projected slope is stale after 5


def test_feed_refreshes_when_ai_state_changes(tmp_path):
    from d1_level_feed import get_d1_extra_levels

    ai_state = tmp_path / "ai_state.json"
    levels_dir = tmp_path / "levels"
    _write_ai_state(ai_state, trendline=104.2)
    first = get_d1_extra_levels(
        "MU", now=NOW, ai_state_path=ai_state, levels_dir=levels_dir
    )
    _write_ai_state(ai_state, trendline=105.5)
    os.utime(ai_state, (NOW.timestamp(), NOW.timestamp() + 60))
    second = get_d1_extra_levels(
        "MU", now=NOW, ai_state_path=ai_state, levels_dir=levels_dir
    )
    trendline = lambda rows: [r["value"] for r in rows if r["family"] == "d1_trendline"]
    assert trendline(first) == [104.2]
    assert trendline(second) == [105.5]


def test_feed_unknown_symbol_and_missing_files_are_empty(tmp_path):
    from d1_level_feed import get_d1_extra_levels

    assert (
        get_d1_extra_levels(
            "ZZZZ",
            now=NOW,
            ai_state_path=tmp_path / "missing.json",
            levels_dir=tmp_path / "missing_levels",
        )
        == []
    )


# ------------------------------------------------------- the shared raw parse
# ai_state is ~38MB and two modules want different slivers of it. The parse
# lives in d1_level_feed and chart_levels borrows it, so a scan's rewrite of
# the file costs ONE read rather than one per consumer. Shaping, thresholds
# and filtering stay each module's own - only the bytes are shared.


def _count_json_load(monkeypatch):
    import json as json_module

    counter = {"parses": 0}
    real_load = json_module.load

    def counting_load(handle, *args, **kwargs):
        counter["parses"] += 1
        return real_load(handle, *args, **kwargs)

    monkeypatch.setattr(json_module, "load", counting_load)
    return counter


def test_both_consumers_share_one_ai_state_parse(tmp_path, monkeypatch):
    import chart_levels
    import d1_level_feed

    ai_state = tmp_path / "ai_state.json"
    _write_ai_state(ai_state)
    # Both projections have to be known before a single read can serve both;
    # each one registers itself on its first call.
    d1_level_feed._load_ai_state_feed(ai_state)
    chart_levels._ai_state_trendlines(ai_state)
    d1_level_feed.reset_ai_state_cache()

    counter = _count_json_load(monkeypatch)
    feed = d1_level_feed._load_ai_state_feed(ai_state)
    trendlines = chart_levels._ai_state_trendlines(ai_state)

    # One read, and each consumer still gets exactly its own sliver.
    assert counter["parses"] == 1
    assert feed["MU"]["smas"]["SMA_50"] == 102.5
    assert feed["MU"]["trendlines"] == [104.2]
    assert trendlines["MU"]["candidate"]["current_line_price"] == 104.2
    assert trendlines["MU"]["last_trade_date"] == "2026-07-15"

    # Unchanged file: neither pays again.
    d1_level_feed._load_ai_state_feed(ai_state)
    chart_levels._ai_state_trendlines(ai_state)
    assert counter["parses"] == 1

    # A rewrite is one more read, shared again - and both slivers move.
    _write_ai_state(ai_state, trendline=105.5)
    os.utime(ai_state, (NOW.timestamp(), NOW.timestamp() + 60))
    assert chart_levels._ai_state_trendlines(ai_state)["MU"]["candidate"][
        "current_line_price"
    ] == 105.5
    assert d1_level_feed._load_ai_state_feed(ai_state)["MU"]["trendlines"] == [105.5]
    assert counter["parses"] == 2


def test_a_symbol_with_no_trendline_still_reaches_this_modules_feed(tmp_path):
    """The projections are independent: one skipping a symbol never hides it
    from the other."""
    import chart_levels
    import d1_level_feed

    ai_state = tmp_path / "ai_state.json"
    ai_state.write_text(
        json.dumps(
            {
                "symbols": {
                    "MU": {
                        "last_trade_date": "2026-07-15",
                        "priority_sma_levels": {"SMA_50": 102.5},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    d1_level_feed.reset_ai_state_cache()

    assert chart_levels._ai_state_trendlines(ai_state) == {}
    feed = d1_level_feed._load_ai_state_feed(ai_state)
    assert feed["MU"]["smas"] == {"SMA_50": 102.5}
    assert feed["MU"]["trendlines"] == []


def test_a_torn_ai_state_keeps_the_last_good_slivers(tmp_path):
    """Uncertainty is not emptiness - a bad read must not blank the levels."""
    import chart_levels
    import d1_level_feed

    ai_state = tmp_path / "ai_state.json"
    _write_ai_state(ai_state)
    d1_level_feed.reset_ai_state_cache()
    assert d1_level_feed._load_ai_state_feed(ai_state)["MU"]["trendlines"] == [104.2]
    assert chart_levels._ai_state_trendlines(ai_state)["MU"]["last_trade_date"]

    ai_state.write_text("{not json", encoding="utf-8")
    os.utime(ai_state, (NOW.timestamp(), NOW.timestamp() + 60))
    assert d1_level_feed._load_ai_state_feed(ai_state)["MU"]["trendlines"] == [104.2]
    assert chart_levels._ai_state_trendlines(ai_state)["MU"][
        "candidate"
    ]["current_line_price"] == 104.2
