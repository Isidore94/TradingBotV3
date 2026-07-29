import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _make_store(tmp_path):
    from focus_picks import FocusPickStore

    return FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "focus_pick_membership.json",
    )


def _symbols(path):
    from watchlist_utils import read_watchlist_symbols

    return read_watchlist_symbols(path)


def test_add_normalizes_and_dedupes(tmp_path):
    store = _make_store(tmp_path)
    assert store.add("aapl", "long") is True
    assert store.add("AAPL", "LONG") is False  # duplicate after normalization
    assert store.focus_longs() == ["AAPL"]
    assert _symbols(tmp_path / "focus_longs.txt") == ["AAPL"]


def test_add_injects_into_correct_shared_watchlist(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long")
    store.add("TSLA", "short")
    assert _symbols(tmp_path / "longs.txt") == ["NVDA"]
    assert _symbols(tmp_path / "shorts.txt") == ["TSLA"]
    membership = store.membership()
    assert membership["NVDA|long"]["shared_file"] == "longs.txt"
    assert membership["TSLA|short"]["shared_file"] == "shorts.txt"


def test_remove_uninjects_only_what_focus_injected(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long")  # injected by focus
    assert _symbols(tmp_path / "longs.txt") == ["NVDA"]
    store.remove("NVDA", "long")
    assert store.focus_longs() == []
    assert _symbols(tmp_path / "longs.txt") == []  # un-injected
    assert "NVDA|long" not in store.membership()


def test_remove_preserves_independently_present_shared_symbol(tmp_path):
    # MSFT already lives in the broad watchlist before Focus Picks ever sees it.
    (tmp_path / "longs.txt").write_text("MSFT\n", encoding="utf-8")
    store = _make_store(tmp_path)
    store.add("MSFT", "long")  # already present -> not recorded as injected
    assert "MSFT|long" not in store.membership()
    store.remove("MSFT", "long")
    assert store.focus_longs() == []
    assert _symbols(tmp_path / "longs.txt") == ["MSFT"]  # NOT deleted


def test_add_many_paste_dedupes(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long")
    added = store.add_many("nvda, aapl\nMSFT", "long")
    assert added == ["AAPL", "MSFT"]  # NVDA already present
    assert store.focus_longs() == ["NVDA", "AAPL", "MSFT"]


def test_clear_removes_injected_keeps_independent(tmp_path):
    (tmp_path / "longs.txt").write_text("B\n", encoding="utf-8")  # independent
    store = _make_store(tmp_path)
    store.add("A", "long")  # injected
    store.add("B", "long")  # already present -> not injected
    assert store.focus_longs() == ["A", "B"]
    removed = store.clear("long")
    assert removed == 2
    assert store.focus_longs() == []
    assert _symbols(tmp_path / "longs.txt") == ["B"]  # injected A gone, B kept
    assert store.membership() == {}


def test_focus_side_and_is_focus(tmp_path):
    store = _make_store(tmp_path)
    store.add("AAPL", "long")
    store.add("TSLA", "short")
    assert store.focus_side("aapl") == "long"
    assert store.focus_side("TSLA") == "short"
    assert store.focus_side("XYZ") is None
    assert store.is_focus("aapl") is True
    assert store.is_focus("aapl", "short") is False
    store.add("AAPL", "short")
    assert store.focus_side("AAPL") == "both"


def test_membership_persists_across_reload(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long")
    reopened = _make_store(tmp_path)  # fresh instance, same paths
    assert reopened.focus_longs() == ["NVDA"]
    assert "NVDA|long" in reopened.membership()


def test_listener_fires_on_change(tmp_path):
    store = _make_store(tmp_path)
    calls = {"n": 0}
    store.add_listener(lambda: calls.__setitem__("n", calls["n"] + 1))
    store.add("NVDA", "long")
    store.remove("NVDA", "long")
    assert calls["n"] == 2


def test_load_focus_map(tmp_path):
    from focus_picks import load_focus_map

    (tmp_path / "focus_longs.txt").write_text("NVDA\nAAPL\n", encoding="utf-8")
    (tmp_path / "focus_shorts.txt").write_text("TSLA\n", encoding="utf-8")
    focus = load_focus_map(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
    )
    assert focus["long"] == {"NVDA", "AAPL"}
    assert focus["short"] == {"TSLA"}


def test_swing_category_uses_own_files_and_swing_watchlists(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing")
    store.add("TSLA", "short", "swing")
    store.add("AAPL", "long")  # default m5 keeps original behavior

    # Separate focus files per category, separate shared-watchlist targets.
    assert _symbols(tmp_path / "focus_swing_longs.txt") == ["NVDA"]
    assert _symbols(tmp_path / "focus_swing_shorts.txt") == ["TSLA"]
    assert _symbols(tmp_path / "focus_longs.txt") == ["AAPL"]
    assert _symbols(tmp_path / "swinglongs.txt") == ["NVDA"]
    assert _symbols(tmp_path / "shortswings.txt") == ["TSLA"]
    assert _symbols(tmp_path / "longs.txt") == ["AAPL"]

    membership = store.membership()
    assert membership["NVDA|long|swing"]["shared_file"] == "swinglongs.txt"
    assert membership["AAPL|long"]["shared_file"] == "longs.txt"  # legacy key format

    assert store.focus_category("NVDA") == "swing"
    assert store.focus_category("AAPL") == "m5"
    assert store.focus_symbols("long") == ["NVDA", "AAPL"]  # swing-first union


def test_swing_remove_only_uninjects_swing_watchlist(tmp_path):
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "swing")
    store.add("NVDA", "long", "m5")
    assert store.focus_category("NVDA") == "both"

    store.remove("NVDA", "long", "swing")
    assert store.focus_category("NVDA") == "m5"
    assert _symbols(tmp_path / "swinglongs.txt") == []  # swing injection undone
    assert _symbols(tmp_path / "longs.txt") == ["NVDA"]  # m5 injection intact


def test_load_focus_maps_by_category_and_union(tmp_path):
    from focus_picks import load_focus_map, load_focus_maps_by_category

    (tmp_path / "focus_longs.txt").write_text("AAPL\n", encoding="utf-8")
    (tmp_path / "focus_swing_longs.txt").write_text("NVDA\n", encoding="utf-8")
    by_category = load_focus_maps_by_category(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
    )
    assert by_category["swing"]["long"] == {"NVDA"}
    assert by_category["m5"]["long"] == {"AAPL"}
    # Explicit-path load_focus_map stays a two-file read (legacy engine callers).
    focus = load_focus_map(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
    )
    assert focus["long"] == {"AAPL"}


# ---------------------------------------------------------------------------
# m5 is a DAY-TRADE list: picks belong to one calendar day (user rule
# 2026-07-29). Swing picks are multi-day and never expire this way.
# ---------------------------------------------------------------------------
def test_m5_focus_resets_on_a_new_calendar_day(tmp_path):
    import json
    from datetime import date, timedelta

    # Independent broad-list entry that must survive the reset untouched.
    (tmp_path / "longs.txt").write_text("MSFT\n", encoding="utf-8")
    store = _make_store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.add("TSLA", "short", "m5")
    store.add("AMD", "long", "swing")
    assert _symbols(tmp_path / "longs.txt") == ["MSFT", "NVDA"]

    # Same-day reload keeps everything.
    reloaded = _make_store(tmp_path)
    assert reloaded.focus_symbols("long", "m5") == ["NVDA"]
    assert reloaded.focus_symbols("short", "m5") == ["TSLA"]

    # Backdate the stamp: the next store load clears m5 (both sides),
    # un-injects only what focus injected, and leaves swing alone.
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    (tmp_path / "focus_m5_state.json").write_text(
        json.dumps({"market_date": yesterday}), encoding="utf-8"
    )
    fresh = _make_store(tmp_path)
    assert fresh.focus_symbols("long", "m5") == []
    assert fresh.focus_symbols("short", "m5") == []
    assert fresh.focus_symbols("long", "swing") == ["AMD"]
    assert _symbols(tmp_path / "longs.txt") == ["MSFT"]  # NVDA un-injected
    assert _symbols(tmp_path / "shorts.txt") == []
    assert _symbols(tmp_path / "focus_longs.txt") == []
    # The stamp now belongs to today, so another reload is a no-op.
    assert fresh.expire_m5_if_new_day() == 0

    # A new m5 add re-stamps today and sticks.
    assert fresh.add("PLTR", "long", "m5") is True
    assert _make_store(tmp_path).focus_symbols("long", "m5") == ["PLTR"]


def test_m5_focus_missing_stamp_is_grandfathered(tmp_path):
    # Pre-upgrade files with no stamp sidecar: nothing is guessed stale.
    (tmp_path / "focus_longs.txt").write_text("NVDA\n", encoding="utf-8")
    store = _make_store(tmp_path)
    assert store.focus_symbols("long", "m5") == ["NVDA"]
    # The first load wrote today's stamp.
    import json

    payload = json.loads((tmp_path / "focus_m5_state.json").read_text(encoding="utf-8"))
    from datetime import date

    assert payload["market_date"] == date.today().isoformat()


def test_load_focus_maps_exclude_stale_m5_read_only(tmp_path):
    import json
    from datetime import date, timedelta

    from focus_picks import load_focus_maps_by_category

    store = _make_store(tmp_path)
    store.add("NVDA", "long", "m5")
    store.add("AMD", "long", "swing")

    # Fresh stamp: m5 included.
    by_cat = load_focus_maps_by_category(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
    )
    assert by_cat["m5"]["long"] == {"NVDA"}

    # Stale stamp: the read-only accessors exclude m5 WITHOUT touching disk
    # (the physical clear belongs to the next store load).
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    (tmp_path / "focus_m5_state.json").write_text(
        json.dumps({"market_date": yesterday}), encoding="utf-8"
    )
    by_cat = load_focus_maps_by_category(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
    )
    assert by_cat["m5"]["long"] == set()
    assert by_cat["swing"]["long"] == {"AMD"}
    assert _symbols(tmp_path / "focus_longs.txt") == ["NVDA"]  # untouched
