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
