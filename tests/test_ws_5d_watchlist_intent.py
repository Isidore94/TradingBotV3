"""WS-5D - a watchlist edit is a dated event, never a verdict.

Every test here drives the REAL surface (the Qt panel's own methods, the real
`FocusPickStore`, the module's own CLI) rather than a helper written for the
test, because the packet's whole claim is about what happens when the trader
edits a list on the desk.

The invariants under test, one per packet item:

* a trader add / remove / re-add appends `trader_edit` rows with aware,
  market-local timestamps; a paste is `trader_paste`;
* a sort and an unchanged save append nothing (membership did not change);
* the Focus store's shared-watchlist injection is `machine_inject` and its
  removal `machine_uninject` - never confusable with a trader edit;
* an edit made OUTSIDE the app is `observed_external`, stamped at the LOAD
  time, never at a guessed earlier time;
* a first load with no stream writes ONE `baseline_recorded` row and no adds;
* an evidence failure never costs the save - the file is written and the
  status carries a suffix;
* hand-entered names survive every path (plan.md sec 5).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- helpers
@pytest.fixture()
def events_path(tmp_path, monkeypatch):
    """Point the store at the sandbox, resolved at CALL time by every writer."""
    import watchlist_intent_events as wie

    target = tmp_path / "watchlist_intent_events.jsonl"
    monkeypatch.setattr(wie, "EVENTS_FILE", target)
    return target


def _rows(events_path, **kwargs):
    import watchlist_intent_events as wie

    return wie.read_events(path=events_path, **kwargs)


def _changes(events_path, **kwargs):
    import watchlist_intent_events as wie

    return [
        row
        for row in wie.read_events(path=events_path, **kwargs)
        if row.get("action") in (wie.ACTION_ADD, wie.ACTION_REMOVE)
    ]


def _panel(path: Path, title: str = "Shared Longs"):
    from ui.panels.watchlists_panel import WatchlistEditorPanel

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return WatchlistEditorPanel(title, path, lambda panel, symbols: None)


def _focus_store(tmp_path):
    from focus_picks import FocusPickStore

    return FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "focus_pick_membership.json",
    )


# --------------------------------------------------------------------------- item 1/2
def test_manual_add_remove_readd_are_dated_trader_edit_rows(tmp_path, events_path):
    import watchlist_intent_events as wie

    longs = tmp_path / "longs.txt"
    panel = _panel(longs)

    panel.add_symbol_input.setText("aapl")
    panel.add_symbol()
    panel.remove_symbols({"AAPL"})
    panel.add_symbol_input.setText("AAPL")
    panel.add_symbol()

    rows = _changes(events_path)
    assert [(row["action"], row["symbol"]) for row in rows] == [
        (wie.ACTION_ADD, "AAPL"),
        (wie.ACTION_REMOVE, "AAPL"),
        (wie.ACTION_ADD, "AAPL"),
    ], "a re-add after a removal is a NEW add row, not a silent no-op"

    for row in rows:
        assert row["schema"] == wie.SCHEMA_WATCHLIST_INTENT_EVENT
        assert row["source"] == wie.SOURCE_TRADER_EDIT
        assert row["list"] == "longs"
        assert row["side"] == "long"
        assert row["horizon"] == "day"
        assert row["writer"], "every row names the module that wrote it"
        stamp = datetime.fromisoformat(row["ts"])
        assert stamp.tzinfo is not None, "an evidence timestamp carries its offset"
        assert stamp.utcoffset() is not None


def test_paste_is_labelled_trader_paste(tmp_path, events_path):
    import watchlist_intent_events as wie

    shorts = tmp_path / "shorts.txt"
    panel = _panel(shorts, "Shared Shorts")

    QApplication.clipboard().setText("NVDA\nTSLA")
    panel.paste_symbols()

    rows = _changes(events_path)
    assert [row["symbol"] for row in rows] == ["NVDA", "TSLA"]
    assert {row["source"] for row in rows} == {wie.SOURCE_TRADER_PASTE}
    assert {row["side"] for row in rows} == {"short"}


def test_sort_and_unchanged_save_append_nothing(tmp_path, events_path):
    longs = tmp_path / "longs.txt"
    longs.write_text("MSFT\nAAPL\n", encoding="utf-8")
    panel = _panel(longs)

    before = len(_rows(events_path))
    panel.sort_symbols()
    panel.force_save()

    assert panel.current_symbols() == ["AAPL", "MSFT"], "the sort really happened"
    assert len(_rows(events_path)) == before, "re-ordering is not a membership change"


def test_swing_lists_carry_the_swing_horizon(tmp_path, events_path):
    swing = tmp_path / "swinglongs.txt"
    panel = _panel(swing, "Swing Longs")

    panel.add_symbol_input.setText("AMD")
    panel.add_symbol()

    row = _changes(events_path)[-1]
    assert (row["list"], row["side"], row["horizon"]) == ("swinglongs", "long", "swing")


# --------------------------------------------------------------------------- item 3
def test_machine_inject_and_uninject_are_labelled(tmp_path, events_path):
    import watchlist_intent_events as wie

    store = _focus_store(tmp_path)
    store.add("NVDA", "long", category="m5")
    store.remove("NVDA", "long", category="m5")

    rows = _changes(events_path)
    assert [(row["source"], row["action"], row["symbol"]) for row in rows] == [
        (wie.SOURCE_MACHINE_INJECT, wie.ACTION_ADD, "NVDA"),
        (wie.SOURCE_MACHINE_UNINJECT, wie.ACTION_REMOVE, "NVDA"),
    ]
    assert {row["list"] for row in rows} == {"longs"}


def test_swing_focus_injection_names_the_swing_list(tmp_path, events_path):
    import watchlist_intent_events as wie

    store = _focus_store(tmp_path)
    store.add("PLTR", "short", category="swing")

    row = _changes(events_path)[-1]
    assert row["source"] == wie.SOURCE_MACHINE_INJECT
    assert (row["list"], row["side"], row["horizon"]) == ("shortswings", "short", "swing")


def test_hand_entered_names_survive_injection_and_removal(tmp_path, events_path):
    from watchlist_utils import read_watchlist_symbols

    longs = tmp_path / "longs.txt"
    longs.write_text("KO\nPEP\n", encoding="utf-8")

    store = _focus_store(tmp_path)
    store.add("NVDA", "long", category="m5")
    assert read_watchlist_symbols(longs) == ["KO", "PEP", "NVDA"]
    store.remove("NVDA", "long", category="m5")
    assert read_watchlist_symbols(longs) == ["KO", "PEP"], (
        "plan.md sec 5: a user-entered watchlist name is never auto-removed"
    )


# --------------------------------------------------------------------------- item 4
def test_first_load_writes_one_baseline_row_and_no_adds(tmp_path, events_path):
    import watchlist_intent_events as wie

    longs = tmp_path / "longs.txt"
    longs.write_text("AAPL\nMSFT\n", encoding="utf-8")
    _panel(longs)

    rows = _rows(events_path)
    assert len(rows) == 1
    assert rows[0]["action"] == wie.ACTION_BASELINE
    assert rows[0]["list"] == "longs"
    assert rows[0]["symbol_count"] == 2
    assert not _changes(events_path), "a baseline is not two invented adds"


def test_external_edit_between_loads_is_observed_external_at_load_time(tmp_path, events_path):
    import watchlist_intent_events as wie

    longs = tmp_path / "longs.txt"
    longs.write_text("AAPL\nMSFT\n", encoding="utf-8")
    _panel(longs)  # baseline recorded

    before = datetime.now().astimezone() - timedelta(seconds=2)
    longs.write_text("AAPL\nTSLA\n", encoding="utf-8")  # Notepad, the DAS, anything
    _panel(longs)  # reopening the page observes the difference

    rows = _changes(events_path)
    assert {(row["action"], row["symbol"]) for row in rows} == {
        (wie.ACTION_REMOVE, "MSFT"),
        (wie.ACTION_ADD, "TSLA"),
    }
    for row in rows:
        assert row["source"] == wie.SOURCE_OBSERVED_EXTERNAL
        assert datetime.fromisoformat(row["ts"]) >= before, (
            "an observed diff is stamped at the load, never at a guessed earlier time"
        )


def test_a_reload_after_the_traders_own_edit_observes_nothing(tmp_path, events_path):
    longs = tmp_path / "longs.txt"
    panel = _panel(longs)
    panel.add_symbol_input.setText("AAPL")
    panel.add_symbol()

    before = len(_rows(events_path))
    panel.refresh_from_disk()
    assert len(_rows(events_path)) == before, (
        "the stream already explains this membership; nothing external happened"
    )


def test_the_baseline_row_is_written_once_per_list(tmp_path, events_path):
    import watchlist_intent_events as wie

    longs = tmp_path / "longs.txt"
    longs.write_text("AAPL\n", encoding="utf-8")
    for _ in range(3):
        _panel(longs)

    baselines = [row for row in _rows(events_path) if row["action"] == wie.ACTION_BASELINE]
    assert len(baselines) == 1, "the stream grows with CHANGES, never with loads"


# --------------------------------------------------------------------------- item 2 (failure)
def test_an_evidence_failure_never_costs_the_save(tmp_path, events_path, monkeypatch):
    import watchlist_intent_events as wie

    longs = tmp_path / "longs.txt"
    panel = _panel(longs)

    def _boom(*args, **kwargs):
        raise RuntimeError("evidence store unavailable")

    monkeypatch.setattr(wie, "record_changes", _boom)

    statuses: list[str] = []
    panel.statusChanged.connect(statuses.append)
    panel.add_symbol_input.setText("AAPL")
    panel.add_symbol()

    assert longs.read_text(encoding="utf-8").splitlines() == ["AAPL"], (
        "an evidence store never costs the thing it records"
    )
    assert statuses and "intent not recorded" in statuses[-1]


# --------------------------------------------------------------------------- item 5
def test_read_events_filters_by_list_and_since(tmp_path, events_path):
    import watchlist_intent_events as wie

    early = datetime(2026, 9, 10, 9, 30).astimezone()
    late = datetime(2026, 9, 11, 9, 30).astimezone()
    wie.record_changes(list_name="longs", added=["AAPL"], source=wie.SOURCE_TRADER_EDIT,
                       writer="test", now=early, path=events_path)
    wie.record_changes(list_name="shorts", added=["NVDA"], source=wie.SOURCE_TRADER_EDIT,
                       writer="test", now=late, path=events_path)

    assert [row["symbol"] for row in wie.read_events(path=events_path)] == ["AAPL", "NVDA"]
    assert [row["symbol"] for row in wie.read_events("longs", path=events_path)] == ["AAPL"]
    assert [
        row["symbol"] for row in wie.read_events(path=events_path, since=late - timedelta(hours=1))
    ] == ["NVDA"]


def test_cli_tail_prints_the_rows(tmp_path, events_path, capsys):
    import watchlist_intent_events as wie

    wie.record_changes(list_name="longs", added=["AAPL"], removed=["MSFT"],
                       source=wie.SOURCE_TRADER_EDIT, writer="test", path=events_path)
    wie.record_changes(list_name="shorts", added=["NVDA"],
                       source=wie.SOURCE_TRADER_EDIT, writer="test", path=events_path)

    assert wie.main(["tail", "--list", "longs", "--path", str(events_path)]) == 0
    out = capsys.readouterr().out
    assert "AAPL" in out and "MSFT" in out
    assert "NVDA" not in out, "--list longs means the longs list"


def test_an_unreadable_stream_is_empty_not_an_exception(tmp_path):
    import watchlist_intent_events as wie

    assert wie.read_events(path=tmp_path / "nope.jsonl") == []
    assert wie.record_changes(list_name="not_a_list", added=["AAPL"],
                              source=wie.SOURCE_TRADER_EDIT, writer="test",
                              path=tmp_path / "x.jsonl") == 0
