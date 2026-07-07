"""Write-avoidance behavior of the shared earnings history.

The history file is tens of MB on a Drive-synced folder; these tests pin the
two mechanisms that keep scan loops from rewriting it per iteration:
material-change detection (identical re-merges skip the save entirely) and
the reentrant deferred-save scope (loops batch real changes into one write).
"""

import os
import sys
import time

import pytest


def _event(ticker="XYZ", date_text="2026-08-07", session="TBD"):
    return {
        "ticker": ticker,
        "earnings_date": date_text,
        "release_session": session,
        "source": "nasdaq",
    }


def _count_saves(monkeypatch, module):
    saves = []
    original_save = module.save_history

    def _counting_save(history, path=None):
        saves.append(str(path))
        return original_save(history, path)

    monkeypatch.setattr(module, "save_history", _counting_save)
    return saves


def test_merge_events_skips_write_when_nothing_material_changed(tmp_path, monkeypatch):
    import earnings_history

    path = tmp_path / "history.json"
    earnings_history.merge_events([_event()], path=path)
    saves = _count_saves(monkeypatch, earnings_history)

    # Identical re-merge (the hourly cache-hit case): no write.
    earnings_history.merge_events([_event()], path=path)
    assert saves == []

    # Material change (TBD -> confirmed AMC session): still writes.
    earnings_history.merge_events(
        [{**_event(session="AMC"), "source_confidence": "confirmed"}], path=path
    )
    assert len(saves) == 1
    events = earnings_history.get_events_for_symbols(["XYZ"], path=path)["XYZ"]
    assert events[0]["release_session"] == "AMC"


def test_deferred_history_save_batches_to_one_write(tmp_path, monkeypatch):
    import earnings_history

    path = tmp_path / "history.json"
    saves = _count_saves(monkeypatch, earnings_history)

    earnings_history.begin_deferred_history_save()
    try:
        earnings_history.merge_events([_event("AAA", "2026-08-07")], path=path)
        earnings_history.merge_events([_event("BBB", "2026-08-14")], path=path)
        earnings_history.merge_events([_event("CCC", "2026-08-21")], path=path)
        assert saves == []
        assert not path.exists()
    finally:
        earnings_history.end_deferred_history_save()

    assert len(saves) == 1
    events = earnings_history.get_events_for_symbols(["AAA", "BBB", "CCC"], path=path)
    assert all(events[ticker] for ticker in ("AAA", "BBB", "CCC"))


def test_deferred_history_save_is_reentrant(tmp_path):
    import earnings_history

    path = tmp_path / "history.json"
    earnings_history.begin_deferred_history_save()
    earnings_history.begin_deferred_history_save()
    try:
        earnings_history.merge_events([_event("AAA")], path=path)
        earnings_history.end_deferred_history_save()
        assert not path.exists()  # inner end must not flush
    finally:
        earnings_history.end_deferred_history_save()
    assert path.exists()
    assert earnings_history.get_events_for_symbols(["AAA"], path=path)["AAA"]


def test_deferred_scope_with_no_changes_writes_nothing(tmp_path, monkeypatch):
    import earnings_history

    path = tmp_path / "history.json"
    earnings_history.merge_events([_event()], path=path)
    saves = _count_saves(monkeypatch, earnings_history)

    earnings_history.begin_deferred_history_save()
    try:
        earnings_history.merge_events([_event()], path=path)
        earnings_history.merge_events([_event()], path=path)
    finally:
        earnings_history.end_deferred_history_save()
    assert saves == []
