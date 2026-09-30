"""Mentor P9: the tilt pack. Today's patterns after a loss (burst, re-entry, size, streak) with leg ids,
and each pattern's base rate of a red rest of day over the journal history, "too few" under 30."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import journal_read, registry, tilt_pack  # noqa: E402

NOW = tilt_pack.FIXTURE_NOW


@pytest.fixture
def journal(tmp_path):
    return tilt_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def test_golden_today_every_pattern_with_its_legs(journal):
    rows = _rows(tilt_pack.build(now=NOW, journal=journal))
    assert rows["tilt:burst:094000"]["text"] == (
        "3 opens in 9 min after a losing close on AMD (legs 2, 3, 4, 5, 7). Observation, not a rule.")
    assert rows["tilt:burst:094000"]["legs"] == [2, 3, 4, 5, 7], "partial fills of one order are one open"
    assert rows["tilt:reentry:AMD:094500"]["text"].startswith("Re-opened AMD LONG 5 min after a losing close on it")
    assert rows["tilt:size:AMD:094500"]["text"].startswith("An open of AMD at 1.6x today's median size after a loss")
    assert rows["tilt:streak:094000"]["legs"] == [2, 6, 8]
    assert rows["tilt:streak:094000"]["before_pnl"] == -170.0
    assert datetime.fromisoformat(rows["tilt:burst:094000"]["at"]).tzinfo is not None


def test_base_rates_say_too_few_honestly(journal):
    rows = _rows(tilt_pack.build(now=NOW, journal=journal))
    streak = rows["tilt:base:streak"]
    assert (streak["n"], streak["red"]) == (2, 1) and streak["too_few"]
    assert streak["text"].endswith("1 of 2 times: too few (n=2, floor 30)")
    assert rows["tilt:base:burst"]["text"].endswith("0 of 0 times: too few (n=0, floor 30)")


def test_nothing_is_seen_before_it_happened(journal):
    early = NOW.replace(hour=9, minute=44)
    rows = _rows(tilt_pack.build(now=early, journal=journal))
    assert "tilt:none" in rows and not tilt_pack.observations(tilt_pack.build(now=early, journal=journal))
    mid = NOW.replace(hour=9, minute=47)
    ids = [r["id"] for r in tilt_pack.observations(tilt_pack.build(now=mid, journal=journal))]
    assert ids == ["tilt:reentry:AMD:094500", "tilt:size:AMD:094500"], "the burst needs its third open"


def test_an_open_trade_is_never_a_loss(journal):
    conn = sqlite3.connect(journal)
    conn.execute("UPDATE trades SET status = 'OPEN', net_pnl = NULL, net_pnl_usd = NULL WHERE trade_id = 'T1'")
    conn.commit()
    conn.close()
    assert tilt_pack.observations(tilt_pack.build(now=NOW.replace(hour=9, minute=59), journal=journal)) == []


def test_ids_are_stable_and_unique(journal):
    one = tilt_pack.build(now=NOW, journal=journal)
    two = tilt_pack.build(now=NOW + timedelta(minutes=30), journal=journal)
    assert [r["id"] for r in tilt_pack.observations(one)] == [r["id"] for r in tilt_pack.observations(two)]
    assert len(one.ids) == len(set(one.ids))


def test_registered_and_read_only():
    assert "tilt_pack" in registry.names()
    source = (SCRIPTS_DIR / "mentor_packs" / "tilt_pack.py").read_text(encoding="utf-8")
    assert "JournalStore(" not in source and "PySide6" not in source and "from ui" not in source
    assert tilt_pack.fixture().ids


def test_leg_signature_changes_only_with_new_legs(journal):
    day = NOW.date().isoformat()
    first = journal_read.leg_signature(journal, day)
    assert first == journal_read.leg_signature(journal, day)
    conn = sqlite3.connect(journal)
    conn.execute("INSERT INTO trade_legs VALUES (50, 'T5', 'SELL', 'CLOSE', 1, 1.0, '2026-09-29T10:25:00-04:00')")
    conn.commit()
    conn.close()
    assert journal_read.leg_signature(journal, day) != first
    assert journal_read.leg_signature(journal.parent / "missing.sqlite3", day) is None


def test_thresholds_are_module_constants():
    assert (tilt_pack.BURST_OPENS, tilt_pack.BURST_WINDOW, tilt_pack.REENTRY_WINDOW) == (
        3, timedelta(minutes=10), timedelta(minutes=15))
    assert (tilt_pack.SIZE_MULTIPLE, tilt_pack.STREAK_LOSSES, tilt_pack.BASE_SESSIONS) == (1.5, 3, 60)
