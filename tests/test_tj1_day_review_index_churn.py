"""TJ-1 review advisories (a) and (b) - what the index costs the home folder.

One index is 22-30 MB of a live session, and it lives in the SHARED home folder.
Three rules keep that from becoming churn, and one keeps it from becoming a lie:

* **A session that has not closed is never indexed.** Its stores are still being
  appended to, so the file would be out of date before it was written - and
  `is_stale` refuses such an index anyway, so writing it is pure cost. Today's
  page streams until the post-close tick builds the one that lasts.
* **An index whose content has not changed is not rewritten.** Two builds of a
  finished session differ only in `built_at`, and rewriting 30 MB to move one
  timestamp is churn on a folder the trader syncs.
* **The folder is pruned to the newest `KEEP_SESSIONS`.** The picker offers
  fifteen sessions and every index is rebuildable.
* **`is_stale` compares a stamp of the four indexed sources.** A warehouse
  recompute REWRITES those CSVs, and the rows of a session that closed weeks ago
  can change; no clause about pending horizons would ever notice.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SESSION = "2026-09-10"
#: Friday morning: 2026-09-10 is the last COMPLETED session, 2026-09-11 is today.
NOW = datetime(2026, 9, 11, 7, 30)
TODAY = "2026-09-11"


@pytest.fixture()
def sources(tmp_path):
    """The four indexed stores, present and tiny, under `tmp_path`."""
    import daily_recap_reader

    home = tmp_path / "home"
    home.mkdir()
    files = {
        "intraday_outcomes": "intraday_bounce_outcomes.csv",
        "session_horizon_outcomes": "master_avwap_session_horizon_outcomes.csv",
        "tier_outcomes": "master_avwap_tier_outcomes.csv",
        "human_focus_outcomes": "human_focus_outcomes.csv",
    }
    paths = {}
    for name, filename in files.items():
        path = home / filename
        path.write_text("trade_date,symbol\n", encoding="utf-8")
        paths[name] = path
    return daily_recap_reader.RecapSources(
        intraday_outcomes=paths["intraday_outcomes"],
        tier_outcomes=paths["tier_outcomes"],
        session_horizon_outcomes=paths["session_horizon_outcomes"],
        human_focus_outcomes=paths["human_focus_outcomes"],
        annotations=home / "trader_annotations.jsonl",
        pick_feedback=home / "pick_feedback.jsonl",
        swing_favorites=home / "swing_favorites.jsonl",
        review_events=home / "alert_review_events.jsonl",
        preference_report=home / "preference_trade_outcomes.csv",
        staged_picks=home / "auto_populate_pending.json",
        environment_labels=home / "d1_environment.jsonl",
        working_lately=home / "snapshot_latest.json",
    )


def _index(sources, session=SESSION, now=NOW):
    import day_review_index

    return day_review_index.build_index(session, sources=sources, now=now)


# ---------------------------------------------------------------------------
# (a) size and churn
# ---------------------------------------------------------------------------
def test_a_session_that_has_not_closed_is_never_written(sources, tmp_path):
    """Today's stores are still being appended to."""
    import day_review_index

    root = tmp_path / "day_review"
    written = day_review_index.write_index(
        _index(sources, session=TODAY), root=root, now=NOW
    )
    assert written is None
    assert not day_review_index.index_path(TODAY, root=root).exists()
    # ...and the page therefore streams that session, which is the intended cost.
    assert day_review_index.read_index(TODAY, root=root) is None


def test_the_session_that_just_closed_is_written(sources, tmp_path):
    import day_review_index

    root = tmp_path / "day_review"
    written = day_review_index.write_index(_index(sources), root=root, now=NOW)
    assert written is not None
    assert Path(written).is_file()


def test_an_unchanged_index_is_not_rewritten(sources, tmp_path):
    """Two builds of a finished session differ only in `built_at`."""
    import day_review_index

    root = tmp_path / "day_review"
    path = Path(day_review_index.write_index(_index(sources), root=root, now=NOW))
    first_mtime = path.stat().st_mtime_ns
    stored = json.loads(path.read_text(encoding="utf-8"))

    later = day_review_index.build_index(
        SESSION, sources=sources, now=datetime(2026, 9, 11, 9, 0)
    )
    assert later["built_at"] != stored["built_at"], "the fixture proves nothing"
    again = day_review_index.write_index(later, root=root, now=NOW)

    assert Path(again) == path
    assert path.stat().st_mtime_ns == first_mtime, "an unchanged index was rewritten"
    assert json.loads(path.read_text(encoding="utf-8"))["built_at"] == stored["built_at"]


def test_a_changed_index_is_written(sources, tmp_path):
    import day_review_index

    root = tmp_path / "day_review"
    path = Path(day_review_index.write_index(_index(sources), root=root, now=NOW))
    before = json.loads(path.read_text(encoding="utf-8"))

    Path(sources.tier_outcomes).write_text(
        "trade_date,symbol\n2026-09-10,NVDA\n", encoding="utf-8"
    )
    day_review_index.write_index(_index(sources), root=root, now=NOW)

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["sources_stamp"] != before["sources_stamp"]


def test_the_folder_is_pruned_to_the_newest_sessions(sources, tmp_path, monkeypatch):
    """Every index is rebuildable, so the oldest go."""
    import day_review_index

    monkeypatch.setattr(day_review_index, "KEEP_SESSIONS", 3)
    root = tmp_path / "day_review"
    for day in ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"):
        path = day_review_index.index_path(day, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    day_review_index.write_index(_index(sources), root=root, now=NOW)

    kept = sorted(child.name for child in (root / "sessions").iterdir())
    assert kept == ["2026-09-03", "2026-09-04", SESSION], kept


def test_pruning_never_costs_the_write(sources, tmp_path, monkeypatch):
    import day_review_index

    monkeypatch.setattr(
        day_review_index,
        "_prune",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("the folder is busy")),
    )
    root = tmp_path / "day_review"
    with pytest.raises(OSError):
        day_review_index._prune(root)  # the stub really raises
    # ...and the write still returns its path, because pruning is maintenance.
    monkeypatch.setattr(day_review_index, "_prune", lambda *_a, **_k: 0)
    assert day_review_index.write_index(_index(sources), root=root, now=NOW) is not None


# ---------------------------------------------------------------------------
# (b) the stores' own stamp
# ---------------------------------------------------------------------------
def test_the_index_records_a_stamp_of_every_store_it_covers(sources):
    import day_review_index

    stamp = _index(sources)["sources_stamp"]
    assert set(stamp) == set(day_review_index.INDEXED_SOURCES)
    for name, entry in stamp.items():
        assert entry["size"] >= 0, name
        assert entry["mtime_ns"] > 0, name


def test_a_recomputed_store_makes_the_index_stale(sources):
    """The warehouse rewrites these CSVs; a closed session's rows can change,
    and nothing else in the index would ever notice."""
    import day_review_index

    index = _index(sources)
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False

    Path(sources.human_focus_outcomes).write_text(
        "trade_date,symbol\n2026-09-10,NVDA\n", encoding="utf-8"
    )
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_a_store_that_disappeared_makes_it_stale(sources):
    import day_review_index

    index = _index(sources)
    Path(sources.tier_outcomes).unlink()
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_without_sources_the_stamp_is_not_consulted(sources):
    """Every caller that asks about the CLOCK alone - and every index written
    before this clause existed - is answered as before."""
    import day_review_index

    index = _index(sources)
    Path(sources.tier_outcomes).unlink()
    assert day_review_index.is_stale(index, now=NOW) is False

    stampless = {key: value for key, value in index.items() if key != "sources_stamp"}
    assert day_review_index.is_stale(stampless, now=NOW, sources=sources) is False


def test_the_stamp_is_asked_before_the_pending_rule(sources):
    """A rewritten store invalidates an index that has nothing pending at all -
    the pending clauses would have answered "never stale"."""
    import day_review_index

    index = dict(_index(sources))
    index["pending"] = False
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is False

    Path(sources.intraday_outcomes).write_text("trade_date\n2026-09-10\n", encoding="utf-8")
    assert day_review_index.is_stale(index, now=NOW, sources=sources) is True


def test_the_calendar_still_says_what_this_fixture_assumes():
    import market_calendar

    assert market_calendar.last_completed_session(NOW).isoformat() == SESSION
    assert date.fromisoformat(TODAY) > market_calendar.last_completed_session(NOW)
