"""Packet ST4, builder-added: two guards the tester's file does not pin.

`tests/test_st4_first_actionable.py` is the packet's contract and is not
touched here. These two cover decisions the build had to make on its own:

1. The compare CLI's SECOND refusal. `PROTECTED_DATA_ROOT` refuses a
   ``--tracker`` / ``--out`` under the live home, but the scratch-script rule
   (CLAUDE.md, incident 2026-09-05) is about the other direction: a run whose
   ``project_paths.DATA_DIR`` resolves under ``C:\\TradingBotData`` must abort
   before it reads anything, because that is the run that overwrote the live
   tracker. Under pytest DATA_DIR is a temp directory, so the guard is
   exercised by pointing it at the live root explicitly.
2. `_scenario_recorded_exit_date` is gated on the CLOSED status. Scenario
   ``events`` gain one entry per exit LEG (`_apply_scenario_exit_event`), so a
   partial exit leaves a dated event behind on a scenario that is still
   running. Reading that date as an exit would let `first_actionable_v2` open a
   second attempt while the first one is still in the trade - the exact thing
   the re-entry rule exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402
import tracker_selection_compare as compare  # noqa: E402
from master_avwap_lib import selection_policy  # noqa: E402


def test_compare_cli_aborts_when_data_dir_resolves_under_the_live_home(tmp_path, monkeypatch, capsys):
    tracker = tmp_path / "tracker.json"
    tracker.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        compare.project_paths,
        "DATA_DIR",
        compare.WELL_KNOWN_LIVE_HOME / "data",
        raising=False,
    )
    rc = compare.main(["--tracker", str(tracker), "--out", str(out_dir)])

    assert rc != 0
    assert not out_dir.exists(), "a refused run creates nothing"
    captured = capsys.readouterr()
    assert "REFUSING" in captured.err
    assert "TRADINGBOTV3_DATA_DIR" in captured.err


def test_a_partial_exit_event_is_not_an_exit_date():
    """A running scenario with a partial's event dated 2026-01-08 reports no
    exit date, so a scan on 2026-01-12 stays the SAME attempt."""
    partially_exited = {
        "status": "PARTIAL",
        "events": [{"trade_date": "2026-01-08", "reason": "PARTIAL_TARGET", "shares": 50}],
    }
    finished = {
        "status": "TARGET_HIT",
        "events": [
            {"trade_date": "2026-01-08", "reason": "PARTIAL_TARGET", "shares": 50},
            {"trade_date": "2026-01-10", "reason": "FINAL_TARGET", "shares": 50},
        ],
    }
    assert m._scenario_recorded_exit_date(partially_exited) == ""
    assert m._scenario_recorded_exit_date(finished) == "2026-01-10"

    def _row(scan_date, exit_date):
        return {
            "symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02",
            "setup_family": "f", "scan_date": scan_date, "closed_setups": 0,
            "representative_exit_date": exit_date,
        }

    still_running = selection_policy.select_episode_rows(
        [_row("2026-01-05", ""), _row("2026-01-12", "")],
        policy=selection_policy.SELECTION_FIRST_ACTIONABLE_V2,
    )
    assert len(still_running) == 1
    assert still_running[0]["scan_date"] == "2026-01-05"

    out_and_back_in = selection_policy.select_episode_rows(
        [_row("2026-01-05", "2026-01-10"), _row("2026-01-12", "")],
        policy=selection_policy.SELECTION_FIRST_ACTIONABLE_V2,
    )
    assert len(out_and_back_in) == 2


def test_an_unknown_selection_policy_raises_rather_than_falling_back():
    """A typo must not silently ship the default and be reported as the
    challenger's numbers."""
    import pytest

    with pytest.raises(ValueError):
        selection_policy.select_episode_rows([], policy="first_actionable_v3")
    with pytest.raises(ValueError):
        m.build_recent_tracker_setup_family_rows(
            {"a": {}}, selection_policy="closed_first_v2"
        )
