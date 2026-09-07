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


def test_v2_never_grades_an_episode_whose_representative_is_pending():
    """Fix round, reviewer blocker 2 on `37e63b9c`.

    ST4.2 stopped `_summarize_tracker_setup_outcome` substituting `avg_closed_r`
    for an open representative, but the family AGGREGATE put it straight back:
    `closed_rows` was `closed_setups > 0` (ANY tradeable scenario closed) and
    the win/loss loop fell back to `avg_closed_r` when
    `representative_closed_r` was None. On the 2026-09-03 mirror that graded
    **271 of 2,712** v2 episodes whose representative was still running - 252
    of them as losses, 19 as wins.

    The fixture is that shape: the representative (`full_band2` on the primary
    stop) is OPEN at +0.40R, an alternate exit plan closed -1.00R. Under v2 the
    episode is PENDING and grades nothing; under v1 it is a loss, unchanged.
    """
    from datetime import date

    def _scenario(scenario_id, template, status, total_r, exit_date=None):
        return {
            "scenario_id": scenario_id, "stop_reference_label": "LOWER_1",
            "stop_reference_level": 95.0, "stop_source_type": "band",
            "exit_template_id": template, "exit_template_label": template,
            "framework_family": "baseline", "framework_version": "baseline",
            "experimental": False, "tradeable": True, "status": status,
            "total_r": total_r, "days_held": 4, "entry_price": 100.0,
            "initial_risk_per_share": 5.0, "initial_risk_usd": 500.0,
            "direction": 1.0,
            "events": (
                [{"trade_date": exit_date, "reason": "STOP", "price": 95.0, "shares": 100}]
                if exit_date else []
            ),
        }

    population = {
        "s": {
            "symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02",
            "scan_date": "2026-01-05", "priority_bucket": "tracked",
            "setup_family": "avwape_bounce", "setup_status": "OPEN",
            "favorite_signals": [],
            "scenarios": {
                "open_rep": _scenario("open_rep", "full_band2", "OPEN", 0.4),
                "closed_alt": _scenario(
                    "closed_alt", "full_band3", "STOPPED", -1.0, "2026-01-09"
                ),
            },
        }
    }
    kwargs = {"reference_date": date(2026, 1, 20), "lookback_days": 45}

    v2 = m.build_recent_tracker_setup_family_rows(
        population,
        selection_policy=selection_policy.SELECTION_FIRST_ACTIONABLE_V2,
        **kwargs,
    )[0]
    assert int(v2["n_episodes"]) == 1
    assert int(v2["n_pending"]) == 1, "an open representative is pending, not a loss"
    assert int(v2["n_wins"]) == 0
    assert int(v2["n_losses"]) == 0
    assert int(v2["closed_setups"]) == 0
    assert v2["win_rate_closed_unweighted"] is None

    v1 = m.build_recent_tracker_setup_family_rows(
        population,
        selection_policy=selection_policy.SELECTION_CLOSED_FIRST_V1,
        **kwargs,
    )[0]
    assert int(v1["n_pending"]) == 0
    assert int(v1["n_losses"]) == 1, "characterized: the alternate's -1.00R is today's grade"
    assert int(v1["closed_setups"]) == 1

    # ST7 (2026-09-06, decision 0019): v2 is the DEFAULT, so the bare call is
    # the pending answer now. v1 keeps its name and its characterization above.
    default = m.build_recent_tracker_setup_family_rows(population, **kwargs)[0]
    assert default["selection_policy"] == selection_policy.SELECTION_FIRST_ACTIONABLE_V2
    assert int(default["n_pending"]) == 1
    assert int(default["n_losses"]) == 0
    assert int(default["closed_setups"]) == 0


def test_a_second_stamped_output_never_rewrites_the_first(tmp_path):
    """Packet test 9's never-overwrite clause, asserted on the FILE COUNT too.

    Two runs inside the same second must produce a second stamped PAIR (the
    `-2` sibling), never reuse or truncate the first, and the two refusals must
    still hold once a legitimate output directory exists.
    """
    import json

    import tracker_selection_compare as compare

    tracker = tmp_path / "tracker.json"
    payload = m._default_setup_tracker_payload()
    payload["data_session"] = "2026-02-02"
    payload["setups"] = {
        "s": {
            "symbol": "AAPL", "side": "LONG", "anchor_date": "2026-01-20",
            "scan_date": "2026-01-23", "priority_bucket": "tracked",
            "setup_family": "post_earnings_52w_break", "setup_status": "CLOSED",
            "favorite_signals": [],
            "scenarios": {
                "a": {
                    "scenario_id": "a", "stop_reference_label": "LOWER_1",
                    "exit_template_id": "full_band2", "framework_family": "baseline",
                    "framework_version": "baseline", "experimental": False,
                    "tradeable": True, "status": "TARGET_HIT", "total_r": 1.1,
                    "days_held": 4, "entry_price": 100.0,
                    "initial_risk_per_share": 5.0, "initial_risk_usd": 500.0,
                    "direction": 1.0,
                    "events": [{"trade_date": "2026-01-29", "reason": "FINAL_TARGET",
                                "price": 110.0, "shares": 100}],
                }
            },
        }
    }
    tracker.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"

    assert compare.main(["--tracker", str(tracker), "--out", str(out_dir)]) == 0
    first = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
    assert len(first) == 2

    assert compare.main(["--tracker", str(tracker), "--out", str(out_dir)]) == 0
    assert len(list(out_dir.iterdir())) == 4, "the second run wrote a new stamped pair"
    for name, original in first.items():
        assert (out_dir / name).read_bytes() == original, f"{name} was rewritten"

    protected = Path(str(compare.PROTECTED_DATA_ROOT))
    assert compare.main(["--tracker", str(protected / "x.json"), "--out", str(out_dir)]) != 0
    assert compare.main(["--tracker", str(tracker), "--out", str(protected / "x")]) != 0
    assert len(list(out_dir.iterdir())) == 4, "a refused run wrote nothing"
