"""ST6 - the Working-lately snapshot, the priority switch and the AWAY Recap.

Written by the TESTER on `claude/st6-working-lately`, based on
`origin/claude/st2-real-counts` (`1145d3b7`), and proven RED there before any
of it existed. The builder makes these pass; it may add tests and may not
weaken one.

Trader, 2026-09-06: *"Implement the already-owed desk Working-lately surface,
priority switch, and Away Recap around one deterministic evidence snapshot ...
Persist a small, deduplicated leader-change event with prior/new leader,
snapshot IDs, timestamp, and cause ... Choose any margin/persistence rule
before inspecting its forward evaluation. The first release can report an
observational leader or no clear leader; do not call every winner proven."*

===========================================================================
THE API THIS FILE PINS
===========================================================================

`scripts/working_lately.py` (additive; ST2's `select_leader`,
`LEADER_MARGIN_LB`, `LEADER_FRESHNESS_SESSIONS` and `LeaderVerdict` are kept
exactly as they are):

* ``EvidenceCell`` - frozen. The packet's fields: ``kind`` (one of
  ``swing_trade_r`` / ``swing_favorable`` / ``daytrade_held_run``), ``side``,
  ``family``, ``outcome_kind``, ``outcome_version``, ``knowledge_basis``,
  ``horizon``, ``window_sessions``, ``latest_measured_session``,
  ``n_eligible``, ``n_pending``, ``n_excluded``, ``n_symbols``,
  ``n_sessions``, ``top_symbol_share``, ``top_session_share``, ``statistic``,
  ``statistic_name``, ``uncertainty_low``, ``uncertainty_kind``,
  ``namespace``.  Any field beyond that list carries a default, so a cell can
  be built from the listed ones alone.

  For a swing cell ``n_eligible`` is the group's ELIGIBLE ROW COUNT and the
  Wilson denominator is ``wins + losses`` - a FLAT is a measured outcome with
  no answer to the win/loss question (R4 B6), so it counts in ``n_eligible``
  and never in the rate's denominator.

* ``EvidenceSnapshot`` - frozen: ``snapshot_id`` (sha1 hex of the sorted cell
  tuples + the policy lines + ``as_of`` - and of NOTHING else, so a source's
  mtime and ``built_at`` cannot move it), ``as_of``, ``built_at``, ``cells``,
  ``verdicts`` (``{kind: LeaderVerdict}``), ``sources``
  (``{name: {"path", "mtime", "rows", "rows_by_session"}}``).

  An ABSENT source has ``rows is None`` - never ``0``. An absent source is a
  question that was not asked; a zero is an answer.

* ``build_snapshot(*, recent_rows, favorable_read, held_run_summaries,
  last_completed_session, previous_verdicts, sources=None) -> EvidenceSnapshot``
  - PURE. Identical inputs give an identical ``snapshot_id`` and identical
  verdicts. ``favorable_read`` is an ST1 ``swing_evidence.EligibleRead`` or
  ``None``; ``held_run_summaries`` is what
  ``held_run_score.dimension_summaries`` returns, or ``None``.

* ``LEADER_PERSISTENCE_SNAPSHOTS = 2`` - declared 2026-09-06 BEFORE any
  forward evaluation. A NEW leader is announced only once it has led in two
  consecutive snapshots with DISTINCT ``as_of``. Until then the verdict is
  ``no_clear_leader``, its reason contains ``awaiting persistence (1 of 2)``,
  and ``coverage["pending_leader"]`` / ``coverage["pending_leader_snapshots"]``
  name the candidate and the count.

* A cell whose ``top_session_share`` or ``top_symbol_share`` EXCEEDS 0.5 is not
  eligible for leadership; the verdict reason contains ``concentrated``.
  Dependence is answered by REFUSING, never by a new formula.

* ``leader_name(verdict) -> str`` - ``"{SIDE} {family}"``, or ``""`` when the
  verdict names no leader. The events file uses it in both directions.

* ``pool_cells(cells)`` - the one helper that would combine cells. It RAISES
  ``ValueError`` when the cells differ in ``kind``, ``side`` or
  ``outcome_kind``, and the message names the axis. Nothing in the desk pools
  across those three, and this is the refusal that keeps it true.

* ``snapshot_stamp(snapshot_or_payload) -> str`` - the one short identity every
  surface prints, containing ``snapshot_id[:8]``.

* ``alert_priority_key(alert) -> (bounce_type, side)`` - derived exactly as
  `alert_center_panel._attach_held_run_suffix` derives it today
  (``payload["feedback"]["bounce_types"].split(";")[0]``, falling back to
  ``alert.trigger``), so the bar, the review list and the suffix cannot drift.

`scripts/ui/services/working_lately_service.py`:

* ``WorkingLatelyService(*, store_dir, parent=None)`` - a ``QObject``.
  ``snapshot_path`` is ``store_dir/"snapshot_latest.json"``, ``events_path`` is
  ``store_dir/"leader_change_events.jsonl"``.  **A test injects `store_dir`;
  the service never resolves %LOCALAPPDATA% for a test.**
* ``previous_verdicts() -> {kind: LeaderVerdict}`` rebuilt from the persisted
  snapshot (``{}`` when there is none).
* ``publish(snapshot) -> list[dict]`` - writes ``snapshot_latest.json``
  (temp-and-rename) and APPENDS to ``leader_change_events.jsonl`` one event per
  kind whose leadership changed; returns only the events appended by THIS call.
* ``events() -> list[dict]``, ``load_snapshot() -> dict | None``.
* ``build_payload() -> dict`` - the WORKER side: calls the three module-level
  readers ``read_recent_rows`` / ``read_favorable_read`` /
  ``read_held_run_summaries``, builds, publishes, returns the payload dict.
* ``_on_payload_ready(payload: dict)`` - the GUI slot. It emits
  ``snapshotChanged(dict)`` and does nothing else: no reader, no file.

An event is ``{ts, kind, prior_leader, new_leader, prior_snapshot_id,
new_snapshot_id, cause, as_of}``. It is appended when the kind's leader NAME
changed, or when the verdict crossed into ``last_reliable_reading`` /
``no_evidence`` from a state that had one. The dedupe key is
``(kind, prior_leader, new_leader, new_snapshot_id)``, so a restart replays
nothing. ``cause`` is one of ``new_outcomes`` / ``window_rollover`` /
``corrected_data`` / ``lost_coverage``.

**Note to the builder on cause precedence.** The packet lists ``window_rollover``
first ("when `as_of` moved"), but `as_of` moves on nearly every build, which
would make ``corrected_data`` and ``lost_coverage`` unreachable. Check them
refusal-first: ``lost_coverage``, then ``corrected_data``, then
``window_rollover``, then ``new_outcomes``. No test below depends on that
ordering - each cause is asserted with the other conditions held constant -
so this is a recommendation, not a pin.

The surfaces:

* ``ui/widgets/working_lately_strip.py`` -> ``WorkingLatelyStrip`` with
  ``set_snapshot(payload)``, ``line_text()`` (one line, starting
  ``Working lately (20 sessions):``) and ``tooltip_text()`` (every cell line).
* ``M5AlertBar.set_working_lately_order([(bounce_type, side), ...])`` and
  ``AlertCenterPanel.set_working_lately_order([(bounce_type, side), ...])``.
* ``MasterAvwapPanel.set_working_lately_order([(side, family), ...])``.
* ``SetupTrackerPanel.set_working_lately_snapshot(payload)``; the banner then
  renders THAT snapshot, its own CSV read staying the labelled fallback.
* ``weekend_verdict.build_verdict(..., working_lately=payload)``.
* ``away_recap.build_recap(..., working_lately=payload, leader_events=(...))``.
* The switch is the ``local_settings`` key ``prioritise_working_lately``
  (default OFF), read AT SORT TIME and never at write time.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SWING_OUTCOME_KIND = "trade_r_representative_exit"


# ---------------------------------------------------------------------------
# calendar helpers - the tests are dated by the exchange calendar, never by a
# literal, so they do not rot and never count in calendar days.
# ---------------------------------------------------------------------------


def _sessions(count: int) -> list[date]:
    """The last ``count`` completed exchange sessions, OLDEST first."""
    import market_calendar

    out = [market_calendar.last_completed_session(datetime.now(market_calendar.MARKET_TZ))]
    while len(out) < count:
        out.append(market_calendar.previous_session(out[-1]))
    return list(reversed(out))


def _qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module", autouse=True)
def _app():
    pytest.importorskip("PySide6")
    yield _qt_app()


# ---------------------------------------------------------------------------
# fixtures that model the real files
# ---------------------------------------------------------------------------


def _recent_row(
    family: str,
    *,
    wins: int,
    losses: int,
    session: date,
    side: str = "LONG",
    namespace: str = "live",
    outcome_kind: str = SWING_OUTCOME_KIND,
) -> dict:
    """One row of `RECENT_SETUP_TYPE_STATS_FILE` as ST2.1 exports it.

    Every column is a STRING, because the panel reads them back out of a CSV
    through `csv.DictReader` and an integer that only exists in a test fixture
    proves nothing about the file.
    """
    n = wins + losses
    return {
        "namespace": namespace,
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "status": "",
        "closed_setups": str(n),
        "tracked_setups": str(n),
        "avg_closed_r": "0.5",
        "target_hit_rate": "0.5",
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "0",
        "n_unmeasured": "0",
        "n_pending": "0",
        "n_symbols": "12",
        "n_entry_sessions": "9",
        "outcome_kind": outcome_kind,
        "outcome_version": "recent_types_v2",
        "knowledge_basis": "entry_scan_row_close_to_representative_exit",
        "horizon_basis": "30d lookback, representative exit",
        "latest_measured_session": session.isoformat(),
    }


def _favorable_row(
    *,
    symbol: str,
    scan_date: date,
    side_return_pct: float,
    family: str = "avwap_breakout",
    side: str = "LONG",
    horizon: int | None = None,
    stale: bool = False,
) -> dict:
    """One row of `master_avwap_tier_outcomes.csv`, columns as the live file.

    The live header (read 2026-09-06) carries `side_return_pct` - a PERCENT
    MOVE, never an R - plus `observation_id`, `scan_date`, `horizon_sessions`
    and `stale_horizon`. Everything is a string, as `csv.DictReader` yields it.
    """
    from swing_headline import SWING_HORIZON_SESSIONS

    sessions = SWING_HORIZON_SESSIONS if horizon is None else horizon
    return {
        "observation_id": f"{symbol}:{scan_date.isoformat()}:{sessions}:{side_return_pct}",
        "scan_date": scan_date.isoformat(),
        "horizon_sessions": str(sessions),
        "tier": "S",
        "symbol": symbol,
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "side_return_pct": str(side_return_pct),
        "win": "True" if side_return_pct > 0 else "False",
        "stale_horizon": "True" if stale else "",
    }


def _outcome_row(
    *,
    symbol: str,
    session: date,
    bounce: str,
    mfe: str = "2.0",
    entry: str = "10:00:00",
    stop_hit: bool = False,
    minutes: str = "60",
) -> dict:
    """One row of `intraday_bounce_outcomes.csv` as `build_episodes` reads it."""
    stamp = session.isoformat()
    event_id = f"{symbol}_long_{stamp.replace('-', '')}_{entry.replace(':', '_')}_{bounce}"
    return {
        "event_id": event_id,
        "trade_date": stamp,
        "symbol": symbol,
        "direction": "long",
        "entry_time": f"{stamp}T{entry}",
        "context_json": json.dumps({"market_environment": "bullish_strong"}),
        "stop_hit": "True" if stop_hit else "False",
        "mfe_r": mfe,
        "minutes_elapsed": minutes,
    }


def _build(
    *,
    recent_rows=(),
    favorable_read=None,
    held_run_summaries=None,
    as_of: date,
    previous_verdicts=None,
    sources=None,
):
    from working_lately import build_snapshot

    return build_snapshot(
        recent_rows=list(recent_rows),
        favorable_read=favorable_read,
        held_run_summaries=held_run_summaries,
        last_completed_session=as_of,
        previous_verdicts=previous_verdicts or {},
        **({} if sources is None else {"sources": sources}),
    )


def _service(tmp_path: Path):
    from ui.services.working_lately_service import WorkingLatelyService

    return WorkingLatelyService(store_dir=tmp_path / "working_lately")


# ===========================================================================
# ST6.1 - one deterministic evidence snapshot
# ===========================================================================


def test_the_same_evidence_in_any_order_builds_the_same_snapshot_id_and_verdicts():
    """Item 1. Two builds, shuffled input, different source mtimes.

    The id is over the sorted cell tuples + the policy lines + `as_of`, so a
    file's mtime and the wall clock the build happened at cannot move it - a
    snapshot that changed identity every half-hour timer tick would make the
    events file a log of the timer.

    `swing_evidence` is ST1's module and is deliberately imported INSIDE the
    test: on this base the import fails, which is the honest red, and after the
    builder merges ST1 the assertions below are real.
    """
    import swing_evidence
    from swing_headline import SWING_HORIZON_SESSIONS, wilson_lower_bound
    from evidence_stats import LATELY_SESSIONS

    days = _sessions(4)
    d1, d2, d3, as_of = days

    recent = [
        _recent_row("alpha", wins=45, losses=15, session=as_of),
        _recent_row("beta", wins=30, losses=30, session=as_of),
    ]
    # Seven eligible rows: four favourable, two against, ONE FLAT. Three
    # symbols and three scan dates, so neither concentration share exceeds 0.5.
    favorable_rows = [
        _favorable_row(symbol="AAA", scan_date=d1, side_return_pct=4.0),
        _favorable_row(symbol="AAA", scan_date=d1, side_return_pct=2.5),
        _favorable_row(symbol="AAA", scan_date=d1, side_return_pct=-1.5),
        _favorable_row(symbol="BBB", scan_date=d2, side_return_pct=1.0),
        _favorable_row(symbol="BBB", scan_date=d2, side_return_pct=-3.0),
        _favorable_row(symbol="BBB", scan_date=d3, side_return_pct=0.75),
        _favorable_row(symbol="CCC", scan_date=d3, side_return_pct=0.0),
        # Never eligible: the wrong horizon, and an explicit stale horizon.
        _favorable_row(symbol="DDD", scan_date=d3, side_return_pct=9.0, horizon=1),
        _favorable_row(symbol="EEE", scan_date=d3, side_return_pct=9.0, stale=True),
    ]
    read = swing_evidence.read_eligible_rows(
        favorable_rows, swing_evidence.POLICY_SCANROW_V1, end=as_of.isoformat()
    )
    assert len(read.rows) == 7, "the ST1 read is the fixture's own premise"

    first = _build(
        recent_rows=recent,
        favorable_read=read,
        as_of=as_of,
        sources={"recent_types": {"path": "a.csv", "mtime": 111.0}},
    )
    second = _build(
        recent_rows=list(reversed(recent)),
        favorable_read=swing_evidence.read_eligible_rows(
            list(reversed(favorable_rows)),
            swing_evidence.POLICY_SCANROW_V1,
            end=as_of.isoformat(),
        ),
        as_of=as_of,
        sources={"recent_types": {"path": "a.csv", "mtime": 999.0}},
    )

    assert len(first.snapshot_id) == 40, first.snapshot_id
    assert first.snapshot_id == second.snapshot_id
    assert {k: (v.state, v.reason) for k, v in first.verdicts.items()} == {
        k: (v.state, v.reason) for k, v in second.verdicts.items()
    }

    swing = [c for c in first.cells if c.kind == "swing_trade_r" and c.family == "alpha"]
    assert len(swing) == 1
    assert swing[0].n_eligible == 60
    assert swing[0].statistic == pytest.approx(0.75)
    assert swing[0].uncertainty_low == pytest.approx(wilson_lower_bound(45, 60))
    assert swing[0].window_sessions == LATELY_SESSIONS
    assert swing[0].namespace == "live"
    assert swing[0].outcome_kind == SWING_OUTCOME_KIND
    assert swing[0].horizon == "30d lookback, representative exit", (
        "the row's own horizon_basis travels; the cell never restates it"
    )
    assert swing[0].latest_measured_session == as_of.isoformat()

    favorable = [c for c in first.cells if c.kind == "swing_favorable"]
    assert len(favorable) == 1, [c.family for c in favorable]
    cell = favorable[0]
    assert cell.side == "LONG" and cell.family == "avwap_breakout"
    assert cell.n_eligible == 7, "the flat is eligible evidence and is counted"
    assert cell.statistic == pytest.approx(4 / 6), "the flat is not in the rate's denominator"
    assert cell.uncertainty_low == pytest.approx(wilson_lower_bound(4, 6))
    assert cell.n_symbols == 3 and cell.n_sessions == 3
    assert cell.top_symbol_share == pytest.approx(3 / 7)
    assert cell.top_session_share == pytest.approx(3 / 7)
    # The horizon carries its UNIT. v1 counts SCAN ROWS and v2 counts exchange
    # sessions (ST1), and a bare "5" is the confusion that packet removed.
    assert str(SWING_HORIZON_SESSIONS) in cell.horizon, cell.horizon
    assert "scan row" in cell.horizon.lower(), cell.horizon
    assert cell.outcome_kind == swing_evidence.POLICY_SCANROW_V1.outcome_kind
    assert cell.knowledge_basis == swing_evidence.POLICY_SCANROW_V1.knowledge_basis


def test_pooling_cells_across_kind_side_or_outcome_kind_refuses():
    """Item 6. The refusal is the mechanism; there is no pooled formula.

    The cells are taken OFF a real snapshot rather than hand-built, so the test
    cannot pass by agreeing with itself about a constructor.
    """
    from working_lately import pool_cells

    days = _sessions(3)
    as_of = days[-1]
    recent = [
        _recent_row("alpha", wins=45, losses=15, session=as_of),
        _recent_row("alpha", wins=40, losses=20, session=as_of, side="SHORT"),
        _recent_row(
            "gamma", wins=44, losses=16, session=as_of, outcome_kind="trade_r_close_2d"
        ),
    ]
    summaries = _held_run_summaries(days[0], days[1])
    snapshot = _build(recent_rows=recent, held_run_summaries=summaries, as_of=as_of)

    def _cell(**match):
        found = [
            c
            for c in snapshot.cells
            if all(getattr(c, key) == value for key, value in match.items())
        ]
        assert found, f"no cell matched {match}: {[(c.kind, c.side, c.family) for c in snapshot.cells]}"
        return found[0]

    long_alpha = _cell(kind="swing_trade_r", side="LONG", family="alpha")
    short_alpha = _cell(kind="swing_trade_r", side="SHORT", family="alpha")
    other_outcome = _cell(kind="swing_trade_r", side="LONG", family="gamma")
    day_cell = _cell(kind="daytrade_held_run")

    with pytest.raises(ValueError) as mixed_kind:
        pool_cells([long_alpha, day_cell])
    assert "kind" in str(mixed_kind.value).lower()

    with pytest.raises(ValueError) as mixed_side:
        pool_cells([long_alpha, short_alpha])
    assert "side" in str(mixed_side.value).lower()

    with pytest.raises(ValueError) as mixed_outcome:
        pool_cells([long_alpha, other_outcome])
    assert "outcome" in str(mixed_outcome.value).lower()

    # Same kind, same side, same outcome kind: allowed, and the only case that is.
    pool_cells([long_alpha, _cell(kind="swing_trade_r", side="LONG", family="alpha")])

    assert set(snapshot.verdicts) == {
        "swing_trade_r",
        "swing_favorable",
        "daytrade_held_run",
    }
    assert all(hasattr(v, "state") for v in snapshot.verdicts.values())


# ===========================================================================
# ST6.2 - Held x Ran carries its identities
# ===========================================================================


def _episodes(*rows):
    import held_run_score as hrs

    return hrs.build_episodes(list(rows))


def _held_run_summaries(session_a: date, session_b: date, *, min_n: int = 6):
    """Two day-trade cells over the SAME two sessions.

    `ema_15` is CONCENTRATED - five of its six held episodes are one symbol -
    and carries the higher held x ran. `vwap_reclaim` is spread over three
    symbols and carries the lower one. If concentration is not refused, the
    concentrated cell wins, which is the whole point.
    """
    import held_run_score as hrs

    rows = []
    for _ in range(3):
        rows.append(_outcome_row(symbol="AAA", session=session_a, bounce="ema_15", mfe="3.0",
                                 entry=f"10:{len(rows):02d}:00"))
    for symbol in ("AAA", "AAA", "BBB"):
        rows.append(_outcome_row(symbol=symbol, session=session_b, bounce="ema_15", mfe="3.0",
                                 entry=f"10:{len(rows):02d}:00"))
    for symbol in ("CCC", "DDD", "EEE"):
        rows.append(_outcome_row(symbol=symbol, session=session_a, bounce="vwap_reclaim",
                                 mfe="1.0", entry=f"11:{len(rows):02d}:00"))
    for symbol in ("CCC", "DDD", "EEE"):
        rows.append(_outcome_row(symbol=symbol, session=session_b, bounce="vwap_reclaim",
                                 mfe="1.0", entry=f"11:{len(rows):02d}:00"))
    episodes = hrs.build_episodes(rows)
    return hrs.dimension_summaries(episodes, min_n=min_n, as_of=session_b.isoformat())


def test_held_run_carries_its_symbols_and_sessions_and_a_concentrated_cell_is_withheld():
    """Item 9. `held_run_score.py:352-355` calls `summarize` with no identities.

    So `concentration.by_symbol`, `concentration.by_session` and the
    session-block `bootstrap` come back UNMEASURED for every held x ran cell -
    the desk's day-trade headline had no way to know it was one name six times.
    The episodes already carry `symbol` and `trade_date`; this passes them.
    """
    days = _sessions(3)
    session_a, session_b, as_of = days
    summaries = _held_run_summaries(session_a, session_b)

    spread = summaries[("bounce_type", "long", "vwap_reclaim")]
    assert spread["n_held"] == 6
    assert spread["concentration"]["by_symbol"]["measured"] is True
    assert spread["concentration"]["by_symbol"]["distinct"] == 3
    assert spread["concentration"]["by_symbol"]["top_share"] == pytest.approx(2 / 6)
    assert spread["concentration"]["by_session"]["measured"] is True
    assert spread["concentration"]["by_session"]["distinct"] == 2
    assert spread["bootstrap"]["measured"] is True, spread["bootstrap"].get("reason")

    lopsided = summaries[("bounce_type", "long", "ema_15")]
    assert lopsided["concentration"]["by_symbol"]["top_share"] == pytest.approx(5 / 6)
    assert lopsided["held_run_score"] == pytest.approx(3.0)
    assert spread["held_run_score"] == pytest.approx(1.0)

    # Held x Ran stays NAME-SELECTION evidence: the snapshot never grows a P&L
    # field for it, and the statistic says what it is.
    first = _build(held_run_summaries=summaries, as_of=session_b)
    second = _build(
        held_run_summaries=summaries,
        as_of=as_of,
        previous_verdicts=first.verdicts,
    )
    # bounce_type x SIDE. `dimension_summaries` also emits the pooled
    # `("bounce_type", "all", ...)` row (R4 B4); a cell built from that would
    # be the same episodes counted twice under a second name.
    day_cells = {(c.side, c.family): c for c in second.cells if c.kind == "daytrade_held_run"}
    assert set(day_cells) == {("long", "ema_15"), ("long", "vwap_reclaim")}, sorted(day_cells)
    lopsided_cell = day_cells[("long", "ema_15")]
    assert lopsided_cell.statistic == pytest.approx(3.0)
    assert lopsided_cell.statistic_name == "held_run_score (P(held 30m) x trimmed MFE_R)"
    assert lopsided_cell.top_symbol_share == pytest.approx(5 / 6)
    assert not any(
        "pnl" in field.lower() or "p_and_l" in field.lower()
        for field in lopsided_cell.__dataclass_fields__
    )

    from working_lately import leader_name

    verdict = second.verdicts["daytrade_held_run"]
    assert verdict.state == "leader", verdict.reason
    name = leader_name(verdict)
    assert "vwap_reclaim" in name, name
    assert "ema_15" not in name, (
        "the concentrated cell carries the higher held x ran and must not lead"
    )
    assert "concentrated" in (
        verdict.reason + json.dumps(verdict.coverage, default=str)
    ).lower(), verdict.reason


# ===========================================================================
# ST6.3 - the service: persistence, dedupe, replay, causes
# ===========================================================================


def _swing_sequence(as_of: date, *, leader: str):
    """Recent rows in which ``leader`` is clear of the other by 0.25 of bound.

    45/60 -> 0.6277, 30/60 -> 0.3774: a gap of 0.250, five times the declared
    `LEADER_MARGIN_LB` of 0.05, so the margin rule cannot be what decides.
    """
    other = "beta" if leader == "alpha" else "alpha"
    return [
        _recent_row(leader, wins=45, losses=15, session=as_of),
        _recent_row(other, wins=30, losses=30, session=as_of),
    ]


def _publish(service, as_of: date, rows, **kwargs):
    snapshot = _build(
        recent_rows=rows,
        as_of=as_of,
        previous_verdicts=service.previous_verdicts(),
        **kwargs,
    )
    return snapshot, service.publish(snapshot)


def test_a_new_leader_is_announced_only_after_two_snapshots_and_writes_one_event(tmp_path):
    """Item 10. `LEADER_PERSISTENCE_SNAPSHOTS = 2`, declared before any forward look."""
    from working_lately import LEADER_PERSISTENCE_SNAPSHOTS

    assert LEADER_PERSISTENCE_SNAPSHOTS == 2

    d1, d2 = _sessions(2)
    service = _service(tmp_path)

    first, events_1 = _publish(service, d1, _swing_sequence(d1, leader="alpha"))
    verdict = first.verdicts["swing_trade_r"]
    assert verdict.state == "no_clear_leader", verdict.reason
    assert verdict.leader is None
    assert "awaiting persistence (1 of 2)" in verdict.reason
    assert verdict.coverage["pending_leader"] == "LONG alpha"
    assert verdict.coverage["pending_leader_snapshots"] == 1
    assert events_1 == [], "nothing has changed hands yet"

    second, events_2 = _publish(service, d2, _swing_sequence(d2, leader="alpha"))
    assert second.verdicts["swing_trade_r"].state == "leader"
    assert second.verdicts["swing_trade_r"].leader["setup_family"] == "alpha"
    assert len(events_2) == 1, events_2
    event = events_2[0]
    assert event["kind"] == "swing_trade_r"
    assert event["prior_leader"] == ""
    assert event["new_leader"] == "LONG alpha"
    assert event["new_snapshot_id"] == second.snapshot_id
    assert event["as_of"] == d2.isoformat()
    assert event["cause"] in {
        "new_outcomes",
        "window_rollover",
        "corrected_data",
        "lost_coverage",
    }

    assert service.publish(second) == [], "the dedupe key holds inside one run too"
    assert len(service.events()) == 1


def test_replayed_snapshots_write_their_events_in_as_of_order_and_twice_is_the_same_file(tmp_path):
    """Item 2. Order is the sessions' order, and the sequence is a function."""

    d1, d2, d3, d4 = _sessions(4)
    plan = [
        (d1, _swing_sequence(d1, leader="alpha")),
        (d2, _swing_sequence(d2, leader="alpha")),
        (d3, _swing_sequence(d3, leader="beta")),
        (d4, _swing_sequence(d4, leader="beta")),
    ]

    def _run(root: Path):
        from ui.services.working_lately_service import WorkingLatelyService

        service = WorkingLatelyService(store_dir=root)
        for as_of, rows in plan:
            _publish(service, as_of, rows)
        return service.events()

    events = _run(tmp_path / "first")
    swing = [e for e in events if e["kind"] == "swing_trade_r"]
    assert [(e["prior_leader"], e["new_leader"], e["as_of"]) for e in swing] == [
        ("", "LONG alpha", d2.isoformat()),
        ("LONG alpha", "", d3.isoformat()),
        ("", "LONG beta", d4.isoformat()),
    ], swing

    stamps = [e["as_of"] for e in events]
    assert stamps == sorted(stamps), "events come out in as_of order, never build order"

    def _without_ts(rows):
        """Everything but the wall clock - which is the only thing allowed to differ."""
        return [{k: v for k, v in row.items() if k != "ts"} for row in rows]

    again = _run(tmp_path / "second")
    assert _without_ts(again) == _without_ts(events)


def test_a_restart_over_the_same_snapshot_appends_nothing(tmp_path):
    """Item 3. The desk restarts; the phone does not get yesterday's news twice."""
    from ui.services.working_lately_service import WorkingLatelyService

    d1, d2 = _sessions(2)
    service = _service(tmp_path)
    _publish(service, d1, _swing_sequence(d1, leader="alpha"))
    latest, _ = _publish(service, d2, _swing_sequence(d2, leader="alpha"))

    lines_before = service.events_path.read_text(encoding="utf-8")
    assert lines_before.strip(), "the run under test wrote at least one event"

    restarted = WorkingLatelyService(store_dir=service.events_path.parent)
    rebuilt = _build(
        recent_rows=_swing_sequence(d2, leader="alpha"),
        as_of=d2,
        previous_verdicts=restarted.previous_verdicts(),
    )
    assert rebuilt.snapshot_id == latest.snapshot_id
    assert restarted.publish(rebuilt) == []
    assert restarted.events_path.read_text(encoding="utf-8") == lines_before


def test_stale_evidence_reads_as_the_last_reliable_reading_and_never_as_new_leadership(tmp_path):
    """Item 4. `as_of` is held FIXED; only the evidence goes stale.

    So `lost_coverage` cannot be confused with a window rollover, and the event
    that is written names the SAME leader in both slots - nothing new was
    learned, the desk simply stopped being able to read it.
    """
    from working_lately import LEADER_FRESHNESS_SESSIONS

    days = _sessions(6)
    as_of = days[-1]
    stale_session = days[0]

    service = _service(tmp_path)
    _publish(service, days[-2], _swing_sequence(days[-2], leader="alpha"))
    _publish(service, as_of, _swing_sequence(as_of, leader="alpha"))
    assert service.previous_verdicts()["swing_trade_r"].state == "leader"

    stale_rows = _swing_sequence(as_of, leader="alpha")
    for row in stale_rows:
        row["latest_measured_session"] = stale_session.isoformat()
    snapshot, events = _publish(service, as_of, stale_rows)

    verdict = snapshot.verdicts["swing_trade_r"]
    assert verdict.state == "last_reliable_reading", verdict.reason
    assert verdict.leader is not None and verdict.leader["setup_family"] == "alpha"
    assert LEADER_FRESHNESS_SESSIONS == 2

    swing = [e for e in events if e["kind"] == "swing_trade_r"]
    assert len(swing) == 1, events
    assert swing[0]["cause"] == "lost_coverage"
    assert swing[0]["prior_leader"] == swing[0]["new_leader"] == "LONG alpha", (
        "a lost-coverage event never announces a NEW leader"
    )
    assert service.publish(snapshot) == []

    # The other half: a source that could not be read is an ABSENT source, not
    # a zero, and it never produces a leader.
    assert snapshot.sources["swing_favorable"]["rows"] is None
    favorable = snapshot.verdicts["swing_favorable"]
    assert favorable.state in {"no_evidence", "last_reliable_reading"}
    assert "source unavailable" in favorable.reason.lower(), favorable.reason


def test_a_change_to_an_earlier_session_is_labelled_corrected_data(tmp_path):
    """Item 5. Same `as_of`; rows for a session at or before it changed.

    A restatement is not news about the market, and an events file that called
    it `new_outcomes` would teach the trader that the desk found something.
    """
    days = _sessions(3)
    earlier, prev, as_of = days

    service = _service(tmp_path)
    base_rows = _swing_sequence(prev, leader="alpha")
    _publish(service, prev, base_rows)
    settled, _ = _publish(service, as_of, _swing_sequence(as_of, leader="alpha"))
    assert settled.verdicts["swing_trade_r"].state == "leader"

    corrected = _swing_sequence(as_of, leader="alpha")
    # A restatement: rows appear for a session at or before the previous
    # `as_of`, and alpha's record is revised down far enough to lose the lead.
    corrected[0]["n_wins"], corrected[0]["n_losses"] = "30", "30"
    corrected.append(_recent_row("delta", wins=44, losses=16, session=earlier))
    snapshot, events = _publish(service, as_of, corrected)

    assert snapshot.as_of == settled.as_of, "the window did not roll"
    swing = [e for e in events if e["kind"] == "swing_trade_r"]
    assert len(swing) == 1, events
    assert swing[0]["prior_leader"] == "LONG alpha"
    assert swing[0]["cause"] == "corrected_data", swing[0]


def test_the_gui_slot_only_formats_and_the_readers_live_on_the_worker(tmp_path, monkeypatch):
    """Item 11. Nothing expensive belongs on the Qt thread, and a read is expensive.

    Two phases, because "the slot did not read" is only worth something once
    the readers are proven to be the seam the WORKER goes through.
    """
    from ui.services import working_lately_service as svc

    calls: list[str] = []

    def _spy(name, value):
        def _reader():
            calls.append(name)
            return value

        return _reader

    monkeypatch.setattr(svc, "read_recent_rows", _spy("recent", []))
    monkeypatch.setattr(svc, "read_favorable_read", _spy("favorable", None))
    monkeypatch.setattr(svc, "read_held_run_summaries", _spy("held", None))

    payload = _service(tmp_path).build_payload()
    assert isinstance(payload, dict)
    assert sorted(calls) == ["favorable", "held", "recent"], calls

    def _boom():
        raise AssertionError("a reader ran on the GUI thread")

    monkeypatch.setattr(svc, "read_recent_rows", _boom)
    monkeypatch.setattr(svc, "read_favorable_read", _boom)
    monkeypatch.setattr(svc, "read_held_run_summaries", _boom)

    fresh = _service(tmp_path / "slot_only")
    seen: list[dict] = []
    fresh.snapshotChanged.connect(seen.append)
    handed = {"snapshot_id": "0" * 40, "as_of": "2026-09-04", "cells": [], "verdicts": {}}
    fresh._on_payload_ready(handed)

    assert seen == [handed], seen
    assert not fresh.snapshot_path.exists(), "the slot writes nothing"
    assert not fresh.events_path.exists(), "the slot writes nothing"


# ===========================================================================
# ST6.4 - one snapshot, four surfaces
# ===========================================================================


def test_every_surface_prints_the_same_snapshot_stamp(tmp_path):
    """Item 8. The strip, the tracker banner, Weekend Prep's card and the recap.

    Four surfaces answered "what is working" and nothing tied their answers to
    one reading. Each real render function is driven with ONE snapshot and must
    print its identity, so a screenshot of two of them can be reconciled.
    """
    import away_recap
    import weekend_verdict
    from working_lately import snapshot_stamp
    from ui.panels import setup_tracker_panel
    from ui.widgets.working_lately_strip import WorkingLatelyStrip

    d1, d2 = _sessions(2)
    service = _service(tmp_path)
    _publish(service, d1, _swing_sequence(d1, leader="alpha"))
    snapshot, _ = _publish(service, d2, _swing_sequence(d2, leader="alpha"))
    payload = service.load_snapshot()
    assert payload["snapshot_id"] == snapshot.snapshot_id

    stamp = snapshot_stamp(payload)
    assert snapshot.snapshot_id[:8] in stamp
    assert snapshot_stamp(snapshot) == stamp

    strip = WorkingLatelyStrip()
    try:
        strip.set_snapshot(payload)
        line = strip.line_text()
        assert line.startswith("Working lately (20 sessions):"), line
        assert "alpha" in line
        assert stamp in strip.tooltip_text()
    finally:
        strip.deleteLater()

    panel = setup_tracker_panel.SetupTrackerPanel()
    _load_the_tracker(panel)
    try:
        panel.set_working_lately_snapshot(payload)
        html = setup_tracker_panel._best_now_banner_html(panel)
    finally:
        panel.deleteLater()
    assert stamp in html, html
    assert "panel read" not in html, "the service's snapshot was present; this is not a fallback"

    card = weekend_verdict.build_verdict(working_lately=payload)
    assert stamp in "\n".join(card.rendered())

    recap = away_recap.build_recap(
        session_date=d2.isoformat(),
        digest_swings=[f"{i}. SYM{i} (LONG) | Favorite | AVWAP band bounce" for i in range(1, 10)],
        working_lately=payload,
        leader_events=service.events(),
    )
    assert stamp in recap["summary"], recap["summary"]
    assert len(recap["best_swings"]) == 9, "every ranked row, never a top five"


# ===========================================================================
# ST6.5 - the priority switch reorders and never withholds
# ===========================================================================


def _m5_alert(symbol: str, bounce: str, *, at: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text=at,
        symbol=symbol,
        side=side,
        trigger=f"[S-TIER] {bounce}",
        timeframe="5m",
        tag="green",
        raw_text=f"{bounce} {symbol} ({side.lower()})",
        payload={"feedback": {"bounce_types": bounce, "symbol": symbol, "direction": side.lower()}},
    )


def _set_switch(on: bool) -> None:
    import project_paths

    project_paths.save_local_setting("prioritise_working_lately", bool(on))
    project_paths.invalidate_local_settings_cache()


def test_the_switch_reorders_three_lists_and_shows_exactly_the_same_rows(tmp_path):
    """Item 7. The identical-visible-rows test CLAUDE.md says is owed WITH the switch.

    Three surfaces, each driven through its real path, each run twice. The set
    of visible names is identical, the order is not, and the two things that
    DO withhold - the M5 bar's repeat fold and the review pane's movers-only
    filter - produce byte-identical results both ways.
    """
    import project_paths
    from ui.models.setup import SetupRow
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    from ui.widgets import m5_alert_bar as bar_module
    from working_lately import alert_priority_key

    assert not project_paths.get_local_setting("prioritise_working_lately", False), (
        "the switch defaults OFF"
    )

    day_order = [("vwap_reclaim", "LONG"), ("ema_15", "LONG")]
    swing_order = [("SHORT", "beta"), ("LONG", "alpha")]

    posts = [
        ("AAA", "ema_15", "07:01:00"),
        ("BBB", "vwap_reclaim", "07:02:00"),
        ("CCC", "ema_15", "07:03:00"),
        ("AAA", "ema_15", "07:04:00"),  # a repeat: folds into AAA's row with x2
    ]
    assert alert_priority_key(_m5_alert("AAA", "ema_15", at="07:01:00")) == ("ema_15", "LONG")

    def _bar_run(switch_on: bool):
        _set_switch(switch_on)
        bar = bar_module.M5AlertBar()
        try:
            bar.set_working_lately_order(day_order)
            for symbol, bounce, at in posts:
                bar.post(_m5_alert(symbol, bounce, at=at))
            symbols = [
                str(getattr(a, "symbol", "")) for a in bar.alerts()
            ]
            folds = {
                str(getattr(bar.list.item(i).data(bar_module._ALERT_ROLE), "symbol", "")):
                int(bar.list.item(i).data(bar_module._REPEAT_ROLE) or 1)
                for i in range(bar.list.count())
            }
            return symbols, folds
        finally:
            bar.deleteLater()

    off_symbols, off_folds = _bar_run(False)
    on_symbols, on_folds = _bar_run(True)
    assert off_symbols == ["AAA", "CCC", "BBB"], off_symbols
    assert on_symbols == ["BBB", "AAA", "CCC"], on_symbols
    assert sorted(off_symbols) == sorted(on_symbols)
    assert off_folds == on_folds == {"AAA": 2, "CCC": 1, "BBB": 1}

    # ---- the WAITING review list, re-sorted where the panel picks the next chart
    from ui.panels.alert_center_panel import PREV_DAY_CLOSED

    def _queue_run(switch_on: bool):
        _set_switch(switch_on)
        panel = AlertCenterPanel(ignored_symbols_path=tmp_path / f"ign-{switch_on}.json")
        try:
            panel.set_working_lately_order(day_order)
            panel._review_movers_only = True
            panel._review_chart_state = lambda alert: (  # noqa: E731
                PREV_DAY_CLOSED if alert.symbol == "DDD" else "shows"
            )
            panel._review_queue = [
                _m5_alert(symbol, bounce, at=at)
                for symbol, bounce, at in (
                    ("AAA", "ema_15", "07:01:00"),
                    ("DDD", "ema_15", "07:02:00"),
                    ("BBB", "vwap_reclaim", "07:03:00"),
                    ("CCC", "ema_15", "07:04:00"),
                    ("EEE", "vwap_reclaim", "07:05:00"),
                )
            ]
            shown = []
            while panel._review_queue or panel._current_review_alert is not None:
                panel._advance_review_queue()
                current = panel._current_review_alert
                if current is None:
                    break
                shown.append(current.symbol)
            return shown, sorted(panel._hidden_inside_range)
        finally:
            panel.deleteLater()

    off_shown, off_hidden = _queue_run(False)
    on_shown, on_hidden = _queue_run(True)
    assert off_shown == ["AAA", "BBB", "CCC", "EEE"], off_shown
    assert on_shown == ["BBB", "EEE", "AAA", "CCC"], on_shown
    assert sorted(off_shown) == sorted(on_shown)
    assert off_hidden == on_hidden == ["DDD"], (off_hidden, on_hidden)

    # ---- the Master AVWAP setups table
    rows = [
        SetupRow(symbol="NVDA", side="LONG", score=90.0, bucket="favorite_setup",
                 raw={"setup_family": "alpha"}),
        SetupRow(symbol="TSLA", side="SHORT", score=80.0, bucket="favorite_setup",
                 raw={"setup_family": "beta"}),
        SetupRow(symbol="AMD", side="LONG", score=70.0, bucket="favorite_setup",
                 raw={"setup_family": "alpha"}),
    ]

    def _table_run(switch_on: bool):
        _set_switch(switch_on)
        panel = MasterAvwapPanel(None)
        try:
            panel.set_working_lately_order(swing_order)
            panel.set_rows(list(rows))
            return [row.symbol for row in panel.filtered_rows()]
        finally:
            panel.deleteLater()

    off_table = _table_run(False)
    on_table = _table_run(True)
    assert off_table == ["NVDA", "TSLA", "AMD"], off_table
    assert on_table == ["TSLA", "NVDA", "AMD"], on_table
    assert sorted(off_table) == sorted(on_table)

    _set_switch(False)


def _load_the_tracker(panel) -> None:
    """G7 trigger: the Setup Tracker's read is no longer a side effect of
    building the widget, so the test asks for it.

    `tests.conftest.refresh_setup_tracker` calls the panel's own `refresh()` -
    the slot the Refresh button calls - and, once G7.2 moves the twelve export
    reads onto a worker, waits for the render that lands on the Qt thread. It is
    a trigger and nothing else: no assertion moved with it.
    """
    from tests.conftest import refresh_setup_tracker

    refresh_setup_tracker(panel)
