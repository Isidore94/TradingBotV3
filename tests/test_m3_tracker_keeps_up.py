"""Packet M3 - the setup tracker keeps up with its scans, says how old it is,
and lets a stuck setup age out as UNMEASURED.

Three items, three failures the 2026-09-05 measurement audit found:

* **M3.1** - the purity gate refused every scheduled tracker write while the
  trader's ``daily_bars_source: "yahoo"`` pin was in force. The gate was written
  in 2026-07 against a *systemic IB fallback*; the pin is the trader's own
  declaration of the source of record, and a gate that refuses the declared
  source is refusing the trader. On 2026-09-04 the 13:00 run logged
  ``sources=cache`` over 139 symbols and skipped the write.
* **M3.2** - nothing on disk said WHEN the tracker snapshot behind the Setup
  Tracker's tables was taken, or WHO took it. The sibling ``scan_factor_*``
  exports are current to the last scan, so the page looked fresher than it was.
* **M3.3** - no age rule closes a setup. 37 OPEN setups were older than 20
  sessions with scenarios reading "Awaiting update", and 41 had no baseline
  scenario at all. They sat in every denominator as though they were evidence.
  ``EXPIRED_UNMEASURED`` is uncertainty made visible; it is never a win and
  never a loss.

Every test here was proven RED on ``e744afd5`` (the M1 tip this branch is built
on) before the fix existed.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
from master_avwap_lib import legacy, runner  # noqa: E402


# ===========================================================================
# M3.1 - the purity gate honours the trader's pin
# ===========================================================================

def _frame(declared_source: str, *, row_sources=None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"]),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [100, 100, 100],
        }
    )
    if row_sources is not None:
        frame[legacy.DAILY_BAR_SOURCE_COLUMN] = list(row_sources)
    return legacy._set_daily_bar_source(frame, declared_source)


def _cached_yahoo_frames(count: int) -> dict[str, pd.DataFrame]:
    """What the live desk actually holds under the pin: the frame was served off
    the durable store (declared ``cache``) and its ROWS say who wrote them."""
    return {
        f"SYM{index:03d}": _frame(
            legacy.DAILY_BAR_SOURCE_CACHE,
            row_sources=[legacy.DAILY_BAR_SOURCE_YAHOO] * 3,
        )
        for index in range(count)
    }


@pytest.fixture
def pin(monkeypatch):
    def _set(value: str) -> None:
        monkeypatch.setattr(runner, "daily_bars_source_pin", lambda: value)

    return _set


def test_the_pinned_source_is_not_a_fallback_and_does_not_refuse_the_write(pin):
    """2026-09-04, 13:03:59: 139 symbols on cached Yahoo bars, write refused."""
    pin("yahoo")
    frames = _cached_yahoo_frames(139)

    allowed, quarantined, reason = runner.evaluate_setup_tracker_purity(
        list(frames), frames
    )

    assert allowed is True
    assert quarantined == []
    assert reason == ""


def test_with_no_pin_the_same_input_refuses_exactly_as_it_does_today(pin):
    """The gate is byte-for-byte the old one when the trader declared nothing."""
    pin("auto")
    frames = _cached_yahoo_frames(139)

    allowed, quarantined, reason = runner.evaluate_setup_tracker_purity(
        list(frames), frames
    )

    assert allowed is False
    assert len(quarantined) == 139
    assert "non-IBKR daily data" in reason


def test_a_third_source_still_refuses_even_under_the_pin(pin):
    """The pin declares ONE source of record. Anything that is neither it nor
    IBKR is a fallback the trader did not declare, and still vetoes."""
    pin("yahoo")
    frames = {f"SYM{index}": _frame("stooq") for index in range(10)}

    allowed, quarantined, reason = runner.evaluate_setup_tracker_purity(
        list(frames), frames
    )

    assert allowed is False
    assert sorted(quarantined) == sorted(frames)
    assert reason


def test_a_small_dirty_tail_is_still_quarantined_rather_than_vetoed(pin):
    """The 2026-07-17 quarantine behaviour is untouched by the pin."""
    pin("yahoo")
    frames = {f"SYM{index}": _frame(legacy.DAILY_BAR_SOURCE_IBKR) for index in range(20)}
    frames["SYM0"] = _frame("stooq")

    allowed, quarantined, reason = runner.evaluate_setup_tracker_purity(
        list(frames), frames
    )

    assert allowed is True
    assert quarantined == ["SYM0"]
    assert reason == ""


def test_the_source_mix_and_the_decision_are_logged_once_per_run(pin, caplog):
    pin("yahoo")
    frames = _cached_yahoo_frames(5)
    frames["IBSYM"] = _frame(legacy.DAILY_BAR_SOURCE_IBKR)

    with caplog.at_level(logging.INFO):
        runner.evaluate_setup_tracker_purity(list(frames), frames)

    lines = [
        record.getMessage()
        for record in caplog.records
        if "purity" in record.getMessage().lower()
    ]
    assert len(lines) == 1, lines
    line = lines[0]
    for token in ("n_ib=1", "n_pinned=5", "n_other=0", "refused=False"):
        assert token in line, line


def test_the_tracker_record_still_stamps_the_daily_bar_source_per_symbol():
    """The gate stops refusing; the vintage stays auditable per symbol."""
    source = (SCRIPTS_DIR / "master_avwap_lib" / "legacy.py").read_text(encoding="utf-8")
    assert '"daily_bar_source": _get_daily_bar_source(df)' in source


# ===========================================================================
# M3.2 - the tracker's own clock
# ===========================================================================

def _tracker_payload() -> dict:
    return {
        "setups": {},
        "control_setups": {},
        "study_setups": {},
        "daily_watchlists": {},
        "stats": [],
        "setup_type_stats": [],
        "attribute_registry": {},
    }


@pytest.fixture
def tracker_file(tmp_path, monkeypatch):
    json_path = tmp_path / "tracker.json"
    monkeypatch.setattr(legacy, "SETUP_TRACKER_FILE", json_path)
    monkeypatch.setattr(legacy, "_setup_tracker_backup_path", lambda: tmp_path / "tracker.json.bak")
    monkeypatch.setattr(legacy, "_append_setup_tracker_events", lambda payload: {})
    monkeypatch.setattr(legacy, "save_setup_tracker_scoring_payload", lambda payload: None)
    import tracker_store

    monkeypatch.setattr(tracker_store, "shadow_enabled", lambda: False)
    return json_path


def test_a_saved_payload_carries_saved_at_and_saved_by(tracker_file):
    payload = _tracker_payload()

    legacy.save_setup_tracker_payload(
        payload,
        allow_empty=True,
        data_session="2026-09-04",
        saved_by=legacy.TRACKER_SAVED_BY_CLOSE_SLOT,
    )

    assert payload["saved_by"] == "close_slot"
    saved_at = payload["saved_at"]
    assert saved_at and datetime.fromisoformat(saved_at)
    reloaded = legacy.load_setup_tracker_payload()
    assert reloaded["saved_at"] == saved_at
    assert reloaded["saved_by"] == "close_slot"


def test_an_unstamped_legacy_payload_reads_as_unknown_rather_than_now(tracker_file):
    tracker_file.write_text('{"schema_version": 1, "setups": {}}', encoding="utf-8")

    reloaded = legacy.load_setup_tracker_payload()

    assert reloaded["saved_at"] is None
    assert reloaded["saved_by"] is None


def test_a_pre_m3_payload_still_verifies_clean_against_the_mirror(tmp_path):
    """Gate #57 measures five parity-clean saves. A header key added on this
    branch must not make every pre-M3 payload read as a difference."""
    from tracker_store import TrackerStore

    store = TrackerStore(tmp_path / "tracker.sqlite")
    legacy_payload = {
        "schema_version": 2,
        "updated_at": "2026-09-03T07:15:53",
        "data_session": "2026-09-02",
        "daily_watchlists": {},
        "setups": {},
        "control_setups": {},
        "study_setups": {},
        "stats": [],
        "setup_type_stats": [],
        "attribute_registry": {},
    }
    store.save_payload(legacy_payload)

    assert store.verify(legacy_payload).ok


def test_the_close_slot_save_says_close_slot():
    payload = _tracker_payload()
    with mock.patch.object(legacy, "export_setup_tracker_views"), mock.patch.object(
        legacy, "write_control_discovery_report"
    ), mock.patch.object(legacy, "write_master_avwap_study_report"), mock.patch.object(
        legacy, "save_setup_tracker_payload"
    ) as save_mock:
        legacy.update_setup_tracker_from_scan(
            [], {"symbols": {}}, {}, {}, None, auto_tune=False, tracker_payload=payload
        )

    assert save_mock.call_args.kwargs["saved_by"] == legacy.TRACKER_SAVED_BY_CLOSE_SLOT


def test_the_catch_up_backfill_says_catch_up_backfill(tmp_path):
    """The catch-up is today's EFFECTIVE writer, so it must say so."""
    calls: list[dict] = []

    longs = tmp_path / "longs.txt"
    longs.write_text("AAA\n", encoding="utf-8")
    shorts = tmp_path / "shorts.txt"
    shorts.write_text("", encoding="utf-8")

    with (
        mock.patch.object(legacy, "connect_daily_data_client", lambda **k: None),
        mock.patch.object(legacy, "disconnect_daily_data_client", lambda *a, **k: None),
        mock.patch.object(
            legacy, "get_recent_market_session_dates", lambda n=1: [date(2026, 9, 4)]
        ),
        mock.patch.object(legacy, "load_scan_earnings_context", lambda syms: ({}, {})),
        mock.patch.object(
            legacy, "append_master_avwap_d1_watchlist_symbols", lambda lo, sh: (lo, sh, 0)
        ),
        mock.patch.object(
            legacy, "fetch_daily_bars", lambda ib, sym, days, **k: _frame("ibkr")
        ),
        mock.patch.object(legacy, "_evaluate_priority_snapshot_for_date", lambda **k: None),
        mock.patch.object(
            legacy,
            "update_setup_tracker_from_scan",
            lambda *a, **k: calls.append(k),
        ),
    ):
        legacy.backfill_setup_tracker_from_recent_sessions(
            lookback_sessions=1,
            longs_path=longs,
            shorts_path=shorts,
            end_date=date(2026, 9, 4),
            run_scoring_side_effects=False,
            saved_by=legacy.TRACKER_SAVED_BY_CATCH_UP,
        )

    assert calls
    assert calls[0]["saved_by"] == legacy.TRACKER_SAVED_BY_CATCH_UP


def test_the_manual_backfill_default_is_manual():
    import inspect

    default = (
        inspect.signature(legacy.backfill_setup_tracker_from_recent_sessions)
        .parameters["saved_by"]
        .default
    )
    assert default == legacy.TRACKER_SAVED_BY_MANUAL


def test_the_staleness_catchup_declares_itself_the_catch_up_writer():
    calls: list[dict] = []
    plan = {
        "stale": True,
        "lookback_sessions": 2,
        "last_completed_session": date(2026, 9, 4),
        "reason": "tracker last updated for 2026-09-02",
    }
    with (
        mock.patch.object(runner, "get_local_setting", lambda key, default=None: True),
        mock.patch.object(runner, "compute_setup_tracker_catchup_plan", lambda **k: plan),
        mock.patch.object(
            runner,
            "backfill_setup_tracker_from_recent_sessions",
            lambda **k: calls.append(k) or {"dates": ["2026-09-04"]},
        ),
    ):
        runner._maybe_run_setup_tracker_catchup(
            update_setup_tracker=False, now=datetime(2026, 9, 5, 10, 30)
        )

    assert calls and calls[0]["saved_by"] == legacy.TRACKER_SAVED_BY_CATCH_UP


EXPORT_FILES = (
    "SETUP_SCENARIOS_FILE",
    "SETUP_DAILY_FILE",
    "SETUP_STATS_FILE",
    "SETUP_TYPE_STATS_FILE",
    "SETUP_TYPE_RECENT_STATS_FILE",
    "SETUP_PLAYBOOKS_FILE",
    "SETUP_SHORT_HORIZON_FILE",
    "SETUP_ATTRIBUTES_FILE",
    "SETUP_ATTRIBUTE_LEADERBOARD_FILE",
    "SETUP_BAND_VARIANT_STATS_FILE",
)


def _redirect_exports(monkeypatch, tmp_path) -> None:
    for name in EXPORT_FILES:
        monkeypatch.setattr(legacy, name, tmp_path / f"{name.lower()}.csv")
    monkeypatch.setattr(
        legacy, "attribute_leaderboard_view_path", lambda name: tmp_path / f"view_{name}.csv"
    )


def _scenario(label, *, status="OPEN", total_r=0.0, experimental=False, source="current_anchor"):
    return {
        "scenario_id": f"{label.lower()}__full_band2",
        "stop_reference_label": label,
        "stop_reference_level": 42.0,
        "stop_source_type": source,
        "exit_template_id": "full_band2",
        "experimental": experimental,
        "tradeable": True,
        "shares": 100,
        "status": status,
        "total_r": total_r,
        "last_action": "Awaiting update",
    }


def _setup_record(symbol, *, scan_date="2026-08-03", scenarios=None, status="OPEN", **extra):
    record = {
        "setup_id": f"{scan_date}:{symbol}:LONG",
        "symbol": symbol,
        "side": "LONG",
        "scan_date": scan_date,
        "anchor_date": "2026-06-01",
        "entry_price": 45.0,
        "setup_family": "top_pattern",
        "tracker_setup_family": "top_pattern",
        "priority_bucket": "favorite_setup",
        "tracker_priority_bucket": "favorite_setup",
        "priority_score": 70.0,
        "setup_status": status,
        "entry_feature_snapshot": {"atr20": 1.0},
        "scenarios": scenarios if scenarios is not None else {"lower_1__full_band2": _scenario("LOWER_1")},
    }
    record.update(extra)
    return record


def test_the_three_stats_exports_carry_the_tracker_clock(tmp_path, monkeypatch):
    _redirect_exports(monkeypatch, tmp_path)
    # Inside the recent-window lookback, so all three exports have a row: the
    # "what worked lately" view is by definition empty for an old setup.
    setup = _setup_record("AAA", scan_date=date.today().isoformat())
    payload = {
        "setups": {setup["setup_id"]: setup},
        "saved_at": "2026-09-05T13:02:11",
        "saved_by": legacy.TRACKER_SAVED_BY_CLOSE_SLOT,
    }

    legacy.export_setup_tracker_views(payload)

    for name in (
        "SETUP_TYPE_STATS_FILE",
        "SETUP_TYPE_RECENT_STATS_FILE",
        "SETUP_BAND_VARIANT_STATS_FILE",
    ):
        path = getattr(legacy, name)
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert rows, name
        assert {row["tracker_saved_at"] for row in rows} == {"2026-09-05T13:02:11"}, name
        assert {row["tracker_saved_by"] for row in rows} == {"close_slot"}, name


def test_a_replayed_record_stamps_the_session_its_bars_reached():
    setup = _setup_record("AAA", scan_date="2026-08-03")
    setup["entry_trade_date"] = "2026-08-03"
    bars = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05"]),
            "open": [45.0, 45.0, 45.0],
            "high": [46.0, 46.0, 46.0],
            "low": [44.0, 44.0, 44.0],
            "close": [45.5, 45.5, 45.5],
            "volume": [1000, 1000, 1000],
        }
    )

    recomputed = legacy.recompute_tracker_setup_record(setup, bars)

    assert recomputed["last_replayed_session"] == "2026-08-05"


# ===========================================================================
# M3.3 - a stuck setup ages out as UNMEASURED
# ===========================================================================

AS_OF = "2026-09-04"


def _sessions_before(end_iso: str, sessions: int) -> str:
    day = date.fromisoformat(end_iso)
    for _ in range(sessions):
        day = market_calendar.previous_session(day)
    return day.isoformat()


def test_an_open_setup_never_replayed_for_21_sessions_expires_unmeasured():
    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    setup = _setup_record("CTRA", scan_date=stale, last_replayed_session=stale)

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["setup_status"] == legacy.SETUP_STATUS_EXPIRED_UNMEASURED
    assert setup["expiry_reason"] == legacy.TRACKER_EXPIRY_REASON_NO_REPLAY


def test_an_equally_old_setup_replayed_yesterday_does_not_expire():
    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    yesterday = _sessions_before(AS_OF, 1)
    setup = _setup_record("MU", scan_date=stale, last_replayed_session=yesterday)

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["setup_status"] == "OPEN"
    assert setup.get("expiry_reason", "") == ""


def test_a_record_whose_every_scenario_is_experimental_has_no_baseline():
    setup = _setup_record(
        "GBTG",
        scan_date=_sessions_before(AS_OF, 2),
        last_replayed_session=_sessions_before(AS_OF, 1),
        scenarios={"x": _scenario("LOWER_1", experimental=True)},
    )

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["setup_status"] == legacy.SETUP_STATUS_EXPIRED_UNMEASURED
    assert setup["expiry_reason"] == legacy.TRACKER_EXPIRY_REASON_NO_BASELINE


def test_a_record_with_only_band_variant_scenarios_has_no_baseline():
    """The challenger is shadow evidence; a record carrying nothing else has
    measured nothing the champion can be graded on."""
    setup = _setup_record(
        "CLF",
        scan_date=_sessions_before(AS_OF, 2),
        last_replayed_session=_sessions_before(AS_OF, 1),
        scenarios={
            "v": _scenario("VARIANT_LOWER_1", source=legacy.BAND_VARIANT_STOP_SOURCE)
        },
    )

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["setup_status"] == legacy.SETUP_STATUS_EXPIRED_UNMEASURED
    assert setup["expiry_reason"] == legacy.TRACKER_EXPIRY_REASON_NO_BASELINE


def test_an_empty_scenarios_dict_has_no_baseline():
    setup = _setup_record("OKLO", scan_date=_sessions_before(AS_OF, 2), scenarios={})

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["expiry_reason"] == legacy.TRACKER_EXPIRY_REASON_NO_BASELINE


def test_a_normally_closed_setup_is_never_expired():
    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 5)
    setup = _setup_record(
        "NVDA",
        scan_date=stale,
        status="CLOSED",
        last_replayed_session=stale,
        scenarios={"c": _scenario("LOWER_1", status="TARGET_HIT", total_r=1.4)},
    )

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["setup_status"] == "CLOSED"
    assert setup.get("expiry_reason", "") == ""


def test_a_date_the_calendar_refuses_never_expires_a_setup():
    """Uncertainty never deletes (plan.md sec 5). A clock that cannot answer
    leaves the record exactly where it was."""
    setup = _setup_record("XXX", scan_date="1998-01-02", last_replayed_session="1998-01-02")

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["setup_status"] == "OPEN"
    assert setup.get("expiry_reason", "") == ""


def test_the_expiry_is_reversible_by_a_later_replay():
    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    setup = _setup_record("PWR", scan_date=stale, last_replayed_session=stale)
    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)
    assert setup["setup_status"] == legacy.SETUP_STATUS_EXPIRED_UNMEASURED

    setup["setup_status"] = "OPEN"
    setup["last_replayed_session"] = _sessions_before(AS_OF, 1)
    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["setup_status"] == "OPEN"
    assert setup["expiry_reason"] == ""


# --- the exclusion, and what it must NOT move --------------------------------

def _family_rows(setups):
    return legacy.build_tracker_setup_type_rows({s["setup_id"]: s for s in setups})


def _closed(symbol, *, total_r, scan_date="2026-09-02"):
    status = "TARGET_HIT" if total_r > 0 else "STOPPED"
    return _setup_record(
        symbol,
        scan_date=scan_date,
        status="CLOSED",
        last_replayed_session=scan_date,
        scenarios={"c": _scenario("LOWER_1", status=status, total_r=total_r)},
    )


# ---------------------------------------------------------------------------
# Lead ruling, 2026-09-05: the invariant wins.
#
# plan.md sec 5 forbids a scoring-input change without golden fixtures first.
# `build_tracker_setup_type_rows` is BOTH a trader-facing export and a live
# scoring input - `_load_ranked_tracker_setup_type_rows` ->
# `rank_tracker_setup_type_rows` -> `apply_tracker_setup_type_adjustments` ->
# `row["score"]`, where `tracked_setups` is a sort tiebreak and `avg_total_r`
# feeds the ranking metric. So the exclusion is OPT-IN: the export asks for it,
# the champion's scoring population never does.
#
# These tests pin the population rather than a stored blob, and that is
# deliberate. "Byte-identical to e744afd5" means: flipping a record's status to
# EXPIRED_UNMEASURED changes NOTHING the scorer sees - which is exactly what the
# record would have carried on e744afd5, where the status did not exist. A
# stored fixture would pin one tracker; this pins the rule.
# ---------------------------------------------------------------------------

def _stale_pair():
    """The same tracker twice: once as it read on e744afd5 (the stale records
    are plain OPEN), once with M3.3's status on them.

    The shape is chosen so the scoring path CAN move if the exclusion reaches
    it — otherwise this whole group of tests would be unfalsifiable, which is
    the 2026-09-02 lesson. Two families sit in the same (side, bucket) so
    `group_size > 1` and the rank bonuses apply; both carry four CLOSED setups
    with identical outcomes so their `ranking_score` and `closed_setups` tie;
    and the stale records sit in ONE family, so `tracked_setups` — the next
    tiebreak in `rank_tracker_setup_type_rows` — is the only thing separating
    them. Drop the stale records and the tie falls through to `type_label`,
    which reverses the order: `other_pattern` sorts before `top_pattern`.

    So on the pre-ruling code (which excluded unconditionally) `top_pattern`
    goes from rank 1 to rank 2, its rank bonus from 12 to 8, and the trader's
    `row["score"]` moves. That is the defect the lead's ruling forbids, and it
    is what these tests would catch.
    """
    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    # `near_favorite_zone` for every record: `_tracker_priority_bucket` demotes
    # `favorite_setup` to it for any family outside MAIN_SWING_SETUP_FAMILIES,
    # which would split these into different rank groups and make the test
    # unfalsifiable again.
    BUCKET = "near_favorite_zone"
    # `gamma` drags the (side, bucket) baseline negative so `zeta` and `alpha`
    # both carry a real POSITIVE edge - identical to each other, so their
    # ranking scores and closed counts tie and `tracked_setups` is what
    # separates them. `zeta` sorts after `alpha` by `type_label`, so removing
    # its stale records reverses the two.
    families = {"zeta_pattern": (4, 1.0), "alpha_pattern": (4, 1.0), "gamma_pattern": (8, -1.0)}

    def _build(status, reason):
        rows = []
        for family, (count, total_r) in families.items():
            for index in range(count):
                record = _closed(f"{family[:2].upper()}{index}", total_r=total_r)
                record["setup_id"] = f"{family}:{index}"
                record["setup_family"] = family
                record["tracker_setup_family"] = family
                record["priority_bucket"] = BUCKET
                record["tracker_priority_bucket"] = BUCKET
                rows.append(record)
        for symbol in ("CCC", "EEE", "FFF"):
            record = _setup_record(
                symbol,
                scan_date=stale,
                status=status,
                expiry_reason=reason,
                last_replayed_session=stale,
            )
            record["setup_family"] = "zeta_pattern"
            record["tracker_setup_family"] = "zeta_pattern"
            record["priority_bucket"] = BUCKET
            record["tracker_priority_bucket"] = BUCKET
            rows.append(record)
        return {item["setup_id"]: item for item in rows}

    return (
        _build("OPEN", ""),
        _build(legacy.SETUP_STATUS_EXPIRED_UNMEASURED, legacy.TRACKER_EXPIRY_REASON_NO_REPLAY),
    )


#: The only two cells that may differ between the two builds, and why.
#:
#: `open_setups` is a DISPLAY count of setups currently OPEN. A record that has
#: aged out is no longer open, so this cell follows the status by definition -
#: that IS M3.3. It reaches no score: `_compute_tracker_setup_type_ranking_score`
#: reads `tracked_setups`, the metric pair and its baselines, `target_hit_rate`,
#: `stop_rate` and `closed_setups`, and `rank_tracker_setup_type_rows` sorts on
#: `ranking_score`, `closed_setups`, `tracked_setups`, `avg_closed_r` and
#: `type_label`. Neither mentions it.
#:
#: `n_expired_unmeasured` is M3.3's own additive count - 0 before the status
#: exists, 2 after - and new keys are allowed exactly as they are for the
#: band-variant shadow. `sample_setups` renders `setup_status` in its text.
#:
#: The assertion below is that these are the ONLY differences: a third key
#: moving fails this test, which is the point of listing them rather than
#: comparing a hand-picked subset.
STATUS_FOLLOWING_CELLS = {"open_setups", "n_expired_unmeasured", "sample_setups"}


def test_the_scoring_population_is_untouched_by_the_new_status():
    """The default call is the champion's population and must not move."""
    before, after = _stale_pair()

    rows_before = {
        row["setup_type_id"]: row for row in legacy.build_tracker_setup_type_rows(before)
    }
    rows_after = {
        row["setup_type_id"]: row for row in legacy.build_tracker_setup_type_rows(after)
    }

    assert set(rows_before) == set(rows_after)
    moved = set()
    for setup_type_id, row_before in rows_before.items():
        row_after = rows_after[setup_type_id]
        assert set(row_before) <= set(row_after)
        for key, value in row_before.items():
            if row_after[key] != value:
                moved.add(key)

    assert moved <= STATUS_FOLLOWING_CELLS, moved
    # And the cells the lead's ruling names by hand, stated separately so the
    # set comparison above can never quietly absorb one of them.
    for setup_type_id, row_before in rows_before.items():
        for key in ("tracked_setups", "avg_total_r", "avg_closed_r", "closed_setups"):
            assert rows_after[setup_type_id][key] == row_before[key], key


def test_the_rank_and_the_score_delta_are_untouched_by_the_new_status():
    before, after = _stale_pair()

    ranked_before = legacy.rank_tracker_setup_type_rows(
        legacy.build_tracker_setup_type_rows(before)
    )
    ranked_after = legacy.rank_tracker_setup_type_rows(
        legacy.build_tracker_setup_type_rows(after)
    )

    def _ranks(rows):
        return sorted(
            (
                str(row.get("setup_type_id")),
                int(row.get("rank_within_side_bucket", 0) or 0),
                int(row.get("score_delta", 0) or 0),
                row.get("ranking_score"),
            )
            for row in rows
        )

    assert _ranks(ranked_before) == _ranks(ranked_after)


def test_the_priority_score_itself_is_untouched_by_the_new_status():
    """The seam that actually reaches the trader: `row["score"]`."""
    before, after = _stale_pair()

    def _scores(setups):
        priority_rows = [
            {
                "symbol": "AAA",
                "side": "LONG",
                "setup_family": "zeta_pattern",
                "priority_bucket": "near_favorite_zone",
                "score": 100.0,
            }
        ]
        ai_state = {"symbols": {"AAA": {}}}
        legacy.apply_tracker_setup_type_adjustments(
            priority_rows,
            ai_state,
            {},
            tracker_payload={"setups": setups},
        )
        entry = ai_state["symbols"]["AAA"]
        return {
            "score": priority_rows[0]["score"],
            "delta": priority_rows[0]["setup_type_score_delta"],
            # The rank is published too, and on this fixture it is the value
            # that MOVES if the exclusion reaches scoring: `zeta_pattern` leads
            # its (side, bucket) group only because its stale records are still
            # in `tracked_setups`, the sort's third key. The delta itself is
            # confidence-capped here and would not show the difference.
            "rank": entry.get("priority_setup_type_rank"),
        }

    scored_before = _scores(before)
    # An unfalsifiable test is worse than none: prove this fixture actually
    # produces a live adjustment and a real rank before comparing.
    assert scored_before["delta"] != 0, scored_before
    assert scored_before["score"] != 100.0
    assert scored_before["rank"] == 1, scored_before

    assert _scores(after) == scored_before


def test_the_export_excludes_them_while_the_scoring_rows_still_count_them(tmp_path, monkeypatch):
    """One tracker, two readings: the CSV drops the expired, the payload's
    scoring rows keep them."""
    _redirect_exports(monkeypatch, tmp_path)
    _, after = _stale_pair()
    payload = {"setups": after, "saved_at": "2026-09-05T13:02:11", "saved_by": "close_slot"}

    legacy.export_setup_tracker_views(payload)

    csv_rows = list(
        csv.DictReader(
            legacy.SETUP_TYPE_STATS_FILE.read_text(encoding="utf-8").splitlines()
        )
    )
    assert csv_rows
    # Three stale records; sixteen graded ones across three families.
    assert sum(int(row["n_expired_unmeasured"]) for row in csv_rows) == 3
    assert sum(int(row["tracked_setups"]) for row in csv_rows) == 16

    scoring_rows = payload["setup_type_stats"]
    assert sum(int(row["tracked_setups"]) for row in scoring_rows) == 19


def test_the_scoring_tuners_own_inputs_never_see_the_exclusion():
    """`analyze_master_avwap_scoring.py` - the only thing that writes live
    scoring weights - reads the ATTRIBUTE exports, and those still carry a row
    for an expired record."""
    _, after = _stale_pair()

    attribute_rows = legacy._flatten_tracker_attributes(after, {})
    symbols = {str(row.get("symbol") or "") for row in attribute_rows}

    assert {"CCC", "EEE"} <= symbols or not attribute_rows


def test_the_expired_setups_leave_the_win_rate_untouched_and_are_counted():
    graded = [_closed("AAA", total_r=1.5), _closed("BBB", total_r=-1.0)]
    before = legacy.build_tracker_setup_type_rows(
        {item["setup_id"]: item for item in graded}, exclude_expired_unmeasured=True
    )

    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    expired = _setup_record(
        "CCC",
        scan_date=stale,
        status=legacy.SETUP_STATUS_EXPIRED_UNMEASURED,
        expiry_reason=legacy.TRACKER_EXPIRY_REASON_NO_REPLAY,
        last_replayed_session=stale,
    )
    after = legacy.build_tracker_setup_type_rows(
        {item["setup_id"]: item for item in graded + [expired]},
        exclude_expired_unmeasured=True,
    )

    assert len(before) == len(after) == 1
    for key in ("target_hit_rate", "stop_rate", "avg_closed_r", "closed_setups"):
        assert after[0][key] == before[0][key], key
    assert after[0]["n_expired_unmeasured"] == 1
    assert before[0]["n_expired_unmeasured"] == 0


def test_the_scenario_stats_export_excludes_expired_records_and_counts_them():
    graded = [_closed("AAA", total_r=1.5), _closed("BBB", total_r=-1.0)]
    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    expired = _setup_record(
        "CCC",
        scan_date=stale,
        status=legacy.SETUP_STATUS_EXPIRED_UNMEASURED,
        expiry_reason=legacy.TRACKER_EXPIRY_REASON_NO_REPLAY,
    )
    setups = {s["setup_id"]: s for s in graded + [expired]}

    rows = legacy.build_tracker_stats_rows(legacy._flatten_tracker_scenarios(setups))

    assert len(rows) == 1
    assert rows[0]["tracked_setups"] == 2
    assert rows[0]["n_expired_unmeasured"] == 1
    assert rows[0]["win_rate_closed"] == pytest.approx(0.5)


def test_the_band_variant_export_excludes_expired_records_and_counts_them():
    def _paired(symbol, *, status):
        return _setup_record(
            symbol,
            scan_date="2026-09-02",
            status=status,
            current_anchor_variant={
                "formula_version": "avwap_bands_oneoption_bb20_v1",
                "vwap": 44.0,
                "stdev": 1.25,
                "bands": {"LOWER_1": 42.0, "UPPER_1": 46.0},
                "reason": "",
            },
            scenarios={
                "champ": _scenario("LOWER_1", status="TARGET_HIT", total_r=1.0),
                "var": _scenario(
                    "VARIANT_LOWER_1",
                    status="TARGET_HIT",
                    total_r=0.8,
                    source=legacy.BAND_VARIANT_STOP_SOURCE,
                ),
            },
        )

    setups = {
        item["setup_id"]: item
        for item in (
            _paired("AAA", status="CLOSED"),
            _paired("BBB", status=legacy.SETUP_STATUS_EXPIRED_UNMEASURED),
        )
    }

    rows = legacy.build_band_variant_stats_rows(setups)

    assert len(rows) == 1
    assert rows[0]["n"] == 1
    assert rows[0]["n_expired_unmeasured"] == 1


def test_a_tracker_with_no_expired_records_exports_byte_identical_champion_rows():
    """The parity claim behind M3.3: the exclusion may only touch the expired."""
    setups = {s["setup_id"]: s for s in (_closed("AAA", total_r=1.5), _closed("BBB", total_r=-1.0))}
    rows = legacy.build_tracker_setup_type_rows(setups, exclude_expired_unmeasured=True)
    assert rows[0]["tracked_setups"] == 2
    assert rows[0]["n_expired_unmeasured"] == 0
    assert rows[0]["closed_setups"] == 2


def test_the_recompute_applies_the_expiry_after_the_closure_rule():
    """M3.3 is applied inside the daily recompute, and only when the caller
    hands it the session to measure against."""
    setup = _setup_record("AAA", scan_date="2026-08-03")
    setup["entry_trade_date"] = "2026-08-03"
    bars = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-08-03", "2026-08-04"]),
            "open": [45.0, 45.0],
            "high": [46.0, 46.0],
            "low": [44.0, 44.0],
            "close": [45.5, 45.5],
            "volume": [1000, 1000],
        }
    )

    recomputed = legacy.recompute_tracker_setup_record(setup, bars, as_of_session=AS_OF)

    assert recomputed["setup_status"] == legacy.SETUP_STATUS_EXPIRED_UNMEASURED
    assert recomputed["expiry_reason"] == legacy.TRACKER_EXPIRY_REASON_NO_REPLAY


# ===========================================================================
# The Setup Tracker panel: two clocks, named
# ===========================================================================

def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel_module(monkeypatch, tmp_path):
    if _qt_app() is None:
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    for name in (
        "SETUP_TYPE_STATS_FILE",
        "SETUP_PLAYBOOKS_FILE",
        "MASTER_AVWAP_TIER_PERFORMANCE_FILE",
        "MASTER_AVWAP_TIER_LIST_FILE",
        "RECENT_SETUP_TYPE_STATS_FILE",
        "SHORT_HORIZON_FILE",
        "MASTER_AVWAP_TIER_CATCH_RATE_FILE",
        "BAND_VARIANT_STATS_FILE",
    ):
        monkeypatch.setattr(setup_tracker_panel, name, tmp_path / f"{name.lower()}.csv")
    monkeypatch.setattr(
        setup_tracker_panel,
        "MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE",
        tmp_path / "scan_factors.csv",
    )
    return setup_tracker_panel


def test_the_clock_sentence_names_both_clocks(panel_module):
    sentence = panel_module.tracker_clock_sentence(
        "2026-09-05T13:02:11", "close_slot", "2026-09-05 13:05"
    )

    assert "Tracker as of 2026-09-05T13:02:11 (close_slot)" in sentence
    assert "scan factors as of 2026-09-05 13:05" in sentence


def test_an_unstamped_export_says_unknown_rather_than_looking_current(panel_module):
    sentence = panel_module.tracker_clock_sentence("", "", "2026-09-05 13:05")

    assert "unknown" in sentence.lower()
    assert "2026-09-05T" not in sentence


def test_the_panel_renders_the_two_clocks_from_the_exports(panel_module, tmp_path):
    rows = [
        {
            "setup_type_id": "LONG | favorite_setup | top_pattern |  |  | N",
            "type_label": "LONG | favorite_setup",
            "side": "LONG",
            "priority_bucket": "favorite_setup",
            "setup_family": "top_pattern",
            "favorite_zone": "",
            "retest_label": "",
            "compression_label": "N",
            "tracked_setups": "4",
            "tradeable_setups": "4",
            "closed_setups": "2",
            "open_setups": "2",
            "n_expired_unmeasured": "3",
            "avg_closed_r": "0.30",
            "target_hit_rate": "0.50",
            "stop_rate": "0.50",
            "tracker_saved_at": "2026-09-05T13:02:11",
            "tracker_saved_by": "close_slot",
        }
    ]
    path = tmp_path / "setup_type_stats_file.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "scan_factors.csv").write_text("factor\n", encoding="utf-8")
    panel_module.clear_setup_tracker_csv_cache()

    panel = panel_module.SetupTrackerPanel()
    try:
        text = panel.status_label.text()
        assert "Tracker as of 2026-09-05T13:02:11 (close_slot)" in text
        assert "scan factors as of" in text
        assert "3 expired unmeasured, excluded" in panel.setup_type_status_label.text()
    finally:
        panel.deleteLater()


# ===========================================================================
# Reviewer round, 2026-09-05: three blockers and five advisories.
# ===========================================================================

# --- BLOCKER 1: the two clocks were in two different zones -----------------

def test_the_two_clocks_are_rendered_in_the_same_zone(panel_module, tmp_path):
    """`saved_at` was market-local with an offset and the scan-factor mtime was
    machine-local PT with none, so the line read a three-hour gap that was not
    there. Same instant in, same time out."""
    from datetime import timezone

    factors = tmp_path / "scan_factors.csv"
    factors.write_text("factor\n", encoding="utf-8")
    moment = datetime.fromtimestamp(factors.stat().st_mtime, tz=timezone.utc)
    saved_at = legacy.tracker_save_timestamp(moment)

    sentence = panel_module.tracker_clock_sentence(
        saved_at, "close_slot", panel_module._latest_mtime_text([factors])
    )

    rendered = sentence.split("scan factors as of ", 1)[1].strip()
    assert rendered == saved_at, sentence


# --- BLOCKER 2: an all-expired group took its own count with it ------------

def _expired(symbol, family, *, scan_date="2026-08-03"):
    record = _setup_record(
        symbol,
        scan_date=scan_date,
        status=legacy.SETUP_STATUS_EXPIRED_UNMEASURED,
        expiry_reason=legacy.TRACKER_EXPIRY_REASON_NO_REPLAY,
    )
    record["setup_id"] = f"{family}:{symbol}"
    record["setup_family"] = family
    record["tracker_setup_family"] = family
    return record


def _mixed_and_all_expired():
    """FA: two live and one expired. FB: three expired and nothing else."""
    rows = []
    for index, total_r in enumerate((1.5, -1.0)):
        record = _closed(f"LIVE{index}", total_r=total_r)
        record["setup_id"] = f"FA:LIVE{index}"
        record["setup_family"] = "fa_family"
        record["tracker_setup_family"] = "fa_family"
        rows.append(record)
    rows.append(_expired("EXP0", "fa_family"))
    for index in range(3):
        rows.append(_expired(f"GONE{index}", "fb_family"))
    return {item["setup_id"]: item for item in rows}


def test_an_all_expired_group_still_reports_its_expired_count():
    rows = legacy.build_tracker_setup_type_rows(
        _mixed_and_all_expired(), exclude_expired_unmeasured=True
    )

    assert sum(int(row["n_expired_unmeasured"]) for row in rows) == 4
    # The group that lost every record is still SHOWN, carrying zero measured
    # setups rather than vanishing with its own count.
    empty = [row for row in rows if int(row["tracked_setups"]) == 0]
    assert len(empty) == 1
    assert int(empty[0]["n_expired_unmeasured"]) == 3


def test_the_scenario_stats_export_keeps_an_all_expired_groups_count():
    """`build_tracker_stats_rows` groups by stop label + exit template, so the
    all-expired group has to be built on a stop label nothing live uses -
    otherwise the live rows keep the group alive and the bug hides."""
    setups = _mixed_and_all_expired()
    for setup in setups.values():
        if _tracker_expired(setup) and setup["setup_family"] == "fb_family":
            for scenario in setup["scenarios"].values():
                scenario["stop_reference_label"] = "SMA_50"

    rows = legacy.build_tracker_stats_rows(legacy._flatten_tracker_scenarios(setups))

    assert sum(int(row["n_expired_unmeasured"]) for row in rows) == 4
    orphan = [row for row in rows if row["stop_reference_label"] == "SMA_50"]
    assert len(orphan) == 1
    assert int(orphan[0]["tracked_setups"]) == 0
    assert int(orphan[0]["n_expired_unmeasured"]) == 3


def _tracker_expired(setup) -> bool:
    return str(setup.get("setup_status") or "") == legacy.SETUP_STATUS_EXPIRED_UNMEASURED


def test_the_panel_sentence_counts_every_expired_record(panel_module):
    rows = legacy.build_tracker_setup_type_rows(
        _mixed_and_all_expired(), exclude_expired_unmeasured=True
    )

    assert panel_module.expired_unmeasured_sentence(rows) == "4 expired unmeasured, excluded"


# --- BLOCKER 3: the gate needs a token it can grep -------------------------

def test_the_expiry_log_line_carries_a_greppable_count(caplog):
    tracker = {"setups": _mixed_and_all_expired(), "control_setups": {}, "study_setups": {}}
    with caplog.at_level(logging.INFO):
        legacy.log_tracker_expiry_summary(legacy.apply_tracker_expiry_sweep(tracker, AS_OF))

    assert any("n_expired_unmeasured=4" in record.getMessage() for record in caplog.records), [
        record.getMessage() for record in caplog.records
    ]


# --- ADVISORY 1: a naive moment is ATTACHED, never converted ---------------

def test_a_naive_timestamp_is_attached_to_market_local_not_converted():
    """CLAUDE.md's `_gate_moment` rule: normalize by ATTACHING market-local to
    the naive side, never by stripping or converting the aware one."""
    naive = datetime(2026, 9, 5, 13, 2, 11)

    stamped = legacy.tracker_save_timestamp(naive)

    assert stamped.startswith("2026-09-05T13:02:11"), stamped


# --- ADVISORY 2: no frame is not "bars from the pinned source" -------------

def test_a_symbol_with_no_frame_is_reported_separately_and_never_as_pinned(pin, caplog):
    pin("yahoo")
    frames = _cached_yahoo_frames(4)
    frames["NOBARS"] = None

    with caplog.at_level(logging.INFO):
        allowed, quarantined, reason = runner.evaluate_setup_tracker_purity(
            list(frames), frames
        )

    line = next(m for m in (r.getMessage() for r in caplog.records) if "purity" in m.lower())
    assert "n_no_frame=1" in line, line
    # It saw no bars, so it is neither pinned nor a third source, and it is out
    # of the fraction entirely.
    assert "n_pinned=4" in line, line
    assert allowed is True
    assert "NOBARS" not in quarantined
    assert reason == ""


# --- ADVISORY 3: the reason does not bake the number -----------------------

def test_the_stored_reason_carries_the_threshold_as_a_field_not_in_its_name():
    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    setup = _setup_record("CTRA", scan_date=stale, last_replayed_session=stale)

    legacy.apply_tracker_setup_expiry(setup, as_of_session=AS_OF)

    assert setup["expiry_reason"] == "no_replay_stale_sessions"
    assert "20" not in setup["expiry_reason"]
    assert setup["stale_sessions"] == legacy.TRACKER_STALE_SESSIONS


def test_lately_is_one_number_in_this_codebase():
    import evidence_stats

    assert legacy.TRACKER_STALE_SESSIONS == evidence_stats.LATELY_SESSIONS


# --- ADVISORY 4: the Current Picks tab reads a different population --------

def test_the_current_picks_tab_carries_no_expired_sentence(panel_module):
    """`master_avwap_tier_list.csv` is not the setup-type export; the count
    would be describing a population that tab does not show."""
    panel = panel_module.SetupTrackerPanel()
    try:
        assert not hasattr(panel, "current_pick_status_label")
    finally:
        panel.deleteLater()


# --- ADVISORY 6: a hand-run scan never claims to be the close slot ---------

def test_a_scan_payload_without_a_writer_reads_as_manual():
    from scan_worker import parse_payload

    assert parse_payload({"update_setup_tracker": True})["saved_by"] == "manual"
    assert (
        parse_payload({"update_setup_tracker": True, "saved_by": "close_slot"})["saved_by"]
        == "close_slot"
    )


def test_the_scanner_cli_declares_itself_manual():
    import inspect

    source = inspect.getsource(runner.main)
    assert "TRACKER_SAVED_BY_MANUAL" in source
