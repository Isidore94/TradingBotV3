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


def test_the_expired_setups_leave_the_win_rate_untouched_and_are_counted():
    graded = [_closed("AAA", total_r=1.5), _closed("BBB", total_r=-1.0)]
    before = _family_rows(graded)

    stale = _sessions_before(AS_OF, legacy.TRACKER_STALE_SESSIONS + 1)
    expired = _setup_record(
        "CCC",
        scan_date=stale,
        status=legacy.SETUP_STATUS_EXPIRED_UNMEASURED,
        expiry_reason=legacy.TRACKER_EXPIRY_REASON_NO_REPLAY,
        last_replayed_session=stale,
    )
    after = _family_rows(graded + [expired])

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
    rows = _family_rows(list(setups.values()))
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
