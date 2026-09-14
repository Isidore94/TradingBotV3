"""Packet D1C-A items 1 and 2 - the claimed-pick STORE and the horizon.

Trader, 2026-09-14:

    "When I like and claim a D1 setup, it becomes a ranked pick I can follow in
    Master AVWAP Setups. ... Determine the trade horizon explicitly; do not rely
    on a stale chart timeframe. ... Claimed picks must survive refreshes,
    rescans and restarts. Reuse the existing lifecycle where appropriate and
    document its expiry/removal rules."

This file pins the append-only store (`scripts/claimed_picks.py`) and the ONE
horizon resolver. Everything here is pure: no Qt, no live store - every call
takes an explicit ``path=`` into ``tmp_path``, which is the seam the packet
names ("a `path` parameter on every function").

The fade clock is NOT a calendar window: a claim is active until the trader
drops it or ``focus_picks.FADE_TRADING_DAYS`` **trading** days pass, counted by
``market_calendar.trading_days_between`` - so the boundary tests below compute
their session dates FROM that calendar rather than subtracting ten days.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# The day every fade assertion is measured from. A fixed Monday inside the
# validated NYSE range, so the boundary never drifts with the wall clock.
AS_OF = date(2026, 9, 14)


def _session_n_trading_days_back(as_of: date, n: int) -> date:
    """The first calendar day whose distance to ``as_of`` is ``n`` sessions.

    Computed from `market_calendar`, never from `timedelta(days=n)`: a window
    in calendar days is the exact mistake this store must not make.
    """
    import market_calendar

    day = as_of
    for _ in range(90):
        day = day - timedelta(days=1)
        if market_calendar.trading_days_between(day, as_of) == n:
            return day
    raise AssertionError(f"no day {n} sessions before {as_of}")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _claim_kwargs(**overrides):
    base = dict(
        symbol="aapl",
        side="LONG",
        horizon="d1",
        claimed_setup_id="avwap_band_bounce",
        source="chart_review:d1_flag_long",
        annotation_ref="2026-09-14T08:25:00-07:00",
        known_at_claim={},
        note="",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# item 1 - the store
# ---------------------------------------------------------------------------
def test_the_claim_store_sits_beside_swing_favourites_in_the_home_folder():
    """`project_paths.CLAIMED_PICKS_FILE` = PERSISTENT_DATA_DIR/claimed_picks.jsonl."""
    import project_paths

    path = project_paths.CLAIMED_PICKS_FILE
    assert Path(path).name == "claimed_picks.jsonl"
    assert Path(path).parent == Path(project_paths.SWING_FAVORITES_FILE).parent


def test_a_claim_row_carries_the_symbol_side_horizon_setup_and_both_clocks(tmp_path):
    """One row, schema by name, the symbol upper-cased, two timestamps."""
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    written = claimed_picks.record_claim(**_claim_kwargs(), path=path)

    assert written is not None, "a writable store must return the row it wrote"
    rows = _read(path)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["schema"] == "claimed_pick_v1"
    assert row["action"] == "claim"
    assert row["symbol"] == "AAPL", "the symbol is stored upper-cased"
    assert row["side"] == "LONG"
    assert row["horizon"] == "d1", "this packet writes no other horizon"
    assert row["claimed_setup_id"] == "avwap_band_bounce"
    assert row["source"] == "chart_review:d1_flag_long"
    assert row["annotation_ref"] == "2026-09-14T08:25:00-07:00", (
        "the annotation ref is what joins this store to trader_annotations.jsonl"
    )
    assert row["known_at_claim"] == {}
    assert row["note"] == ""
    # Two clocks, the swing-favourites convention: one tz-aware machine-local
    # stamp and one market-local session date. Neither alone answers "which
    # session was this?" across an evening write.
    assert row["claim_at"], "claim_at is the tz-aware machine-local stamp"
    assert row["claim_at"][10] == "T" and ("+" in row["claim_at"][11:] or "-" in row["claim_at"][11:]), (
        f"claim_at must be tz-aware ISO, got {row['claim_at']!r}"
    )
    assert row["claim_at_utc"], "claim_at_utc travels beside it"
    assert len(row["session_date"]) == 10 and row["session_date"][4] == "-"


def test_a_second_claim_of_an_active_key_writes_no_row_and_reports_the_duplicate(
    tmp_path,
):
    """Repeated clicks never create a second pick (D1C0 decision 5)."""
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    first = claimed_picks.record_claim(**_claim_kwargs(), path=path)
    second = claimed_picks.record_claim(**_claim_kwargs(note="again"), path=path)

    assert len(_read(path)) == 1, "the duplicate must append NOTHING"
    assert second is not None, "a duplicate is a success - the pick exists"
    assert second.get("duplicate") is True, "and it says so"
    assert second["claim_at"] == first["claim_at"], (
        "the duplicate returns the EXISTING active row, not a fresh one"
    )
    assert len(claimed_picks.active_claims(path, as_of=AS_OF)) == 1


def test_a_drop_then_a_new_claim_is_two_more_rows_and_one_active_row(tmp_path):
    """Append-only: the record of what the trader did, in the order they did it."""
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    claimed_picks.record_claim(**_claim_kwargs(), path=path)
    claimed_picks.record_drop("AAPL", "LONG", "avwap_band_bounce", path=path)
    claimed_picks.record_claim(**_claim_kwargs(note="back on"), path=path)

    rows = _read(path)
    assert [row["action"] for row in rows] == ["claim", "drop", "claim"]
    active = claimed_picks.active_claims(path, as_of=AS_OF)
    assert len(active) == 1, active
    assert active[0]["note"] == "back on"


def test_a_dropped_claim_is_not_active(tmp_path):
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    claimed_picks.record_claim(**_claim_kwargs(), path=path)
    claimed_picks.record_drop("AAPL", "LONG", "avwap_band_bounce", path=path)

    assert claimed_picks.active_claims(path, as_of=AS_OF) == []
    assert claimed_picks.active_keys(path, as_of=AS_OF) == set()


def test_the_identity_key_is_symbol_side_and_setup(tmp_path):
    """The same symbol LONG and SHORT, or under two setups, are separate picks."""
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    claimed_picks.record_claim(**_claim_kwargs(), path=path)
    claimed_picks.record_claim(**_claim_kwargs(side="SHORT"), path=path)
    claimed_picks.record_claim(
        **_claim_kwargs(claimed_setup_id="avwap_breakout"), path=path
    )
    # Dropping one leaves the other two standing.
    claimed_picks.record_drop("AAPL", "LONG", "avwap_band_bounce", path=path)

    active = claimed_picks.active_claims(path, as_of=AS_OF)
    assert {(row["side"], row["claimed_setup_id"]) for row in active} == {
        ("SHORT", "avwap_band_bounce"),
        ("LONG", "avwap_breakout"),
    }
    assert claimed_picks.active_keys(path, as_of=AS_OF) == {
        ("AAPL", "SHORT"),
        ("AAPL", "LONG"),
    }, "the queue gate keys on (symbol, side) only"


def test_active_claims_replays_a_real_shaped_file_in_order(tmp_path):
    """An OLD row has its keys PRESENT and EMPTY, not absent.

    The file below is what this store looks like after a week: an early row
    written before `known_at_claim` carried anything, a trader drop, a machine
    expire, and a live claim. The replay is file order, last action per key.
    """
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    old_session = _session_n_trading_days_back(AS_OF, 3).isoformat()
    rows = [
        # keys present and EMPTY - the 2026-08-24 lesson
        {
            "schema": "claimed_pick_v1",
            "action": "claim",
            "symbol": "MSFT",
            "side": "LONG",
            "horizon": "d1",
            "claimed_setup_id": "avwap_breakout",
            "claim_at": f"{old_session}T07:10:00-07:00",
            "claim_at_utc": f"{old_session}T14:10:00+00:00",
            "session_date": old_session,
            "source": "",
            "annotation_ref": "",
            "known_at_claim": {},
            "note": "",
        },
        {
            "schema": "claimed_pick_v1",
            "action": "claim",
            "symbol": "NVDA",
            "side": "SHORT",
            "horizon": "d1",
            "claimed_setup_id": "avwap_band_bounce",
            "claim_at": f"{old_session}T07:11:00-07:00",
            "claim_at_utc": f"{old_session}T14:11:00+00:00",
            "session_date": old_session,
            "source": "chart_review:d1_flag_short",
            "annotation_ref": "a2",
            "known_at_claim": {"priority_score": 71.0},
            "note": "lost the band",
        },
        {
            "schema": "claimed_pick_v1",
            "action": "drop",
            "symbol": "NVDA",
            "side": "SHORT",
            "horizon": "d1",
            "claimed_setup_id": "avwap_band_bounce",
            "claim_at": f"{old_session}T09:00:00-07:00",
            "claim_at_utc": f"{old_session}T16:00:00+00:00",
            "session_date": old_session,
            "source": "setups_table",
            "annotation_ref": "",
            "known_at_claim": {},
            "note": "",
        },
        {
            "schema": "claimed_pick_v1",
            "action": "expire",
            "symbol": "TSLA",
            "side": "LONG",
            "horizon": "d1",
            "claimed_setup_id": "sma_breakout",
            "claim_at": f"{old_session}T16:05:00-07:00",
            "claim_at_utc": f"{old_session}T23:05:00+00:00",
            "session_date": old_session,
            "source": "machine",
            "annotation_ref": "",
            "known_at_claim": {},
            "note": "",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    active = claimed_picks.active_claims(path, as_of=AS_OF)

    assert [(row["symbol"], row["side"]) for row in active] == [("MSFT", "LONG")], (
        "NVDA was dropped and TSLA expired; only the untouched MSFT claim stands"
    )
    assert active[0]["known_at_claim"] == {}, "an empty measurement dict is not a hole"


def test_a_claim_ten_trading_days_old_is_active_and_eleven_is_not(tmp_path):
    """`focus_picks.FADE_TRADING_DAYS` (10), counted on the exchange calendar.

    The two session dates are computed from `market_calendar`, so this asserts
    the NUMBER of sessions, not a count of calendar days that happens to agree
    over one particular fortnight.
    """
    import claimed_picks
    import focus_picks

    assert focus_picks.FADE_TRADING_DAYS == 10, "the lifecycle is reused by reference"

    path = tmp_path / "claimed_picks.jsonl"
    on_the_boundary = _session_n_trading_days_back(AS_OF, 10).isoformat()
    over_the_boundary = _session_n_trading_days_back(AS_OF, 11).isoformat()
    for symbol, session in (("ONTIME", on_the_boundary), ("STALE", over_the_boundary)):
        claimed_picks.append_row(
            claimed_picks.build_claim_row(
                **_claim_kwargs(symbol=symbol),
                session_date=session,
            ),
            path=path,
        )

    active = claimed_picks.active_claims(path, as_of=AS_OF)

    assert [row["symbol"] for row in active] == ["ONTIME"], (
        f"a claim exactly {focus_picks.FADE_TRADING_DAYS} sessions old is still "
        "active; one session past the fade is not"
    )


def test_a_calendar_that_cannot_answer_keeps_every_claim_active(tmp_path, monkeypatch):
    """Uncertainty never deletes (plan.md sec 5)."""
    import claimed_picks
    import market_calendar

    def _boom(*_args, **_kwargs):
        raise market_calendar.SessionCalendarError("outside the validated range")

    monkeypatch.setattr(market_calendar, "trading_days_between", _boom)
    if hasattr(claimed_picks, "trading_days_between"):
        monkeypatch.setattr(claimed_picks, "trading_days_between", _boom)

    path = tmp_path / "claimed_picks.jsonl"
    claimed_picks.append_row(
        claimed_picks.build_claim_row(
            **_claim_kwargs(symbol="ANCIENT"), session_date="2026-01-05"
        ),
        path=path,
    )

    assert [row["symbol"] for row in claimed_picks.active_claims(path, as_of=AS_OF)] == [
        "ANCIENT"
    ]
    assert claimed_picks.sweep_expired(path, as_of=AS_OF) == [], (
        "a calendar failure expires nothing"
    )
    assert len(_read(path)) == 1, "and appends nothing"


def test_sweep_expired_appends_one_expire_row_per_faded_claim_and_repeats_cleanly(
    tmp_path,
):
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    fresh = _session_n_trading_days_back(AS_OF, 2).isoformat()
    faded = _session_n_trading_days_back(AS_OF, 11).isoformat()
    claimed_picks.append_row(
        claimed_picks.build_claim_row(**_claim_kwargs(symbol="LIVE"), session_date=fresh),
        path=path,
    )
    claimed_picks.append_row(
        claimed_picks.build_claim_row(**_claim_kwargs(symbol="OLD"), session_date=faded),
        path=path,
    )

    swept = claimed_picks.sweep_expired(path, as_of=AS_OF)

    assert [row["symbol"] for row in swept] == ["OLD"]
    rows = _read(path)
    assert [(row["action"], row["symbol"]) for row in rows] == [
        ("claim", "LIVE"),
        ("claim", "OLD"),
        ("expire", "OLD"),
    ]
    # Idempotent: the second call has nothing left to expire and writes nothing.
    assert claimed_picks.sweep_expired(path, as_of=AS_OF) == []
    assert len(_read(path)) == 3, "a second sweep must not append a second expire row"
    assert [row["symbol"] for row in claimed_picks.active_claims(path, as_of=AS_OF)] == [
        "LIVE"
    ]


def test_an_unwritable_store_never_raises_and_the_claim_reports_the_failure(tmp_path):
    """An evidence store never costs the thing it records - but the CALLER has
    to know, because this one decides whether the chart is retired."""
    import claimed_picks

    # A directory where the file should be: every open() for append fails.
    blocked = tmp_path / "claimed_picks.jsonl"
    blocked.mkdir()

    assert claimed_picks.append_row({"action": "claim"}, path=blocked) is False
    assert claimed_picks.record_claim(**_claim_kwargs(), path=blocked) is None, (
        "a failed write returns None so the pane can keep the chart and say so"
    )
    assert claimed_picks.load_rows(blocked) == []


def test_load_rows_skips_junk_lines_without_raising(tmp_path):
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"
    claimed_picks.record_claim(**_claim_kwargs(), path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n\n")

    assert len(claimed_picks.load_rows(path)) == 1


# ---------------------------------------------------------------------------
# item 2 - the horizon is resolved explicitly, once
# ---------------------------------------------------------------------------
def _alert(**overrides):
    from ui.models.bounce import BounceAlert

    base = dict(
        time_text="08:25:00",
        symbol="AAPL",
        side="LONG",
        trigger="(long) zone1 reject at AVWAPE",
        timeframe="D1",
        tag="d1_flag_long",
        raw_text="MASTER_AVWAP_D1_ZONE: AAPL (long) zone1 reject",
        is_d1=True,
    )
    base.update(overrides)
    return BounceAlert(**base)


def test_a_d1_scan_alert_resolves_to_the_d1_horizon():
    import claimed_picks

    assert (
        claimed_picks.claim_horizon(
            _alert(), "avwap_band_bounce", is_m5_review=False
        )
        == "d1"
    )


def test_an_m5_review_alert_resolves_to_the_m5_horizon():
    """The panel's `_is_m5_review_alert` answer is handed in as a flag - the
    widget never imports the panel."""
    import claimed_picks

    m5 = _alert(
        timeframe="M5",
        tag="",
        is_d1=False,
        raw_text="BOUNCE: AAPL (long) 5m reclaim",
    )
    assert claimed_picks.claim_horizon(m5, "avwap_band_bounce", is_m5_review=True) == "m5"


def test_a_manual_look_takes_its_horizon_from_the_claimed_setups_registry_group():
    """Neither `is_d1` nor the M5 flag: fall back to what was CLAIMED.

    The registry's swing-side groups are `Main swing` and `Earnings cycle`;
    both are D1 theses, so a manual chart claimed as one of them places as a
    D1 pick.
    """
    import claimed_picks

    manual = _alert(tag="manual_chart", timeframe="", is_d1=False, raw_text="AAPL")
    assert (
        claimed_picks.claim_horizon(manual, "avwap_band_bounce", is_m5_review=False)
        == "d1"
    ), "avwap_band_bounce is a Main swing family"
    assert (
        claimed_picks.claim_horizon(
            manual, "post_earnings_52w_break", is_m5_review=False
        )
        == "d1"
    ), "post_earnings_52w_break is an Earnings cycle family"


def test_an_unnameable_setup_resolves_to_no_horizon_at_all():
    """`none_of_these` and an id the registry never heard of place NOTHING."""
    import claimed_picks

    manual = _alert(tag="manual_chart", timeframe="", is_d1=False, raw_text="AAPL")
    assert claimed_picks.claim_horizon(manual, "none_of_these", is_m5_review=False) == ""
    assert claimed_picks.claim_horizon(manual, "not_a_setup", is_m5_review=False) == ""
    assert claimed_picks.claim_horizon(manual, "", is_m5_review=False) == ""


def test_the_horizon_never_reads_the_rails_timeframe():
    """A stale `CaptureRail._timeframe` is the risk the trader named.

    `claim_horizon` takes the ALERT and the CLAIM. There is no rail in its
    signature, so a rail left on "D1" from construction cannot reach it.
    """
    import inspect

    import claimed_picks

    parameters = set(inspect.signature(claimed_picks.claim_horizon).parameters)
    assert "timeframe" not in parameters and "rail" not in parameters, parameters
    source = inspect.getsource(claimed_picks.claim_horizon)
    assert "_timeframe" not in source, "the resolver must never read the rail"


def test_a_claim_the_json_writer_cannot_serialise_fails_without_raising(tmp_path):
    """"Never raises" has to mean never, not "never on an OSError".

    `known_at_claim` comes from an ALERT PAYLOAD - whatever the scanner put
    there - so a value `json.dumps` refuses is a live possibility, and it must
    cost the row rather than the click.
    """
    import claimed_picks

    path = tmp_path / "claimed_picks.jsonl"

    assert claimed_picks.append_row({"action": "claim", "bad": object()}, path=path) is False
    assert claimed_picks.record_claim(
        **_claim_kwargs(known_at_claim={"payload": object()}), path=path
    ) is None
    assert _read(path) == [], "a refused row leaves nothing behind"
    # ...and the store still works for the next, well-formed claim.
    assert claimed_picks.record_claim(**_claim_kwargs(), path=path) is not None
    assert len(_read(path)) == 1
