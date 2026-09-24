"""Evidence lane: setup suggestions from what the desk logged before an entry."""

from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

PT = ZoneInfo("America/Los_Angeles")


def _trade(symbol="AAA", direction="LONG", opened_at="2026-09-10T10:30:00-04:00", **extra):
    return {
        "trade_id": extra.pop("trade_id", "T1"),
        "symbol": symbol,
        "direction": direction,
        "status": "CLOSED",
        "opened_at": opened_at,
        "closed_at": extra.pop("closed_at", "2026-09-10T11:30:00-04:00"),
        "trade_date": extra.pop("trade_date", "2026-09-10"),
        **extra,
    }


def _fired(symbol="AAA", side="LONG", ts="2026-09-10T07:10:00", kind="vwap_bounce", action="watch_fired"):
    # Review-event ts is naive machine-local time (Pacific on the desk).
    return {
        "action": action,
        "symbol": symbol,
        "side": side,
        "ts": ts,
        "trade_date": ts[:10],
        "detail": {"kind": kind, "message": f"{kind} (long)"},
        "review_record_id": f"rr-{symbol}-{ts}",
    }


def _index(events=(), claims=(), focus=()):
    from journal_setup_evidence import EvidenceIndex

    return EvidenceIndex.load(
        review_events=list(events), claimed_rows=list(claims), focus_rows=list(focus), naive_tz=PT
    )


# --------------------------------------------------------------- pure parts --


def test_option_symbols_resolve_to_their_underlying():
    from journal_setup_evidence import option_underlying, underlying_view

    assert option_underlying("BK250905C00102000") == ("BK", "C")
    assert option_underlying("QQQ26JUN26P700.00") == ("QQQ", "P")
    assert option_underlying("NVDA") == ("NVDA", "")
    # A long put is short the underlying; a short put is long it.
    assert underlying_view({"symbol": "IWM251114P00236000", "direction": "LONG"}) == ("IWM", "SHORT")
    assert underlying_view({"symbol": "DRAM261218P00055000", "direction": "SHORT"}) == ("DRAM", "LONG")
    assert underlying_view({"symbol": "HPE260911C00050000", "direction": "LONG"}) == ("HPE", "LONG")


def test_an_alert_that_fired_just_before_the_entry_clears_the_bulk_threshold():
    from journal_bulk_tag import DEFAULT_CONFIDENCE_THRESHOLD

    # 07:10 PT = 10:10 ET, twenty minutes before the 10:30 ET entry.
    found = _index([_fired()]).candidates_for(_trade())

    assert [item["tag"] for item in found] == ["vwap_bounce"]
    top = found[0]
    assert top["source"] == "evidence:alert_fired"
    assert top["confidence"] >= DEFAULT_CONFIDENCE_THRESHOLD
    assert "20 min before entry" in top["rationale"]
    assert top["context_row_id"].startswith("rr-AAA")


def test_an_alert_stamped_after_the_entry_is_never_evidence():
    """Point in time: only what was known at the first fill."""
    found = _index([_fired(ts="2026-09-10T07:31:00")]).candidates_for(_trade())
    assert found == []


def test_the_opposite_side_never_matches():
    found = _index([_fired(side="SHORT")]).candidates_for(_trade())
    assert found == []


def test_an_intraday_alert_from_an_earlier_session_explains_nothing():
    found = _index([_fired(ts="2026-09-09T07:10:00")]).candidates_for(_trade())
    assert found == []


def test_a_d1_flag_a_few_days_earlier_is_a_weaker_suggestion():
    from journal_bulk_tag import DEFAULT_CONFIDENCE_THRESHOLD

    event = _fired(ts="2026-09-08T12:00:00", kind="avwape_dev1_bounce", action="focus_d1_flag")
    found = _index([event]).candidates_for(_trade())

    assert [item["tag"] for item in found] == ["avwape_dev1_bounce"]
    assert found[0]["confidence"] < DEFAULT_CONFIDENCE_THRESHOLD
    assert "day(s) before entry" in found[0]["rationale"]


def test_a_claim_before_the_entry_is_the_strongest_evidence_and_a_drop_retires_it():
    claim = {
        "action": "claim",
        "symbol": "AAA",
        "side": "LONG",
        "claimed_setup_id": "trendline_break",
        "claim_at": "2026-09-08T12:00:00-07:00",
        "horizon": "d1",
        "annotation_ref": "ann-1",
    }
    found = _index(claims=[claim]).candidates_for(_trade())
    assert found[0]["tag"] == "trendline_break"
    assert found[0]["source"] == "evidence:claimed_pick"
    assert found[0]["confidence"] >= 0.85

    drop = {**claim, "action": "drop", "claim_at": "2026-09-09T12:00:00-07:00"}
    assert _index(claims=[claim, drop]).candidates_for(_trade()) == []


def test_focus_membership_corroborates_but_never_names_a_setup():
    focus = {
        "trade_date": "2026-09-10",
        "symbol": "AAA",
        "side": "LONG",
        "source": "focus_m5",
        "snapshotted_at": "2026-09-09T22:37:30",
    }
    assert _index(focus=[focus]).candidates_for(_trade()) == []

    plain = _index([_fired(ts="2026-09-10T05:00:00")]).candidates_for(_trade())
    boosted = _index([_fired(ts="2026-09-10T05:00:00")], focus=[focus]).candidates_for(_trade())
    assert boosted[0]["confidence"] > plain[0]["confidence"]
    assert "Focus" in boosted[0]["rationale"]


def test_an_option_trade_is_matched_on_its_underlying_and_its_effective_side():
    # A long put is SHORT the underlying, so a short-side alert explains it.
    trade = _trade(symbol="AAA260918P00050000", direction="LONG")
    found = _index([_fired(side="SHORT")]).candidates_for(trade)
    assert [item["tag"] for item in found] == ["vwap_bounce"]
    assert _index([_fired(side="LONG")]).candidates_for(trade) == []


def test_a_date_only_fill_uses_only_earlier_sessions():
    trade = _trade(opened_at="2026-09-10T00:00:00-04:00")
    same_day = _fired(ts="2026-09-10T07:10:00", kind="ema15_reject", action="focus_d1_flag")
    day_before = _fired(ts="2026-09-09T07:10:00", kind="new_5d_high", action="focus_d1_flag")
    found = _index([same_day, day_before]).candidates_for(trade)
    assert [item["tag"] for item in found] == ["new_5d_high"]


def test_generic_card_tags_name_no_setup():
    shown = {
        "action": "shown",
        "symbol": "AAA",
        "side": "LONG",
        "ts": "2026-09-10T07:00:00",
        "tag": "manual_chart",
        "chart_watch_kind": "",
        "bounce_types": "",
        "timeframe": "M5",
    }
    assert _index([shown]).candidates_for(_trade()) == []
    shown_with_kind = {**shown, "bounce_types": "eod_vwap;impulse_retest_vwap_eod"}
    assert [item["tag"] for item in _index([shown_with_kind]).candidates_for(_trade())] == ["eod_vwap"]


# ------------------------------------------------------- AutoTagger + P6a ---


def _empty_tagger(tmp_path, evidence):
    from journal_analytics import AutoTagger

    return AutoTagger(
        setup_tracker_path=tmp_path / "none_tracker.json",
        focus_path=tmp_path / "none_focus.json",
        avwap_signals_path=tmp_path / "none_signals.csv",
        intraday_bounces_path=tmp_path / "none_bounces.csv",
        evidence=evidence,
    )


def test_the_auto_tagger_carries_the_evidence_lane(tmp_path):
    tagger = _empty_tagger(tmp_path, _index([_fired()]))
    suggestions = tagger.suggest_for_trade(_trade())
    assert [item["tag"] for item in suggestions] == ["vwap_bounce"]
    assert suggestions[0]["source"] == "evidence:alert_fired"


def test_the_scanner_lane_matches_an_option_on_its_underlying(tmp_path):
    """Before: 87 option trades on the live journal matched nothing, ever."""
    import json

    from journal_analytics import AutoTagger, clear_context_row_cache

    clear_context_row_cache()
    tracker = tmp_path / "tracker.json"
    tracker.write_text(
        json.dumps(
            {
                "setups": {
                    "s1": {
                        "symbol": "AAA",
                        "side": "SHORT",
                        "scan_date": "2026-09-10",
                        "setup_family": "avwap_breakout",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    tagger = AutoTagger(
        setup_tracker_path=tracker,
        focus_path=tmp_path / "none_focus.json",
        avwap_signals_path=tmp_path / "none_signals.csv",
        intraday_bounces_path=tmp_path / "none_bounces.csv",
    )
    long_put = _trade(symbol="AAA260918P00050000", direction="LONG")
    suggestions = tagger.suggest_for_trade(long_put)
    assert suggestions and suggestions[0]["tag"] == "avwap_breakout"
    # Same-day tracker + agreeing side: the P6a arithmetic's 0.72.
    assert suggestions[0]["confidence"] >= 0.70
    clear_context_row_cache()


def test_the_bulk_tagger_applies_strong_evidence_and_parks_weak_evidence(tmp_path):
    """Strong evidence becomes a provisional tag; weak evidence stays a candidate only."""
    import journal_bulk_tag as bulk
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "journal.sqlite3")
    with store.connection() as conn:
        for trade_id, symbol in (("strong", "AAA"), ("weak", "BBB")):
            conn.execute(
                """
                INSERT INTO trades(trade_id, broker, account_number, symbol, direction, status,
                    opened_at, closed_at, trade_date, updated_at)
                VALUES(?, 'QUESTRADE', '1', ?, 'LONG', 'CLOSED', ?, ?, '2026-09-10', '2026-09-10')
                """,
                (trade_id, symbol, "2026-09-10T10:30:00-04:00", "2026-09-10T11:30:00-04:00"),
            )
    events = [
        _fired(symbol="AAA"),
        _fired(symbol="BBB", ts="2026-09-08T12:00:00", kind="new_5d_high", action="focus_d1_flag"),
    ]
    store.refresh_auto_tags(_empty_tagger(tmp_path, _index(events)))
    plan = bulk.build_plan(store, refresh=False)
    bulk.apply_plan(store, plan)

    strong = store.annotation_state("strong")
    assert strong["setup_tags"] == "vwap_bounce"
    assert strong["tag_status"] == "provisional"
    weak = store.annotation_state("weak")
    assert weak["setup_tags"] == ""
    assert weak["tag_status"] == "needs_review"
    assert [row["tag"] for row in store.list_auto_tag_candidates("weak")][0] == "new_5d_high"


def test_an_aware_utc_stamp_is_compared_as_an_instant():
    from journal_setup_evidence import aware_moment

    moment = aware_moment("2026-09-10T14:10:00+00:00")
    assert moment.tzinfo is not None
    assert moment.astimezone(timezone.utc).hour == 14
