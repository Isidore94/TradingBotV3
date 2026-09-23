"""The daily digest names its best and worst trades (trader 2026-09-23).

The v2 fact pack carried only aggregate outcome stats, so the narration's
best_candidates came back empty or generic. v3 adds a compact, capped `names`
block: the day's top D1 swing setups, best/worst settled swing outcomes,
best/worst M5 alerts and the trader's own journal trades. Missing data is
"unknown", never invented, and the narration may name only tickers the pack
holds.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import digest  # noqa: E402

DAY = "2026-09-22"
NOW = datetime(2026, 9, 23, 2, 0, tzinfo=timezone.utc)


def _final(symbol, close_r, *, direction="long", trigger="bounce_confirmed", mfe_r=2.0):
    return {
        "symbol": symbol, "direction": direction, "trade_date": DAY,
        "entry_time": f"{DAY}T06:45:00-07:00", "market_environment": "bullish",
        "env_key": "bullish|opening_drive", "close_r": close_r, "mfe_r": mfe_r,
        "mae_r": -0.5, "bounce_type": trigger,
    }


def _setup(symbol, tier, score, *, side="LONG", family="avwap_band_bounce", zone="LOWER_1 to AVWAPE"):
    return {"symbol": symbol, "side": side, "tier": tier, "priority_score": str(score),
            "setup_family": family, "favorite_zone": zone, "current_band_zone": "",
            "scan_date": DAY}


def _outcome(symbol, ret, *, side="LONG", horizon=5, family="avwap_breakout", tier="A"):
    return {"symbol": symbol, "side": side, "setup_family": family, "tier": tier,
            "scan_date": "2026-09-15", "target_session": DAY, "horizon_sessions": str(horizon),
            "side_return_pct": str(ret), "measured": "True"}


def _trade(symbol, pnl, *, direction="LONG", status="CLOSED"):
    return {"symbol": symbol, "direction": direction, "status": status, "net_pnl": pnl,
            "currency": "USD", "opened_at": f"{DAY}T07:00:00", "closed_at": f"{DAY}T08:00:00"}


def _pack(**overrides):
    payload = {
        "session_date": DAY,
        "is_session": True,
        "finals": [_final("AAPL", 1.5), _final("MSFT", -1.0), _final("NVDA", 0.4)],
        "top_setups": digest.names_source([
            _setup("VST", "A", 225.7), _setup("ALOY", "S", 302.0), _setup("KO", "B", 400.0),
        ]),
        "swing_outcomes": digest.names_source([
            _outcome("VKTX", 9.5), _outcome("FSLY", -7.7), _outcome("ABSI", 3.1),
        ]),
        "journal_trades": digest.names_source([_trade("TSLA", -94.4), _trade("AMD", 0.0, status="OPEN")]),
        "now": NOW,
    }
    payload.update(overrides)
    return digest.build_fact_pack(**payload)


def _symbols(rows):
    return [row["symbol"] for row in rows]


# ---------------------------------------------------------------------------
# the names block
# ---------------------------------------------------------------------------


def test_the_schema_is_v3_because_the_shape_grew():
    assert digest.FACTS_SCHEMA == "daily_digest_facts_v3"
    assert _pack()["schema"] == "daily_digest_facts_v3"


def test_top_setups_rank_by_tier_then_score_with_their_numbers():
    top = _pack()["names"]["d1_top_setups"]
    assert top["status"] == "ok" and top["n"] == 3
    assert _symbols(top["rows"]) == ["ALOY", "VST", "KO"]
    first = top["rows"][0]
    assert first == {"symbol": "ALOY", "side": "LONG", "tier": "S", "score": 302.0,
                     "setup": "avwap_band_bounce", "zone": "LOWER_1 to AVWAPE"}


def test_top_setups_are_capped_and_the_cap_is_counted():
    rows = [_setup(f"S{index:02d}", "B", index) for index in range(25)]
    top = _pack(top_setups=digest.names_source(rows))["names"]["d1_top_setups"]
    assert len(top["rows"]) == digest.TOP_SETUPS_CAP == 10
    assert top["n"] == 25 and top["capped"] == 15


def test_settled_swing_outcomes_list_best_and_worst_without_repeating_a_row():
    swing = _pack()["names"]["swing_settled"]
    assert swing["status"] == "ok" and swing["n"] == 3
    assert _symbols(swing["best"])[0] == "VKTX"
    assert swing["best"][0]["ret_pct"] == 9.5 and swing["best"][0]["h"] == 5
    assert _symbols(swing["worst"])[0] == "FSLY"
    listed = _symbols(swing["best"]) + _symbols(swing["worst"])
    assert len(listed) == len(set(listed)) == 3


def test_m5_alerts_come_from_the_champion_finals_with_their_trigger():
    m5 = _pack()["names"]["m5_alerts"]
    assert m5["n"] == 3
    assert m5["best"][0] == {"symbol": "AAPL", "side": "LONG", "trigger": "bounce_confirmed",
                             "time": "06:45", "close_r": 1.5, "mfe_r": 2.0}
    assert _symbols(m5["worst"])[0] == "MSFT"


def test_m5_ranks_by_the_one_exit_policy_measured_most_and_never_mixes_them():
    """2026-09-22 had 545 usable finals and only 3 with a close_r."""
    swept = [{**_final(f"S{index}", None), "r_last_measured": index - 2.0} for index in range(5)]
    m5 = _pack(finals=[_final("AAPL", 9.0)] + swept)["names"]["m5_alerts"]
    assert m5["rank_by"] == "last_measured_r"
    assert m5["unranked"] == 1  # AAPL has a close_r but no last-measured R
    assert m5["best"][0]["symbol"] == "S4" and m5["best"][0]["last_measured_r"] == 2.0
    assert all("close_r" not in row for row in m5["best"] + m5["worst"])


def test_journal_trades_carry_pnl_and_an_unknown_r_never_a_zero():
    journal = _pack()["names"]["journal_trades"]
    assert _symbols(journal["rows"]) == ["TSLA", "AMD"]
    tsla, amd = journal["rows"]
    assert tsla["net_pnl"] == -94.4 and tsla["r"] is None
    # An open trade's P&L is not settled: unknown, not zero.
    assert amd["status"] == "OPEN" and amd["net_pnl"] is None


def test_an_unread_source_is_unknown_with_its_reason_and_no_rows():
    pack = _pack(
        top_setups=digest.names_source(status="unknown", reason="the scan file holds 2026-09-19"),
        swing_outcomes=None,
    )
    top = pack["names"]["d1_top_setups"]
    assert top["status"] == "unknown" and top["rows"] == [] and "2026-09-19" in top["reason"]
    swing = pack["names"]["swing_settled"]
    assert swing["status"] == "unknown" and swing["best"] == [] and swing["reason"]


def test_a_non_session_names_nothing():
    pack = _pack(is_session=False)
    for section in pack["names"].values():
        if isinstance(section, dict) and "status" in section:
            assert section["status"] == "not_a_session"


def test_a_crowded_day_is_trimmed_to_fit_the_cap_and_says_so(monkeypatch):
    monkeypatch.setattr(digest, "FACT_PACK_HARD_CAP_BYTES", 12_000)
    long_family = "x" * 40
    pack = _pack(
        finals=[_final(f"M{index:03d}", index / 10) for index in range(60)],
        top_setups=digest.names_source(
            [_setup(f"T{index:03d}", "S", index, family=long_family, zone="y" * 40)
             for index in range(40)]),
        swing_outcomes=digest.names_source(
            [_outcome(f"W{index:03d}", index - 20, family=long_family) for index in range(40)]),
        journal_trades=digest.names_source([_trade(f"J{index:03d}", index) for index in range(40)]),
    )
    assert digest.fact_pack_bytes(pack) <= digest.FACT_PACK_HARD_CAP_BYTES
    trimmed = sum(
        section.get("trimmed", 0) for section in pack["names"].values() if isinstance(section, dict)
    )
    assert trimmed > 0


def test_every_named_row_puts_its_symbol_first_so_a_metric_ref_can_resolve_it():
    pack = _pack()
    for section in pack["names"].values():
        if not isinstance(section, dict):
            continue
        for key in ("rows", "best", "worst"):
            for row in section.get(key) or []:
                assert next(iter(row)) == "symbol"


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_tier_list_reader_keeps_only_this_sessions_scan(tmp_path):
    path = _write_csv(tmp_path / "tier.csv", [_setup("ALOY", "S", 302.0),
                                              {**_setup("OLD", "S", 1.0), "scan_date": "2026-09-19"}])
    source = digest.read_top_setups(DAY, path=path)
    assert source["status"] == "ok" and _symbols(source["rows"]) == ["ALOY"]

    other = digest.read_top_setups("2026-09-23", path=path)
    assert other["status"] == "unknown" and "2026-09-22" in other["reason"]

    absent = digest.read_top_setups(DAY, path=tmp_path / "missing.csv")
    assert absent["status"] == "absent" and absent["rows"] == []


def test_the_horizon_reader_keeps_measured_rows_settling_this_session(tmp_path):
    path = _write_csv(tmp_path / "horizon.csv", [
        _outcome("VKTX", 9.5),
        {**_outcome("LATE", 1.0), "target_session": "2026-09-23"},
        {**_outcome("PEND", 1.0), "measured": "False"},
    ])
    source = digest.read_swing_outcomes(DAY, path=path)
    assert source["status"] == "ok" and _symbols(source["rows"]) == ["VKTX"]


def test_the_journal_reader_is_read_only_and_reads_this_sessions_trades(tmp_path):
    path = tmp_path / "journal.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE trades (trade_id TEXT, symbol TEXT, direction TEXT, status TEXT, "
            "opened_at TEXT, closed_at TEXT, trade_date TEXT, net_pnl REAL, currency TEXT)"
        )
        conn.execute("INSERT INTO trades VALUES ('t1','TSLA','LONG','CLOSED',?,?,?,-94.4,'USD')",
                     (f"{DAY}T07:00:00", f"{DAY}T08:00:00", DAY))
        conn.execute("INSERT INTO trades VALUES ('t2','OLD','LONG','CLOSED',?,?,?,1.0,'USD')",
                     ("2026-09-01T07:00:00", "2026-09-01T08:00:00", "2026-09-01"))
    before = path.read_bytes()
    source = digest.read_journal_trades(DAY, path=path)
    assert source["status"] == "ok" and _symbols(source["rows"]) == ["TSLA"]
    assert path.read_bytes() == before

    absent = digest.read_journal_trades(DAY, path=tmp_path / "none.sqlite3")
    assert absent["status"] == "absent"
    assert not (tmp_path / "none.sqlite3").exists(), "a missing journal is never created"


# ---------------------------------------------------------------------------
# narration: name the tickers, and only the tickers the pack holds
# ---------------------------------------------------------------------------


def _item(statement):
    return {"statement": statement, "evidence_refs": ["digest.facts"], "confidence": "medium"}


def _summary(best=(), working=(), executive="Swing names led the day."):
    return {
        "executive_summary": executive,
        "what_is_working": [_item(text) for text in working],
        "what_is_not_working": [],
        "best_candidates": [_item(text) for text in best],
        "lessons_for_tomorrow": [],
        "risk_notes": [],
    }


def test_the_package_tells_the_narrator_to_name_the_tickers():
    package = digest.narration_evidence_package(_pack())
    rules = " ".join(package["narration_rules"])
    assert "best_candidates" in rules and "names" in rules
    assert "digest.facts" in rules


def test_a_narration_naming_pack_tickers_with_pack_numbers_passes():
    summary = _summary(best=["ALOY LONG tier S score 302.0", "VKTX settled +9.5 percent over 5 sessions"])
    assert digest.check_narration_names(summary, _pack()) == []


def test_a_narration_naming_a_ticker_the_pack_does_not_hold_is_rejected():
    summary = _summary(best=["ALOY tier S"], working=["GME looked strong"])
    problems = digest.check_narration_names(summary, _pack())
    assert problems and "GME" in problems[0]


def test_empty_best_candidates_is_rejected_when_the_pack_names_rows():
    problems = digest.check_narration_names(_summary(), _pack())
    assert any("best_candidates" in problem for problem in problems)


def test_empty_best_candidates_is_fine_when_the_pack_names_nothing():
    pack = _pack(finals=[], top_setups=digest.names_source([]),
                 swing_outcomes=digest.names_source([]), journal_trades=digest.names_source([]))
    assert digest.check_narration_names(_summary(), pack) == []


def test_a_best_candidate_number_that_is_not_that_tickers_number_is_rejected():
    problems = digest.check_narration_names(_summary(best=["VKTX settled +12.25 percent"]), _pack())
    assert problems and "12.25" in problems[0]


def test_trading_words_in_capitals_are_not_mistaken_for_tickers():
    summary = _summary(best=["ALOY SHORT, tier S, AVWAP band bounce; MFE and EOD R not blended"])
    assert digest.check_narration_names(summary, _pack()) == []


def _fake_ai(monkeypatch, answers):
    import ai_summary

    calls = []

    def request(**kwargs):
        calls.append(kwargs.get("previous_error", ""))
        return {"model": "fake", "summary": answers[min(len(calls), len(answers)) - 1]}

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: "fake")
    monkeypatch.setattr(ai_summary, "request_ai_summary", request)
    return calls


def test_a_bad_narration_is_retried_once_with_the_reason(monkeypatch):
    calls = _fake_ai(monkeypatch, [_summary(best=["GME"]), _summary(best=["ALOY tier S"])])
    result = digest._narrate(pack=_pack(), now=NOW)
    assert len(calls) == 2 and "GME" in calls[1]
    assert result["narration"]["best_candidates"][0]["statement"] == "ALOY tier S"


def test_a_narration_still_naming_unknown_tickers_is_not_published(tmp_path, monkeypatch):
    _fake_ai(monkeypatch, [_summary(best=["GME"])])
    monkeypatch.setattr(digest, "_read_names_sources", lambda day, unavailable: {})
    result = digest.run_daily_digest(
        session_date=DAY, now=NOW, root=tmp_path, finals=[_final("AAPL", 1.0)],
    )
    assert result["status"] == digest.STATUS_DEGRADED
    assert "GME" in result["reason"]
    assert digest.facts_path(tmp_path, DAY).is_file()
    assert not digest.narration_path(tmp_path, DAY).exists()


def test_run_daily_digest_writes_the_names_it_read(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "_read_names_sources", lambda day, unavailable: {
        "top_setups": digest.names_source([_setup("ALOY", "S", 302.0)]),
    })
    digest.run_daily_digest(session_date=DAY, now=NOW, root=tmp_path,
                            finals=[_final("AAPL", 1.0)], narrate=False)
    pack = json.loads(digest.facts_path(tmp_path, DAY).read_text(encoding="utf-8"))
    assert pack["names"]["d1_top_setups"]["rows"][0]["symbol"] == "ALOY"
    assert pack["names"]["m5_alerts"]["best"][0]["symbol"] == "AAPL"
    assert pack["names"]["journal_trades"]["status"] == "unknown"


@pytest.mark.parametrize("path_name", ["facts"])
def test_a_v2_pack_on_disk_still_rolls_up(tmp_path, path_name):
    """Old readers keep working: a v2 pack has no names block."""
    target = digest.facts_path(tmp_path, DAY)
    target.parent.mkdir(parents=True)
    v2 = _pack()
    v2.pop("names")
    v2["schema"] = "daily_digest_facts_v2"
    target.write_text(digest.render_fact_pack(v2), encoding="utf-8")
    assert digest.rollup(tmp_path, since=DAY, until=DAY)["sessions"] == 1
