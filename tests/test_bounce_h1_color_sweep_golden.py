"""What the H1 colour sweep records, frozen (S10a).

`check_h1_color_setups` runs the real `_evaluate_h1_color_signals`,
`detect_h1_color_signals` and `_emit_h1_color_alert` over hand-built hourly
tapes. Only the recorders are stubbed: the candidate event, the outcome
registration, the tier record and the symbol log. The fixture freezes every
call they received.

Three names, one per H1 colour type:

| name | side | type |
|---|---|---|
| BLUE | long | h1_blue_after_red |
| BNCE | long | h1_ema10_bounce |
| DUMP | short | h1_green_to_yellow |
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_NAME = "h1_color_sweep_v1"


def _to_ib(rows):
    from bounce_bot_lib.legacy import IbBar

    return [
        IbBar(
            dt=datetime.fromisoformat(row["dt"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
        for row in rows
    ]


def _run(fixture, monkeypatch) -> dict:
    """Drive the shipped sweep; record every recorder call in order."""
    import bounce_bot_lib.legacy as legacy
    from bounce_bot_lib.legacy import BounceBot

    # The fixture carries closed H1 bars directly, so the M5->H1 fold is identity.
    monkeypatch.setattr(legacy, "_closed_h1_bars", lambda bars: list(bars))

    series = {name: _to_ib(rows) for name, rows in fixture["cases"].items()}
    spy = _to_ib(fixture["spy"])
    series["SPY"] = spy
    watch = fixture["watch"]

    records: list[dict] = []
    bot = object.__new__(BounceBot)
    bot._spy_session_bars = lambda: ([spy[-1]], spy[-2].close)
    bot._h1_color_state = None
    bot._h1_color_sweep_symbols = lambda side: list(watch.get(side, []))
    bot.get_cached_5m_bars = lambda symbol: series.get(symbol, [])

    def _log_event(kind, symbol, side, levels, bounce_candle, current_candle, reason=""):
        event_id = f"{symbol}_{side}_{'_'.join(sorted(levels))}"
        records.append(
            {
                "call": "candidate_event",
                "kind": kind,
                "symbol": symbol,
                "side": side,
                "levels": dict(levels),
                "bounce_candle": dict(bounce_candle),
                "current_candle": dict(current_candle),
                "reason": reason,
                "event_id": event_id,
            }
        )
        return {"event_id": event_id, "symbol": symbol, "direction": side}

    def _register(symbol, side, levels, bounce_candle, current_candle, event_id):
        records.append(
            {
                "call": "register_outcome",
                "symbol": symbol,
                "side": side,
                "levels": dict(levels),
                "bounce_candle": dict(bounce_candle),
                "current_candle": dict(current_candle),
                "event_id": event_id,
            }
        )

    bot._log_bounce_candidate_event = _log_event
    bot._register_bounce_outcome = _register
    bot._evaluate_bounce_alert_quality = lambda side, levels, row: {"tier": "B"}
    bot.record_alert_tier = lambda event_id, quality: records.append(
        {"call": "alert_tier", "event_id": event_id, "tier": quality.get("tier")}
    )
    bot.log_symbol = lambda symbol, message: records.append(
        {"call": "log_symbol", "symbol": symbol, "message": message}
    )
    bot._measured_exit_suffix = lambda *a, **k: ""
    bot.log_bounce_to_file = lambda **k: records.append({"call": "bounce_file", "symbol": k.get("symbol")})
    bot.gui_callback = None

    hits = bot.check_h1_color_setups()
    return {
        "hits": [
            {"symbol": hit["symbol"], "side": hit["side"], "type": hit["type"]} for hit in hits
        ],
        "records": records,
    }


def test_h1_color_sweep_golden_fixture(monkeypatch):
    fixture = load_fixture_contract(FIXTURE_NAME)
    assert fixture.schema == "h1_color_sweep_v1"
    actual = _run(fixture, monkeypatch)
    assert actual == fixture["expected"]
