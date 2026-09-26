"""P10 options chase: the pure picker, the service with a fake IB, the Opt column, the log."""

from __future__ import annotations

import json
import time
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import options_chase as oc  # noqa: E402

NY = ZoneInfo("America/New_York")
# Monday 2026-09-28; weeklies Fri 10/02 (4 sessions) and Fri 10/09.
TODAY = date(2026, 9, 28)
NOW = datetime(2026, 9, 28, 10, 40, 20, tzinfo=NY)


def _pop(**values):
    row = {"symbol": "ABC", "side": "long", "last": 25.0, "atr": 0.20, "rvol": 3.0, "move15": 2.4}
    row.update(values)
    return row


def _q(expiry, strike, right, bid, ask, delta, iv=0.62):
    return {"expiry": expiry, "strike": strike, "right": right, "bid": bid, "ask": ask,
            "delta": delta, "iv": iv}


def _chain():
    quotes = [
        _q("2026-10-02", 26.0, "C", 1.40, 1.50, 0.38),
        _q("2026-10-02", 27.0, "C", 0.85, 0.95, 0.26),
        _q("2026-10-02", 28.0, "C", 0.40, 0.46, 0.14),
        _q("2026-10-02", 24.0, "C", 1.9, 2.0, 0.60),  # ITM call: never picked
        _q("2026-10-02", 23.0, "P", 0.70, 0.78, -0.24),
        _q("2026-10-02", 22.0, "P", 0.30, 0.34, -0.12),
        _q("2026-10-09", 27.0, "C", 1.10, 1.20, 0.25),
    ]
    return {"expiries": ["20260930", "20261002", "20261009"],
            "strikes": [22.0, 23.0, 24.0, 26.0, 27.0, 28.0], "quotes": quotes}


# ---------------------------------------------------------------- pure picker
def test_picks_nearest_weekly_otm_call_nearest_quarter_delta():
    result = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=0.41)
    assert result["status"] == oc.STATUS_CANDIDATE
    assert (result["expiry"], result["strike"], result["right"]) == ("2026-10-02", 27.0, "C")
    assert result["delta"] == pytest.approx(0.26)
    assert result["mid"] == pytest.approx(0.90)
    assert result["spread_pct"] == pytest.approx(0.10 / 0.90 * 100)
    assert result["iv_vs_hv"] == pytest.approx(0.62 / 0.41)
    assert oc.cell_text(result) == "27C 10/02 · 0.85x0.95 · 11% · IV 62 (HV 41)"


def test_short_pop_picks_the_otm_put_with_abs_delta():
    result = oc.pick_candidate(_pop(side="short"), _chain(), today=TODAY, hv=0.41)
    assert (result["status"], result["strike"], result["right"]) == ("candidate", 23.0, "P")
    assert result["delta"] == pytest.approx(0.24)


def test_the_wednesday_daily_is_not_a_weekly():
    assert oc.weekly_expiries(["20260930", "20261002", "20261009"]) == [
        date(2026, 10, 2), date(2026, 10, 9)]


def test_expiry_under_two_sessions_rolls_to_the_next_weekly():
    # Thursday 10/01: Friday 10/02 has one session to go, so 10/09 is the pick.
    result = oc.pick_candidate(_pop(), _chain(), today=date(2026, 10, 1), hv=0.41)
    assert result["expiry"] == "2026-10-09"
    assert result["strike"] == 27.0 and result["status"] == "candidate"
    expiry, sessions, _ = oc.pick_expiry(["20261002"], date(2026, 10, 1))
    assert expiry is None and sessions is None


def test_refuses_rvol_under_two_and_unknown_rvol():
    assert oc.pick_candidate(_pop(rvol=1.8), _chain(), today=TODAY)["reason"] == "RVOL 1.8 < 2"
    unknown = oc.pick_candidate(_pop(rvol=None), _chain(), today=TODAY)
    assert (unknown["status"], unknown["reason"]) == ("refused", "RVOL unknown")


def test_refuses_empty_chain():
    for chain in (None, {}, {"expiries": ["20261002"], "quotes": []}):
        result = oc.pick_candidate(_pop(), chain, today=TODAY)
        assert (result["status"], result["reason"]) == ("refused", "empty chain")


def test_refuses_when_no_delta_in_range():
    chain = _chain()
    chain["quotes"] = [_q("2026-10-02", 26.0, "C", 1.4, 1.5, 0.40),
                       _q("2026-10-02", 28.0, "C", 0.4, 0.46, 0.10)]
    result = oc.pick_candidate(_pop(), chain, today=TODAY)
    assert (result["status"], result["reason"]) == ("refused", "no delta in 0.15-0.35")
    chain["quotes"] = [_q("2026-10-02", 27.0, "C", 0.85, 0.95, None)]
    assert oc.pick_candidate(_pop(), chain, today=TODAY)["reason"] == "no delta quoted"


def test_refuses_a_wide_spread_but_still_names_the_contract():
    chain = _chain()
    chain["quotes"][1] = _q("2026-10-02", 27.0, "C", 0.80, 1.00, 0.26)
    result = oc.pick_candidate(_pop(), chain, today=TODAY, hv=0.41)
    assert result["status"] == "refused"
    assert result["reason"] == "spread 22% of mid > 15%"
    assert result["strike"] == 27.0 and result["mid"] == pytest.approx(0.90)
    assert oc.cell_text(result) == "no chase (spread 22% of mid > 15%)"
    assert "refused: spread 22% of mid > 15%" in oc.detail_text(result)


def test_missing_quote_parts_are_unknown_never_guessed():
    chain = _chain()
    chain["quotes"][1] = _q("2026-10-02", 27.0, "C", None, 0.95, 0.26, iv=None)
    result = oc.pick_candidate(_pop(), chain, today=TODAY, hv=None)
    assert (result["status"], result["reason"]) == ("refused", "no two-sided quote")
    assert result["mid"] is None and result["iv"] is None and result["iv_vs_hv"] is None
    assert oc.pick_candidate(_pop(last=None), _chain(), today=TODAY)["reason"] == "last price unknown"
    ok = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=None)
    assert ok["status"] == "candidate" and ok["iv_vs_hv"] is None
    assert oc.cell_text(ok).endswith("IV 62 (HV —)")


def test_realized_vol_is_annualised_close_to_close_over_20_sessions():
    closes = [100.0 * (1.01 if i % 2 else 0.99) for i in range(21)]
    returns = [math.log(b / a) for a, b in zip(closes[:-1], closes[1:], strict=True)]
    mean = sum(returns) / len(returns)
    expected = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)) * math.sqrt(252)
    assert oc.realized_vol(closes) == pytest.approx(expected)
    assert oc.realized_vol(closes[:20]) is None  # 19 returns: unknown


def test_read_daily_closes_drops_today_and_later(tmp_path):
    path = tmp_path / "ABC.csv"
    path.write_text("datetime,open,high,low,close,volume\n"
                    "2026-09-24,1,1,1,10,1\n2026-09-25,1,1,1,11,1\n2026-09-28,1,1,1,99,1\n",
                    encoding="utf-8")
    assert oc.read_daily_closes(path, before=TODAY) == [10.0, 11.0]
    assert oc.read_daily_closes(tmp_path / "missing.csv", before=TODAY) == []


def test_quote_plan_spreads_otm_strikes_over_the_delta_window():
    strikes = [20 + 0.5 * i for i in range(40)]  # 20.0 .. 39.5
    plan = oc.quote_plan(25.0, "long", strikes, sessions=4, hv=0.60)
    assert plan and all(s > 25.0 for s in plan) and len(plan) <= oc.QUOTE_STRIKES
    puts = oc.quote_plan(25.0, "short", strikes, sessions=4, hv=0.60)
    assert puts and all(s < 25.0 for s in puts)
    assert oc.quote_plan(None, "long", strikes, sessions=4, hv=0.6) == []


# ---------------------------------------------------------------- log rows
def _at(hour, minute):
    return datetime(2026, 9, 28, hour, minute, 20, tzinfo=NY)


def test_tracker_flags_once_per_answer_and_owes_30_60_and_close_rows():
    tracker = oc.ChaseOutcomeTracker()
    candidate = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=0.41)
    rows = tracker.flag([candidate], now=_at(10, 40))
    assert len(rows) == 1 and rows[0]["kind"] == "flag" and rows[0]["mid"] == pytest.approx(0.90)
    assert tracker.flag([candidate], now=_at(10, 45)) == []  # same answer: no new flag
    assert tracker.observe({"ABC": 25.2}, now=_at(11, 5)) == []
    mids = {"ABC": 1.08}
    thirty = tracker.observe({"ABC": 25.2}, now=_at(11, 10),
                             option_mid=lambda flag: mids.get(flag["symbol"]))
    assert [r["horizon"] for r in thirty] == ["+30m"]
    assert thirty[0]["move_atr"] == pytest.approx(1.0)
    assert thirty[0]["move_atr_chase"] == pytest.approx(1.0)
    assert thirty[0]["option_mid"] == pytest.approx(1.08)
    assert thirty[0]["option_mid_change_pct"] == pytest.approx(20.0)
    sixty = tracker.observe({"ABC": 24.9}, now=_at(11, 40))
    assert [r["horizon"] for r in sixty] == ["+60m"]
    assert sixty[0]["option_mid"] is None and sixty[0]["option_mid_change_pct"] is None
    close = tracker.observe({"ABC": 25.4}, now=_at(16, 0))
    assert [r["horizon"] for r in close] == ["close"]
    assert close[0]["move_atr"] == pytest.approx(2.0)
    assert tracker.observe({"ABC": 25.4}, now=_at(16, 5)) == []


def test_a_short_flag_measures_the_move_in_the_chase_direction_and_refusals_are_flagged():
    tracker = oc.ChaseOutcomeTracker()
    refused = oc.pick_candidate(_pop(side="short", rvol=1.5), _chain(), today=TODAY)
    rows = tracker.flag([refused], now=_at(15, 50))
    assert rows[0]["status"] == "refused" and rows[0]["reason"] == "RVOL 1.5 < 2"
    # The close comes before +30: only the close row is written.
    out = tracker.observe({"ABC": 24.8}, now=_at(16, 0))
    assert [r["horizon"] for r in out] == ["close"]
    assert out[0]["move_atr"] == pytest.approx(-1.0)
    assert out[0]["move_atr_chase"] == pytest.approx(1.0)
    assert out[0]["option_mid"] is None


def test_no_data_rows_are_never_logged():
    tracker = oc.ChaseOutcomeTracker()
    assert tracker.flag([oc.no_data(_pop(), "IB not connected")], now=_at(10, 40)) == []


def test_log_append_summary_and_cli(tmp_path, capsys):
    path = tmp_path / "options_chase_log.jsonl"
    tracker = oc.ChaseOutcomeTracker()
    candidate = oc.pick_candidate(_pop(), _chain(), today=TODAY, hv=0.41)
    refused = oc.pick_candidate(_pop(symbol="XYZ", rvol=1.2), _chain(), today=TODAY)
    assert oc.append_records(path, tracker.flag([candidate, refused], now=_at(10, 40)))
    assert oc.append_records(path, tracker.observe(
        {"ABC": 25.2, "XYZ": 25.0}, now=_at(11, 10), option_mid=lambda _f: 1.08))
    rows = oc.load_records(path)
    assert [r["kind"] for r in rows] == ["flag", "flag", "outcome", "outcome"]
    summary = oc.summarize(rows)
    assert (summary["flags"], summary["candidates"], summary["refused"]) == (2, 1, 1)
    assert summary["horizons"]["+30m"]["outcomes"] == 2
    assert summary["horizons"]["+30m"]["with_option_mid"] == 1
    assert summary["horizons"]["+30m"]["median_option_mid_change_pct"] == pytest.approx(20.0)
    assert oc.main(["--summary", "--path", str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["candidates"] == 1


def test_a_failed_log_write_returns_false_and_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert oc.append_records(blocker / "sub" / "log.jsonl", [{"kind": "flag"}]) is False


def test_log_path_is_a_project_paths_constant():
    import project_paths

    assert Path(project_paths.OPTIONS_CHASE_LOG_FILE).name == "options_chase_log.jsonl"


# ---------------------------------------------------------------- the service (fake IB)
def _svc():
    from ui.services import options_chase_service as ocs

    return ocs


def _pop_row(symbol, score, rvol, last=25.0):
    return {"symbol": symbol, "pop_score": score, "rvol": rvol, "last": last, "atr": 0.2,
            "move15_pct": 2.0}


def _movers_board():
    return {"pop": {
        "long": [_pop_row("AAA", 3.0, 3.0), _pop_row("CCC", 4.0, 1.5),
                 _pop_row("BBB", 2.5, 2.5), _pop_row("DDD", 1.0, 2.1)],
        "short": [_pop_row("EEE", -2.0, 4.0)],
    }}


class FakeClient:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.closed = False

    def fetch_chain(self, symbol, *, side, last, hv, today):
        self.calls.append((symbol, side))
        if self.error is not None:
            raise self.error
        return _chain()

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _service(tmp_path, client, clock=None):
    ocs = _svc()
    return ocs.OptionsChaseService(
        client_factory=lambda: client, hv_provider=lambda _s, _d: 0.41,
        log_path=tmp_path / "options_chase_log.jsonl", monotonic=clock or Clock())


def test_service_chases_top_three_rvol2_pop_names_and_caches_five_minutes(tmp_path):
    ocs = _svc()
    assert ocs.OPTIONS_CHASE_MAX_PER_TICK == 3 and ocs.OPTIONS_CHASE_CACHE_SECONDS == 300
    client, clock = FakeClient(), Clock()
    service = _service(tmp_path, client, clock)
    results = service.run(_movers_board(), {"AAA": 25.0}, now=NOW)
    # CCC has the biggest move but RVOL 1.5: never chased; DDD is fourth.
    assert client.calls == [("AAA", "long"), ("BBB", "long"), ("EEE", "short")]
    assert set(results) == {"AAA|long", "BBB|long", "EEE|short"}
    assert results["AAA|long"]["status"] == "candidate" and results["EEE|short"]["right"] == "P"
    clock.t += 299
    service.run(_movers_board(), {}, now=NOW)
    assert len(client.calls) == 3  # cached
    clock.t += 2
    service.run(_movers_board(), {}, now=NOW)
    assert len(client.calls) == 6  # expired: fetched again
    board = service.annotate(_movers_board())
    texts = {r["symbol"]: oc.cell_text(r["opt"]) for r in board["pop"]["long"]}
    assert texts["AAA"] == "27C 10/02 · 0.85x0.95 · 11% · IV 62 (HV 41)"
    assert texts["CCC"] == "—" and texts["DDD"] == "—"
    flags = [r for r in oc.load_records(tmp_path / "options_chase_log.jsonl") if r["kind"] == "flag"]
    assert len(flags) == 3  # one flag per answer, not per tick


def test_annotate_never_mutates_the_board_it_was_given(tmp_path):
    service = _service(tmp_path, FakeClient())
    service.run(_movers_board(), {}, now=NOW)
    original = _movers_board()
    out = service.annotate(original)
    assert "opt" not in original["pop"]["long"][0]
    assert out["pop"]["long"][0]["opt"]["symbol"] == "AAA"


def test_not_connected_says_so_once_and_every_pop_row_reads_no_option_data(tmp_path, caplog):
    ocs = _svc()
    client = FakeClient(ocs.OptionDataError("IB not connected", connection=True))
    service = _service(tmp_path, client)
    with caplog.at_level("WARNING"):
        service.run(_movers_board(), {}, now=NOW)
        service.run(_movers_board(), {}, now=NOW)
    assert len(client.calls) == 2  # one connect attempt per tick, not per name
    assert sum("no option data this session" in r.message for r in caplog.records) == 1
    board = service.annotate(_movers_board())
    for row in board["pop"]["long"] + board["pop"]["short"]:
        assert oc.cell_text(row["opt"]) == "no option data (IB not connected)"
    assert not (tmp_path / "options_chase_log.jsonl").exists()  # no data is never logged


def _fake_app_class(mode):
    """A fake IB app: the real callbacks; the socket methods answer from a script."""
    ocs = _svc()

    class FakeApp(ocs._OptionApp):
        def __init__(self):
            super().__init__()
            self.connected = False
            self.requests = []
            self.cancelled = []
            self.data_types = []

        def connect(self, host, port, clientId):  # noqa: N803
            self.requests.append(("connect", host, port, clientId))
            self.connected = True

        def isConnected(self):  # noqa: N802
            return self.connected

        def run(self):
            self.nextValidId(1)
            while self.connected:
                time.sleep(0.005)

        def disconnect(self):
            self.connected = False

        def reqMarketDataType(self, kind):  # noqa: N802
            self.data_types.append(kind)

        def reqContractDetails(self, reqId, contract):  # noqa: N802,N803
            self.requests.append(("details", contract.symbol))
            self.contractDetails(reqId, SimpleNamespace(contract=SimpleNamespace(conId=123)))
            self.contractDetailsEnd(reqId)

        def reqSecDefOptParams(self, reqId, symbol, exchange, sec_type, con_id):  # noqa: N802,N803
            self.requests.append(("secdef", symbol, con_id))
            strikes = {20 + 0.5 * i for i in range(40)}
            self.securityDefinitionOptionParameter(reqId, "CBOE", 123, "ABC", "100",
                                                   {"20261002"}, {25.0})
            self.securityDefinitionOptionParameter(reqId, "SMART", 123, "ABC", "100",
                                                   {"20260930", "20261002", "20261009"}, strikes)
            self.securityDefinitionOptionParameterEnd(reqId)

        def reqMktData(self, reqId, contract, generic, snapshot, regulatory, options):  # noqa: N802,N803
            self.requests.append(("mkt", contract.strike, contract.right,
                                  contract.lastTradeDateOrContractMonth, snapshot, regulatory))
            if mode == "denied":
                self.error(reqId, 354, "Requested market data is not subscribed.")
                return
            if mode == "competing":
                self.error(reqId, 10197, "No market data during competing live session")
                return
            if mode == "no_security":
                self.error(reqId, 200, "No security definition has been found")
                return
            if mode == "silent":  # never answers: every snapshot times out
                return
            delta = max(0.02, 0.5 - 0.18 * (contract.strike - 25.0))
            self.tickPrice(reqId, 1, 0.50, None)
            self.tickPrice(reqId, 2, 0.54, None)
            self.tickOptionComputation(reqId, 13, 0, 0.7, delta, 0.52, 0, 0, 0, 0, 25.0)
            self.tickSnapshotEnd(reqId)

        def cancelMktData(self, reqId):  # noqa: N802,N803
            self.cancelled.append(reqId)

    return FakeApp


def _client(mode):
    ocs = _svc()
    apps = []

    def factory():
        apps.append(_fake_app_class(mode)())
        return apps[-1]

    client = ocs.IBOptionChainClient(host="127.0.0.1", port=7496, client_id=9145,
                                     app_factory=factory, connect_timeout_s=1.0,
                                     quote_timeout_s=0.5, request_gap_s=0)
    return client, apps


def test_ib_client_fetches_chain_and_snapshot_quotes_on_its_own_client_id():
    client, apps = _client("ok")
    try:
        chain = client.fetch_chain("ABC", side="long", last=25.0, hv=0.41, today=TODAY)
    finally:
        client.close()
    app = apps[0]
    assert app.requests[0] == ("connect", "127.0.0.1", 7496, 9145)
    assert app.data_types == [1]
    mkt = [r for r in app.requests if r[0] == "mkt"]
    assert 0 < len(mkt) <= oc.QUOTE_STRIKES
    assert all(r[2] == "C" and r[3] == "20261002" and r[4] is True and r[5] is False for r in mkt)
    assert len(app.cancelled) == len(mkt)
    result = oc.pick_candidate(_pop(), chain, today=TODAY, hv=0.41)
    assert result["status"] == "candidate" and result["expiry"] == "2026-10-02"
    assert oc.DELTA_MIN <= result["delta"] <= oc.DELTA_MAX
    assert result["mid"] == pytest.approx(0.52)


def test_no_option_permission_stops_requests_for_the_session(tmp_path, caplog):
    ocs = _svc()
    client, apps = _client("denied")
    service = ocs.OptionsChaseService(
        client_factory=lambda: client, hv_provider=lambda _s, _d: 0.41,
        log_path=tmp_path / "log.jsonl")
    try:
        with caplog.at_level("WARNING"):
            first = service.run(_movers_board(), {}, now=NOW)
            requests = len(apps[0].requests)
            second = service.run(_movers_board(), {}, now=NOW)
    finally:
        client.close()
    reason = "no option market-data permission (IB 354)"
    assert all(r["status"] == "no_data" and r["reason"] == reason for r in first.values())
    assert all(r["reason"] == reason for r in second.values())
    assert len(apps[0].requests) == requests  # nothing more asked of IB this session
    assert sum("no option data this session" in r.message for r in caplog.records) == 1
    board = service.annotate(_movers_board())
    assert oc.cell_text(board["pop"]["long"][1]["opt"]) == f"no option data ({reason})"


@pytest.mark.parametrize("mode, reason", [("silent", "no quotes (IB timeout)"),
                                          ("no_security", "no quotes (IB 200)")])
def test_no_quote_at_all_is_no_option_data_never_a_cached_refusal(tmp_path, mode, reason):
    ocs = _svc()
    client, apps = _client(mode)
    try:
        with pytest.raises(ocs.OptionDataError) as caught:
            client.fetch_chain("ABC", side="long", last=25.0, hv=0.41, today=TODAY)
        assert caught.value.reason == reason
        service = ocs.OptionsChaseService(
            client_factory=lambda: client, hv_provider=lambda _s, _d: 0.41,
            log_path=tmp_path / "log.jsonl", max_per_tick=1)
        first = service.run(_movers_board(), {}, now=NOW)
        asked = len(apps[0].requests)
        service.run(_movers_board(), {}, now=NOW)
    finally:
        client.close()
    assert first["AAA|long"]["status"] == "no_data" and first["AAA|long"]["reason"] == reason
    assert oc.cell_text(first["AAA|long"]) == f"no option data ({reason})"
    assert len(apps[0].requests) > asked  # not cached: asked again next tick
    assert not (tmp_path / "log.jsonl").exists()  # never logged as a refusal


def test_competing_live_session_is_retried_next_tick_not_latched_for_the_day(tmp_path):
    ocs = _svc()
    client, apps = _client("competing")
    service = ocs.OptionsChaseService(
        client_factory=lambda: client, hv_provider=lambda _s, _d: 0.41,
        log_path=tmp_path / "log.jsonl")
    try:
        first = service.run(_movers_board(), {}, now=NOW)
        asked = len(apps[0].requests)
        service.run(_movers_board(), {}, now=NOW)
    finally:
        client.close()
    reason = "competing live session (IB 10197)"
    assert first["AAA|long"]["reason"] == reason and service.down_reason == reason
    assert len(apps[0].requests) > asked  # retried on the next tick
    assert 10197 not in ocs.NO_PERMISSION_CODES


def test_connection_errors_are_bounded():
    ocs = _svc()
    app = ocs._OptionApp()
    for i in range(300):
        app.error(-1, 1100, f"lost {i}")
    assert len(app.connection_errors) == 200 and app.connection_errors[-1] == "1100: lost 299"


def test_movers_service_runs_the_chase_after_the_final_board_and_republishes(tmp_path):
    from ui.services import movers_service as ms

    chase = _service(tmp_path, FakeClient())
    service = ms.MoversService(autostart=False, options_chase=chase, clock=lambda: NOW)
    emitted = []
    service.moversChanged.connect(emitted.append)
    service._board = _movers_board()
    service._run_options_chase({"AAA": [{"close": 25.1}]}, NOW)
    assert emitted and emitted[-1]["pop"]["long"][0]["opt"]["status"] == "candidate"
    assert service.board()["pop"]["long"][0]["opt"]["strike"] == 27.0


def test_a_failing_chase_leaves_the_board_alone():
    from ui.services import movers_service as ms

    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("boom")

        def annotate(self, board):
            return dict(board)

        def close(self):
            pass

    service = ms.MoversService(autostart=False, options_chase=Broken(), clock=lambda: NOW)
    emitted = []
    service.moversChanged.connect(emitted.append)
    service._board = _movers_board()
    service._run_options_chase({}, NOW)
    assert emitted == [] and service.board() == _movers_board()


def test_a_failed_log_write_never_touches_the_board(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    ocs = _svc()
    service = ocs.OptionsChaseService(
        client_factory=FakeClient, hv_provider=lambda _s, _d: 0.41,
        log_path=blocker / "sub" / "options_chase_log.jsonl")
    results = service.run(_movers_board(), {}, now=NOW)
    assert results["AAA|long"]["status"] == "candidate"
    board = service.annotate(_movers_board())
    assert board["pop"]["long"][0]["opt"]["strike"] == 27.0


def test_log_rows_carry_the_option_mid_at_flag_and_a_fresh_cached_mid_later(tmp_path):
    clock = Clock()
    service = _service(tmp_path, FakeClient(), clock)
    service.run(_movers_board(), {"AAA": 25.0}, now=NOW)
    clock.t += 200  # cache still fresh at +30m in this fake clock
    later = datetime(2026, 9, 28, 11, 10, 20, tzinfo=NY)
    service.run(_movers_board(), {"AAA": 25.2, "BBB": 25.0, "EEE": 25.0}, now=later)
    rows = oc.load_records(tmp_path / "options_chase_log.jsonl")
    flag = next(r for r in rows if r["kind"] == "flag" and r["symbol"] == "AAA")
    assert flag["mid"] == pytest.approx(0.90) and flag["strike"] == 27.0
    outcome = next(r for r in rows if r["kind"] == "outcome" and r["symbol"] == "AAA")
    assert outcome["horizon"] == "+30m" and outcome["move_atr"] == pytest.approx(1.0)
    assert outcome["option_mid"] == pytest.approx(0.90)
    assert service.fresh_mid(flag) == pytest.approx(0.90)
    clock.t += 200  # the cached quote is now 400 s old: no mid is claimed
    assert service.fresh_mid(flag) is None


# ---------------------------------------------------------------- the Opt column
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _board_widget(board):
    from ui.widgets.movers_board import MoversBoard

    widget = MoversBoard(persist=False)
    widget.resize(420, 400)
    widget.update_board(board)
    widget.flush_pending_refresh()
    return widget


def _opt_index(widget, symbol):
    columns = [k for k, _h in widget.model._columns]
    row = [r["symbol"] for r in widget.model.rows()].index(symbol)
    return widget.model.index(row, columns.index("opt"))


def _chased_board(tmp_path):
    board = {"as_of": "2026-09-28T10:40:00-04:00",
             "state": {"state": "up_day", "pullback": False, "bounce": False},
             **_movers_board(), "dip": {"long": [], "short": []}}
    service = _service(tmp_path, FakeClient())
    service.run(board, {}, now=NOW)
    return service.annotate(board)


def test_pop_table_has_an_opt_column_with_candidate_text_and_full_hover(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from ui.widgets.movers_board import COLUMNS

    assert [k for k, _h in COLUMNS["pop"]][:5] == ["symbol", "move15_pct", "rvol", "lvl", "opt"]
    widget = _board_widget(_chased_board(tmp_path))
    index = _opt_index(widget, "AAA")
    assert widget.model.data(index) == "27C 10/02 · 0.85x0.95 · 11% · IV 62 (HV 41)"
    hover = widget.model.data(index, Qt.ItemDataRole.ToolTipRole)
    assert "candidate: 27C 10/02" in hover and "delta 0.26" in hover and "IV/HV 1.51x" in hover
    unchecked = _opt_index(widget, "CCC")
    assert widget.model.data(unchecked) == "—"
    assert "not checked" in widget.model.data(unchecked, Qt.ItemDataRole.ToolTipRole)
    assert not widget.table.isColumnHidden(index.column())  # fits the 420 px board


def test_opt_column_shows_no_option_data_and_the_refusal_reason(qapp, tmp_path):
    from PySide6.QtCore import Qt

    board = _chased_board(tmp_path)
    board["pop"]["long"][0]["opt"] = oc.no_data(_pop(symbol="AAA"), "IB not connected")
    board["pop"]["long"][2]["opt"] = oc.pick_candidate(
        _pop(symbol="BBB", rvol=1.4), _chain(), today=TODAY)
    widget = _board_widget(board)
    assert widget.model.data(_opt_index(widget, "AAA")) == "no option data (IB not connected)"
    index = _opt_index(widget, "BBB")
    assert widget.model.data(index) == "no chase (RVOL 1.4 < 2)"
    assert "refused: RVOL 1.4 < 2" in widget.model.data(index, Qt.ItemDataRole.ToolTipRole)


def test_clicking_the_opt_cell_only_copies_its_text(qapp, tmp_path):
    from PySide6.QtWidgets import QApplication

    widget = _board_widget(_chased_board(tmp_path))
    opened, focus = [], []
    widget.symbolActivated.connect(lambda *a: opened.append(a))
    widget.focusAddRequested.connect(lambda *a: focus.append(a))
    source = _opt_index(widget, "AAA")
    widget._on_clicked(widget.proxy.mapFromSource(source))
    assert QApplication.clipboard().text() == "27C 10/02 · 0.85x0.95 · 11% · IV 62 (HV 41)"
    assert opened == [] and focus == []
    assert "Copied" in widget.status_label.text()
    # Any other cell still opens the chart, as before.
    symbol_cell = widget.model.index(source.row(), 0)
    widget._on_clicked(widget.proxy.mapFromSource(symbol_cell))
    assert opened == [("AAA", "LONG")]


def test_the_desk_wires_one_options_chase_into_the_movers_service():
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert source.count("OptionsChaseService(") == 1
    assert "options_chase=OptionsChaseService()" in source
