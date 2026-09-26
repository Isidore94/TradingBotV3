"""The `momentum_scanner` universe source: IB-scanner strong names on a rolling, capped list.

Trader, 2026-09-26: "Help me capture a larger array of stocks then. A momentum universe sounds
PERFECT because we can use IB API market scanner like we do for the intraday stuff."
Names sighted by the scans stay 60 observed sessions after their last sighting, at most 300, join
the LONG list only (never universe_all.txt, which the Strength Board reads), never move a name any
other source holds, never count toward the write floor, and an IB outage keeps the list.
No test here reaches IB: the scanner is a fake app or an injected fetch.
"""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ib_market_scanner as ims  # noqa: E402
import momentum_universe as mu  # noqa: E402
import universe_builder as ub  # noqa: E402

DAY = date(2026, 9, 28)
GOOD = (50.0, 2.5e8)  # $50, $250M a day


def _sessions(count: int, *, end: date = DAY) -> list[date]:
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


# ---------------------------------------------------------------------------
# Scanner result parsing
# ---------------------------------------------------------------------------
class _Contract:
    def __init__(self, symbol):
        self.symbol = symbol


class _Details:
    def __init__(self, symbol):
        self.contract = _Contract(symbol)


class _FakeApp(ims._ScannerApp):
    """Answers each scan code with its own ranked rows, out of rank order."""

    ROWS = {
        "HIGH_VS_52W_HL": [(1, "NVDA"), (0, "BRK B"), (2, "XYZ WS")],
        "HIGH_VS_26W_HL": [(0, "NVDA"), (1, "PLTR")],
        "HIGH_VS_13W_HL": [(0, "CHEAP")],
        "TOP_PERC_GAIN": [(0, "THIN"), (1, "UNPRICED")],
    }
    subscriptions: list = []

    def connect(self, host, port, clientId):  # noqa: N803
        self._connected = True
        self.client_id_used = clientId

    def isConnected(self):  # noqa: N802
        return getattr(self, "_connected", False)

    def run(self):
        self.nextValidId(1)

    def disconnect(self):
        self._connected = False

    def reqScannerSubscription(self, reqId, sub, options, filters):  # noqa: N802,N803
        type(self).subscriptions.append((sub.scanCode, sub.numberOfRows, sub.abovePrice, self.client_id_used))
        for rank, symbol in self.ROWS[sub.scanCode]:
            self.scannerData(reqId, rank, _Details(symbol), "", "", "")
        self.scannerDataEnd(reqId)

    def cancelScannerSubscription(self, reqId):  # noqa: N802,N803
        pass


def test_scanner_results_parse_into_admitted_names_with_their_scans():
    _FakeApp.subscriptions = []
    scanner_cls = ims.IBMarketScanner

    def factory(**kwargs):
        return scanner_cls(app_factory=_FakeApp, **kwargs)

    with patch.object(ims, "IBMarketScanner", factory):
        results = mu.fetch_momentum_scans()
    assert [row[0] for row in _FakeApp.subscriptions] == list(mu.MOMENTUM_SCAN_CODES)
    assert {row[1] for row in _FakeApp.subscriptions} == {50}
    assert {row[2] for row in _FakeApp.subscriptions} == {5.0}
    assert {row[3] for row in _FakeApp.subscriptions} == {mu.MOMENTUM_CLIENT_ID}
    assert results["HIGH_VS_52W_HL"] == ["BRK-B", "NVDA"]  # rank order, warrant dropped, IB class -> dash
    metrics = {"NVDA": GOOD, "BRK-B": GOOD, "PLTR": GOOD, "CHEAP": (4.99, 9e8), "THIN": (20.0, 49.9e6),
               "UNPRICED": (None, None)}
    sighted = mu.sighted_symbols(results, metrics)
    assert sighted == {"BRK-B": ["HIGH_VS_52W_HL"], "NVDA": ["HIGH_VS_52W_HL", "HIGH_VS_26W_HL"],
                       "PLTR": ["HIGH_VS_26W_HL"]}


def test_the_dollar_volume_floor_is_the_lowest_s15_bucket_edge():
    import setup_permutations

    assert mu.MOMENTUM_MIN_DOLLAR_VOLUME_M == setup_permutations.DOLLAR_VOLUME_EDGES_M[0]
    assert mu.MOMENTUM_MIN_PRICE == 5.0
    assert mu.MOMENTUM_KEEP_SESSIONS == 60 and mu.MOMENTUM_MAX_SYMBOLS == 300


# ---------------------------------------------------------------------------
# Rolling membership and cap
# ---------------------------------------------------------------------------
def test_a_name_stays_sixty_observed_sessions_after_its_last_sighting():
    days = _sessions(62)
    membership = mu.empty_membership()
    membership = mu.update_membership(membership, {"OLD": ["HIGH_VS_52W_HL"]}, today=days[0])
    for day in days[1:61]:  # 60 more sessions without a sighting
        membership = mu.update_membership(membership, {"NEW": ["TOP_PERC_GAIN"]}, today=day)
    assert "OLD" in mu.active_members(membership)
    membership = mu.update_membership(membership, {}, today=days[61])  # the 61st session after
    assert "OLD" not in mu.active_members(membership)
    assert "OLD" not in membership["members"]
    assert membership["members"]["NEW"]["hits"] == 60
    assert membership["members"]["NEW"]["first_seen"] == days[1].isoformat()


def test_a_resighting_resets_the_clock_and_a_same_day_rerun_counts_once():
    days = _sessions(70)
    membership = mu.update_membership(mu.empty_membership(), {"NVDA": ["HIGH_VS_52W_HL"]}, today=days[0])
    membership = mu.update_membership(membership, {"NVDA": ["TOP_PERC_GAIN"]}, today=days[30])
    membership = mu.update_membership(membership, {"NVDA": ["TOP_PERC_GAIN"]}, today=days[30])
    for day in days[31:70]:
        membership = mu.update_membership(membership, {}, today=day)
    assert mu.active_members(membership) == ["NVDA"]
    assert membership["members"]["NVDA"]["hits"] == 2
    assert membership["members"]["NVDA"]["scans"] == ["TOP_PERC_GAIN", "HIGH_VS_52W_HL"]


def test_the_cap_keeps_the_most_recently_sighted_then_the_most_sighted():
    days = _sessions(3)
    membership = mu.update_membership(mu.empty_membership(), {"AAA": ["x"], "BBB": ["x"]}, today=days[0])
    membership = mu.update_membership(membership, {"BBB": ["x"], "CCC": ["x"]}, today=days[1])
    membership = mu.update_membership(membership, {"DDD": ["x"]}, today=days[2], cap=3)
    assert mu.active_members(membership) == ["DDD", "BBB", "CCC"]  # BBB's two sightings beat CCC's one
    assert set(membership["members"]) == {"BBB", "CCC", "DDD"}
    assert mu.active_members(membership, cap=2) == ["DDD", "BBB"]
    big = mu.update_membership(mu.empty_membership(), {f"S{i:03d}": ["x"] for i in range(400)}, today=DAY)
    assert len(big["members"]) == mu.MOMENTUM_MAX_SYMBOLS == len(mu.active_members(big))


# ---------------------------------------------------------------------------
# The store: once a day, IB down keeps it, unreadable is never overwritten
# ---------------------------------------------------------------------------
def test_ib_down_keeps_yesterdays_membership_and_never_writes(tmp_path):
    store = tmp_path / "momentum.json"
    days = _sessions(2)
    report = mu.refresh_membership({"NVDA": GOOD}, today=days[0], path=store,
                                   fetch=lambda: {"HIGH_VS_52W_HL": ["NVDA"]})
    assert report["refreshed"] and report["members"] == ["NVDA"]
    before = store.read_bytes()

    def down():
        raise ims.ScannerError("connect 127.0.0.1:7496 failed: refused")

    report = mu.refresh_membership({"NVDA": GOOD}, today=days[1], path=store, fetch=down)
    assert report["members"] == ["NVDA"] and not report["refreshed"] and "refused" in report["error"]
    assert store.read_bytes() == before


def test_the_scanner_is_asked_once_a_day(tmp_path):
    store = tmp_path / "momentum.json"
    calls = []

    def fetch():
        calls.append(1)
        return {"HIGH_VS_52W_HL": ["NVDA"]}

    mu.refresh_membership({"NVDA": GOOD}, today=DAY, path=store, fetch=fetch)
    report = mu.refresh_membership({"NVDA": GOOD}, today=DAY, path=store, fetch=fetch)
    assert len(calls) == 1 and report["members"] == ["NVDA"] and not report["refreshed"]
    mu.refresh_membership({"NVDA": GOOD}, today=DAY + timedelta(days=1), path=store, fetch=fetch, write=False)
    assert len(calls) == 1


def test_an_unreadable_store_is_never_overwritten(tmp_path):
    store = tmp_path / "momentum.json"
    store.write_text("{not json", encoding="utf-8")
    report = mu.refresh_membership({"NVDA": GOOD}, today=DAY, path=store,
                                   fetch=lambda: {"HIGH_VS_52W_HL": ["NVDA"]})
    assert report["members"] == [] and report["error"].startswith("unreadable")
    assert store.read_text(encoding="utf-8") == "{not json"


# ---------------------------------------------------------------------------
# build_universe
# ---------------------------------------------------------------------------
def _metrics(symbols, *, long_side=True, mixed=()):
    rows = []
    for symbol in symbols:
        side = long_side and symbol not in mixed
        rows.append({
            "symbol": symbol, "last_price": 50.0, "avg_volume_20d": 5e6, "dollar_volume_20d": 2.5e8,
            "sma_50": 40.0, "sma_100": 40.0, "sma_200": 40.0, "above_sma_50": side, "above_sma_100": side,
            "above_sma_200": side, "below_sma_50": not long_side and symbol not in mixed,
            "below_sma_100": not long_side and symbol not in mixed,
            "below_sma_200": not long_side and symbol not in mixed,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def home(tmp_path):
    files = {name: tmp_path / f"universe_{name}.txt" for name in ("all", "longs", "shorts")}
    with ExitStack() as stack:
        for attr, value in (("UNIVERSE_ALL_FILE", files["all"]), ("UNIVERSE_LONGS_FILE", files["longs"]),
                            ("UNIVERSE_SHORTS_FILE", files["shorts"]),
                            ("UNIVERSE_METADATA_FILE", tmp_path / "universe_metadata.csv")):
            stack.enter_context(patch.object(ub, attr, value))
        stack.enter_context(patch.object(
            ub, "UNIVERSE_INCLUDE_FILES", {name: tmp_path / f"universe_include_{name}.txt" for name in files}))
        stack.enter_context(patch.object(ub, "_universe_ledger_path", lambda: tmp_path / "ledger.jsonl"))
        stack.enter_context(patch.object(ub, "_snapshot_universe_lists", lambda: ""))
        stack.enter_context(patch.object(mu, "MOMENTUM_UNIVERSE_MEMBERSHIP_FILE", tmp_path / "momentum.json"))
        yield tmp_path, files


def _build(metrics, *, scans=None, journal=None, **kwargs):
    history = pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])

    def fetch():
        if scans is None:
            raise ims.ScannerError("IB down")
        return scans

    with patch.object(ub, "fetch_all_listed_symbols", return_value=["AAPL"]), \
            patch.object(ub, "fetch_optionable_symbols", return_value=["AAPL"]), \
            patch.object(ub, "fetch_price_history", return_value=history), \
            patch.object(ub, "compute_universe_metrics", return_value=metrics), \
            patch.object(ub, "fetch_market_caps", return_value={}), \
            patch.object(ub, "journal_traded_symbols", return_value=journal or {}), \
            patch.object(mu, "fetch_momentum_scans", fetch):
        return ub.build_universe(write_outputs=True, **kwargs)


def _read(path):
    return path.read_text(encoding="utf-8").split()


def test_momentum_names_join_the_long_list_only_and_nothing_else_moves(home):
    root, files = home
    (root / "universe_include_all.txt").write_text("TYPED\n", encoding="utf-8")
    (root / "universe_include_shorts.txt").write_text("TSHORT\n", encoding="utf-8")
    screen = [f"S{index:03d}" for index in range(30)]
    shorts = _metrics(["SH1"], long_side=False)
    # MOMO and MIXED (in the screen, on no side list) are strong but not on a side list.
    extra = _metrics(["MOMO", "MIXED", "TYPED", "TSHORT", "JNL"], mixed=("MIXED", "TYPED", "TSHORT", "JNL", "MOMO"))
    metrics = pd.concat([_metrics(screen), shorts, extra], ignore_index=True)
    # MOMO is priced but fails the screen, so only this source can bring it in.
    with patch.object(ub, "apply_universe_screen", side_effect=lambda frame, **_k: frame[frame["symbol"] != "MOMO"]):
        _build(metrics, force=True, journal={"JNL": "SHORT"})
    plain_lists = {name: _read(path) for name, path in files.items()}
    (root / "momentum.json").unlink(missing_ok=True)
    scans = {"HIGH_VS_52W_HL": ["MOMO", "S000", "SH1", "TYPED", "TSHORT", "JNL", "MIXED"]}
    with patch.object(ub, "apply_universe_screen", side_effect=lambda frame, **_k: frame[frame["symbol"] != "MOMO"]):
        result = _build(metrics, scans=scans, force=True, journal={"JNL": "SHORT"})
    lists = {name: _read(path) for name, path in files.items()}
    assert lists["all"] == plain_lists["all"]  # the Strength Board's list does not grow
    assert lists["shorts"] == plain_lists["shorts"]
    assert set(lists["longs"]) - set(plain_lists["longs"]) == {"MOMO", "MIXED"}
    assert set(plain_lists["longs"]) <= set(lists["longs"])
    assert result["momentum_scanner"] == ["MIXED", "MOMO"]
    row = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert row["stages"]["momentum_scanner"] == 2
    assert row["momentum_scanner"]["members"] == 7 and row["momentum_scanner"]["refreshed"]


def test_the_write_floor_ignores_momentum_names(home):
    root, files = home
    files["all"].write_text("\n".join(f"P{index:04d}" for index in range(1487)) + "\n", encoding="utf-8")
    momo = [f"M{index:03d}" for index in range(300)]
    metrics = _metrics([f"S{index:03d}" for index in range(700)] + momo, mixed=tuple(momo))
    screen = patch.object(ub, "apply_universe_screen", side_effect=lambda frame, **_k: frame[~frame["symbol"].isin(momo)])
    # 700 screen names are under the 743 floor; 300 momentum names must not lift them over it.
    with screen, pytest.raises(ub.UniverseWriteRefused):
        _build(metrics, scans={"HIGH_VS_52W_HL": momo})
    row = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert row["refused"] and row["floor"] == 743 and row["stages"]["momentum_scanner"] == 300
    assert len(_read(files["all"])) == 1487


def test_ib_down_rebuild_keeps_the_stored_momentum_names(home):
    root, files = home
    screen = [f"S{index:03d}" for index in range(30)]
    metrics = _metrics(screen + ["MOMO"], mixed=("MOMO",))
    _build(metrics, scans={"HIGH_VS_52W_HL": ["MOMO"]}, force=True)
    assert "MOMO" in _read(files["longs"])
    store = json.loads((root / "momentum.json").read_text(encoding="utf-8"))
    store["refreshed_on"] = "2026-01-01"  # a later day: the scanner is asked again and is down
    (root / "momentum.json").write_text(json.dumps(store), encoding="utf-8")
    result = _build(metrics, scans=None, force=True)
    assert result["momentum_scanner"] == ["MOMO"] and "MOMO" in _read(files["longs"])
    row = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert row["momentum_scanner"]["error"] == "IB down" and not row["momentum_scanner"]["refreshed"]


def test_a_broken_momentum_stage_never_fails_the_rebuild(home):
    _root, files = home
    screen = [f"S{index:03d}" for index in range(30)]
    with patch.object(mu, "refresh_membership", side_effect=RuntimeError("boom")):
        result = _build(_metrics(screen), force=True)
    assert result["momentum_scanner"] == [] and len(_read(files["all"])) == 30
