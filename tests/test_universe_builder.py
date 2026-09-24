"""Tests for the self-sufficient universe builder (pure parsing/screening only)."""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import universe_builder as ub  # noqa: E402


@contextlib.contextmanager
def _fake_module(name, module):
    """Swap one sys.modules entry and restore only that one.

    patch.dict(sys.modules) would also drop every module first imported inside
    the block (pyarrow's pandas types), and re-importing them later fails.
    """
    saved = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved

NASDAQ_SAMPLE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
ZTEST|Test Listing|Q|Y|N|100|N|N
File Creation Time: 0630202522:01|||||||
"""

OTHER_SAMPLE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Class B|N|BRK B|N|100|N|BRK=B
SPY|SPDR S&P 500|P|SPY|Y|100|N|SPY
BAD$|Structured Product|N|BAD$|N|100|N|BAD$
"""


class SymbolDirectoryTests(unittest.TestCase):
    def test_parse_drops_etfs_tests_and_structured(self):
        symbols = ub.parse_symbol_directory(NASDAQ_SAMPLE, OTHER_SAMPLE)
        self.assertIn("AAPL", symbols)
        self.assertIn("BRK-B", symbols)  # dot converted to Yahoo dash form
        self.assertNotIn("QQQ", symbols)  # ETF
        self.assertNotIn("SPY", symbols)  # ETF
        self.assertNotIn("ZTEST", symbols)  # test issue
        self.assertFalse(any("$" in s for s in symbols))


class WeeklysParseTests(unittest.TestCase):
    def test_parse_weeklys_extracts_tickers(self):
        text = 'Available Weeklys,Name\nAAPL,"Apple Inc"\nTSLA,"Tesla"\n"Standard Weeklys",\n'
        symbols = ub.parse_weeklys_csv(text)
        self.assertIn("AAPL", symbols)
        self.assertIn("TSLA", symbols)
        self.assertNotIn("STANDARD WEEKLYS", symbols)


class OptionableParseTests(unittest.TestCase):
    def test_parse_cboe_symbol_directory(self):
        text = (
            "Company Name, Stock Symbol, DPM Name, Post/Station, Global Trading Hours DPM\n"
            '"Apple Inc","AAPL","Citadel Securities LLC","9/1","-"\n'
            '"Berkshire Hathaway CL B","BRK.B","Belvedere Trading LLC","1/1","-"\n'
            '"Bad Row","N/A$","X","1/1","-"\n'
        )
        symbols = ub.parse_cboe_symbol_directory(text)
        self.assertIn("AAPL", symbols)
        self.assertIn("BRK-B", symbols)  # dot converted to Yahoo dash form
        self.assertNotIn("STOCK SYMBOL", symbols)  # header skipped
        self.assertFalse(any("$" in s for s in symbols))


def _history(symbol: str, *, price: float, volume: float, rising: bool, periods: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2025-08-01", periods=periods)
    step = 0.2 if rising else -0.2
    start = price - step * periods
    rows = [
        {"symbol": symbol, "datetime": dt, "close": start + step * i, "volume": volume}
        for i, dt in enumerate(dates)
    ]
    return pd.DataFrame(rows)


class ScreenTests(unittest.TestCase):
    def test_metrics_and_screen(self):
        history = pd.concat(
            [
                _history("GOOD", price=50.0, volume=2_000_000, rising=True),
                _history("THIN", price=50.0, volume=100_000, rising=True),
                _history("CHEAP", price=2.0, volume=5_000_000, rising=True),
                _history("DOWN", price=40.0, volume=3_000_000, rising=False),
            ],
            ignore_index=True,
        )
        metrics = ub.compute_universe_metrics(history)
        self.assertEqual(len(metrics), 4)

        screened = ub.apply_universe_screen(
            metrics,
            market_caps_m={"GOOD": 5000.0, "DOWN": 8000.0, "THIN": 5000.0, "CHEAP": 5000.0},
        )
        symbols = set(screened["symbol"])
        self.assertIn("GOOD", symbols)
        self.assertIn("DOWN", symbols)  # base screen keeps downtrends; trend split happens later
        self.assertNotIn("THIN", symbols)
        self.assertNotIn("CHEAP", symbols)

        good = screened[screened["symbol"] == "GOOD"].iloc[0]
        self.assertTrue(good["above_sma_100"] and good["above_sma_200"])
        self.assertFalse(good["below_sma_50"])
        down = screened[screened["symbol"] == "DOWN"].iloc[0]
        self.assertFalse(down["above_sma_100"] or down["above_sma_200"])
        # The short screen needs all three (50/100/200) below-flags true.
        self.assertTrue(down["below_sma_50"] and down["below_sma_100"] and down["below_sma_200"])

    def test_compare_symbol_lists_normalizes_and_diffs(self):
        result = ub.compare_symbol_lists(
            ours=["AAPL", "BRK-B", "NVDA", "EXTRA"],
            theirs=["aapl", "BRK.B", "NVDA", "MISSING"],
        )
        self.assertEqual(result["matched"], ["AAPL", "BRK-B", "NVDA"])
        self.assertEqual(result["only_ours"], ["EXTRA"])
        self.assertEqual(result["only_theirs"], ["MISSING"])
        self.assertEqual(result["theirs_count"], 4)
        self.assertEqual(result["overlap_pct"], 75.0)

    def test_compare_symbol_lists_empty_external(self):
        result = ub.compare_symbol_lists(ours=["AAPL"], theirs=[])
        self.assertEqual(result["overlap_pct"], 0.0)
        self.assertEqual(result["matched"], [])

    def test_compare_maps_separatorless_class_shares(self):
        # TC2000 writes BRKB for what Yahoo calls BRK-B: same company, one match.
        result = ub.compare_symbol_lists(ours=["BRK-B", "AAPL"], theirs=["BRKB", "AAPL"])
        self.assertEqual(result["matched"], ["AAPL", "BRK-B"])
        self.assertEqual(result["only_ours"], [])
        self.assertEqual(result["only_theirs"], [])
        self.assertEqual(result["overlap_pct"], 100.0)

    def test_merge_external_is_durable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            longs_file = root / "universe_longs.txt"
            include_file = root / "universe_include_longs.txt"
            longs_file.write_text("AAPL\nNVDA\n", encoding="utf-8")
            with (
                patch.dict(ub.UNIVERSE_LIST_FILES, {"longs": longs_file}),
                patch.dict(ub.UNIVERSE_INCLUDE_FILES, {"longs": include_file}),
            ):
                result = ub.merge_external_into_universe("longs", ["NVDA", "SEM", "BRKB"])
                self.assertEqual(result["added"], ["BRKB", "SEM"])
                self.assertEqual(result["total"], 4)
                # Union written to the list file, additions remembered in the include file.
                self.assertEqual(
                    longs_file.read_text(encoding="utf-8").split(), ["AAPL", "BRKB", "NVDA", "SEM"]
                )
                self.assertEqual(include_file.read_text(encoding="utf-8").split(), ["BRKB", "SEM"])
                # Second merge with the same list is a no-op.
                again = ub.merge_external_into_universe("longs", ["SEM"])
                self.assertEqual(again["added_count"], 0)

    def test_small_cap_dropped_but_unknown_cap_kept(self):
        metrics = ub.compute_universe_metrics(
            pd.concat(
                [
                    _history("SMALL", price=30.0, volume=2_000_000, rising=True),
                    _history("UNKNOWN", price=30.0, volume=2_000_000, rising=True),
                ],
                ignore_index=True,
            )
        )
        screened = ub.apply_universe_screen(metrics, market_caps_m={"SMALL": 300.0, "UNKNOWN": 0.0})
        symbols = set(screened["symbol"])
        self.assertNotIn("SMALL", symbols)
        self.assertIn("UNKNOWN", symbols)




class PriceHistoryShapeTests(unittest.TestCase):
    """A chunk sub-frame whose date axis is not named ``Date``.

    Defect found on the desk 2026-08-17 06:00: the Monday universe rebuild died
    with ``KeyError: "['datetime'] not in index"`` (autopilot.log 06:00:16),
    raised by the column selection at the end of ``fetch_price_history``'s
    per-symbol loop. yfinance normally names the daily index ``Date``, so
    ``reset_index()`` yields a ``Date`` column the rename turns into
    ``datetime``; when the response arrives with an unnamed index instead,
    ``reset_index()`` yields ``index`` and the selection raises. One malformed
    sub-frame aborted the entire rebuild rather than being skipped like every
    other per-symbol fault in that loop.
    """

    @staticmethod
    def _daily_frame(index_name):
        index = pd.date_range("2026-01-02", periods=30, freq="D")
        index.name = index_name
        return pd.DataFrame(
            {
                "Open": [10.0] * 30,
                "High": [11.0] * 30,
                "Low": [9.0] * 30,
                "Close": [10.5] * 30,
                "Adj Close": [10.5] * 30,
                "Volume": [2_000_000] * 30,
            },
            index=index,
        )

    def _download_stub(self, index_name):
        raw = pd.concat({"AAPL": self._daily_frame(index_name)}, axis=1)

        def _download(**_kwargs):
            return raw

        return _download

    def _fetch(self, index_name):
        module = type(sys)("yfinance")
        module.download = self._download_stub(index_name)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "price_history.parquet"
            # conftest's offline guard replaces fetch_price_history with a
            # stub returning an empty frame - which is the point of the guard,
            # and exactly wrong here, because this fetch IS the subject. The
            # guard stashes the original for precisely this case; taking it
            # back is still hermetic, because yfinance itself is faked above.
            original = getattr(
                ub, "_offline_original_fetch_price_history", ub.fetch_price_history
            )
            with _fake_module("yfinance", module), \
                    patch.object(ub, "PRICE_HISTORY_CACHE", cache):
                return original(["AAPL"], refresh=True)

    def test_named_date_index_still_works(self):
        history = self._fetch("Date")
        self.assertEqual(list(history.columns), ["symbol", "datetime", "close", "volume"])
        self.assertEqual(len(history), 30)

    def test_unnamed_index_does_not_abort_the_rebuild(self):
        history = self._fetch(None)
        self.assertEqual(list(history.columns), ["symbol", "datetime", "close", "volume"])
        self.assertEqual(len(history), 30)

    def test_intraday_style_datetime_index_is_accepted(self):
        history = self._fetch("Datetime")
        self.assertEqual(list(history.columns), ["symbol", "datetime", "close", "volume"])
        self.assertEqual(len(history), 30)


class UniverseWriteGuardTests(unittest.TestCase):
    """A fetch outage must not blank the universe.

    plan.md sec 5: "A failed publish never destroys the last verified report."
    ``build_universe`` used to write ``universe_all/longs/shorts`` unconditionally,
    so a screen that produced nothing overwrote a good universe with an empty
    file. The rebuild must fail loudly and leave the previous lists in place.
    """

    def test_empty_screen_refuses_to_write(self):
        empty = pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])
        with patch.object(ub, "fetch_all_listed_symbols", return_value=["AAPL"]), \
                patch.object(ub, "fetch_optionable_symbols", return_value=["AAPL"]), \
                patch.object(ub, "fetch_price_history", return_value=empty), \
                patch.object(ub, "_write_watchlist") as write_watchlist:
            with self.assertRaises(RuntimeError) as caught:
                ub.build_universe(write_outputs=True)
        write_watchlist.assert_not_called()
        self.assertIn("0 symbols", str(caught.exception))

    def test_empty_screen_without_outputs_still_returns(self):
        empty = pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])
        with patch.object(ub, "fetch_all_listed_symbols", return_value=["AAPL"]), \
                patch.object(ub, "fetch_optionable_symbols", return_value=["AAPL"]), \
                patch.object(ub, "fetch_price_history", return_value=empty):
            result = ub.build_universe(write_outputs=False)
        self.assertEqual(result["all"], [])


class UniverseWriteFloorTests(unittest.TestCase):
    """A partial rebuild must not overwrite a good universe (plan.md R9.1).

    On 2026-08-20 13:31-13:35 PT a rebuild that priced ~25% of the listing
    replaced a 1,487-name universe with a few hundred, and the D1 scanner ran
    409-533 symbols for the whole of 2026-08-21 instead of its usual 1,088-1,513.
    ``build_universe`` refused to write only at *exactly* zero symbols, so there
    was no floor between "everything" and "nothing".
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.all_file = root / "universe_all.txt"
        self.longs_file = root / "universe_longs.txt"
        self.shorts_file = root / "universe_shorts.txt"
        self.metadata_file = root / "universe_metadata.csv"
        self.ledger_file = root / "job_ledger.jsonl"
        for attr, value in (
            ("UNIVERSE_ALL_FILE", self.all_file),
            ("UNIVERSE_LONGS_FILE", self.longs_file),
            ("UNIVERSE_SHORTS_FILE", self.shorts_file),
            ("UNIVERSE_METADATA_FILE", self.metadata_file),
        ):
            patcher = patch.object(ub, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Manual include files live in the same throwaway root so a developer's
        # real include lists can never leak into the counts under test.
        patcher = patch.object(
            ub,
            "UNIVERSE_INCLUDE_FILES",
            {name: root / f"universe_include_{name}.txt" for name in ("all", "longs", "shorts")},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(ub, "_universe_ledger_path", lambda: self.ledger_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _metrics(count: int) -> pd.DataFrame:
        """A screened-shaped frame of ``count`` names that all pass and all rank long."""
        symbols = [f"S{i:05d}" for i in range(count)]
        return pd.DataFrame(
            {
                "symbol": symbols,
                "last_price": [50.0] * count,
                "avg_volume_20d": [5_000_000.0] * count,
                "dollar_volume_20d": [250_000_000.0] * count,
                "sma_50": [40.0] * count,
                "sma_100": [40.0] * count,
                "sma_200": [40.0] * count,
                "above_sma_50": [True] * count,
                "above_sma_100": [True] * count,
                "above_sma_200": [True] * count,
                "below_sma_50": [False] * count,
                "below_sma_100": [False] * count,
                "below_sma_200": [False] * count,
            }
        )

    def _seed_previous(self, count: int) -> None:
        self.all_file.write_text("\n".join(f"P{i:05d}" for i in range(count)) + "\n", encoding="utf-8")

    def _build(self, produced: int, **kwargs):
        history = pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])
        with patch.object(ub, "fetch_all_listed_symbols", return_value=["AAPL"]), \
                patch.object(ub, "fetch_optionable_symbols", return_value=["AAPL"]), \
                patch.object(ub, "fetch_price_history", return_value=history), \
                patch.object(ub, "compute_universe_metrics", return_value=self._metrics(produced)), \
                patch.object(ub, "fetch_market_caps", return_value={}):
            return ub.build_universe(write_outputs=True, **kwargs)

    def _ledger_rows(self) -> list[dict]:
        if not self.ledger_file.exists():
            return []
        return [
            json.loads(line)
            for line in self.ledger_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # -- the four pinned cases -------------------------------------------
    def test_the_2026_08_20_collapse_refuses_to_write(self):
        """1,487 -> ~400 is the shape that actually happened. It must be stopped."""
        self._seed_previous(1487)
        with self.assertRaises(RuntimeError) as caught:
            self._build(400)
        self.assertIn("floor", str(caught.exception).lower())
        # The good universe is still on disk, untouched.
        self.assertEqual(len(self.all_file.read_text(encoding="utf-8").split()), 1487)
        self.assertFalse(self.longs_file.exists())

    def test_a_normal_shrink_still_writes(self):
        """1,487 -> 1,450 is ordinary churn and must not be blocked."""
        self._seed_previous(1487)
        result = self._build(1450)
        self.assertEqual(len(result["all"]), 1450)
        self.assertEqual(len(self.all_file.read_text(encoding="utf-8").split()), 1450)

    def test_zero_symbols_still_refuses(self):
        """The pre-existing zero guard is unchanged and is not a floor case."""
        self._seed_previous(1487)
        with self.assertRaises(RuntimeError) as caught:
            self._build(0)
        self.assertIn("0 symbols", str(caught.exception))
        self.assertEqual(len(self.all_file.read_text(encoding="utf-8").split()), 1487)

    def test_unreadable_prior_universe_fails_open(self):
        """Never leave the desk with no universe because we could not measure the old one."""
        self._seed_previous(1487)
        with patch.object(ub, "_read_universe_count", side_effect=OSError("locked")):
            result = self._build(400)
        self.assertEqual(len(result["all"]), 400)
        self.assertEqual(len(self.all_file.read_text(encoding="utf-8").split()), 400)

    # -- the carve-out and the floor arithmetic --------------------------
    def test_force_bypasses_the_floor(self):
        """A manual rebuild carves out exactly as the quiet-hours gate does."""
        self._seed_previous(1487)
        result = self._build(400, force=True)
        self.assertEqual(len(result["all"]), 400)
        self.assertEqual(len(self.all_file.read_text(encoding="utf-8").split()), 400)

    def test_force_does_not_bypass_the_zero_guard(self):
        """plan.md sec 5: a failed publish never destroys the last verified report."""
        self._seed_previous(1487)
        with self.assertRaises(RuntimeError):
            self._build(0, force=True)
        self.assertEqual(len(self.all_file.read_text(encoding="utf-8").split()), 1487)

    def test_floor_is_the_larger_of_500_and_half(self):
        self.assertEqual(ub.universe_write_floor(1487), 743)
        self.assertEqual(ub.universe_write_floor(1200), 600)
        # Below 1,000 the absolute floor binds instead of the fraction.
        self.assertEqual(ub.universe_write_floor(900), 500)
        # Nothing to protect: a missing or empty prior universe fails open.
        self.assertEqual(ub.universe_write_floor(None), 0)
        self.assertEqual(ub.universe_write_floor(0), 0)

    def test_first_ever_build_writes(self):
        """No prior file at all is the first build, not a collapse."""
        result = self._build(600)
        self.assertEqual(len(result["all"]), 600)

    # -- the ledger row --------------------------------------------------
    def test_a_refused_rebuild_is_still_recorded(self):
        self._seed_previous(1487)
        with self.assertRaises(RuntimeError):
            self._build(400)
        rows = [r for r in self._ledger_rows() if r.get("event") == "universe_rebuild"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["refused"])
        self.assertEqual(row["before"]["all"], 1487)
        self.assertEqual(row["after"]["all"], 400)
        self.assertEqual(row["floor"], 743)
        self.assertFalse(row["forced"])

    def test_a_successful_rebuild_is_recorded_too(self):
        self._seed_previous(1487)
        self._build(1450)
        rows = [r for r in self._ledger_rows() if r.get("event") == "universe_rebuild"]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["refused"])
        self.assertEqual(rows[0]["before"]["all"], 1487)
        self.assertEqual(rows[0]["after"]["all"], 1450)

    def test_the_ledger_row_never_becomes_a_phantom_job(self):
        """It is an audit row in the job ledger, not a job.

        ``JobLedger._replay`` only reduces events carrying a ``key``; ours has
        none, so a rebuild can never invent a QUEUED job that
        ``operations_audit`` would then report on.
        """
        self._seed_previous(1487)
        self._build(1450)
        rows = [r for r in self._ledger_rows() if r.get("event") == "universe_rebuild"]
        self.assertTrue(rows)
        self.assertNotIn("key", rows[0])
        from job_ledger import JobLedger

        self.assertEqual(JobLedger(self.ledger_file).jobs_for_date(""), [])

    def test_a_ledger_outage_never_blocks_the_rebuild(self):
        """The audit row is best effort; it can never cost the desk its universe.

        The outage is simulated by making the WRITE raise, not by naming a path
        that happens to be unwritable. ``Q:/nope/...`` was a Windows drive that
        does not exist - and on macOS or Linux ``Q:`` is a perfectly legal
        directory name, so the ledger wrote successfully into the repository
        root and this test passed without ever exercising an outage.
        """
        self._seed_previous(1487)

        def _refuse(*args, **kwargs):
            raise OSError("ledger unavailable")

        with patch.object(ub, "_universe_ledger_path", _refuse):
            result = self._build(1450)
        self.assertEqual(len(result["all"]), 1450)


    # -- P0-1: a typed refusal and per-stage evidence --------------------
    def test_the_refusal_carries_its_counts(self):
        self._seed_previous(1455)
        with self.assertRaises(ub.UniverseWriteRefused) as caught:
            self._build(343)
        self.assertEqual(caught.exception.produced, 343)
        self.assertEqual(caught.exception.floor, 727)
        self.assertEqual(caught.exception.kept, 1455)
        self.assertIsInstance(caught.exception, RuntimeError)

    def test_the_ledger_row_counts_every_stage(self):
        self._seed_previous(1487)
        self._build(1450)
        row = [r for r in self._ledger_rows() if r.get("event") == "universe_rebuild"][0]
        stages = row["stages"]
        for key in (
            "directory",
            "after_options_filter",
            "priced",
            "passed_screen",
            "after_include_lists",
        ):
            self.assertIn(key, stages)
        self.assertEqual(stages["priced"], 1450)
        self.assertEqual(stages["after_include_lists"], 1450)
        self.assertIn("yfinance", row)


class PriceFetchBatchErrorTests(unittest.TestCase):
    """The 104 s run on 2026-09-23 hints at failed yfinance batches; count them."""

    def test_failed_batches_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[str] = []

            def fake_download(**kwargs):
                calls.append(kwargs["tickers"])
                raise ValueError("rate limited")

            fake_yf = type(sys)("yfinance")
            fake_yf.download = fake_download
            stats: dict = {}
            with _fake_module("yfinance", fake_yf), \
                    patch.object(ub, "PRICE_HISTORY_CACHE", Path(tmp) / "ph.parquet"), \
                    patch.object(ub, "YF_CHUNK_RETRY_PAUSE_SECONDS", 0), \
                    patch.object(ub, "YF_CHUNK_PAUSE_SECONDS", 0), \
                    patch.object(ub, "YF_CHUNK_SIZE", 2):
                # conftest's offline guard stubs fetch_price_history; yfinance is faked above.
                fetch = getattr(ub, "_offline_original_fetch_price_history", ub.fetch_price_history)
                history = fetch(["A", "B", "C"], refresh=True, stats=stats)
        self.assertTrue(history.empty)
        self.assertEqual(stats["batches"], 2)
        self.assertEqual(stats["batch_errors"], 2)
        self.assertEqual(stats["batch_retry_failures"], 2)
        self.assertEqual(len(calls), 4)


class UniverseRestoreSnapshotTests(unittest.TestCase):
    """``--restore-snapshot`` is the tested CLI for putting a good universe back."""

    STAMP = "20260922T130004"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.lists = {
            "all": root / "universe_all.txt",
            "longs": root / "universe_longs.txt",
            "shorts": root / "universe_shorts.txt",
        }
        self.snapshots = root / "snapshots"
        self.ledger_file = root / "job_ledger.jsonl"
        for attr, value in (
            ("UNIVERSE_ALL_FILE", self.lists["all"]),
            ("UNIVERSE_LONGS_FILE", self.lists["longs"]),
            ("UNIVERSE_SHORTS_FILE", self.lists["shorts"]),
            ("UNIVERSE_SNAPSHOT_DIR", self.snapshots),
        ):
            patcher = patch.object(ub, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(ub, "_universe_ledger_path", lambda: self.ledger_file)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The collapsed lists now on disk.
        for name, count in (("all", 343), ("longs", 109), ("shorts", 165)):
            self._write(self.lists[name], "C", count)
        # The good snapshot.
        good = self.snapshots / f"universe-{self.STAMP}"
        good.mkdir(parents=True)
        for name, count in (("all", 1455), ("longs", 542), ("shorts", 584)):
            self._write(good / f"universe_{name}.txt", "G", count)

    @staticmethod
    def _write(path: Path, prefix: str, count: int) -> None:
        path.write_text("\n".join(f"{prefix}{i:05d}" for i in range(count)) + "\n", encoding="utf-8")

    @staticmethod
    def _count(path: Path) -> int:
        return len(path.read_text(encoding="utf-8").split())

    def _rows(self) -> list[dict]:
        if not self.ledger_file.exists():
            return []
        return [json.loads(line) for line in self.ledger_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_round_trip_restores_all_three_lists_and_records_it(self):
        self.assertEqual(ub.main(["--restore-snapshot", self.STAMP]), 0)
        self.assertEqual(self._count(self.lists["all"]), 1455)
        self.assertEqual(self._count(self.lists["longs"]), 542)
        self.assertEqual(self._count(self.lists["shorts"]), 584)
        rows = [r for r in self._rows() if r.get("event") == "universe_restore"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stamp"], self.STAMP)
        self.assertEqual(rows[0]["before"]["all"], 343)
        self.assertEqual(rows[0]["after"]["all"], 1455)
        self.assertNotIn("key", rows[0])
        # The collapsed lists were snapshotted first, so the restore is reversible.
        aside = [p for p in self.snapshots.iterdir() if p.name != f"universe-{self.STAMP}"]
        self.assertEqual(len(aside), 1)
        self.assertEqual(self._count(aside[0] / "universe_all.txt"), 343)

    def test_the_universe_prefix_is_accepted(self):
        self.assertEqual(ub.main(["--restore-snapshot", f"universe-{self.STAMP}"]), 0)
        self.assertEqual(self._count(self.lists["all"]), 1455)

    def test_a_restore_survives_its_own_snapshot_pruning(self):
        """A full snapshot folder prunes the oldest; the source may be that one."""
        with patch.object(ub, "UNIVERSE_SNAPSHOT_KEEP", 1):
            self.assertEqual(ub.main(["--restore-snapshot", self.STAMP]), 0)
        self.assertEqual(self._count(self.lists["all"]), 1455)
        self.assertEqual(self._count(self.lists["shorts"]), 584)

    def test_a_missing_snapshot_changes_nothing(self):
        self.assertEqual(ub.main(["--restore-snapshot", "20990101T000000"]), 2)
        self.assertEqual(self._count(self.lists["all"]), 343)
        self.assertEqual([r for r in self._rows() if r.get("event") == "universe_restore"], [])

    def test_a_path_like_stamp_is_refused(self):
        self.assertEqual(ub.main(["--restore-snapshot", "../universe-" + self.STAMP]), 2)
        self.assertEqual(self._count(self.lists["all"]), 343)

    def test_an_incomplete_snapshot_is_refused(self):
        (self.snapshots / f"universe-{self.STAMP}" / "universe_shorts.txt").unlink()
        self.assertEqual(ub.main(["--restore-snapshot", self.STAMP]), 2)
        self.assertEqual(self._count(self.lists["all"]), 343)
        self.assertEqual(self._count(self.lists["shorts"]), 165)

    def test_an_empty_snapshot_is_refused(self):
        (self.snapshots / f"universe-{self.STAMP}" / "universe_all.txt").write_text("", encoding="utf-8")
        self.assertEqual(ub.main(["--restore-snapshot", self.STAMP]), 2)
        self.assertEqual(self._count(self.lists["all"]), 343)


class UniverseForceCarveOutWiringTests(unittest.TestCase):
    """The floor's carve-out is only real if a manual rebuild actually reaches it.

    plan.md R9.1: "a manual rebuild keeps a ``force=True`` carve-out exactly as
    the quiet-hours gate does". Both manual entry points are pinned here so the
    wiring cannot be dropped while the flag quietly survives.
    """

    def test_only_override_floor_reaches_the_write_floor(self):
        """``skip_stale_check`` and ``override_floor`` are separate (P0-1).

        On 2026-09-23 the 13:02 stale tick passed ``force=True`` to skip the
        stale check, and the same flag skipped the write floor: 343 names
        replaced 1,455.
        """
        import autopilot_core as core

        seen: list[bool] = []

        def fake_build(**kwargs):
            seen.append(bool(kwargs.get("force")))
            return {"all": ["AAPL"], "longs": ["AAPL"], "shorts": []}

        with patch.dict(
            sys.modules,
            {"universe_builder": type(sys)("universe_builder")},
        ):
            sys.modules["universe_builder"].build_universe = fake_build
            sys.modules["universe_builder"].DEFAULT_OPTIONS_FILTER = "optionable"
            sys.modules["universe_builder"].UniverseWriteRefused = ub.UniverseWriteRefused
            # The manual button: override all the way down to the write floor.
            self.assertEqual(
                core.rebuild_universe_if_stale(skip_stale_check=True, override_floor=True, built_at=None),
                "rebuilt",
            )
            # The scheduled stale tick: skips the stale check, never the floor.
            self.assertEqual(
                core.rebuild_universe_if_stale(skip_stale_check=True, override_floor=False, built_at=None),
                "rebuilt",
            )
        self.assertEqual(seen, [True, False])

    def test_the_old_force_flag_is_gone(self):
        import inspect

        import autopilot_core as core

        params = inspect.signature(core.rebuild_universe_if_stale).parameters
        self.assertNotIn("force", params)
        self.assertIn("skip_stale_check", params)
        self.assertIn("override_floor", params)

    def test_the_universe_tab_button_forces(self):
        """The Build button is an operator looking straight at the result."""
        source = (ROOT_DIR / "scripts" / "ui" / "panels" / "universe_panel.py").read_text(encoding="utf-8")
        self.assertIn("force=True", source)
        self.assertIn("build_universe(options_filter=options_filter, force=True)", source)


if __name__ == "__main__":
    unittest.main()
