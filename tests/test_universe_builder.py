"""Tests for the self-sufficient universe builder (pure parsing/screening only)."""

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
            with patch.dict(sys.modules, {"yfinance": module}), \
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
        """The audit row is best effort; it can never cost the desk its universe."""
        self._seed_previous(1487)
        with patch.object(ub, "_universe_ledger_path", lambda: Path("Q:/nope/job_ledger.jsonl")):
            result = self._build(1450)
        self.assertEqual(len(result["all"]), 1450)


class UniverseForceCarveOutWiringTests(unittest.TestCase):
    """The floor's carve-out is only real if a manual rebuild actually reaches it.

    plan.md R9.1: "a manual rebuild keeps a ``force=True`` carve-out exactly as
    the quiet-hours gate does". Both manual entry points are pinned here so the
    wiring cannot be dropped while the flag quietly survives.
    """

    def test_the_autopilot_manual_rebuild_forwards_force(self):
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
            # The manual button: force=True all the way down to the write floor.
            self.assertEqual(core.rebuild_universe_if_stale(force=True, built_at=None), "rebuilt")
            # The scheduled stale tick: never carves out.
            self.assertEqual(core.rebuild_universe_if_stale(force=False, built_at=None), "rebuilt")
        self.assertEqual(seen, [True, False])

    def test_the_universe_tab_button_forces(self):
        """The Build button is an operator looking straight at the result."""
        source = (ROOT_DIR / "scripts" / "ui" / "panels" / "universe_panel.py").read_text(encoding="utf-8")
        self.assertIn("force=True", source)
        self.assertIn("build_universe(options_filter=options_filter, force=True)", source)


if __name__ == "__main__":
    unittest.main()
