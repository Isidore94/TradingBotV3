"""Tests for the 2nd-stdev breakout study family + the recent ("last 30 days")
setup-type stats that surface it.

2nd-dev breakouts (fresh cross up through UPPER_2 / down through LOWER_2) are
recorded in the isolated study namespace (no scoring impact) so realized R
accumulates and the recent-window stats can show whether they work before any
promotion to a scored family.
"""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _daily(d, o, h, l, c):
    return {"date": d, "open": o, "high": h, "low": l, "close": c, "volume": 1_000_000}


def _long_rows(*, fresh: bool, band: float = 100.0):
    dates = [f"2026-06-2{i}" for i in range(2, 7)]  # 22..26 priors
    if fresh:
        # clean lookback: highs well below the 2nd band, no touches
        priors = [_daily(d, 98.0, 98.5, 97.0, 98.0) for d in dates]
        today = _daily("2026-06-29", 99.0, 102.0, 98.8, 101.0)  # first clean break above band
    else:
        # chop-at-band: priors keep tagging the 2nd band -> not a fresh break
        priors = [_daily(d, 99.5, 100.8, 98.5, 100.2) for d in dates]
        today = _daily("2026-06-29", 100.0, 102.0, 99.0, 101.0)
    return priors + [today]


def _short_rows(*, band: float = 100.0):
    dates = [f"2026-06-2{i}" for i in range(2, 7)]
    priors = [_daily(d, 102.0, 103.0, 101.5, 102.0) for d in dates]  # clean, above band
    today = _daily("2026-06-29", 101.0, 101.2, 98.0, 99.0)  # first clean break below band
    return priors + [today]


def _ai_state(symbol, side, rows, band):
    bands = {"UPPER_2": band, "LOWER_2": band - 20} if side == "LONG" else {"LOWER_2": band, "UPPER_2": band + 20}
    return {
        "symbols": {
            symbol: {
                "side": side,
                "current_anchor": {"bands": bands},
                "daily_ohlc": rows,
                "atr20": 2.0,
                "last_trade_date": rows[-1]["date"],
            }
        }
    }


def _priority_rows(symbol, side, rows):
    return [{"symbol": symbol, "side": side, "last_trade_date": rows[-1]["date"], "atr20": 2.0}]


class SecondDevBreakoutDetectionTests(unittest.TestCase):
    def test_fresh_long_breakout_emits_study_row(self):
        rows = _long_rows(fresh=True)
        study = m.enrich_priority_rows_with_second_dev_breakouts(
            _priority_rows("NVDA", "LONG", rows),
            {},
            ai_state=_ai_state("NVDA", "LONG", rows, 100.0),
        )
        self.assertEqual(len(study), 1)
        self.assertEqual(study[0]["setup_family"], m.SECOND_DEV_BREAKOUT_STUDY_FAMILY)
        self.assertEqual(study[0]["priority_bucket"], m.SECOND_DEV_BREAKOUT_STUDY_BUCKET)
        self.assertEqual(study[0]["side"], "LONG")
        self.assertFalse(study[0]["is_favorite_setup"])
        self.assertIn("2NDDEV_BREAKOUT", study[0]["setup_tags"])

    def test_fresh_short_breakout_emits_study_row(self):
        rows = _short_rows()
        study = m.enrich_priority_rows_with_second_dev_breakouts(
            _priority_rows("LULU", "SHORT", rows),
            {},
            ai_state=_ai_state("LULU", "SHORT", rows, 100.0),
        )
        self.assertEqual(len(study), 1)
        self.assertEqual(study[0]["side"], "SHORT")
        self.assertEqual(study[0]["setup_family"], m.SECOND_DEV_BREAKOUT_STUDY_FAMILY)

    def test_chop_at_band_does_not_emit(self):
        rows = _long_rows(fresh=False)
        study = m.enrich_priority_rows_with_second_dev_breakouts(
            _priority_rows("CHOP", "LONG", rows),
            {},
            ai_state=_ai_state("CHOP", "LONG", rows, 100.0),
        )
        self.assertEqual(study, [])

    def test_missing_band_or_data_is_skipped(self):
        # No current_anchor bands -> nothing to detect, no crash.
        state = {"symbols": {"X": {"side": "LONG", "daily_ohlc": _long_rows(fresh=True), "atr20": 2.0,
                                    "last_trade_date": "2026-06-29"}}}
        study = m.enrich_priority_rows_with_second_dev_breakouts(
            [{"symbol": "X", "side": "LONG", "last_trade_date": "2026-06-29"}], {}, ai_state=state
        )
        self.assertEqual(study, [])


def _closed_2nddev(symbol, closed_r, scan, side="LONG"):
    return {
        "symbol": symbol,
        "side": side,
        "anchor_date": scan,
        "scan_date": scan,
        "setup_family": m.SECOND_DEV_BREAKOUT_STUDY_FAMILY,
        "study_kind": m.SECOND_DEV_BREAKOUT_STUDY_FAMILY,
        "priority_bucket": m.SECOND_DEV_BREAKOUT_STUDY_BUCKET,
        "scenarios": {
            "s1": {
                "tradeable": True,
                "status": "TARGET_HIT" if closed_r > 0 else "STOPPED",
                "total_r": closed_r,
                "stop_reference_label": "LOWER_1" if side == "LONG" else "UPPER_1",
            }
        },
    }


class RecentSetupTypeStatsTests(unittest.TestCase):
    def test_recent_family_rows_include_2nddev_breakout(self):
        ref = date(2026, 6, 29)
        study = {
            f"study:w{i}": _closed_2nddev(f"W{i}", 1.2, scan=f"2026-06-2{i}")
            for i in range(2, 7)  # scan dates 22..26, ages 3..7 days
        }
        rows = m.build_recent_tracker_setup_family_rows(study, lookback_days=30, reference_date=ref)
        fam = [r for r in rows if r["setup_family"] == m.SECOND_DEV_BREAKOUT_STUDY_FAMILY]
        self.assertTrue(fam)
        self.assertGreater(fam[0]["closed_setups"], 0)
        self.assertGreater(fam[0]["avg_closed_r"], 0)

    def test_build_recent_setup_type_stat_rows_tags_namespace(self):
        base = date.today()
        study = {
            f"study:s{i}": _closed_2nddev(f"S{i}", 0.8, scan=(base - timedelta(days=i + 2)).isoformat())
            for i in range(5)
        }
        payload = {"setups": {}, "study_setups": study}
        rows = m.build_recent_setup_type_stat_rows(payload)
        breakout = [r for r in rows if r["setup_family"] == m.SECOND_DEV_BREAKOUT_STUDY_FAMILY]
        self.assertTrue(breakout)
        self.assertEqual(breakout[0]["namespace"], "study")
        self.assertEqual(breakout[0]["lookback_days"], m.RECENT_SETUP_TYPE_LOOKBACK_DAYS)


if __name__ == "__main__":
    unittest.main()
