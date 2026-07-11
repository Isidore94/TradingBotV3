"""Move forensics: episode finding, no-lookahead snapshots, pattern mining."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import move_forensics as forensics  # noqa: E402


def _frame(closes, *, start="2026-01-02", spread=0.5, volume=1_000_000.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    rows = []
    prev = closes[0]
    for dt, close in zip(dates, closes):
        rows.append(
            {
                "datetime": dt,
                "open": prev,
                "high": max(prev, close) + spread,
                "low": min(prev, close) - spread,
                "close": close,
                "volume": volume,
            }
        )
        prev = close
    return pd.DataFrame(rows)


def _arrays(closes, spread=0.5):
    frame = _frame(closes, spread=spread)
    return (
        frame["high"].to_numpy(float),
        frame["low"].to_numpy(float),
        frame["close"].to_numpy(float),
        np.full(len(frame), 1.0),  # ATR
    )


class TestFindMoveEpisodes:
    def test_clean_move_found_once_no_overlap(self):
        closes = [100.0] * 50 + [100.0 + 3.0 * k for k in range(1, 11)] + [130.0] * 10
        highs, lows, closes_arr, atr = _arrays(closes)
        episodes = forensics.find_move_episodes(highs, lows, closes_arr, atr, 30)
        assert len(episodes) == 1, "one continuing move must be one episode"
        episode = episodes[0]
        # Peak extension captures the WHOLE run (~30 ATR), not just the
        # in-horizon slice that qualified it.
        assert episode["move_atr"] >= 20
        assert episode["adverse_atr"] <= forensics.MOVE_MAX_ADVERSE_ATR
        assert episode["days_to_peak"] >= forensics.MOVE_HORIZON_SESSIONS
        # the climb starts at bar 50; the signal cannot postdate the liftoff
        assert episode["signal_idx"] <= 50

    def test_choppy_move_rejected_until_after_the_drawdown(self):
        # Drops 5 ATR first, then rallies 8 ATR: days before the bottom see
        # too much adverse excursion, so the episode starts at/after the low.
        closes = (
            [100.0] * 40
            + [100.0 - 1.0 * k for k in range(1, 6)]  # bars 40-44 fall to 95
            + [95.0 + 2.0 * k for k in range(1, 9)]  # bars 45-52 rally to 111
            + [111.0] * 12
        )
        highs, lows, closes_arr, atr = _arrays(closes, spread=0.1)
        episodes = forensics.find_move_episodes(highs, lows, closes_arr, atr, 30)
        assert episodes, "the rally leg itself is a clean move"
        assert all(e["signal_idx"] >= 43 for e in episodes)

    def test_no_full_forward_window_no_episode(self):
        closes = [100.0] * 35 + [100.0 + 3.0 * k for k in range(1, 6)]
        highs, lows, closes_arr, atr = _arrays(closes)
        episodes = forensics.find_move_episodes(highs, lows, closes_arr, atr, 30)
        assert episodes == []  # move exists but the horizon extends past the data


class TestScanSymbol:
    def test_long_and_short_moves_detected_with_flags(self):
        up = [100.0] * 60 + [100.0 + 3.0 * k for k in range(1, 11)] + [130.0] * 12
        frame = _frame(up)
        movers, baseline = forensics.scan_symbol("UPPY", frame, [], days=60)
        long_moves = [row for row in movers if row["side"] == "LONG"]
        assert long_moves
        row = long_moves[0]
        assert row["move_atr"] >= 3.0 and row["days_to_peak"] >= 1
        for flag in forensics.condition_flag_names():
            assert flag in row
        assert baseline, "ordinary days should be sampled"
        assert all(b["side"] in {"LONG", "SHORT"} for b in baseline)

        down = [100.0] * 60 + [100.0 - 3.0 * k for k in range(1, 11)] + [70.0] * 12
        movers_down, _ = forensics.scan_symbol("DOWNY", _frame(down), [], days=60)
        assert any(row["side"] == "SHORT" for row in movers_down)

    def test_baseline_days_do_not_overlap_episodes(self):
        closes = [100.0] * 60 + [100.0 + 3.0 * k for k in range(1, 11)] + [130.0] * 12
        movers, baseline = forensics.scan_symbol("UPPY", _frame(closes), [], days=60)
        episode_spans = {
            (row["side"], date)
            for row in movers
            for date in (row["signal_date"], row["peak_date"])
        }
        for row in baseline:
            assert (row["side"], row["signal_date"]) not in episode_spans

    def test_condition_flags_have_no_lookahead(self):
        base = [100.0] * 70
        variant_a = base + [100.0 + 2.0 * k for k in range(1, 13)]
        variant_b = base + [100.0 + 4.0 * k for k in range(1, 13)]
        earnings = ["2026-01-20"]
        frame_a, frame_b = _frame(variant_a), _frame(variant_b)
        ctx_a = forensics.build_symbol_context("SAME", frame_a, earnings, scan_start_idx=40)
        ctx_b = forensics.build_symbol_context("SAME", frame_b, earnings, scan_start_idx=40)
        streaks = forensics.compute_weekly_streak_series(frame_a)
        i = 69  # both variants identical through this bar
        flags_a = forensics.collect_condition_flags(ctx_a, i, weekly_streak=int(streaks[i]), side="LONG")
        flags_b = forensics.collect_condition_flags(ctx_b, i, weekly_streak=int(streaks[i]), side="LONG")
        assert flags_a == flags_b


class TestPatternMining:
    def _rows(self):
        movers = []
        for k in range(20):  # co-occurring pair present in most movers
            movers.append({"side": "LONG", "move_atr": 4.0, "ema15_bounce": True, "volume_2x_recent": True})
        for k in range(10):
            movers.append({"side": "LONG", "move_atr": 3.0, "second_dev_breakout": True})
        baseline = [{"side": "LONG"} for _ in range(95)]
        baseline += [{"side": "LONG", "ema15_bounce": True} for _ in range(5)]
        return movers, baseline

    def test_lift_ranks_over_represented_conditions(self):
        movers, baseline = self._rows()
        patterns = forensics.mine_patterns(movers, baseline)
        strong = [p for p in patterns if p["movers_with"] >= forensics.PATTERN_MIN_MOVER_SUPPORT]
        assert strong[0]["lift"] > 3.0
        names = [p["pattern"] for p in strong]
        assert "volume_2x_recent" in names and "ema15_bounce" in names
        pair = next(p for p in strong if p["kind"] == "pair")
        assert set(pair["pattern"].split(" + ")) == {"ema15_bounce", "volume_2x_recent"}
        assert pair["movers_with"] == 20

    def test_novel_flag_marks_non_playbook_patterns(self):
        movers, baseline = self._rows()
        patterns = forensics.mine_patterns(movers, baseline)
        by_name = {p["pattern"]: p for p in patterns}
        assert by_name["ema15_bounce"]["novel"] is True  # context flag, not a tracked family
        assert by_name["second_dev_breakout"]["novel"] is False  # existing playbook family

    def test_thin_support_sinks_but_stays_visible(self):
        movers = [{"side": "LONG", "move_atr": 9.0, "gap_up_hold": True} for _ in range(2)]
        movers += [{"side": "LONG", "move_atr": 3.0, "trend20_aligned": True} for _ in range(20)]
        baseline = [{"side": "LONG"} for _ in range(50)]
        patterns = forensics.mine_patterns(movers, baseline)
        assert patterns[0]["pattern"] == "trend20_aligned"
        thin = next(p for p in patterns if p["pattern"] == "gap_up_hold")
        assert patterns.index(thin) > patterns.index(patterns[0])


class TestReportAndDigest:
    def test_report_and_digest_shape(self):
        closes = [100.0] * 60 + [100.0 + 3.0 * k for k in range(1, 11)] + [130.0] * 12
        movers, baseline = forensics.scan_symbol("UPPY", _frame(closes), [], days=60)
        patterns = forensics.mine_patterns(movers, baseline)
        params = {"min_move_atr": 3.0, "horizon": 10, "max_adverse_atr": 1.0}
        report = forensics.render_report(movers, patterns, days=60, params=params)
        assert "MOVE FORENSICS" in report
        assert "LONG MOVES" in report and "SHORT MOVES" in report
        assert "NOT IN THE PLAYBOOK" in report
        digest = forensics.build_ai_digest(movers, patterns, days=60, params=params)
        assert digest["mover_counts"]["LONG"] >= 1
        assert "how_to_analyze" in digest and digest["files"]["movers_csv"]
        assert isinstance(digest["top_patterns"], list)
