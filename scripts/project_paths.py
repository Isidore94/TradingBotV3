from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"
PERSISTENT_DATA_DIR = Path(
    os.environ.get("TRADINGBOTV3_DATA_DIR")
    or Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "TradingBotV3"
)

CACHE_DIR = DATA_DIR / "cache"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"
REPORTS_DIR = OUTPUT_DIR / "reports"
PERSISTENT_RUNTIME_DATA_DIR = PERSISTENT_DATA_DIR / "runtime"

LONGS_FILE = ROOT_DIR / "longs.txt"
SHORTS_FILE = ROOT_DIR / "shorts.txt"

SECTOR_ETF_MAP_FILE = DATA_DIR / "sector_etf_map.json"
INDUSTRY_ETF_MAP_FILE = DATA_DIR / "industry_etf_map.json"
SYMBOL_CLASSIFICATION_CACHE_FILE = DATA_DIR / "symbol_classification.csv"
EARNINGS_ANCHORS_FILE = DATA_DIR / "earnings_avwap_anchors.csv"
EARNINGS_ANCHOR_CANDIDATES_FILE = RUNTIME_DATA_DIR / "earnings_anchor_candidates.csv"

EARNINGS_CACHE_FILE = CACHE_DIR / "earnings_cache.json"
PREV_EARNINGS_CACHE_FILE = CACHE_DIR / "prev_earnings_cache.json"
EARNINGS_DATES_CACHE_FILE = CACHE_DIR / "earnings_dates_cache.json"

AVWAP_SIGNALS_FILE = RUNTIME_DATA_DIR / "avwap_signals.csv"
D1_FEATURES_FILE = RUNTIME_DATA_DIR / "d1_features.csv"
INTRADAY_BOUNCES_FILE = RUNTIME_DATA_DIR / "intraday_bounces.csv"
MASTER_AVWAP_AI_STATE_FILE = RUNTIME_DATA_DIR / "master_avwap_ai_state.json"
MASTER_AVWAP_HISTORY_FILE = RUNTIME_DATA_DIR / "master_avwap_history.json"
MASTER_POSITIONS_FILE = RUNTIME_DATA_DIR / "master_positions.json"
PREVIOUS_GAP_UPS_FILE = RUNTIME_DATA_DIR / "previous_gap_ups.csv"
ANCHOR_AVWAP_SIGNALS_FILE = RUNTIME_DATA_DIR / "master_anchor_avwap_signals.csv"
MASTER_AVWAP_FOCUS_FILE = RUNTIME_DATA_DIR / "master_avwap_focus.json"
MASTER_AVWAP_SETUP_TRACKER_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_tracker.json"
MASTER_AVWAP_SETUP_SCENARIOS_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_scenarios.csv"
MASTER_AVWAP_SETUP_DAILY_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_daily.csv"
MASTER_AVWAP_SETUP_STATS_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_stats.csv"

MASTER_AVWAP_REPORT_FILE = REPORTS_DIR / "master_avwap_events.txt"
MASTER_AVWAP_EVENT_TICKERS_FILE = REPORTS_DIR / "master_avwap_event_tickers.txt"
MASTER_AVWAP_PRIORITY_SETUPS_FILE = REPORTS_DIR / "master_avwap_priority_setups.txt"
MASTER_AVWAP_STDEV_REPORT_FILE = REPORTS_DIR / "master_avwap_stdev2_3.txt"
MASTER_ANCHOR_AVWAP_REPORT_FILE = REPORTS_DIR / "master_anchor_avwap_events.txt"
MASTER_AVWAP_TRADINGVIEW_REPORT_FILE = REPORTS_DIR / "master_avwap_tradingview.txt"
EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE = REPORTS_DIR / "earnings_anchor_candidates.txt"

BOUNCE_LOG_FILE = LOG_DIR / "bouncers.txt"
TRADING_BOT_LOG_FILE = LOG_DIR / "trading_bot.log"
MASTER_AVWAP_LOG_FILE = LOG_DIR / "master_avwap.log"
RRS_STRENGTH_LOG_FILE = LOG_DIR / "rrs_strength_extremes.csv"
RRS_GROUP_STRENGTH_LOG_FILE = LOG_DIR / "rrs_group_strength_extremes.csv"


def _ensure_directories() -> None:
    for path in (
        DATA_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        CACHE_DIR,
        RUNTIME_DATA_DIR,
        REPORTS_DIR,
        PERSISTENT_DATA_DIR,
        PERSISTENT_RUNTIME_DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_file(legacy_path: Path, new_path: Path) -> None:
    if not legacy_path.exists() or new_path.exists():
        return
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_path), str(new_path))


def migrate_legacy_layout() -> None:
    _ensure_directories()

    legacy_moves = [
        (DATA_DIR / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (DATA_DIR / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (DATA_DIR / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (DATA_DIR / "avwap_signals.csv", AVWAP_SIGNALS_FILE),
        (DATA_DIR / "d1_features.csv", D1_FEATURES_FILE),
        (DATA_DIR / "intraday_bounces.csv", INTRADAY_BOUNCES_FILE),
        (DATA_DIR / "master_avwap_ai_state.json", MASTER_AVWAP_AI_STATE_FILE),
        (DATA_DIR / "master_avwap_history.json", MASTER_AVWAP_HISTORY_FILE),
        (OUTPUT_DIR / "master_positions.json", MASTER_POSITIONS_FILE),
        (OUTPUT_DIR / "previous_gap_ups.csv", PREVIOUS_GAP_UPS_FILE),
        (OUTPUT_DIR / "master_anchor_avwap_signals.csv", ANCHOR_AVWAP_SIGNALS_FILE),
        (OUTPUT_DIR / "master_avwap_events.txt", MASTER_AVWAP_REPORT_FILE),
        (OUTPUT_DIR / "master_avwap_event_tickers.txt", MASTER_AVWAP_EVENT_TICKERS_FILE),
        (OUTPUT_DIR / "master_avwap_priority_setups.txt", MASTER_AVWAP_PRIORITY_SETUPS_FILE),
        (OUTPUT_DIR / "master_avwap_stdev2_3.txt", MASTER_AVWAP_STDEV_REPORT_FILE),
        (OUTPUT_DIR / "master_anchor_avwap_events.txt", MASTER_ANCHOR_AVWAP_REPORT_FILE),
    ]
    for legacy_path, new_path in legacy_moves:
        _migrate_legacy_file(legacy_path, new_path)


migrate_legacy_layout()
