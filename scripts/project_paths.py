from __future__ import annotations

import os
import shutil
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = ROOT_DIR / "data"
REPO_OUTPUT_DIR = ROOT_DIR / "output"
REPO_LOG_DIR = ROOT_DIR / "logs"


def _default_local_settings_dir() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "TradingBotV3"

    if sys.platform == "darwin":
        preferred = Path.home() / "Library" / "Application Support" / "TradingBotV3"
    else:
        preferred = Path.home() / ".local" / "share" / "TradingBotV3"

    legacy = Path.home() / "AppData" / "Local" / "TradingBotV3"
    if legacy.exists() and not preferred.exists():
        return legacy
    return preferred


LOCAL_SETTINGS_DIR = _default_local_settings_dir()
LOCAL_SETTINGS_FILE = LOCAL_SETTINGS_DIR / "local_settings.json"


def _load_local_settings() -> dict:
    if not LOCAL_SETTINGS_FILE.exists():
        return {}
    try:
        payload = json.loads(LOCAL_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_persistent_data_dir() -> tuple[Path, str]:
    env_value = os.environ.get("TRADINGBOTV3_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser(), "environment"

    settings = _load_local_settings()
    config_value = settings.get("shared_data_dir")
    if isinstance(config_value, str) and config_value.strip():
        return Path(config_value).expanduser(), "local_config"

    return LOCAL_SETTINGS_DIR, "default_local"


PERSISTENT_DATA_DIR, PERSISTENT_DATA_DIR_SOURCE = _resolve_persistent_data_dir()

SHARED_HOME_DIR = PERSISTENT_DATA_DIR
DATA_DIR = PERSISTENT_DATA_DIR / "data"
OUTPUT_DIR = PERSISTENT_DATA_DIR / "output"
LOG_DIR = PERSISTENT_DATA_DIR / "logs"

LOCAL_MACHINE_CACHE_DIR = LOCAL_SETTINGS_DIR / "machine_cache"
CACHE_DIR = LOCAL_MACHINE_CACHE_DIR
RUNTIME_DATA_DIR = DATA_DIR / "runtime"
REPORTS_DIR = OUTPUT_DIR / "reports"
PERSISTENT_RUNTIME_DATA_DIR = RUNTIME_DATA_DIR

LONGS_FILE = PERSISTENT_DATA_DIR / "longs.txt"
SHORTS_FILE = PERSISTENT_DATA_DIR / "shorts.txt"

SECTOR_ETF_MAP_FILE = DATA_DIR / "sector_etf_map.json"
INDUSTRY_ETF_MAP_FILE = DATA_DIR / "industry_etf_map.json"
SYMBOL_CLASSIFICATION_CACHE_FILE = DATA_DIR / "symbol_classification.csv"
EARNINGS_ANCHORS_FILE = DATA_DIR / "earnings_avwap_anchors.csv"
EARNINGS_ANCHOR_CANDIDATES_FILE = RUNTIME_DATA_DIR / "earnings_anchor_candidates.csv"

EARNINGS_CACHE_FILE = CACHE_DIR / "earnings_cache.json"
PREV_EARNINGS_CACHE_FILE = CACHE_DIR / "prev_earnings_cache.json"
EARNINGS_DATES_CACHE_FILE = CACHE_DIR / "earnings_dates_cache.json"
EARNINGS_CALENDAR_CACHE_FILE = CACHE_DIR / "earnings_calendar_rows.json"
YAHOO_SYMBOL_META_CACHE_FILE = CACHE_DIR / "yahoo_symbol_metadata.json"
DAILY_BARS_CACHE_DIR = CACHE_DIR / "daily_bars"

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
MASTER_AVWAP_SETUP_ATTRIBUTES_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_attributes.csv"
MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_setup_attribute_leaderboard.csv"
MASTER_AVWAP_SCORING_CONFIG_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scoring_config.json"
MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scoring_recommendations.json"
MASTER_AVWAP_SCORING_TUNER_REPORT_FILE = PERSISTENT_RUNTIME_DATA_DIR / "master_avwap_scoring_tuner_report.txt"

MASTER_AVWAP_REPORT_FILE = REPORTS_DIR / "master_avwap_events.txt"
MASTER_AVWAP_EVENT_TICKERS_FILE = REPORTS_DIR / "master_avwap_event_tickers.txt"
MASTER_AVWAP_PRIORITY_SETUPS_FILE = REPORTS_DIR / "master_avwap_priority_setups.txt"
MASTER_AVWAP_STDEV_REPORT_FILE = REPORTS_DIR / "master_avwap_stdev2_3.txt"
MASTER_ANCHOR_AVWAP_REPORT_FILE = REPORTS_DIR / "master_anchor_avwap_events.txt"
MASTER_AVWAP_TRADINGVIEW_REPORT_FILE = REPORTS_DIR / "master_avwap_tradingview.txt"
EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE = REPORTS_DIR / "earnings_anchor_candidates.txt"

BOUNCE_LOG_FILE = LOG_DIR / "bouncers.txt"
APP_LOG_FILE = LOG_DIR / "trading_bot.log"
APP_LOG_BACKUP_COUNT = 1
TRADING_BOT_LOG_FILE = APP_LOG_FILE
MASTER_AVWAP_LOG_FILE = APP_LOG_FILE
RRS_STRENGTH_LOG_FILE = LOG_DIR / "rrs_strength_extremes.csv"
RRS_GROUP_STRENGTH_LOG_FILE = LOG_DIR / "rrs_group_strength_extremes.csv"
RRS_ENVIRONMENT_FOCUS_HISTORY_FILE = RUNTIME_DATA_DIR / "rrs_environment_focus_history.json"


def get_tracker_storage_details() -> dict[str, str]:
    source_labels = {
        "environment": "Environment variable",
        "local_config": "Saved local setting",
        "default_local": "Default local storage",
    }
    return {
        "data_dir": str(PERSISTENT_DATA_DIR),
        "shared_root_dir": str(SHARED_HOME_DIR),
        "mutable_data_dir": str(DATA_DIR),
        "logs_dir": str(LOG_DIR),
        "output_dir": str(OUTPUT_DIR),
        "runtime_dir": str(PERSISTENT_RUNTIME_DATA_DIR),
        "local_cache_dir": str(LOCAL_MACHINE_CACHE_DIR),
        "source": PERSISTENT_DATA_DIR_SOURCE,
        "source_label": source_labels.get(PERSISTENT_DATA_DIR_SOURCE, PERSISTENT_DATA_DIR_SOURCE),
        "settings_file": str(LOCAL_SETTINGS_FILE),
    }


def get_shared_watchlist_paths() -> tuple[Path, Path]:
    return (LONGS_FILE, SHORTS_FILE)


def get_shared_watchlist_details() -> dict[str, str]:
    longs_path, shorts_path = get_shared_watchlist_paths()
    return {
        "longs_path": str(longs_path),
        "shorts_path": str(shorts_path),
        "longs_exists": "yes" if longs_path.exists() else "no",
        "shorts_exists": "yes" if shorts_path.exists() else "no",
    }


def save_tracker_storage_dir(path: str) -> Path:
    target = Path(path).expanduser()
    LOCAL_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _load_local_settings()
    payload["shared_data_dir"] = str(target)
    LOCAL_SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def clear_tracker_storage_dir() -> None:
    if not LOCAL_SETTINGS_FILE.exists():
        return
    payload = _load_local_settings()
    payload.pop("shared_data_dir", None)
    if payload:
        LOCAL_SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    LOCAL_SETTINGS_FILE.unlink(missing_ok=True)


def get_local_setting(key: str, default=None):
    payload = _load_local_settings()
    return payload.get(key, default)


def save_local_setting(key: str, value) -> None:
    LOCAL_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _load_local_settings()
    payload[key] = value
    LOCAL_SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def open_path_in_file_manager(path: Path) -> None:
    target = Path(path).expanduser()
    if sys.platform == "win32":
        os.startfile(str(target))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def _ensure_directories() -> None:
    for path in (
        CACHE_DIR,
        RUNTIME_DATA_DIR,
        REPORTS_DIR,
        PERSISTENT_DATA_DIR,
        PERSISTENT_RUNTIME_DATA_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_file(legacy_path: Path, new_path: Path) -> None:
    if not legacy_path.exists() or new_path.exists():
        return
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(new_path))
    except OSError:
        # Cloud-synced folders can briefly lock files; don't block app startup on legacy migration.
        return


def _append_legacy_text_file(source_path: Path, destination_path: Path) -> None:
    if not source_path.exists() or source_path.is_dir():
        return
    try:
        if source_path.resolve() == destination_path.resolve():
            return
    except Exception:
        pass

    try:
        content = source_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""

    if content.strip():
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            had_content = destination_path.exists() and destination_path.stat().st_size > 0
            with destination_path.open("a", encoding="utf-8") as destination_file:
                if had_content:
                    destination_file.write("\n")
                destination_file.write(
                    f"=== migrated from {source_path} at {datetime.now().isoformat(timespec='seconds')} ===\n"
                )
                destination_file.write(content.rstrip())
                destination_file.write("\n")
        except OSError:
            return

    try:
        source_path.unlink(missing_ok=True)
    except OSError:
        return


def _consolidate_log_variants(destination_path: Path, base_name: str, search_dir: Path, keep_backups: int) -> None:
    direct_path = search_dir / base_name
    _append_legacy_text_file(direct_path, destination_path)

    for rotated_path in sorted(search_dir.glob(f"{base_name}.*")):
        suffix = rotated_path.name[len(base_name) + 1 :]
        if not suffix.isdigit():
            continue
        if search_dir == destination_path.parent and base_name == destination_path.name:
            if int(suffix) <= max(0, int(keep_backups)):
                continue
        _append_legacy_text_file(rotated_path, destination_path)


def _consolidate_legacy_logs() -> None:
    legacy_log_dir = LOCAL_SETTINGS_DIR / "logs"

    for source_path in (
        ROOT_DIR / "trading_bot.log",
        ROOT_DIR / "scripts" / "trading_bot.log",
        ROOT_DIR / "bouncers.txt",
    ):
        destination = BOUNCE_LOG_FILE if source_path.name == "bouncers.txt" else APP_LOG_FILE
        _append_legacy_text_file(source_path, destination)

    for search_dir in (REPO_LOG_DIR, legacy_log_dir):
        if search_dir == LOG_DIR:
            continue
        _consolidate_log_variants(APP_LOG_FILE, "trading_bot.log", search_dir, keep_backups=0)
        _consolidate_log_variants(APP_LOG_FILE, "master_avwap.log", search_dir, keep_backups=0)
        _consolidate_log_variants(BOUNCE_LOG_FILE, "bouncers.txt", search_dir, keep_backups=0)

    _consolidate_log_variants(APP_LOG_FILE, "master_avwap.log", LOG_DIR, keep_backups=0)
    _consolidate_log_variants(APP_LOG_FILE, APP_LOG_FILE.name, LOG_DIR, keep_backups=APP_LOG_BACKUP_COUNT)


def migrate_legacy_layout() -> None:
    _ensure_directories()

    legacy_moves = [
        (ROOT_DIR / "longs.txt", LONGS_FILE),
        (ROOT_DIR / "shorts.txt", SHORTS_FILE),
        (REPO_DATA_DIR / "sector_etf_map.json", SECTOR_ETF_MAP_FILE),
        (REPO_DATA_DIR / "industry_etf_map.json", INDUSTRY_ETF_MAP_FILE),
        (REPO_DATA_DIR / "symbol_classification.csv", SYMBOL_CLASSIFICATION_CACHE_FILE),
        (REPO_DATA_DIR / "earnings_avwap_anchors.csv", EARNINGS_ANCHORS_FILE),
        (REPO_DATA_DIR / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (REPO_DATA_DIR / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (REPO_DATA_DIR / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (REPO_DATA_DIR / "avwap_signals.csv", AVWAP_SIGNALS_FILE),
        (REPO_DATA_DIR / "d1_features.csv", D1_FEATURES_FILE),
        (REPO_DATA_DIR / "intraday_bounces.csv", INTRADAY_BOUNCES_FILE),
        (REPO_DATA_DIR / "master_avwap_ai_state.json", MASTER_AVWAP_AI_STATE_FILE),
        (REPO_DATA_DIR / "master_avwap_history.json", MASTER_AVWAP_HISTORY_FILE),
        (REPO_DATA_DIR / "runtime" / "earnings_anchor_candidates.csv", EARNINGS_ANCHOR_CANDIDATES_FILE),
        (REPO_DATA_DIR / "runtime" / "avwap_signals.csv", AVWAP_SIGNALS_FILE),
        (REPO_DATA_DIR / "runtime" / "d1_features.csv", D1_FEATURES_FILE),
        (REPO_DATA_DIR / "runtime" / "intraday_bounces.csv", INTRADAY_BOUNCES_FILE),
        (REPO_DATA_DIR / "runtime" / "rrs_environment_focus_history.json", RRS_ENVIRONMENT_FOCUS_HISTORY_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_ai_state.json", MASTER_AVWAP_AI_STATE_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_history.json", MASTER_AVWAP_HISTORY_FILE),
        (REPO_DATA_DIR / "runtime" / "master_positions.json", MASTER_POSITIONS_FILE),
        (REPO_DATA_DIR / "runtime" / "previous_gap_ups.csv", PREVIOUS_GAP_UPS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_anchor_avwap_signals.csv", ANCHOR_AVWAP_SIGNALS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_focus.json", MASTER_AVWAP_FOCUS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_tracker.json", MASTER_AVWAP_SETUP_TRACKER_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_scenarios.csv", MASTER_AVWAP_SETUP_SCENARIOS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_daily.csv", MASTER_AVWAP_SETUP_DAILY_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_stats.csv", MASTER_AVWAP_SETUP_STATS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_attributes.csv", MASTER_AVWAP_SETUP_ATTRIBUTES_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_setup_attribute_leaderboard.csv", MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scoring_config.json", MASTER_AVWAP_SCORING_CONFIG_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scoring_recommendations.json", MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE),
        (REPO_DATA_DIR / "runtime" / "master_avwap_scoring_tuner_report.txt", MASTER_AVWAP_SCORING_TUNER_REPORT_FILE),
        (REPO_DATA_DIR / "cache" / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (REPO_DATA_DIR / "cache" / "daily_bars", DAILY_BARS_CACHE_DIR),
        (REPO_OUTPUT_DIR / "master_positions.json", MASTER_POSITIONS_FILE),
        (REPO_OUTPUT_DIR / "previous_gap_ups.csv", PREVIOUS_GAP_UPS_FILE),
        (REPO_OUTPUT_DIR / "master_anchor_avwap_signals.csv", ANCHOR_AVWAP_SIGNALS_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_events.txt", MASTER_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_event_tickers.txt", MASTER_AVWAP_EVENT_TICKERS_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_priority_setups.txt", MASTER_AVWAP_PRIORITY_SETUPS_FILE),
        (REPO_OUTPUT_DIR / "master_avwap_stdev2_3.txt", MASTER_AVWAP_STDEV_REPORT_FILE),
        (REPO_OUTPUT_DIR / "master_anchor_avwap_events.txt", MASTER_ANCHOR_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_events.txt", MASTER_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_event_tickers.txt", MASTER_AVWAP_EVENT_TICKERS_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_priority_setups.txt", MASTER_AVWAP_PRIORITY_SETUPS_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_stdev2_3.txt", MASTER_AVWAP_STDEV_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_anchor_avwap_events.txt", MASTER_ANCHOR_AVWAP_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "master_avwap_tradingview.txt", MASTER_AVWAP_TRADINGVIEW_REPORT_FILE),
        (REPO_OUTPUT_DIR / "reports" / "earnings_anchor_candidates.txt", EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE),
        (REPO_LOG_DIR / "bouncers.txt", BOUNCE_LOG_FILE),
        (REPO_LOG_DIR / "trading_bot.log", TRADING_BOT_LOG_FILE),
        (REPO_LOG_DIR / "master_avwap.log", MASTER_AVWAP_LOG_FILE),
        (REPO_LOG_DIR / "rrs_strength_extremes.csv", RRS_STRENGTH_LOG_FILE),
        (REPO_LOG_DIR / "rrs_group_strength_extremes.csv", RRS_GROUP_STRENGTH_LOG_FILE),
        (DATA_DIR / "cache" / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (DATA_DIR / "cache" / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (DATA_DIR / "cache" / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (DATA_DIR / "cache" / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (DATA_DIR / "cache" / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (DATA_DIR / "cache" / "daily_bars", DAILY_BARS_CACHE_DIR),
        (LOCAL_SETTINGS_DIR / "data" / "sector_etf_map.json", SECTOR_ETF_MAP_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "industry_etf_map.json", INDUSTRY_ETF_MAP_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "symbol_classification.csv", SYMBOL_CLASSIFICATION_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "earnings_avwap_anchors.csv", EARNINGS_ANCHORS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "earnings_cache.json", EARNINGS_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "prev_earnings_cache.json", PREV_EARNINGS_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "earnings_dates_cache.json", EARNINGS_DATES_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "earnings_calendar_rows.json", EARNINGS_CALENDAR_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "yahoo_symbol_metadata.json", YAHOO_SYMBOL_META_CACHE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "cache" / "daily_bars", DAILY_BARS_CACHE_DIR),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "earnings_anchor_candidates.csv", EARNINGS_ANCHOR_CANDIDATES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "avwap_signals.csv", AVWAP_SIGNALS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "d1_features.csv", D1_FEATURES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "intraday_bounces.csv", INTRADAY_BOUNCES_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "rrs_environment_focus_history.json", RRS_ENVIRONMENT_FOCUS_HISTORY_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_ai_state.json", MASTER_AVWAP_AI_STATE_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_history.json", MASTER_AVWAP_HISTORY_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_positions.json", MASTER_POSITIONS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "previous_gap_ups.csv", PREVIOUS_GAP_UPS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_anchor_avwap_signals.csv", ANCHOR_AVWAP_SIGNALS_FILE),
        (LOCAL_SETTINGS_DIR / "data" / "runtime" / "master_avwap_focus.json", MASTER_AVWAP_FOCUS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_tracker.json", MASTER_AVWAP_SETUP_TRACKER_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_scenarios.csv", MASTER_AVWAP_SETUP_SCENARIOS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_daily.csv", MASTER_AVWAP_SETUP_DAILY_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_stats.csv", MASTER_AVWAP_SETUP_STATS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_attributes.csv", MASTER_AVWAP_SETUP_ATTRIBUTES_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_setup_attribute_leaderboard.csv", MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scoring_config.json", MASTER_AVWAP_SCORING_CONFIG_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scoring_recommendations.json", MASTER_AVWAP_SCORING_RECOMMENDATIONS_FILE),
        (LOCAL_SETTINGS_DIR / "runtime" / "master_avwap_scoring_tuner_report.txt", MASTER_AVWAP_SCORING_TUNER_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_events.txt", MASTER_AVWAP_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_event_tickers.txt", MASTER_AVWAP_EVENT_TICKERS_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_priority_setups.txt", MASTER_AVWAP_PRIORITY_SETUPS_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_stdev2_3.txt", MASTER_AVWAP_STDEV_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_anchor_avwap_events.txt", MASTER_ANCHOR_AVWAP_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "master_avwap_tradingview.txt", MASTER_AVWAP_TRADINGVIEW_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "output" / "reports" / "earnings_anchor_candidates.txt", EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "bouncers.txt", BOUNCE_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "trading_bot.log", TRADING_BOT_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "master_avwap.log", MASTER_AVWAP_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "rrs_strength_extremes.csv", RRS_STRENGTH_LOG_FILE),
        (LOCAL_SETTINGS_DIR / "logs" / "rrs_group_strength_extremes.csv", RRS_GROUP_STRENGTH_LOG_FILE),
    ]
    for legacy_path, new_path in legacy_moves:
        _migrate_legacy_file(legacy_path, new_path)
    _consolidate_legacy_logs()


migrate_legacy_layout()
