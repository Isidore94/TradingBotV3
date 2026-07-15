# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import re
from pathlib import Path
# Suppress Tk deprecation warnings on macOS
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import math
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
import zoneinfo
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter.font as tkFont

import pandas as pd
import yfinance as yf
from market_session import (
    get_market_local_now,
    get_market_session_close_naive,
    get_market_session_open_naive,
    get_market_session_window,
)

# IB API imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# For colored console output (still used for logging)
from colorama import init, Fore, Style
init(autoreset=True)

IB_INFO_STATUS_CODES = {
    2104: "Market data farm status",
    2106: "Historical data farm status",
    2107: "Historical data farm idle",
    2108: "Market data farm idle",
    2158: "Security definition farm status",
}
IB_WARNING_STATUS_CODES = {2103, 2105}
YFINANCE_NOISY_LOGGER_NAMES = (
    "yfinance",
    "yfinance.base",
    "yfinance.scrapers",
    "yfinance.scrapers.quote",
)

from master_avwap_shared import (
    build_master_avwap_active_level_map,
    build_master_avwap_d1_flag_events,
    build_master_avwap_second_stdev_cross_map,
    describe_master_avwap_focus,
    describe_master_avwap_second_stdev_cross,
    load_master_avwap_d1_upgrade_alerts as load_master_avwap_d1_upgrade_alerts_map,
    load_master_avwap_d1_watchlist as load_master_avwap_d1_watchlist_map,
    load_master_avwap_events_for_date,
    load_master_avwap_focus_map,
    normalize_master_avwap_event_row,
)
from focus_picks import load_focus_map
from project_paths import (
    DATA_DIR,
    LOG_DIR,
    ROOT_DIR,
    LONGS_FILE,
    SHORTS_FILE,
    AUTO_LONGS_FILE,
    AUTO_SHORTS_FILE,
    REPORTS_DIR,
    BOUNCE_LOG_FILE,
    SafeRotatingFileHandler,
    TRADING_BOT_LOG_FILE,
    INTRADAY_BOUNCES_FILE,
    INTRADAY_BOUNCE_CANDIDATES_FILE,
    INTRADAY_BOUNCE_OUTCOMES_FILE,
    INTRADAY_BOUNCE_OUTCOME_STATE_FILE,
    INTRADAY_BOUNCE_FEEDBACK_FILE,
    AVWAP_SIGNALS_FILE,
    RRS_STRENGTH_LOG_FILE,
    RRS_GROUP_STRENGTH_LOG_FILE,
    RRS_ENVIRONMENT_FOCUS_HISTORY_FILE,
    SECTOR_ETF_MAP_FILE,
    INDUSTRY_ETF_MAP_FILE,
    SYMBOL_CLASSIFICATION_CACHE_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE,
    MASTER_AVWAP_D1_WATCHLIST_FILE,
    REGIME_PAUSE_OBSERVATIONS_FILE,
    APP_LOG_BACKUP_COUNT,
    get_tracker_storage_details,
    get_local_setting,
    save_local_setting,
    save_tracker_storage_dir,
)

##########################################
# Adjustable Parameters
##########################################
LONGS_FILENAME = LONGS_FILE
SHORTS_FILENAME = SHORTS_FILE
# Auto Pilot's own morning picks (see autopilot_core). Scanned with the same
# treatment as the trader's lists so the bot's picks earn their own tracked
# outcome history; the trader's files always win direction conflicts.
AUTO_LONGS_FILENAME = AUTO_LONGS_FILE
AUTO_SHORTS_FILENAME = AUTO_SHORTS_FILE
AUTO_WATCHLIST_PRIORITY = True  # every-candle treatment (pacing governor protects IB)
BOUNCE_LOG_FILENAME = BOUNCE_LOG_FILE
TRADING_BOT_LOG_FILENAME = TRADING_BOT_LOG_FILE
INTRADAY_BOUNCES_CSV = INTRADAY_BOUNCES_FILE
INTRADAY_BOUNCE_CANDIDATES_CSV = INTRADAY_BOUNCE_CANDIDATES_FILE
INTRADAY_BOUNCE_OUTCOMES_CSV = INTRADAY_BOUNCE_OUTCOMES_FILE
INTRADAY_BOUNCE_OUTCOME_STATE_JSON = INTRADAY_BOUNCE_OUTCOME_STATE_FILE
INTRADAY_BOUNCE_FEEDBACK_CSV = INTRADAY_BOUNCE_FEEDBACK_FILE
INTRADAY_BOUNCE_PERFORMANCE_CSV = INTRADAY_BOUNCE_CANDIDATES_CSV.with_name("intraday_bounce_performance.csv")
INTRADAY_BOUNCE_PERFORMANCE_REPORT = REPORTS_DIR / "intraday_bounce_performance.txt"
MASTER_AVWAP_SIGNALS_FILENAME = AVWAP_SIGNALS_FILE
MASTER_AVWAP_FOCUS_FILENAME = MASTER_AVWAP_FOCUS_FILE
MASTER_AVWAP_D1_UPGRADE_ALERTS_FILENAME = MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE
MASTER_AVWAP_D1_WATCHLIST_FILENAME = MASTER_AVWAP_D1_WATCHLIST_FILE
STRENGTH_SCAN_LOG_FILENAME = RRS_STRENGTH_LOG_FILE
GROUP_STRENGTH_SCAN_LOG_FILENAME = RRS_GROUP_STRENGTH_LOG_FILE
ENVIRONMENT_FOCUS_HISTORY_FILENAME = RRS_ENVIRONMENT_FOCUS_HISTORY_FILE
ENVIRONMENT_FOCUS_HISTORY_SCHEMA_VERSION = 1
ENVIRONMENT_FOCUS_HISTORY_KEEP_DAYS = 30
NASDAQ_EARNINGS_REACTION_CACHE_FILES = (
    ROOT_DIR / "data" / "cache" / "nasdaq_earnings_calendar_cache.json",
    DATA_DIR / "cache" / "nasdaq_earnings_calendar_cache.json",
)
MANUAL_EARNINGS_REACTION_FILES = (
    ROOT_DIR / "config" / "manual_earnings_calendar.json",
    DATA_DIR / "manual_earnings_calendar.json",
)
SECTOR_ETF_MAP_FILENAME = SECTOR_ETF_MAP_FILE
INDUSTRY_ETF_MAP_FILENAME = INDUSTRY_ETF_MAP_FILE
SYMBOL_CLASSIFICATION_CACHE_FILENAME = SYMBOL_CLASSIFICATION_CACHE_FILE
ATR_PERIOD = 20
THRESHOLD_MULTIPLIER = 0.02
EMA_21_TOUCH_BUFFER_ATR = 0.002
EMA_21_TOUCH_BUFFER_MIN = 0.01
MID_EARNINGS_ABOVE_SECOND_STDEV_FAMILY = "mid_earnings_above_2nd_stdev"
TOP_PATTERN_FAMILIES = {"top_pattern", "top_pattern_tracking"}
H1_MID_EARNINGS_BOUNCE_LEVELS = {
    "h1_ema_15": "H1 15 EMA",
    "h1_sma_20": "H1 20 SMA",
}
H1_TOP_PATTERN_ENTRY_LEVELS = {"h1_ema_15"}
H1_MID_EARNINGS_MIN_BARS = 21
EMA_FRESH_TOUCH_LOOKBACK_BARS = 3
EMA_FRESH_TOUCH_BUFFER_ATR = 0.03
CONSECUTIVE_CANDLES = 6  # Number of candles price must respect level before bounce
CHECK_CONSECUTIVE_CANDLES = True  # Parameter to enable/disable this check
# Defaults rebalanced 2026-07-02 from the outcome tracker's full history
# (learning rows keep measuring disabled types, so these stay evidence-based):
# - dynamic_vwap bands turned ON: dynamic_vwap_lower_band shorts were the best
#   measured segment (+1.42R avg, n=18, MAE -0.64); upper-band longs +0.78R (n=30).
# - 10_candle turned ON: positive both sides (+0.34R long n=66, +0.47R short n=34).
# - eod_vwap_upper/lower band and vwap_lower_band stay OFF: measured -0.82R/-0.25R/
#   -0.06R. ema_21 stays OFF (-0.60R shorts). Direction/time-specific mutes are
#   handled by bounce_bot_lib.learning at alert time, not by these type toggles.
CHECK_BOUNCE_VVWAP = True
CHECK_BOUNCE_DYNAMIC_VVWAP = True
CHECK_BOUNCE_EOD_VWAP = True
CHECK_BOUNCE_VWAP_EOD_CONFLUENCE = True
CHECK_BOUNCE_IMPULSE_RETEST_VWAP_EOD = True
CHECK_BOUNCE_8_EMA = True
CHECK_BOUNCE_15_EMA = True
CHECK_BOUNCE_21_EMA = False
CHECK_BOUNCE_10_CANDLE = True
CHECK_BOUNCE_PREV_DAY_HIGH = True
CHECK_BOUNCE_PREV_DAY_LOW = True
CHECK_BOUNCE_VWAP_UPPER_BAND = False
CHECK_BOUNCE_VWAP_LOWER_BAND = False
CHECK_BOUNCE_DYNAMIC_VWAP_UPPER_BAND = True
CHECK_BOUNCE_DYNAMIC_VWAP_LOWER_BAND = True
CHECK_BOUNCE_EOD_VWAP_UPPER_BAND = False
CHECK_BOUNCE_EOD_VWAP_LOWER_BAND = False
LOGGING_MODE = True
SCAN_OUTSIDE_MARKET_HOURS = True
LOG_PRICE_APPROACHING = True
USE_GUI = True  # New parameter to toggle GUI on/off
RECLAIM_LOOKBACK_CANDLES = 3
CONFLUENCE_MAX_SPREAD_ATR = 0.25
VWAP_BOUNCE_NEAR_ATR = 0.08
VWAP_BOUNCE_PIERCE_ATR = 0.08
VWAP_BOUNCE_CLOSE_BUFFER_ATR = 0.03
IMPULSE_LOOKBACK_BARS = 12
IMPULSE_MAX_BARS = 6
IMPULSE_MIN_ATR = 0.8
IMPULSE_INTRADAY_ATR_PERIOD = 14
IMPULSE_RETRACE_MIN_FRAC = 0.05
IMPULSE_RETRACE_MAX_FRAC = 0.90
IMPULSE_RETEST_MAX_BARS = 8
IMPULSE_RETEST_TOUCH_ATR = 0.12
IMPULSE_RETEST_PIERCE_ATR = 0.40
IMPULSE_RETEST_CLOSE_BUFFER_ATR = 0.02
IMPULSE_CHOP_LOOKBACK_BARS = 8
IMPULSE_MAX_NEAR_BARS = 6
IMPULSE_LEVEL_BAND_ATR = 0.14
IMPULSE_MAX_CROSSES = 2

BOUNCE_TYPE_DEFAULTS = {
    "10_candle": CHECK_BOUNCE_10_CANDLE,
    "vwap": CHECK_BOUNCE_VVWAP,
    "dynamic_vwap": CHECK_BOUNCE_DYNAMIC_VVWAP,
    "eod_vwap": CHECK_BOUNCE_EOD_VWAP,
    "vwap_eod_confluence": CHECK_BOUNCE_VWAP_EOD_CONFLUENCE,
    "impulse_retest_vwap_eod": CHECK_BOUNCE_IMPULSE_RETEST_VWAP_EOD,
    "ema_8": CHECK_BOUNCE_8_EMA,
    "ema_15": CHECK_BOUNCE_15_EMA,
    "ema_21": CHECK_BOUNCE_21_EMA,
    "vwap_upper_band": CHECK_BOUNCE_VWAP_UPPER_BAND,
    "vwap_lower_band": CHECK_BOUNCE_VWAP_LOWER_BAND,
    "dynamic_vwap_upper_band": CHECK_BOUNCE_DYNAMIC_VWAP_UPPER_BAND,
    "dynamic_vwap_lower_band": CHECK_BOUNCE_DYNAMIC_VWAP_LOWER_BAND,
    "eod_vwap_upper_band": CHECK_BOUNCE_EOD_VWAP_UPPER_BAND,
    "eod_vwap_lower_band": CHECK_BOUNCE_EOD_VWAP_LOWER_BAND,
    "prev_day_high": CHECK_BOUNCE_PREV_DAY_HIGH,
    "prev_day_low": CHECK_BOUNCE_PREV_DAY_LOW,
}

BOUNCE_TYPE_LABELS = {
    "10_candle": "10-Candle",
    "vwap": "Std VWAP",
    "dynamic_vwap": "Dynamic VWAP",
    "eod_vwap": "EOD VWAP",
    "vwap_eod_confluence": "VWAP+EOD Confluence",
    "impulse_retest_vwap_eod": "Impulse Retest VWAP/EOD",
    "ema_8": "8 EMA",
    "ema_15": "15 EMA",
    "ema_21": "21 EMA",
    "h1_ema_15": "H1 15 EMA",
    "h1_sma_20": "H1 20 SMA",
    "vwap_upper_band": "VWAP 1SD Upper",
    "vwap_lower_band": "VWAP 1SD Lower",
    "dynamic_vwap_upper_band": "Dynamic VWAP 1SD Upper",
    "dynamic_vwap_lower_band": "Dynamic VWAP 1SD Lower",
    "eod_vwap_upper_band": "EOD VWAP 1SD Upper",
    "eod_vwap_lower_band": "EOD VWAP 1SD Lower",
    "prev_day_high": "Previous Day High",
    "prev_day_low": "Previous Day Low",
    "regime_pause_rw": "Regime Pause RW",
    "regime_pause_rs": "Regime Pause RS",
    "orb_breakout": "5m ORB Breakout (30m+)",
    "orb_breakdown": "5m ORB Breakdown (30m+)",
    "ema8_grind_hod": "8-EMA Grind New HOD",
    "ema8_grind_lod": "8-EMA Grind New LOD",
    "h1_ema10_bounce": "H1 10-EMA Bounce",
    "h1_blue_after_red": "H1 Blue After Red",
    "h1_green_to_yellow": "H1 Green->Yellow Fade",
}
BOUNCE_LEARNING_TYPE_KEYS = set(BOUNCE_TYPE_DEFAULTS)

# Priority watchlist emphasis (2026-07-02): longs.txt / shorts.txt (the
# trader's live intraday RS/RW dumps) and human focus picks get the full
# treatment every candle; everything else (master AVWAP D1 watch/upgrade
# symbols) is "background" and only refreshes every N cycles, with its cached
# 5-min bars surviving the off-cycles. Cuts IB historical-bar requests roughly
# in proportion to the background share of the scan set.
PRIORITY_WATCHLIST_EMPHASIS = True
BACKGROUND_SYMBOL_REFRESH_EVERY_CYCLES = 3

# D1 focus alert gate (2026-07-02): only surface D1 flags whose upgrade lands
# in an actionable bucket (the near -> favorite/high-conviction move, i.e. the
# A/S-caliber transition) AND that clear the evidence bar - non-negative
# Expected-R when the master scan provides it, and no learning-state mute for
# the context. Suppressed flags are still logged so the decision is auditable.
D1_FLAG_ACTIONABLE_BUCKETS = {"favorite_setup", "high_conviction"}
D1_FLAG_MIN_EXPECTED_R = 0.0

# Connection & Request settings
MAX_CONCURRENT_REQUESTS = 1
REQUEST_DELAY = 0.1  # seconds between IB historical data requests

# IB pacing protection (2026-07-04): the open is heavy on M5 history pulls, so
# pacing complaints (error 100, or any message mentioning "pacing") put an
# escalating cooldown on historical requests instead of hammering the API.
# Repeat violations inside the reset window double the cooldown up to the max.
IB_PACING_BACKOFF_INITIAL_SECONDS = 30.0
IB_PACING_BACKOFF_MAX_SECONDS = 300.0
IB_PACING_VIOLATION_RESET_SECONDS = 600.0

# RRS (Real Relative Strength) settings
RRS_DEFAULT_THRESHOLD = 2.0
RRS_LENGTH = 12
RRS_TIMEOUT = 3.0
RRS_TIMEFRAMES = {
    "5m": {"label": "5 min", "bar_size": "5 mins", "duration": "5 D", "minutes": 5},
    "15m": {"label": "15 min", "bar_size": "15 mins", "duration": "5 D", "minutes": 15},
    "30m": {"label": "30 min", "bar_size": "30 mins", "duration": "5 D", "minutes": 30},
    "1h": {"label": "1 hour", "bar_size": "1 hour", "duration": "5 D", "minutes": 60},
}
SCAN_EXTREME_COUNT = 5
GROUP_STRENGTH_TIMEFRAMES = {
    "D1": {"bar_size": "1 day", "duration": "6 M"},
    "H1": {"bar_size": "1 hour", "duration": "5 D"},
    "M5": {"bar_size": "5 mins", "duration": "5 D"},
}
ENVIRONMENT_FOCUS_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]+$")
DEFAULT_SECTOR_ETF_MAP = {
    "communication-services": "XLC",
    "consumer-cyclical": "XLY",
    "consumer-defensive": "XLP",
    "energy": "XLE",
    "financial-services": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "basic-materials": "XLB",
    "real-estate": "XLRE",
    "technology": "XLK",
    "utilities": "XLU",
}
DEFAULT_INDUSTRY_ETF_MAP = {
    "aerospace-defense": {"sectorKey": "industrials", "industry": "Aerospace & Defense", "sector": "Industrials", "etf": "ITA"},
    "airlines": {"sectorKey": "industrials", "industry": "Airlines", "sector": "Industrials", "etf": "JETS"},
    "asset-management": {"sectorKey": "financial-services", "industry": "Asset Management", "sector": "Financial Services", "etf": "KCE"},
    "auto-manufacturers": {"sectorKey": "consumer-cyclical", "industry": "Auto Manufacturers", "sector": "Consumer Cyclical", "etf": "CARZ"},
    "banks-diversified": {"sectorKey": "financial-services", "industry": "Banks - Diversified", "sector": "Financial Services", "etf": "KBE"},
    "banks-regional": {"sectorKey": "financial-services", "industry": "Banks - Regional", "sector": "Financial Services", "etf": "KRE"},
    "biotechnology": {"sectorKey": "healthcare", "industry": "Biotechnology", "sector": "Healthcare", "etf": "XBI"},
    "building-products-equipment": {"sectorKey": "industrials", "industry": "Building Products & Equipment", "sector": "Industrials", "etf": "XHB"},
    "capital-markets": {"sectorKey": "financial-services", "industry": "Capital Markets", "sector": "Financial Services", "etf": "KCE"},
    "chemicals": {"sectorKey": "basic-materials", "industry": "Chemicals", "sector": "Basic Materials", "etf": "XLB"},
    "communication-equipment": {"sectorKey": "technology", "industry": "Communication Equipment", "sector": "Technology", "etf": "IGN"},
    "computer-hardware": {"sectorKey": "technology", "industry": "Computer Hardware", "sector": "Technology", "etf": "XLK"},
    "consumer-electronics": {"sectorKey": "technology", "industry": "Consumer Electronics", "sector": "Technology", "etf": "XLK"},
    "copper": {"sectorKey": "basic-materials", "industry": "Copper", "sector": "Basic Materials", "etf": "COPX"},
    "credit-services": {"sectorKey": "financial-services", "industry": "Credit Services", "sector": "Financial Services", "etf": "IPAY"},
    "cybersecurity": {"sectorKey": "technology", "industry": "Cybersecurity", "sector": "Technology", "etf": "HACK"},
    "diagnostics-research": {"sectorKey": "healthcare", "industry": "Diagnostics & Research", "sector": "Healthcare", "etf": "IHI"},
    "drug-manufacturers-general": {"sectorKey": "healthcare", "industry": "Drug Manufacturers - General", "sector": "Healthcare", "etf": "PPH"},
    "drug-manufacturers-specialty-generic": {"sectorKey": "healthcare", "industry": "Drug Manufacturers - Specialty & Generic", "sector": "Healthcare", "etf": "IHE"},
    "electrical-equipment-parts": {"sectorKey": "industrials", "industry": "Electrical Equipment & Parts", "sector": "Industrials", "etf": "GRID"},
    "electronic-components": {"sectorKey": "technology", "industry": "Electronic Components", "sector": "Technology", "etf": "SOXX"},
    "engineering-construction": {"sectorKey": "industrials", "industry": "Engineering & Construction", "sector": "Industrials", "etf": "PAVE"},
    "entertainment": {"sectorKey": "communication-services", "industry": "Entertainment", "sector": "Communication Services", "etf": "XLC"},
    "farm-heavy-construction-machinery": {"sectorKey": "industrials", "industry": "Farm & Heavy Construction Machinery", "sector": "Industrials", "etf": "XLI"},
    "gold": {"sectorKey": "basic-materials", "industry": "Gold", "sector": "Basic Materials", "etf": "GDX"},
    "health-information-services": {"sectorKey": "healthcare", "industry": "Health Information Services", "sector": "Healthcare", "etf": "ARKG"},
    "healthcare-plans": {"sectorKey": "healthcare", "industry": "Healthcare Plans", "sector": "Healthcare", "etf": "IHF"},
    "home-improvement-retail": {"sectorKey": "consumer-cyclical", "industry": "Home Improvement Retail", "sector": "Consumer Cyclical", "etf": "XHB"},
    "internet-content-information": {"sectorKey": "communication-services", "industry": "Internet Content & Information", "sector": "Communication Services", "etf": "FDN"},
    "internet-retail": {"sectorKey": "consumer-cyclical", "industry": "Internet Retail", "sector": "Consumer Cyclical", "etf": "IBUY"},
    "insurance-property-casualty": {"sectorKey": "financial-services", "industry": "Insurance - Property & Casualty", "sector": "Financial Services", "etf": "KIE"},
    "insurance-diversified": {"sectorKey": "financial-services", "industry": "Insurance - Diversified", "sector": "Financial Services", "etf": "KIE"},
    "leisure": {"sectorKey": "consumer-cyclical", "industry": "Leisure", "sector": "Consumer Cyclical", "etf": "PEJ"},
    "lithium": {"sectorKey": "basic-materials", "industry": "Lithium", "sector": "Basic Materials", "etf": "LIT"},
    "lodging": {"sectorKey": "consumer-cyclical", "industry": "Lodging", "sector": "Consumer Cyclical", "etf": "PEJ"},
    "medical-devices": {"sectorKey": "healthcare", "industry": "Medical Devices", "sector": "Healthcare", "etf": "IHI"},
    "medical-instruments-supplies": {"sectorKey": "healthcare", "industry": "Medical Instruments & Supplies", "sector": "Healthcare", "etf": "IHI"},
    "memory": {"sectorKey": "technology", "industry": "Memory", "sector": "Technology", "etf": "DRAM"},
    "oil-gas-e-p": {"sectorKey": "energy", "industry": "Oil & Gas E&P", "sector": "Energy", "etf": "XOP"},
    "oil-gas-equipment-services": {"sectorKey": "energy", "industry": "Oil & Gas Equipment & Services", "sector": "Energy", "etf": "OIH"},
    "oil-gas-integrated": {"sectorKey": "energy", "industry": "Oil & Gas Integrated", "sector": "Energy", "etf": "XLE"},
    "packaged-foods": {"sectorKey": "consumer-defensive", "industry": "Packaged Foods", "sector": "Consumer Defensive", "etf": "XLP"},
    "pharmaceutical-retailers": {"sectorKey": "healthcare", "industry": "Pharmaceutical Retailers", "sector": "Healthcare", "etf": "IHE"},
    "pollution-treatment-controls": {"sectorKey": "industrials", "industry": "Pollution & Treatment Controls", "sector": "Industrials", "etf": "PHO"},
    "real-estate-development": {"sectorKey": "real-estate", "industry": "Real Estate - Development", "sector": "Real Estate", "etf": "XLRE"},
    "real-estate-services": {"sectorKey": "real-estate", "industry": "Real Estate Services", "sector": "Real Estate", "etf": "VNQ"},
    "recreational-vehicles": {"sectorKey": "consumer-cyclical", "industry": "Recreational Vehicles", "sector": "Consumer Cyclical", "etf": "CARZ"},
    "reit-diversified": {"sectorKey": "real-estate", "industry": "REIT - Diversified", "sector": "Real Estate", "etf": "VNQ"},
    "reit-industrial": {"sectorKey": "real-estate", "industry": "REIT - Industrial", "sector": "Real Estate", "etf": "VNQ"},
    "reit-retail": {"sectorKey": "real-estate", "industry": "REIT - Retail", "sector": "Real Estate", "etf": "VNQ"},
    "renewable-utilities": {"sectorKey": "utilities", "industry": "Renewable Utilities", "sector": "Utilities", "etf": "ICLN"},
    "residential-construction": {"sectorKey": "consumer-cyclical", "industry": "Residential Construction", "sector": "Consumer Cyclical", "etf": "XHB"},
    "restaurants": {"sectorKey": "consumer-cyclical", "industry": "Restaurants", "sector": "Consumer Cyclical", "etf": "EATZ"},
    "semiconductor-equipment-materials": {"sectorKey": "technology", "industry": "Semiconductor Equipment & Materials", "sector": "Technology", "etf": "SOXX"},
    "semiconductors": {"sectorKey": "technology", "industry": "Semiconductors", "sector": "Technology", "etf": "SMH"},
    "software-application": {"sectorKey": "technology", "industry": "Software - Application", "sector": "Technology", "etf": "IGV"},
    "software-infrastructure": {"sectorKey": "technology", "industry": "Software - Infrastructure", "sector": "Technology", "etf": "IGV"},
    "solar": {"sectorKey": "technology", "industry": "Solar", "sector": "Technology", "etf": "TAN"},
    "specialty-industrial-machinery": {"sectorKey": "industrials", "industry": "Specialty Industrial Machinery", "sector": "Industrials", "etf": "XLI"},
    "specialty-retail": {"sectorKey": "consumer-cyclical", "industry": "Specialty Retail", "sector": "Consumer Cyclical", "etf": "XRT"},
    "steel": {"sectorKey": "basic-materials", "industry": "Steel", "sector": "Basic Materials", "etf": "SLX"},
    "telecom-services": {"sectorKey": "communication-services", "industry": "Telecom Services", "sector": "Communication Services", "etf": "IYZ"},
    "travel-services": {"sectorKey": "consumer-cyclical", "industry": "Travel Services", "sector": "Consumer Cyclical", "etf": "PEJ"},
    "trucking": {"sectorKey": "industrials", "industry": "Trucking", "sector": "Industrials", "etf": "IYT"},
    "uranium": {"sectorKey": "energy", "industry": "Uranium", "sector": "Energy", "etf": "URA"},
    "utilities-regulated-electric": {"sectorKey": "utilities", "industry": "Utilities - Regulated Electric", "sector": "Utilities", "etf": "XLU"},
}

MARKET_ENVIRONMENTS = {
    "bearish_strong": {"label": "Bearish Strong"},
    "bearish_weak": {"label": "Bearish Weak"},
    "bullish_strong": {"label": "Bullish Strong"},
    "bullish_weak": {"label": "Bullish Weak"},
    # 2026-07-08: mature session, SPY not band-held and VWAP position
    # disagreeing with the day color - no trend to lean on. Direction-favored
    # logic treats it as neither bullish nor bearish (startswith checks fail).
    "neutral_chop": {"label": "Neutral / Chop"},
}
# Auto intraday market regime (2026-07-03): SPY green/red vs yesterday's close
# decides bullish/bearish; the magnitude decides strong/weak. The bot tracks
# this itself every cycle; a manual GUI selection overrides it for the session.
MARKET_REGIME_AUTO_ENABLED = True
MARKET_REGIME_STRONG_ABS_PCT = 0.5
# VWAP-based regime read (2026-07-04): SPY holding above VWAP+1stdev for most
# of the session is bullish_strong; above VWAP and green on the day but under
# the band is bullish_weak. Inverted for bearish. Needs at least an hour of
# 5m bars (and volume data) - before that the day%-vs-prev-close rule applies.
MARKET_REGIME_VWAP_MIN_BARS = 12
MARKET_REGIME_VWAP_BAND_FRACTION = 0.60

# Regime-pause banger detection (2026-07-03). In a bearish tape, when SPY
# prints a green 5m candle or stops making new session lows for a few candles,
# whatever is STILL falling or refusing to bounce is the real weakness - the
# AAOI pattern: much weaker than SPY all day, then keeps dropping through the
# SPY pause. Inverted for bullish tape. Candidates come from the trader's
# longs.txt / shorts.txt only.
REGIME_PAUSE_NO_NEW_EXTREME_CANDLES = 3
REGIME_PAUSE_SWEEP_MAX_CANDLES = 3
REGIME_BANGER_DAY_EXCESS_PCT = 0.75
REGIME_BANGER_WINDOW_EXCESS_PCT = 0.20
REGIME_PAUSE_RW_TYPE = "regime_pause_rw"
REGIME_PAUSE_RS_TYPE = "regime_pause_rs"

# Entry assist (2026-07-08): regime-tailored entry timing on one button.
# Strong regimes run a pullback/bounce WINDOW (manual click or auto SPY-pause
# detection marks the start; the end ranks which names held the trade
# direction best through the counter-move). Weak regimes emit instant
# strongest/weakest trailing-30m lists; neutral/chop emits both sides.
ENTRY_WINDOW_TOP_N = 8
ENTRY_WINDOW_MIN_BARS = 2
ENTRY_MOVERS_MINUTES = 30
ENTRY_AUTO_MOVERS_INTERVAL_MIN = 30


def entry_assist_mode_for_env(env_key) -> dict:
    """Which entry-assist behavior a regime gets: window vs instant movers."""
    env = str(env_key or "").strip().lower()
    if env == "bullish_strong":
        return {"mode": "window", "sides": ("long",)}
    if env == "bearish_strong":
        return {"mode": "window", "sides": ("short",)}
    if env == "bullish_weak":
        return {"mode": "movers", "sides": ("long",)}
    if env == "bearish_weak":
        return {"mode": "movers", "sides": ("short",)}
    # neutral_chop (and anything unknown): no trend to pause against - emit
    # both the strongest and weakest lists ("does everything").
    return {"mode": "movers", "sides": ("long", "short")}

# Delayed 5m opening-range breaks (2026-07-04). The trader's ORB variant: the
# 5-minute opening range (the 9:30 candle) has to survive the first 30 minutes
# untouched on a closing basis; the FIRST 5m close through it at 30+ minutes
# after the open is the alert. An early close through the range kills the
# setup for the day (wicks through are fine - failed pokes keep it alive).
# longs.txt names break the OR high, shorts.txt names break the OR low.
ORB_DELAY_MINUTES = 30
# A break older than this many bars when first seen (e.g. after a restart) is
# stale news, not a live trigger - mark it dead instead of alerting.
ORB_FRESH_BREAK_MAX_BARS = 2
ORB_BREAKOUT_TYPE = "orb_breakout"
ORB_BREAKDOWN_TYPE = "orb_breakdown"

# 8-EMA grind squeeze (2026-07-04). Longs: a strong name pulls back after the
# initial push, then rides the 5m 8-EMA without a single close below it; the
# candle-to-EMA gap compresses (the "squeeze") and the alert fires on the bar
# that pushes into a new HOD. Inverted for shorts (grind below, new LOD).
EMA8_GRIND_LENGTH = 8
EMA8_GRIND_MIN_BARS = 6  # >=30 min of closes holding the 8-EMA before the push
EMA8_GRIND_PULLBACK_MIN_BARS = 3  # prior HOD/LOD must be at least this stale
EMA8_GRIND_MIN_DAY_PCT = 0.75  # "strong" gate: day move with the trade direction
EMA8_GRIND_SQUEEZE_BARS = 4  # bars right before the push that must hug the EMA
EMA8_GRIND_SQUEEZE_MAX_ATR = 0.35  # max |close - ema8| in intraday ATRs
EMA8_GRIND_HOD_TYPE = "ema8_grind_hod"
EMA8_GRIND_LOD_TYPE = "ema8_grind_lod"

# H1 candle-color sweeps (2026-07-06): the trader's hourly regime colors.
# EMA15 vs SMA20 defines the hourly trend; the close's position classifies
# the candle (green/orange/yellow up-structure, red/blue down-structure).
# Signals under test, swept on longs.txt / shorts.txt closed H1 candles:
#   - light blue: green regime + open above the H1 10-EMA with the low
#     tagging (within 0.02 ATR) or piercing it and the close recovering.
#   - blue-after-red: red H1 followed by a blue reclaim candle (long).
#   - green-to-yellow: green H1 followed by a yellow breakdown candle (short).
H1_COLOR_MIN_BARS = 22  # SMA20 must exist on both the signal and prior candle
H1_COLOR_ATR_LENGTH = 14
H1_EMA10_BOUNCE_TOUCH_ATR = 0.02  # trader spec: |low - EMA10| <= 0.02 * ATR
H1_EMA10_BOUNCE_TYPE = "h1_ema10_bounce"
H1_BLUE_AFTER_RED_TYPE = "h1_blue_after_red"
H1_GREEN_TO_YELLOW_TYPE = "h1_green_to_yellow"
MIN_MOVE_RATIO_FOR_SIGNAL = 0.25
MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL = 0.15
MASTER_AVWAP_FOCUS_MIN_ABS_RRS = 2.75
MASTER_AVWAP_FOCUS_MIN_MOVE_RATIO = 0.45
MASTER_AVWAP_FOCUS_MIN_EXCESS_MOVE_RATIO = 0.25
BOUNCE_INDUSTRY_RRS_BONUS_CAP = 10
EMIT_MASTER_AVWAP_FOCUS_RRS_ALERTS = False
ENVIRONMENT_HIGHLIGHT_LIMIT = 6
ENVIRONMENT_SCAN_LIMIT = 25
ENVIRONMENT_RECENT_PROFILE_BARS = 3
ENVIRONMENT_PROFILE_EMA_ALPHA = 0.70
ENVIRONMENT_RECENT_BIAS_MIN_SCORE = 0.15
ENVIRONMENT_TOTAL_BIAS_MIN_SCORE = 0.05
ENVIRONMENT_SCAN_DELAY_MINUTES = 60
ENVIRONMENT_CONTEXT_MIN_RRS_FRACTION = 0.35
ENVIRONMENT_HOLD_MOVE_RATIO = 0.10
SPY_COMPRESSION_THRESHOLD = 0.35
SPY_UP_THRESHOLD = 0.20
SPY_PULLBACK_DELTA_THRESHOLD = 0.25
IMPULSE_RRS_PROFILE_LENGTH = 6
IMPULSE_RRS_RECENT_PHASE_BARS = 3
IMPULSE_RRS_FAVORED_TREND_RATIO = 0.60
IMPULSE_RRS_FAVORED_EXCESS_RATIO = 0.20
IMPULSE_RRS_UNFAVORED_TREND_RATIO = 0.90
IMPULSE_RRS_UNFAVORED_EXCESS_RATIO = 0.45
IMPULSE_RRS_COUNTER_RRS = 0.35
IMPULSE_RRS_COUNTER_MOVE_RATIO = 0.20
IMPULSE_RRS_COUNTER_BARS_REQUIRED = 2


def _default_environment_focus_history_payload():
    return {
        "schema_version": ENVIRONMENT_FOCUS_HISTORY_SCHEMA_VERSION,
        "updated_at": None,
        "days": {},
    }


def _load_environment_focus_history_payload():
    if not ENVIRONMENT_FOCUS_HISTORY_FILENAME.exists():
        return _default_environment_focus_history_payload()
    try:
        payload = json.loads(ENVIRONMENT_FOCUS_HISTORY_FILENAME.read_text(encoding="utf-8"))
    except Exception:
        return _default_environment_focus_history_payload()
    if not isinstance(payload, dict):
        return _default_environment_focus_history_payload()

    history = _default_environment_focus_history_payload()
    history["schema_version"] = int(
        payload.get("schema_version", ENVIRONMENT_FOCUS_HISTORY_SCHEMA_VERSION)
        or ENVIRONMENT_FOCUS_HISTORY_SCHEMA_VERSION
    )
    history["updated_at"] = payload.get("updated_at")
    history["days"] = payload.get("days", {}) if isinstance(payload.get("days"), dict) else {}
    return history


def _save_environment_focus_history_payload(payload):
    history = _default_environment_focus_history_payload()
    if isinstance(payload, dict):
        history["days"] = payload.get("days", {}) if isinstance(payload.get("days"), dict) else {}
    history["schema_version"] = ENVIRONMENT_FOCUS_HISTORY_SCHEMA_VERSION
    history["updated_at"] = get_market_local_now().isoformat(timespec="seconds")
    ENVIRONMENT_FOCUS_HISTORY_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT_FOCUS_HISTORY_FILENAME.write_text(json.dumps(history, indent=2), encoding="utf-8")
BOUNCE_LEVEL_LOOKBACK_BARS = 8
BOUNCE_LEVEL_BAND_ATR = 0.12
BOUNCE_MAX_NEAR_BARS = 4
BOUNCE_MAX_CROSSES = 2
BOUNCE_MIN_SEPARATION_ATR = 0.45
BOUNCE_MIN_LEVEL_SLOPE_ATR = 0.04
BOUNCE_LEVEL_SLOPE_LOOKBACK = 3
BOUNCE_ONE_SIDED_BARS = 4
BOUNCE_ONE_SIDED_RATIO = 0.75
BOUNCE_REACTION_LOOKBACK_BARS = 6
BOUNCE_MIN_BODY_TO_MEDIAN = 1.10
BOUNCE_MIN_RANGE_TO_MEDIAN = 1.05
BOUNCE_VOLUME_LOOKBACK_BARS = 6
BOUNCE_MIN_RELATIVE_VOLUME = 1.10
BOUNCE_CONFIRMATION_MAX_CANDLES = 3
FAST_CONFIRM_BOUNCE_TYPES = {"impulse_retest_vwap_eod", "ema_8", "ema_15", "ema_21"}
FAST_CONFIRMATION_MAX_CANDLES = 1
PREV_DAY_LEVEL_MIN_PRIOR_BARS = 4
PREV_DAY_LEVEL_RESPECT_RATIO = 0.85
PREV_DAY_LEVEL_MAX_WRONG_SIDE_CLOSES = 1
PREV_DAY_LEVEL_RECENT_RESPECT_BARS = 6

# Session structure gate (2026-07-04): most bounce setups should be a stock
# MOVING with the trade direction that makes a simple retest of a measured
# level - not a choppy tape pulling back over and over, and not a name pinned
# in compression. Longs: a real advance off the open must exist, the day's
# move must still be intact (close above the session open), and at most N
# significant pullbacks (zigzag swings against the move) may have printed -
# the live retest counts as one. Inverted for shorts.
BOUNCE_SESSION_STRUCTURE_GATE = True
BOUNCE_STRUCTURE_MIN_BARS = 4  # too few bars -> not enough structure to judge
BOUNCE_MIN_ADVANCE_ATR = 2.0  # day leg with the trade direction, intraday ATRs
BOUNCE_MAX_SIGNIFICANT_PULLBACKS = 2
BOUNCE_PULLBACK_MIN_ATR = 1.2  # swing depth for a pullback to count...
BOUNCE_PULLBACK_MIN_ADVANCE_FRAC = 0.30  # ...and as a fraction of the advance

BOUNCE_LEARNING_SCHEMA_VERSION = 4
BOUNCE_OUTCOME_MILESTONE_BARS = (1, 3, 6, 12)
BOUNCE_EOD_FINALIZE_GRACE_MINUTES = 10
BOUNCE_PERFORMANCE_MIN_SAMPLES = 5
BOUNCE_PERFORMANCE_R_CLIP = 4.0
BOUNCE_CANDIDATE_EVENT_COLUMNS = [
    "schema_version",
    "event_id",
    "event_type",
    "logged_at",
    "trade_date",
    "symbol",
    "direction",
    "bounce_types",
    "reason",
    "score",
    "entry_trigger",
    "entry_price",
    "stop_price",
    "risk_per_share",
    "target_1r",
    "target_2r",
    "atr",
    "threshold",
    "rrs_timeframe",
    "rrs_spy",
    "rrs_sector",
    "rrs_industry",
    "sector",
    "industry",
    "sector_etf",
    "industry_etf",
    "market_environment",
    "human_focus_pick",
    "human_focus_side",
    "master_avwap_focus_label",
    "master_avwap_priority_bucket",
    "master_avwap_setup_family",
    "master_avwap_h1_focus_type",
    "master_avwap_swing_traits",
    "candles_waited",
    "levels_json",
    "candle_json",
    "metrics_json",
    "context_json",
]
BOUNCE_OUTCOME_COLUMNS = [
    "schema_version",
    "event_id",
    "event_type",
    "logged_at",
    "trade_date",
    "symbol",
    "direction",
    "entry_time",
    "entry_price",
    "stop_price",
    "risk_per_share",
    "bars_elapsed",
    "minutes_elapsed",
    "close_r",
    "mfe_r",
    "mae_r",
    "best_price",
    "worst_price",
    "target_1r_hit",
    "target_2r_hit",
    "stop_hit",
    "status",
    "milestone_bar",
    "context_json",
    "outcome_mode",
    "eod_close",
    "eod_move_pct",
    "mfe_pct",
    "mae_pct",
]
BOUNCE_FEEDBACK_COLUMNS = [
    "schema_version",
    "feedback_id",
    "logged_at",
    "source",
    "rating",
    "reason",
    "event_id",
    "event_type",
    "trade_date",
    "symbol",
    "direction",
    "bounce_types",
    "alert_time",
    "alert_message",
    "entry_price",
    "stop_price",
    "risk_per_share",
    "score",
    "levels_json",
    "candle_json",
    "metrics_json",
    "context_json",
]
VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES = 4
APP_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s]: %(message)s"


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify_key(value):
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-")


def load_sector_etf_map():
    SECTOR_ETF_MAP_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    if not SECTOR_ETF_MAP_FILENAME.exists():
        with open(SECTOR_ETF_MAP_FILENAME, "w") as fh:
            json.dump(DEFAULT_SECTOR_ETF_MAP, fh, indent=2, sort_keys=True)
    try:
        with open(SECTOR_ETF_MAP_FILENAME, "r") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            normalized = {}
            for k, v in data.items():
                if not v:
                    continue
                key = slugify_key(k)
                if key:
                    normalized[key] = str(v).strip().upper()
            if normalized:
                return normalized
    except Exception as exc:
        logging.warning(f"Failed loading sector ETF map: {exc}")
    return dict(DEFAULT_SECTOR_ETF_MAP)


def _default_industry_etf_map_payload():
    now = utc_now_iso()
    refs = {}
    for industry_key, entry in DEFAULT_INDUSTRY_ETF_MAP.items():
        refs[industry_key] = {
            "sectorKey": entry.get("sectorKey", ""),
            "industry": entry.get("industry", ""),
            "sector": entry.get("sector", ""),
            "etf": (entry.get("etf") or "").strip().upper() or None,
            "source": "default",
            "first_seen_utc": now,
            "last_seen_utc": now,
            "seen_count": 0,
        }
    return {"version": 1, "updated_utc": now, "yahoo_industryKey_to_ref": refs}


def _load_industry_etf_map_file():
    INDUSTRY_ETF_MAP_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    default_map = _default_industry_etf_map_payload()
    if not INDUSTRY_ETF_MAP_FILENAME.exists():
        with open(INDUSTRY_ETF_MAP_FILENAME, "w") as fh:
            json.dump(default_map, fh, indent=2, sort_keys=True)
        return default_map
    try:
        with open(INDUSTRY_ETF_MAP_FILENAME, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data.setdefault("version", 1)
    data.setdefault("updated_utc", utc_now_iso())
    refs = data.setdefault("yahoo_industryKey_to_ref", {})
    changed = False
    for industry_key, default_entry in default_map["yahoo_industryKey_to_ref"].items():
        existing = refs.get(industry_key)
        if not isinstance(existing, dict):
            refs[industry_key] = dict(default_entry)
            changed = True
            continue
        for field in ("sectorKey", "industry", "sector"):
            if not existing.get(field) and default_entry.get(field):
                existing[field] = default_entry[field]
                changed = True
        if not existing.get("etf") and default_entry.get("etf"):
            existing["etf"] = default_entry["etf"]
            existing.setdefault("source", "default")
            changed = True
        existing.setdefault("first_seen_utc", default_entry.get("first_seen_utc") or utc_now_iso())
        existing.setdefault("last_seen_utc", existing.get("first_seen_utc") or utc_now_iso())
        existing.setdefault("seen_count", 0)
    if changed:
        data["updated_utc"] = utc_now_iso()
        try:
            with open(INDUSTRY_ETF_MAP_FILENAME, "w") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
        except Exception as exc:
            logging.warning(f"Failed updating industry ETF map defaults: {exc}")
    return data


def load_industry_etf_map():
    return _load_industry_etf_map_file()


def load_and_update_industry_etf_map(industryKey, sectorKey, industry_name, sector_name):
    data = _load_industry_etf_map_file()
    refs = data.setdefault("yahoo_industryKey_to_ref", {})
    now = utc_now_iso()
    key = (industryKey or "").strip()
    if key:
        entry = refs.get(key)
        if not isinstance(entry, dict):
            entry = {
                "sectorKey": sectorKey or "",
                "industry": industry_name or "",
                "sector": sector_name or "",
                "etf": None,
                "first_seen_utc": now,
                "last_seen_utc": now,
                "seen_count": 1,
            }
        else:
            entry["sectorKey"] = entry.get("sectorKey") or (sectorKey or "")
            entry["industry"] = entry.get("industry") or (industry_name or "")
            entry["sector"] = entry.get("sector") or (sector_name or "")
            entry["etf"] = entry.get("etf") if entry.get("etf") else None
            entry["first_seen_utc"] = entry.get("first_seen_utc") or now
            entry["last_seen_utc"] = now
            entry["seen_count"] = int(entry.get("seen_count", 0) or 0) + 1
        refs[key] = entry
        data["updated_utc"] = now
        with open(INDUSTRY_ETF_MAP_FILENAME, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    return data


def resolve_sector_etf(sectorKey, sector_map=None):
    key = (sectorKey or "").strip()
    mapping = sector_map if isinstance(sector_map, dict) else load_sector_etf_map()
    if not key:
        return "SPY"
    candidates = [
        key,
        key.lower(),
        key.replace("_", "-").lower(),
        slugify_key(key),
    ]
    for candidate in candidates:
        if candidate and candidate in mapping:
            return mapping[candidate]
    return "SPY"


def resolve_industry_ref_etf(industryKey, sectorKey, industry_map_data=None, sector_map=None):
    if industryKey:
        data = industry_map_data if isinstance(industry_map_data, dict) else _load_industry_etf_map_file()
        refs = data.get("yahoo_industryKey_to_ref", {})
        candidates = [
            str(industryKey or "").strip(),
            str(industryKey or "").strip().lower(),
            slugify_key(industryKey),
        ]
        for candidate in candidates:
            entry = refs.get(candidate, {}) if isinstance(refs, dict) else {}
            etf = (entry.get("etf") or "").strip().upper() if isinstance(entry, dict) else ""
            if etf:
                return etf
    return resolve_sector_etf(sectorKey, sector_map)


def load_symbol_classification_cache():
    cache = {}
    if not SYMBOL_CLASSIFICATION_CACHE_FILENAME.exists():
        return cache
    try:
        with open(SYMBOL_CLASSIFICATION_CACHE_FILENAME, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                symbol = (row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                cache[symbol] = {
                    "symbol": symbol,
                    "sectorKey": slugify_key(row.get("sectorKey") or row.get("sector") or ""),
                    "industryKey": slugify_key(row.get("industryKey") or row.get("industry") or ""),
                    "sector": row.get("sector", ""),
                    "industry": row.get("industry", ""),
                    "updated_utc": row.get("updated_utc", ""),
                }
    except Exception as exc:
        logging.warning(f"Failed loading symbol classification cache: {exc}")
    return cache

##########################################
# Logging Filter
##########################################
class HistoricalDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Code=2176" in msg:
            return False
        if ("REQUEST reqHistoricalData" in msg or
            "SENDING reqHistoricalData" in msg or
            "REQUEST startApi" in msg or
            "SENDING startApi" in msg or
            "HistoricalDataEnd" in msg or
            "Building ATR cache for new symbols" in msg or
            "Calculated shares" in msg or
            "Not enough" in msg):
            return False
        return True

##########################################
# Utility Functions
##########################################
def wait_for_candle_close(stop_event=None):
    now = get_market_local_now()
    sec_since_5 = (now.minute % 5) * 60 + now.second
    sec_to_go = 300 - sec_since_5
    logging.info(f"Waiting for candle to close: {sec_to_go} seconds remaining.")
    if stop_event is not None:
        # Cancellable wait: shutdown must not block up to a full 5-minute bar.
        if stop_event.wait(max(0, sec_to_go)):
            return
    else:
        while sec_to_go > 0:
            time.sleep(1)
            sec_to_go -= 1
    logging.info("Candle has closed.")

def read_tickers(file_path):
    if not os.path.exists(file_path):
        logging.warning(f"{file_path} does not exist.")
        return []
    with open(file_path, "r") as f:
        tickers = [line.strip().upper() for line in f
                  if line.strip() and "Symbols from TC2000" not in line]
    logging.debug(f"Loaded tickers from {file_path}: {tickers}")
    return tickers


def _parse_iso_date_safe(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _feedback_json_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _normalize_bounce_feedback_context(context):
    if not isinstance(context, dict):
        context = {}
    normalized = dict(context)
    display_text = str(
        normalized.get("alert_message")
        or normalized.get("display_text")
        or normalized.get("text")
        or ""
    ).strip()
    if display_text:
        normalized["alert_message"] = display_text

    if not normalized.get("symbol") and display_text:
        match = re.match(
            r"^\s*(?P<symbol>[A-Z0-9.\-]+)\s*:\s*Bounce confirmed\s*\((?P<direction>long|short)\)\s*from\s*(?P<levels>.+?)\s*$",
            display_text,
            re.IGNORECASE,
        )
        if match:
            normalized["symbol"] = match.group("symbol").upper()
            normalized["direction"] = match.group("direction").lower()
            if not normalized.get("bounce_types"):
                level_text = match.group("levels")
                levels = [
                    item
                    for item in re.findall(r"[A-Za-z0-9_]+", level_text)
                    if item.lower() not in {"from", "long", "short"}
                ]
                normalized["bounce_types"] = ";".join(levels)

    normalized["symbol"] = str(normalized.get("symbol") or "").strip().upper()
    normalized["direction"] = str(normalized.get("direction") or "").strip().lower()
    normalized["event_id"] = str(normalized.get("event_id") or "").strip()
    normalized["event_type"] = str(normalized.get("event_type") or "confirmed").strip() or "confirmed"
    normalized["trade_date"] = str(normalized.get("trade_date") or get_market_local_now().date().isoformat()).strip()
    normalized["bounce_types"] = str(normalized.get("bounce_types") or "").strip()
    normalized["alert_time"] = str(normalized.get("alert_time") or "").strip()
    return normalized


def _normalize_alert_message_payload(msg):
    if isinstance(msg, dict):
        display_text = str(
            msg.get("text")
            or msg.get("display_text")
            or msg.get("message")
            or ""
        ).strip()
        feedback = msg.get("feedback")
        if isinstance(feedback, dict):
            feedback_context = dict(feedback)
            if display_text and not feedback_context.get("alert_message"):
                feedback_context["alert_message"] = display_text
            return display_text, _normalize_bounce_feedback_context(feedback_context)
        return display_text, {}
    return str(msg), {}


def record_bounce_feedback(feedback_context, rating, reason="", source="gui") -> Path:
    context = _normalize_bounce_feedback_context(feedback_context)
    rating = str(rating or "").strip().lower()
    if rating in {"ok", "yes", "confirm", "confirmed"}:
        rating = "good"
    elif rating in {"bad", "wrong", "not_good", "issue"}:
        rating = "issue"
    if rating not in {"good", "issue"}:
        rating = rating or "unspecified"

    logged_at = get_market_local_now().isoformat(timespec="seconds")
    feedback_key = "|".join(
        [
            context.get("event_id") or context.get("symbol") or "bounce",
            context.get("direction") or "",
            context.get("alert_time") or "",
            logged_at,
            rating,
        ]
    )
    feedback_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", feedback_key).strip("_")[:180]
    row = {
        "schema_version": BOUNCE_LEARNING_SCHEMA_VERSION,
        "feedback_id": feedback_id,
        "logged_at": logged_at,
        "source": str(source or "gui").strip() or "gui",
        "rating": rating,
        "reason": str(reason or "").strip(),
        "event_id": context.get("event_id", ""),
        "event_type": context.get("event_type", ""),
        "trade_date": context.get("trade_date", ""),
        "symbol": context.get("symbol", ""),
        "direction": context.get("direction", ""),
        "bounce_types": context.get("bounce_types", ""),
        "alert_time": context.get("alert_time", ""),
        "alert_message": context.get("alert_message", ""),
        "entry_price": context.get("entry_price", ""),
        "stop_price": context.get("stop_price", ""),
        "risk_per_share": context.get("risk_per_share", ""),
        "score": context.get("score", ""),
        "levels_json": _feedback_json_value(context.get("levels_json", "")),
        "candle_json": _feedback_json_value(context.get("candle_json", "")),
        "metrics_json": _feedback_json_value(context.get("metrics_json", "")),
        "context_json": _feedback_json_value(context.get("context_json", "")),
    }
    try:
        INTRADAY_BOUNCE_FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not INTRADAY_BOUNCE_FEEDBACK_CSV.exists()
            or INTRADAY_BOUNCE_FEEDBACK_CSV.stat().st_size == 0
        )
        with INTRADAY_BOUNCE_FEEDBACK_CSV.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=BOUNCE_FEEDBACK_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in BOUNCE_FEEDBACK_COLUMNS})
    except Exception as exc:
        logging.error(f"Failed writing bounce feedback to {INTRADAY_BOUNCE_FEEDBACK_CSV}: {exc}")
        raise
    return INTRADAY_BOUNCE_FEEDBACK_CSV


def _bounce_perf_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _bounce_perf_clip_r(value, limit: float = BOUNCE_PERFORMANCE_R_CLIP):
    numeric = _bounce_perf_float(value)
    if numeric is None:
        return None
    clip_limit = max(0.5, float(limit))
    return max(-clip_limit, min(clip_limit, float(numeric)))


def _bounce_perf_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _split_delimited_text(value) -> list[str]:
    items = []
    for item in re.split(r"[;,]", str(value or "")):
        item = item.strip()
        if item:
            items.append(item)
    return sorted(set(items))


def _normalize_bounce_type_key(value) -> str:
    text = str(value or "").strip()
    if text in {"10_candle_low", "10_candle_high"}:
        return "10_candle"
    return text


def _bounce_type_keys_from_levels(levels) -> list[str]:
    keys = levels.keys() if isinstance(levels, dict) else levels or []
    return sorted(
        {
            normalized
            for normalized in (_normalize_bounce_type_key(key) for key in keys)
            if normalized
        }
    )


def _split_bounce_type_text(value) -> list[str]:
    return sorted(
        {
            normalized
            for normalized in (_normalize_bounce_type_key(item) for item in _split_delimited_text(value))
            if normalized
        }
    )


def _read_bounce_learning_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    desired = set(columns)
    try:
        return pd.read_csv(path, usecols=lambda column: column in desired)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _bounce_time_bucket(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "unknown"
    from bounce_bot_lib.learning import time_bucket_for

    return time_bucket_for(parsed.to_pydatetime())


def _bounce_quality_time(row: dict) -> datetime | None:
    """Prefer the signal/entry bar over delayed notification wall time."""
    for key in ("signal_time", "entry_time", "bar_time", "timestamp"):
        value = row.get(key)
        if value in (None, ""):
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return parsed.to_pydatetime()
    return None


def _bounce_rrs_alignment(row: dict) -> str:
    direction = str(row.get("direction") or "").strip().lower()
    rrs_value = _bounce_perf_float(row.get("rrs_spy"))
    if rrs_value is None:
        return "unknown"
    if direction == "long":
        if rrs_value >= 2.0:
            return "strong_aligned"
        if rrs_value > 0:
            return "aligned"
        return "counter"
    if direction == "short":
        if rrs_value <= -2.0:
            return "strong_aligned"
        if rrs_value < 0:
            return "aligned"
        return "counter"
    return "unknown"


def _migrate_csv_header(path, fieldnames):
    """One-time widen of an append-only CSV when new columns are added.

    Existing rows get blank values for the new columns; no-op when the header
    already matches or the file does not exist.
    """
    try:
        if not Path(path).exists():
            return
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == list(fieldnames):
                return
            if not reader.fieldnames or not set(reader.fieldnames) <= set(fieldnames):
                return  # unknown layout; never destroy data
            rows = list(reader)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    except Exception as exc:
        logging.warning("CSV header migration skipped for %s: %s", path, exc)


def _format_bounce_alert_message(symbol, direction, levels_list, event_row, quality, exit_note="") -> str:
    """Alert text: tier + confirmation + trade plan + the measured reasons.

    PROVEN bounces (a segment with strong measured avg AND median R matched
    this alert) carry the token + the evidence so the trader sees WHY it is a
    take-this-one alert; the Alert Center gives the token banger treatment."""
    row = event_row if isinstance(event_row, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    tier = str(quality.get("tier") or "B")
    proven_token = "PROVEN " if quality.get("proven") else ""
    parts = [f"[{tier}-TIER] {proven_token}{symbol}: Bounce confirmed ({direction}) from {levels_list}"]

    def _num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    entry = _num(row.get("entry_price"))
    stop = _num(row.get("stop_price"))
    risk = _num(row.get("risk_per_share"))
    target_1r = _num(row.get("target_1r"))
    if entry is not None and stop is not None:
        plan = f"entry {entry:.2f}, stop {stop:.2f}"
        if risk:
            plan += f" (risk {risk:.2f})"
        parts.append(plan)
    if target_1r is not None:
        # Exit discipline from the tracker: most bounces touch +1R (60-80%) but
        # round-trip; harvesting the partial is the measured edge.
        parts.append(f"take 50% at +1R {target_1r:.2f}, trail the rest")
    if exit_note:
        parts.append(str(exit_note))
    proven_reasons = quality.get("proven_reasons") or []
    if proven_reasons:
        parts.append("proven: " + "; ".join(proven_reasons))
    reasons = quality.get("reasons") or []
    if reasons:
        parts.append("why: " + "; ".join(reasons[:3]))
    return " | ".join(parts)


def _latest_bounce_outcome_rows(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    if outcomes_df.empty or "event_id" not in outcomes_df.columns:
        return pd.DataFrame()
    outcomes = outcomes_df.copy()
    event_type_series = (
        outcomes["event_type"]
        if "event_type" in outcomes.columns
        else pd.Series([""] * len(outcomes), index=outcomes.index)
    )
    status_series = (
        outcomes["status"]
        if "status" in outcomes.columns
        else pd.Series([""] * len(outcomes), index=outcomes.index)
    )
    outcomes = outcomes[
        (event_type_series.astype(str).str.lower() == "final")
        & (status_series.astype(str).str.lower() == "eod_complete")
    ].copy()
    if outcomes.empty:
        return pd.DataFrame()
    bars_elapsed_series = (
        outcomes["bars_elapsed"]
        if "bars_elapsed" in outcomes.columns
        else pd.Series([0] * len(outcomes), index=outcomes.index)
    )
    logged_at_series = (
        outcomes["logged_at"]
        if "logged_at" in outcomes.columns
        else pd.Series([""] * len(outcomes), index=outcomes.index)
    )
    outcomes["_bars_elapsed"] = pd.to_numeric(bars_elapsed_series, errors="coerce").fillna(0)
    outcomes["_logged_at"] = pd.to_datetime(
        logged_at_series,
        errors="coerce",
    )
    outcomes = outcomes.sort_values(["event_id", "_bars_elapsed", "_logged_at"])
    return outcomes.drop_duplicates(subset=["event_id"], keep="last")


def build_intraday_bounce_performance_rows(
    *,
    candidates_path: Path = INTRADAY_BOUNCE_CANDIDATES_CSV,
    outcomes_path: Path = INTRADAY_BOUNCE_OUTCOMES_CSV,
    min_samples: int = BOUNCE_PERFORMANCE_MIN_SAMPLES,
) -> list[dict]:
    candidate_columns = [
        "event_id",
        "event_type",
        "logged_at",
        "trade_date",
        "symbol",
        "direction",
        "bounce_types",
        "score",
        "risk_per_share",
        "rrs_spy",
        "rrs_sector",
        "rrs_industry",
        "market_environment",
        "master_avwap_focus_label",
        "master_avwap_priority_bucket",
        "master_avwap_setup_family",
        "master_avwap_h1_focus_type",
        "master_avwap_swing_traits",
    ]
    outcome_columns = [
        "event_id",
        "event_type",
        "logged_at",
        "entry_time",
        "bars_elapsed",
        "close_r",
        "mfe_r",
        "mae_r",
        "target_1r_hit",
        "target_2r_hit",
        "stop_hit",
        "status",
        "outcome_mode",
        "eod_close",
        "eod_move_pct",
        "mfe_pct",
        "mae_pct",
    ]
    candidates = _read_bounce_learning_csv(candidates_path, candidate_columns)
    outcomes = _read_bounce_learning_csv(outcomes_path, outcome_columns)
    if candidates.empty or outcomes.empty or "event_id" not in candidates.columns:
        return []

    if "event_type" not in candidates.columns:
        return []
    candidates = candidates[candidates["event_type"].astype(str).str.lower() == "confirmed"].copy()
    if candidates.empty:
        return []
    candidate_logged_series = (
        candidates["logged_at"]
        if "logged_at" in candidates.columns
        else pd.Series([""] * len(candidates), index=candidates.index)
    )
    candidates["_logged_at"] = pd.to_datetime(candidate_logged_series, errors="coerce")
    candidates = candidates.sort_values(["event_id", "_logged_at"]).drop_duplicates(subset=["event_id"], keep="last")
    latest_outcomes = _latest_bounce_outcome_rows(outcomes)
    if latest_outcomes.empty:
        return []

    joined = candidates.merge(latest_outcomes, on="event_id", how="inner", suffixes=("", "_outcome"))
    if joined.empty:
        return []

    observations = []
    for record in joined.to_dict("records"):
        close_r = _bounce_perf_clip_r(record.get("close_r"))
        if close_r is None:
            continue
        direction = str(record.get("direction") or "").strip().lower() or "unknown"
        bounce_types = _split_bounce_type_text(record.get("bounce_types"))
        combo = "+".join(bounce_types) if bounce_types else "unknown"
        common = {
            "direction": direction,
            "event_id": record.get("event_id"),
            "symbol": str(record.get("symbol") or "").strip().upper(),
            "score": _bounce_perf_float(record.get("score")),
            "risk_per_share": _bounce_perf_float(record.get("risk_per_share")),
            "eod_r": close_r,
            "close_r": close_r,
            "mfe_r": _bounce_perf_clip_r(record.get("mfe_r")),
            "mae_r": _bounce_perf_clip_r(record.get("mae_r")),
            "eod_move_pct": _bounce_perf_float(record.get("eod_move_pct")),
            "mfe_pct": _bounce_perf_float(record.get("mfe_pct")),
            "mae_pct": _bounce_perf_float(record.get("mae_pct")),
            "target_1r_hit": _bounce_perf_bool(record.get("target_1r_hit")),
            "target_2r_hit": _bounce_perf_bool(record.get("target_2r_hit")),
            "stop_hit": _bounce_perf_bool(record.get("stop_hit")),
        }
        dimensions = [
            ("bounce_combo", combo),
            ("market_environment", str(record.get("market_environment") or "unknown").strip() or "unknown"),
            ("rrs_alignment", _bounce_rrs_alignment(record)),
            ("time_bucket", _bounce_time_bucket(record.get("entry_time") or record.get("logged_at"))),
            (
                "master_avwap_focus",
                str(record.get("master_avwap_focus_label") or "none").strip() or "none",
            ),
            (
                "master_avwap_priority_bucket",
                str(record.get("master_avwap_priority_bucket") or "none").strip() or "none",
            ),
            (
                "master_avwap_setup_family",
                str(record.get("master_avwap_setup_family") or "none").strip() or "none",
            ),
            (
                "master_avwap_h1_focus_type",
                str(record.get("master_avwap_h1_focus_type") or "none").strip() or "none",
            ),
        ]
        dimensions.extend(("bounce_type", bounce_type) for bounce_type in bounce_types)
        dimensions.extend(
            ("master_avwap_swing_trait", trait)
            for trait in _split_delimited_text(record.get("master_avwap_swing_traits"))
        )
        h1_focus_type = str(record.get("master_avwap_h1_focus_type") or "").strip().lower()
        if h1_focus_type == "top_pattern" and "h1_ema_15" in bounce_types:
            dimensions.append(("top_pattern_entry_timing", "h1_15ema_bounce"))
        for dimension, segment in dimensions:
            observations.append({**common, "dimension": dimension, "segment": segment})

    if not observations:
        return []

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for observation in observations:
        key = (
            str(observation.get("dimension") or ""),
            str(observation.get("direction") or ""),
            str(observation.get("segment") or ""),
        )
        grouped.setdefault(key, []).append(observation)

    rows = []
    min_samples = max(1, int(min_samples or 1))
    for (dimension, direction, segment), group_rows in grouped.items():
        close_values = [row["eod_r"] for row in group_rows if row.get("eod_r") is not None]
        mfe_values = [row["mfe_r"] for row in group_rows if row.get("mfe_r") is not None]
        mae_values = [row["mae_r"] for row in group_rows if row.get("mae_r") is not None]
        eod_move_pct_values = [
            row["eod_move_pct"] for row in group_rows if row.get("eod_move_pct") is not None
        ]
        mfe_pct_values = [row["mfe_pct"] for row in group_rows if row.get("mfe_pct") is not None]
        mae_pct_values = [row["mae_pct"] for row in group_rows if row.get("mae_pct") is not None]
        score_values = [row["score"] for row in group_rows if row.get("score") is not None]
        risk_values = [row["risk_per_share"] for row in group_rows if row.get("risk_per_share") is not None]
        sample_count = len(close_values)
        if sample_count <= 0:
            continue
        target_1r_rate = sum(1 for row in group_rows if row.get("target_1r_hit")) / float(len(group_rows))
        target_2r_rate = sum(1 for row in group_rows if row.get("target_2r_hit")) / float(len(group_rows))
        stop_rate = sum(1 for row in group_rows if row.get("stop_hit")) / float(len(group_rows))
        avg_close_r = sum(close_values) / float(len(close_values))
        median_close_r = float(pd.Series(close_values).median())
        positive_eod_rate = sum(1 for value in close_values if value > 0.0) / float(len(close_values))
        sample_weight = min(1.0, math.log1p(sample_count) / math.log1p(max(min_samples, 2)))
        edge_score = (
            (0.70 * avg_close_r)
            + (0.30 * median_close_r)
            + (0.50 * (positive_eod_rate - 0.50))
        ) * sample_weight
        if sample_count < min_samples:
            recommendation = "watch_more_samples"
        elif avg_close_r >= 0.25 and median_close_r >= 0.0 and positive_eod_rate >= 0.55:
            recommendation = "focus"
        elif avg_close_r <= -0.20 or positive_eod_rate <= 0.40:
            recommendation = "avoid"
        else:
            recommendation = "neutral"
        rows.append(
            {
                "dimension": dimension,
                "direction": direction,
                "segment": segment,
                "sample_count": sample_count,
                "avg_score": (sum(score_values) / float(len(score_values))) if score_values else None,
                "avg_risk_per_share": (sum(risk_values) / float(len(risk_values))) if risk_values else None,
                "avg_eod_r": avg_close_r,
                "median_eod_r": median_close_r,
                "avg_close_r": avg_close_r,
                "median_close_r": median_close_r,
                "avg_mfe_r": (sum(mfe_values) / float(len(mfe_values))) if mfe_values else None,
                "avg_mae_r": (sum(mae_values) / float(len(mae_values))) if mae_values else None,
                "avg_eod_move_pct": (
                    sum(eod_move_pct_values) / float(len(eod_move_pct_values))
                    if eod_move_pct_values
                    else None
                ),
                "avg_mfe_pct": (sum(mfe_pct_values) / float(len(mfe_pct_values))) if mfe_pct_values else None,
                "avg_mae_pct": (sum(mae_pct_values) / float(len(mae_pct_values))) if mae_pct_values else None,
                "positive_eod_rate": positive_eod_rate,
                "target_1r_rate": target_1r_rate,
                "target_2r_rate": target_2r_rate,
                "stop_rate": stop_rate,
                "edge_score": edge_score,
                "recommendation": recommendation,
                "example_symbols": ", ".join(
                    sorted({str(row.get("symbol") or "") for row in group_rows if row.get("symbol")})[:8]
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row.get("recommendation") != "focus",
            -float(row.get("edge_score", 0.0) or 0.0),
            -int(row.get("sample_count", 0) or 0),
            str(row.get("dimension") or ""),
            str(row.get("direction") or ""),
            str(row.get("segment") or ""),
        )
    )
    return rows


def _format_bounce_perf_r(value) -> str:
    numeric = _bounce_perf_float(value)
    return "n/a" if numeric is None else f"{numeric:+.2f}R"


def _format_bounce_perf_pct(value) -> str:
    numeric = _bounce_perf_float(value)
    return "n/a" if numeric is None else f"{numeric * 100:.0f}%"


def _format_bounce_price_pct(value) -> str:
    numeric = _bounce_perf_float(value)
    return "n/a" if numeric is None else f"{numeric:+.2f}%"


def write_intraday_bounce_performance_report(
    rows: list[dict],
    *,
    report_path: Path = INTRADAY_BOUNCE_PERFORMANCE_REPORT,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Intraday BounceBot performance",
        "=" * 80,
        f"Generated at {get_market_local_now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Primary metric: confirmed bounce EOD R from confirmation close to regular-session close.",
        f"EOD R is clipped to +/-{BOUNCE_PERFORMANCE_R_CLIP:.1f}R for ranking.",
        "",
    ]
    if not rows:
        lines.append("No confirmed EOD bounce outcomes were available.")
        report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return report_path

    def add_section(title: str, section_rows: list[dict]) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        if not section_rows:
            lines.append("None")
            lines.append("")
            return
        for idx, row in enumerate(section_rows[:12], start=1):
            lines.append(
                f"{idx:>2}. {row.get('direction')} {row.get('dimension')}={row.get('segment')} "
                f"samples={int(row.get('sample_count', 0) or 0)} "
                f"avgEOD={_format_bounce_perf_r(row.get('avg_eod_r', row.get('avg_close_r')))} "
                f"medianEOD={_format_bounce_perf_r(row.get('median_eod_r', row.get('median_close_r')))} "
                f"green={_format_bounce_perf_pct(row.get('positive_eod_rate'))} "
                f"avgMove={_format_bounce_price_pct(row.get('avg_eod_move_pct'))} "
                f"MFE={_format_bounce_perf_r(row.get('avg_mfe_r'))} "
                f"MAE={_format_bounce_perf_r(row.get('avg_mae_r'))} "
                f"1Rseen={_format_bounce_perf_pct(row.get('target_1r_rate'))} "
                f"stopSeen={_format_bounce_perf_pct(row.get('stop_rate'))} "
                f"rec={row.get('recommendation')}"
            )
            if row.get("example_symbols"):
                lines.append(f"    examples: {row.get('example_symbols')}")
        lines.append("")

    focus_rows = [
        row for row in rows
        if row.get("recommendation") == "focus"
    ]
    avoid_rows = [
        row for row in rows
        if row.get("recommendation") == "avoid"
    ]
    avoid_rows = sorted(
        avoid_rows,
        key=lambda row: (
            float(row.get("edge_score", 0.0) or 0.0),
            -int(row.get("sample_count", 0) or 0),
        ),
    )
    add_section("DT focus candidates", focus_rows)
    add_section("Weak DT cohorts / avoid", avoid_rows)
    add_section(
        "Best individual bounce types",
        [
            row for row in rows
            if row.get("dimension") == "bounce_type"
        ],
    )
    add_section(
        "Best bounce combos",
        [row for row in rows if row.get("dimension") == "bounce_combo"],
    )
    add_section(
        "Master AVWAP focus context",
        [
            row for row in rows
            if str(row.get("dimension") or "").startswith("master_avwap_")
        ],
    )
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def refresh_intraday_bounce_performance_report(
    *,
    performance_path: Path = INTRADAY_BOUNCE_PERFORMANCE_CSV,
    report_path: Path = INTRADAY_BOUNCE_PERFORMANCE_REPORT,
    min_samples: int = BOUNCE_PERFORMANCE_MIN_SAMPLES,
) -> tuple[Path, Path, int]:
    rows = build_intraday_bounce_performance_rows(min_samples=min_samples)
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(performance_path, index=False)
    written_report = write_intraday_bounce_performance_report(rows, report_path=report_path)
    return performance_path, written_report, len(rows)


def reset_log_files():
    try:
        TRADING_BOT_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        Path(TRADING_BOT_LOG_FILENAME).touch(exist_ok=True)
    except OSError as e:
        print(f"Could not prepare trading log {TRADING_BOT_LOG_FILENAME}: {e}")

    try:
        if os.path.exists(BOUNCE_LOG_FILENAME):
            os.remove(BOUNCE_LOG_FILENAME)
            print(f"Previous session bounce list deleted: {BOUNCE_LOG_FILENAME}")
        with open(BOUNCE_LOG_FILENAME, "w", encoding="utf-8"):
            pass
        print(f"Created fresh bounce list: {BOUNCE_LOG_FILENAME}")
    except Exception as e:
        print(f"Error resetting bounce list {BOUNCE_LOG_FILENAME}: {e}")


def configure_app_logging():
    if getattr(configure_app_logging, "_configured", False):
        return

    TRADING_BOT_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))
    handlers = [console_handler]
    try:
        file_handler = SafeRotatingFileHandler(
            TRADING_BOT_LOG_FILENAME,
            maxBytes=1_000_000,
            backupCount=APP_LOG_BACKUP_COUNT,
        )
    except OSError as exc:
        logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
        logging.getLogger().warning(f"File logging disabled for {TRADING_BOT_LOG_FILENAME}: {exc}")
    else:
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))
        handlers.append(file_handler)
        logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    logger = logging.getLogger()
    logger.addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi").addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi.client").addFilter(HistoricalDataFilter())
    for logger_name in YFINANCE_NOISY_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    configure_app_logging._configured = True


@dataclass(frozen=True)
class IbBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _parse_ib_bar_datetime(value):
    text = str(value).strip()
    for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _bars_to_ib(bars):
    ib_bars = []
    for bar in bars:
        dt = _parse_ib_bar_datetime(bar.get("time"))
        if dt is None:
            continue
        try:
            volume = float(bar.get("volume", 0.0) or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        ib_bars.append(
            IbBar(
                dt=dt,
                open=float(bar.get("open", 0.0)),
                high=float(bar.get("high", 0.0)),
                low=float(bar.get("low", 0.0)),
                close=float(bar.get("close", 0.0)),
                volume=volume,
            )
        )
    return ib_bars


def _dedupe_bars(bars):
    by_dt = {}
    for bar in bars:
        by_dt[bar.dt] = bar
    return sorted(by_dt.values(), key=lambda b: b.dt)


def _align_bars_with_map(symbol_bars, spy_by_dt):
    if not symbol_bars or not spy_by_dt:
        return [], []
    sym_by_dt = {bar.dt: bar for bar in symbol_bars}
    common = sorted(sym_by_dt.keys() & spy_by_dt.keys())
    if not common:
        return [], []
    return [sym_by_dt[dt] for dt in common], [spy_by_dt[dt] for dt in common]


def _aggregate_bars_timeframe(bars, timeframe_minutes, *, drop_partial_tail=True):
    if timeframe_minutes <= 5:
        return list(bars)
    expected = max(1, timeframe_minutes // 5)
    buckets = {}
    order = []
    counts = {}
    for bar in bars:
        dt = bar.dt
        market_open = get_market_session_open_naive(reference=dt)
        minutes_since = int((dt - market_open).total_seconds() // 60)
        if minutes_since < 0:
            bucket_minute = (dt.minute // timeframe_minutes) * timeframe_minutes
            bucket_start = dt.replace(minute=bucket_minute, second=0, microsecond=0)
        else:
            bucket_index = minutes_since // timeframe_minutes
            bucket_start = market_open + timedelta(minutes=bucket_index * timeframe_minutes)
        if bucket_start not in buckets:
            buckets[bucket_start] = IbBar(
                dt=bucket_start,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
            )
            counts[bucket_start] = 1
            order.append(bucket_start)
        else:
            agg = buckets[bucket_start]
            buckets[bucket_start] = IbBar(
                dt=agg.dt,
                open=agg.open,
                high=max(agg.high, bar.high),
                low=min(agg.low, bar.low),
                close=bar.close,
            )
            counts[bucket_start] += 1

    if not order:
        return []

    result = []
    last_bucket = order[-1]
    for bucket in order:
        if drop_partial_tail and bucket == last_bucket and counts.get(bucket, 0) < expected:
            continue
        result.append(buckets[bucket])
    return result


def _ib_bar_to_candle_dict(bar):
    return {
        "time": bar.dt.strftime("%Y%m%d  %H:%M:%S"),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
    }


def detect_mid_earnings_h1_bounce(
    h1_bars,
    direction,
    atr20,
    reference_date=None,
    allowed_level_keys=None,
):
    bars = _dedupe_bars(h1_bars or [])
    if len(bars) < H1_MID_EARNINGS_MIN_BARS:
        return None

    normalized_direction = str(direction or "").strip().lower()
    if normalized_direction not in {"long", "short"}:
        return None

    try:
        atr_value = float(atr20)
    except (TypeError, ValueError):
        return None
    if atr_value <= 0:
        return None

    frame = pd.DataFrame(
        {
            "dt": [bar.dt for bar in bars],
            "open": [float(bar.open) for bar in bars],
            "high": [float(bar.high) for bar in bars],
            "low": [float(bar.low) for bar in bars],
            "close": [float(bar.close) for bar in bars],
        }
    ).sort_values("dt")
    frame["h1_ema_15"] = frame["close"].ewm(span=15, adjust=False).mean()
    frame["h1_sma_20"] = frame["close"].rolling(20).mean()

    touch_row = frame.iloc[-2]
    confirm_row = frame.iloc[-1]
    confirm_dt = confirm_row["dt"]
    if reference_date is not None and confirm_dt.date() != reference_date:
        return None

    threshold = THRESHOLD_MULTIPLIER * atr_value
    allowed_level_keys = (
        {str(level_key) for level_key in allowed_level_keys}
        if allowed_level_keys is not None
        else None
    )
    levels = {}
    triggered_levels = []
    labels = []
    for level_key, level_label in H1_MID_EARNINGS_BOUNCE_LEVELS.items():
        if allowed_level_keys is not None and level_key not in allowed_level_keys:
            continue
        level_value = touch_row.get(level_key)
        if pd.isna(level_value):
            continue
        level_value = float(level_value)
        if normalized_direction == "long":
            touched = abs(float(touch_row["low"]) - level_value) <= threshold
            confirmed = float(confirm_row["close"]) > level_value
        else:
            touched = abs(float(touch_row["high"]) - level_value) <= threshold
            confirmed = float(confirm_row["close"]) < level_value
        if touched and confirmed:
            levels[level_key] = level_value
            triggered_levels.append(level_key)
            labels.append(level_label)

    if not triggered_levels:
        return None

    touch_bar = IbBar(
        dt=touch_row["dt"],
        open=float(touch_row["open"]),
        high=float(touch_row["high"]),
        low=float(touch_row["low"]),
        close=float(touch_row["close"]),
    )
    confirm_bar = IbBar(
        dt=confirm_row["dt"],
        open=float(confirm_row["open"]),
        high=float(confirm_row["high"]),
        low=float(confirm_row["low"]),
        close=float(confirm_row["close"]),
    )
    close_side = "above" if normalized_direction == "long" else "below"
    touch_side = "low" if normalized_direction == "long" else "high"
    joined_labels = "/".join(labels)
    return {
        "levels": levels,
        "triggered_levels": triggered_levels,
        "candle": _ib_bar_to_candle_dict(touch_bar),
        "confirmation_candle": _ib_bar_to_candle_dict(confirm_bar),
        "confirm_immediately": True,
        "max_confirmation_candles": 1,
        "threshold": threshold,
        "reason": (
            f"H1 confirmation candle closed {close_side} {joined_labels} "
            f"after prior H1 {touch_side} touched the level."
        ),
    }


def _wilder_atr_last(bars, length):
    if len(bars) < length + 1:
        return None
    true_ranges = []
    for idx in range(1, len(bars)):
        high = bars[idx].high
        low = bars[idx].low
        prev_close = bars[idx - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < length:
        return None
    atr = sum(true_ranges[:length]) / float(length)
    for tr in true_ranges[length:]:
        atr = ((atr * (length - 1)) + tr) / float(length)
    return atr if atr > 0 else None


def classify_h1_candle_color(close, ema15, sma20):
    """Trader's hourly regime color for one closed H1 candle.

    Uptrend structure (EMA15 > SMA20): green above the EMA15, orange between
    the averages, yellow below both. Downtrend structure: red below the EMA15,
    blue when the close reclaims both averages; a close between them has no
    color (no signal either way).
    """
    if close is None or ema15 is None or sma20 is None:
        return None
    if ema15 > sma20:
        if close > ema15:
            return "green"
        if close > sma20:
            return "orange"
        return "yellow"
    if close > ema15 and close > sma20:
        return "blue"
    if close < ema15:
        return "red"
    return None


def h1_candle_colors(h1_bars):
    """Color per H1 bar, aligned 1:1; None until the SMA20 has 20 bars."""
    bars = list(h1_bars or [])
    ema15 = _ema_series(bars, 15)
    closes = [bar.close for bar in bars]
    colors = []
    for idx, bar in enumerate(bars):
        if idx < 19:
            colors.append(None)
            continue
        sma20 = sum(closes[idx - 19 : idx + 1]) / 20.0
        colors.append(classify_h1_candle_color(bar.close, ema15[idx], sma20))
    return colors


def _closed_h1_bars(bars_5m):
    """Aggregate 5m bars to H1 and drop the still-forming final bucket.

    An hourly bucket is closed once the cached 5m bars reach its end - either
    bucket start + 60 minutes or the session close. Uses a time-based rule
    instead of the aggregator's bar-count tail rule so the short final RTH
    candle (a 30-minute hour) still counts once the bell rings.
    """
    bars = list(bars_5m or [])
    if not bars:
        return []
    h1 = _aggregate_bars_timeframe(bars, 60, drop_partial_tail=False)
    if not h1:
        return []
    bucket_end = h1[-1].dt + timedelta(minutes=60)
    try:
        session_close = get_market_session_close_naive(reference=h1[-1].dt)
        if session_close > h1[-1].dt:
            bucket_end = min(bucket_end, session_close)
    except Exception:
        pass
    latest_data_end = bars[-1].dt + timedelta(minutes=5)
    if latest_data_end < bucket_end:
        h1 = h1[:-1]
    return h1


def detect_h1_color_signals(h1_bars, side):
    """Signal hits on the LAST closed H1 candle for the color strategies.

    Longs (strong names): green-regime 10-EMA touch-and-recover candles and
    blue reclaim candles right after a red one. Shorts (weak names): a green
    candle followed by a yellow breakdown through both averages.
    """
    bars = list(h1_bars or [])
    if len(bars) < H1_COLOR_MIN_BARS:
        return []
    normalized_side = str(side or "").strip().lower()
    if normalized_side not in {"long", "short"}:
        return []
    colors = h1_candle_colors(bars)
    color, prev_color = colors[-1], colors[-2]
    if not color or not prev_color:
        return []
    bar = bars[-1]
    closes = [item.close for item in bars]
    ema10 = _ema_series(bars, 10)[-1]
    ema15 = _ema_series(bars, 15)[-1]
    sma20 = sum(closes[-20:]) / 20.0
    hits = []
    if normalized_side == "long":
        atr = _wilder_atr_last(bars, H1_COLOR_ATR_LENGTH)
        if color == "green" and atr and bar.open > ema10 and bar.close > ema10:
            tagged = bar.low >= ema10 and abs(bar.low - ema10) <= H1_EMA10_BOUNCE_TOUCH_ATR * atr
            pierced = bar.low < ema10
            if tagged or pierced:
                hits.append(
                    {
                        "type": H1_EMA10_BOUNCE_TYPE,
                        "level": ema10,
                        "signal_bar": bar,
                        "color": color,
                        "prev_color": prev_color,
                        "detail": (
                            f"green H1 {'pierced' if pierced else 'tagged'} the 10-EMA "
                            f"{ema10:.2f} and closed back above it"
                        ),
                    }
                )
        if prev_color == "red" and color == "blue":
            hits.append(
                {
                    "type": H1_BLUE_AFTER_RED_TYPE,
                    "level": ema15,
                    "signal_bar": bar,
                    "color": color,
                    "prev_color": prev_color,
                    "detail": (
                        f"blue reclaim candle after a red H1: close {bar.close:.2f} back above "
                        f"the 15-EMA {ema15:.2f} and 20-SMA {sma20:.2f}"
                    ),
                }
            )
    elif prev_color == "green" and color == "yellow":
        hits.append(
            {
                "type": H1_GREEN_TO_YELLOW_TYPE,
                "level": sma20,
                "signal_bar": bar,
                "color": color,
                "prev_color": prev_color,
                "detail": (
                    f"green H1 failed straight to yellow: close {bar.close:.2f} broke below "
                    f"the 15-EMA {ema15:.2f} and 20-SMA {sma20:.2f}"
                ),
            }
        )
    return hits


def real_relative_strength(symbol_bars, spy_bars, length=RRS_LENGTH):
    if not symbol_bars or not spy_bars:
        return None, None
    min_bars = length + 2
    if len(symbol_bars) < min_bars or len(spy_bars) < min_bars:
        return None, None
    sym_move = symbol_bars[-1].close - symbol_bars[-1 - length].close
    spy_move = spy_bars[-1].close - spy_bars[-1 - length].close
    sym_atr = _wilder_atr_last(symbol_bars[:-1], length)
    spy_atr = _wilder_atr_last(spy_bars[:-1], length)
    if sym_atr is None or spy_atr is None or sym_atr == 0 or spy_atr == 0:
        return None, None
    power_index = spy_move / spy_atr
    rrs = (sym_move - (power_index * sym_atr)) / sym_atr
    return rrs, power_index


def _spy_vwap_regime_stats(today_bars, prev_close):
    """The VWAP-position regime read plus the measurements behind it, or None.

    Returns {vwap, stdev, above_band_frac, below_band_frac, last_close,
    classification} where classification is one of the four regime keys or
    None (mixed tape - the caller falls back to the day% rule). Returns None
    outright when the session is too young or volume data is missing.
    """
    if not prev_close or len(today_bars) < MARKET_REGIME_VWAP_MIN_BARS:
        return None
    if sum(bar.volume for bar in today_bars) <= 0:
        return None

    above_upper = 0
    below_lower = 0
    cum_vol = 0.0
    cum_pv = 0.0
    cum_pv2 = 0.0
    vwap = None
    stdev = 0.0
    for bar in today_bars:
        typical = (bar.high + bar.low + bar.close) / 3.0
        volume = max(0.0, float(bar.volume))
        cum_vol += volume
        cum_pv += typical * volume
        cum_pv2 += typical * typical * volume
        if cum_vol <= 0:
            return None
        vwap = cum_pv / cum_vol
        variance = max(0.0, (cum_pv2 / cum_vol) - vwap * vwap)
        stdev = math.sqrt(variance)
        if bar.close > vwap + stdev:
            above_upper += 1
        elif bar.close < vwap - stdev:
            below_lower += 1

    fraction = float(MARKET_REGIME_VWAP_BAND_FRACTION)
    total = len(today_bars)
    last_close = today_bars[-1].close
    classification = None
    if above_upper / total >= fraction:
        classification = "bullish_strong"
    elif below_lower / total >= fraction:
        classification = "bearish_strong"
    elif last_close > vwap and last_close > prev_close:
        classification = "bullish_weak"
    elif last_close < vwap and last_close < prev_close:
        classification = "bearish_weak"
    return {
        "vwap": vwap,
        "stdev": stdev,
        "above_band_frac": above_upper / total,
        "below_band_frac": below_lower / total,
        "last_close": last_close,
        "classification": classification,
    }


def _classify_spy_vwap_regime(today_bars, prev_close):
    """VWAP-position regime read from SPY's session 5m bars, or None.

    bullish_strong: closes held above VWAP+1stdev for most of the day.
    bullish_weak:   above VWAP and green on the day, but under the band.
    Inverted for bearish. Returns None (caller falls back to the day% rule)
    when the session is too young or volume data is missing.
    """
    stats = _spy_vwap_regime_stats(today_bars, prev_close)
    return stats["classification"] if stats else None


def _session_structure_report(today_df, direction, ref_atr):
    """(ok, reason) for the trend-then-simple-retest session gate.

    The tape we want under a bounce (longs): the stock is moving higher, then
    makes a simple retest of a level. So: a real advance off the open, still
    intact, with at most BOUNCE_MAX_SIGNIFICANT_PULLBACKS zigzag swings
    against the move (the live retest included). Shorts run on the mirrored
    tape. Too little data to judge passes through - the per-level checks
    still gate those.
    """
    if today_df is None or len(today_df) < BOUNCE_STRUCTURE_MIN_BARS:
        return True, "not enough bars to judge structure"
    if not ref_atr or ref_atr <= 0:
        return True, "no intraday ATR reference"

    if direction == "long":
        eff_high = today_df["high"].astype(float).tolist()
        eff_low = today_df["low"].astype(float).tolist()
        eff_closes = today_df["close"].astype(float).tolist()
        eff_open = float(today_df["open"].iloc[0])
    else:
        eff_high = (-today_df["low"].astype(float)).tolist()
        eff_low = (-today_df["high"].astype(float)).tolist()
        eff_closes = (-today_df["close"].astype(float)).tolist()
        eff_open = -float(today_df["open"].iloc[0])

    advance = max(eff_high) - eff_open
    if advance < BOUNCE_MIN_ADVANCE_ATR * ref_atr:
        return False, (
            f"no directional leg / compressed: advance {advance:.2f} < "
            f"{BOUNCE_MIN_ADVANCE_ATR} x intraday ATR ({ref_atr:.2f})"
        )
    if eff_closes[-1] <= eff_open:
        return False, "day's move has been given back"

    depth_min = max(
        BOUNCE_PULLBACK_MIN_ATR * ref_atr,
        BOUNCE_PULLBACK_MIN_ADVANCE_FRAC * advance,
    )
    # Zigzag on CLOSES against wick pivots: one wide candle is not a pullback;
    # a swing only counts once price *closes* depth_min away from the pivot.
    pullbacks = 0
    trend_with = True
    pivot_high = eff_high[0]
    pivot_low = eff_low[0]
    for high, low, close in zip(eff_high, eff_low, eff_closes):
        if trend_with:
            if high > pivot_high:
                pivot_high = high
            if pivot_high - close >= depth_min:
                trend_with = False
                pivot_low = low
                pullbacks += 1
        else:
            if low < pivot_low:
                pivot_low = low
            if close - pivot_low >= depth_min:
                trend_with = True
                pivot_high = high
    if pullbacks > BOUNCE_MAX_SIGNIFICANT_PULLBACKS:
        return False, (
            f"choppy tape: {pullbacks} pullbacks of >= {depth_min:.2f} "
            f"(max {BOUNCE_MAX_SIGNIFICANT_PULLBACKS})"
        )
    return True, f"advance {advance:.2f}, pullbacks {pullbacks}"


def _ema_series(bars, length):
    """Close-based EMA per bar (ewm span semantics), aligned 1:1 with bars."""
    if not bars:
        return []
    alpha = 2.0 / (length + 1.0)
    ema = bars[0].close
    values = [ema]
    for bar in bars[1:]:
        ema = alpha * bar.close + (1.0 - alpha) * ema
        values.append(ema)
    return values


##########################################
# Request Queue Class
##########################################
class RequestQueue:
    def __init__(self):
        self.queue = []
        self.active_requests = 0
        self.lock = threading.Lock()

    def add_request(self, func, *args):
        self.queue.append((func, args))
        self.process_queue()

    def process_queue(self):
        with self.lock:
            while self.queue and self.active_requests < MAX_CONCURRENT_REQUESTS:
                func, args = self.queue.pop(0)
                self.active_requests += 1
                threading.Thread(target=self._execute_request, args=(func, args)).start()

    def _execute_request(self, func, args):
        try:
            func(*args)
            time.sleep(REQUEST_DELAY)
        finally:
            with self.lock:
                self.active_requests -= 1
                self.process_queue()

##########################################
# BounceBot Class with GUI callback
##########################################
class BounceBot(EWrapper, EClient):
    def __init__(self, gui_callback=None, start_scanning_enabled=True):
        EClient.__init__(self, self)
        self.connection_status = False
        self.reqIdCount = 1000
        self.data_lock = threading.Lock()
        self.reqId_lock = threading.Lock()
        self.connect_lock = threading.Lock()
        self.ib_host = None
        self.ib_port = None
        self.ib_client_id_base = None
        self.ib_client_id = None
        self.ib_client_id_offset = 0
        self.client_id_conflict = False
        self.client_id_namespace = max(1, os.getpid() % 1000) * 1000
        self.api_thread = None
        self.strategy_thread = None
        # Cooperative shutdown: run_strategy exits, sleeps abort, and
        # ensure_connected stops retrying once this is set (plan.md Packet A).
        self._stop_event = threading.Event()

        self.data = {}
        self.data_ready_events = {}
        self.reqid_to_symbol = {}
        self.invalid_security_symbols = set()

        # IB pacing backoff state (see IB_PACING_* constants).
        self.pacing_lock = threading.Lock()
        self.pacing_backoff_seconds = 0.0
        self.pacing_backoff_until = 0.0
        self.last_pacing_violation_at = 0.0

        # After-close learning refresh: once per session, so tiers/mutes work
        # from today's outcomes even when the bot never restarts.
        self._learning_refresh_date = None

        self.longs = read_tickers(LONGS_FILENAME)
        self.shorts = read_tickers(SHORTS_FILENAME)
        self.auto_longs = read_tickers(AUTO_LONGS_FILENAME)
        self.auto_shorts = read_tickers(AUTO_SHORTS_FILENAME)
        self.atr_cache = {}
        self.symbol_metrics = {}  # Store precomputed VWAP and level metrics

        self.alerted_symbols = set()
        self.bounce_candidates = {}  # Track candidate bounces
        self.pending_bounce_outcomes = self._load_pending_bounce_outcomes()
        self.logged_near_miss_events = set()
        self.request_queue = RequestQueue()
        self.gui_callback = gui_callback  # Callback to update the GUI

        # Add this to track which symbols we've already warned about
        self.warned_symbols = set()

        # RRS settings (thread-safe for GUI updates)
        self.rrs_lock = threading.Lock()
        self.rrs_threshold = RRS_DEFAULT_THRESHOLD
        self.rrs_length = RRS_LENGTH
        self.rrs_timeframe_key = "5m"
        self.rrs_bar_size = RRS_TIMEFRAMES[self.rrs_timeframe_key]["bar_size"]
        self.rrs_duration = RRS_TIMEFRAMES[self.rrs_timeframe_key]["duration"]

        # Cache latest 5-minute bars per symbol for reuse (RRS)
        self.latest_bars = {}
        self.latest_scan_extremes = {}
        self.latest_group_extremes = {}
        self.market_environment = "bullish_strong"
        self.market_environment_lock = threading.Lock()
        self.market_environment_user_override = False
        self._regime_pause_state = None
        self._regime_pause_alert_log = None
        self._regime_pause_observations = None
        self._orb_break_state = None
        self._ema8_grind_state = None
        self._h1_color_state = None
        self.latest_rrs_payload = None
        self.earnings_reaction_filter_cache = {}

        self.bounce_type_toggles = dict(BOUNCE_TYPE_DEFAULTS)
        self.scanning_enabled = bool(start_scanning_enabled)
        self.scanning_lock = threading.Lock()
        self._scan_cycle_index = 0

        self.master_avwap_events = {}
        self.master_avwap_last_scan_date = None
        self._master_avwap_events_cache_key = None
        self.emitted_master_avwap_events = set()
        self.master_avwap_focus_map = {}
        self.human_focus_map = {"long": set(), "short": set()}
        self.master_avwap_second_stdev_cross_map = {}
        self.master_avwap_d1_upgrade_alerts = {}
        self.master_avwap_d1_watchlist = {}
        self.emitted_master_avwap_focus_alerts = set()
        self.emitted_master_avwap_second_stdev_alerts = set()
        self.emitted_master_avwap_d1_flags = set()
        self.master_avwap_d1_flags_primed_date = None
        self.emitted_h1_mid_earnings_bounce_alerts = set()

        self.sector_etf_map = load_sector_etf_map()
        self.industry_map_data = _load_industry_etf_map_file()
        self.symbol_classification_cache = {}
        self._load_symbol_classification_cache()

        # Learning-loop maintenance runs off-thread so startup stays instant:
        # refresh the alert-time learning state if stale (daily) and compact
        # the candidates CSV when the JSON-blob columns have bloated it.
        if str(os.environ.get("TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE") or "").strip() != "1":
            threading.Thread(target=self._run_bounce_learning_maintenance, daemon=True).start()

    def _run_bounce_learning_maintenance(self):
        try:
            from bounce_bot_lib.learning import (
                compact_bounce_candidates_csv,
                refresh_bounce_learning_if_stale,
            )

            compact_bounce_candidates_csv(
                INTRADAY_BOUNCE_CANDIDATES_CSV,
                max_age_days=365,
                min_bytes_to_bother=150_000_000,
            )
            if refresh_bounce_learning_if_stale():
                logging.info("Bounce learning state refreshed at startup.")
        except Exception as exc:  # maintenance must never break the bot
            logging.warning("Bounce learning maintenance failed: %s", exc)

    def _load_pending_bounce_outcomes(self):
        if not INTRADAY_BOUNCE_OUTCOME_STATE_JSON.exists():
            return {}
        try:
            payload = json.loads(INTRADAY_BOUNCE_OUTCOME_STATE_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
        pending = payload.get("pending", {}) if isinstance(payload, dict) else {}
        return pending if isinstance(pending, dict) else {}

    def _save_pending_bounce_outcomes(self):
        try:
            INTRADAY_BOUNCE_OUTCOME_STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": BOUNCE_LEARNING_SCHEMA_VERSION,
                "updated_at": get_market_local_now().isoformat(timespec="seconds"),
                "pending": self.pending_bounce_outcomes,
            }
            INTRADAY_BOUNCE_OUTCOME_STATE_JSON.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logging.debug(f"Failed saving pending bounce outcome state: {exc}")

    def _learning_csv_header(self, path, fieldnames):
        if not path.exists() or path.stat().st_size == 0:
            return list(fieldnames)
        try:
            with path.open("r", newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                existing_header = next(reader, [])
        except Exception:
            return list(fieldnames)
        existing_header = [str(item or "").strip() for item in existing_header if str(item or "").strip()]
        if not existing_header:
            return list(fieldnames)
        missing = [field for field in fieldnames if field not in existing_header]
        if not missing:
            return existing_header

        widened_header = existing_header + missing
        try:
            with path.open("r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile, restkey="_extra")
                existing_rows = list(reader)
            with path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=widened_header, extrasaction="ignore")
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({key: existing_row.get(key, "") for key in widened_header})
        except Exception as exc:
            logging.debug(f"Failed widening learning CSV header for {path}: {exc}")
            return existing_header
        return widened_header

    def _append_learning_row(self, path, fieldnames, row):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists() or path.stat().st_size == 0
            writer_fieldnames = self._learning_csv_header(path, fieldnames)
            with path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=writer_fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in writer_fieldnames})
        except Exception as exc:
            logging.debug(f"Failed writing learning row to {path}: {exc}")

    def _json_for_learning(self, value):
        def sanitize(item):
            if isinstance(item, pd.Series):
                return sanitize(item.to_dict())
            if isinstance(item, dict):
                return {str(key): sanitize(val) for key, val in item.items()}
            if isinstance(item, (list, tuple, set)):
                return [sanitize(val) for val in item]
            if isinstance(item, (datetime, date)):
                return item.isoformat()
            try:
                if pd.isna(item):
                    return None
            except Exception:
                pass
            if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                return None
            return item

        # Use `default=str` so unknown/non-JSON-native objects are converted
        # to a string representation instead of raising TypeError.
        return json.dumps(sanitize(value), ensure_ascii=True, sort_keys=True, default=str)

    def _to_float_or_blank(self, value):
        try:
            if value is None or pd.isna(value):
                return ""
        except Exception:
            if value is None:
                return ""
        try:
            return float(value)
        except (TypeError, ValueError):
            return ""

    def _parse_bar_time(self, value):
        if isinstance(value, datetime):
            return value
        parsed = pd.to_datetime(str(value or ""), format="%Y%m%d  %H:%M:%S", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(str(value or ""), errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()

    def _extract_rrs_entry(self, payload, key, symbol):
        if not isinstance(payload, dict):
            return {}
        symbol = str(symbol or "").strip().upper()
        for row in payload.get(key, []) or []:
            try:
                signal, row_symbol, rrs_value, power_index = row[:4]
            except Exception:
                continue
            if str(row_symbol or "").strip().upper() == symbol:
                return {
                    "signal": signal,
                    "rrs": self._to_float_or_blank(rrs_value),
                    "power_index": self._to_float_or_blank(power_index),
                }
        return {}

    def _build_bounce_context_snapshot(self, symbol, direction):
        payload = self.latest_rrs_payload if isinstance(self.latest_rrs_payload, dict) else {}
        classification = self.symbol_classification_cache.get(str(symbol or "").strip().upper(), {})
        sector_key = classification.get("sectorKey", "") if isinstance(classification, dict) else ""
        industry_key = classification.get("industryKey", "") if isinstance(classification, dict) else ""
        sector_etf = resolve_sector_etf(sector_key, self.sector_etf_map) if sector_key else ""
        industry_etf = resolve_industry_ref_etf(industry_key, sector_key) if industry_key else ""
        spy_rrs = self._extract_rrs_entry(payload, "results", symbol)
        if not spy_rrs:
            # ``results`` only carries threshold-crossers; fall back to the
            # full-universe map so learning rows always capture the RRS value.
            rrs_all = payload.get("rrs_all") or {}
            cached = rrs_all.get(str(symbol or "").strip().upper())
            if cached:
                rrs_value, power_index = cached
                spy_rrs = {
                    "signal": "",
                    "rrs": self._to_float_or_blank(rrs_value),
                    "power_index": self._to_float_or_blank(power_index),
                }
        sector_rrs = self._extract_rrs_entry(payload, "results_sector", symbol)
        industry_rrs = self._extract_rrs_entry(payload, "results_industry", symbol)
        symbol_context = {}
        for entry in payload.get("symbol_context", []) or []:
            if str(entry.get("symbol") or "").strip().upper() == str(symbol or "").strip().upper():
                symbol_context = dict(entry)
                break
        return {
            "rrs_timeframe": payload.get("timeframe_key", ""),
            "rrs_spy": spy_rrs.get("rrs", ""),
            "rrs_spy_signal": spy_rrs.get("signal", ""),
            "rrs_sector": sector_rrs.get("rrs", ""),
            "rrs_sector_signal": sector_rrs.get("signal", ""),
            "rrs_industry": industry_rrs.get("rrs", ""),
            "rrs_industry_signal": industry_rrs.get("signal", ""),
            "sector": classification.get("sector", "") if isinstance(classification, dict) else "",
            "industry": classification.get("industry", "") if isinstance(classification, dict) else "",
            "sector_etf": sector_etf,
            "industry_etf": industry_etf,
            "market_environment": self.get_market_environment(),
            "watchlist_bias": self.get_symbol_direction(symbol),
            "symbol_context": symbol_context,
        }

    def _evaluate_bounce_alert_quality(self, direction, levels, event_row):
        """Tier + mute verdict for a confirmed bounce, from the learning state."""
        try:
            from bounce_bot_lib.learning import (
                evaluate_bounce_quality,
                load_bounce_learning_state,
                time_bucket_for,
            )

            row = event_row if isinstance(event_row, dict) else {}
            bounce_type_keys = _bounce_type_keys_from_levels(levels or {})
            return evaluate_bounce_quality(
                load_bounce_learning_state(),
                direction=direction,
                bounce_types=bounce_type_keys,
                time_bucket=time_bucket_for(_bounce_quality_time(row) or get_market_local_now()),
                market_environment=str(row.get("market_environment") or ""),
                priority_bucket=str(row.get("master_avwap_priority_bucket") or ""),
                focus_label=str(row.get("master_avwap_focus_label") or ""),
                # The best measured segments live in dimensions the composite
                # ignores; matching any PROVEN one flags the alert live.
                bounce_combo="+".join(bounce_type_keys),
                setup_family=str(row.get("master_avwap_setup_family") or ""),
                swing_traits=_split_delimited_text(row.get("master_avwap_swing_traits")),
            )
        except Exception as exc:  # learning must never block an alert
            logging.warning("Bounce learning evaluation failed (alerting anyway): %s", exc)
            return {"tier": "B", "muted": False, "mute_reasons": [], "reasons": []}

    def _measured_exit_suffix(self, direction, levels):
        """Tracker-measured exit stats for this bounce type ("" when unproven).

        Exits are the measured leak (MFE 2-3R vs 0.3-0.7R closed), so alerts
        carry the evidence. Never blocks an alert.
        """
        try:
            from bounce_bot_lib.learning import load_bounce_learning_state, measured_exit_note

            return measured_exit_note(
                load_bounce_learning_state(),
                direction=direction,
                bounce_types=_bounce_type_keys_from_levels(levels or {}),
            )
        except Exception as exc:
            logging.debug("Measured exit note unavailable: %s", exc)
            return ""

    def _make_bounce_event_id(self, symbol, direction, bounce_candle, levels):
        candle_time = ""
        if isinstance(bounce_candle, dict):
            candle_time = str(bounce_candle.get("time") or "")
        level_key = "-".join(sorted(str(key) for key in (levels or {}).keys()))
        raw = f"{symbol}|{direction}|{candle_time}|{level_key}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")

    def _build_bounce_trade_plan(self, direction, levels, bounce_candle, current_candle=None):
        candle = current_candle if current_candle is not None else bounce_candle
        if isinstance(candle, pd.Series):
            candle = candle.to_dict()
        if isinstance(bounce_candle, pd.Series):
            bounce_candle = bounce_candle.to_dict()
        candle = candle if isinstance(candle, dict) else {}
        bounce_candle = bounce_candle if isinstance(bounce_candle, dict) else {}
        entry_price = self._to_float_or_blank(candle.get("close"))
        if entry_price == "":
            entry_price = self._to_float_or_blank(bounce_candle.get("close"))
        if direction == "long":
            stop_price = self._to_float_or_blank(bounce_candle.get("low"))
            entry_trigger = self._to_float_or_blank(bounce_candle.get("high"))
            risk = float(entry_price) - float(stop_price) if entry_price != "" and stop_price != "" else ""
            target_1r = float(entry_price) + risk if risk != "" and risk > 0 else ""
            target_2r = float(entry_price) + (2 * risk) if risk != "" and risk > 0 else ""
        else:
            stop_price = self._to_float_or_blank(bounce_candle.get("high"))
            entry_trigger = self._to_float_or_blank(bounce_candle.get("low"))
            risk = float(stop_price) - float(entry_price) if entry_price != "" and stop_price != "" else ""
            target_1r = float(entry_price) - risk if risk != "" and risk > 0 else ""
            target_2r = float(entry_price) - (2 * risk) if risk != "" and risk > 0 else ""
        if risk != "" and risk <= 0:
            risk = ""
            target_1r = ""
            target_2r = ""
        return {
            "entry_trigger": entry_trigger,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "risk_per_share": risk,
            "target_1r": target_1r,
            "target_2r": target_2r,
        }

    def _score_bounce_candidate_snapshot(self, direction, levels, context):
        score = 10 * len(levels or {})
        level_names = set((levels or {}).keys())
        if "vwap_eod_confluence" in level_names:
            score += 14
        if "impulse_retest_vwap_eod" in level_names:
            score += 18
        if any(name in level_names for name in ("prev_day_high", "prev_day_low")):
            score += 8
        rrs_spy = context.get("rrs_spy")
        try:
            rrs_value = float(rrs_spy)
            if direction == "long" and rrs_value > 0:
                score += min(12, max(0, rrs_value * 2))
            elif direction == "short" and rrs_value < 0:
                score += min(12, max(0, abs(rrs_value) * 2))
        except (TypeError, ValueError):
            pass
        if context.get("rrs_sector") != "":
            score += 4
        try:
            industry_rrs = float(context.get("rrs_industry"))
            if direction == "long" and industry_rrs > 0:
                score += min(BOUNCE_INDUSTRY_RRS_BONUS_CAP, max(4, industry_rrs * 2))
            elif direction == "short" and industry_rrs < 0:
                score += min(BOUNCE_INDUSTRY_RRS_BONUS_CAP, max(4, abs(industry_rrs) * 2))
        except (TypeError, ValueError):
            pass
        return round(float(score), 2)

    def _master_avwap_swing_trait_tags(self, focus_entry):
        if not isinstance(focus_entry, dict):
            return []

        trait_specs = [
            ("preferred_swing_focus", "preferred_swing_focus"),
            ("top_pattern_watch", "top_pattern_watch"),
            ("top_pattern_entry", "top_pattern_entry"),
            ("post_earnings_active", "post_earnings_active"),
            ("post_earnings_break_intraday", "post_earnings_break_intraday"),
            ("post_earnings_break_close", "post_earnings_break_close"),
            ("mid_earnings_active_second_stdev_hold", "mid_earnings_2nd_stdev_hold"),
            ("sma_breakout_watch", "sma_breakout_watch"),
            ("sma_breakout_confirmed", "sma_breakout_confirmed"),
            ("retest_followthrough", "retest_followthrough"),
            ("breakout_5d", "breakout_5d"),
            ("previous_day_range_break", "previous_day_range_break"),
            ("trendline_break_recent", "trendline_break_recent"),
            ("extreme_move_watch", "extreme_move_watch"),
        ]
        tags = [
            tag
            for key, tag in trait_specs
            if bool(focus_entry.get(key))
        ]
        bucket = str(focus_entry.get("priority_bucket") or "").strip().lower()
        if bucket in {"favorite_setup", "near_favorite_zone"}:
            tags.append(bucket)
        setup_family = str(focus_entry.get("setup_family") or "").strip().lower()
        if setup_family:
            tags.append(f"family:{setup_family}")

        seen = set()
        deduped = []
        for tag in tags:
            if tag in seen:
                continue
            seen.add(tag)
            deduped.append(tag)
        return deduped

    def _merge_bounce_candidate_learning_levels(self, candidate_info, learning_info):
        if not isinstance(candidate_info, dict) or not isinstance(learning_info, dict):
            return candidate_info

        merged = dict(candidate_info)
        merged_levels = dict(learning_info.get("levels") or {})
        merged_levels.update(candidate_info.get("levels") or {})

        merged_triggers = []
        for source in (candidate_info, learning_info):
            for level in source.get("triggered_levels") or []:
                level = str(level or "").strip()
                if level and level not in merged_triggers:
                    merged_triggers.append(level)

        merged["levels"] = merged_levels
        merged["triggered_levels"] = merged_triggers
        merged["learning_bounce_types"] = _bounce_type_keys_from_levels(merged_levels)
        return merged

    def _log_bounce_candidate_event(
        self,
        event_type,
        symbol,
        direction,
        levels,
        bounce_candle,
        current_candle=None,
        threshold="",
        reason="",
        candidate_id="",
        candles_waited=0,
    ):
        if isinstance(current_candle, pd.Series):
            current_candle = current_candle.to_dict()
        if isinstance(bounce_candle, pd.Series):
            bounce_candle = bounce_candle.to_dict()
        bounce_candle = bounce_candle if isinstance(bounce_candle, dict) else {}
        current_candle = current_candle if isinstance(current_candle, dict) else {}
        event_id = candidate_id or self._make_bounce_event_id(symbol, direction, bounce_candle, levels)
        trade_dt = self._parse_bar_time(current_candle.get("time") or bounce_candle.get("time"))
        trade_date = trade_dt.date().isoformat() if trade_dt else get_market_local_now().date().isoformat()
        context = self._build_bounce_context_snapshot(symbol, direction)
        plan = self._build_bounce_trade_plan(direction, levels, bounce_candle, current_candle or None)
        atr = self.atr_cache.get(symbol)
        metrics = self.symbol_metrics.get(symbol, {})
        focus_entry = getattr(self, "master_avwap_focus_map", {}).get(symbol)
        focus_label = self._describe_master_avwap_focus(focus_entry) if isinstance(focus_entry, dict) else ""
        swing_traits = self._master_avwap_swing_trait_tags(focus_entry)
        human_focus_side = self._human_focus_side_for_symbol(symbol, direction=direction)
        human_focus_pick = bool(human_focus_side in {"LONG", "SHORT"})
        level_names = {str(key) for key in (levels or {}).keys()}
        master_avwap_h1_focus_type = ""
        if bool(level_names & {"h1_ema_15", "h1_sma_20"}):
            if "h1_ema_15" in level_names and self._is_top_pattern_h1_entry_focus(symbol, direction):
                master_avwap_h1_focus_type = "top_pattern"
            elif self._is_mid_earnings_h1_bounce_focus(symbol, direction):
                master_avwap_h1_focus_type = "mid_earnings"
        row = {
            "schema_version": BOUNCE_LEARNING_SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type,
            "logged_at": get_market_local_now().isoformat(timespec="seconds"),
            # Used immediately for the live verdict. Persistence already gets
            # the same bar time through the outcome row's ``entry_time``;
            # keeping this out of the candidate CSV avoids a 186 MB header
            # migration on the first alert after upgrade.
            "signal_time": trade_dt.isoformat(timespec="seconds") if trade_dt else "",
            "trade_date": trade_date,
            "symbol": symbol,
            "direction": direction,
            "bounce_types": ";".join(_bounce_type_keys_from_levels(levels or {})),
            "reason": reason,
            "score": self._score_bounce_candidate_snapshot(direction, levels, context),
            "entry_trigger": plan["entry_trigger"],
            "entry_price": plan["entry_price"],
            "stop_price": plan["stop_price"],
            "risk_per_share": plan["risk_per_share"],
            "target_1r": plan["target_1r"],
            "target_2r": plan["target_2r"],
            "atr": self._to_float_or_blank(atr),
            "threshold": self._to_float_or_blank(threshold),
            "rrs_timeframe": context.get("rrs_timeframe", ""),
            "rrs_spy": context.get("rrs_spy", ""),
            "rrs_sector": context.get("rrs_sector", ""),
            "rrs_industry": context.get("rrs_industry", ""),
            "sector": context.get("sector", ""),
            "industry": context.get("industry", ""),
            "sector_etf": context.get("sector_etf", ""),
            "industry_etf": context.get("industry_etf", ""),
            "market_environment": context.get("market_environment", ""),
            "human_focus_pick": bool(human_focus_pick),
            "human_focus_side": human_focus_side if human_focus_pick else "",
            "master_avwap_focus_label": focus_label,
            "master_avwap_priority_bucket": (
                str(focus_entry.get("priority_bucket") or "").strip()
                if isinstance(focus_entry, dict)
                else ""
            ),
            "master_avwap_setup_family": (
                str(focus_entry.get("setup_family") or "").strip()
                if isinstance(focus_entry, dict)
                else ""
            ),
            "master_avwap_h1_focus_type": master_avwap_h1_focus_type,
            "master_avwap_swing_traits": ";".join(swing_traits),
            "candles_waited": int(candles_waited or 0),
            "levels_json": self._json_for_learning(levels or {}),
            "candle_json": self._json_for_learning({"bounce": bounce_candle, "current": current_candle}),
            "metrics_json": self._json_for_learning(metrics),
            "context_json": self._json_for_learning(context),
        }
        self._append_learning_row(INTRADAY_BOUNCE_CANDIDATES_CSV, BOUNCE_CANDIDATE_EVENT_COLUMNS, row)
        return row

    def _build_bounce_feedback_alert_payload(self, display_text, candidate_row):
        row = candidate_row if isinstance(candidate_row, dict) else {}
        feedback_context = {
            "event_id": row.get("event_id", ""),
            "event_type": row.get("event_type", "confirmed"),
            "trade_date": row.get("trade_date", ""),
            "symbol": row.get("symbol", ""),
            "direction": row.get("direction", ""),
            "bounce_types": row.get("bounce_types", ""),
            "alert_message": display_text,
            "entry_price": row.get("entry_price", ""),
            "stop_price": row.get("stop_price", ""),
            "risk_per_share": row.get("risk_per_share", ""),
            "score": row.get("score", ""),
            "levels_json": row.get("levels_json", ""),
            "candle_json": row.get("candle_json", ""),
            "metrics_json": row.get("metrics_json", ""),
            "context_json": row.get("context_json", ""),
            "is_focus_pick": bool(row.get("human_focus_pick")),
            "focus_side": str(row.get("human_focus_side") or "").strip().upper(),
        }
        return {
            "kind": "bounce_alert",
            "text": str(display_text or ""),
            "feedback": feedback_context,
        }

    def _collect_near_miss_levels(self, symbol, direction, candle):
        metrics = self.symbol_metrics.get(symbol, {})
        atr = self.atr_cache.get(symbol)
        if not metrics or atr is None:
            return {}
        if isinstance(candle, pd.Series):
            candle = candle.to_dict()
        high_value = self._to_float_or_blank(candle.get("high"))
        low_value = self._to_float_or_blank(candle.get("low"))
        if high_value == "" or low_value == "":
            return {}
        threshold = THRESHOLD_MULTIPLIER * float(atr)

        level_specs = [
            ("vwap", "std_vwap"),
            ("dynamic_vwap", "dynamic_vwap"),
            ("eod_vwap", "eod_vwap"),
            ("ema_8", "ema_8"),
            ("ema_15", "ema_15"),
            ("ema_21", "ema_21"),
        ]
        if direction == "long":
            level_specs.extend(
                [
                    ("prev_day_high", "prev_high"),
                    ("vwap_upper_band", "vwap_1stdev_upper"),
                    ("dynamic_vwap_upper_band", "dynamic_vwap_1stdev_upper"),
                    ("eod_vwap_upper_band", "eod_vwap_1stdev_upper"),
                ]
            )
        else:
            level_specs.extend(
                [
                    ("prev_day_low", "prev_low"),
                    ("vwap_lower_band", "vwap_1stdev_lower"),
                    ("dynamic_vwap_lower_band", "dynamic_vwap_1stdev_lower"),
                    ("eod_vwap_lower_band", "eod_vwap_1stdev_lower"),
                ]
            )

        near_levels = {}
        for bounce_type, metric_key in level_specs:
            if not self.is_bounce_type_enabled(bounce_type):
                continue
            level_value = metrics.get(metric_key)
            if level_value is None:
                continue
            try:
                level_float = float(level_value)
            except (TypeError, ValueError):
                continue
            if float(high_value) >= (level_float - threshold) and float(low_value) <= (level_float + threshold):
                near_levels[bounce_type] = level_float
        if (
            self.is_bounce_type_enabled("vwap_eod_confluence")
            and metrics.get("std_vwap") is not None
            and metrics.get("eod_vwap") is not None
        ):
            std_vwap = float(metrics["std_vwap"])
            eod_vwap = float(metrics["eod_vwap"])
            if abs(std_vwap - eod_vwap) <= CONFLUENCE_MAX_SPREAD_ATR * float(atr):
                low_zone, high_zone = sorted([std_vwap, eod_vwap])
                if float(high_value) >= (low_zone - threshold) and float(low_value) <= (high_zone + threshold):
                    near_levels["vwap_eod_confluence"] = (low_zone + high_zone) / 2.0
        return near_levels

    def _maybe_log_near_miss_bounce(self, symbol, direction, candle):
        near_levels = self._collect_near_miss_levels(symbol, direction, candle)
        if not near_levels:
            return
        candle_time = str(candle.get("time") if isinstance(candle, dict) else "")
        level_key = "-".join(sorted(near_levels))
        cache_key = f"{symbol}|{direction}|{candle_time}|{level_key}"
        if cache_key in self.logged_near_miss_events:
            return
        self.logged_near_miss_events.add(cache_key)
        self._log_bounce_candidate_event(
            "near_miss",
            symbol,
            direction,
            near_levels,
            candle,
            candle,
            threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
            reason="Touched an enabled level but did not pass all bounce-quality or confirmation setup checks.",
            candidate_id=self._make_bounce_event_id(symbol, direction, candle, near_levels),
            candles_waited=0,
        )

    def _pending_bounce_symbols(self):
        symbols = set()
        for state in getattr(self, "pending_bounce_outcomes", {}).values():
            if not isinstance(state, dict):
                continue
            symbol = str(state.get("symbol") or "").strip().upper()
            if symbol:
                symbols.add(symbol)
        return symbols

    def _rows_after_bounce_entry_for_session(self, state, df, entry_dt):
        if df is None or df.empty or "datetime" not in df.columns:
            return pd.DataFrame()
        trade_date = _parse_iso_date_safe(state.get("trade_date")) or entry_dt.date()
        session = get_market_session_window(reference=entry_dt)
        open_naive = session.open_local.replace(tzinfo=None)
        close_naive = session.close_local.replace(tzinfo=None)

        frame = df.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"])
        if frame.empty:
            return pd.DataFrame()
        session_rows = frame[
            (frame["datetime"].dt.date == trade_date)
            & (frame["datetime"] > pd.Timestamp(entry_dt))
            & (frame["datetime"] >= pd.Timestamp(open_naive))
            & (frame["datetime"] <= pd.Timestamp(close_naive))
        ].copy()
        return session_rows.sort_values("datetime").reset_index(drop=True)

    def _is_eod_finalization_due(self, entry_dt):
        session = get_market_session_window(reference=entry_dt)
        close_with_grace = session.close_local + timedelta(minutes=BOUNCE_EOD_FINALIZE_GRACE_MINUTES)
        now_local = get_market_local_now()
        if now_local.tzinfo is None:
            now_local = now_local.replace(tzinfo=session.close_local.tzinfo)
        else:
            now_local = now_local.astimezone(session.close_local.tzinfo)
        return now_local >= close_with_grace

    def _register_bounce_outcome(self, symbol, direction, levels, bounce_candle, current_candle, candidate_id):
        if isinstance(current_candle, pd.Series):
            current_candle = current_candle.to_dict()
        if isinstance(bounce_candle, pd.Series):
            bounce_candle = bounce_candle.to_dict()
        current_candle = current_candle if isinstance(current_candle, dict) else {}
        bounce_candle = bounce_candle if isinstance(bounce_candle, dict) else {}
        plan = self._build_bounce_trade_plan(direction, levels, bounce_candle, current_candle)
        if plan.get("risk_per_share") == "":
            return
        entry_dt = self._parse_bar_time(current_candle.get("time"))
        if entry_dt is None:
            return
        context = self._build_bounce_context_snapshot(symbol, direction)
        event_id = candidate_id or self._make_bounce_event_id(symbol, direction, bounce_candle, levels)
        self.pending_bounce_outcomes[event_id] = {
            "event_id": event_id,
            "symbol": symbol,
            "direction": direction,
            "trade_date": entry_dt.date().isoformat(),
            "entry_time": entry_dt.isoformat(timespec="seconds"),
            "entry_price": float(plan["entry_price"]),
            "stop_price": float(plan["stop_price"]),
            "risk_per_share": float(plan["risk_per_share"]),
            "target_1r": float(plan["target_1r"]) if plan["target_1r"] != "" else None,
            "target_2r": float(plan["target_2r"]) if plan["target_2r"] != "" else None,
            "milestones_logged": [],
            "outcome_mode": "eod_hold",
            "context": context,
        }
        self._save_pending_bounce_outcomes()
        self._append_bounce_outcome_row(self.pending_bounce_outcomes[event_id], "registered", 0, None, pd.DataFrame())

    def _append_bounce_outcome_row(self, state, event_type, bars_elapsed, milestone_bar, rows_after_entry, *, finalize_eod=False):
        direction = state.get("direction")
        entry_price = float(state.get("entry_price"))
        risk = float(state.get("risk_per_share"))
        best_price = ""
        worst_price = ""
        close_r = ""
        mfe_r = ""
        mae_r = ""
        target_1r_hit = False
        target_2r_hit = False
        stop_hit = False
        minutes_elapsed = ""
        status = "open"
        eod_close = ""
        eod_move_pct = ""
        mfe_pct = ""
        mae_pct = ""
        entry_dt = self._parse_bar_time(state.get("entry_time"))
        if rows_after_entry is not None and not rows_after_entry.empty and risk > 0:
            high_max = float(rows_after_entry["high"].max())
            low_min = float(rows_after_entry["low"].min())
            last_row = rows_after_entry.iloc[-1]
            last_close = float(last_row["close"])
            last_dt = last_row.get("datetime")
            if entry_dt is not None and isinstance(last_dt, pd.Timestamp):
                minutes_elapsed = int((last_dt.to_pydatetime() - entry_dt).total_seconds() // 60)
            if direction == "long":
                best_price = high_max
                worst_price = low_min
                mfe_r = (high_max - entry_price) / risk
                mae_r = (low_min - entry_price) / risk
                close_r = (last_close - entry_price) / risk
                if entry_price > 0:
                    mfe_pct = ((high_max - entry_price) / entry_price) * 100.0
                    mae_pct = ((low_min - entry_price) / entry_price) * 100.0
                    eod_move_pct = ((last_close - entry_price) / entry_price) * 100.0
                target_1r_hit = state.get("target_1r") is not None and high_max >= float(state["target_1r"])
                target_2r_hit = state.get("target_2r") is not None and high_max >= float(state["target_2r"])
                stop_hit = low_min <= float(state.get("stop_price"))
            else:
                best_price = low_min
                worst_price = high_max
                mfe_r = (entry_price - low_min) / risk
                mae_r = (entry_price - high_max) / risk
                close_r = (entry_price - last_close) / risk
                if entry_price > 0:
                    mfe_pct = ((entry_price - low_min) / entry_price) * 100.0
                    mae_pct = ((entry_price - high_max) / entry_price) * 100.0
                    eod_move_pct = ((entry_price - last_close) / entry_price) * 100.0
                target_1r_hit = state.get("target_1r") is not None and low_min <= float(state["target_1r"])
                target_2r_hit = state.get("target_2r") is not None and low_min <= float(state["target_2r"])
                stop_hit = high_max >= float(state.get("stop_price"))
            eod_close = last_close if finalize_eod else ""
        elif finalize_eod and risk > 0:
            best_price = entry_price
            worst_price = entry_price
            close_r = 0.0
            mfe_r = 0.0
            mae_r = 0.0
            eod_close = entry_price
            eod_move_pct = 0.0
            mfe_pct = 0.0
            mae_pct = 0.0
        if finalize_eod:
            status = "eod_complete"
            if minutes_elapsed == "" and entry_dt is not None:
                session = get_market_session_window(reference=entry_dt)
                close_naive = session.close_local.replace(tzinfo=None)
                minutes_elapsed = int((close_naive - entry_dt).total_seconds() // 60)
        row = {
            "schema_version": BOUNCE_LEARNING_SCHEMA_VERSION,
            "event_id": state.get("event_id"),
            "event_type": event_type,
            "logged_at": get_market_local_now().isoformat(timespec="seconds"),
            "trade_date": state.get("trade_date"),
            "symbol": state.get("symbol"),
            "direction": direction,
            "entry_time": state.get("entry_time"),
            "entry_price": entry_price,
            "stop_price": state.get("stop_price"),
            "risk_per_share": risk,
            "bars_elapsed": int(bars_elapsed or 0),
            "minutes_elapsed": minutes_elapsed,
            "close_r": round(close_r, 4) if close_r != "" else "",
            "mfe_r": round(mfe_r, 4) if mfe_r != "" else "",
            "mae_r": round(mae_r, 4) if mae_r != "" else "",
            "best_price": best_price,
            "worst_price": worst_price,
            "target_1r_hit": bool(target_1r_hit),
            "target_2r_hit": bool(target_2r_hit),
            "stop_hit": bool(stop_hit),
            "status": status,
            "milestone_bar": "" if milestone_bar is None else int(milestone_bar),
            "context_json": self._json_for_learning(state.get("context", {})),
            "outcome_mode": state.get("outcome_mode") or "eod_hold",
            "eod_close": round(eod_close, 4) if eod_close != "" else "",
            "eod_move_pct": round(eod_move_pct, 4) if eod_move_pct != "" else "",
            "mfe_pct": round(mfe_pct, 4) if mfe_pct != "" else "",
            "mae_pct": round(mae_pct, 4) if mae_pct != "" else "",
        }
        self._append_learning_row(INTRADAY_BOUNCE_OUTCOMES_CSV, BOUNCE_OUTCOME_COLUMNS, row)
        return status

    def _update_pending_bounce_outcomes(self, symbol, df):
        if not self.pending_bounce_outcomes or df is None or df.empty:
            return
        if "datetime" not in df.columns:
            return
        symbol_key = str(symbol or "").strip().upper()
        changed = False
        for event_id, state in list(self.pending_bounce_outcomes.items()):
            if str(state.get("symbol") or "").strip().upper() != symbol_key:
                continue
            entry_dt = self._parse_bar_time(state.get("entry_time"))
            if entry_dt is None:
                self.pending_bounce_outcomes.pop(event_id, None)
                changed = True
                continue
            rows_after_entry = self._rows_after_bounce_entry_for_session(state, df, entry_dt)
            bars_elapsed = len(rows_after_entry)
            eod_due = self._is_eod_finalization_due(entry_dt)
            if rows_after_entry.empty and not eod_due:
                continue
            logged = set(int(item) for item in state.get("milestones_logged", []) if str(item).isdigit())
            for milestone in BOUNCE_OUTCOME_MILESTONE_BARS:
                if bars_elapsed < milestone or milestone in logged:
                    continue
                milestone_rows = rows_after_entry.head(milestone)
                status = self._append_bounce_outcome_row(
                    state,
                    f"{milestone}_bar",
                    bars_elapsed=milestone,
                    milestone_bar=milestone,
                    rows_after_entry=milestone_rows,
                )
                logged.add(milestone)
                changed = True
            state["milestones_logged"] = sorted(logged)
            if not rows_after_entry.empty:
                self._append_bounce_outcome_row(
                    state,
                    "update",
                    bars_elapsed=bars_elapsed,
                    milestone_bar=None,
                    rows_after_entry=rows_after_entry,
                )
            if eod_due:
                self._append_bounce_outcome_row(
                    state,
                    "final",
                    bars_elapsed=bars_elapsed,
                    milestone_bar=None,
                    rows_after_entry=rows_after_entry,
                    finalize_eod=True,
                )
                self.pending_bounce_outcomes.pop(event_id, None)
                changed = True
        if changed:
            self._save_pending_bounce_outcomes()


    def getReqId(self):
        with self.reqId_lock:
            reqId = self.reqIdCount
            self.reqIdCount += 1
            return reqId

    def create_stock_contract(self, symbol):
        logging.debug(f"Creating contract for symbol: {symbol}")
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def set_connection_info(self, host, port, client_id):
        self.ib_host = host
        self.ib_port = port
        self.ib_client_id_base = int(client_id) if client_id is not None else 125
        if self.ib_client_id is None:
            self.ib_client_id = self._allocate_client_id()

    def _allocate_client_id(self):
        base = int(self.ib_client_id_base or self.ib_client_id or 125)
        client_id = (base * 10000) + self.client_id_namespace + self.ib_client_id_offset
        self.ib_client_id_offset += 1
        self.ib_client_id = client_id
        self.client_id_conflict = False
        return client_id

    def _ensure_api_thread(self):
        if self.api_thread is None or not self.api_thread.is_alive():
            self.api_thread = threading.Thread(target=self.run, daemon=True)
            self.api_thread.start()

    def ensure_connected(self, timeout=10):
        if self.connection_status:
            return True
        if self.ib_host is None or self.ib_port is None or self.ib_client_id is None:
            logging.error("IB connection info not set.")
            return False
        deadline = time.time() + timeout
        with self.connect_lock:
            if self.connection_status:
                return True
            while time.time() < deadline:
                if self.client_id_conflict or self.ib_client_id is None:
                    old_client_id = self.ib_client_id
                    new_client_id = self._allocate_client_id()
                    if old_client_id is not None:
                        logging.warning(
                            f"IB client id {old_client_id} is already in use; retrying with {new_client_id}."
                        )
                try:
                    try:
                        self.disconnect()
                    except Exception:
                        pass
                    current_client_id = self.ib_client_id
                    self.connect(self.ib_host, self.ib_port, clientId=current_client_id)
                    self._ensure_api_thread()
                    attempt_deadline = min(deadline, time.time() + 2.5)
                    while not self.connection_status and time.time() < attempt_deadline:
                        if self.client_id_conflict:
                            break
                        time.sleep(0.2)
                    if self.connection_status:
                        logging.info(f"Reconnected to IB with clientId={self.ib_client_id}.")
                        return True
                    if self.client_id_conflict:
                        continue
                except Exception as e:
                    # A single attempt can fail on a transient ibapi race: the
                    # stale EReader thread from the prior connection calls
                    # Connection.disconnect() (nulling the socket) while
                    # Connection.connect() is mid-handshake, surfacing as
                    # "'NoneType' object has no attribute 'settimeout'". These
                    # self-heal once the old reader exits, so retry until the
                    # deadline instead of abandoning the whole reconnect.
                    logging.warning(f"Reconnect attempt failed ({e}); retrying.")
                    try:
                        self.disconnect()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    continue
            logging.error("Failed to reconnect to IB within timeout.")
            return False

    def _maybe_refresh_learning_after_close(self):
        """Rebuild the day-trade learning chain once per day after the close.

        Runs in a worker thread (no IB needed) so the scan loop never waits;
        keeps alert tiers/mutes current without a bot restart or Auto Pilot.
        """
        if time.time() - getattr(self, "_learning_refresh_last_check", 0.0) < 60:
            return
        self._learning_refresh_last_check = time.time()
        now = get_market_local_now()
        if now.weekday() >= 5:
            return
        today = now.date()
        if self._learning_refresh_date == today:
            return
        session = get_market_session_window(now)
        if now < session.close_local + timedelta(minutes=BOUNCE_EOD_FINALIZE_GRACE_MINUTES):
            return
        self._learning_refresh_date = today

        def worker():
            try:
                from bounce_bot_lib.learning import refresh_bounce_learning_state

                state = refresh_bounce_learning_state()
                segments = sum(len(value) for value in (state or {}).get("segments", {}).values())
                logging.info("After-close learning refresh complete (%s measured segments).", segments)
                if self.gui_callback:
                    self.gui_callback(
                        f"Learning refreshed after the close ({segments} measured segments).", "blue"
                    )
            except Exception:
                logging.exception("After-close learning refresh failed")

        threading.Thread(target=worker, name="learning-refresh", daemon=True).start()

    def _register_pacing_violation(self, error_code):
        now = time.time()
        with self.pacing_lock:
            previous = self.pacing_backoff_seconds
            if previous and (now - self.last_pacing_violation_at) <= IB_PACING_VIOLATION_RESET_SECONDS:
                backoff = min(previous * 2.0, IB_PACING_BACKOFF_MAX_SECONDS)
            else:
                backoff = float(IB_PACING_BACKOFF_INITIAL_SECONDS)
            self.pacing_backoff_seconds = backoff
            self.pacing_backoff_until = now + backoff
            self.last_pacing_violation_at = now
        logging.warning(
            f"IB pacing violation (code {error_code}): backing off historical requests for {backoff:.0f}s."
        )
        if self.gui_callback:
            self.gui_callback(f"IB pacing limit hit - slowing historical requests for {backoff:.0f}s.", "blue")

    def pacing_delay_remaining(self):
        with self.pacing_lock:
            return max(0.0, self.pacing_backoff_until - time.time())

    def _respect_pacing_backoff(self):
        """Block the requesting thread until the pacing cooldown has passed."""
        remaining = self.pacing_delay_remaining()
        if remaining <= 0:
            return
        logging.info(f"IB pacing backoff active: waiting {remaining:.0f}s before the next historical request.")
        deadline = time.time() + IB_PACING_BACKOFF_MAX_SECONDS
        while remaining > 0 and time.time() < deadline:
            time.sleep(min(remaining, 1.0))
            remaining = self.pacing_delay_remaining()

    def set_bounce_type_enabled(self, bounce_type, enabled):
        if bounce_type in self.bounce_type_toggles:
            self.bounce_type_toggles[bounce_type] = bool(enabled)

    def is_bounce_type_enabled(self, bounce_type):
        return self.bounce_type_toggles.get(bounce_type, False)

    def set_scanning_enabled(self, enabled):
        with self.scanning_lock:
            self.scanning_enabled = bool(enabled)
        state = "enabled" if self.scanning_enabled else "paused"
        logging.info(f"Scanning {state}.")
        if self.gui_callback:
            self.gui_callback(f"Scanning {state}.", "blue")

    def is_scanning_enabled(self):
        with self.scanning_lock:
            return self.scanning_enabled

    def _normalize_master_avwap_event_row(self, row):
        return normalize_master_avwap_event_row(row)

    def load_master_avwap_events_today(self):
        """
        Load today's NEW cross/bounce events from master_avwap signal output.
        Keeps parsing generic so new signal/level names continue to work.
        """
        today = datetime.now().date()
        try:
            stat = Path(MASTER_AVWAP_SIGNALS_FILENAME).stat()
            cache_key = (today, stat.st_mtime_ns, stat.st_size)
        except OSError:
            cache_key = (today, None, None)

        if (
            self.master_avwap_last_scan_date == today
            and getattr(self, "_master_avwap_events_cache_key", None) == cache_key
        ):
            return

        self.master_avwap_events = load_master_avwap_events_for_date(
            trade_date=today,
            signals_path=MASTER_AVWAP_SIGNALS_FILENAME,
        )
        self.master_avwap_last_scan_date = today
        self._master_avwap_events_cache_key = cache_key

    def _build_master_avwap_active_level_map(self):
        return build_master_avwap_active_level_map(self.master_avwap_events)

    def _build_master_avwap_second_stdev_cross_map(self):
        return build_master_avwap_second_stdev_cross_map(self.master_avwap_events)

    def find_active_master_avwap_bounces(self):
        self.load_master_avwap_events_today()
        active = {}
        for symbol, events in self.master_avwap_events.items():
            bounce_levels = sorted(
                {
                    event.get("level") or event.get("signal_type")
                    for event in events
                    if str(event.get("signal_type", "")).startswith("BOUNCE")
                }
            )
            if bounce_levels:
                active[symbol] = bounce_levels
        return active

    def _master_avwap_event_key(self, event):
        return (
            event.get("symbol"),
            event.get("trade_date"),
            event.get("signal_type"),
            event.get("anchor_type"),
            event.get("anchor_date"),
        )

    def load_master_avwap_focus(self):
        self.master_avwap_focus_map = load_master_avwap_focus_map(
            focus_path=MASTER_AVWAP_FOCUS_FILENAME,
        )

    def load_human_focus_picks(self):
        try:
            focus = load_focus_map()
        except Exception as exc:
            logging.debug(f"Failed loading human focus picks: {exc}")
            focus = {}
        self.human_focus_map = {
            "long": {
                str(item or "").strip().upper()
                for item in (focus.get("long") or set())
                if str(item or "").strip()
            },
            "short": {
                str(item or "").strip().upper()
                for item in (focus.get("short") or set())
                if str(item or "").strip()
            },
        }

    def _human_focus_sets(self):
        focus = getattr(self, "human_focus_map", {}) or {}
        return {
            "long": {
                str(item or "").strip().upper()
                for item in (focus.get("long") or set())
                if str(item or "").strip()
            },
            "short": {
                str(item or "").strip().upper()
                for item in (focus.get("short") or set())
                if str(item or "").strip()
            },
        }

    def _human_focus_symbols(self):
        focus = self._human_focus_sets()
        return set(focus["long"]) | set(focus["short"])

    def _human_focus_side_for_symbol(self, symbol, direction=None):
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return ""
        focus = self._human_focus_sets()
        direction = str(direction or "").strip().lower()
        in_long = symbol in focus["long"]
        in_short = symbol in focus["short"]
        if direction == "long":
            return "LONG" if in_long else ""
        if direction == "short":
            return "SHORT" if in_short else ""
        if in_long and not in_short:
            return "LONG"
        if in_short and not in_long:
            return "SHORT"
        if in_long and in_short:
            return "BOTH"
        return ""

    def _is_human_focus_bounce(self, symbol, direction):
        side = self._human_focus_side_for_symbol(symbol, direction=direction)
        return (direction == "long" and side == "LONG") or (direction == "short" and side == "SHORT")

    def _bounce_eval_filter_options_for_symbol(self, symbol, direction, allowed_bounce_types):
        if self._is_human_focus_bounce(symbol, direction):
            return None, True
        return allowed_bounce_types, False

    def load_master_avwap_d1_watchlist(self):
        self.master_avwap_d1_watchlist = load_master_avwap_d1_watchlist_map(
            watchlist_path=MASTER_AVWAP_D1_WATCHLIST_FILENAME,
        )

    def load_master_avwap_d1_upgrade_alerts(self):
        self.master_avwap_d1_upgrade_alerts = load_master_avwap_d1_upgrade_alerts_map(
            alerts_path=MASTER_AVWAP_D1_UPGRADE_ALERTS_FILENAME,
        )

    def get_master_avwap_d1_watch_symbols(self):
        symbols = set(getattr(self, "master_avwap_d1_watchlist", {}).keys())
        symbols.update(getattr(self, "master_avwap_d1_upgrade_alerts", {}).keys())
        return sorted(symbols)

    def get_symbol_direction(self, symbol):
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return ""
        human_focus_side = self._human_focus_side_for_symbol(symbol)
        if human_focus_side == "LONG":
            return "long"
        if human_focus_side == "SHORT":
            return "short"
        long_symbols = {str(item or "").strip().upper() for item in self.longs}
        short_symbols = {str(item or "").strip().upper() for item in self.shorts}
        if symbol in long_symbols:
            return "long"
        if symbol in short_symbols:
            return "short"

        # The bot's own morning picks (autolongs/autoshorts.txt). After the
        # trader's lists so a manual call always wins a direction conflict.
        if symbol in self._auto_watch_symbols("long"):
            return "long"
        if symbol in self._auto_watch_symbols("short"):
            return "short"

        watch_entry = self.master_avwap_d1_watchlist.get(symbol) or {}
        side = str(watch_entry.get("side") or "").strip().upper()
        if side == "LONG":
            return "long"
        if side == "SHORT":
            return "short"

        upgrade_entry = getattr(self, "master_avwap_d1_upgrade_alerts", {}).get(symbol) or {}
        side = str(upgrade_entry.get("side") or "").strip().upper()
        if side == "LONG":
            return "long"
        if side == "SHORT":
            return "short"

        focus_entry = self.master_avwap_focus_map.get(symbol) or {}
        side = str(focus_entry.get("side") or "").strip().upper()
        if side == "LONG":
            return "long"
        if side == "SHORT":
            return "short"
        return ""

    def _auto_watch_symbols(self, side=None):
        if side == "long":
            source = getattr(self, "auto_longs", [])
        elif side == "short":
            source = getattr(self, "auto_shorts", [])
        else:
            source = list(getattr(self, "auto_longs", [])) + list(getattr(self, "auto_shorts", []))
        return {str(item or "").strip().upper() for item in source if str(item or "").strip()}

    def get_scan_symbol_set(self):
        base_symbols = {
            str(item or "").strip().upper()
            for item in (self.longs + self.shorts)
            if str(item or "").strip()
        }
        return (
            base_symbols
            | self._auto_watch_symbols()
            | set(self.get_master_avwap_d1_watch_symbols())
            | self._human_focus_symbols()
        )

    def get_priority_scan_symbols(self):
        """Every-candle names: the trader's longs.txt / shorts.txt intraday
        dumps, human focus picks, and (so the bot's picks earn a clean tracked
        history) the auto watchlists."""
        base_symbols = {
            str(item or "").strip().upper()
            for item in (self.longs + self.shorts)
            if str(item or "").strip()
        }
        priority = base_symbols | self._human_focus_symbols()
        if AUTO_WATCHLIST_PRIORITY:
            priority |= self._auto_watch_symbols()
        return priority

    def _is_background_refresh_cycle(self):
        if not PRIORITY_WATCHLIST_EMPHASIS:
            return True
        every = max(1, int(BACKGROUND_SYMBOL_REFRESH_EVERY_CYCLES))
        return int(getattr(self, "_scan_cycle_index", 0)) % every == 0

    def _prune_latest_bars_for_cycle(self, refresh_background, background_symbols):
        """Cycle-start bar-cache policy: priority symbols, SPY and the sector/
        industry ETFs always refetch; background symbols keep their cached bars
        through off-cycles so the RRS scan costs them nothing."""
        if refresh_background or not self.latest_bars:
            self.latest_bars = {}
            return
        keep = {str(s or "").strip().upper() for s in background_symbols}
        self.latest_bars = {
            key: bars
            for key, bars in self.latest_bars.items()
            if str(key).split("|", 1)[0].strip().upper() in keep
        }

    def _master_avwap_trigger_float(self, value):
        try:
            if value is None or pd.isna(value):
                return None
        except Exception:
            if value is None:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _find_master_avwap_intraday_trigger_events(self, symbol, today_df):
        symbol = str(symbol or "").strip().upper()
        watch_entry = (
            getattr(self, "master_avwap_d1_upgrade_alerts", {}).get(symbol)
            or getattr(self, "master_avwap_d1_watchlist", {}).get(symbol)
            or {}
        )
        if not watch_entry or not bool(watch_entry.get("active_current_scan")):
            return []

        trigger_levels = watch_entry.get("trigger_levels") or []
        if not trigger_levels or today_df is None or today_df.empty:
            return []

        working = today_df.copy()
        if "datetime" not in working.columns:
            working["datetime"] = pd.to_datetime(
                working.get("time"),
                format="%Y%m%d  %H:%M:%S",
                errors="coerce",
            )
            if working["datetime"].isna().all():
                working["datetime"] = pd.to_datetime(working.get("time"), errors="coerce")
        working = working.dropna(subset=["datetime"]).sort_values("datetime")
        if working.empty:
            return []
        latest_row = working.iloc[-1]
        previous_close = None
        if len(working) >= 2:
            previous_close = self._master_avwap_trigger_float(working.iloc[-2].get("close"))

        side = str(watch_entry.get("side") or "").strip().upper()
        if side not in {"LONG", "SHORT"}:
            side = "LONG" if self.get_symbol_direction(symbol) == "long" else "SHORT"

        events = []
        for trigger in trigger_levels:
            if not isinstance(trigger, dict):
                continue
            level_value = self._master_avwap_trigger_float(trigger.get("level"))
            if level_value is None:
                continue
            action = str(trigger.get("action") or "").strip().lower()
            if action not in {"break_above", "break_below"}:
                action = "break_below" if side == "SHORT" else "break_above"

            reference_price = previous_close
            if reference_price is None:
                reference_price = self._master_avwap_trigger_float(trigger.get("armed_price"))
            high_value = self._master_avwap_trigger_float(latest_row.get("high"))
            low_value = self._master_avwap_trigger_float(latest_row.get("low"))
            close_value = self._master_avwap_trigger_float(latest_row.get("close"))
            if high_value is None or low_value is None or close_value is None or reference_price is None:
                continue

            if action == "break_above":
                crossed = reference_price < level_value and high_value >= level_value
            else:
                crossed = reference_price > level_value and low_value <= level_value
            if not crossed:
                continue

            bar_time = latest_row.get("datetime")
            if hasattr(bar_time, "to_pydatetime"):
                bar_time = bar_time.to_pydatetime()
            level_label = str(trigger.get("label") or "LEVEL").strip().upper()
            alert_label = str(trigger.get("alert_label") or level_label).strip()
            source_run = str(watch_entry.get("watchlist_run_date") or trigger.get("armed_at") or "").strip()
            reason_parts = [str(trigger.get("reason") or "").strip()]
            if source_run:
                reason_parts.append(f"master_run={source_run}")
            if bar_time:
                reason_parts.append(f"bar={bar_time:%H:%M}")
            reason = "; ".join(part for part in reason_parts if part)
            events.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "direction": "short" if side == "SHORT" else "long",
                    "event_type": str(trigger.get("event_type") or "preloaded_level_break").strip(),
                    "label": alert_label,
                    "reason": reason,
                    "sort_rank": 6,
                    "source": "watchlist_trigger",
                    "priority_score": watch_entry.get("priority_score"),
                    "trade_date": working["datetime"].iloc[-1].date().isoformat(),
                    "trigger_id": str(trigger.get("trigger_id") or f"{level_label}:{level_value:.4f}"),
                    "trigger_action": action,
                    "level_label": level_label,
                    "level": level_value,
                    "current_price": close_value,
                    "bar_high": high_value,
                    "bar_low": low_value,
                    "bar_time": bar_time.isoformat(timespec="minutes") if bar_time else "",
                    "trigger_source": str(trigger.get("source") or "").strip(),
                    "target_tier": str(trigger.get("target_tier") or "").strip(),
                    "upgrade_only": bool(trigger.get("upgrade_only")),
                }
            )
        return events

    def emit_master_avwap_intraday_trigger_flags(self, symbol, today_df):
        # Challenger in shadow (HIGH_CONVICTION plan 16.2): staged confirmation
        # engine sees the same bars. Generic champion level flags retain their
        # live behavior; A/S target touches are research observations until a
        # full scan confirms a Favorite / High Conviction bucket transition.
        try:
            from greatness_shadow import record_d1_shadow

            record_d1_shadow(self, symbol, today_df)
        except Exception:
            logging.debug("Greatness shadow hook failed.", exc_info=True)
        events = self._find_master_avwap_intraday_trigger_events(symbol, today_df)
        if not events or not self.gui_callback:
            return 0

        emitted_count = 0
        today_iso = datetime.now().date().isoformat()
        for event in events:
            event_key = self._master_avwap_d1_flag_key(event, today_iso=today_iso)
            if event_key in self.emitted_master_avwap_d1_flags:
                continue
            self.emitted_master_avwap_d1_flags.add(event_key)
            direction = str(event.get("direction") or "").strip().lower()
            message = self._format_master_avwap_d1_flag_event(event)
            if not self._master_avwap_d1_event_is_research_only(event):
                gui_tag = "d1_flag_long" if direction == "long" else "d1_flag_short" if direction == "short" else "d1_flag_watch"
                self.gui_callback(message, gui_tag)
            self.log_symbol(symbol, message)
            emitted_count += 1
        return emitted_count

    def _master_avwap_d1_event_is_research_only(self, event):
        source = str(event.get("source") or "").strip()
        return bool(
            source == "watchlist_upgrade_target"
            or (
                source == "watchlist_trigger"
                and (str(event.get("target_tier") or "").strip() or bool(event.get("upgrade_only")))
            )
        )

    def _format_master_avwap_d1_flag_event(self, event):
        symbol = str(event.get("symbol") or "").strip().upper()
        direction = str(event.get("direction") or "").strip().lower()
        label = str(event.get("label") or "D1 flag").strip()
        reason = str(event.get("reason") or "").strip()
        source = str(event.get("source") or "").strip()
        score = event.get("priority_score")
        suffix_parts = []
        level_value = self._master_avwap_trigger_float(event.get("level"))
        target_tier = str(event.get("target_tier") or "").strip()
        research_only = self._master_avwap_d1_event_is_research_only(event)
        if research_only:
            # These events are level observations (often an intraday wick), not
            # a completed full-scan promotion. Keep the evidence while removing
            # the old, misleading A/S-upgrade language.
            label = re.sub(r"^(?:A/S\s+)?upgrade:\s*", "", label, flags=re.IGNORECASE)
            reason = re.sub(
                r"^A/S\s+upgrade\s+target:\s*",
                "Developing target: ",
                reason,
                flags=re.IGNORECASE,
            )
            label = f"Developing level observation: {label}"
            suffix_parts.append(
                "status=research-only; await completed D1 scan confirmation into Favorite/High Conviction"
            )
        elif target_tier and "upgrade" not in label.lower():
            label = f"{target_tier} upgrade: {label}"
        if event.get("trigger_id") and level_value is not None:
            level_label = str(event.get("level_label") or "").strip()
            label = f"{label} {level_label}@{level_value:.2f}".strip()
            current_price = self._master_avwap_trigger_float(event.get("current_price"))
            if current_price is not None:
                suffix_parts.append(f"price={current_price:.2f}")
        if reason:
            suffix_parts.append(reason)
        if score is not None:
            try:
                suffix_parts.append(f"score={float(score):.0f}")
            except (TypeError, ValueError):
                pass
        if source:
            suffix_parts.append(f"source={source}")
        suffix = f" [{'; '.join(suffix_parts)}]" if suffix_parts else ""
        prefix = (
            "MASTER_AVWAP_D1_BUCKET_UPGRADE"
            if source == "bucket_upgrade"
            else "MASTER_AVWAP_D1_RESEARCH"
            if research_only
            else "MASTER_AVWAP_D1_FLAG"
        )
        return f"{prefix}: {symbol} ({direction or 'watch'}) {label}{suffix}"

    def _build_master_avwap_d1_flag_events(self):
        bucket_upgrade_alerts = {
            symbol: entry
            for symbol, entry in getattr(self, "master_avwap_d1_upgrade_alerts", {}).items()
            if isinstance(entry, dict) and entry.get("bucket_upgrade_events")
        }
        return build_master_avwap_d1_flag_events(
            {},
            {},
            {},
            trade_date=datetime.now().date(),
            d1_upgrade_alerts=bucket_upgrade_alerts,
        )

    def _master_avwap_d1_flag_key(self, event, today_iso=None):
        symbol = str(event.get("symbol") or "").strip().upper()
        trigger_id = str(event.get("trigger_id") or "").strip()
        if trigger_id:
            source = str(event.get("source") or "").strip()
            trigger_state = "intraday_trigger" if source == "watchlist_trigger" else "watch_target"
            return (
                today_iso or datetime.now().date().isoformat(),
                symbol,
                event.get("side"),
                trigger_state,
                trigger_id,
            )
        return (
            today_iso or datetime.now().date().isoformat(),
            symbol,
            event.get("side"),
            event.get("event_type"),
            event.get("label"),
            event.get("reason"),
        )

    def prime_master_avwap_d1_flags(self, events=None):
        today_iso = datetime.now().date().isoformat()
        active_events = events if events is not None else self._build_master_avwap_d1_flag_events()
        for event in active_events:
            self.emitted_master_avwap_d1_flags.add(self._master_avwap_d1_flag_key(event, today_iso=today_iso))
        self.master_avwap_d1_flags_primed_date = today_iso
        logging.info(
            "Master AVWAP D1 flags primed without GUI emission for %s existing flag(s).",
            len(active_events),
        )
        return len(active_events)

    def _should_emit_d1_flag(self, event):
        """Data-backed gate for D1 focus alerts. Returns (emit, reason)."""
        symbol = str(event.get("symbol") or "").strip().upper()
        if self._human_focus_side_for_symbol(symbol):
            return True, "human focus pick"

        bucket = str(event.get("priority_bucket") or event.get("bucket") or "").strip().lower()
        if bucket and bucket not in D1_FLAG_ACTIONABLE_BUCKETS:
            return False, f"upgrade lands in '{bucket}', not an actionable bucket"

        expected_r = None
        try:
            raw_expected = event.get("expected_r")
            expected_r = float(raw_expected) if raw_expected not in (None, "") else None
        except (TypeError, ValueError):
            expected_r = None
        if expected_r is not None and expected_r < D1_FLAG_MIN_EXPECTED_R:
            return False, f"expected R {expected_r:+.2f} below {D1_FLAG_MIN_EXPECTED_R:+.2f}"

        try:
            from bounce_bot_lib.learning import (
                evaluate_bounce_quality,
                load_bounce_learning_state,
                time_bucket_for,
            )

            focus_entry = getattr(self, "master_avwap_focus_map", {}).get(symbol)
            focus_label = self._describe_master_avwap_focus(focus_entry) if isinstance(focus_entry, dict) else ""
            quality = evaluate_bounce_quality(
                load_bounce_learning_state(),
                direction=str(event.get("side") or event.get("direction") or "").lower(),
                time_bucket=time_bucket_for(_bounce_quality_time(event) or get_market_local_now()),
                market_environment=self.get_market_environment(),
                priority_bucket=bucket,
                focus_label=focus_label,
            )
            if quality.get("muted"):
                return False, "; ".join(quality.get("mute_reasons") or ["learning-state mute"])
        except Exception as exc:  # gate must fail open, never block on errors
            logging.debug("D1 flag learning check skipped: %s", exc)
        return True, ""

    def emit_master_avwap_d1_flags(self):
        events = self._build_master_avwap_d1_flag_events()
        today_iso = datetime.now().date().isoformat()
        if self.master_avwap_d1_flags_primed_date != today_iso:
            self.prime_master_avwap_d1_flags(events)
            return
        if not self.gui_callback:
            return
        for event in events:
            symbol = str(event.get("symbol") or "").strip().upper()
            event_key = self._master_avwap_d1_flag_key(event, today_iso=today_iso)
            if event_key in self.emitted_master_avwap_d1_flags:
                continue
            self.emitted_master_avwap_d1_flags.add(event_key)
            direction = str(event.get("direction") or "").strip().lower()
            if not direction:
                direction = self.get_symbol_direction(symbol) or "watch"
            message = self._format_master_avwap_d1_flag_event({**event, "direction": direction})
            emit, suppress_reason = self._should_emit_d1_flag(event)
            if not emit:
                self.log_symbol(symbol, f"D1_FLAG_SUPPRESSED: {message} | {suppress_reason}")
                continue
            gui_tag = "d1_flag_long" if direction == "long" else "d1_flag_short" if direction == "short" else "d1_flag_watch"
            self.gui_callback(message, gui_tag)
            self.log_symbol(symbol, message)

    def _describe_master_avwap_focus(self, focus_entry):
        return describe_master_avwap_focus(focus_entry)

    def _is_mid_earnings_h1_bounce_focus(self, symbol, direction):
        focus_entry = self.master_avwap_focus_map.get(symbol)
        if not isinstance(focus_entry, dict):
            return False
        focus_side = str(focus_entry.get("side") or "").strip().upper()
        if direction == "long" and focus_side != "LONG":
            return False
        if direction == "short" and focus_side != "SHORT":
            return False

        setup_family = str(focus_entry.get("setup_family") or "").strip().lower()
        bucket = str(focus_entry.get("priority_bucket") or "").strip().lower()
        return bool(
            setup_family == MID_EARNINGS_ABOVE_SECOND_STDEV_FAMILY
            or (
                bucket == "stdev_retest_tracking"
                and focus_entry.get("mid_earnings_active_second_stdev_hold")
            )
        )

    def _is_top_pattern_h1_entry_focus(self, symbol, direction):
        if str(direction or "").strip().lower() != "long":
            return False
        focus_entry = self.master_avwap_focus_map.get(symbol)
        if not isinstance(focus_entry, dict):
            return False
        focus_side = str(focus_entry.get("side") or "").strip().upper()
        if focus_side != "LONG":
            return False
        setup_family = str(focus_entry.get("setup_family") or "").strip().lower()
        bucket = str(focus_entry.get("priority_bucket") or "").strip().lower()
        return bool(
            setup_family in TOP_PATTERN_FAMILIES
            or bucket in {"top_pattern_tracking", "top_strength_watchlist"}
            or focus_entry.get("top_pattern_watch")
            or focus_entry.get("top_pattern_entry")
        )

    def _evaluate_master_avwap_mid_earnings_h1_bounce(self, symbol, direction, reference_date):
        top_pattern_focus = self._is_top_pattern_h1_entry_focus(symbol, direction)
        mid_earnings_focus = self._is_mid_earnings_h1_bounce_focus(symbol, direction)
        if not top_pattern_focus and not mid_earnings_focus:
            return None
        atr = self.atr_cache.get(symbol)
        bars = self.latest_bars.get(symbol, [])
        h1_bars = _aggregate_bars_timeframe(bars, 60)
        candidate = detect_mid_earnings_h1_bounce(
            h1_bars,
            direction,
            atr,
            reference_date=reference_date,
            allowed_level_keys=H1_TOP_PATTERN_ENTRY_LEVELS if top_pattern_focus else None,
        )
        if not candidate:
            return None

        h1_event_type = "h1_top_pattern_15ema_entry" if top_pattern_focus else "h1_mid_earnings_bounce"
        confirmation_candle = candidate.get("confirmation_candle") or {}
        confirmation_time = str(confirmation_candle.get("time") or "")
        confirmation_dt = self._parse_bar_time(confirmation_time)
        alert_date = (
            confirmation_dt.date().isoformat()
            if confirmation_dt is not None
            else get_market_local_now().date().isoformat()
        )
        event_key = (
            alert_date,
            h1_event_type,
            symbol,
            direction,
            confirmation_time,
            tuple(sorted(str(level) for level in candidate.get("triggered_levels", []))),
        )
        if event_key in self.emitted_h1_mid_earnings_bounce_alerts:
            return None
        candidate["h1_event_key"] = event_key
        candidate["master_avwap_h1_focus_type"] = "top_pattern" if top_pattern_focus else "mid_earnings"
        return candidate

    def _emit_master_avwap_focus_bounce_alert(self, symbol, direction, levels_list):
        if not self.gui_callback:
            return
        focus_entry = self.master_avwap_focus_map.get(symbol)
        if not focus_entry:
            return
        focus_side = focus_entry.get("side")
        if direction == "long" and focus_side != "LONG":
            return
        if direction == "short" and focus_side != "SHORT":
            return

        normalized_levels = tuple(sorted(str(level) for level in (levels_list or [])))
        alert_key = (datetime.now().date().isoformat(), "bounce", symbol, direction, normalized_levels)
        if alert_key in self.emitted_master_avwap_focus_alerts:
            return
        self.emitted_master_avwap_focus_alerts.add(alert_key)

        reason = self._describe_master_avwap_focus(focus_entry)
        cross_entry = self.master_avwap_second_stdev_cross_map.get(symbol)
        if cross_entry:
            cross_side = cross_entry.get("side")
            if (direction == "long" and cross_side == "LONG") or (direction == "short" and cross_side == "SHORT"):
                reason = f"{reason}; {self._describe_master_avwap_second_stdev_cross(cross_entry)}"
        level_text = ", ".join(normalized_levels) if normalized_levels else "level"
        message = f"MASTER_AVWAP_FOCUS_BOUNCE: {symbol} ({direction}) {level_text} [{reason}]"
        gui_tag = "master_avwap_focus_long" if direction == "long" else "master_avwap_focus_short"
        self.gui_callback(message, gui_tag)
        self.log_symbol(symbol, message)

    def _describe_master_avwap_second_stdev_cross(self, cross_entry):
        return describe_master_avwap_second_stdev_cross(cross_entry)

    def _emit_master_avwap_second_stdev_bounce_alert(self, symbol, direction, levels_list):
        if not self.gui_callback:
            return
        cross_entry = self.master_avwap_second_stdev_cross_map.get(symbol)
        if not cross_entry:
            return

        focus_entry = self.master_avwap_focus_map.get(symbol)
        if focus_entry:
            focus_side = focus_entry.get("side")
            if (direction == "long" and focus_side == "LONG") or (direction == "short" and focus_side == "SHORT"):
                return

        cross_side = cross_entry.get("side")
        if direction == "long" and cross_side != "LONG":
            return
        if direction == "short" and cross_side != "SHORT":
            return

        normalized_levels = tuple(sorted(str(level) for level in (levels_list or [])))
        alert_key = (
            datetime.now().date().isoformat(),
            "2nd_stdev_bounce",
            symbol,
            direction,
            normalized_levels,
            cross_entry.get("signal_type", ""),
        )
        if alert_key in self.emitted_master_avwap_second_stdev_alerts:
            return
        self.emitted_master_avwap_second_stdev_alerts.add(alert_key)

        reason = self._describe_master_avwap_second_stdev_cross(cross_entry)
        level_text = ", ".join(normalized_levels) if normalized_levels else "bounce"
        message = f"MASTER_AVWAP_2ND_STDEV_BOUNCE: {symbol} ({direction}) {level_text} [{reason}]"
        gui_tag = "master_avwap_focus_long" if direction == "long" else "master_avwap_focus_short"
        self.gui_callback(message, gui_tag)
        self.log_symbol(symbol, message)

    def _focus_rrs_is_significant(self, entry, threshold):
        rrs_value = entry.get("rrs")
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        if rrs_value is None or move_ratio is None or excess_move_ratio is None:
            return False
        if not self._is_move_significant(entry):
            return False
        return (
            abs(rrs_value) >= max(MASTER_AVWAP_FOCUS_MIN_ABS_RRS, float(threshold) + 0.75)
            and abs(move_ratio) >= MASTER_AVWAP_FOCUS_MIN_MOVE_RATIO
            and abs(excess_move_ratio) >= MASTER_AVWAP_FOCUS_MIN_EXCESS_MOVE_RATIO
        )

    def _emit_master_avwap_focus_rrs_alerts(self, symbol_context, threshold, timeframe_key):
        if not EMIT_MASTER_AVWAP_FOCUS_RRS_ALERTS:
            return
        if not self.gui_callback or not self.master_avwap_focus_map:
            return

        timeframe_label = RRS_TIMEFRAMES.get(timeframe_key, {}).get("label", timeframe_key)
        today_iso = datetime.now().date().isoformat()
        for entry in symbol_context:
            symbol = entry.get("symbol")
            if not symbol:
                continue
            focus_entry = self.master_avwap_focus_map.get(symbol)
            if not focus_entry:
                continue

            focus_side = focus_entry.get("side")
            signal = entry.get("signal")
            if focus_side == "LONG" and signal != "RS":
                continue
            if focus_side == "SHORT" and signal != "RW":
                continue
            if not self._focus_rrs_is_significant(entry, threshold):
                continue

            alert_key = (today_iso, timeframe_key, symbol, signal)
            if alert_key in self.emitted_master_avwap_focus_alerts:
                continue
            self.emitted_master_avwap_focus_alerts.add(alert_key)

            reason = self._describe_master_avwap_focus(focus_entry)
            move_ratio = entry.get("move_ratio")
            excess_move_ratio = entry.get("excess_move_ratio")
            message = (
                f"MASTER_AVWAP_FOCUS_RRS: {symbol} ({focus_side.lower()}) {signal} "
                f"{entry['rrs']:+.2f} ER={excess_move_ratio:+.2f} MR={move_ratio:+.2f} "
                f"[{reason}, {timeframe_label}]"
            )
            gui_tag = "master_avwap_focus_long" if focus_side == "LONG" else "master_avwap_focus_short"
            self.gui_callback(message, gui_tag)
            self.log_symbol(symbol, message)


    def update_watchlists_from_master_avwap(self):
        self.load_master_avwap_events_today()
        self.master_avwap_second_stdev_cross_map = self._build_master_avwap_second_stdev_cross_map()
        active_level_map = self._build_master_avwap_active_level_map()
        event_symbols = set(active_level_map)
        if not event_symbols:
            logging.info("Master AVWAP: no new cross/bounce events for today.")
            return

        monitored = self.get_scan_symbol_set()
        matched_symbols = sorted(event_symbols & monitored)
        if not matched_symbols:
            logging.info(
                "Master AVWAP: today's event symbols do not intersect bounce bot watchlists."
            )
            return

        for symbol in matched_symbols:
            levels = active_level_map.get(symbol, [])
            side = "LONG" if self.get_symbol_direction(symbol) == "long" else "SHORT"
            symbol_events = self.master_avwap_events.get(symbol, [])
            newly_emitted = 0
            for event in symbol_events:
                event_key = self._master_avwap_event_key(event)
                if event_key in self.emitted_master_avwap_events:
                    continue
                self.emitted_master_avwap_events.add(event_key)
                newly_emitted += 1
                signal_type = event.get("signal_type", "")
                msg = f"MASTER_AVWAP_EVENT: {symbol} ({side}) {signal_type}"
                self.log_symbol(symbol, msg)

            if newly_emitted == 0:
                continue

            summary_msg = (
                f"MASTER_AVWAP_ACTIVE_EVENT: {symbol} ({side}) levels={levels} new_events={newly_emitted}"
            )
            logging.info(summary_msg)

    def set_rrs_threshold(self, value):
        with self.rrs_lock:
            self.rrs_threshold = float(value)

    def set_rrs_timeframe(self, key):
        if key not in RRS_TIMEFRAMES:
            return
        with self.rrs_lock:
            self.rrs_timeframe_key = key
            self.rrs_bar_size = RRS_TIMEFRAMES[key]["bar_size"]
            self.rrs_duration = RRS_TIMEFRAMES[key]["duration"]

    def get_rrs_settings(self):
        with self.rrs_lock:
            return (
                self.rrs_threshold,
                self.rrs_bar_size,
                self.rrs_duration,
                self.rrs_length,
                self.rrs_timeframe_key,
            )

    def set_market_environment(self, env_key, *, source="user"):
        if env_key not in MARKET_ENVIRONMENTS:
            return
        with self.market_environment_lock:
            changed = env_key != self.market_environment
            self.market_environment = env_key
        if source == "user" and changed:
            # A real manual selection pins the regime for the session; the
            # GUI's startup sync (same value) does not count as an override.
            self.market_environment_user_override = True
        status_msg = f"Market environment set to {MARKET_ENVIRONMENTS[env_key]['label']}"
        if source == "auto":
            status_msg = f"Auto market regime: {MARKET_ENVIRONMENTS[env_key]['label']}"
        self._refresh_rrs_gui(status_msg=status_msg)

    def clear_market_environment_override(self):
        self.market_environment_user_override = False

    def get_market_environment(self):
        with self.market_environment_lock:
            return self.market_environment

    def _spy_session_bars(self, cached_only=False):
        """(today_bars, prev_close) from the cached SPY 5m series.

        cached_only=True never triggers an IB fetch (safe from the GUI thread);
        it reads whatever series the scan/paused loop last cached.
        """
        if cached_only:
            spy_bars = self.latest_bars.get("SPY|5 D|5 mins") or []
        else:
            spy_bars = self.get_cached_5m_bars("SPY")
        if not spy_bars:
            return [], None
        today = spy_bars[-1].dt.date()
        today_bars = [bar for bar in spy_bars if bar.dt.date() == today]
        prior_bars = [bar for bar in spy_bars if bar.dt.date() < today]
        prev_close = prior_bars[-1].close if prior_bars else None
        return today_bars, prev_close

    def update_auto_market_environment(self):
        """Track the intraday regime from SPY's VWAP position + day color.

        Preferred read: SPY vs its session VWAP/1stdev band (see
        _classify_spy_vwap_regime). Early in the session (or without volume
        data) it falls back to green/red vs yesterday's close with |day %|
        deciding strong/weak. Never fights a manual GUI selection (user
        override wins until cleared).
        """
        if not MARKET_REGIME_AUTO_ENABLED or self.market_environment_user_override:
            return None
        today_bars, prev_close = self._spy_session_bars()
        if not today_bars or not prev_close:
            return None
        day_pct = (today_bars[-1].close - prev_close) / prev_close * 100.0
        stats = _spy_vwap_regime_stats(today_bars, prev_close)
        if stats is None:
            # Session too young / no volume: the day% rule decides.
            direction = "bullish" if day_pct >= 0 else "bearish"
            strength = "strong" if abs(day_pct) >= MARKET_REGIME_STRONG_ABS_PCT else "weak"
            env_key = f"{direction}_{strength}"
        elif stats["classification"] is not None:
            env_key = stats["classification"]
        else:
            # Mature session, no band hold, and VWAP position disagrees with
            # the day color: that's chop, not a trend to lean on.
            env_key = "neutral_chop"
        if env_key != self.get_market_environment():
            logging.info("Auto market regime: SPY %+.2f%% on the day -> %s", day_pct, env_key)
            self.set_market_environment(env_key, source="auto")
        return env_key

    def get_auto_regime_reading(self):
        """Read-only snapshot of what auto regime tracking thinks RIGHT NOW.

        Never mutates state and ignores any manual override, so the desk can
        always show the auto read - and how close the other regimes are -
        even while the trader is forcing a regime. Uses only cached SPY bars
        (no IB round-trip), so it is safe to call from the GUI thread; bar
        freshness comes from the scan loop (or the paused-mode refresh).
        Returns None until a SPY session series is available.
        """
        today_bars, prev_close = self._spy_session_bars(cached_only=True)
        if not today_bars or not prev_close:
            return None
        last_close = today_bars[-1].close
        day_pct = (last_close - prev_close) / prev_close * 100.0
        stats = _spy_vwap_regime_stats(today_bars, prev_close)
        source = "vwap"
        if stats is None:
            direction = "bullish" if day_pct >= 0 else "bearish"
            strength = "strong" if abs(day_pct) >= MARKET_REGIME_STRONG_ABS_PCT else "weak"
            env_key = f"{direction}_{strength}"
            source = "day_pct"
        elif stats["classification"] is not None:
            env_key = stats["classification"]
        else:
            env_key = "neutral_chop"  # mature mixed tape (see update_auto_market_environment)
        active_env = self.get_market_environment()
        reading = {
            "env_key": env_key,
            "label": MARKET_ENVIRONMENTS.get(env_key, {}).get("label", env_key),
            "source": source,
            "day_pct": day_pct,
            "last_close": last_close,
            "prev_close": prev_close,
            "bar_time": today_bars[-1].dt.strftime("%H:%M"),
            "override_active": bool(self.market_environment_user_override),
            "active_env_key": active_env,
            "active_label": MARKET_ENVIRONMENTS.get(active_env, {}).get("label", active_env),
            "strong_abs_pct": float(MARKET_REGIME_STRONG_ABS_PCT),
            "band_fraction_needed": float(MARKET_REGIME_VWAP_BAND_FRACTION),
        }
        if stats:
            reading.update(
                {
                    "vwap": stats["vwap"],
                    "stdev": stats["stdev"],
                    "above_band_frac": stats["above_band_frac"],
                    "below_band_frac": stats["below_band_frac"],
                }
            )
        return reading

    def _maybe_refresh_auto_regime_while_paused(self):
        """Keep the auto-regime read alive while scanning is paused.

        The scan loop normally refreshes SPY bars + regime every cycle, but a
        paused loop skips all of it; this refetches the SPY 5m series about
        once a minute so the desk's regime readout (and auto tracking, when
        no override is set) never goes stale.
        """
        if not MARKET_REGIME_AUTO_ENABLED:
            return
        now = time.time()
        if now - float(getattr(self, "_paused_regime_refresh_ts", 0.0)) < 60.0:
            return
        self._paused_regime_refresh_ts = now
        if not self.connection_status:
            return
        self.latest_bars.pop("SPY|5 D|5 mins", None)
        self.get_cached_5m_bars("SPY")  # refetch fresh series for the reading
        self.update_auto_market_environment()  # applies only when not overridden

    # ------------------------------------------------------------------
    # Regime-pause sweeps: when SPY pauses against the tape, whatever
    # refuses to participate is flagged (see constants block). Reworked
    # 2026-07-07: the per-symbol tiered bangers were screening information
    # dressed as entries (ten identical-ExpR alerts in one second, no
    # entry trigger), so the feed now gets ONE untiered summary line per
    # sweep batch while per-symbol pause-defiance counts persist to
    # REGIME_PAUSE_OBSERVATIONS_FILE as evidence for the hourly master
    # scan's swing rows. Tracker outcome logging is unchanged so the
    # signal keeps earning (or losing) measured WR/PF.
    # ------------------------------------------------------------------
    def _detect_spy_pause_start(self, spy_today, side):
        """Return the pause-start bar datetime, or None while the tape trends.

        Bearish tape (hunting shorts): a pause is a green SPY 5m candle, or no
        new session low across the last N candles. Bullish tape inverts.
        """
        lookback = int(REGIME_PAUSE_NO_NEW_EXTREME_CANDLES)
        if len(spy_today) < lookback + 2:
            return None
        last = spy_today[-1]
        recent = spy_today[-lookback:]
        earlier = spy_today[:-lookback]
        if side == "short":
            if last.close > last.open:
                return last.dt
            if min(bar.low for bar in recent) > min(bar.low for bar in earlier):
                return recent[0].dt
        else:
            if last.close < last.open:
                return last.dt
            if max(bar.high for bar in recent) < max(bar.high for bar in earlier):
                return recent[0].dt
        return None

    @staticmethod
    def _window_change_pct(bars, start_dt):
        window = [bar for bar in bars if bar.dt >= start_dt]
        if not window or not window[0].open:
            return None, window
        return (window[-1].close - window[0].open) / window[0].open * 100.0, window

    def _regime_pause_day_alerted(self, today, side):
        """Day-scoped alerted set: one banger per symbol per side per day.

        SPY pauses many times on a choppy day; the per-pause state resets each
        time the tape resumes, so without this the same non-participating
        names re-alert on every pause.
        """
        log = getattr(self, "_regime_pause_alert_log", None)
        if not log or log.get("date") != today:
            log = {"date": today, "sides": {}}
            self._regime_pause_alert_log = log
        return log["sides"].setdefault(side, set())

    def check_regime_pause_setups(self):
        """SPY paused against the tape: flag the non-participants hard."""
        env = self.get_market_environment()
        side = "short" if env.startswith("bearish") else "long"
        spy_today, _prev_close = self._spy_session_bars()
        if not spy_today:
            return []
        today = spy_today[-1].dt.date()
        pause_start = self._detect_spy_pause_start(spy_today, side)
        # Challenger in shadow (plan.md sec 16): the pure market-state engine
        # sees the same bars and logs agreement/divergence with this legacy
        # detector. It must never influence the live decision below.
        try:
            from market_state_bridge import record_spy_shadow

            record_spy_shadow(
                spy_today,
                _prev_close,
                legacy_pause_start=pause_start,
                side=side,
            )
        except Exception:
            logging.debug("SPY shadow-state hook failed.", exc_info=True)
        if pause_start is None:
            self._regime_pause_state = None
            return []
        state = self._regime_pause_state
        if not state or state.get("date") != today or state.get("side") != side:
            state = {
                "date": today,
                "side": side,
                "start_dt": pause_start,
                "alerted": self._regime_pause_day_alerted(today, side),
                # Per-pause-window set: a symbol's defiance counts once per
                # pause toward the observations file, however many sweep
                # cycles the pause lasts.
                "observed": set(),
            }
            self._regime_pause_state = state
        pause_candles = sum(1 for bar in spy_today if bar.dt >= state["start_dt"])
        if pause_candles > int(REGIME_PAUSE_SWEEP_MAX_CANDLES):
            return []
        return self._sweep_regime_pause_bangers(state, spy_today, side)

    def _sweep_regime_pause_bangers(self, state, spy_today, side):
        watchlist = self.shorts if side == "short" else self.longs
        symbols = sorted(
            {str(item or "").strip().upper() for item in watchlist if str(item or "").strip()}
        )
        if not symbols:
            return []
        sign = -1.0 if side == "short" else 1.0
        spy_open = spy_today[0].open
        spy_day = (spy_today[-1].close - spy_open) / spy_open * 100.0 if spy_open else None
        spy_window, _ = self._window_change_pct(spy_today, state["start_dt"])
        if spy_day is None or spy_window is None:
            return []
        # The pause must actually run against the tape direction.
        if (side == "short" and spy_window < 0) or (side == "long" and spy_window > 0):
            return []

        observed = state.setdefault("observed", set())
        flagged = []
        observations_dirty = False
        for symbol in symbols:
            if symbol in observed:
                continue
            bars = self.get_cached_5m_bars(symbol)
            if not bars:
                continue
            sym_today = [bar for bar in bars if bar.dt.date() == state["date"]]
            if len(sym_today) < 2 or not sym_today[0].open:
                continue
            sym_day = (sym_today[-1].close - sym_today[0].open) / sym_today[0].open * 100.0
            sym_window, window_bars = self._window_change_pct(sym_today, state["start_dt"])
            if sym_window is None:
                continue
            # Day-anchored: the symbol must have been notably weaker (shorts) /
            # stronger (longs) than SPY since the open - the "obvious earlier
            # in the day" filter.
            day_excess = sign * (sym_day - spy_day)
            if day_excess < REGIME_BANGER_DAY_EXCESS_PCT:
                continue
            # Pause-window: still moving with the trade direction, or refusing
            # to follow SPY's pause (compression counts).
            window_excess = sign * (sym_window - spy_window)
            still_trending = sign * sym_window > 0
            pre_window = [bar for bar in sym_today if bar.dt < state["start_dt"]]
            if not pre_window:
                made_new_extreme = False
            elif side == "short":
                made_new_extreme = min(bar.low for bar in window_bars) < min(bar.low for bar in pre_window)
            else:
                made_new_extreme = max(bar.high for bar in window_bars) > max(bar.high for bar in pre_window)
            if not (still_trending or made_new_extreme or window_excess >= REGIME_BANGER_WINDOW_EXCESS_PCT):
                continue
            observed.add(symbol)
            self._record_regime_pause_observation(
                symbol,
                side,
                state["date"],
                day_excess=day_excess,
                sym_day=sym_day,
            )
            observations_dirty = True
            if symbol in state["alerted"]:
                continue
            state["alerted"].add(symbol)
            hit = {
                "symbol": symbol,
                "side": side,
                "sym_day": sym_day,
                "spy_day": spy_day,
                "sym_window": sym_window,
                "spy_window": spy_window,
                "day_excess": day_excess,
                "last_bar": sym_today[-1],
            }
            self._record_regime_pause_banger(hit)
            flagged.append(hit)
        if observations_dirty:
            self._save_regime_pause_observations()
        if flagged:
            self._emit_regime_pause_summary(side, spy_window, flagged, state)
        return flagged

    def _record_regime_pause_banger(self, hit):
        """Tracker-side bookkeeping for a pause-defiance hit (no feed alert).

        Candidate event, outcome registration, and the bounce log keep
        flowing so the tracker can measure whether pause defiance carries
        real edge; the GUI surface is the batch summary line only.
        """
        symbol = hit["symbol"]
        side = hit["side"]
        bounce_type = REGIME_PAUSE_RW_TYPE if side == "short" else REGIME_PAUSE_RS_TYPE
        last_bar = hit["last_bar"]
        bar_dict = {
            "time": last_bar.dt.strftime("%Y%m%d  %H:%M:%S"),
            "open": last_bar.open,
            "high": last_bar.high,
            "low": last_bar.low,
            "close": last_bar.close,
        }
        levels = {bounce_type: last_bar.close}
        event_row = self._log_bounce_candidate_event(
            "confirmed",
            symbol,
            side,
            levels,
            bar_dict,
            bar_dict,
            reason=(
                f"SPY pause banger: day {hit['sym_day']:+.2f}% vs SPY {hit['spy_day']:+.2f}%; "
                f"pause window {hit['sym_window']:+.2f}% vs SPY {hit['spy_window']:+.2f}%"
            ),
        )
        self._register_bounce_outcome(symbol, side, levels, bar_dict, bar_dict, event_row.get("event_id", ""))
        quality = self._evaluate_bounce_alert_quality(side, levels, event_row)
        message = (
            f"REGIME PAUSE {symbol} ({side}): SPY paused ({hit['spy_window']:+.2f}% window) "
            f"but {symbol} isn't participating - day {hit['sym_day']:+.2f}% vs SPY {hit['spy_day']:+.2f}% "
            f"(excess {hit['day_excess']:+.2f}%), window {hit['sym_window']:+.2f}%. Swing evidence recorded."
        )
        self.log_symbol(symbol, f"SWEEP: {message}")
        self.log_bounce_to_file(
            symbol=symbol,
            direction=side,
            levels=levels,
            bounce_candle=bar_dict,
            current_candle=bar_dict,
            threshold=0.0,
            quality=quality,
        )

    def _emit_regime_pause_summary(self, side, spy_window, hits, state):
        """One untiered feed line per sweep batch (symbols day-deduped).

        Untiered on purpose: no [X-TIER] stamp and no BANGER token, so the
        line passes the feed's tier gate quietly and never fires the sound.
        """
        symbols = ", ".join(hit["symbol"] for hit in hits)
        pressing = "pressing lows" if side == "short" else "holding highs"
        total_today = len(state.get("alerted") or ())
        message = (
            f"REGIME PAUSE WATCH ({side}): SPY paused ({spy_window:+.2f}% window) - "
            f"{len(hits)} swing {side}{'s' if len(hits) != 1 else ''} still {pressing}: {symbols}"
            f" ({total_today} today). Recorded as swing-scan evidence, not an entry signal."
        )
        if self.gui_callback:
            self.gui_callback(message, "red" if side == "short" else "green")

    # ------------------------------------------------------------------
    # Entry assist (2026-07-08): regime-tailored entry timing.
    # Strong regimes: a pullback (bullish) / bounce (bearish) WINDOW - the
    # button or auto SPY-pause detection marks the start; the end ranks
    # which names held the trade direction best through the counter-move.
    # Weak regimes: instant strongest/weakest trailing-30m lists.
    # Neutral/chop: both lists at once; auto emits them on a 30m cadence.
    # ------------------------------------------------------------------
    def _entry_candidates(self, side):
        pools = (
            (getattr(self, "longs", None), getattr(self, "auto_longs", None))
            if side == "long"
            else (getattr(self, "shorts", None), getattr(self, "auto_shorts", None))
        )
        symbols = set()
        for pool in pools:
            symbols.update(
                str(item or "").strip().upper() for item in (pool or []) if str(item or "").strip()
            )
        symbols.discard("SPY")
        return sorted(symbols)

    def start_entry_window(self, sides, source="manual", start_dt=None):
        spy_today, _prev = self._spy_session_bars()
        if not spy_today:
            return {"ok": False, "note": "No SPY session bars yet - cannot open an entry window."}
        # Auto passes the detected pause-start bar so the whole counter-move
        # is measured; manual clicks anchor at the current bar.
        start_dt = start_dt or spy_today[-1].dt
        self._entry_window = {
            "sides": tuple(sides),
            "source": source,
            "date": start_dt.date(),
            "start_dt": start_dt,
        }
        sides_text = "/".join(sides)
        tracking = "RS holders" if "long" in sides else "RW leaders"
        if self.gui_callback:
            self.gui_callback(
                f"ENTRY WINDOW OPEN ({sides_text}): tracking {tracking} while SPY counter-moves [{source}].",
                "blue",
            )
        return {"ok": True, "note": f"Entry window opened @ {start_dt.strftime('%H:%M')} ({sides_text})."}

    def end_entry_window(self, source="manual"):
        window = getattr(self, "_entry_window", None)
        if not window:
            return {"ok": False, "note": "No entry window active."}
        self._entry_window = None
        spy_today, _prev = self._spy_session_bars()
        spy_bars_today = [bar for bar in spy_today if bar.dt.date() == window["date"]]
        spy_window, spy_window_bars = self._window_change_pct(spy_bars_today, window["start_dt"])
        if spy_window is None or len(spy_window_bars) < ENTRY_WINDOW_MIN_BARS:
            note = "Entry window too short to rank - no output."
            if source == "manual" and self.gui_callback:
                self.gui_callback(f"ENTRY WINDOW: {note}", "blue")
            return {"ok": False, "note": note}
        results = {}
        for side in window["sides"]:
            ranked = self._rank_entry_window_side(side, window["start_dt"], window["date"], spy_window)
            results[side] = ranked
            self._emit_entry_window_summary(side, window, spy_window, ranked, source)
        return {"ok": True, "note": f"Entry window closed - ranked output emitted [{source}].", "results": results}

    def _rank_entry_window_side(self, side, start_dt, today, spy_window, cached_only=False):
        sign = 1.0 if side == "long" else -1.0
        rows = []
        for symbol in self._entry_candidates(side):
            if cached_only:
                bars = self.latest_bars.get(f"{symbol}|5 D|5 mins")
            else:
                bars = self.get_cached_5m_bars(symbol)
            sym_today = [bar for bar in bars or [] if bar.dt.date() == today]
            sym_window, _window_bars = self._window_change_pct(sym_today, start_dt)
            if sym_window is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "window_pct": sym_window,
                    "excess": sign * (sym_window - spy_window),
                }
            )
        rows.sort(key=lambda row: -row["excess"])
        return rows[:ENTRY_WINDOW_TOP_N]

    def _emit_entry_window_summary(self, side, window, spy_window, ranked, source):
        if not self.gui_callback:
            return
        start_text = window["start_dt"].strftime("%H:%M")
        if not ranked:
            self.gui_callback(
                f"ENTRY WINDOW ({side}): no candidates with fresh bars for the {start_text} window.",
                "blue",
            )
            return
        names = ", ".join(
            f"{row['symbol']} {row['window_pct']:+.2f}% (x{row['excess']:+.2f})" for row in ranked
        )
        held = "held strongest" if side == "long" else "stayed weakest"
        self.gui_callback(
            f"ENTRY WINDOW ({side}): SPY {spy_window:+.2f}% since {start_text} - "
            f"{held} through it: {names} [{source}]",
            "green" if side == "long" else "red",
        )

    @staticmethod
    def _trailing_return_pct(bars, minutes):
        if not bars:
            return None
        cutoff = bars[-1].dt - timedelta(minutes=int(minutes))
        window = [bar for bar in bars if bar.dt > cutoff]
        if len(window) < 2 or not window[0].open:
            return None
        return (window[-1].close - window[0].open) / window[0].open * 100.0

    def _rank_trailing_movers(self, side, minutes, spy_bars, cached_only=False):
        sign = 1.0 if side == "long" else -1.0
        spy_change = self._trailing_return_pct(spy_bars or [], minutes) or 0.0
        rows = []
        for symbol in self._entry_candidates(side):
            if cached_only:
                symbol_bars = self.latest_bars.get(f"{symbol}|5 D|5 mins")
            else:
                symbol_bars = self.get_cached_5m_bars(symbol)
            change = self._trailing_return_pct(symbol_bars or [], minutes)
            if change is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "change_pct": change,
                    "excess": sign * (change - spy_change),
                }
            )
        rows.sort(key=lambda row: -row["excess"])
        return rows[:ENTRY_WINDOW_TOP_N]

    def emit_trailing_movers(self, sides, minutes=ENTRY_MOVERS_MINUTES, source="manual"):
        spy_bars = self.get_cached_5m_bars("SPY")
        results = {}
        for side in sides:
            ranked = self._rank_trailing_movers(side, minutes, spy_bars)
            results[side] = ranked
            if not self.gui_callback:
                continue
            label = "STRONGEST" if side == "long" else "WEAKEST"
            if ranked:
                names = ", ".join(f"{row['symbol']} {row['change_pct']:+.2f}%" for row in ranked)
            else:
                names = "none with fresh bars"
            self.gui_callback(f"{label} {int(minutes)}M ({side}): {names} [{source}]",
                              "green" if side == "long" else "red")
        emitted = sum(len(rows) for rows in results.values())
        return {
            "ok": emitted > 0,
            "note": f"Emitted {int(minutes)}m movers ({'/'.join(sides)}): {emitted} name(s) [{source}].",
            "results": results,
        }

    def entry_assist_action(self):
        """The strip button: window toggle in strong regimes, instant lists otherwise."""
        mode = entry_assist_mode_for_env(self.get_market_environment())
        if mode["mode"] == "window":
            if getattr(self, "_entry_window", None):
                return self.end_entry_window(source="manual")
            return self.start_entry_window(mode["sides"], source="manual")
        return self.emit_trailing_movers(mode["sides"], source="manual")

    def entry_assist_state(self):
        """Read-only state for the GUI button label (safe from the GUI thread)."""
        env = self.get_market_environment()
        window = getattr(self, "_entry_window", None)
        return {
            "env_key": env,
            "mode": entry_assist_mode_for_env(env)["mode"],
            "window_active": bool(window),
            "window_started": window["start_dt"].strftime("%H:%M") if window else "",
            "window_sides": list(window["sides"]) if window else [],
            "window_source": window.get("source", "") if window else "",
        }

    def spy_m5_chart_bars(self, max_sessions=2):
        """Cached SPY 5m bars as plain dicts for the GUI chart.

        Reads ``latest_bars`` only (never triggers an IB fetch), so it is safe
        from the GUI thread; the scan / paused loop keeps the series fresh."""
        bars = self.latest_bars.get("SPY|5 D|5 mins") or []
        if not bars:
            return []
        dates = sorted({bar.dt.date() for bar in bars})
        keep = set(dates[-max(1, int(max_sessions)):])
        return [
            {
                "dt": bar.dt,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(getattr(bar, "volume", 0) or 0),
            }
            for bar in bars
            if bar.dt.date() in keep
        ]

    @staticmethod
    def _m5_bar_completed(bar_dt, now=None):
        """Whether a timestamped M5 bar is closed at ``now``.

        IB's historical cache has no complete/forming marker. The timestamp is
        the bar start, so a transition-grade bar closes five minutes later.
        """
        if not isinstance(bar_dt, datetime):
            return False
        moment = now or datetime.now(tz=bar_dt.tzinfo)
        if bar_dt.tzinfo is None and moment.tzinfo is not None:
            moment = moment.replace(tzinfo=None)
        elif bar_dt.tzinfo is not None and moment.tzinfo is None:
            moment = moment.replace(tzinfo=bar_dt.tzinfo)
        return bar_dt + timedelta(minutes=5) <= moment

    def cached_m5_window_bars(self, start_dt, end_dt, *, completed_only=True, now=None):
        """All cached M5 symbols over a GUI window, without provider I/O.

        This read model feeds advisory industry composites. It deliberately
        exposes only keys from the existing ``5 D|5 mins`` cache and never
        launches a second market-data loop.
        """
        result = {}
        suffix = "|5 D|5 mins"
        for key, cached in (getattr(self, "latest_bars", {}) or {}).items():
            if not str(key).endswith(suffix):
                continue
            symbol = str(key)[: -len(suffix)].strip().upper()
            if not symbol:
                continue
            window = [
                bar
                for bar in cached or []
                if isinstance(getattr(bar, "dt", None), datetime)
                and start_dt <= bar.dt <= end_dt
                and (not completed_only or self._m5_bar_completed(bar.dt, now=now))
            ]
            if not window:
                continue
            window.sort(key=lambda bar: bar.dt)
            result[symbol] = [
                {
                    "dt": bar.dt,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(getattr(bar, "volume", 0) or 0),
                }
                for bar in window
            ]
        return result

    def rank_window_movers(
        self,
        start_dt,
        end_dt,
        sides=("long", "short"),
        *,
        completed_only=False,
        now=None,
    ):
        """RS/RW ranking over an arbitrary from->to window (GUI chart selection).

        Same excess-vs-SPY math as the entry windows, but with an explicit end
        instead of "now", so the trader can measure any stretch of the session.
        Cached bars only - safe from the GUI thread. ``completed_only`` is the
        advisory GUI contract: exclude forming bars and require every ranked
        symbol to share SPY's exact endpoints with at least 80% timestamp
        coverage. The legacy preview callers keep their existing default.
        """

        def window_bars(bars):
            return sorted(
                [
                    bar
                    for bar in bars
                    if start_dt <= bar.dt <= end_dt
                    and (not completed_only or self._m5_bar_completed(bar.dt, now=now))
                ],
                key=lambda bar: bar.dt,
            )

        def window_pct(window):
            if not window or not window[0].open:
                return None
            return (window[-1].close - window[0].open) / window[0].open * 100.0

        spy_bars = window_bars(self.latest_bars.get("SPY|5 D|5 mins") or [])
        spy_pct = window_pct(spy_bars)
        if spy_pct is None:
            return {"ok": False, "note": "No cached SPY bars covering that window yet.", "rows": []}
        spy_timestamps = {bar.dt for bar in spy_bars}
        first_spy_ts = spy_bars[0].dt
        last_spy_ts = spy_bars[-1].dt
        rows = []
        candidates_considered = 0
        for side in sides:
            sign = 1.0 if side == "long" else -1.0
            for symbol in self._entry_candidates(side):
                candidates_considered += 1
                symbol_window = window_bars(
                    self.latest_bars.get(f"{symbol}|5 D|5 mins") or []
                )
                timestamp_coverage = None
                if completed_only:
                    symbol_timestamps = {bar.dt for bar in symbol_window}
                    timestamp_coverage = len(symbol_timestamps & spy_timestamps) / max(
                        1, len(spy_timestamps)
                    )
                    if (
                        first_spy_ts not in symbol_timestamps
                        or last_spy_ts not in symbol_timestamps
                        or timestamp_coverage < 0.8
                    ):
                        continue
                pct = window_pct(symbol_window)
                if pct is None:
                    continue
                row = {
                    "symbol": symbol,
                    "side": "LONG" if side == "long" else "SHORT",
                    "window_pct": pct,
                    "spy_pct": spy_pct,
                    "excess": sign * (pct - spy_pct),
                }
                if completed_only:
                    row.update(
                        {
                            "timestamp_coverage": timestamp_coverage,
                            "alignment_status": "exact_completed",
                        }
                    )
                rows.append(row)
        rows.sort(key=lambda row: -row["excess"])
        result = {"ok": True, "spy_pct": spy_pct, "rows": rows}
        if completed_only:
            result.update(
                {
                    "completed_only": True,
                    "candidate_coverage": len(rows) / max(1, candidates_considered),
                    "candidates_considered": candidates_considered,
                    "candidates_ranked": len(rows),
                    "data_complete_through": (
                        spy_bars[-1].dt.isoformat(timespec="minutes") if spy_bars else ""
                    ),
                }
            )
        return result

    def entry_assist_board_snapshot(self, movers_minutes=ENTRY_MOVERS_MINUTES):
        """The always-on RS/RW board: everything entry assist can say, fresh.

        Auto regime + live SPY pause detection + the ACTIVE window's ranking
        as it stands right now (not just at close) + a pause-preview ranking
        when SPY is pausing but no window is open + trailing strongest AND
        weakest movers for both sides regardless of regime. Cached bars only
        (never triggers an IB fetch) - safe to poll from the GUI thread."""
        spy_bars = self.latest_bars.get("SPY|5 D|5 mins") or []
        if not spy_bars:
            return {}
        env = self.get_market_environment()
        today = spy_bars[-1].dt.date()
        spy_today = [bar for bar in spy_bars if bar.dt.date() == today]
        snapshot = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "env_key": env,
            "env_label": str(MARKET_ENVIRONMENTS.get(env, {}).get("label", env)),
            "bar_time": spy_today[-1].dt.strftime("%H:%M") if spy_today else "",
            "movers_minutes": int(movers_minutes),
        }

        trend_side = "short" if env.startswith("bearish") else "long"
        pause_start = self._detect_spy_pause_start(spy_today, trend_side) if spy_today else None
        snapshot["pause"] = {
            "trend_side": trend_side,
            "detected": pause_start is not None,
            "since": pause_start.strftime("%H:%M") if pause_start else "",
        }

        window = getattr(self, "_entry_window", None)
        if window and spy_today:
            window_bars = [bar for bar in spy_today if bar.dt.date() == window["date"]]
            spy_window, _bars = self._window_change_pct(window_bars, window["start_dt"])
            rankings = {}
            if spy_window is not None:
                for side in window["sides"]:
                    rankings[side] = self._rank_entry_window_side(
                        side, window["start_dt"], window["date"], spy_window, cached_only=True
                    )
            snapshot["window"] = {
                "active": True,
                "sides": list(window["sides"]),
                "started": window["start_dt"].strftime("%H:%M"),
                "source": window.get("source", ""),
                "spy_pct": spy_window,
                "rankings": rankings,
            }
        else:
            snapshot["window"] = {"active": False}
            # SPY is pausing but nothing is tracking it yet: show what a
            # window opened at the pause start WOULD say right now.
            if pause_start is not None:
                preview = self.rank_window_movers(pause_start, spy_today[-1].dt, sides=(trend_side,))
                if preview.get("ok"):
                    snapshot["pause_preview"] = {
                        "side": trend_side,
                        "since": pause_start.strftime("%H:%M"),
                        "spy_pct": preview.get("spy_pct"),
                        "rows": (preview.get("rows") or [])[:ENTRY_WINDOW_TOP_N],
                    }

        snapshot["movers"] = {
            side: self._rank_trailing_movers(side, movers_minutes, spy_bars, cached_only=True)
            for side in ("long", "short")
        }
        return snapshot

    def entry_assist_command(self, command):
        """GUI button array: explicit entry-assist actions, independent of regime.

        ``pullback_window`` / ``bounce_window`` toggle the tracking window for
        their side (opening one while the other side's window is active first
        closes it, emitting its ranked list). The movers commands emit the
        trailing-30m list(s) immediately.
        """
        command = str(command or "").strip().lower()
        if command in ("pullback_window", "bounce_window"):
            sides = ("long",) if command == "pullback_window" else ("short",)
            window = getattr(self, "_entry_window", None)
            if window:
                same_sides = tuple(window.get("sides") or ()) == sides
                result = self.end_entry_window(source="manual")
                if same_sides:
                    return result
            return self.start_entry_window(sides, source="manual")
        if command == "strongest_30m":
            return self.emit_trailing_movers(("long",), source="manual")
        if command == "weakest_30m":
            return self.emit_trailing_movers(("short",), source="manual")
        if command == "movers_30m":
            return self.emit_trailing_movers(("long", "short"), source="manual")
        return {"ok": False, "note": f"Unknown entry-assist command: {command or '(empty)'}."}

    def entry_assist_auto_tick(self):
        """Auto mode: run the whole entry-assist cycle without clicks.

        Strong regimes: open a window when SPY pauses against the tape and
        close it (emitting the ranked list) when the tape resumes. Weak and
        chop regimes: emit the trailing-30m list(s) every 30 minutes.
        """
        env = self.get_market_environment()
        mode = entry_assist_mode_for_env(env)
        window = getattr(self, "_entry_window", None)
        if mode["mode"] == "window":
            spy_today, _prev = self._spy_session_bars()
            if not spy_today:
                return
            trend_side = "short" if env.startswith("bearish") else "long"
            pause_start = self._detect_spy_pause_start(spy_today, trend_side)
            if window is None and pause_start is not None:
                self.start_entry_window(mode["sides"], source="auto", start_dt=pause_start)
            elif window is not None and window.get("source") == "auto" and pause_start is None:
                # Tape resumed: the pullback/bounce is over - rank and emit.
                self.end_entry_window(source="auto")
            return
        # A regime flip mid-window (e.g. strong -> weak) strands a window;
        # drop it quietly so the button and auto stay in sync.
        if window is not None and window.get("source") == "auto":
            self._entry_window = None
        now = time.time()
        if now - float(getattr(self, "_entry_movers_last_ts", 0.0)) >= ENTRY_AUTO_MOVERS_INTERVAL_MIN * 60.0:
            self._entry_movers_last_ts = now
            self.emit_trailing_movers(mode["sides"], source="auto")

    def _regime_pause_observation_store(self, today):
        store = getattr(self, "_regime_pause_observations", None)
        today_iso = today.isoformat()
        if not isinstance(store, dict) or store.get("date") != today_iso:
            store = {"date": today_iso, "sides": {"long": {}, "short": {}}}
            # Survive restarts: fold in counts already written for this day.
            try:
                path = Path(REGIME_PAUSE_OBSERVATIONS_FILE)
                if path.exists():
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and existing.get("date") == today_iso:
                        sides = existing.get("sides")
                        if isinstance(sides, dict):
                            for side_key in ("long", "short"):
                                side_map = sides.get(side_key)
                                if isinstance(side_map, dict):
                                    store["sides"][side_key] = {
                                        str(sym).strip().upper(): dict(entry)
                                        for sym, entry in side_map.items()
                                        if isinstance(entry, dict)
                                    }
            except Exception as exc:
                logging.debug("Regime-pause observations reload skipped: %s", exc)
            self._regime_pause_observations = store
        return store

    def _record_regime_pause_observation(self, symbol, side, today, *, day_excess, sym_day):
        store = self._regime_pause_observation_store(today)
        entries = store["sides"].setdefault(side, {})
        now_text = datetime.now().strftime("%H:%M")
        entry = entries.get(symbol)
        if not isinstance(entry, dict):
            entry = {"pause_count": 0, "first_seen": now_text, "max_day_excess_pct": None}
            entries[symbol] = entry
        entry["pause_count"] = int(entry.get("pause_count", 0) or 0) + 1
        entry["last_seen"] = now_text
        entry["last_day_pct"] = round(float(sym_day), 2)
        previous = entry.get("max_day_excess_pct")
        entry["max_day_excess_pct"] = round(
            max(float(day_excess), float(previous)) if previous is not None else float(day_excess),
            2,
        )

    def _save_regime_pause_observations(self):
        """Persist today's pause-defiance counts for the hourly master scan.

        Best-effort by design: the observations file is swing-scan evidence,
        never a blocker for the intraday sweep itself.
        """
        store = getattr(self, "_regime_pause_observations", None)
        if not isinstance(store, dict):
            return
        payload = {
            "schema_version": 1,
            "date": store.get("date", ""),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "sides": store.get("sides", {}),
        }
        try:
            path = Path(REGIME_PAUSE_OBSERVATIONS_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_name(f".{path.name}.tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temp_path, path)
        except Exception as exc:
            logging.warning("Regime-pause observations save failed: %s", exc)

    # ------------------------------------------------------------------
    # Trader-favorite day-trade sweeps (2026-07-04): delayed 5m opening-
    # range breaks and 8-EMA grind squeezes into a new session extreme.
    # Candidates come from longs.txt / shorts.txt only.
    # ------------------------------------------------------------------
    def _watchlist_day_sweep_symbols(self, side, symbols=None):
        watchlist = self.shorts if side == "short" else self.longs
        selected = {
            str(item or "").strip().upper() for item in watchlist if str(item or "").strip()
        }
        if symbols is not None:
            selected &= {
                str(item or "").strip().upper() for item in symbols if str(item or "").strip()
            }
        return sorted(selected)

    def _symbol_session_bars(self, symbol, today):
        bars = self.get_cached_5m_bars(symbol)
        return [bar for bar in bars or [] if bar.dt.date() == today]

    def check_orb_break_setups(self, symbols=None):
        """First 5m close through the opening range at 30+ minutes after open."""
        spy_today, _prev_close = self._spy_session_bars()
        if not spy_today:
            return []
        today = spy_today[-1].dt.date()
        state = self._orb_break_state
        if not state or state.get("date") != today:
            state = {"date": today, "alerted": set(), "dead": set()}
            self._orb_break_state = state
        hits = []
        for side in ("long", "short"):
            for symbol in self._watchlist_day_sweep_symbols(side, symbols):
                key = f"{symbol}|{side}"
                if key in state["alerted"] or key in state["dead"]:
                    continue
                hit = self._evaluate_orb_break(symbol, side, today, state)
                if hit:
                    state["alerted"].add(key)
                    hits.append(hit)
        for hit in hits:
            self._emit_orb_break_alert(hit)
        return hits

    def _evaluate_orb_break(self, symbol, side, today, state):
        sym_today = self._symbol_session_bars(symbol, today)
        if len(sym_today) < 2:
            return None
        opening = sym_today[0]
        # RTH bars: the session's first 5m candle is the opening range. Guard
        # against partial data where the cached series misses the open. Bar
        # timestamps are machine-local naive, so the open must be derived the
        # same way (9:30 ET is 06:30 on a Pacific box) - a hard-coded 9:30
        # check silently killed this sweep on non-Eastern machines.
        if opening.dt != get_market_session_open_naive(reference=opening.dt):
            return None
        earliest_break_dt = opening.dt + timedelta(minutes=ORB_DELAY_MINUTES)
        key = f"{symbol}|{side}"
        sign = 1.0 if side == "long" else -1.0
        level = opening.high if side == "long" else opening.low
        for index, bar in enumerate(sym_today[1:], start=1):
            if sign * (bar.close - level) <= 0:
                continue
            if bar.dt < earliest_break_dt:
                # Early close through the range: the delayed break is gone.
                state["dead"].add(key)
                return None
            if index < len(sym_today) - ORB_FRESH_BREAK_MAX_BARS:
                # First seen long after the fact (restart/stale cache): the
                # trigger already played out - don't alert hours late.
                state["dead"].add(key)
                return None
            return {
                "symbol": symbol,
                "side": side,
                "level": level,
                "or_high": opening.high,
                "or_low": opening.low,
                "break_bar": bar,
                "minutes_after_open": int((bar.dt - opening.dt).total_seconds() // 60),
            }
        return None

    def _emit_orb_break_alert(self, hit):
        symbol = hit["symbol"]
        side = hit["side"]
        bounce_type = ORB_BREAKOUT_TYPE if side == "long" else ORB_BREAKDOWN_TYPE
        break_bar = hit["break_bar"]
        bar_dict = {
            "time": break_bar.dt.strftime("%Y%m%d  %H:%M:%S"),
            "open": break_bar.open,
            "high": break_bar.high,
            "low": break_bar.low,
            "close": break_bar.close,
        }
        levels = {bounce_type: hit["level"]}
        edge = "high" if side == "long" else "low"
        event_row = self._log_bounce_candidate_event(
            "confirmed",
            symbol,
            side,
            levels,
            bar_dict,
            bar_dict,
            reason=(
                f"Delayed 5m ORB: first close through the opening-range {edge} "
                f"{hit['level']:.2f} at {break_bar.dt:%H:%M} "
                f"({hit['minutes_after_open']} min after open)"
            ),
        )
        self._register_bounce_outcome(symbol, side, levels, bar_dict, bar_dict, event_row.get("event_id", ""))
        quality = self._evaluate_bounce_alert_quality(side, levels, event_row)
        label = "ORB BREAKOUT" if side == "long" else "ORB BREAKDOWN"
        message = (
            f"[{quality.get('tier', 'B')}-TIER] {label} {symbol} ({side}): first 5m close "
            f"{'above' if side == 'long' else 'below'} the opening-range {edge} {hit['level']:.2f} "
            f"at {break_bar.dt:%H:%M} ({hit['minutes_after_open']} min after open); "
            f"OR {hit['or_low']:.2f}-{hit['or_high']:.2f}."
        )
        exit_note = self._measured_exit_suffix(side, levels)
        if exit_note:
            message += f" | {exit_note}"
        payload = self._build_bounce_feedback_alert_payload(message, event_row)
        if self.gui_callback:
            self.gui_callback(payload, "red" if side == "short" else "green")
        self.log_symbol(symbol, f"ALERT: {message}")
        self.log_bounce_to_file(
            symbol=symbol,
            direction=side,
            levels=levels,
            bounce_candle=bar_dict,
            current_candle=bar_dict,
            threshold=0.0,
            quality=quality,
        )

    def check_ema8_grind_setups(self, symbols=None):
        """Strong name rides the 5m 8-EMA, squeezes, then pushes to a new extreme."""
        spy_today, _prev_close = self._spy_session_bars()
        if not spy_today:
            return []
        today = spy_today[-1].dt.date()
        state = self._ema8_grind_state
        if not state or state.get("date") != today:
            state = {"date": today, "alerted": set()}
            self._ema8_grind_state = state
        hits = []
        for side in ("long", "short"):
            for symbol in self._watchlist_day_sweep_symbols(side, symbols):
                key = f"{symbol}|{side}"
                if key in state["alerted"]:
                    continue
                hit = self._evaluate_ema8_grind(symbol, side, today)
                if hit:
                    state["alerted"].add(key)
                    hits.append(hit)
        for hit in hits:
            self._emit_ema8_grind_alert(hit)
        return hits

    def _evaluate_ema8_grind(self, symbol, side, today):
        bars = self.get_cached_5m_bars(symbol)
        if not bars:
            return None
        sym_today = [bar for bar in bars if bar.dt.date() == today]
        if len(sym_today) < EMA8_GRIND_MIN_BARS + EMA8_GRIND_PULLBACK_MIN_BARS + 1:
            return None
        sign = 1.0 if side == "long" else -1.0
        push = sym_today[-1]
        prior = sym_today[:-1]
        # 1) The push bar prints a new session extreme (HOD longs / LOD shorts).
        if side == "long":
            prior_extreme = max(bar.high for bar in prior)
            if push.high <= prior_extreme:
                return None
            extreme_index = max(range(len(prior)), key=lambda i: prior[i].high)
        else:
            prior_extreme = min(bar.low for bar in prior)
            if push.low >= prior_extreme:
                return None
            extreme_index = min(range(len(prior)), key=lambda i: prior[i].low)
        # 2) The old extreme must be stale: a straight one-way rip has no
        #    pullback-and-grind and is not this setup.
        if len(prior) - 1 - extreme_index < EMA8_GRIND_PULLBACK_MIN_BARS:
            return None
        # 3) Strength gate: the day's move runs with the trade direction.
        session_open = sym_today[0].open
        if not session_open:
            return None
        day_pct = (push.close - session_open) / session_open * 100.0
        if sign * day_pct < EMA8_GRIND_MIN_DAY_PCT:
            return None
        # 4) The grind: not a single 5m close across the 8-EMA before the push.
        ema_values = _ema_series(bars, EMA8_GRIND_LENGTH)
        ema_today = ema_values[len(bars) - len(sym_today):]
        grind_bars = sym_today[-(EMA8_GRIND_MIN_BARS + 1):-1]
        grind_emas = ema_today[-(EMA8_GRIND_MIN_BARS + 1):-1]
        if len(grind_bars) < EMA8_GRIND_MIN_BARS:
            return None
        if any(sign * (bar.close - ema) < 0 for bar, ema in zip(grind_bars, grind_emas)):
            return None
        # 5) The squeeze: closes hugging the 8-EMA right before the push.
        atr = _wilder_atr_last(bars[:-1], IMPULSE_INTRADAY_ATR_PERIOD)
        if not atr:
            return None
        squeeze_gap = max(
            abs(bar.close - ema)
            for bar, ema in zip(
                grind_bars[-EMA8_GRIND_SQUEEZE_BARS:], grind_emas[-EMA8_GRIND_SQUEEZE_BARS:]
            )
        ) / atr
        if squeeze_gap > EMA8_GRIND_SQUEEZE_MAX_ATR:
            return None
        return {
            "symbol": symbol,
            "side": side,
            "push_bar": push,
            "ema8": ema_today[-1],
            "day_pct": day_pct,
            "squeeze_gap_atr": squeeze_gap,
            "grind_bars": len(grind_bars),
            "new_extreme": push.high if side == "long" else push.low,
        }

    def _emit_ema8_grind_alert(self, hit):
        symbol = hit["symbol"]
        side = hit["side"]
        bounce_type = EMA8_GRIND_HOD_TYPE if side == "long" else EMA8_GRIND_LOD_TYPE
        push_bar = hit["push_bar"]
        bar_dict = {
            "time": push_bar.dt.strftime("%Y%m%d  %H:%M:%S"),
            "open": push_bar.open,
            "high": push_bar.high,
            "low": push_bar.low,
            "close": push_bar.close,
        }
        levels = {bounce_type: hit["ema8"]}
        extreme_label = "HOD" if side == "long" else "LOD"
        event_row = self._log_bounce_candidate_event(
            "confirmed",
            symbol,
            side,
            levels,
            bar_dict,
            bar_dict,
            reason=(
                f"8-EMA grind: {hit['grind_bars']} bars holding the 5m 8-EMA "
                f"(squeeze {hit['squeeze_gap_atr']:.2f} ATR), day {hit['day_pct']:+.2f}%, "
                f"pushing into a new {extreme_label} {hit['new_extreme']:.2f}"
            ),
        )
        self._register_bounce_outcome(symbol, side, levels, bar_dict, bar_dict, event_row.get("event_id", ""))
        quality = self._evaluate_bounce_alert_quality(side, levels, event_row)
        message = (
            f"[{quality.get('tier', 'B')}-TIER] 8-EMA GRIND {symbol} ({side}): held the 5m 8-EMA "
            f"{hit['grind_bars']} bars, squeezed to {hit['squeeze_gap_atr']:.2f} ATR, now pushing "
            f"into a new {extreme_label} {hit['new_extreme']:.2f} (day {hit['day_pct']:+.2f}%, "
            f"8-EMA {hit['ema8']:.2f})."
        )
        exit_note = self._measured_exit_suffix(side, levels)
        if exit_note:
            message += f" | {exit_note}"
        payload = self._build_bounce_feedback_alert_payload(message, event_row)
        if self.gui_callback:
            self.gui_callback(payload, "red" if side == "short" else "green")
        self.log_symbol(symbol, f"ALERT: {message}")
        self.log_bounce_to_file(
            symbol=symbol,
            direction=side,
            levels=levels,
            bounce_candle=bar_dict,
            current_candle=bar_dict,
            threshold=0.0,
            quality=quality,
        )

    # ------------------------------------------------------------------
    # H1 candle-color sweeps (2026-07-06): the trader's hourly regime
    # colors under test. Longs from longs.txt: green-regime 10-EMA bounce
    # candles and blue-after-red reclaims. Shorts from shorts.txt: a green
    # H1 that fails straight to yellow. Signals fire once per closed H1
    # candle and feed the bounce outcome tracker for edge measurement.
    # ------------------------------------------------------------------
    def check_h1_color_setups(self):
        """Sweep watchlists for H1 color signals on the last closed hourly candle."""
        spy_today, _prev_close = self._spy_session_bars()
        if not spy_today:
            return []
        today = spy_today[-1].dt.date()
        state = self._h1_color_state
        if not state or state.get("date") != today:
            state = {"date": today, "alerted": set()}
            self._h1_color_state = state
        hits = []
        for side in ("long", "short"):
            for symbol in self._watchlist_day_sweep_symbols(side):
                for hit in self._evaluate_h1_color_signals(symbol, side, today):
                    key = f"{symbol}|{hit['type']}|{hit['signal_bar'].dt:%H:%M}"
                    if key in state["alerted"]:
                        continue
                    state["alerted"].add(key)
                    hits.append(hit)
        for hit in hits:
            self._emit_h1_color_alert(hit)
        return hits

    def _evaluate_h1_color_signals(self, symbol, side, today):
        bars = self.get_cached_5m_bars(symbol)
        if not bars:
            return []
        h1 = _closed_h1_bars(bars)
        if not h1 or h1[-1].dt.date() != today:
            # Only the freshly closed candle is a live signal; anything older
            # (restart, stale cache) already played out.
            return []
        spy_bars = self.get_cached_5m_bars("SPY")
        spy_h1 = _closed_h1_bars(spy_bars) if spy_bars else []
        if not spy_h1 or h1[-1].dt != spy_h1[-1].dt:
            # "Today" is not fresh enough: a symbol whose cache stopped an
            # hour ago must not replay that old H1 candle during a later broad
            # sweep. SPY is the session clock because it is continuously
            # refreshed and uses the same RTH bucket boundaries.
            return []
        hits = detect_h1_color_signals(h1, side)
        for hit in hits:
            hit["symbol"] = symbol
            hit["side"] = side
        return hits

    def _emit_h1_color_alert(self, hit):
        symbol = hit["symbol"]
        side = hit["side"]
        bounce_type = hit["type"]
        signal_bar = hit["signal_bar"]
        bar_dict = {
            "time": signal_bar.dt.strftime("%Y%m%d  %H:%M:%S"),
            "open": signal_bar.open,
            "high": signal_bar.high,
            "low": signal_bar.low,
            "close": signal_bar.close,
        }
        levels = {bounce_type: hit["level"]}
        event_row = self._log_bounce_candidate_event(
            "confirmed",
            symbol,
            side,
            levels,
            bar_dict,
            bar_dict,
            reason=f"H1 color signal: {hit['detail']} (H1 candle {signal_bar.dt:%H:%M})",
        )
        self._register_bounce_outcome(symbol, side, levels, bar_dict, bar_dict, event_row.get("event_id", ""))
        quality = self._evaluate_bounce_alert_quality(side, levels, event_row)
        label = BOUNCE_TYPE_LABELS.get(bounce_type, bounce_type).upper()
        message = (
            f"[{quality.get('tier', 'B')}-TIER] {label} {symbol} ({side}): {hit['detail']} "
            f"on the {signal_bar.dt:%H:%M} H1 candle."
        )
        exit_note = self._measured_exit_suffix(side, levels)
        if exit_note:
            message += f" | {exit_note}"
        payload = self._build_bounce_feedback_alert_payload(message, event_row)
        if self.gui_callback:
            self.gui_callback(payload, "red" if side == "short" else "green")
        self.log_symbol(symbol, f"ALERT: {message}")
        self.log_bounce_to_file(
            symbol=symbol,
            direction=side,
            levels=levels,
            bounce_candle=bar_dict,
            current_candle=bar_dict,
            threshold=0.0,
            quality=quality,
        )

    def _earnings_reaction_source_paths(self):
        paths = []
        for path in list(NASDAQ_EARNINGS_REACTION_CACHE_FILES) + list(MANUAL_EARNINGS_REACTION_FILES):
            candidate = Path(path)
            if candidate not in paths:
                paths.append(candidate)
        return paths

    def _earnings_reaction_source_signature(self, paths):
        signature = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                signature.append((str(path), None))
            else:
                signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def _previous_market_date_from_bars(self, bars, current_date):
        session_dates = sorted(
            {
                bar.dt.date()
                for bar in bars or []
                if getattr(bar, "dt", None) is not None and bar.dt.date() < current_date
            }
        )
        if session_dates:
            return session_dates[-1]
        return current_date - timedelta(days=1)

    def _load_earnings_reaction_symbols(self, market_date, previous_market_date):
        if market_date is None or previous_market_date is None:
            return set()

        source_paths = self._earnings_reaction_source_paths()
        source_signature = self._earnings_reaction_source_signature(source_paths)
        cache_key = (
            market_date.isoformat(),
            previous_market_date.isoformat(),
            source_signature,
        )
        cached = self.earnings_reaction_filter_cache
        if cached.get("key") == cache_key:
            return set(cached.get("symbols", set()))

        symbols = set()
        for path in source_paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logging.debug(f"Failed reading earnings reaction source {path}: {exc}")
                continue
            for row, fallback_date in self._iter_earnings_rows(payload):
                event_date = self._parse_earnings_event_date(row, fallback_date)
                if event_date is None:
                    continue
                release_bucket = self._earnings_release_bucket(row.get("time") or row.get("release_time"))
                if (
                    (event_date == previous_market_date and release_bucket == "AMC")
                    or (event_date == market_date and release_bucket == "BMO")
                ):
                    symbols.update(self._symbol_aliases(row.get("symbol") or row.get("ticker")))

        self.earnings_reaction_filter_cache = {"key": cache_key, "symbols": set(symbols)}
        return symbols

    def _iter_earnings_rows(self, payload):
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row, row.get("date")
            return

        if not isinstance(payload, dict):
            return

        dates_payload = payload.get("dates")
        if isinstance(dates_payload, dict):
            for date_key, entry in dates_payload.items():
                rows = entry.get("rows") if isinstance(entry, dict) else []
                for row in rows or []:
                    if isinstance(row, dict):
                        yield row, date_key

        for key in ("earnings", "events", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        yield row, row.get("date")

    def _parse_earnings_event_date(self, row, fallback_date=None):
        for value in (
            row.get("date") if isinstance(row, dict) else None,
            row.get("earningsDate") if isinstance(row, dict) else None,
            row.get("reportDate") if isinstance(row, dict) else None,
            fallback_date,
        ):
            try:
                text = str(value or "").strip()
                if not text:
                    continue
                return datetime.fromisoformat(text[:10]).date()
            except ValueError:
                continue
        return None

    def _earnings_release_bucket(self, value):
        text = str(value or "").strip().lower().replace("_", "-")
        if not text:
            return ""
        if text in {"bmo", "pre", "pre-market", "premarket"}:
            return "BMO"
        if text in {"amc", "post", "post-market", "after-hours", "afterhours"}:
            return "AMC"
        if "pre" in text or "before" in text:
            return "BMO"
        if "after" in text or "post" in text or "close" in text:
            return "AMC"
        return str(value or "").strip().upper()

    def _symbol_aliases(self, symbol):
        text = str(symbol or "").strip().upper()
        if not text:
            return set()
        aliases = {text}
        if "." in text:
            aliases.add(text.replace(".", "-"))
        if "-" in text:
            aliases.add(text.replace("-", "."))
        return aliases

    def _symbol_matches_alias_set(self, symbol, aliases):
        return bool(self._symbol_aliases(symbol) & set(aliases or []))

    def _refresh_rrs_gui(self, status_msg=None):
        if not self.latest_rrs_payload or not self.gui_callback:
            return
        snapshot = self._decorate_snapshot(self.latest_rrs_payload)
        self.gui_callback(snapshot, "rrs_snapshot")
        if status_msg:
            self.gui_callback(status_msg, "rrs_status")

    def _decorate_snapshot(self, base_snapshot):
        decorated = dict(base_snapshot)
        env_key = self.get_market_environment()
        decorated["market_environment"] = env_key
        decorated["market_environment_label"] = MARKET_ENVIRONMENTS.get(env_key, {}).get("label", env_key)
        decorated["environment_highlights"] = self._build_environment_highlights(
            base_snapshot.get("symbol_context", []),
            base_snapshot.get("spy_move_ratio"),
            base_snapshot.get("environment_scan"),
        )
        return decorated

    def _environment_scan_is_active(self, latest_bar_dt=None):
        now_local = get_market_local_now()
        current_session = get_market_session_window(reference=now_local)
        first_hour_end = current_session.open_local + timedelta(minutes=ENVIRONMENT_SCAN_DELAY_MINUTES)
        if now_local < first_hour_end:
            return False
        if latest_bar_dt is None:
            return True
        latest_session = get_market_session_window(reference=latest_bar_dt)
        return latest_session.market_date == current_session.market_date

    def _build_environment_highlights(self, symbol_context, spy_ratio, environment_scan=None):
        if environment_scan is not None:
            sections = self._build_intraday_environment_sections(environment_scan)
            if sections:
                return sections
            scenario = self.get_market_environment()
            long_eval_bars = int(environment_scan.get("long_eval_window_count", 0) or 0)
            short_eval_bars = int(environment_scan.get("short_eval_window_count", 0) or 0)
            if scenario == "bullish_strong":
                message = "No intraday long leaders are available by average RRS yet."
            elif scenario == "bearish_strong":
                message = "No intraday short leaders are available by average RRS yet."
            elif scenario == "bullish_weak" and long_eval_bars <= 0:
                message = "No SPY-weak or compressed bars have been measured yet for bullish weak focus."
            elif scenario == "bearish_weak" and short_eval_bars <= 0:
                message = "No SPY-strong or compressed bars have been measured yet for bearish weak focus."
            else:
                message = "No intraday RS/RW candidates proved the required opposite-SPY behavior."
            return [
                {
                    "title": "Environment focus",
                    "rows": [{"text": message, "tag": "rrs_hdr"}],
                    "tag": "rrs_hdr",
                }
            ]

        scenario = self.get_market_environment()
        strong_entries = sorted(
            [entry for entry in symbol_context if entry.get("signal") == "RS"],
            key=lambda row: (-(row.get("excess_move_ratio") or 0.0), -row["rrs"]),
        )
        weak_entries = sorted(
            [entry for entry in symbol_context if entry.get("signal") == "RW"],
            key=lambda row: ((row.get("excess_move_ratio") or 0.0), row["rrs"]),
        )
        strong_long_entries = [entry for entry in strong_entries if entry.get("watchlist_bias") == "long"]
        weak_short_entries = [entry for entry in weak_entries if entry.get("watchlist_bias") == "short"]
        sections = []

        def append_section(title, entries, tag):
            section = self._build_section(title, entries, tag)
            if section:
                sections.append(section)

        if scenario == "bearish_strong":
            append_section("Weak Stocks (Bearish Strong focus)", weak_entries, "rrs_rw")
        elif scenario == "bearish_weak":
            append_section("Weak Stocks (Bearish Weak focus)", weak_entries, "rrs_rw")
            append_section("Strong Stocks (Bearish Weak focus)", strong_long_entries, "rrs_rs")
        elif scenario == "bullish_strong":
            append_section("Strong Stocks (Bullish Strong focus)", strong_entries, "rrs_rs")
        elif scenario == "bullish_weak":
            append_section("Strong Stocks (Bullish Weak focus)", strong_entries, "rrs_rs")
            append_section("Weak Stocks (Bullish Weak focus)", weak_short_entries, "rrs_rw")
        else:
            append_section("Strong Stocks", strong_entries, "rrs_rs")
            append_section("Weak Stocks", weak_entries, "rrs_rw")

        spy_section = self._build_spy_section(spy_ratio, weak_entries)
        if spy_section:
            sections.append(spy_section)

        if not sections:
            sections.append(
                {
                    "title": "Environment focus",
                    "rows": [{"text": "No RS/RW candidates available", "tag": "rrs_hdr"}],
                    "tag": "rrs_hdr",
                }
            )
        return sections

    def _build_spy_context_windows(self, spy_bars, length):
        if not spy_bars or len(spy_bars) < length + 2:
            return []

        current_date = spy_bars[-1].dt.date()
        windows = []
        for idx in range(length + 1, len(spy_bars)):
            bar = spy_bars[idx]
            if bar.dt.date() != current_date:
                continue
            # Score each SPY bar independently so the "weak" environments can
            # rank symbols against the exact opposite-tape bars instead of a
            # single blended session state.
            move_ratio = self._calc_move_ratio(spy_bars[:idx + 1], length)
            if move_ratio is None:
                continue
            prev_move_ratio = windows[-1]["move_ratio"] if windows else None
            delta = move_ratio - prev_move_ratio if prev_move_ratio is not None else 0.0
            compression = abs(move_ratio) < SPY_COMPRESSION_THRESHOLD
            # Only treat delta-only turns as valid context when SPY is near flat/compressed.
            # This avoids calling a small bounce inside a weak tape "SPY strength".
            rising = move_ratio >= SPY_UP_THRESHOLD or (
                delta >= SPY_PULLBACK_DELTA_THRESHOLD and move_ratio >= -SPY_COMPRESSION_THRESHOLD
            )
            falling = move_ratio <= -SPY_UP_THRESHOLD or (
                delta <= -SPY_PULLBACK_DELTA_THRESHOLD and move_ratio <= SPY_COMPRESSION_THRESHOLD
            )
            windows.append(
                {
                    "dt": bar.dt,
                    "move_ratio": move_ratio,
                    "delta": delta,
                    "compression": compression,
                    "spy_weak": falling,
                    "spy_strong": rising,
                    "long_eval": compression or falling,
                    "short_eval": compression or rising,
                }
            )
        return windows

    def _summarize_environment_scan(self, intraday_profiles, spy_windows, threshold):
        if not intraday_profiles:
            return None

        window_map = {item["dt"]: item for item in spy_windows}
        long_candidates = []
        short_candidates = []
        weak_spy_window_count = sum(1 for item in spy_windows if item.get("spy_weak"))
        strong_spy_window_count = sum(1 for item in spy_windows if item.get("spy_strong"))
        compression_window_count = sum(1 for item in spy_windows if item.get("compression"))
        long_eval_window_count = sum(1 for item in spy_windows if item.get("long_eval"))
        short_eval_window_count = sum(1 for item in spy_windows if item.get("short_eval"))

        def summarize_bucket(profile, direction, flag_key=None):
            samples = []
            hits = []
            for item in profile:
                if item.get("rrs") is None:
                    continue
                if flag_key:
                    window = window_map.get(item.get("dt"))
                    if not window or not window.get(flag_key):
                        continue
                samples.append(item)
                if flag_key:
                    window = window_map.get(item.get("dt"))
                    if self._profile_item_matches_environment_context(
                        item,
                        direction,
                        window,
                        threshold,
                    ):
                        hits.append(item)
                elif direction == "long":
                    if item["rrs"] >= threshold:
                        hits.append(item)
                else:
                    if item["rrs"] <= -threshold:
                        hits.append(item)

            sample_rrs = [item["rrs"] for item in samples if item.get("rrs") is not None]
            sample_excess = [item["excess_move_ratio"] for item in samples if item.get("excess_move_ratio") is not None]
            avg_rrs = (sum(sample_rrs) / len(sample_rrs)) if sample_rrs else None
            best_rrs = None
            if sample_rrs:
                best_rrs = max(sample_rrs) if direction == "long" else min(sample_rrs)
            avg_excess = (sum(sample_excess) / len(sample_excess)) if sample_excess else None
            return {
                "windows": len(samples),
                "hits": len(hits),
                "hit_rate": (len(hits) / len(samples)) if samples else 0.0,
                "avg_rrs": avg_rrs,
                "best_rrs": best_rrs,
                "avg_excess": avg_excess,
            }

        def _signed_rrs(value, direction):
            if value is None:
                return None
            return value if direction == "long" else -value

        def finalize_candidate(symbol, direction, profile, overall_summary, context_summary, compression_summary):
            if overall_summary["windows"] <= 0:
                return None
            latest_profile = profile[-1] if profile else {}
            recent_profile = profile[-ENVIRONMENT_RECENT_PROFILE_BARS:] if profile else []
            recent_profile_hits = sum(
                1
                for item in recent_profile
                if self._profile_item_matches_direction(item, direction, require_significant=True)
            )
            recent_profile_hit_rate = (
                recent_profile_hits / len(recent_profile) if recent_profile else 0.0
            )
            current_direction_ok = self._profile_item_matches_direction(
                latest_profile,
                direction,
                require_significant=True,
            )
            current_countertrend = self._profile_item_is_countertrend(latest_profile, direction)
            current_rrs = latest_profile.get("rrs") if latest_profile else None
            current_move_ratio = latest_profile.get("move_ratio") if latest_profile else None
            current_excess = latest_profile.get("excess_move_ratio") if latest_profile else None
            current_direction_bias = self._profile_direction_bias_score(
                latest_profile,
                direction,
                threshold,
            )
            recent_direction_bias = self._ema_profile_direction_bias(
                recent_profile,
                direction,
                threshold,
            )
            total_direction_bias = self._ema_profile_direction_bias(
                profile,
                direction,
                threshold,
            )
            active_direction_bias_ok = (
                not current_countertrend and (
                    current_direction_ok
                    or (
                        recent_direction_bias is not None
                        and recent_direction_bias >= ENVIRONMENT_RECENT_BIAS_MIN_SCORE
                        and total_direction_bias is not None
                        and total_direction_bias >= ENVIRONMENT_TOTAL_BIAS_MIN_SCORE
                    )
                )
            )
            overall_signed_avg_rrs = _signed_rrs(overall_summary["avg_rrs"], direction)
            if (
                (overall_signed_avg_rrs is None or overall_signed_avg_rrs <= 0)
                and overall_summary["hits"] <= 0
                and context_summary["hits"] <= 0
                and compression_summary["hits"] <= 0
                and recent_profile_hits <= 0
                and not current_direction_ok
            ):
                return None
            context_score = (
                context_summary["hits"] * 160.0
                + context_summary["hit_rate"] * 110.0
                + (abs(context_summary["avg_rrs"]) * 18.0 if context_summary["avg_rrs"] is not None else 0.0)
                + (abs(context_summary["best_rrs"]) * 8.0 if context_summary["best_rrs"] is not None else 0.0)
                + (abs(context_summary["avg_excess"]) * 10.0 if context_summary["avg_excess"] is not None else 0.0)
                + compression_summary["hits"] * 55.0
                + compression_summary["hit_rate"] * 35.0
                + (abs(compression_summary["avg_rrs"]) * 6.0 if compression_summary["avg_rrs"] is not None else 0.0)
                + (abs(compression_summary["best_rrs"]) * 3.0 if compression_summary["best_rrs"] is not None else 0.0)
                + (abs(compression_summary["avg_excess"]) * 4.0 if compression_summary["avg_excess"] is not None else 0.0)
                + recent_profile_hits * 75.0
                + recent_profile_hit_rate * 50.0
                + (total_direction_bias * 70.0 if total_direction_bias is not None else 0.0)
                + (recent_direction_bias * 110.0 if recent_direction_bias is not None else 0.0)
                + (current_direction_bias * 140.0 if current_direction_bias is not None else 0.0)
            )
            overall_score = (
                overall_summary["hits"] * 120.0
                + overall_summary["hit_rate"] * 85.0
                + (abs(overall_summary["avg_rrs"]) * 95.0 if overall_summary["avg_rrs"] is not None else 0.0)
                + (abs(overall_summary["best_rrs"]) * 16.0 if overall_summary["best_rrs"] is not None else 0.0)
                + (abs(overall_summary["avg_excess"]) * 18.0 if overall_summary["avg_excess"] is not None else 0.0)
                + context_summary["hits"] * 28.0
                + context_summary["hit_rate"] * 18.0
                + recent_profile_hits * 75.0
                + recent_profile_hit_rate * 55.0
                + (total_direction_bias * 75.0 if total_direction_bias is not None else 0.0)
                + (recent_direction_bias * 120.0 if recent_direction_bias is not None else 0.0)
                + (current_direction_bias * 150.0 if current_direction_bias is not None else 0.0)
            )
            if current_direction_ok:
                context_score += 35.0
                overall_score += 35.0
            if current_countertrend:
                context_score -= 140.0
                overall_score -= 140.0
            elif not active_direction_bias_ok:
                context_score -= 45.0
                overall_score -= 45.0
            return {
                "symbol": symbol,
                "direction": direction,
                "overall_hits": overall_summary["hits"],
                "overall_windows": overall_summary["windows"],
                "overall_hit_rate": overall_summary["hit_rate"],
                "overall_avg_rrs": overall_summary["avg_rrs"],
                "overall_best_rrs": overall_summary["best_rrs"],
                "overall_avg_excess": overall_summary["avg_excess"],
                "context_hits": context_summary["hits"],
                "context_windows": context_summary["windows"],
                "context_hit_rate": context_summary["hit_rate"],
                "context_avg_rrs": context_summary["avg_rrs"],
                "context_best_rrs": context_summary["best_rrs"],
                "context_avg_excess": context_summary["avg_excess"],
                "compression_hits": compression_summary["hits"],
                "compression_windows": compression_summary["windows"],
                "compression_hit_rate": compression_summary["hit_rate"],
                "compression_avg_rrs": compression_summary["avg_rrs"],
                "compression_best_rrs": compression_summary["best_rrs"],
                "compression_avg_excess": compression_summary["avg_excess"],
                "current_rrs": current_rrs,
                "current_move_ratio": current_move_ratio,
                "current_excess": current_excess,
                "current_direction_ok": current_direction_ok,
                "current_countertrend": current_countertrend,
                "current_direction_bias": current_direction_bias,
                "recent_direction_bias": recent_direction_bias,
                "total_direction_bias": total_direction_bias,
                "active_direction_bias_ok": active_direction_bias_ok,
                "recent_profile_hits": recent_profile_hits,
                "recent_profile_hit_rate": recent_profile_hit_rate,
                "context_score": context_score,
                "overall_score": overall_score,
                "score": context_score,
            }

        for symbol, profile in intraday_profiles.items():
            if not profile:
                continue

            direction = self.get_symbol_direction(symbol)
            if direction == "long":
                overall_summary = summarize_bucket(profile, "long")
                context_summary = summarize_bucket(profile, "long", "long_eval")
                compression_summary = summarize_bucket(profile, "long", "compression")
                candidate = finalize_candidate(
                    symbol,
                    "long",
                    profile,
                    overall_summary,
                    context_summary,
                    compression_summary,
                )
                if candidate:
                    long_candidates.append(candidate)

            if direction == "short":
                overall_summary = summarize_bucket(profile, "short")
                context_summary = summarize_bucket(profile, "short", "short_eval")
                compression_summary = summarize_bucket(profile, "short", "compression")
                candidate = finalize_candidate(
                    symbol,
                    "short",
                    profile,
                    overall_summary,
                    context_summary,
                    compression_summary,
                )
                if candidate:
                    short_candidates.append(candidate)

        long_candidates.sort(
            key=lambda row: (
                0 if row.get("active_direction_bias_ok") else 1,
                0 if row.get("current_direction_ok") else 1,
                -(
                    row.get("overall_avg_rrs")
                    if row.get("overall_avg_rrs") is not None else float("-inf")
                ),
                -(
                    row.get("recent_direction_bias")
                    if row.get("recent_direction_bias") is not None else float("-inf")
                ),
                -(
                    row.get("current_direction_bias")
                    if row.get("current_direction_bias") is not None else float("-inf")
                ),
                -(row.get("recent_profile_hits", 0) or 0),
                -(row.get("recent_profile_hit_rate", 0.0) or 0.0),
                -(row.get("overall_score", 0.0) or 0.0),
                -row["score"],
                -row["context_hits"],
                -(row["context_avg_rrs"] if row["context_avg_rrs"] is not None else float("-inf")),
                row["symbol"],
            )
        )
        short_candidates.sort(
            key=lambda row: (
                0 if row.get("active_direction_bias_ok") else 1,
                0 if row.get("current_direction_ok") else 1,
                (
                    row.get("overall_avg_rrs")
                    if row.get("overall_avg_rrs") is not None else float("inf")
                ),
                -(
                    row.get("recent_direction_bias")
                    if row.get("recent_direction_bias") is not None else float("-inf")
                ),
                -(
                    row.get("current_direction_bias")
                    if row.get("current_direction_bias") is not None else float("-inf")
                ),
                -(row.get("recent_profile_hits", 0) or 0),
                -(row.get("recent_profile_hit_rate", 0.0) or 0.0),
                -(row.get("overall_score", 0.0) or 0.0),
                -row["score"],
                -row["context_hits"],
                (row["context_avg_rrs"] if row["context_avg_rrs"] is not None else float("inf")),
                row["symbol"],
            )
        )
        return {
            "long_candidates": long_candidates,
            "short_candidates": short_candidates,
            "weak_spy_window_count": weak_spy_window_count,
            "strong_spy_window_count": strong_spy_window_count,
            "compression_window_count": compression_window_count,
            "long_eval_window_count": long_eval_window_count,
            "short_eval_window_count": short_eval_window_count,
            "spy_windows": len(spy_windows),
        }

    def _build_intraday_environment_sections(self, environment_scan):
        scenario = self.get_market_environment()
        long_candidates = environment_scan.get("long_candidates", [])
        short_candidates = environment_scan.get("short_candidates", [])
        weak_spy_windows = int(environment_scan.get("weak_spy_window_count", 0) or 0)
        strong_spy_windows = int(environment_scan.get("strong_spy_window_count", 0) or 0)
        compression_windows = int(environment_scan.get("compression_window_count", 0) or 0)
        long_eval_windows = int(environment_scan.get("long_eval_window_count", 0) or 0)
        short_eval_windows = int(environment_scan.get("short_eval_window_count", 0) or 0)
        sections = []

        def _signed_entry_rrs(entry, key):
            value = entry.get(key)
            if value is None:
                return None
            return value if entry.get("direction") == "long" else -value

        def _descending(value):
            return -(value if value is not None else float("-inf"))

        def candidate_sort_key(entry, focus_mode):
            if focus_mode == "overall":
                primary_avg = _signed_entry_rrs(entry, "overall_avg_rrs")
                primary_hit_rate = entry.get("overall_hit_rate")
                primary_hits = entry.get("overall_hits", 0)
                score_value = entry.get("overall_score", 0.0)
            elif focus_mode == "compression":
                primary_avg = _signed_entry_rrs(entry, "compression_avg_rrs")
                primary_hit_rate = entry.get("compression_hit_rate")
                primary_hits = entry.get("compression_hits", 0)
                score_value = entry.get("score", 0.0)
            else:
                primary_avg = _signed_entry_rrs(entry, "context_avg_rrs")
                primary_hit_rate = entry.get("context_hit_rate")
                primary_hits = entry.get("context_hits", 0)
                score_value = entry.get("context_score", entry.get("score", 0.0))
            return (
                0 if entry.get("active_direction_bias_ok") else 1,
                _descending(primary_avg),
                -float(primary_hit_rate or 0.0),
                -int(primary_hits or 0),
                0 if entry.get("current_direction_ok") else 1,
                _descending(
                    entry.get("recent_direction_bias")
                    if entry.get("recent_direction_bias") is not None else None
                ),
                _descending(
                    entry.get("current_direction_bias")
                    if entry.get("current_direction_bias") is not None else None
                ),
                -(entry.get("recent_profile_hits", 0) or 0),
                -(entry.get("recent_profile_hit_rate", 0.0) or 0.0),
                _descending(_signed_entry_rrs(entry, "overall_avg_rrs")),
                -(score_value or 0.0),
                entry.get("symbol", ""),
            )

        def focus_metric_present(entry, focus_mode):
            if focus_mode == "overall":
                primary_avg = _signed_entry_rrs(entry, "overall_avg_rrs")
                primary_hits = entry.get("overall_hits", 0)
            elif focus_mode == "compression":
                primary_avg = _signed_entry_rrs(entry, "compression_avg_rrs")
                primary_hits = entry.get("compression_hits", 0)
            else:
                primary_avg = _signed_entry_rrs(entry, "context_avg_rrs")
                primary_hits = entry.get("context_hits", 0)
            return (
                (primary_avg is not None and primary_avg > 0)
                or int(primary_hits or 0) > 0
            )

        def prepare_candidates(candidates, focus_mode):
            ordered = sorted(candidates, key=lambda entry: candidate_sort_key(entry, focus_mode))
            return [entry for entry in ordered if focus_metric_present(entry, focus_mode)]

        long_overall_candidates = prepare_candidates(long_candidates, "overall")
        short_overall_candidates = prepare_candidates(short_candidates, "overall")
        long_context_candidates = prepare_candidates(
            [entry for entry in long_candidates if entry.get("context_hits", 0) > 0],
            "context",
        )
        short_context_candidates = prepare_candidates(
            [entry for entry in short_candidates if entry.get("context_hits", 0) > 0],
            "context",
        )

        def add_candidate_section(title, candidates, tag, focus_mode):
            if not candidates:
                return
            rows = []
            for entry in candidates[:ENVIRONMENT_SCAN_LIMIT]:
                rows.append({
                    "text": self._format_environment_scan_entry(entry, focus_mode=focus_mode),
                    "tag": tag,
                    "symbol": entry.get("symbol"),
                })
            sections.append({"title": title, "rows": rows, "tag": tag})

        if scenario == "bullish_strong":
            add_candidate_section(
                "Strongest Longs by Avg RRS",
                long_overall_candidates,
                "rrs_rs",
                "overall",
            )
        elif scenario == "bearish_strong":
            add_candidate_section(
                "Weakest Shorts by Avg RRS",
                short_overall_candidates,
                "rrs_rw",
                "overall",
            )
        elif scenario == "bullish_weak":
            add_candidate_section(
                (
                    f"Strong Longs When SPY Weak/Compressed "
                    f"({long_eval_windows} bars; {weak_spy_windows} weak, {compression_windows} compressed)"
                ),
                long_context_candidates if long_eval_windows > 0 else [],
                "rrs_rs",
                "context",
            )
        elif scenario == "bearish_weak":
            add_candidate_section(
                (
                    f"Weak Shorts When SPY Strong/Compressed "
                    f"({short_eval_windows} bars; {strong_spy_windows} strong, {compression_windows} compressed)"
                ),
                short_context_candidates if short_eval_windows > 0 else [],
                "rrs_rw",
                "context",
            )
        else:
            add_candidate_section("Strongest Longs by Avg RRS", long_overall_candidates, "rrs_rs", "overall")
            add_candidate_section("Weakest Shorts by Avg RRS", short_overall_candidates, "rrs_rw", "overall")

        if sections:
            return sections
        return None

    def _record_environment_focus_history(self, environment_scan, timeframe_key):
        if not isinstance(environment_scan, dict):
            return

        now_local = get_market_local_now()
        current_date = now_local.date().isoformat()
        history = _load_environment_focus_history_payload()
        days = history.setdefault("days", {})

        cutoff_date = now_local.date() - timedelta(days=ENVIRONMENT_FOCUS_HISTORY_KEEP_DAYS)
        for day_key in list(days.keys()):
            try:
                day_value = datetime.fromisoformat(str(day_key)).date()
            except ValueError:
                days.pop(day_key, None)
                continue
            if day_value < cutoff_date:
                days.pop(day_key, None)

        day_payload = days.setdefault(
            current_date,
            {
                "updated_at": now_local.isoformat(timespec="seconds"),
                "bullish_weak_longs": {"symbols": {}},
                "bearish_weak_shorts": {"symbols": {}},
            },
        )
        day_payload["updated_at"] = now_local.isoformat(timespec="seconds")

        bucket_inputs = (
            (
                "bullish_weak_longs",
                [entry for entry in environment_scan.get("long_candidates", []) if int(entry.get("context_hits", 0) or 0) > 0],
            ),
            (
                "bearish_weak_shorts",
                [entry for entry in environment_scan.get("short_candidates", []) if int(entry.get("context_hits", 0) or 0) > 0],
            ),
        )

        current_environment = self.get_market_environment()
        for bucket_name, candidates in bucket_inputs:
            bucket_payload = day_payload.setdefault(bucket_name, {"symbols": {}})
            bucket_payload["updated_at"] = now_local.isoformat(timespec="seconds")
            symbols_payload = bucket_payload.setdefault("symbols", {})
            for entry in candidates:
                symbol = str(entry.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                existing = symbols_payload.setdefault(
                    symbol,
                    {
                        "first_seen_at": now_local.isoformat(timespec="seconds"),
                        "last_seen_at": now_local.isoformat(timespec="seconds"),
                        "hit_count": 0,
                        "max_score": 0.0,
                        "max_context_hits": 0,
                        "max_context_hit_rate": 0.0,
                        "max_compression_hits": 0,
                        "max_compression_hit_rate": 0.0,
                        "timeframes": [],
                        "market_environments": [],
                    },
                )
                existing["last_seen_at"] = now_local.isoformat(timespec="seconds")
                existing["hit_count"] = int(existing.get("hit_count", 0) or 0) + 1
                existing["max_score"] = max(
                    float(existing.get("max_score", 0.0) or 0.0),
                    float(entry.get("score", 0.0) or 0.0),
                )
                existing["max_context_hits"] = max(
                    int(existing.get("max_context_hits", 0) or 0),
                    int(entry.get("context_hits", 0) or 0),
                )
                existing["max_context_hit_rate"] = max(
                    float(existing.get("max_context_hit_rate", 0.0) or 0.0),
                    float(entry.get("context_hit_rate", 0.0) or 0.0),
                )
                existing["max_compression_hits"] = max(
                    int(existing.get("max_compression_hits", 0) or 0),
                    int(entry.get("compression_hits", 0) or 0),
                )
                existing["max_compression_hit_rate"] = max(
                    float(existing.get("max_compression_hit_rate", 0.0) or 0.0),
                    float(entry.get("compression_hit_rate", 0.0) or 0.0),
                )
                existing["timeframes"] = sorted(
                    {
                        str(item).strip()
                        for item in list(existing.get("timeframes", []) or []) + [timeframe_key]
                        if str(item).strip()
                    }
                )
                existing["market_environments"] = sorted(
                    {
                        str(item).strip()
                        for item in list(existing.get("market_environments", []) or []) + [current_environment]
                        if str(item).strip()
                    }
                )

        _save_environment_focus_history_payload(history)

    def _format_environment_scan_entry(self, entry, focus_mode="context"):
        direction = entry.get("direction")
        if focus_mode == "overall":
            focus_label = "dayRS" if direction == "long" else "dayRW"
            focus_hits = entry.get("overall_hits", 0)
            focus_windows = entry.get("overall_windows", 0)
            avg_rrs = entry.get("overall_avg_rrs")
            best_rrs = entry.get("overall_best_rrs")
            avg_excess = entry.get("overall_avg_excess")
            secondary_label = "weakSPY" if direction == "long" else "strongSPY"
            secondary_hits = entry.get("context_hits", 0)
            secondary_windows = entry.get("context_windows", 0)
        elif focus_mode == "compression":
            focus_label = "comp"
            focus_hits = entry.get("compression_hits", 0)
            focus_windows = entry.get("compression_windows", 0)
            avg_rrs = entry.get("compression_avg_rrs")
            best_rrs = entry.get("compression_best_rrs")
            avg_excess = entry.get("compression_avg_excess")
            secondary_label = "weakSPY" if direction == "long" else "strongSPY"
            secondary_hits = entry.get("context_hits", 0)
            secondary_windows = entry.get("context_windows", 0)
        else:
            focus_label = "weak/compSPY" if direction == "long" else "strong/compSPY"
            focus_hits = entry.get("context_hits", 0)
            focus_windows = entry.get("context_windows", 0)
            avg_rrs = entry.get("context_avg_rrs")
            best_rrs = entry.get("context_best_rrs")
            avg_excess = entry.get("context_avg_excess")
            secondary_label = "comp"
            secondary_hits = entry.get("compression_hits", 0)
            secondary_windows = entry.get("compression_windows", 0)

        avg_rrs_text = f"{avg_rrs:+.2f}" if avg_rrs is not None else "n/a"
        best_rrs_text = f"{best_rrs:+.2f}" if best_rrs is not None else "n/a"
        line = (
            f"{entry['symbol']} {focus_label}={focus_hits}/{focus_windows} "
            f"avgRRS={avg_rrs_text} best={best_rrs_text}"
        )
        if avg_excess is not None:
            line += f" avgER={avg_excess:+.2f}"
        current_excess = entry.get("current_excess")
        if current_excess is not None:
            line += f" nowER={current_excess:+.2f}"
        recent_direction_bias = entry.get("recent_direction_bias")
        if recent_direction_bias is not None:
            line += f" ema={recent_direction_bias:+.2f}"
        current_direction_bias = entry.get("current_direction_bias")
        if current_direction_bias is not None:
            line += f" nowBias={current_direction_bias:+.2f}"
        if secondary_windows:
            line += f" {secondary_label}={secondary_hits}/{secondary_windows}"
        return line

    def _build_section(self, title, entries, tag):
        if not entries:
            return None
        significant = [entry for entry in entries if self._is_move_significant(entry)]
        source = significant if significant else entries
        rows = []
        for entry in source[:ENVIRONMENT_HIGHLIGHT_LIMIT]:
            rows.append({
                "text": self._format_environment_entry(entry),
                "tag": tag,
                "symbol": entry.get("symbol"),
            })
        if not rows:
            return None
        return {"title": title, "rows": rows, "tag": tag}

    def _build_spy_section(self, spy_ratio, weak_entries):
        if spy_ratio is None or not weak_entries:
            return None
        if spy_ratio >= SPY_UP_THRESHOLD:
            title = "SPY rising / flat -> weak stocks to watch"
        elif abs(spy_ratio) < SPY_COMPRESSION_THRESHOLD:
            title = "SPY compressing -> weak names to monitor"
        else:
            return None
        return self._build_section(title, weak_entries, "rrs_rw")

    def _format_environment_entry(self, entry):
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        move_text = f"{move_ratio:+.2f}" if move_ratio is not None else "n/a"
        excess_text = f"{excess_move_ratio:+.2f}" if excess_move_ratio is not None else "n/a"
        signal = entry.get("signal", "")
        marker = "UP" if signal == "RS" else "DN"
        return f"{entry['symbol']} {marker} {signal} {entry['rrs']:+.2f} ER={excess_text} MR={move_text}"

    def _is_move_significant(self, entry):
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        if move_ratio is None or excess_move_ratio is None:
            return False
        return (
            abs(move_ratio) >= MIN_MOVE_RATIO_FOR_SIGNAL
            and abs(excess_move_ratio) >= MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL
        )

    def _profile_item_matches_direction(self, entry, direction, require_significant=False):
        if not isinstance(entry, dict):
            return False
        rrs_value = entry.get("rrs")
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        if rrs_value is None:
            return False
        if require_significant:
            if move_ratio is None or excess_move_ratio is None:
                return False
            if direction == "long":
                return (
                    move_ratio >= MIN_MOVE_RATIO_FOR_SIGNAL
                    and excess_move_ratio >= MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL
                )
            return (
                move_ratio <= -MIN_MOVE_RATIO_FOR_SIGNAL
                and excess_move_ratio <= -MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL
            )
        if direction == "long":
            return (
                rrs_value > 0
                and (move_ratio is None or move_ratio > 0)
                and (excess_move_ratio is None or excess_move_ratio > 0)
            )
        return (
            rrs_value < 0
            and (move_ratio is None or move_ratio < 0)
            and (excess_move_ratio is None or excess_move_ratio < 0)
        )

    def _profile_item_is_countertrend(self, entry, direction):
        opposite_direction = "short" if direction == "long" else "long"
        return self._profile_item_matches_direction(
            entry,
            opposite_direction,
            require_significant=True,
        )

    def _profile_item_matches_environment_context(self, entry, direction, spy_window, threshold):
        if not isinstance(entry, dict) or not isinstance(spy_window, dict):
            return False

        rrs_value = entry.get("rrs")
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        if rrs_value is None or move_ratio is None or excess_move_ratio is None:
            return False

        signed_rrs = rrs_value if direction == "long" else -rrs_value
        signed_move = move_ratio if direction == "long" else -move_ratio
        signed_excess = excess_move_ratio if direction == "long" else -excess_move_ratio
        min_rrs = max(0.0, abs(float(threshold or RRS_DEFAULT_THRESHOLD)) * ENVIRONMENT_CONTEXT_MIN_RRS_FRACTION)
        if signed_rrs < min_rrs or signed_excess < MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL:
            return False

        opposite_spy_move = (
            (direction == "long" and spy_window.get("spy_weak"))
            or (direction == "short" and spy_window.get("spy_strong"))
        )
        if opposite_spy_move:
            return signed_move >= -ENVIRONMENT_HOLD_MOVE_RATIO
        if spy_window.get("compression"):
            return signed_move >= MIN_MOVE_RATIO_FOR_SIGNAL
        return False

    def _profile_direction_bias_score(self, entry, direction, threshold):
        if not isinstance(entry, dict):
            return None

        threshold = abs(float(threshold or 0.0))
        if threshold <= 0.0:
            threshold = float(RRS_DEFAULT_THRESHOLD)

        def clamp(value, limit=3.0):
            return max(-limit, min(limit, value))

        components = []

        rrs_value = entry.get("rrs")
        if rrs_value is not None:
            signed_rrs = rrs_value if direction == "long" else -rrs_value
            components.append(clamp(signed_rrs / threshold))

        move_ratio = entry.get("move_ratio")
        if move_ratio is not None and MIN_MOVE_RATIO_FOR_SIGNAL > 0:
            signed_move = move_ratio if direction == "long" else -move_ratio
            components.append(clamp(signed_move / MIN_MOVE_RATIO_FOR_SIGNAL))

        excess_move_ratio = entry.get("excess_move_ratio")
        if excess_move_ratio is not None and MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL > 0:
            signed_excess = excess_move_ratio if direction == "long" else -excess_move_ratio
            components.append(clamp(signed_excess / MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL))

        if not components:
            return None
        return sum(components) / len(components)

    def _ema_profile_direction_bias(self, profile, direction, threshold):
        ema_value = None
        for item in profile or []:
            bias_score = self._profile_direction_bias_score(item, direction, threshold)
            if bias_score is None:
                continue
            if ema_value is None:
                ema_value = bias_score
            else:
                ema_value = (
                    (ENVIRONMENT_PROFILE_EMA_ALPHA * bias_score)
                    + ((1.0 - ENVIRONMENT_PROFILE_EMA_ALPHA) * ema_value)
                )
        return ema_value

    def _calc_move_ratio(self, bars, length):
        if not bars or len(bars) < length + 1:
            return None
        move = bars[-1].close - bars[-(length + 1)].close
        atr = _wilder_atr_last(bars[:-1], length)
        if atr is None or atr == 0:
            return None
        return move / atr

    def _build_intraday_rrs_profile(self, symbol_bars, spy_bars, length=IMPULSE_RRS_PROFILE_LENGTH):
        if not symbol_bars or not spy_bars:
            return []
        aligned_sym, aligned_spy = _align_bars_with_map(symbol_bars, {bar.dt: bar for bar in spy_bars})
        if len(aligned_sym) < length + 2 or len(aligned_spy) < length + 2:
            return []

        profile = []
        for idx in range(length + 1, len(aligned_sym)):
            sym_slice = aligned_sym[:idx + 1]
            spy_slice = aligned_spy[:idx + 1]
            rrs_value, power_index = real_relative_strength(sym_slice, spy_slice, length=length)
            if rrs_value is None:
                continue
            sym_move_ratio = self._calc_move_ratio(sym_slice, length)
            spy_move_ratio = self._calc_move_ratio(spy_slice, length)
            excess_move_ratio = None
            if sym_move_ratio is not None and spy_move_ratio is not None:
                excess_move_ratio = sym_move_ratio - spy_move_ratio
            profile.append(
                {
                    "dt": sym_slice[-1].dt,
                    "rrs": rrs_value,
                    "power_index": power_index,
                    "move_ratio": sym_move_ratio,
                    "spy_move_ratio": spy_move_ratio,
                    "excess_move_ratio": excess_move_ratio,
                }
            )
        return profile

    def _impulse_regime_transition_ok(self, symbol, direction, profile):
        if len(profile) < (IMPULSE_RRS_RECENT_PHASE_BARS + 2):
            logging.debug(f"{symbol}: Insufficient RS/RW profile points for impulse regime filter.")
            return False

        env = self.get_market_environment()
        favored_env = (
            (direction == "long" and env.startswith("bullish"))
            or (direction == "short" and env.startswith("bearish"))
        )
        weak_env = env.endswith("weak")

        pre_phase = profile[:-IMPULSE_RRS_RECENT_PHASE_BARS]
        recent_phase = profile[-IMPULSE_RRS_RECENT_PHASE_BARS:]
        if not pre_phase or not recent_phase:
            return False

        trend_ratio_req = (
            IMPULSE_RRS_FAVORED_TREND_RATIO if favored_env else IMPULSE_RRS_UNFAVORED_TREND_RATIO
        )
        excess_ratio_req = (
            IMPULSE_RRS_FAVORED_EXCESS_RATIO if (favored_env and weak_env) else 0.0
        )
        if not favored_env:
            excess_ratio_req = IMPULSE_RRS_UNFAVORED_EXCESS_RATIO

        if direction == "long":
            best_trend_ratio = max((item.get("move_ratio") for item in pre_phase if item.get("move_ratio") is not None), default=None)
            best_excess_ratio = max((item.get("excess_move_ratio") for item in pre_phase if item.get("excess_move_ratio") is not None), default=None)
            counter_rrs_count = sum(
                1 for item in recent_phase
                if (item.get("rrs") is not None and item.get("rrs") <= -IMPULSE_RRS_COUNTER_RRS)
            )
            counter_move_count = sum(
                1 for item in recent_phase
                if (item.get("move_ratio") is not None and item.get("move_ratio") <= -IMPULSE_RRS_COUNTER_MOVE_RATIO)
            )
            trend_ok = best_trend_ratio is not None and best_trend_ratio >= trend_ratio_req
            excess_ok = excess_ratio_req <= 0.0 or (best_excess_ratio is not None and best_excess_ratio >= excess_ratio_req)
        else:
            best_trend_ratio = min((item.get("move_ratio") for item in pre_phase if item.get("move_ratio") is not None), default=None)
            best_excess_ratio = min((item.get("excess_move_ratio") for item in pre_phase if item.get("excess_move_ratio") is not None), default=None)
            counter_rrs_count = sum(
                1 for item in recent_phase
                if (item.get("rrs") is not None and item.get("rrs") >= IMPULSE_RRS_COUNTER_RRS)
            )
            counter_move_count = sum(
                1 for item in recent_phase
                if (item.get("move_ratio") is not None and item.get("move_ratio") >= IMPULSE_RRS_COUNTER_MOVE_RATIO)
            )
            trend_ok = best_trend_ratio is not None and best_trend_ratio <= -trend_ratio_req
            excess_ok = excess_ratio_req <= 0.0 or (best_excess_ratio is not None and best_excess_ratio <= -excess_ratio_req)

        counter_ok = (
            counter_rrs_count >= IMPULSE_RRS_COUNTER_BARS_REQUIRED
            and counter_move_count >= IMPULSE_RRS_COUNTER_BARS_REQUIRED
        )

        if trend_ok and excess_ok and counter_ok:
            logging.debug(
                f"{symbol}: Impulse RS/RW regime accepted. Env={env}, Direction={direction}, "
                f"TrendRatio={best_trend_ratio}, ExcessRatio={best_excess_ratio}, "
                f"CounterRRSBars={counter_rrs_count}, CounterMoveBars={counter_move_count}"
            )
            return True

        logging.debug(
            f"{symbol}: Rejecting impulse retest on RS/RW regime filter. Env={env}, Direction={direction}, "
            f"TrendOK={trend_ok}, ExcessOK={excess_ok}, CounterOK={counter_ok}, "
            f"TrendRatio={best_trend_ratio}, ExcessRatio={best_excess_ratio}, "
            f"CounterRRSBars={counter_rrs_count}, CounterMoveBars={counter_move_count}"
        )
        return False

    def _build_intraday_level_series_maps(self, df, current_date):
        if "datetime" in df.columns and "typical_price" in df.columns:
            prepared = df.copy()
        else:
            prepared = self._prepare_vwap_frame(df)
        if prepared.empty:
            return {}

        today_prepared = prepared[prepared["datetime"].dt.date == current_date].copy()
        if today_prepared.empty:
            return {}

        series_maps = {}

        std_work = today_prepared.copy()
        std_work["cum_vol"] = std_work["volume"].cumsum()
        std_work["cum_vol_price"] = (std_work["typical_price"] * std_work["volume"]).cumsum()
        std_work["level_value"] = std_work["cum_vol_price"] / std_work["cum_vol"]
        series_maps["vwap"] = {
            row["datetime"]: float(row["level_value"])
            for _, row in std_work.iterrows()
        }

        previous_sessions = sorted(d for d in prepared["datetime"].dt.date.unique() if d < current_date)
        previous_session_df = (
            prepared[prepared["datetime"].dt.date == previous_sessions[-1]].copy()
            if previous_sessions
            else pd.DataFrame()
        )

        dynamic_source = (
            pd.concat([previous_session_df, today_prepared], ignore_index=True)
            if not previous_session_df.empty
            else today_prepared.copy()
        )
        dynamic_skip_rows = len(previous_session_df)
        dynamic_source["cum_vol"] = dynamic_source["volume"].cumsum()
        dynamic_source["cum_vol_price"] = (dynamic_source["typical_price"] * dynamic_source["volume"]).cumsum()
        dynamic_source["level_value"] = dynamic_source["cum_vol_price"] / dynamic_source["cum_vol"]
        series_maps["dynamic_vwap"] = {
            row["datetime"]: float(row["level_value"])
            for _, row in dynamic_source.iloc[dynamic_skip_rows:].iterrows()
        }

        if previous_sessions:
            prev_tail = previous_session_df.tail(1).copy()
            eod_source = pd.concat([prev_tail, today_prepared], ignore_index=True)
            skip_rows = len(prev_tail)
        else:
            eod_source = today_prepared.copy()
            skip_rows = 0

        eod_source["cum_vol"] = eod_source["volume"].cumsum()
        eod_source["cum_vol_price"] = (eod_source["typical_price"] * eod_source["volume"]).cumsum()
        eod_source["level_value"] = eod_source["cum_vol_price"] / eod_source["cum_vol"]
        series_maps["eod_vwap"] = {
            row["datetime"]: float(row["level_value"])
            for _, row in eod_source.iloc[skip_rows:].iterrows()
        }

        confluence_map = {}
        dt_keys = sorted(set(series_maps["vwap"]) | set(series_maps["eod_vwap"]))
        for dt in dt_keys:
            std_val = series_maps["vwap"].get(dt)
            eod_val = series_maps["eod_vwap"].get(dt)
            if std_val is None and eod_val is None:
                continue
            if std_val is None:
                confluence_map[dt] = (eod_val, eod_val)
            elif eod_val is None:
                confluence_map[dt] = (std_val, std_val)
            else:
                confluence_map[dt] = (min(std_val, eod_val), max(std_val, eod_val))
        series_maps["vwap_eod_confluence"] = confluence_map
        return series_maps

    def _build_vwap_invalidation_snapshot(self, df, direction):
        prepared = self._prepare_vwap_frame(df)
        if prepared.empty:
            return None

        current_date = prepared["datetime"].iloc[-1].date()
        today_prepared = prepared[prepared["datetime"].dt.date == current_date].copy()
        if today_prepared.empty:
            return None

        level_series_maps = self._build_intraday_level_series_maps(prepared, current_date)
        if not level_series_maps:
            return None

        consecutive_bad_closes = 0
        recent_rows = []
        for _, row in today_prepared.sort_values("datetime", ascending=False).iterrows():
            bar_dt = row["datetime"]
            close_value = float(row["close"])
            levels = {
                "vwap": level_series_maps.get("vwap", {}).get(bar_dt),
                "dynamic_vwap": level_series_maps.get("dynamic_vwap", {}).get(bar_dt),
                "eod_vwap": level_series_maps.get("eod_vwap", {}).get(bar_dt),
            }
            if any(level is None for level in levels.values()):
                break

            if direction == "long":
                violates_levels = all(close_value < level for level in levels.values())
            else:
                violates_levels = all(close_value > level for level in levels.values())

            recent_rows.append(
                {
                    "datetime": bar_dt.isoformat(),
                    "close": close_value,
                    "vwap": float(levels["vwap"]),
                    "dynamic_vwap": float(levels["dynamic_vwap"]),
                    "eod_vwap": float(levels["eod_vwap"]),
                    "violates_levels": violates_levels,
                }
            )

            if not violates_levels:
                break
            consecutive_bad_closes += 1

        recent_rows.reverse()
        last_row = recent_rows[-1] if recent_rows else None
        return {
            "direction": direction,
            "required_consecutive_closes": VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES,
            "consecutive_bad_closes": consecutive_bad_closes,
            "bars_reviewed": len(recent_rows),
            "last_row": last_row,
            "recent_rows": recent_rows,
        }

    def _load_symbol_classification_cache(self):
        if not SYMBOL_CLASSIFICATION_CACHE_FILENAME.exists():
            return
        try:
            with open(SYMBOL_CLASSIFICATION_CACHE_FILENAME, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    symbol = (row.get("symbol") or "").strip().upper()
                    if symbol:
                        sector_key = slugify_key(row.get("sectorKey") or row.get("sector") or "")
                        industry_key = slugify_key(row.get("industryKey") or row.get("industry") or "")
                        self.symbol_classification_cache[symbol] = {
                            "symbol": symbol,
                            "sectorKey": sector_key,
                            "industryKey": industry_key,
                            "sector": row.get("sector", ""),
                            "industry": row.get("industry", ""),
                            "updated_utc": row.get("updated_utc", ""),
                        }
        except Exception as exc:
            logging.warning(f"Failed loading symbol classification cache: {exc}")

    def _write_symbol_classification_cache(self):
        SYMBOL_CLASSIFICATION_CACHE_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        with open(SYMBOL_CLASSIFICATION_CACHE_FILENAME, "w", newline="") as fh:
            fieldnames = ["symbol", "sectorKey", "industryKey", "sector", "industry", "updated_utc"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for symbol in sorted(self.symbol_classification_cache):
                writer.writerow(self.symbol_classification_cache[symbol])

    def _fetch_symbol_classification(self, symbol):
        sector_name = ""
        industry_name = ""
        sector_key = ""
        industry_key = ""
        try:
            info = yf.Ticker(symbol).get_info()
            sector_name = (info.get("sectorDisp") or info.get("sector") or "").strip()
            industry_name = (info.get("industryDisp") or info.get("industry") or "").strip()
            sector_key = (info.get("sectorKey") or "").strip()
            industry_key = (info.get("industryKey") or "").strip()
        except Exception as exc:
            logging.warning(f"Yahoo classification failed for {symbol}: {exc}")
        sector_key = slugify_key(sector_key or sector_name)
        industry_key = slugify_key(industry_key or industry_name)
        return {
            "symbol": symbol,
            "sectorKey": sector_key,
            "industryKey": industry_key,
            "sector": sector_name,
            "industry": industry_name,
            "updated_utc": utc_now_iso(),
        }

    def get_symbol_classification(self, symbol):
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return None
        cached = self.symbol_classification_cache.get(symbol)
        if cached is not None:
            return cached
        row = self._fetch_symbol_classification(symbol)
        self.symbol_classification_cache[symbol] = row
        self._write_symbol_classification_cache()
        if row.get("industryKey"):
            self.industry_map_data = load_and_update_industry_etf_map(
                row.get("industryKey"),
                row.get("sectorKey"),
                row.get("industry"),
                row.get("sector"),
            )
        return row

    def _get_cached_bars(self, symbol, duration, bar_size):
        key = f"{symbol}|{duration}|{bar_size}"
        bars = self.latest_bars.get(key)
        if bars:
            return bars
        bars_raw = self.request_historical_bars(symbol, duration, bar_size, timeout=RRS_TIMEOUT)
        if not bars_raw:
            return []
        bars_ib = _dedupe_bars(_bars_to_ib(bars_raw))
        self.latest_bars[key] = bars_ib
        self.latest_bars.setdefault(symbol, bars_ib)
        return bars_ib

    def request_historical_bars(self, symbol, duration, bar_size, timeout=RRS_TIMEOUT):
        if symbol in self.invalid_security_symbols:
            logging.debug(f"{symbol}: Skipping historical request due to prior IB security-definition failure.")
            return None
        if not self.ensure_connected():
            logging.warning("Not connected to IB; skipping historical request.")
            return None
        self._respect_pacing_backoff()
        reqId = self.getReqId()
        with self.data_lock:
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            self.reqid_to_symbol[reqId] = symbol
        contract = self.create_stock_contract(symbol)
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        if not self.data_ready_events[reqId].wait(timeout=timeout):
            with self.data_lock:
                del self.data_ready_events[reqId]
                self.data.pop(reqId, None)
                self.reqid_to_symbol.pop(reqId, None)
            logging.warning(f"{symbol}: Timeout waiting for RRS data.")
            return None
        with self.data_lock:
            bars = self.data.get(reqId, [])
            del self.data_ready_events[reqId]
            self.data.pop(reqId, None)
            self.reqid_to_symbol.pop(reqId, None)
        return bars

    def get_cached_5m_bars(self, symbol):
        return self._get_cached_bars(symbol, "5 D", "5 mins")

    def run_rrs_scan(self, timeframe_key_override=None, emit_gui=True):
        self.load_master_avwap_focus()
        self.load_master_avwap_d1_watchlist()
        self.load_master_avwap_d1_upgrade_alerts()
        threshold, bar_size, duration, length, timeframe_key = self.get_rrs_settings()
        if timeframe_key_override in RRS_TIMEFRAMES:
            timeframe_key = timeframe_key_override
            bar_size = RRS_TIMEFRAMES[timeframe_key]["bar_size"]
            duration = RRS_TIMEFRAMES[timeframe_key]["duration"]
        timeframe_minutes = RRS_TIMEFRAMES[timeframe_key]["minutes"]
        if emit_gui and self.gui_callback:
            self.gui_callback(f"RRS scan running ({RRS_TIMEFRAMES[timeframe_key]['label']})...", "rrs_status")

        spy_5m = self.get_cached_5m_bars("SPY")
        if not spy_5m:
            if emit_gui and self.gui_callback:
                self.gui_callback("RRS scan: SPY data unavailable.", "rrs_status")
            return

        spy_bars = spy_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(spy_5m, timeframe_minutes)
        spy_by_dt = {bar.dt: bar for bar in spy_bars}
        spy_move_ratio = self._calc_move_ratio(spy_bars, length)

        results = []
        sector_results = []
        industry_results = []
        symbol_context = []
        all_scores = []
        intraday_profiles = {}
        current_market_date = spy_5m[-1].dt.date()
        previous_market_date = self._previous_market_date_from_bars(spy_5m, current_market_date)
        earnings_reaction_aliases = self._load_earnings_reaction_symbols(
            current_market_date,
            previous_market_date,
        )
        raw_symbols = sorted(self.get_scan_symbol_set() - {"SPY"})
        earnings_reaction_symbols = [
            symbol for symbol in raw_symbols
            if self._symbol_matches_alias_set(symbol, earnings_reaction_aliases)
        ]
        all_symbols = [
            symbol for symbol in raw_symbols
            if symbol not in earnings_reaction_symbols
        ]
        environment_scan_active = self._environment_scan_is_active(
            spy_5m[-1].dt if spy_5m else None
        )
        spy_context_windows = self._build_spy_context_windows(spy_5m, length) if environment_scan_active else []
        for symbol in all_symbols:
            sym_5m = self.get_cached_5m_bars(symbol)
            if not sym_5m:
                continue
            intraday_profile = self._build_intraday_rrs_profile(sym_5m, spy_5m, length=length)
            if intraday_profile:
                current_date = spy_5m[-1].dt.date()
                today_profile = [item for item in intraday_profile if item.get("dt") and item["dt"].date() == current_date]
                if today_profile:
                    intraday_profiles[symbol] = today_profile
            sym_bars = sym_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(sym_5m, timeframe_minutes)
            aligned_sym, aligned_spy = _align_bars_with_map(sym_bars, spy_by_dt)
            rrs_value, power_index = real_relative_strength(aligned_sym, aligned_spy, length=length)
            if rrs_value is None:
                continue
            symbol_move_ratio = self._calc_move_ratio(sym_bars, length)
            excess_move_ratio = None
            if symbol_move_ratio is not None and spy_move_ratio is not None:
                excess_move_ratio = symbol_move_ratio - spy_move_ratio
            all_scores.append((symbol, rrs_value, power_index))
            environment_signal = None
            if rrs_value >= threshold:
                environment_signal = "RS"
            elif rrs_value <= -threshold:
                environment_signal = "RW"

            symbol_direction = self.get_symbol_direction(symbol)
            if environment_signal:
                symbol_context.append(
                    {
                        "symbol": symbol,
                        "signal": environment_signal,
                        "rrs": rrs_value,
                        "move_ratio": symbol_move_ratio,
                        "excess_move_ratio": excess_move_ratio,
                        "power_index": power_index,
                        "watchlist_bias": symbol_direction,
                    }
                )

            if symbol_direction == "long" and rrs_value >= threshold:
                results.append(("RS", symbol, rrs_value, power_index))
            elif symbol_direction == "short" and rrs_value <= -threshold:
                results.append(("RW", symbol, rrs_value, power_index))

            classification = self.get_symbol_classification(symbol)
            if not classification:
                continue
            sector_key = classification.get("sectorKey", "")
            industry_key = classification.get("industryKey", "")
            if industry_key:
                self.industry_map_data = load_and_update_industry_etf_map(
                    industry_key,
                    sector_key,
                    classification.get("industry", ""),
                    classification.get("sector", ""),
                )

            sector_etf = resolve_sector_etf(sector_key, self.sector_etf_map)
            sec_5m = self.get_cached_5m_bars(sector_etf)
            if sec_5m:
                sec_bars = sec_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(sec_5m, timeframe_minutes)
                aligned_sym_sec, aligned_sec = _align_bars_with_map(sym_bars, {bar.dt: bar for bar in sec_bars})
                sec_rrs, sec_power = real_relative_strength(aligned_sym_sec, aligned_sec, length=length)
                if sec_rrs is not None:
                    if symbol_direction == "long" and sec_rrs >= threshold:
                        sector_results.append(("RS", symbol, sec_rrs, sec_power))
                    elif symbol_direction == "short" and sec_rrs <= -threshold:
                        sector_results.append(("RW", symbol, sec_rrs, sec_power))

            industry_ref = resolve_industry_ref_etf(industry_key, sector_key)
            ind_5m = self.get_cached_5m_bars(industry_ref)
            if ind_5m:
                ind_bars = ind_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(ind_5m, timeframe_minutes)
                aligned_sym_ind, aligned_ind = _align_bars_with_map(sym_bars, {bar.dt: bar for bar in ind_bars})
                ind_rrs, ind_power = real_relative_strength(aligned_sym_ind, aligned_ind, length=length)
                if ind_rrs is not None:
                    if symbol_direction == "long" and ind_rrs >= threshold:
                        industry_results.append(("RS", symbol, ind_rrs, ind_power))
                    elif symbol_direction == "short" and ind_rrs <= -threshold:
                        industry_results.append(("RW", symbol, ind_rrs, ind_power))

        strongest = sorted(all_scores, key=lambda row: row[1], reverse=True)[:SCAN_EXTREME_COUNT]
        weakest = sorted(all_scores, key=lambda row: row[1])[:SCAN_EXTREME_COUNT]
        self.latest_scan_extremes[timeframe_key] = strongest + weakest
        if strongest or weakest:
            self._log_scan_extremes(timeframe_key, strongest, weakest)

        rs_results = sorted([r for r in results if r[0] == "RS"], key=lambda r: -r[2])
        rw_results = sorted([r for r in results if r[0] == "RW"], key=lambda r: r[2])
        ordered_results = rs_results + rw_results

        def _ordered(rows):
            rs = sorted([r for r in rows if r[0] == "RS"], key=lambda r: -r[2])
            rw = sorted([r for r in rows if r[0] == "RW"], key=lambda r: r[2])
            return rs + rw

        sector_payload = _ordered(sector_results)
        industry_payload = _ordered(industry_results)
        environment_scan = (
            self._summarize_environment_scan(intraday_profiles, spy_context_windows, threshold)
            if environment_scan_active else None
        )
        if environment_scan:
            self._record_environment_focus_history(environment_scan, timeframe_key)
        snapshot_payload = {
            "timestamp": datetime.now(),
            "threshold": threshold,
            "timeframe_key": timeframe_key,
            "results": ordered_results,
            "results_sector": sector_payload,
            "results_industry": industry_payload,
            "group_strength": self.compute_group_strengths(),
            "symbol_context": symbol_context,
            "spy_move_ratio": spy_move_ratio,
            "environment_scan": environment_scan,
            "excluded_earnings_reaction_symbols": earnings_reaction_symbols,
            # Every scanned symbol's SPY RRS, not just threshold-crossers.
            # ``results`` only holds alert candidates, which left the learning
            # rows' rrs fields blank for ~85% of bounces ("unknown" alignment).
            "rrs_all": {
                str(symbol).strip().upper(): (rrs_value, power_index)
                for symbol, rrs_value, power_index in all_scores
            },
        }
        if emit_gui:
            self._emit_master_avwap_focus_rrs_alerts(symbol_context, threshold, timeframe_key)
        self.latest_rrs_payload = snapshot_payload
        if emit_gui and self.gui_callback:
            decorated_snapshot = self._decorate_snapshot(snapshot_payload)
            self.gui_callback(decorated_snapshot, "rrs_snapshot")
            status_msg = (
                f"RRS scan complete ({len(ordered_results)} SPY, {len(sector_payload)} sector, "
                f"{len(industry_payload)} industry refs)"
            )
            if not environment_scan_active:
                market_session = get_market_session_window(reference=get_market_local_now())
                first_hour_end = market_session.open_local + timedelta(minutes=ENVIRONMENT_SCAN_DELAY_MINUTES)
                status_msg += f" | SPY context windows start after {first_hour_end.strftime('%H:%M')}"
            if earnings_reaction_symbols:
                status_msg += f" | excluded {len(earnings_reaction_symbols)} earnings reactions"
            self.gui_callback(
                status_msg,
                "rrs_status",
            )

    def compute_group_strengths(self):
        results = {}
        industry_refs = self.industry_map_data.get("yahoo_industryKey_to_ref", {})
        industry_etfs = sorted({(v.get("etf") or "").strip().upper() for v in industry_refs.values() if isinstance(v, dict) and (v.get("etf") or "").strip()})
        for tf_key, tf in GROUP_STRENGTH_TIMEFRAMES.items():
            spy = self._get_cached_bars("SPY", tf["duration"], tf["bar_size"])
            if not spy:
                continue
            spy_by_dt = {bar.dt: bar for bar in spy}
            sectors = []
            for sector_key, etf in sorted(self.sector_etf_map.items()):
                bars = self._get_cached_bars(etf, tf["duration"], tf["bar_size"])
                if not bars:
                    continue
                aligned_etf, aligned_spy = _align_bars_with_map(bars, spy_by_dt)
                rrs, power = real_relative_strength(aligned_etf, aligned_spy, length=self.rrs_length)
                if rrs is not None:
                    sectors.append({"group_key": sector_key, "etf": etf, "rrs": rrs, "power_index": power})
            industries = []
            for etf in industry_etfs:
                bars = self._get_cached_bars(etf, tf["duration"], tf["bar_size"])
                if not bars:
                    continue
                aligned_etf, aligned_spy = _align_bars_with_map(bars, spy_by_dt)
                rrs, power = real_relative_strength(aligned_etf, aligned_spy, length=self.rrs_length)
                if rrs is not None:
                    industries.append({"group_key": etf, "etf": etf, "rrs": rrs, "power_index": power})
            results[tf_key] = {
                "sectors": sorted(sectors, key=lambda x: x["rrs"], reverse=True),
                "industries": sorted(industries, key=lambda x: x["rrs"], reverse=True),
            }
        self.latest_group_extremes = results
        self._log_group_strength_extremes(results)
        return results

    def _log_group_strength_extremes(self, grouped):
        timestamp_local = get_market_local_now().strftime("%Y-%m-%d %H:%M:%S %Z")
        GROUP_STRENGTH_SCAN_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        write_header = not GROUP_STRENGTH_SCAN_LOG_FILENAME.exists()
        with open(GROUP_STRENGTH_SCAN_LOG_FILENAME, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["timestamp_local", "timeframe", "group_type", "group_key", "etf", "rrs", "power_index"],
            )
            if write_header:
                writer.writeheader()
            for timeframe, payload in grouped.items():
                for group_type, items in (("sector", payload.get("sectors", [])), ("industry", payload.get("industries", []))):
                    top = items[:SCAN_EXTREME_COUNT]
                    bottom = list(reversed(items[-SCAN_EXTREME_COUNT:])) if len(items) > SCAN_EXTREME_COUNT else []
                    for item in top + bottom:
                        writer.writerow({
                            "timestamp_local": timestamp_local,
                            "timeframe": timeframe,
                            "group_type": group_type,
                            "group_key": item.get("group_key", ""),
                            "etf": item.get("etf", ""),
                            "rrs": f"{item.get('rrs', 0.0):.4f}",
                            "power_index": f"{item.get('power_index', 0.0):.4f}",
                        })

    def _log_scan_extremes(self, timeframe_key, strongest, weakest):
        timestamp_local = get_market_local_now()
        STRENGTH_SCAN_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        write_header = not STRENGTH_SCAN_LOG_FILENAME.exists()
        with open(STRENGTH_SCAN_LOG_FILENAME, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["timestamp_local", "timeframe", "bucket", "symbol", "rrs", "power_index"],
            )
            if write_header:
                writer.writeheader()
            for bucket, items in (("strongest", strongest), ("weakest", weakest)):
                for symbol, rrs_value, power_index in items:
                    writer.writerow(
                        {
                            "timestamp_local": timestamp_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
                            "timeframe": timeframe_key,
                            "bucket": bucket,
                            "symbol": symbol,
                            "rrs": f"{rrs_value:.4f}",
                            "power_index": (
                                f"{power_index:.4f}" if power_index is not None else ""
                            ),
                        }
                    )

    def get_monitored_extreme_symbols(self):
        symbols = set()
        for entries in self.latest_scan_extremes.values():
            symbols.update(item[0] for item in entries)
        symbols.update(self._human_focus_symbols())
        for symbol, entry in self.master_avwap_focus_map.items():
            if not isinstance(entry, dict):
                continue
            focus_side = str(entry.get("side") or "").upper()
            if focus_side == "LONG" and symbol in self.longs:
                symbols.add(symbol)
            elif focus_side == "SHORT" and symbol in self.shorts:
                symbols.add(symbol)
        return symbols

    def has_minimum_candles_completed(self, required=CONSECUTIVE_CANDLES):
        now = get_market_local_now()
        market_session = get_market_session_window(reference=now)
        if now < market_session.open_local:
            return False
        elapsed = (now - market_session.open_local).total_seconds()
        return elapsed >= required * 300

    def nextValidId(self, orderId):
        self.connection_status = True
        self.client_id_conflict = False
        logging.info(f"Connected to IB API. NextValidId={orderId}")

    def error(self, reqId, errorCode, errorString):
        if errorCode in IB_INFO_STATUS_CODES:
            logging.info(f"IB Status. ReqId={reqId}, Code={errorCode}, Msg={errorString}")
        elif errorCode in IB_WARNING_STATUS_CODES:
            logging.warning(f"IB Warning. ReqId={reqId}, Code={errorCode}, Msg={errorString}")
        else:
            logging.error(f"IB Error. ReqId={reqId}, Code={errorCode}, Msg={errorString}")
        if errorCode == 200:
            with self.data_lock:
                symbol = self.reqid_to_symbol.get(reqId)
                if symbol:
                    self.invalid_security_symbols.add(symbol)
                if reqId in self.data_ready_events:
                    self.data_ready_events[reqId].set()
        if errorCode in (162, 366):
            # Failed/cancelled historical query: release the waiting request
            # instead of letting it burn its full timeout.
            with self.data_lock:
                if reqId in self.data_ready_events:
                    self.data_ready_events[reqId].set()
        if errorCode == 100 or "pacing" in str(errorString or "").lower():
            self._register_pacing_violation(errorCode)
        if errorCode == 326:
            self.connection_status = False
            self.client_id_conflict = True
        if errorCode in (504, 1100) or "not connected" in errorString.lower():
            self.connection_status = False
        elif errorCode in (1101, 1102):
            self.connection_status = True

    def connectionClosed(self):
        self.connection_status = False
        logging.warning("IB connection closed.")

    def historicalData(self, reqId, bar):
        with self.data_lock:
            if reqId not in self.data:
                self.data[reqId] = []
            self.data[reqId].append({
                "time": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })
            logging.debug(f"ReqId={reqId} | Received bar: {bar.date} O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close}")

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        logging.debug(f"Historical data end: ReqId={reqId}, Start={start}, End={end}. Total bars: {len(self.data.get(reqId, []))}")
        with self.data_lock:
            if reqId in self.data_ready_events:
                self.data_ready_events[reqId].set()

    def calculate_atr(self, df_daily, period=ATR_PERIOD):
        # Calculate True Range
        df_daily["prev_close"] = df_daily["close"].shift(1)
        df_daily["h_pc"] = abs(df_daily["high"] - df_daily["prev_close"])
        df_daily["l_pc"] = abs(df_daily["low"] - df_daily["prev_close"])
        df_daily["h_l"] = abs(df_daily["high"] - df_daily["low"])
        df_daily["tr"] = df_daily[["h_pc", "l_pc", "h_l"]].max(axis=1)

        # Initialize ATR
        df_daily["atr"] = float('nan')

        # First ATR value is the simple average of TR
        first_atr = df_daily["tr"].iloc[:period].mean()
        df_daily.loc[df_daily.index[period-1], "atr"] = first_atr

        # Apply Wilder's smoothing formula
        multiplier = 1.0 / period
        for i in range(period, len(df_daily)):
            prev_atr = df_daily["atr"].iloc[i-1]
            current_tr = df_daily["tr"].iloc[i]
            df_daily.loc[df_daily.index[i], "atr"] = (prev_atr * (period - 1) + current_tr) * multiplier

        return df_daily["atr"].iloc[-1]




    def calculate_vwap(self, df):
        if df.empty or ('volume' not in df) or (df['volume'].sum() == 0):
            return None

        df = df.copy()

        # TC2000 style typical price calculation
        df["typical"] = (df["high"] + df["low"] + df["open"] + df["close"]) / 4

        # Calculate VWAP
        vwap = (df["typical"] * df["volume"]).sum() / df["volume"].sum()

        return vwap

    def calculate_vwap_with_stdev_bands(self, df, band_mult=1.0):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None, None, None

        current_date = prepared_df["datetime"].iloc[-1].date()
        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        vwap, upper_band, lower_band = self._calculate_vwap_bands(today_df, band_mult=band_mult)

        if LOGGING_MODE and vwap is not None:
            logging.debug(f"Standard VWAP calculation: VWAP={vwap:.4f}")
            logging.debug(f"Standard VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}")

        return vwap, upper_band, lower_band




    def _prepare_vwap_frame(self, df):
        if df.empty:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy["datetime"] = pd.to_datetime(
            df_copy["time"],
            format="%Y%m%d  %H:%M:%S",
            errors="coerce",
        )
        if df_copy["datetime"].isna().all():
            df_copy["datetime"] = pd.to_datetime(df_copy["time"], errors="coerce")
        if df_copy["datetime"].isna().all():
            logging.error("Failed to convert time to datetime for VWAP calculation")
            return pd.DataFrame()

        df_copy = df_copy.sort_values("datetime").reset_index(drop=True)
        df_copy["typical_price"] = (
            df_copy["high"] + df_copy["low"] + df_copy["open"] + df_copy["close"]
        ) / 4.0
        return df_copy

    def _calculate_vwap_value(self, df):
        if df.empty or df["volume"].sum() == 0:
            return None

        vol_price = df["typical_price"] * df["volume"]
        return vol_price.sum() / df["volume"].sum()

    def _calculate_vwap_bands(self, df, band_mult=1.0):
        if df.empty or df["volume"].sum() == 0:
            return None, None, None

        df = df.copy()
        df["vol_times_price"] = df["typical_price"] * df["volume"]
        cum_vol = df["volume"].cumsum()
        cum_vol_price = df["vol_times_price"].cumsum()
        df["vwap"] = cum_vol_price / cum_vol

        df["sq_dev"] = ((df["typical_price"] - df["vwap"]) ** 2) * df["volume"]
        cum_sq_dev = df["sq_dev"].cumsum()
        df["stdev"] = (cum_sq_dev / cum_vol).apply(lambda x: math.sqrt(x) if x > 0 else 0)

        vwap = df["vwap"].iloc[-1]
        stdev = df["stdev"].iloc[-1]
        upper_band = vwap + stdev * band_mult
        lower_band = vwap - stdev * band_mult
        return vwap, upper_band, lower_band

    def calculate_standard_vwap(self, df):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None

        current_date = prepared_df["datetime"].iloc[-1].date()
        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        if today_df.empty:
            logging.warning("No today's data available for standard VWAP calculation")
            return None

        logging.debug(f"Standard VWAP using data from {current_date} only")
        logging.debug(f"Today's data count: {len(today_df)}")

        return self._calculate_vwap_value(today_df)

    def calculate_dynamic_vwap(self, df):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        previous_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        logging.debug(
            f"Dynamic VWAP using data from {previous_date} and {current_date}"
        )
        logging.debug(
            f"Previous day data count: {len(previous_df)}, Today's data count: {len(today_df)}"
        )

        combined_df = (
            pd.concat([previous_df, today_df])
            if not previous_df.empty
            else today_df
        )
        return self._calculate_vwap_value(combined_df)

    def calculate_eod_vwap(self, df):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        previous_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        logging.debug(
            f"EOD VWAP using data from {current_date} and last candle from {previous_date}"
        )
        logging.debug(
            f"Previous day data count: {len(previous_df)}, Today's data count: {len(today_df)}"
        )

        if previous_df.empty:
            return self._calculate_vwap_value(today_df)

        eod_df = pd.concat([previous_df.tail(1), today_df])
        return self._calculate_vwap_value(eod_df)

    def calculate_dynamic_vwap_with_stdev_bands(self, df, band_mult=1.0):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None, None, None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        prev_day_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        if prev_day_df.empty and previous_date:
            symbol_key = f"dynamic_vwap_{current_date}"
            if symbol_key not in self.warned_symbols:
                logging.warning(
                    f"No previous trading day data for dynamic VWAP bands (looking for {previous_date})"
                )
                self.warned_symbols.add(symbol_key)

        combined_df = (
            pd.concat([prev_day_df, today_df])
            if not prev_day_df.empty
            else today_df
        )
        dynamic_vwap, upper_band, lower_band = self._calculate_vwap_bands(
            combined_df,
            band_mult=band_mult,
        )

        if LOGGING_MODE and dynamic_vwap is not None:
            logging.debug(f"Dynamic VWAP calculation: VWAP={dynamic_vwap:.4f}")
            logging.debug(
                f"Dynamic VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}"
            )

        return dynamic_vwap, upper_band, lower_band



    def calculate_eod_vwap_with_stdev_bands(self, df, band_mult=1.0):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None, None, None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        prev_day_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        if prev_day_df.empty and previous_date:
            symbol_key = f"eod_vwap_{current_date}"
            if symbol_key not in self.warned_symbols:
                logging.warning(
                    f"No previous trading day data for EOD VWAP bands (looking for {previous_date})"
                )
                self.warned_symbols.add(symbol_key)

        eod_df = (
            pd.concat([prev_day_df.tail(1), today_df])
            if not prev_day_df.empty
            else today_df
        )
        eod_vwap, upper_band, lower_band = self._calculate_vwap_bands(
            eod_df,
            band_mult=band_mult,
        )

        if LOGGING_MODE and eod_vwap is not None:
            logging.debug(f"EOD VWAP calculation: VWAP={eod_vwap:.4f}")
            logging.debug(
                f"EOD VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}"
            )

        return eod_vwap, upper_band, lower_band





    def calculate_dynamic_vwap2(self, df):
        try:
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
        except Exception as e:
            logging.warning(f"Datetime conversion error in dynamic VWAP2: {e}")
            return None

        if df.empty:
            return None

        today_date = df["datetime"].iloc[-1].date()
        two_days_ago = today_date - timedelta(days=2)

        today_df = df[df["datetime"].dt.date == today_date].copy()
        two_days_ago_df = df[df["datetime"].dt.date == two_days_ago].copy()

        dynamic_df = pd.concat([two_days_ago_df, today_df]) if not two_days_ago_df.empty else today_df

        return self.calculate_vwap(dynamic_df)

    def get_previous_day_extremes(self, df):
        try:
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
        except Exception as e:
            logging.warning(f"Error converting datetime: {e}")
            return None, None
        current_date = df["datetime"].iloc[-1].date()
        prev_session = df[df["datetime"].dt.date < current_date]
        if prev_session.empty:
            return None, None
        prev_high = prev_session["high"].max()
        prev_low = prev_session["low"].min()
        return prev_high, prev_low



    def build_atr_cache(self):
        all_symbols = self.get_scan_symbol_set()
        to_fetch = [s for s in all_symbols if s not in self.atr_cache]
        if not to_fetch:
            logging.debug("No new symbols for ATR update.")
            return
        for sym in to_fetch:
            if not sym.isalnum():
                logging.error(f"Invalid symbol: {sym}. Skipping ATR calculation.")
                self.atr_cache[sym] = None
                continue
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(sym)
            logging.debug(f"Requesting 20-day ATR data for {sym} with reqId {reqId}")
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="20 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                if len(bars) < ATR_PERIOD:
                    self.atr_cache[sym] = None
                    logging.warning(f"{sym}: Insufficient data for ATR calculation.")
                else:
                    df_daily = pd.DataFrame(bars)
                    df_daily["datetime"] = pd.to_datetime(df_daily["time"], errors="coerce")
                    atr_val = self.calculate_atr(df_daily, period=ATR_PERIOD)
                    self.atr_cache[sym] = atr_val
                    logging.info(f"{sym}: 20-day ATR = {atr_val:.2f}")
            else:
                self.atr_cache[sym] = None
                logging.warning(f"{sym}: ATR data request timed out.")
            del self.data_ready_events[reqId]

    # Helper function for color coded logging (non-ATR messages)
    # Option 1: Update the method definition to accept a tag parameter with a default value
    def log_symbol(self, symbol, msg, tag=None):
        if "ATR" in msg:
            logging.info(msg)
        else:
            direction = self.get_symbol_direction(symbol)
            if direction == "long":
                colored_msg = Fore.GREEN + msg + Style.RESET_ALL
            elif direction == "short":
                colored_msg = Fore.RED + msg + Style.RESET_ALL
            else:
                colored_msg = msg
            logging.info(colored_msg)


    def log_symbol_metrics(self, symbol, df_5m):
        # Get the current price from the last candle
        current_price = df_5m["close"].iloc[-1] if not df_5m.empty else None

        # Calculate the different VWAPs directly
        standard_vwap = self.calculate_standard_vwap(df_5m)
        dynamic_vwap = self.calculate_dynamic_vwap(df_5m)
        eod_vwap = self.calculate_eod_vwap(df_5m)

        # Get previous day extremes
        prev_high, prev_low = self.get_previous_day_extremes(df_5m)

        # Get ATR value
        atr_val = self.atr_cache.get(symbol, float('nan'))

        # Build the log message
        msg = f"{symbol} Metrics -> "
        msg += f"Price: {current_price:.2f}, " if current_price is not None else "Price: N/A, "
        msg += f"Std VWAP: {standard_vwap:.2f}, " if standard_vwap is not None else "Std VWAP: N/A, "
        msg += f"Dynamic VWAP: {dynamic_vwap:.2f}, " if dynamic_vwap is not None else "Dynamic VWAP: N/A, "
        msg += f"EOD VWAP: {eod_vwap:.2f}, " if eod_vwap is not None else "EOD VWAP: N/A, "

        direction = self.get_symbol_direction(symbol)
        if direction == "long":
            msg += f"Prev Day High: {prev_high:.2f}, " if prev_high is not None else "Prev Day High: N/A, "
        elif direction == "short":
            msg += f"Prev Day Low: {prev_low:.2f}, " if prev_low is not None else "Prev Day Low: N/A, "

        msg += f"ATR: {atr_val:.2f}"
        self.log_symbol(symbol, msg)



    def evaluate_bounce_candidate(self, symbol, df, allowed_bounce_types=None, *, include_disabled_bounce_types=False):
        if len(df) < 10:
            return None

        # Get the current candle
        current_candle = df.iloc[-1].copy()

        # Ensure we're working with scalar values, not Series objects
        def get_scalar(value):
            if isinstance(value, pd.Series):
                return value.iloc[0] if len(value) > 0 else value
            return value

        # Normalize current candle data
        current_candle_data = {
            "open": get_scalar(current_candle["open"]),
            "high": get_scalar(current_candle["high"]),
            "low": get_scalar(current_candle["low"]),
            "close": get_scalar(current_candle["close"]),
            "volume": get_scalar(current_candle["volume"]),
            "time": current_candle["time"]
        }

        # Get ATR for this symbol
        atr = self.atr_cache.get(symbol, None)
        if atr is None:
            logging.debug(f"{symbol}: No ATR available, skipping bounce evaluation")
            return None

        # Set threshold for proximity to levels
        threshold = THRESHOLD_MULTIPLIER * atr
        direction = self.get_symbol_direction(symbol)
        if direction not in {"long", "short"}:
            logging.debug(f"{symbol}: No long/short direction available, skipping bounce evaluation")
            return None

        # Get the metrics from the cache
        metrics = self.symbol_metrics.get(symbol, {})
        if not metrics:
            logging.debug(f"{symbol}: No metrics available, skipping bounce evaluation")
            return None

        # Add datetime column if it doesn't exist
        if "datetime" not in df.columns:
            try:
                df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            except Exception as e:
                logging.error(f"{symbol}: Error in datetime conversion: {e}")
                return None

        # Get today's candles
        current_date = df["datetime"].iloc[-1].date()
        today_df = df[df["datetime"].dt.date == current_date].copy()
        level_series_maps = self._build_intraday_level_series_maps(df, current_date)

        # Skip if we don't have enough candles for today
        if len(today_df) < CONSECUTIVE_CANDLES + 1:  # +1 for current candle
            logging.debug(f"{symbol}: Not enough candles for today ({len(today_df)}) to check consecutive condition")
            # We can still continue without the consecutive check if it's disabled
            if CHECK_CONSECUTIVE_CANDLES:
                return None

        # Estimate intraday ATR from today's 5-minute candles for impulse sizing.
        intraday_impulse_atr = None
        try:
            if len(today_df) >= 3:
                intraday_df = today_df[["high", "low", "close"]].copy()
                intraday_df["prev_close"] = intraday_df["close"].shift(1)
                tr_components = pd.concat(
                    [
                        (intraday_df["high"] - intraday_df["low"]).abs(),
                        (intraday_df["high"] - intraday_df["prev_close"]).abs(),
                        (intraday_df["low"] - intraday_df["prev_close"]).abs(),
                    ],
                    axis=1,
                )
                intraday_df["tr"] = tr_components.max(axis=1)
                lookback = min(IMPULSE_INTRADAY_ATR_PERIOD, len(intraday_df))
                intraday_impulse_atr = float(intraday_df["tr"].tail(lookback).mean())
                if intraday_impulse_atr <= 0 or math.isnan(intraday_impulse_atr):
                    intraday_impulse_atr = None
        except Exception:
            intraday_impulse_atr = None

        intraday_rrs_profile = []
        try:
            spy_5m = self.latest_bars.get("SPY") or self.get_cached_5m_bars("SPY")
            if spy_5m:
                today_symbol_bars = _dedupe_bars(_bars_to_ib(today_df.to_dict("records")))
                today_spy_bars = [bar for bar in spy_5m if bar.dt.date() == current_date]
                intraday_rrs_profile = self._build_intraday_rrs_profile(
                    today_symbol_bars,
                    today_spy_bars,
                    length=IMPULSE_RRS_PROFILE_LENGTH,
                )
        except Exception as exc:
            logging.debug(f"{symbol}: Failed building intraday RS/RW profile for impulse filter: {exc}")
            intraday_rrs_profile = []

        # General structure gate for ALL bounce types: we want a stock moving
        # with the trade direction making a simple retest of a level - not a
        # choppy multi-pullback tape, not a compressed one.
        if BOUNCE_SESSION_STRUCTURE_GATE:
            structure_ok, structure_reason = _session_structure_report(
                today_df, direction, intraday_impulse_atr
            )
            if not structure_ok:
                logging.debug(
                    f"{symbol}: Session structure gate rejected bounce scan ({structure_reason})"
                )
                return None

        # Log the metrics being used for evaluation
        if LOGGING_MODE:
            std_vwap_str = f"{metrics.get('std_vwap'):.4f}" if metrics.get('std_vwap') is not None else "None"
            dynamic_vwap_str = f"{metrics.get('dynamic_vwap'):.4f}" if metrics.get('dynamic_vwap') is not None else "None"
            eod_vwap_str = f"{metrics.get('eod_vwap'):.4f}" if metrics.get('eod_vwap') is not None else "None"
            ema_8_str = f"{metrics.get('ema_8'):.4f}" if metrics.get('ema_8') is not None else "None"
            ema_15_str = f"{metrics.get('ema_15'):.4f}" if metrics.get('ema_15') is not None else "None"
            ema_21_str = f"{metrics.get('ema_21'):.4f}" if metrics.get('ema_21') is not None else "None"
            upper_band_str = f"{metrics.get('vwap_1stdev_upper'):.4f}" if metrics.get('vwap_1stdev_upper') is not None else "None"
            lower_band_str = f"{metrics.get('vwap_1stdev_lower'):.4f}" if metrics.get('vwap_1stdev_lower') is not None else "None"
            dynamic_upper_str = f"{metrics.get('dynamic_vwap_1stdev_upper'):.4f}" if metrics.get('dynamic_vwap_1stdev_upper') is not None else "None"
            dynamic_lower_str = f"{metrics.get('dynamic_vwap_1stdev_lower'):.4f}" if metrics.get('dynamic_vwap_1stdev_lower') is not None else "None"
            eod_upper_str = f"{metrics.get('eod_vwap_1stdev_upper'):.4f}" if metrics.get('eod_vwap_1stdev_upper') is not None else "None"
            eod_lower_str = f"{metrics.get('eod_vwap_1stdev_lower'):.4f}" if metrics.get('eod_vwap_1stdev_lower') is not None else "None"

            logging.debug(f"{symbol} evaluation using - Std VWAP: {std_vwap_str}, Dynamic VWAP: {dynamic_vwap_str}, EOD VWAP: {eod_vwap_str}, EMA8: {ema_8_str}, EMA15: {ema_15_str}, EMA21: {ema_21_str}")
            logging.debug(f"{symbol} bands - Std 1SD Upper: {upper_band_str}, Std 1SD Lower: {lower_band_str}")
            logging.debug(f"{symbol} dyn bands - Dynamic 1SD Upper: {dynamic_upper_str}, Dynamic 1SD Lower: {dynamic_lower_str}")
            logging.debug(f"{symbol} eod bands - EOD 1SD Upper: {eod_upper_str}, EOD 1SD Lower: {eod_lower_str}")

        # Initialize dictionary for reference levels that triggered bounce condition
        ref_levels = {}
        triggered_levels = []
        allowed_types = set(allowed_bounce_types) if allowed_bounce_types is not None else None

        def bounce_type_allowed(bounce_type):
            return allowed_types is None or bounce_type in allowed_types

        def bounce_type_enabled(bounce_type):
            return bool(include_disabled_bounce_types or self.is_bounce_type_enabled(bounce_type))

        def mark_trigger(level_name):
            if level_name not in triggered_levels:
                triggered_levels.append(level_name)

        def is_touch_reject(level_value):
            if level_value is None:
                return False
            interacted = (
                current_candle_data["high"] >= (level_value - threshold)
                and current_candle_data["low"] <= (level_value + threshold)
            )
            if direction == "long":
                return (
                    interacted
                    and current_candle_data["close"] >= (level_value - threshold)
                    and current_candle_data["close"] > current_candle_data["open"]
                )
            return (
                interacted
                and current_candle_data["close"] <= (level_value + threshold)
                    and current_candle_data["close"] < current_candle_data["open"]
                )

        def candle_interacts_with_level(level_value, touch_buffer):
            if level_value is None:
                return False
            buffer = max(0.0, float(touch_buffer or 0.0))
            return (
                current_candle_data["high"] >= (level_value - buffer)
                and current_candle_data["low"] <= (level_value + buffer)
            )

        def is_recent_reclaim(level_value, level_name):
            if level_value is None or len(today_df) < 2:
                return False
            lookback = min(RECLAIM_LOOKBACK_CANDLES, len(today_df) - 1)
            previous_candles = today_df.iloc[-(lookback + 1):-1]
            if previous_candles.empty:
                return False

            if direction == "long":
                was_below = (previous_candles["close"] <= (level_value + threshold)).any()
                reclaimed_now = (
                    current_candle_data["close"] > level_value
                    and current_candle_data["high"] >= (level_value - threshold)
                    and current_candle_data["low"] <= (level_value + threshold)
                    and current_candle_data["close"] > current_candle_data["open"]
                )
            else:
                was_above = (previous_candles["close"] >= (level_value - threshold)).any()
                reclaimed_now = (
                    current_candle_data["close"] < level_value
                    and current_candle_data["high"] >= (level_value - threshold)
                    and current_candle_data["low"] <= (level_value + threshold)
                    and current_candle_data["close"] < current_candle_data["open"]
                )
                was_below = was_above

            reclaimed = bool(was_below and reclaimed_now)
            if reclaimed:
                logging.debug(
                    f"{symbol}: {level_name} reclaim detected within last {lookback} candles"
                )
            return reclaimed

        def is_vwap_zone_reaction(level_low, level_high, level_name):
            if level_low is None or level_high is None:
                return False

            near_buffer = max(threshold, VWAP_BOUNCE_NEAR_ATR * atr)
            pierce_buffer = max(threshold, VWAP_BOUNCE_PIERCE_ATR * atr)
            close_buffer = max(threshold, VWAP_BOUNCE_CLOSE_BUFFER_ATR * atr)

            if direction == "long":
                approached_zone = current_candle_data["low"] <= (level_high + near_buffer)
                limited_undershoot = current_candle_data["low"] >= (level_low - pierce_buffer)
                recovered = (
                    current_candle_data["close"] >= (level_high - close_buffer)
                    and current_candle_data["close"] > current_candle_data["open"]
                )
                distance = level_high - current_candle_data["low"]
            else:
                approached_zone = current_candle_data["high"] >= (level_low - near_buffer)
                limited_overshoot = current_candle_data["high"] <= (level_high + pierce_buffer)
                recovered = (
                    current_candle_data["close"] <= (level_low + close_buffer)
                    and current_candle_data["close"] < current_candle_data["open"]
                )
                limited_undershoot = limited_overshoot
                distance = current_candle_data["high"] - level_low

            reacted = approached_zone and limited_undershoot and recovered
            if reacted:
                logging.debug(
                    f"{symbol}: {level_name} zone reaction accepted. "
                    f"ZoneLow={level_low:.4f}, ZoneHigh={level_high:.4f}, "
                    f"Distance={distance:.4f}, NearBuffer={near_buffer:.4f}, "
                    f"PierceBuffer={pierce_buffer:.4f}, CloseBuffer={close_buffer:.4f}"
                )
            return reacted

        def has_impulse_retest_structure():
            prior_candles = today_df.iloc[:-1].copy().reset_index(drop=True)
            if len(prior_candles) < 4:
                return False
            window = prior_candles.tail(IMPULSE_LOOKBACK_BARS).reset_index(drop=True)
            if len(window) < 4:
                return False

            if direction == "long":
                start_pos = int(window["low"].idxmin())
                if start_pos >= len(window) - 1:
                    return False
                tail = window.iloc[start_pos:]
                end_pos = int(tail["high"].idxmax())
                if end_pos < start_pos or end_pos >= len(window):
                    return False
                impulse_low = float(window.iloc[start_pos]["low"])
                impulse_high = float(window.iloc[end_pos]["high"])
                retrace = impulse_high - float(current_candle_data["low"])
            else:
                start_pos = int(window["high"].idxmax())
                if start_pos >= len(window) - 1:
                    return False
                tail = window.iloc[start_pos:]
                end_pos = int(tail["low"].idxmin())
                if end_pos < start_pos or end_pos >= len(window):
                    return False
                impulse_high = float(window.iloc[start_pos]["high"])
                impulse_low = float(window.iloc[end_pos]["low"])
                retrace = float(current_candle_data["high"]) - impulse_low

            impulse_size = impulse_high - impulse_low
            if impulse_size <= 0:
                return False
            impulse_atr_reference = intraday_impulse_atr if intraday_impulse_atr is not None else atr
            if impulse_size < (IMPULSE_MIN_ATR * impulse_atr_reference):
                return False

            bars_in_leg = end_pos - start_pos + 1
            if bars_in_leg > IMPULSE_MAX_BARS:
                return False

            window_start_in_prior = len(prior_candles) - len(window)
            end_pos_global = window_start_in_prior + end_pos
            bars_since_end = len(prior_candles) - 1 - end_pos_global
            if bars_since_end > IMPULSE_RETEST_MAX_BARS:
                return False

            retrace_frac = retrace / impulse_size
            if retrace_frac < IMPULSE_RETRACE_MIN_FRAC or retrace_frac > IMPULSE_RETRACE_MAX_FRAC:
                return False

            logging.debug(
                f"{symbol}: Impulse/retest structure accepted. "
                f"Impulse={impulse_size:.4f}, BarsInLeg={bars_in_leg}, "
                f"BarsSinceImpulse={bars_since_end}, RetraceFrac={retrace_frac:.3f}"
            )
            return True

        def is_compressed_around_levels(level_low, level_high):
            if level_low is None or level_high is None:
                return False
            prior = today_df.iloc[-(IMPULSE_CHOP_LOOKBACK_BARS + 1):-1].copy()
            if prior.empty:
                return False

            band = IMPULSE_LEVEL_BAND_ATR * atr
            near_mask = prior["close"].between(level_low - band, level_high + band)
            near_count = int(near_mask.sum())
            if near_count > IMPULSE_MAX_NEAR_BARS:
                logging.debug(
                    f"{symbol}: Rejecting impulse retest due to compression near VWAP/EOD. "
                    f"NearBars={near_count}, MaxNearBars={IMPULSE_MAX_NEAR_BARS}"
                )
                return True

            states = []
            upper = level_high + band
            lower = level_low - band
            for close_val in prior["close"]:
                if close_val > upper:
                    states.append(1)
                elif close_val < lower:
                    states.append(-1)
                else:
                    states.append(0)

            crosses = 0
            prev_state = 0
            for state in states:
                if state == 0:
                    continue
                if prev_state != 0 and state != prev_state:
                    crosses += 1
                prev_state = state

            if crosses > IMPULSE_MAX_CROSSES:
                logging.debug(
                    f"{symbol}: Rejecting impulse retest due to repeated level crossing. "
                    f"Crosses={crosses}, MaxCrosses={IMPULSE_MAX_CROSSES}"
                )
                return True
            return False

        def build_zone_bounds(zone_key, bars, fallback_low, fallback_high):
            if bars.empty:
                return []
            zone_map = level_series_maps.get(zone_key, {})
            bounds = []
            for _, row in bars.iterrows():
                dt = row.get("datetime")
                if zone_key == "vwap_eod_confluence":
                    low_high = zone_map.get(dt)
                    if low_high is None:
                        low_val, high_val = fallback_low, fallback_high
                    else:
                        low_val, high_val = low_high
                else:
                    value = zone_map.get(dt)
                    if value is None:
                        low_val = fallback_low
                        high_val = fallback_high
                    else:
                        low_val = value
                        high_val = value
                bounds.append((row, low_val, high_val))
            return bounds

        def passes_level_quality_filter(zone_key, level_low, level_high, label):
            min_required_bars = max(BOUNCE_ONE_SIDED_BARS, 3)
            if len(today_df) <= min_required_bars:
                return True

            prior = today_df.iloc[-(BOUNCE_LEVEL_LOOKBACK_BARS + 1):-1].copy()
            if len(prior) < min_required_bars:
                return True

            zone_rows = build_zone_bounds(zone_key, prior, level_low, level_high)
            if not zone_rows:
                return True

            band = BOUNCE_LEVEL_BAND_ATR * atr
            near_count = 0
            crosses = 0
            prev_state = 0
            max_separation = 0.0
            centers = []
            directional_accepts = 0
            acceptance_rows = zone_rows[-BOUNCE_ONE_SIDED_BARS:]

            for idx, (row, row_low, row_high) in enumerate(zone_rows):
                close_val = float(row["close"])
                high_val = float(row["high"])
                low_val = float(row["low"])

                if (row_low - band) <= close_val <= (row_high + band):
                    near_count += 1
                    state = 0
                elif close_val > (row_high + band):
                    state = 1
                else:
                    state = -1

                if prev_state != 0 and state != 0 and state != prev_state:
                    crosses += 1
                if state != 0:
                    prev_state = state

                centers.append((row_low + row_high) / 2.0)

                if direction == "long":
                    max_separation = max(max_separation, (high_val - row_high) / atr)
                else:
                    max_separation = max(max_separation, (row_low - low_val) / atr)

                if idx >= len(zone_rows) - len(acceptance_rows):
                    if direction == "long" and close_val >= (row_high - band):
                        directional_accepts += 1
                    elif direction == "short" and close_val <= (row_low + band):
                        directional_accepts += 1

            one_sided_ratio = directional_accepts / max(1, len(acceptance_rows))
            one_sided_ok = one_sided_ratio >= BOUNCE_ONE_SIDED_RATIO
            separation_ok = max_separation >= BOUNCE_MIN_SEPARATION_ATR

            slope_ok = True
            if len(centers) > BOUNCE_LEVEL_SLOPE_LOOKBACK:
                slope_delta = (centers[-1] - centers[-1 - BOUNCE_LEVEL_SLOPE_LOOKBACK]) / atr
                if direction == "long":
                    slope_ok = slope_delta >= BOUNCE_MIN_LEVEL_SLOPE_ATR
                else:
                    slope_ok = slope_delta <= -BOUNCE_MIN_LEVEL_SLOPE_ATR

            reaction_prior = today_df.iloc[-(BOUNCE_REACTION_LOOKBACK_BARS + 1):-1].copy()
            body_ok = True
            range_ok = True
            if not reaction_prior.empty:
                prior_bodies = (reaction_prior["close"] - reaction_prior["open"]).abs()
                prior_ranges = reaction_prior["high"] - reaction_prior["low"]
                median_body = float(prior_bodies.median()) if not prior_bodies.empty else 0.0
                median_range = float(prior_ranges.median()) if not prior_ranges.empty else 0.0
                current_body = abs(current_candle_data["close"] - current_candle_data["open"])
                current_range = current_candle_data["high"] - current_candle_data["low"]
                if median_body > 0:
                    body_ok = current_body >= (median_body * BOUNCE_MIN_BODY_TO_MEDIAN)
                if median_range > 0:
                    range_ok = current_range >= (median_range * BOUNCE_MIN_RANGE_TO_MEDIAN)

            volume_ok = True
            volume_prior = today_df.iloc[-(BOUNCE_VOLUME_LOOKBACK_BARS + 1):-1].copy()
            if not volume_prior.empty:
                avg_volume = float(volume_prior["volume"].mean())
                if avg_volume > 0:
                    volume_ok = current_candle_data["volume"] >= (avg_volume * BOUNCE_MIN_RELATIVE_VOLUME)

            passes = (
                near_count <= BOUNCE_MAX_NEAR_BARS
                and crosses <= BOUNCE_MAX_CROSSES
                and one_sided_ok
                and separation_ok
                and slope_ok
                and body_ok
                and range_ok
                and volume_ok
            )

            if not passes:
                logging.debug(
                    f"{symbol}: Rejecting {label} bounce due to chop/quality filter. "
                    f"NearBars={near_count}, Crosses={crosses}, Separation={max_separation:.2f}, "
                    f"OneSidedRatio={one_sided_ratio:.2f}, SlopeOK={slope_ok}, "
                    f"BodyOK={body_ok}, RangeOK={range_ok}, VolumeOK={volume_ok}"
                )
            return passes

        # Function to check if a candle respects a level based on direction
        def respects_level(candle, level_value, direction, threshold):
            if direction == "long":
                # For longs, low should be above level (with threshold leeway)
                return candle["low"] >= (level_value - threshold)
            else:
                # For shorts, high should be below level (with threshold leeway)
                return candle["high"] <= (level_value + threshold)

        # Function to check consecutive candles respect a level
        def check_consecutive_respect(level_value, level_name):
            if not CHECK_CONSECUTIVE_CANDLES or level_value is None:
                return True  # Skip check if disabled or level is None

            if len(today_df) <= CONSECUTIVE_CANDLES:
                return False  # Not enough candles to check

            # Get N candles prior to current candle
            prev_n_candles = today_df.iloc[-(CONSECUTIVE_CANDLES+1):-1]

            # Check each candle respects the level
            respect_count = sum(1 for _, candle in prev_n_candles.iterrows()
                            if respects_level(candle, level_value, direction, threshold))

            level_respected = respect_count >= CONSECUTIVE_CANDLES

            if level_respected:
                logging.debug(f"{symbol}: {level_name} respected for {respect_count}/{CONSECUTIVE_CANDLES} candles")
            else:
                logging.debug(f"{symbol}: {level_name} only respected for {respect_count}/{CONSECUTIVE_CANDLES} candles, skipping")

            return level_respected

        def check_previous_day_level_session_respect(level_value, level_name):
            if level_value is None:
                return False

            prior = today_df.iloc[:-1].copy()
            if len(prior) < PREV_DAY_LEVEL_MIN_PRIOR_BARS:
                logging.debug(
                    f"{symbol}: Not enough prior candles for {level_name} session-respect check "
                    f"({len(prior)}/{PREV_DAY_LEVEL_MIN_PRIOR_BARS})"
                )
                return False

            if direction == "long":
                respect_mask = prior["close"] >= (level_value - threshold)
            else:
                respect_mask = prior["close"] <= (level_value + threshold)

            respect_count = int(respect_mask.sum())
            wrong_side_count = len(prior) - respect_count
            respect_ratio = respect_count / max(1, len(prior))

            recent = prior.tail(min(PREV_DAY_LEVEL_RECENT_RESPECT_BARS, len(prior)))
            if direction == "long":
                recent_respected = bool((recent["close"] >= (level_value - threshold)).all())
            else:
                recent_respected = bool((recent["close"] <= (level_value + threshold)).all())

            respected = (
                respect_ratio >= PREV_DAY_LEVEL_RESPECT_RATIO
                and wrong_side_count <= PREV_DAY_LEVEL_MAX_WRONG_SIDE_CLOSES
                and recent_respected
            )
            if respected:
                logging.debug(
                    f"{symbol}: {level_name} session respect accepted. "
                    f"RespectRatio={respect_ratio:.2f}, WrongSideCloses={wrong_side_count}, "
                    f"RecentRespected={recent_respected}"
                )
            else:
                logging.debug(
                    f"{symbol}: Rejecting {level_name} bounce because the level was not respected "
                    f"through the session. RespectRatio={respect_ratio:.2f}, "
                    f"WrongSideCloses={wrong_side_count}, RecentRespected={recent_respected}"
                )
            return respected

        def has_recent_prior_ema_touch(ema_key, ema_value, ema_label):
            if ema_value is None or len(today_df) <= 1:
                return False

            span_by_key = {"ema_8": 8, "ema_15": 15, "ema_21": 21}
            span = span_by_key.get(ema_key)
            if span is None or len(today_df) < span:
                return False

            lookback = min(EMA_FRESH_TOUCH_LOOKBACK_BARS, len(today_df) - 1)
            ema_frame = today_df[["high", "low"]].copy()
            ema_frame["ema"] = today_df["close"].ewm(span=span, adjust=False).mean()
            prior = ema_frame.iloc[-(lookback + 1):-1]
            touch_buffer = max(threshold, EMA_FRESH_TOUCH_BUFFER_ATR * atr)

            for _, row in prior.iterrows():
                if row["high"] >= (row["ema"] - touch_buffer) and row["low"] <= (row["ema"] + touch_buffer):
                    logging.debug(
                        f"{symbol}: Rejecting {ema_label} bounce because EMA was already touched "
                        f"within the prior {lookback} candles."
                    )
                    return True
            return False

        # Check for 10-candle bounce if enabled
        if bounce_type_allowed("10_candle") and bounce_type_enabled("10_candle") and len(df) >= 11:
            if direction == "long":
                # For longs, check if current candle creates a new lowest low
                last_10_candles = df.iloc[-11:-1].copy()  # Exclude current candle
                lowest_low_prev = last_10_candles["low"].min()

                # Bounce condition: current candle creates a new low and closes above the open
                if current_candle_data["low"] < lowest_low_prev and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["10_candle_low"] = current_candle_data["low"]
                    mark_trigger("10_candle_low")
                    logging.debug(f"{symbol}: 10-candle LONG bounce candidate found. New low: {current_candle_data['low']:.2f}, Previous lowest: {lowest_low_prev:.2f}")
            else:
                # For shorts, check if current candle creates a new highest high
                last_10_candles = df.iloc[-11:-1].copy()
                highest_high_prev = last_10_candles["high"].max()

                # Bounce condition: current candle creates a new high and closes below the open
                if current_candle_data["high"] > highest_high_prev and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["10_candle_high"] = current_candle_data["high"]
                    mark_trigger("10_candle_high")
                    logging.debug(f"{symbol}: 10-candle SHORT bounce candidate found. New high: {current_candle_data['high']:.2f}, Previous highest: {highest_high_prev:.2f}")

        # Check for standard VWAP bounces if enabled
        if bounce_type_allowed("vwap") and bounce_type_enabled("vwap") and metrics.get("std_vwap") is not None:
            std_vwap = metrics.get("std_vwap")
            # Check if price respected standard VWAP for consecutive candles
            respected = check_consecutive_respect(std_vwap, "Standard VWAP")
            reclaim = is_recent_reclaim(std_vwap, "Standard VWAP")
            zone_reaction = is_vwap_zone_reaction(std_vwap, std_vwap, "Standard VWAP")
            quality_ok = passes_level_quality_filter("vwap", std_vwap, std_vwap, "Standard VWAP")
            touch_reject = is_touch_reject(std_vwap)
            if quality_ok and ((respected and (touch_reject or zone_reaction)) or reclaim):
                ref_levels["vwap"] = std_vwap
                mark_trigger("vwap")
                logging.debug(
                    f"{symbol}: Standard VWAP bounce candidate found ({'reclaim' if reclaim else 'zone reaction' if zone_reaction and not touch_reject else 'touch/reject'}). "
                    f"VWAP: {std_vwap:.2f}, Close: {current_candle_data['close']:.2f}"
                )

        # Check for Dynamic VWAP bounces if enabled
        if bounce_type_allowed("dynamic_vwap") and bounce_type_enabled("dynamic_vwap") and metrics.get("dynamic_vwap") is not None:
            dynamic_vwap = metrics.get("dynamic_vwap")
            # Check if price respected dynamic VWAP for consecutive candles
            respected = check_consecutive_respect(dynamic_vwap, "Dynamic VWAP")
            zone_reaction = is_vwap_zone_reaction(dynamic_vwap, dynamic_vwap, "Dynamic VWAP")
            quality_ok = passes_level_quality_filter("dynamic_vwap", dynamic_vwap, dynamic_vwap, "Dynamic VWAP")
            touch_reject = is_touch_reject(dynamic_vwap)
            if quality_ok and respected and (touch_reject or zone_reaction):
                ref_levels["dynamic_vwap"] = dynamic_vwap
                mark_trigger("dynamic_vwap")
                logging.debug(
                    f"{symbol}: Dynamic VWAP bounce candidate found "
                    f"({'zone reaction' if zone_reaction and not touch_reject else 'touch/reject'}). "
                    f"DVWAP: {dynamic_vwap:.2f}, Close: {current_candle_data['close']:.2f}"
                )

        # Check for EOD VWAP bounces if enabled
        if bounce_type_allowed("eod_vwap") and bounce_type_enabled("eod_vwap") and metrics.get("eod_vwap") is not None:
            eod_vwap = metrics.get("eod_vwap")
            # Check if price respected EOD VWAP for consecutive candles
            respected = check_consecutive_respect(eod_vwap, "EOD VWAP")
            reclaim = is_recent_reclaim(eod_vwap, "EOD VWAP")
            zone_reaction = is_vwap_zone_reaction(eod_vwap, eod_vwap, "EOD VWAP")
            quality_ok = passes_level_quality_filter("eod_vwap", eod_vwap, eod_vwap, "EOD VWAP")
            touch_reject = is_touch_reject(eod_vwap)
            if quality_ok and ((respected and (touch_reject or zone_reaction)) or reclaim):
                ref_levels["eod_vwap"] = eod_vwap
                mark_trigger("eod_vwap")
                logging.debug(
                    f"{symbol}: EOD VWAP bounce candidate found ({'reclaim' if reclaim else 'zone reaction' if zone_reaction and not touch_reject else 'touch/reject'}). "
                    f"EOD VWAP: {eod_vwap:.2f}, Close: {current_candle_data['close']:.2f}"
                )

        if (
            bounce_type_allowed("vwap_eod_confluence")
            and bounce_type_enabled("vwap_eod_confluence")
        ):
            std_vwap = metrics.get("std_vwap")
            eod_vwap = metrics.get("eod_vwap")
            if std_vwap is not None and eod_vwap is not None:
                level_spread = abs(std_vwap - eod_vwap)
                max_allowed_spread = CONFLUENCE_MAX_SPREAD_ATR * atr
                if level_spread <= max_allowed_spread:
                    confluence_level = max(std_vwap, eod_vwap) if direction == "long" else min(std_vwap, eod_vwap)
                    touch_reject = is_touch_reject(confluence_level)
                    zone_reaction = is_vwap_zone_reaction(
                        min(std_vwap, eod_vwap),
                        max(std_vwap, eod_vwap),
                        "VWAP+EOD Confluence",
                    )
                    reclaim = is_recent_reclaim(confluence_level, "VWAP+EOD Confluence")
                    quality_ok = passes_level_quality_filter(
                        "vwap_eod_confluence",
                        min(std_vwap, eod_vwap),
                        max(std_vwap, eod_vwap),
                        "VWAP+EOD Confluence",
                    )
                    if quality_ok and (touch_reject or zone_reaction or reclaim):
                        ref_levels["vwap_eod_confluence"] = confluence_level
                        mark_trigger("vwap_eod_confluence")
                        ref_levels.setdefault("vwap", std_vwap)
                        ref_levels.setdefault("eod_vwap", eod_vwap)
                        logging.debug(
                            f"{symbol}: VWAP+EOD confluence bounce candidate found "
                            f"({'reclaim' if reclaim else 'zone reaction' if zone_reaction and not touch_reject else 'touch/reject'}). "
                            f"Spread: {level_spread:.4f} (max {max_allowed_spread:.4f})"
                        )

        if (
            bounce_type_allowed("impulse_retest_vwap_eod")
            and bounce_type_enabled("impulse_retest_vwap_eod")
        ):
            std_vwap = metrics.get("std_vwap")
            eod_vwap = metrics.get("eod_vwap")
            if std_vwap is not None and eod_vwap is not None:
                std_vwap = float(std_vwap)
                eod_vwap = float(eod_vwap)
                level_low = min(std_vwap, eod_vwap)
                level_high = max(std_vwap, eod_vwap)
                touch_buffer = max(threshold, IMPULSE_RETEST_TOUCH_ATR * atr)
                pierce_buffer = max(threshold, IMPULSE_RETEST_PIERCE_ATR * atr)
                close_buffer = IMPULSE_RETEST_CLOSE_BUFFER_ATR * atr
                touched_retest = None
                level_spread = abs(std_vwap - eod_vwap)

                if level_spread <= CONFLUENCE_MAX_SPREAD_ATR * atr:
                    if direction == "long":
                        touched_zone = (
                            current_candle_data["low"] <= (level_high + touch_buffer)
                            and current_candle_data["low"] >= (level_low - pierce_buffer)
                        )
                        recovered_zone = (
                            current_candle_data["close"] >= (level_high - close_buffer)
                            and current_candle_data["close"] > current_candle_data["open"]
                        )
                    else:
                        touched_zone = (
                            current_candle_data["high"] >= (level_low - touch_buffer)
                            and current_candle_data["high"] <= (level_high + pierce_buffer)
                        )
                        recovered_zone = (
                            current_candle_data["close"] <= (level_low + close_buffer)
                            and current_candle_data["close"] < current_candle_data["open"]
                        )
                    if touched_zone and recovered_zone:
                        touched_retest = {
                            "level": (level_low + level_high) / 2.0,
                            "low": level_low,
                            "high": level_high,
                            "keys": ("vwap", "eod_vwap"),
                            "label": "VWAP/EOD confluence",
                        }

                if touched_retest is None:
                    touched_levels = []
                    for key, level_value in (("vwap", std_vwap), ("eod_vwap", eod_vwap)):
                        if direction == "long":
                            touched_level = (
                                current_candle_data["low"] <= (level_value + touch_buffer)
                                and current_candle_data["low"] >= (level_value - pierce_buffer)
                            )
                            recovered_level = (
                                current_candle_data["close"] >= (level_value - close_buffer)
                                and current_candle_data["close"] > current_candle_data["open"]
                            )
                        else:
                            touched_level = (
                                current_candle_data["high"] >= (level_value - touch_buffer)
                                and current_candle_data["high"] <= (level_value + pierce_buffer)
                            )
                            recovered_level = (
                                current_candle_data["close"] <= (level_value + close_buffer)
                                and current_candle_data["close"] < current_candle_data["open"]
                            )
                        if touched_level and recovered_level:
                            touched_levels.append((key, level_value))

                    if touched_levels:
                        selected_key, selected_level = (
                            max(touched_levels, key=lambda item: item[1])
                            if direction == "long"
                            else min(touched_levels, key=lambda item: item[1])
                        )
                        touched_retest = {
                            "level": selected_level,
                            "low": selected_level,
                            "high": selected_level,
                            "keys": (selected_key,),
                            "label": BOUNCE_TYPE_LABELS.get(selected_key, selected_key),
                        }

                if touched_retest is None:
                    logging.debug(
                        f"{symbol}: Rejecting impulse retest because current candle did not retest "
                        f"VWAP/EOD directly. StdVWAP={std_vwap:.4f}, EOD={eod_vwap:.4f}, "
                        f"TouchBuffer={touch_buffer:.4f}, PierceBuffer={pierce_buffer:.4f}"
                    )

                if (
                    touched_retest is not None
                    and has_impulse_retest_structure()
                    and self._impulse_regime_transition_ok(symbol, direction, intraday_rrs_profile)
                ):
                    if not is_compressed_around_levels(touched_retest["low"], touched_retest["high"]):
                        ref_levels["impulse_retest_vwap_eod"] = touched_retest["level"]
                        mark_trigger("impulse_retest_vwap_eod")
                        if "vwap" in touched_retest["keys"]:
                            ref_levels.setdefault("vwap", std_vwap)
                        if "eod_vwap" in touched_retest["keys"]:
                            ref_levels.setdefault("eod_vwap", eod_vwap)
                        logging.debug(
                            f"{symbol}: Impulse retest VWAP/EOD candidate found. "
                            f"Retest={touched_retest['label']}, Level={touched_retest['level']:.4f}, "
                            f"TouchBuffer={touch_buffer:.4f}, PierceBuffer={pierce_buffer:.4f}"
                        )

        # Check EMA bounces (must also be on the correct side of standard VWAP)
        for ema_key, ema_label in (("ema_8", "8 EMA"), ("ema_15", "15 EMA"), ("ema_21", "21 EMA")):
            if not bounce_type_allowed(ema_key) or not bounce_type_enabled(ema_key):
                continue
            ema_value = metrics.get(ema_key)
            std_vwap = metrics.get("std_vwap")
            if ema_value is None or std_vwap is None:
                continue
            if not check_consecutive_respect(ema_value, ema_label):
                continue
            if has_recent_prior_ema_touch(ema_key, ema_value, ema_label):
                continue

            if direction == "long":
                is_above_vwap = current_candle_data["close"] > std_vwap
                if ema_key == "ema_21":
                    touch_buffer = max(EMA_21_TOUCH_BUFFER_MIN, EMA_21_TOUCH_BUFFER_ATR * atr)
                    touched_ema = candle_interacts_with_level(ema_value, touch_buffer)
                else:
                    touched_ema = abs(current_candle_data["low"] - ema_value) <= threshold
                clean_bounce = (
                    touched_ema
                    and current_candle_data["close"] > current_candle_data["open"]
                    and current_candle_data["close"] > ema_value
                )
                if is_above_vwap and clean_bounce:
                    ref_levels[ema_key] = ema_value
                    mark_trigger(ema_key)
                    logging.debug(
                        f"{symbol}: {ema_label} LONG bounce candidate found. {ema_label}: {ema_value:.2f}, Std VWAP: {std_vwap:.2f}"
                    )
            elif direction == "short":
                is_below_vwap = current_candle_data["close"] < std_vwap
                if ema_key == "ema_21":
                    touch_buffer = max(EMA_21_TOUCH_BUFFER_MIN, EMA_21_TOUCH_BUFFER_ATR * atr)
                    touched_ema = candle_interacts_with_level(ema_value, touch_buffer)
                else:
                    touched_ema = abs(current_candle_data["high"] - ema_value) <= threshold
                clean_bounce = (
                    touched_ema
                    and current_candle_data["close"] < current_candle_data["open"]
                    and current_candle_data["close"] < ema_value
                )
                if is_below_vwap and clean_bounce:
                    ref_levels[ema_key] = ema_value
                    mark_trigger(ema_key)
                    logging.debug(
                        f"{symbol}: {ema_label} SHORT bounce candidate found. {ema_label}: {ema_value:.2f}, Std VWAP: {std_vwap:.2f}"
                    )

        # Check for VWAP upper band bounces for longs
        if bounce_type_allowed("vwap_upper_band") and bounce_type_enabled("vwap_upper_band") and direction == "long" and metrics.get("vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_upper"), "VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["vwap_upper_band"] = metrics.get("vwap_1stdev_upper")
                    mark_trigger("vwap_upper_band")
                    logging.debug(f"{symbol}: VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for VWAP lower band bounces for shorts
        if bounce_type_allowed("vwap_lower_band") and bounce_type_enabled("vwap_lower_band") and direction == "short" and metrics.get("vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_lower"), "VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["vwap_lower_band"] = metrics.get("vwap_1stdev_lower")
                    mark_trigger("vwap_lower_band")
                    logging.debug(f"{symbol}: VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for Dynamic VWAP upper band bounces for longs
        if bounce_type_allowed("dynamic_vwap_upper_band") and bounce_type_enabled("dynamic_vwap_upper_band") and direction == "long" and metrics.get("dynamic_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_upper"), "Dynamic VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("dynamic_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["dynamic_vwap_upper_band"] = metrics.get("dynamic_vwap_1stdev_upper")
                    mark_trigger("dynamic_vwap_upper_band")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('dynamic_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for Dynamic VWAP lower band bounces for shorts
        if bounce_type_allowed("dynamic_vwap_lower_band") and bounce_type_enabled("dynamic_vwap_lower_band") and direction == "short" and metrics.get("dynamic_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_lower"), "Dynamic VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("dynamic_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["dynamic_vwap_lower_band"] = metrics.get("dynamic_vwap_1stdev_lower")
                    mark_trigger("dynamic_vwap_lower_band")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('dynamic_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for EOD VWAP upper band bounces for longs
        if bounce_type_allowed("eod_vwap_upper_band") and bounce_type_enabled("eod_vwap_upper_band") and direction == "long" and metrics.get("eod_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_upper"), "EOD VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("eod_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["eod_vwap_upper_band"] = metrics.get("eod_vwap_1stdev_upper")
                    mark_trigger("eod_vwap_upper_band")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('eod_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for EOD VWAP lower band bounces for shorts
        if bounce_type_allowed("eod_vwap_lower_band") and bounce_type_enabled("eod_vwap_lower_band") and direction == "short" and metrics.get("eod_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_lower"), "EOD VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("eod_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["eod_vwap_lower_band"] = metrics.get("eod_vwap_1stdev_lower")
                    mark_trigger("eod_vwap_lower_band")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('eod_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for previous day high/low bounces if enabled
        if direction == "long" and bounce_type_allowed("prev_day_high") and bounce_type_enabled("prev_day_high") and metrics.get("prev_high") is not None:
            # Check if price respected previous day high for consecutive candles
            if (
                check_consecutive_respect(metrics.get("prev_high"), "Previous Day High")
                and check_previous_day_level_session_respect(metrics.get("prev_high"), "Previous Day High")
            ):
                # Only consider bounce if price respected the level through the session.
                if abs(current_candle_data["low"] - metrics.get("prev_high")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["prev_day_high"] = metrics.get("prev_high")
                    mark_trigger("prev_day_high")
                    logging.debug(f"{symbol}: Previous Day High LONG bounce candidate found. Prev High: {metrics.get('prev_high'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        elif direction == "short" and bounce_type_allowed("prev_day_low") and bounce_type_enabled("prev_day_low") and metrics.get("prev_low") is not None:
            # Check if price respected previous day low for consecutive candles
            if (
                check_consecutive_respect(metrics.get("prev_low"), "Previous Day Low")
                and check_previous_day_level_session_respect(metrics.get("prev_low"), "Previous Day Low")
            ):
                # Only consider bounce if price respected the level through the session.
                if abs(current_candle_data["high"] - metrics.get("prev_low")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["prev_day_low"] = metrics.get("prev_low")
                    mark_trigger("prev_day_low")
                    logging.debug(f"{symbol}: Previous Day Low SHORT bounce candidate found. Prev Low: {metrics.get('prev_low'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Return None if no reference levels were found, otherwise return the details
        fast_confirm = any(level in FAST_CONFIRM_BOUNCE_TYPES for level in triggered_levels)
        return {
            "levels": ref_levels,
            "triggered_levels": triggered_levels,
            "candle": current_candle_data,
            "confirm_immediately": fast_confirm,
            "max_confirmation_candles": (
                FAST_CONFIRMATION_MAX_CANDLES if fast_confirm else BOUNCE_CONFIRMATION_MAX_CANDLES
            ),
        } if ref_levels else None



    def request_and_detect_bounce(self, symbol, allowed_bounce_types=None, *, scan_for_new_bounces=True):
        symbol = str(symbol or "").strip().upper()
        direction = self.get_symbol_direction(symbol)
        if direction not in {"long", "short"}:
            logging.debug(f"{symbol}: No long/short direction available, skipping bounce detection.")
            return

        # Only scan within market hours (if enabled)
        if not SCAN_OUTSIDE_MARKET_HOURS:
            current_time = get_market_local_now()
            market_session = get_market_session_window(reference=current_time)
            if not (market_session.open_local <= current_time <= market_session.close_local):
                logging.debug(f"{symbol}: Not within market hours for bounce detection.")
                return

        # Request 5 days of data to ensure we get enough market days
        five_day_reqId = self.getReqId()
        self.data[five_day_reqId] = []
        self.data_ready_events[five_day_reqId] = threading.Event()
        contract = self.create_stock_contract(symbol)

        # Request historical data from IB
        self.reqHistoricalData(
            reqId=five_day_reqId,
            contract=contract,
            endDateTime="",  # up to now
            durationStr="5 D",  # Increased to 5 days to account for weekends/holidays
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        # Wait for data with timeout
        if not self.data_ready_events[five_day_reqId].wait(timeout=15):
            logging.warning(f"{symbol}: Timeout waiting for historical data.")
            del self.data_ready_events[five_day_reqId]
            return

        all_bars = self.data.get(five_day_reqId, [])
        if len(all_bars) < 10:
            logging.warning(f"{symbol}: Insufficient historical data, only {len(all_bars)} bars received")
            del self.data_ready_events[five_day_reqId]
            return

        # Clean up
        del self.data_ready_events[five_day_reqId]

        # Cache 5-minute bars for RRS reuse
        self.latest_bars[symbol] = _dedupe_bars(_bars_to_ib(all_bars))

        # Convert to DataFrame
        df = pd.DataFrame(all_bars)

        # Add datetime column for date filtering
        try:
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            if df["datetime"].isna().all():
                logging.error(f"{symbol}: Failed to convert any timestamps to datetime objects")
                return
        except Exception as e:
            logging.error(f"{symbol}: Error in datetime conversion: {e}")
            return

        # Get unique dates in the data, sorted in ascending order
        unique_dates = sorted(df["datetime"].dt.date.unique())
        logging.debug(f"{symbol}: Found data for these dates: {unique_dates}")

        if len(unique_dates) < 2:
            logging.warning(f"{symbol}: Need at least 2 days of data, but only found {len(unique_dates)} days")

        # The last date is today, the second-to-last date is the previous trading day
        current_date = unique_dates[-1] if unique_dates else None
        previous_date = unique_dates[-2] if len(unique_dates) >= 2 else None

        logging.debug(f"{symbol}: Current date = {current_date}, Previous trading date = {previous_date}")

        # Create separate dataframes for different time periods
        today_df = df[df["datetime"].dt.date == current_date].copy() if current_date else pd.DataFrame()
        prev_day_df = df[df["datetime"].dt.date == previous_date].copy() if previous_date else pd.DataFrame()

        logging.debug(f"{symbol}: Found {len(today_df)} bars for today and {len(prev_day_df)} bars for previous trading day")

        if today_df.empty:
            logging.warning(f"{symbol}: No data for today")
            return

        self.emit_master_avwap_intraday_trigger_flags(symbol, today_df)
        self._update_pending_bounce_outcomes(symbol, df)
        if not scan_for_new_bounces:
            return

        # 1. Calculate Standard VWAP (today only)
        standard_vwap = self.calculate_standard_vwap(df)
        if standard_vwap is not None:
            logging.debug(
                f"{symbol}: Standard VWAP calculated from {len(today_df)} today's bars = {standard_vwap:.4f}"
            )

        # 2. Calculate Dynamic VWAP (previous trading day + today)
        dynamic_vwap = self.calculate_dynamic_vwap(df)
        if dynamic_vwap is not None and not prev_day_df.empty:
            logging.debug(
                f"{symbol}: Dynamic VWAP calculated from {len(prev_day_df) + len(today_df)} bars (previous day + today) = {dynamic_vwap:.4f}"
            )
        elif dynamic_vwap is not None:
            logging.debug(
                f"{symbol}: Dynamic VWAP calculated from today's bars only = {dynamic_vwap:.4f}"
            )

        # 3. Calculate EOD VWAP (previous trading day's last candle + today)
        eod_vwap = self.calculate_eod_vwap(df)
        if eod_vwap is not None and not prev_day_df.empty:
            logging.debug(
                f"{symbol}: EOD VWAP calculated from {len(today_df) + 1} bars (prev day's last + today) = {eod_vwap:.4f}"
            )
        elif eod_vwap is not None:
            logging.debug(
                f"{symbol}: EOD VWAP calculated from today's bars only = {eod_vwap:.4f}"
            )

        # 4. Get previous day extremes
        prev_high = prev_day_df["high"].max() if not prev_day_df.empty else None
        prev_low = prev_day_df["low"].min() if not prev_day_df.empty else None

        logging.debug(f"{symbol}: Previous day high = {prev_high}, low = {prev_low}")

        # 5. Calculate short EMAs (today only)
        ema_8 = None
        ema_15 = None
        ema_21 = None
        if not today_df.empty:
            if len(today_df) >= 8:
                ema_8 = today_df["close"].ewm(span=8, adjust=False).mean().iloc[-1]
            if len(today_df) >= 15:
                ema_15 = today_df["close"].ewm(span=15, adjust=False).mean().iloc[-1]
            if len(today_df) >= 21:
                ema_21 = today_df["close"].ewm(span=21, adjust=False).mean().iloc[-1]

        # 6. Get current price
        current_price = today_df["close"].iloc[-1] if not today_df.empty else None

        # Calculate standard VWAP with bands
        vwap_value, vwap_upper_band, vwap_lower_band = self.calculate_vwap_with_stdev_bands(today_df)
        logging.debug(f"{symbol}: VWAP: {vwap_value:.4f}, Upper 1SD: {vwap_upper_band:.4f}, Lower 1SD: {vwap_lower_band:.4f}")

        # Calculate dynamic VWAP with bands
        dynamic_vwap, dynamic_upper_band, dynamic_lower_band = self.calculate_dynamic_vwap_with_stdev_bands(df)

        # Fix the logging to handle None values
        dynamic_vwap_str = f"{dynamic_vwap:.4f}" if dynamic_vwap is not None else "None"
        dynamic_upper_str = f"{dynamic_upper_band:.4f}" if dynamic_upper_band is not None else "None"
        dynamic_lower_str = f"{dynamic_lower_band:.4f}" if dynamic_lower_band is not None else "None"
        logging.debug(f"{symbol}: Dynamic VWAP: {dynamic_vwap_str}, Upper 1SD: {dynamic_upper_str}, Lower 1SD: {dynamic_lower_str}")

        # Calculate EOD VWAP with bands
        eod_vwap, eod_upper_band, eod_lower_band = self.calculate_eod_vwap_with_stdev_bands(df)

        # Fix the logging to handle None values
        eod_vwap_str = f"{eod_vwap:.4f}" if eod_vwap is not None else "None"
        eod_upper_str = f"{eod_upper_band:.4f}" if eod_upper_band is not None else "None"
        eod_lower_str = f"{eod_lower_band:.4f}" if eod_lower_band is not None else "None"
        logging.debug(f"{symbol}: EOD VWAP: {eod_vwap_str}, Upper 1SD: {eod_upper_str}, Lower 1SD: {eod_lower_str}")

        vwap_invalidation = self._build_vwap_invalidation_snapshot(df, direction)

        # Store all metrics in one comprehensive dictionary
        self.symbol_metrics[symbol] = {
            "std_vwap": standard_vwap,
            "dynamic_vwap": dynamic_vwap,
            "eod_vwap": eod_vwap,
            "prev_high": prev_high,
            "prev_low": prev_low,
            "price": current_price,
            "vwap_1stdev_upper": vwap_upper_band,
            "vwap_1stdev_lower": vwap_lower_band,
            "dynamic_vwap_1stdev_upper": dynamic_upper_band,
            "dynamic_vwap_1stdev_lower": dynamic_lower_band,
            "eod_vwap_1stdev_upper": eod_upper_band,
            "eod_vwap_1stdev_lower": eod_lower_band,
            "ema_8": ema_8,
            "ema_15": ema_15,
            "ema_21": ema_21,
            "vwap_invalidation": vwap_invalidation,
        }

        # Then continue with detailed logging if LOGGING_MODE is enabled
        if LOGGING_MODE:
            msg = f"{symbol} Metrics -> "
            msg += f"Price: {current_price:.2f}, " if current_price is not None else "Price: N/A, "
            msg += f"Std VWAP: {standard_vwap:.4f}, " if standard_vwap is not None else "Std VWAP: N/A, "
            msg += f"Dynamic VWAP: {dynamic_vwap:.4f}, " if dynamic_vwap is not None else "Dynamic VWAP: N/A, "
            msg += f"EOD VWAP: {eod_vwap:.4f}, " if eod_vwap is not None else "EOD VWAP: N/A, "
            msg += f"VWAP 1SD Upper: {vwap_upper_band:.4f}, " if vwap_upper_band is not None else "VWAP 1SD Upper: N/A, "
            msg += f"VWAP 1SD Lower: {vwap_lower_band:.4f}, " if vwap_lower_band is not None else "VWAP 1SD Lower: N/A, "
            msg += f"Dynamic VWAP 1SD Upper: {dynamic_upper_band:.4f}, " if dynamic_upper_band is not None else "Dynamic VWAP 1SD Upper: N/A, "
            msg += f"Dynamic VWAP 1SD Lower: {dynamic_lower_band:.4f}, " if dynamic_lower_band is not None else "Dynamic VWAP 1SD Lower: N/A, "
            msg += f"EOD VWAP 1SD Upper: {eod_upper_band:.4f}, " if eod_upper_band is not None else "EOD VWAP 1SD Upper: N/A, "
            msg += f"EOD VWAP 1SD Lower: {eod_lower_band:.4f}, " if eod_lower_band is not None else "EOD VWAP 1SD Lower: N/A, "
            msg += f"21 EMA: {ema_21:.4f}, " if ema_21 is not None else "21 EMA: N/A, "

            if direction == "long":
                msg += f"Prev Day High: {prev_high:.4f}, " if prev_high is not None else "Prev Day High: N/A, "
            elif direction == "short":
                msg += f"Prev Day Low: {prev_low:.4f}, " if prev_low is not None else "Prev Day Low: N/A, "

            atr = self.atr_cache.get(symbol, None)
            msg += f"ATR: {atr:.4f}" if atr is not None else "ATR: N/A"
            self.log_symbol(symbol, msg)


        # Continue with evaluating bounce candidates. Human focus picks get
        # side-matching always-on alerts, so their candidate detection bypasses
        # the visible per-type toggles while normal symbols keep those filters.
        effective_allowed_types, include_disabled_for_focus = self._bounce_eval_filter_options_for_symbol(
            symbol,
            direction,
            allowed_bounce_types,
        )
        candidate_info = self.evaluate_bounce_candidate(
            symbol,
            df,
            allowed_bounce_types=effective_allowed_types,
            include_disabled_bounce_types=include_disabled_for_focus,
        )
        learning_candidate_info = self.evaluate_bounce_candidate(
            symbol,
            df,
            allowed_bounce_types=BOUNCE_LEARNING_TYPE_KEYS,
            include_disabled_bounce_types=True,
        )
        h1_mid_earnings_candidate = self._evaluate_master_avwap_mid_earnings_h1_bounce(
            symbol,
            direction,
            current_date,
        )
        if h1_mid_earnings_candidate:
            candidate_info = self._merge_bounce_candidate_learning_levels(
                h1_mid_earnings_candidate,
                learning_candidate_info,
            )
        elif candidate_info:
            candidate_info = self._merge_bounce_candidate_learning_levels(
                candidate_info,
                learning_candidate_info,
            )
        elif learning_candidate_info:
            candidate_info = dict(learning_candidate_info)
            candidate_info["learning_only"] = True

        def confirm_bounce_alert(
            levels,
            levels_list,
            bounce_candle,
            current_candle,
            *,
            candidate_id="",
            candles_waited=0,
            reason=None,
            learning_only=False,
        ):
            event_row = self._log_bounce_candidate_event(
                "confirmed",
                symbol,
                direction,
                levels,
                bounce_candle,
                current_candle,
                threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
                reason=reason or (
                    "Confirmed by close above bounce candle high."
                    if direction == "long"
                    else "Confirmed by close below bounce candle low."
                ),
                candidate_id=candidate_id,
                candles_waited=candles_waited,
            )
            self._register_bounce_outcome(
                symbol,
                direction,
                levels,
                bounce_candle,
                current_candle.to_dict() if isinstance(current_candle, pd.Series) else current_candle,
                candidate_id,
            )

            # Learning-loop verdict: tier from measured segment performance, and
            # an evidence-based mute for proven-negative segments. Muted bounces
            # keep recording (outcome already registered above) so a segment can
            # earn its way back; human focus picks are never muted.
            quality = self._evaluate_bounce_alert_quality(direction, levels, event_row)
            human_pick = bool(event_row.get("human_focus_pick"))
            muted_by_learning = bool(quality.get("muted") and not human_pick)
            bounce_msg = _format_bounce_alert_message(
                symbol, direction, levels_list, event_row, quality,
                exit_note=self._measured_exit_suffix(direction, levels),
            )
            if learning_only or muted_by_learning:
                mute_note = "; ".join(quality.get("mute_reasons") or []) or "learning-only candidate"
                self.log_symbol(symbol, f"LEARNING_ONLY [{quality.get('tier', '?')}]: {bounce_msg} | {mute_note}")
                return

            bounce_payload = self._build_bounce_feedback_alert_payload(bounce_msg, event_row)
            if self.gui_callback:
                self.gui_callback(bounce_payload, "green" if direction == "long" else "red")
            self._emit_master_avwap_second_stdev_bounce_alert(symbol, direction, levels_list)
            self.log_symbol(symbol, f"ALERT: {bounce_msg}")
            self.log_bounce_to_file(
                symbol=symbol,
                direction=direction,
                levels=levels,
                bounce_candle=bounce_candle,
                current_candle=current_candle,
                threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
                quality=quality,
            )
            self.alerted_symbols.add(symbol)

        if h1_mid_earnings_candidate and h1_mid_earnings_candidate.get("confirm_immediately"):
            active_h1_candidate = candidate_info or h1_mid_earnings_candidate
            levels = active_h1_candidate["levels"]
            levels_list = active_h1_candidate.get("triggered_levels") or list(levels.keys())
            h1_event_key = h1_mid_earnings_candidate.get("h1_event_key")
            if h1_event_key and h1_event_key not in self.emitted_h1_mid_earnings_bounce_alerts:
                candidate_id = self._make_bounce_event_id(
                    symbol,
                    direction,
                    active_h1_candidate["candle"],
                    levels,
                )
                confirm_bounce_alert(
                    levels,
                    levels_list,
                    active_h1_candidate["candle"],
                    active_h1_candidate.get("confirmation_candle", today_df.iloc[-1]),
                    candidate_id=candidate_id,
                    reason=active_h1_candidate.get("reason"),
                    learning_only=bool(active_h1_candidate.get("learning_only")),
                )
                self.emitted_h1_mid_earnings_bounce_alerts.add(h1_event_key)
                self.bounce_candidates.pop(symbol, None)
                return

        # STEP 1: First check if we have an existing bounce candidate to confirm
        if symbol in self.bounce_candidates:
            bounce_data = self.bounce_candidates[symbol]
            bounce_candle = bounce_data["bounce_candle"]

            # Convert time strings to datetime objects for comparison
            try:
                bounce_time = pd.to_datetime(bounce_candle["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
                current_time = pd.to_datetime(today_df.iloc[-1]["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")

                # Check if this is a new candle (different from the bounce candidate candle)
                if current_time > bounce_time:
                    logging.debug(f"{symbol}: Checking confirmation - Bounce time: {bounce_time}, Current time: {current_time}")
                    current_candle = today_df.iloc[-1]
                    bounce_data["candles_waited"] += 1
                    candles_waited = bounce_data["candles_waited"]

                    if direction == "long" and current_candle["close"] < bounce_candle["low"]:
                        logging.debug(
                            f"{symbol}: Invalidating long bounce candidate - close {current_candle['close']:.2f} below bounce low {bounce_candle['low']:.2f}"
                        )
                        levels_list = bounce_data.get("triggered_levels") or list(bounce_data.get("levels", {}).keys())
                        self.log_symbol(
                            symbol,
                            f"{symbol}: Bounce candidate invalidated (long) from {levels_list} - "
                            f"close {current_candle['close']:.2f} fell below bounce low {bounce_candle['low']:.2f}"
                        )
                        self._log_bounce_candidate_event(
                            "invalidated",
                            symbol,
                            direction,
                            bounce_data.get("levels", {}),
                            bounce_candle,
                            current_candle,
                            threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
                            reason="Long candidate closed below bounce candle low.",
                            candidate_id=bounce_data.get("candidate_id", ""),
                            candles_waited=candles_waited,
                        )
                        self.bounce_candidates.pop(symbol)
                        return
                    if direction == "short" and current_candle["close"] > bounce_candle["high"]:
                        logging.debug(
                            f"{symbol}: Invalidating short bounce candidate - close {current_candle['close']:.2f} above bounce high {bounce_candle['high']:.2f}"
                        )
                        levels_list = bounce_data.get("triggered_levels") or list(bounce_data.get("levels", {}).keys())
                        self.log_symbol(
                            symbol,
                            f"{symbol}: Bounce candidate invalidated (short) from {levels_list} - "
                            f"close {current_candle['close']:.2f} rose above bounce high {bounce_candle['high']:.2f}"
                        )
                        self._log_bounce_candidate_event(
                            "invalidated",
                            symbol,
                            direction,
                            bounce_data.get("levels", {}),
                            bounce_candle,
                            current_candle,
                            threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
                            reason="Short candidate closed above bounce candle high.",
                            candidate_id=bounce_data.get("candidate_id", ""),
                            candles_waited=candles_waited,
                        )
                        self.bounce_candidates.pop(symbol)
                        return

                    # Confirm only after a later candle closes through the bounce candle extreme.
                    if direction == "long" and current_candle["close"] > bounce_candle["high"]:
                        levels = bounce_data["levels"]
                        levels_list = bounce_data.get("triggered_levels") or list(levels.keys())
                        confirm_bounce_alert(
                            levels,
                            levels_list,
                            bounce_candle,
                            current_candle,
                            candidate_id=bounce_data.get("candidate_id", ""),
                            candles_waited=candles_waited,
                            learning_only=bool(bounce_data.get("learning_only")),
                        )
                        self.bounce_candidates.pop(symbol)
                        return  # Exit after confirming a bounce

                    elif direction == "short" and current_candle["close"] < bounce_candle["low"]:
                        levels = bounce_data["levels"]
                        levels_list = bounce_data.get("triggered_levels") or list(levels.keys())
                        confirm_bounce_alert(
                            levels,
                            levels_list,
                            bounce_candle,
                            current_candle,
                            candidate_id=bounce_data.get("candidate_id", ""),
                            candles_waited=candles_waited,
                            learning_only=bool(bounce_data.get("learning_only")),
                        )
                        self.bounce_candidates.pop(symbol)
                        return  # Exit after confirming a bounce
                    else:
                        max_confirmation_candles = int(
                            bounce_data.get("max_confirmation_candles", BOUNCE_CONFIRMATION_MAX_CANDLES)
                            or BOUNCE_CONFIRMATION_MAX_CANDLES
                        )
                        if candles_waited >= max_confirmation_candles:
                            levels_list = bounce_data.get("triggered_levels") or list(bounce_data.get("levels", {}).keys())
                            logging.debug(
                                f"{symbol}: Removing bounce candidate after {candles_waited} candles without confirmation"
                            )
                            self.log_symbol(
                                symbol,
                                f"{symbol}: Bounce candidate expired after {candles_waited} candles without confirmation "
                                f"from {levels_list}"
                            )
                            self._log_bounce_candidate_event(
                                "expired",
                                symbol,
                                direction,
                                bounce_data.get("levels", {}),
                                bounce_candle,
                                current_candle,
                                threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
                                reason=f"No confirmation within {max_confirmation_candles} candles.",
                                candidate_id=bounce_data.get("candidate_id", ""),
                                candles_waited=candles_waited,
                            )
                            self.bounce_candidates.pop(symbol)
                            return
                        # Remove stale candidates after a certain time period (e.g., 4 hours)
                        detection_time = bounce_data["detection_time"]
                        if (datetime.now() - detection_time).total_seconds() > 14400:  # 4 hours in seconds
                            levels_list = bounce_data.get("triggered_levels") or list(bounce_data.get("levels", {}).keys())
                            logging.debug(f"{symbol}: Removing stale bounce candidate detected at {detection_time}")
                            self.log_symbol(
                                symbol,
                                f"{symbol}: Bounce candidate removed as stale from {levels_list} "
                                f"(detected at {detection_time:%H:%M:%S})"
                            )
                            self._log_bounce_candidate_event(
                                "stale",
                                symbol,
                                direction,
                                bounce_data.get("levels", {}),
                                bounce_candle,
                                current_candle,
                                threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
                                reason="Candidate exceeded stale timeout.",
                                candidate_id=bounce_data.get("candidate_id", ""),
                                candles_waited=candles_waited,
                            )
                            self.bounce_candidates.pop(symbol)
                else:
                    logging.debug(f"{symbol}: Current candle is the same as bounce candle, waiting for next candle")
            except Exception as e:
                logging.error(f"{symbol}: Error during bounce confirmation: {e}")
                self.bounce_candidates.pop(symbol, None)  # Remove problematic candidate

        # STEP 2: Check if current candle is a new bounce candidate - FIXED INDENTATION
        if candidate_info and symbol not in self.bounce_candidates:
            candidate_id = self._make_bounce_event_id(
                symbol,
                direction,
                candidate_info["candle"],
                candidate_info["levels"],
            )
            if candidate_info.get("confirm_immediately"):
                levels = candidate_info["levels"]
                levels_list = candidate_info.get("triggered_levels") or list(levels.keys())
                h1_event_key = candidate_info.get("h1_event_key")
                if h1_event_key and h1_event_key in self.emitted_h1_mid_earnings_bounce_alerts:
                    return
                confirmation_candle = candidate_info.get("confirmation_candle", today_df.iloc[-1])
                logging.debug(
                    f"{symbol}: Fast-confirming bounce on signal candle from {levels_list}"
                )
                confirm_bounce_alert(
                    levels,
                    levels_list,
                    candidate_info["candle"],
                    confirmation_candle,
                    candidate_id=candidate_id,
                    reason=candidate_info.get("reason") or "Fast-confirmed on signal candle.",
                    learning_only=bool(candidate_info.get("learning_only")),
                )
                if h1_event_key:
                    self.emitted_h1_mid_earnings_bounce_alerts.add(h1_event_key)
                return

            self.bounce_candidates[symbol] = {
                "candidate_id": candidate_id,
                "levels": candidate_info["levels"],
                "triggered_levels": candidate_info.get("triggered_levels", []),
                "bounce_candle": candidate_info["candle"],
                "detection_time": datetime.now(),
                "candles_waited": 0,
                "max_confirmation_candles": candidate_info.get(
                    "max_confirmation_candles",
                    BOUNCE_CONFIRMATION_MAX_CANDLES,
                ),
                "learning_only": bool(candidate_info.get("learning_only")),
            }
            self._log_bounce_candidate_event(
                "detected",
                symbol,
                direction,
                candidate_info["levels"],
                candidate_info["candle"],
                candidate_info["candle"],
                threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0),
                reason="Level reaction detected; waiting for confirmation candle.",
                candidate_id=candidate_id,
                candles_waited=0,
            )

                        # In the evaluate_bounce_candidate function, where price approaching is logged:
            if LOG_PRICE_APPROACHING and not candidate_info.get("learning_only"):
                # Filter out 10-candle levels for approaching alerts
                display_level_names = set(candidate_info.get("triggered_levels", candidate_info["levels"].keys()))
                approaching_levels = {k: v for k, v in candidate_info["levels"].items()
                                    if k in display_level_names
                                    if "10_candle" not in k}

                # Only show approaching alert if there are non-10-candle levels
                if approaching_levels:
                    level_details = ", ".join(f"{lvl}: {val:.2f}" for lvl, val in approaching_levels.items())
                    approaching_msg = f"{symbol}: Price approaching levels ({direction}) - {level_details}"

                    # Use self.log_symbol to ensure proper symbol coloring
                    self.log_symbol(symbol, approaching_msg, "approaching")

                    if self.gui_callback:
                        direction_tag = "approaching_green" if direction == "long" else "approaching_red"
                        self.gui_callback(approaching_msg, direction_tag)
        elif not candidate_info and symbol not in self.bounce_candidates:
            try:
                self._maybe_log_near_miss_bounce(
                    symbol,
                    direction,
                    today_df.iloc[-1].to_dict(),
                )
            except Exception as exc:
                logging.debug(f"{symbol}: near-miss logging failed: {exc}")


    def check_removal_conditions(self):
        if not self.has_minimum_candles_completed(required=VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES):
            logging.info(
                f"Skipping removal conditions until {VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES} completed 5-minute candles."
            )
            return
        for watchlist_path, direction in [(LONGS_FILENAME, "long"), (SHORTS_FILENAME, "short")]:
            tickers = read_tickers(watchlist_path)
            for symbol in tickers:
                if symbol not in self.symbol_metrics:
                    continue

                metrics = self.symbol_metrics[symbol]
                invalidation = metrics.get("vwap_invalidation") or {}
                consecutive_bad_closes = int(invalidation.get("consecutive_bad_closes") or 0)
                if consecutive_bad_closes < VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES:
                    continue

                last_row = invalidation.get("last_row") or {}
                close_value = last_row.get("close")
                vwap_value = last_row.get("vwap")
                dynamic_vwap = last_row.get("dynamic_vwap")
                eod_vwap = last_row.get("eod_vwap")
                level_side = "below" if direction == "long" else "above"

                self.remove_from_watchlist(symbol, direction)
                removal_msg = (
                    f"{symbol} removed from {direction}s watchlist - "
                    f"{consecutive_bad_closes} completed M5 closes {level_side} VWAP/Dynamic VWAP/EOD VWAP "
                    f"(close={close_value:.2f}, vwap={vwap_value:.2f}, dvwap={dynamic_vwap:.2f}, eod={eod_vwap:.2f})"
                )
                logging.info(removal_msg)
                if self.gui_callback:
                    self.gui_callback(removal_msg, "blue")

    def _maybe_refresh_auto_populated_watchlists(self):
        """Refresh the auto-owned watchlist slice every ~30 min while scanning.

        The heavy work (yfinance bulk + parquet reads) runs on a one-shot
        daemon thread so the scan cycle never stalls behind it. The legacy
        SPY pause detector is passed through for side-symmetric pullback
        discovery; the shadow market-state engine remains advisory only.
        """
        from autopilot_core import (
            AGGRESSIVE_SPY_PULLBACK_MIN_MOVE_PCT,
            AUTO_POPULATE_REFRESH_MINUTES,
            minutes_since_open,
        )

        now = time.time()
        try:
            since_open = minutes_since_open(datetime.now())
        except Exception:
            since_open = None
        # Wait for a readable tape (30m in) and stop after the close.
        if since_open is None or since_open < 30 or since_open > 390:
            return
        env = self.get_market_environment()
        spy_pullback_active = False
        pullback_key = ""
        if str(env).startswith(("bullish", "bearish")):
            try:
                spy_today, _prev_close = self._spy_session_bars(cached_only=True)
                trend_side = "short" if str(env).startswith("bearish") else "long"
                pause_start = (
                    self._detect_spy_pause_start(spy_today, trend_side) if spy_today else None
                )
                spy_change_pct, _window = (
                    self._window_change_pct(spy_today, pause_start)
                    if pause_start is not None
                    else (None, [])
                )
                minimum_move = float(AGGRESSIVE_SPY_PULLBACK_MIN_MOVE_PCT)
                spy_pullback_active = bool(
                    spy_change_pct is not None
                    and (
                        (trend_side == "short" and spy_change_pct >= minimum_move)
                        or (trend_side == "long" and spy_change_pct <= -minimum_move)
                    )
                )
                if spy_pullback_active:
                    pullback_key = trend_side
            except Exception:
                logging.debug("Auto-populate legacy SPY pause read failed.", exc_info=True)
        if getattr(self, "_auto_populate_running", False):
            return
        prior_pullback_key = str(getattr(self, "_auto_populate_spy_pullback_key", "") or "")
        new_pullback = bool(pullback_key and pullback_key != prior_pullback_key)
        self._auto_populate_spy_pullback_key = pullback_key
        if (
            not new_pullback
            and now - float(getattr(self, "_auto_populate_last_ts", 0.0))
            < AUTO_POPULATE_REFRESH_MINUTES * 60.0
        ):
            return
        self._auto_populate_last_ts = now
        self._auto_populate_running = True

        def worker():
            try:
                from autopilot_core import refresh_auto_populated_watchlists

                summary = refresh_auto_populated_watchlists(
                    env,
                    spy_pullback_active=spy_pullback_active,
                    preserve_existing_auto=new_pullback,
                    log=logging.info,
                )
                if summary and self.gui_callback:
                    long_info = summary.get("long", {})
                    short_info = summary.get("short", {})
                    self.gui_callback(
                        f"AUTO WATCHLIST ({env}): longs {long_info.get('total_auto', 0)} auto "
                        f"(+{len(long_info.get('added', []))}/-{len(long_info.get('rotated_out', []))}), "
                        f"shorts {short_info.get('total_auto', 0)} auto "
                        f"(+{len(short_info.get('added', []))}/-{len(short_info.get('rotated_out', []))}) "
                        f"from {summary.get('scanned', 0)} universe names.",
                        "blue",
                    )
            except Exception:
                logging.exception("Auto-populate worker failed.")
            finally:
                self._auto_populate_running = False

        threading.Thread(target=worker, name="auto-watchlist-populate", daemon=True).start()

    def remove_from_watchlist(self, symbol, direction):
        filename = LONGS_FILENAME if direction == "long" else SHORTS_FILENAME
        try:
            from autopilot_core import record_auto_watchlist_cut

            # Day-scoped blacklist so the auto-populator can't re-add a name
            # the triple-VWAP rule just cut.
            record_auto_watchlist_cut(symbol, direction)
        except Exception:
            pass
        try:
            with open(filename, 'r') as f:
                symbols = f.read().splitlines()
            symbols = [s for s in symbols if s.strip() != symbol]
            with open(filename, 'w') as f:
                f.write('\n'.join(symbols))
            if direction == "long":
                self.longs = symbols
            else:
                self.shorts = symbols
            self.bounce_candidates.pop(symbol, None)
            self.symbol_metrics.pop(symbol, None)
            self.latest_bars.pop(symbol, None)
            stale_bar_keys = [key for key in self.latest_bars if str(key).startswith(f"{symbol}|")]
            for key in stale_bar_keys:
                self.latest_bars.pop(key, None)
            logging.info(f"{symbol} removed from {filename} due to removal condition.")
        except Exception as e:
            logging.error(f"Error removing {symbol} from {filename}: {e}")

    def stop(self, timeout: float = 10.0) -> None:
        """Cooperative shutdown: end the strategy loop, disconnect IB, and
        join owned threads (bounded). Safe to call repeatedly."""
        self._stop_event.set()
        try:
            self.disconnect()
        except Exception:
            pass
        for thread in (self.strategy_thread, self.api_thread):
            if (
                thread is not None
                and thread.is_alive()
                and thread is not threading.current_thread()
            ):
                thread.join(timeout=timeout)

    def is_stopping(self) -> bool:
        return self._stop_event.is_set()

    def _scan_human_focus_fast_lane(self, enabled_bounce_types):
        """Refresh trader-picked M5 names before the broad RRS/watchlist pass.

        A large longs/shorts universe can take many minutes to hydrate from IB.
        Focus picks are the explicit low-latency lane: fetch/detect them first,
        then reuse those bars in the broad scan.  Returning the processed set
        prevents a second historical request later in the same cycle.
        """
        focus_symbols = sorted(self._human_focus_symbols() & self.get_scan_symbol_set())
        processed = set()
        if not focus_symbols:
            return processed
        logging.info(
            "M5 Focus fast lane: scanning %s trader-picked symbol(s) before the broad sweep.",
            len(focus_symbols),
        )
        for symbol in focus_symbols:
            if not self.is_scanning_enabled():
                break
            if self.atr_cache.get(symbol) is None:
                continue
            self.request_and_detect_bounce(symbol, allowed_bounce_types=enabled_bounce_types)
            processed.add(symbol)
        # These two M5 pattern families consume the bars just fetched above.
        self.check_orb_break_setups(symbols=processed)
        self.check_ema8_grind_setups(symbols=processed)
        return processed

    def run_strategy(self):
        last_warning_reset = datetime.now().date()

        while not self._stop_event.is_set():
            try:
                # Bookkeeping that needs no IB and no scanning: keep the
                # learning state fresh even on paused/disconnected evenings.
                try:
                    self._maybe_refresh_learning_after_close()
                except Exception:
                    logging.exception("After-close learning refresh scheduling failed.")

                if not self.is_scanning_enabled():
                    # The regime read stays live even while paused (user ask
                    # 2026-07-08: always consider regime changes).
                    try:
                        self._maybe_refresh_auto_regime_while_paused()
                    except Exception:
                        logging.exception("Paused-mode auto regime refresh failed.")
                    self._stop_event.wait(0.5)
                    continue

                if not self.ensure_connected():
                    logging.warning("IB not connected; retrying in 5 seconds...")
                    self._stop_event.wait(5)
                    continue
                # Reset warning cache daily
                current_date = datetime.now().date()
                if current_date != last_warning_reset:
                    self.warned_symbols.clear()
                    self.emitted_master_avwap_events.clear()
                    self.emitted_master_avwap_focus_alerts.clear()
                    self.emitted_master_avwap_second_stdev_alerts.clear()
                    self.emitted_master_avwap_d1_flags.clear()
                    self.master_avwap_d1_flags_primed_date = None
                    self.emitted_h1_mid_earnings_bounce_alerts.clear()
                    self.logged_near_miss_events.clear()
                    last_warning_reset = current_date
                    logging.info("Daily warning cache reset completed")

                self.longs = read_tickers(LONGS_FILENAME)
                self.shorts = read_tickers(SHORTS_FILENAME)
                self.auto_longs = read_tickers(AUTO_LONGS_FILENAME)
                self.auto_shorts = read_tickers(AUTO_SHORTS_FILENAME)
                self.load_master_avwap_focus()
                self.load_human_focus_picks()
                self.load_master_avwap_d1_watchlist()
                self.load_master_avwap_d1_upgrade_alerts()
                self.update_watchlists_from_master_avwap()
                self.emit_master_avwap_d1_flags()
                self.alerted_symbols.clear()
                self.symbol_metrics = {}
                self._scan_cycle_index = int(getattr(self, "_scan_cycle_index", -1)) + 1
                all_symbols = self.get_scan_symbol_set()
                priority_symbols = self.get_priority_scan_symbols() & all_symbols
                background_symbols = all_symbols - priority_symbols
                refresh_background = self._is_background_refresh_cycle()
                self._prune_latest_bars_for_cycle(refresh_background, background_symbols)
                self.build_atr_cache()

                # Auto-track the intraday regime (SPY vs yesterday's close)
                # before anything downstream reads the environment.
                try:
                    self.update_auto_market_environment()
                except Exception:
                    logging.exception("Auto market regime update failed.")

                enabled_bounce_types = {
                    bounce_type for bounce_type, enabled in self.bounce_type_toggles.items() if enabled
                }
                try:
                    focus_fast_lane_symbols = self._scan_human_focus_fast_lane(enabled_bounce_types)
                except Exception:
                    focus_fast_lane_symbols = set()
                    logging.exception("M5 Focus fast-lane scan failed.")

                # Log strongest/weakest names for key intraday timeframes each cycle.
                for timeframe_key in ("5m", "15m", "1h"):
                    self.run_rrs_scan(timeframe_key_override=timeframe_key, emit_gui=False)
                # Keep the GUI view synced with user-selected RRS timeframe.
                self.run_rrs_scan()

                # Regime-pause bangers: SPY paused against the tape -> flag the
                # longs/shorts.txt names that refuse to participate.
                try:
                    self.check_regime_pause_setups()
                except Exception:
                    logging.exception("Regime pause sweep failed.")

                # Entry assist in auto mode: open/close pullback windows in
                # strong regimes, 30m movers cadence in weak/chop regimes.
                try:
                    self.entry_assist_auto_tick()
                except Exception:
                    logging.exception("Entry assist auto tick failed.")

                # Universe auto-populate: keep longs/shorts.txt stocked with
                # PDH/PDL breakers moving > 0.5 ADR (regime-capped top-N).
                try:
                    self._maybe_refresh_auto_populated_watchlists()
                except Exception:
                    logging.exception("Auto-populate watchlist refresh failed.")

                # Trader-favorite day-trade sweeps on longs/shorts.txt: delayed
                # 5m opening-range breaks and 8-EMA grind squeezes into HOD/LOD.
                try:
                    self.check_orb_break_setups()
                except Exception:
                    logging.exception("ORB break sweep failed.")
                try:
                    self.check_ema8_grind_setups()
                except Exception:
                    logging.exception("8-EMA grind sweep failed.")

                # H1 candle-color signals under test: green-regime 10-EMA
                # bounces + blue-after-red reclaims (longs), green->yellow
                # breakdowns (shorts) on closed hourly candles.
                try:
                    self.check_h1_color_setups()
                except Exception:
                    logging.exception("H1 color sweep failed.")

                d1_watch_symbols = set(self.get_master_avwap_d1_watch_symbols())
                monitored_symbols = self.get_monitored_extreme_symbols() | d1_watch_symbols
                pending_outcome_symbols = self._pending_bounce_symbols()
                if PRIORITY_WATCHLIST_EMPHASIS and not refresh_background:
                    # Off-cycle: only priority names and open-outcome symbols get
                    # fresh bounce-detection requests; background names wait for
                    # their refresh cycle (their cached bars fed the RRS scan).
                    deferred = len(background_symbols - pending_outcome_symbols)
                    logging.info(
                        "Priority emphasis: scanning %s priority symbol(s); deferring %s background "
                        "symbol(s) until refresh cycle (every %s candles).",
                        len(priority_symbols),
                        deferred,
                        BACKGROUND_SYMBOL_REFRESH_EVERY_CYCLES,
                    )
                    scannable_symbols = priority_symbols | (background_symbols & pending_outcome_symbols)
                else:
                    scannable_symbols = set(all_symbols)
                logging.info(f"Monitoring {len(monitored_symbols)} strongest/weakest symbols for EMA bounces.")
                processed_symbols = set(focus_fast_lane_symbols)
                outcome_update_symbols = set(focus_fast_lane_symbols)
                non_ema_extreme_bounce_types = enabled_bounce_types - {"ema_8", "ema_15"}

                # 1) Prioritize strongest/weakest names first.
                for sym in sorted(monitored_symbols):
                    if not self.is_scanning_enabled():
                        break
                    if sym not in scannable_symbols or self.atr_cache.get(sym) is None:
                        continue
                    self.request_and_detect_bounce(sym, allowed_bounce_types=enabled_bounce_types)
                    processed_symbols.add(sym)
                    outcome_update_symbols.add(sym)

                # 2) Then scan all remaining symbols for non-EMA-8/15 bounce types.
                for sym in sorted(scannable_symbols - processed_symbols):
                    if not self.is_scanning_enabled():
                        break
                    if self.atr_cache.get(sym) is None:
                        continue
                    self.request_and_detect_bounce(sym, allowed_bounce_types=non_ema_extreme_bounce_types)
                    outcome_update_symbols.add(sym)

                # Keep EOD outcome tracking alive even if a confirmed bounce was
                # removed from the watchlist or skipped by live-scan gates before
                # the session close.
                for sym in sorted(pending_outcome_symbols - outcome_update_symbols):
                    if not self.is_scanning_enabled():
                        break
                    self.request_and_detect_bounce(
                        sym,
                        allowed_bounce_types=set(),
                        scan_for_new_bounces=False,
                    )

                if not self.is_scanning_enabled():
                    continue

                self.check_removal_conditions()
                wait_for_candle_close(self._stop_event)
                if self._stop_event.is_set():
                    break
                if self.gui_callback:
                    self.gui_callback("Candle has closed", "candle_line")
            except Exception as e:
                logging.exception(f"Error in strategy loop: {e}")
                self._stop_event.wait(5)


    def check_dynamic_vwap_touches(self):
        results = []
        all_symbols = self.get_scan_symbol_set()
        for symbol in all_symbols:
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(symbol)
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                df = pd.DataFrame(bars)
                if not df.empty:
                    dvwap = self.calculate_dynamic_vwap(df)
                    if dvwap is not None:
                        touched = any(row['low'] <= dvwap <= row['high'] for _, row in df.iterrows())
                        if touched:
                            results.append(f"{symbol} touched DVWAP")
            del self.data_ready_events[reqId]
        return results

    def check_dynamic_vwap2_touches(self):
        results = []
        all_symbols = self.get_scan_symbol_set()
        for symbol in all_symbols:
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(symbol)
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="3 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                df = pd.DataFrame(bars)
                if not df.empty:
                    dvwap2 = self.calculate_dynamic_vwap2(df)
                    if dvwap2 is not None:
                        df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
                        today_data = df[df["datetime"].dt.date == df["datetime"].iloc[-1].date()]
                        touched = any(row['low'] <= dvwap2 <= row['high'] for _, row in today_data.iterrows())
                        if touched:
                            results.append(f"{symbol} touched DVWAP2")
            del self.data_ready_events[reqId]
        return results

    def check_eod_vwap_touches(self):
        results = []
        all_symbols = self.get_scan_symbol_set()
        for symbol in all_symbols:
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(symbol)
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                df = pd.DataFrame(bars)
                if not df.empty:
                    eod_vwap = self.calculate_eod_vwap(df)
                    if eod_vwap is not None:
                        df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
                        today_data = df[df["datetime"].dt.date == df["datetime"].iloc[-1].date()]
                        touched = any(row['low'] <= eod_vwap <= row['high'] for _, row in today_data.iterrows())
                        if touched:
                            results.append(f"{symbol} touched EOD VWAP")
            del self.data_ready_events[reqId]
        return results

    def log_bounce_to_file(self, symbol, direction, levels, bounce_candle, current_candle, threshold, quality=None):
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            bounce_types_list = list(levels.keys())
            bounce_types_str = ", ".join(bounce_types_list)
            quality = quality if isinstance(quality, dict) else {}
            tier = str(quality.get("tier") or "").strip().upper()
            composite = quality.get("composite_r")
            tier_str = f"{tier}-TIER" if tier else ""
            if tier and isinstance(composite, (int, float)):
                tier_str = f"{tier_str} ({composite:+.2f}R)"

            candle_time = str(current_candle.get("time", "")).strip() if current_candle is not None else ""
            trade_dt = None
            if candle_time:
                for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S"):
                    try:
                        trade_dt = datetime.strptime(candle_time, fmt)
                        break
                    except ValueError:
                        continue
            if trade_dt is None:
                trade_dt = datetime.now()
            trade_date_str = trade_dt.strftime("%Y-%m-%d")

            BOUNCE_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
            with open(BOUNCE_LOG_FILENAME, "a") as f:
                f.write(f"{timestamp} | {symbol} | {bounce_types_str} | {direction} | {tier_str}\n")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            fieldnames = ["time_local", "trade_date", "symbol", "direction", "bounce_types", "tier", "composite_r"]
            _migrate_csv_header(INTRADAY_BOUNCES_CSV, fieldnames)
            file_exists = INTRADAY_BOUNCES_CSV.exists()
            with INTRADAY_BOUNCES_CSV.open("a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "time_local": timestamp,
                        "trade_date": trade_date_str,
                        "symbol": symbol,
                        "direction": direction,
                        "bounce_types": ", ".join(bounce_types_list),
                        "tier": tier,
                        "composite_r": composite if isinstance(composite, (int, float)) else "",
                    }
                )

            logging.info(
                f"Simplified bounce details for {symbol} logged to {BOUNCE_LOG_FILENAME}"
            )
        except Exception as e:
            logging.error(f"Error logging bounce to file: {e}")



##########################################
# Run Bot with GUI Integration
##########################################
def run_bot_with_gui(gui_callback, start_scanning_enabled=False):
    configure_app_logging()

    bot = BounceBot(
        gui_callback=gui_callback,
        start_scanning_enabled=start_scanning_enabled,
    )
    bot.set_connection_info("127.0.0.1", 7496, 125)
    try:
        bot.connect("127.0.0.1", 7496, clientId=bot.ib_client_id)
    except Exception as exc:
        logging.warning(f"Initial IB connection attempt failed: {exc}")
    api_thread = threading.Thread(target=bot.run, daemon=True)
    bot.api_thread = api_thread
    api_thread.start()
    logging.info("Starting strategy loop (connection will auto-retry until IB is available)...")
    strategy_thread = threading.Thread(target=bot.run_strategy, daemon=True)
    bot.strategy_thread = strategy_thread
    strategy_thread.start()
    return bot

##########################################
# GUI Code using Tkinter
##########################################
# Find and replace the light_grey variable definition with dark theme colors
# Around line 677 in the start_gui() function

from .gui import (
    append_alert_message,
    build_environment_focus_copy_text,
    choose_gui_mode,
    configure_alert_tags,
    copy_text_to_clipboard,
    create_rrs_confirmed_panel,
    prompt_change_home_folder,
    start_gui,
    start_lightweight_gui,
)


##########################################
# Main
##########################################

def main():
    print("Starting script...")
    if "--analyze_bounce_performance" not in sys.argv:
        reset_log_files()  # Use the new function instead of reset_log_file()
        print("Runtime files prepared.")
    else:
        print("Analysis-only mode; live runtime files left untouched.")

    # Rest of the code remains the same...


    try:
        import argparse
        import sys
        print("Imports successful.")

        parser = argparse.ArgumentParser()
        parser.add_argument("--use_gui", action="store_true", help="Use the Tkinter GUI")
        parser.add_argument(
            "--gui_mode",
            choices=("prompt", "full", "lightweight"),
            default="prompt",
            help="Choose GUI startup mode when launching with the GUI.",
        )
        parser.add_argument(
            "--analyze_bounce_performance",
            action="store_true",
            help="Refresh intraday bounce performance CSV/report and exit without connecting to IB.",
        )
        parser.add_argument(
            "--bounce_perf_min_samples",
            type=int,
            default=BOUNCE_PERFORMANCE_MIN_SAMPLES,
            help="Minimum sample count used for intraday bounce focus/avoid recommendations.",
        )
        print("Parser created.")

        args = parser.parse_args()
        print(f"Arguments parsed: {args}")

        if args.analyze_bounce_performance:
            performance_path, report_path, row_count = refresh_intraday_bounce_performance_report(
                min_samples=args.bounce_perf_min_samples,
            )
            print(f"Intraday bounce performance rows: {row_count}")
            print(f"CSV: {performance_path}")
            print(f"Report: {report_path}")
            sys.exit(0)

        # Determine whether to use GUI based on args or global setting
        use_gui = args.use_gui if args.use_gui else USE_GUI
        print(f"Using GUI: {use_gui}")

        if use_gui:
            print("Initializing GUI mode...")
            start_gui(mode=args.gui_mode)
        else:
            print("Initializing console mode...")
            configure_app_logging()
            print("Logging configured.")
            print("Filters applied.")

            # Initialize and run the bot without GUI
            print("Creating BounceBot instance...")
            bot = BounceBot()
            bot.set_connection_info("127.0.0.1", 7496, 125)
            print("Connecting to IB API...")
            bot.connect("127.0.0.1", 7496, clientId=bot.ib_client_id)

            # Wait for connection
            print("Waiting for connection...")
            connection_timeout = 30  # seconds
            connection_start = time.time()
            while not bot.connection_status and time.time() - connection_start < connection_timeout:
                time.sleep(1)
                print(".", end="", flush=True)

            if not bot.connection_status:
                print("\nConnection failed!")
                logging.error("Failed to connect to IB API within timeout period.")
                sys.exit(1)

            print("\nConnection successful!")
            logging.info("BounceBot is connected. Starting strategy loop...")

            # Run the API in a separate thread
            print("Starting API thread...")
            api_thread = threading.Thread(target=bot.run, daemon=True)
            api_thread.start()

            # Run the strategy
            print("Starting strategy loop...")
            bot.run_strategy()  # This will run in the main thread
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
