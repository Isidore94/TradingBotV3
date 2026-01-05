# -*- coding: utf-8 -*-

import os
import sys
import csv
from pathlib import Path
# Suppress Tk deprecation warnings on macOS
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import math
import time
import threading
import logging
from datetime import datetime, timedelta
import zoneinfo
import queue
import tkinter as tk
from tkinter import scrolledtext
import tkinter.font as tkFont

import pandas as pd

# IB API imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# For colored console output (still used for logging)
from colorama import init, Fore, Style
init(autoreset=True)

##########################################
# Adjustable Parameters
##########################################
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LONGS_FILENAME = ROOT_DIR / "longs.txt"
SHORTS_FILENAME = ROOT_DIR / "shorts.txt"
BOUNCE_LOG_FILENAME = LOG_DIR / "bouncers.txt"
INTRADAY_BOUNCES_CSV = DATA_DIR / "intraday_bounces.csv"
ATR_PERIOD = 20
THRESHOLD_MULTIPLIER = 0.02
CONSECUTIVE_CANDLES = 6  # Number of candles price must respect level before bounce
CHECK_CONSECUTIVE_CANDLES = True  # Parameter to enable/disable this check
CHECK_BOUNCE_VVWAP = True
CHECK_BOUNCE_DYNAMIC_VVWAP = True
CHECK_BOUNCE_EOD_VWAP = True 
CHECK_BOUNCE_10_CANDLE = False
CHECK_BOUNCE_PREV_DAY_HIGH = True
CHECK_BOUNCE_PREV_DAY_LOW = True
CHECK_BOUNCE_VWAP_UPPER_BAND = False 
CHECK_BOUNCE_VWAP_LOWER_BAND = False  
CHECK_BOUNCE_DYNAMIC_VWAP_UPPER_BAND = False 
CHECK_BOUNCE_DYNAMIC_VWAP_LOWER_BAND = False  
CHECK_BOUNCE_EOD_VWAP_UPPER_BAND = False
CHECK_BOUNCE_EOD_VWAP_LOWER_BAND = False 
LOGGING_MODE = True
SCAN_OUTSIDE_MARKET_HOURS = True
LOG_PRICE_APPROACHING = True
USE_GUI = True  # New parameter to toggle GUI on/off

# Connection & Request settings
MAX_CONCURRENT_REQUESTS = 4
REQUEST_DELAY = 0.1  # seconds between IB historical data requests

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
def wait_for_candle_close():
    now = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    sec_since_5 = (now.minute % 5) * 60 + now.second
    sec_to_go = 300 - sec_since_5
    logging.info(f"Waiting for candle to close: {sec_to_go} seconds remaining.")
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

def reset_log_files():
    files_to_reset = ["trading_bot.log", BOUNCE_LOG_FILENAME]

    for log_file_path in files_to_reset:
        try:
            # Check if file exists before attempting to remove it
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                print(f"Previous log file deleted: {log_file_path}")
            
            # Create an empty file
            with open(log_file_path, 'w') as f:
                pass
            
            print(f"Created fresh log file: {log_file_path}")
        except Exception as e:
            print(f"Error resetting log file {log_file_path}: {e}")


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
    def __init__(self, gui_callback=None):
        EClient.__init__(self, self)
        self.connection_status = False
        self.reqIdCount = 1000

        self.data = {}
        self.data_ready_events = {}

        self.longs = read_tickers(LONGS_FILENAME)
        self.shorts = read_tickers(SHORTS_FILENAME)
        self.atr_cache = {}
        self.symbol_metrics = {}  # Store precomputed VWAP and level metrics

        self.alerted_symbols = set()
        self.bounce_candidates = {}  # Track candidate bounces
        self.request_queue = RequestQueue()
        self.gui_callback = gui_callback  # Callback to update the GUI
        
        # Add this to track which symbols we've already warned about
        self.warned_symbols = set()


    def getReqId(self):
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

    def nextValidId(self, orderId):
        self.connection_status = True
        logging.info(f"Connected to IB API. NextValidId={orderId}")

    def error(self, reqId, errorCode, errorString):
        logging.error(f"IB Error. ReqId={reqId}, Code={errorCode}, Msg={errorString}")

    def historicalData(self, reqId, bar):
        with threading.Lock():
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
        if df.empty or ('volume' not in df) or (df['volume'].sum() == 0):
            return None, None, None
        
        df = df.copy()
        
        # TC2000 style typical price calculation
        df["typical"] = (df["high"] + df["low"] + df["open"] + df["close"]) / 4
        
        # Calculate cumulative values
        df["vol_times_price"] = df["typical"] * df["volume"]
        cum_vol = df["volume"].cumsum()
        cum_vol_price = df["vol_times_price"].cumsum()
        
        # Calculate VWAP for each point
        df["vwap"] = cum_vol_price / cum_vol
        
        # TradingView-style standard deviation calculation
        # Calculate squared deviation from VWAP
        df["sq_dev"] = ((df["typical"] - df["vwap"]) ** 2) * df["volume"]
        
        # Calculate standard deviation at each point (TradingView style)
        cum_sq_dev = df["sq_dev"].cumsum()
        df["stdev"] = (cum_sq_dev / cum_vol).apply(lambda x: math.sqrt(x) if x > 0 else 0)
        
        # Get final values
        vwap = df["vwap"].iloc[-1]
        stdev = df["stdev"].iloc[-1]
        
        # Calculate bands with specified multiplier
        upper_band = vwap + stdev * band_mult
        lower_band = vwap - stdev * band_mult
        
        if LOGGING_MODE:
            logging.debug(f"Standard VWAP calculation: VWAP={vwap:.4f}, StdDev={stdev:.4f}")
            logging.debug(f"Standard VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}")
        
        return vwap, upper_band, lower_band




    def calculate_standard_vwap(self, df):
        if df.empty:
            return None
            
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Convert time to datetime properly
        try:
            df_copy["datetime"] = pd.to_datetime(df_copy["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            # Check if conversion was successful
            if df_copy["datetime"].isna().all():
                logging.error("Failed to convert time to datetime for standard VWAP")
                return None
        except Exception as e:
            logging.error(f"Error in datetime conversion for standard VWAP: {e}")
            return None
        
        # Filter for today's data only
        current_date = df_copy["datetime"].iloc[-1].date()
        today_df = df_copy[df_copy["datetime"].dt.date == current_date]
        
        if today_df.empty:
            logging.warning("No today's data available for standard VWAP calculation")
            return None
        
        # Debug log the date range
        logging.debug(f"Standard VWAP using data from {current_date} only")
        logging.debug(f"Today's data count: {len(today_df)}")
        
        # TC2000 style typical price calculation
        today_df["typical_price"] = (today_df["high"] + today_df["low"] + 
                                    today_df["open"] + today_df["close"]) / 4.0
        
        # Check if we have volume
        if today_df["volume"].sum() == 0:
            logging.warning("No volume data for standard VWAP calculation")
            return None
            
        # Calculate VWAP
        cum_vol_price = (today_df["typical_price"] * today_df["volume"]).sum()
        cum_vol = today_df["volume"].sum()
        standard_vwap = cum_vol_price / cum_vol
        
        return standard_vwap

    def calculate_dynamic_vwap(self, df):
        if df.empty:
            return None
            
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Convert time to datetime properly
        try:
            df_copy["datetime"] = pd.to_datetime(df_copy["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            # Check if conversion was successful
            if df_copy["datetime"].isna().all():
                logging.error("Failed to convert time to datetime for dynamic VWAP")
                return None
        except Exception as e:
            logging.error(f"Error in datetime conversion for dynamic VWAP: {e}")
            return None
        
        # Get today and yesterday dates
        current_date = df_copy["datetime"].iloc[-1].date()
        yesterday_date = current_date - timedelta(days=1)
        
        # Filter for today's and yesterday's data
        today_df = df_copy[df_copy["datetime"].dt.date == current_date]
        yesterday_df = df_copy[df_copy["datetime"].dt.date == yesterday_date]
        
        # Debug log
        logging.debug(f"Dynamic VWAP using data from {yesterday_date} and {current_date}")
        logging.debug(f"Yesterday's data count: {len(yesterday_df)}, Today's data count: {len(today_df)}")
        
        # Skip calculation if no yesterday's data
        if yesterday_df.empty:
            logging.warning("No yesterday's data for dynamic VWAP, falling back to standard VWAP")
            return self.calculate_standard_vwap(df)
        
        # Combine yesterday and today's data
        combined_df = pd.concat([yesterday_df, today_df])
        
        if combined_df.empty or combined_df["volume"].sum() == 0:
            logging.warning("Insufficient data for dynamic VWAP calculation")
            return None
        
        # TC2000 style typical price calculation
        combined_df["typical_price"] = (combined_df["high"] + combined_df["low"] + 
                                    combined_df["open"] + combined_df["close"]) / 4.0
        
        # Calculate VWAP
        cum_vol_price = (combined_df["typical_price"] * combined_df["volume"]).sum()
        cum_vol = combined_df["volume"].sum()
        dynamic_vwap = cum_vol_price / cum_vol
        
        return dynamic_vwap

    def calculate_eod_vwap(self, df):
        if df.empty:
            return None
            
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Convert time to datetime properly
        try:
            df_copy["datetime"] = pd.to_datetime(df_copy["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            # Check if conversion was successful
            if df_copy["datetime"].isna().all():
                logging.error("Failed to convert time to datetime for EOD VWAP")
                return None
        except Exception as e:
            logging.error(f"Error in datetime conversion for EOD VWAP: {e}")
            return None
        
        # Get today and yesterday dates
        current_date = df_copy["datetime"].iloc[-1].date()
        yesterday_date = current_date - timedelta(days=1)
        
        # Filter for today's data
        today_df = df_copy[df_copy["datetime"].dt.date == current_date]
        
        # Get only the last candle from yesterday
        yesterday_df = df_copy[df_copy["datetime"].dt.date == yesterday_date]
        
        # Debug log
        logging.debug(f"EOD VWAP using data from {current_date} and last candle from {yesterday_date}")
        logging.debug(f"Yesterday's data count: {len(yesterday_df)}, Today's data count: {len(today_df)}")
        
        if yesterday_df.empty:
            logging.warning("No yesterday's data for EOD VWAP, falling back to standard VWAP")
            return self.calculate_standard_vwap(df)
        
        # Get last candle from yesterday
        last_candle_yesterday = yesterday_df.iloc[[-1]]
        
        # Combine the last candle from yesterday with today's data
        eod_df = pd.concat([last_candle_yesterday, today_df])
        
        if eod_df.empty or eod_df["volume"].sum() == 0:
            logging.warning("Insufficient data for EOD VWAP calculation")
            return None
        
        # TC2000 style typical price calculation
        eod_df["typical_price"] = (eod_df["high"] + eod_df["low"] + 
                                eod_df["open"] + eod_df["close"]) / 4.0
        
        # Calculate VWAP
        cum_vol_price = (eod_df["typical_price"] * eod_df["volume"]).sum()
        cum_vol = eod_df["volume"].sum()
        eod_vwap = cum_vol_price / cum_vol
        
        return eod_vwap
    
    def calculate_dynamic_vwap_with_stdev_bands(self, df, band_mult=1.0):
        if df.empty:
            return None, None, None
                
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Convert time to datetime properly
        try:
            df_copy["datetime"] = pd.to_datetime(df_copy["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            if df_copy["datetime"].isna().all():
                logging.error("Failed to convert time to datetime for dynamic VWAP bands")
                return None, None, None
        except Exception as e:
            logging.error(f"Error in datetime conversion for dynamic VWAP bands: {e}")
            return None, None, None
        
        # Get unique dates in the data, sorted in ascending order
        unique_dates = sorted(df_copy["datetime"].dt.date.unique())
        
        if len(unique_dates) < 2:
            # Only log warning once per symbol to avoid spam
            symbol_key = f"dynamic_vwap_insufficient_dates"
            if symbol_key not in self.warned_symbols:
                logging.warning("Insufficient trading days for dynamic VWAP bands calculation")
                self.warned_symbols.add(symbol_key)
            return None, None, None
        
        # The last date is today, the second-to-last date is the previous trading day
        current_date = unique_dates[-1]
        previous_trading_date = unique_dates[-2]
        
        # Filter for today's and previous trading day's data
        today_df = df_copy[df_copy["datetime"].dt.date == current_date]
        prev_day_df = df_copy[df_copy["datetime"].dt.date == previous_trading_date]
        
        if prev_day_df.empty:
            # Only log warning once per symbol to avoid spam
            symbol_key = f"dynamic_vwap_{current_date}"
            if symbol_key not in self.warned_symbols:
                logging.warning(f"No previous trading day data for dynamic VWAP bands (looking for {previous_trading_date})")
                self.warned_symbols.add(symbol_key)
            return None, None, None
        
        # Combine previous trading day and today's data
        combined_df = pd.concat([prev_day_df, today_df])
        
        if combined_df.empty or combined_df["volume"].sum() == 0:
            logging.warning("Insufficient data for dynamic VWAP bands calculation")
            return None, None, None
        
        # Calculate typical price
        combined_df["typical"] = (combined_df["high"] + combined_df["low"] + 
                            combined_df["open"] + combined_df["close"]) / 4
        
        # Calculate cumulative values
        combined_df["vol_times_price"] = combined_df["typical"] * combined_df["volume"]
        cum_vol = combined_df["volume"].cumsum()
        cum_vol_price = combined_df["vol_times_price"].cumsum()
        
        # Calculate VWAP for each point
        combined_df["vwap"] = cum_vol_price / cum_vol
        
        # TradingView-style standard deviation calculation
        combined_df["sq_dev"] = ((combined_df["typical"] - combined_df["vwap"]) ** 2) * combined_df["volume"]
        cum_sq_dev = combined_df["sq_dev"].cumsum()
        combined_df["stdev"] = (cum_sq_dev / cum_vol).apply(lambda x: math.sqrt(x) if x > 0 else 0)
        
        # Get final values
        dynamic_vwap = combined_df["vwap"].iloc[-1]
        stdev = combined_df["stdev"].iloc[-1]
        
        # Calculate bands with specified multiplier
        upper_band = dynamic_vwap + stdev * band_mult
        lower_band = dynamic_vwap - stdev * band_mult
        
        if LOGGING_MODE:
            logging.debug(f"Dynamic VWAP calculation: VWAP={dynamic_vwap:.4f}, StdDev={stdev:.4f}")
            logging.debug(f"Dynamic VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}")
        
        return dynamic_vwap, upper_band, lower_band



    def calculate_eod_vwap_with_stdev_bands(self, df, band_mult=1.0):
        if df.empty:
            return None, None, None
                
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Convert time to datetime properly
        try:
            df_copy["datetime"] = pd.to_datetime(df_copy["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            if df_copy["datetime"].isna().all():
                logging.error("Failed to convert time to datetime for EOD VWAP bands")
                return None, None, None
        except Exception as e:
            logging.error(f"Error in datetime conversion for EOD VWAP bands: {e}")
            return None, None, None
        
        # Get unique dates in the data, sorted in ascending order
        unique_dates = sorted(df_copy["datetime"].dt.date.unique())
        
        if len(unique_dates) < 2:
            # Only log warning once per symbol to avoid spam
            symbol_key = f"eod_vwap_insufficient_dates"
            if symbol_key not in self.warned_symbols:
                logging.warning("Insufficient trading days for EOD VWAP bands calculation")
                self.warned_symbols.add(symbol_key)
            return None, None, None
        
        # The last date is today, the second-to-last date is the previous trading day
        current_date = unique_dates[-1]
        previous_trading_date = unique_dates[-2]
        
        # Filter for today's data
        today_df = df_copy[df_copy["datetime"].dt.date == current_date]
        
        # Get only the last candle from the previous trading day
        prev_day_df = df_copy[df_copy["datetime"].dt.date == previous_trading_date]
        
        if prev_day_df.empty:
            # Only log warning once per symbol to avoid spam
            symbol_key = f"eod_vwap_{current_date}"
            if symbol_key not in self.warned_symbols:
                logging.warning(f"No previous trading day data for EOD VWAP bands (looking for {previous_trading_date})")
                self.warned_symbols.add(symbol_key)
            return None, None, None
        
        # Get last candle from previous trading day
        last_candle_prev = prev_day_df.iloc[[-1]]
        
        # Combine the last candle from previous trading day with today's data
        eod_df = pd.concat([last_candle_prev, today_df])
        
        if eod_df.empty or eod_df["volume"].sum() == 0:
            logging.warning("Insufficient data for EOD VWAP bands calculation")
            return None, None, None
        
        # Calculate typical price
        eod_df["typical"] = (eod_df["high"] + eod_df["low"] + 
                        eod_df["open"] + eod_df["close"]) / 4
        
        # Calculate cumulative values
        eod_df["vol_times_price"] = eod_df["typical"] * eod_df["volume"]
        cum_vol = eod_df["volume"].cumsum()
        cum_vol_price = eod_df["vol_times_price"].cumsum()
        
        # Calculate VWAP for each point
        eod_df["vwap"] = cum_vol_price / cum_vol
        
        # TradingView-style standard deviation calculation
        eod_df["sq_dev"] = ((eod_df["typical"] - eod_df["vwap"]) ** 2) * eod_df["volume"]
        cum_sq_dev = eod_df["sq_dev"].cumsum()
        eod_df["stdev"] = (cum_sq_dev / cum_vol).apply(lambda x: math.sqrt(x) if x > 0 else 0)
        
        # Get final values
        eod_vwap = eod_df["vwap"].iloc[-1]
        stdev = eod_df["stdev"].iloc[-1]
        
        # Calculate bands with specified multiplier
        upper_band = eod_vwap + stdev * band_mult
        lower_band = eod_vwap - stdev * band_mult
        
        if LOGGING_MODE:
            logging.debug(f"EOD VWAP calculation: VWAP={eod_vwap:.4f}, StdDev={stdev:.4f}")
            logging.debug(f"EOD VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}")
        
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
        current_date = df["datetimde"].iloc[-1].date()
        prev_session = df[df["datetime"].dt.date < current_date]
        if prev_session.empty:
            return None, None
        prev_high = prev_session["high"].max()
        prev_low = prev_session["low"].min()
        return prev_high, prev_low
    


    def build_atr_cache(self):
        all_symbols = set(self.longs + self.shorts)
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
            if symbol in self.longs:
                colored_msg = Fore.GREEN + msg + Style.RESET_ALL
            elif symbol in self.shorts:
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
        
        if symbol in self.longs:
            msg += f"Prev Day High: {prev_high:.2f}, " if prev_high is not None else "Prev Day High: N/A, "
        elif symbol in self.shorts:
            msg += f"Prev Day Low: {prev_low:.2f}, " if prev_low is not None else "Prev Day Low: N/A, "
        
        msg += f"ATR: {atr_val:.2f}"
        self.log_symbol(symbol, msg)



    def evaluate_bounce_candidate(self, symbol, df):
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
        direction = "long" if symbol in self.longs else "short"
        
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
        
        # Skip if we don't have enough candles for today
        if len(today_df) < CONSECUTIVE_CANDLES + 1:  # +1 for current candle
            logging.debug(f"{symbol}: Not enough candles for today ({len(today_df)}) to check consecutive condition")
            # We can still continue without the consecutive check if it's disabled
            if CHECK_CONSECUTIVE_CANDLES:
                return None
        
        # Log the metrics being used for evaluation
        if LOGGING_MODE:
            std_vwap_str = f"{metrics.get('std_vwap'):.4f}" if metrics.get('std_vwap') is not None else "None"
            dynamic_vwap_str = f"{metrics.get('dynamic_vwap'):.4f}" if metrics.get('dynamic_vwap') is not None else "None"
            eod_vwap_str = f"{metrics.get('eod_vwap'):.4f}" if metrics.get('eod_vwap') is not None else "None"
            upper_band_str = f"{metrics.get('vwap_1stdev_upper'):.4f}" if metrics.get('vwap_1stdev_upper') is not None else "None"
            lower_band_str = f"{metrics.get('vwap_1stdev_lower'):.4f}" if metrics.get('vwap_1stdev_lower') is not None else "None"
            dynamic_upper_str = f"{metrics.get('dynamic_vwap_1stdev_upper'):.4f}" if metrics.get('dynamic_vwap_1stdev_upper') is not None else "None"
            dynamic_lower_str = f"{metrics.get('dynamic_vwap_1stdev_lower'):.4f}" if metrics.get('dynamic_vwap_1stdev_lower') is not None else "None"
            eod_upper_str = f"{metrics.get('eod_vwap_1stdev_upper'):.4f}" if metrics.get('eod_vwap_1stdev_upper') is not None else "None"
            eod_lower_str = f"{metrics.get('eod_vwap_1stdev_lower'):.4f}" if metrics.get('eod_vwap_1stdev_lower') is not None else "None"
            
            logging.debug(f"{symbol} evaluation using - Std VWAP: {std_vwap_str}, Dynamic VWAP: {dynamic_vwap_str}, EOD VWAP: {eod_vwap_str}")
            logging.debug(f"{symbol} bands - Std 1SD Upper: {upper_band_str}, Std 1SD Lower: {lower_band_str}")
            logging.debug(f"{symbol} dyn bands - Dynamic 1SD Upper: {dynamic_upper_str}, Dynamic 1SD Lower: {dynamic_lower_str}")
            logging.debug(f"{symbol} eod bands - EOD 1SD Upper: {eod_upper_str}, EOD 1SD Lower: {eod_lower_str}")

        # Initialize dictionary for reference levels that triggered bounce condition
        ref_levels = {}

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

        # Check for 10-candle bounce if enabled
        if CHECK_BOUNCE_10_CANDLE and len(df) >= 11:
            if direction == "long":
                # For longs, check if current candle creates a new lowest low
                last_10_candles = df.iloc[-11:-1].copy()  # Exclude current candle
                lowest_low_prev = last_10_candles["low"].min()
                
                # Bounce condition: current candle creates a new low and closes above the open
                if current_candle_data["low"] < lowest_low_prev and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["10_candle_low"] = current_candle_data["low"]
                    logging.debug(f"{symbol}: 10-candle LONG bounce candidate found. New low: {current_candle_data['low']:.2f}, Previous lowest: {lowest_low_prev:.2f}")
            else:
                # For shorts, check if current candle creates a new highest high
                last_10_candles = df.iloc[-11:-1].copy()
                highest_high_prev = last_10_candles["high"].max()
                
                # Bounce condition: current candle creates a new high and closes below the open
                if current_candle_data["high"] > highest_high_prev and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["10_candle_high"] = current_candle_data["high"]
                    logging.debug(f"{symbol}: 10-candle SHORT bounce candidate found. New high: {current_candle_data['high']:.2f}, Previous highest: {highest_high_prev:.2f}")

        # Check for standard VWAP bounces if enabled
        if CHECK_BOUNCE_VVWAP and metrics.get("std_vwap") is not None:
            # Check if price respected standard VWAP for consecutive candles
            if check_consecutive_respect(metrics.get("std_vwap"), "Standard VWAP"):
                if direction == "long" and abs(current_candle_data["low"] - metrics.get("std_vwap")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["vwap"] = metrics.get("std_vwap")
                    logging.debug(f"{symbol}: Standard VWAP LONG bounce candidate found. VWAP: {metrics.get('std_vwap'):.2f}, Current Low: {current_candle_data['low']:.2f}")
                elif direction == "short" and abs(current_candle_data["high"] - metrics.get("std_vwap")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["vwap"] = metrics.get("std_vwap")
                    logging.debug(f"{symbol}: Standard VWAP SHORT bounce candidate found. VWAP: {metrics.get('std_vwap'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for Dynamic VWAP bounces if enabled
        if CHECK_BOUNCE_DYNAMIC_VVWAP and metrics.get("dynamic_vwap") is not None:
            # Check if price respected dynamic VWAP for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap"), "Dynamic VWAP"):
                if direction == "long" and abs(current_candle_data["low"] - metrics.get("dynamic_vwap")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["dynamic_vwap"] = metrics.get("dynamic_vwap")
                    logging.debug(f"{symbol}: Dynamic VWAP LONG bounce candidate found. DVWAP: {metrics.get('dynamic_vwap'):.2f}, Current Low: {current_candle_data['low']:.2f}")
                elif direction == "short" and abs(current_candle_data["high"] - metrics.get("dynamic_vwap")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["dynamic_vwap"] = metrics.get("dynamic_vwap")
                    logging.debug(f"{symbol}: Dynamic VWAP SHORT bounce candidate found. DVWAP: {metrics.get('dynamic_vwap'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for EOD VWAP bounces if enabled
        if CHECK_BOUNCE_EOD_VWAP and metrics.get("eod_vwap") is not None:
            # Check if price respected EOD VWAP for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap"), "EOD VWAP"):
                if direction == "long" and abs(current_candle_data["low"] - metrics.get("eod_vwap")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["eod_vwap"] = metrics.get("eod_vwap")
                    logging.debug(f"{symbol}: EOD VWAP LONG bounce candidate found. EOD VWAP: {metrics.get('eod_vwap'):.2f}, Current Low: {current_candle_data['low']:.2f}")
                elif direction == "short" and abs(current_candle_data["high"] - metrics.get("eod_vwap")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["eod_vwap"] = metrics.get("eod_vwap")
                    logging.debug(f"{symbol}: EOD VWAP SHORT bounce candidate found. EOD VWAP: {metrics.get('eod_vwap'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for VWAP upper band bounces for longs
        if CHECK_BOUNCE_VWAP_UPPER_BAND and direction == "long" and metrics.get("vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_upper"), "VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["vwap_upper_band"] = metrics.get("vwap_1stdev_upper")
                    logging.debug(f"{symbol}: VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for VWAP lower band bounces for shorts
        if CHECK_BOUNCE_VWAP_LOWER_BAND and direction == "short" and metrics.get("vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_lower"), "VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["vwap_lower_band"] = metrics.get("vwap_1stdev_lower")
                    logging.debug(f"{symbol}: VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for Dynamic VWAP upper band bounces for longs
        if CHECK_BOUNCE_DYNAMIC_VWAP_UPPER_BAND and direction == "long" and metrics.get("dynamic_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_upper"), "Dynamic VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("dynamic_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["dynamic_vwap_upper_band"] = metrics.get("dynamic_vwap_1stdev_upper")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('dynamic_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for Dynamic VWAP lower band bounces for shorts
        if CHECK_BOUNCE_DYNAMIC_VWAP_LOWER_BAND and direction == "short" and metrics.get("dynamic_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_lower"), "Dynamic VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("dynamic_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["dynamic_vwap_lower_band"] = metrics.get("dynamic_vwap_1stdev_lower")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('dynamic_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for EOD VWAP upper band bounces for longs
        if CHECK_BOUNCE_EOD_VWAP_UPPER_BAND and direction == "long" and metrics.get("eod_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_upper"), "EOD VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("eod_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["eod_vwap_upper_band"] = metrics.get("eod_vwap_1stdev_upper")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('eod_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for EOD VWAP lower band bounces for shorts
        if CHECK_BOUNCE_EOD_VWAP_LOWER_BAND and direction == "short" and metrics.get("eod_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_lower"), "EOD VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("eod_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["eod_vwap_lower_band"] = metrics.get("eod_vwap_1stdev_lower")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('eod_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for previous day high/low bounces if enabled
        if direction == "long" and CHECK_BOUNCE_PREV_DAY_HIGH and metrics.get("prev_high") is not None:
            # Check if price respected previous day high for consecutive candles
            if check_consecutive_respect(metrics.get("prev_high"), "Previous Day High"):
                # Only consider bounce if price respected the level all day
                if abs(current_candle_data["low"] - metrics.get("prev_high")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["prev_day_high"] = metrics.get("prev_high")
                    logging.debug(f"{symbol}: Previous Day High LONG bounce candidate found. Prev High: {metrics.get('prev_high'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        elif direction == "short" and CHECK_BOUNCE_PREV_DAY_LOW and metrics.get("prev_low") is not None:
            # Check if price respected previous day low for consecutive candles
            if check_consecutive_respect(metrics.get("prev_low"), "Previous Day Low"):
                # Only consider bounce if price respected the level all day
                if abs(current_candle_data["high"] - metrics.get("prev_low")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["prev_day_low"] = metrics.get("prev_low")
                    logging.debug(f"{symbol}: Previous Day Low SHORT bounce candidate found. Prev Low: {metrics.get('prev_low'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Return None if no reference levels were found, otherwise return the details
        return {"levels": ref_levels, "candle": current_candle_data} if ref_levels else None



    def request_and_detect_bounce(self, symbol):
        # Only scan within market hours (if enabled)
        if not SCAN_OUTSIDE_MARKET_HOURS:
            current_time = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
            market_open = current_time.replace(hour=6, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=13, minute=0, second=0, microsecond=0)
            if not (market_open <= current_time <= market_close):
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
        
        # 1. Calculate Standard VWAP (today only)
        standard_vwap = None
        if not today_df.empty and today_df["volume"].sum() > 0:
            today_df["typical"] = (today_df["high"] + today_df["low"] + today_df["open"] + today_df["close"]) / 4
            standard_vwap = (today_df["typical"] * today_df["volume"]).sum() / today_df["volume"].sum()
            logging.debug(f"{symbol}: Standard VWAP calculated from {len(today_df)} today's bars = {standard_vwap:.4f}")
        
        # 2. Calculate Dynamic VWAP (previous trading day + today)
        dynamic_vwap = None
        if not prev_day_df.empty and not today_df.empty:
            combined_df = pd.concat([prev_day_df, today_df])
            if combined_df["volume"].sum() > 0:
                combined_df["typical"] = (combined_df["high"] + combined_df["low"] + combined_df["open"] + combined_df["close"]) / 4
                dynamic_vwap = (combined_df["typical"] * combined_df["volume"]).sum() / combined_df["volume"].sum()
                logging.debug(f"{symbol}: Dynamic VWAP calculated from {len(combined_df)} bars (previous day + today) = {dynamic_vwap:.4f}")
        else:
            logging.debug(f"{symbol}: No previous day data, Dynamic VWAP calculation skipped")
            # Don't fall back to standard VWAP
        
        # 3. Calculate EOD VWAP (previous trading day's last candle + today)
        eod_vwap = None
        if not prev_day_df.empty and not today_df.empty:
            last_candle_prev = prev_day_df.iloc[[-1]]
            eod_df = pd.concat([last_candle_prev, today_df])
            if eod_df["volume"].sum() > 0:
                eod_df["typical"] = (eod_df["high"] + eod_df["low"] + eod_df["open"] + eod_df["close"]) / 4
                eod_vwap = (eod_df["typical"] * eod_df["volume"]).sum() / eod_df["volume"].sum()
                logging.debug(f"{symbol}: EOD VWAP calculated from {len(eod_df)} bars (prev day's last + today) = {eod_vwap:.4f}")
        else:
            logging.debug(f"{symbol}: No previous day data, EOD VWAP calculation skipped")
            # Don't fall back to standard VWAP
        
        # 4. Get previous day extremes
        prev_high = prev_day_df["high"].max() if not prev_day_df.empty else None
        prev_low = prev_day_df["low"].min() if not prev_day_df.empty else None
        
        logging.debug(f"{symbol}: Previous day high = {prev_high}, low = {prev_low}")
        
        # 5. Get current price
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
            "eod_vwap_1stdev_lower": eod_lower_band
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
            
            if symbol in self.longs:
                msg += f"Prev Day High: {prev_high:.4f}, " if prev_high is not None else "Prev Day High: N/A, "
            elif symbol in self.shorts:
                msg += f"Prev Day Low: {prev_low:.4f}, " if prev_low is not None else "Prev Day Low: N/A, "
            
            atr = self.atr_cache.get(symbol, None)
            msg += f"ATR: {atr:.4f}" if atr is not None else "ATR: N/A"
            self.log_symbol(symbol, msg)


        # Continue with evaluating bounce candidates
        direction = "long" if symbol in self.longs else "short"
        candidate_info = self.evaluate_bounce_candidate(symbol, df)

        
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
                    
                    # For longs: confirm if the current candle's high is greater than the bounce candle's high
                    if direction == "long" and current_candle["high"] > bounce_candle["high"]:
                        levels = bounce_data["levels"]
                        levels_list = list(levels.keys())
                        bounce_msg = f"{symbol}: Bounce confirmed (long) from {levels_list}"
                        if self.gui_callback:
                            self.gui_callback(bounce_msg, "green")
                        self.log_symbol(symbol, f"ALERT: {bounce_msg}")
                        self.log_bounce_to_file(
                            symbol=symbol,
                            direction="long",
                            levels=levels,
                            bounce_candle=bounce_candle,
                            current_candle=current_candle,
                            threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0)
                        )
                        self.alerted_symbols.add(symbol)
                        self.bounce_candidates.pop(symbol)
                        return  # Exit after confirming a bounce
                
                    # For shorts: confirm if the current candle's low is less than the bounce candle's low
                    elif direction == "short" and current_candle["low"] < bounce_candle["low"]:
                        levels = bounce_data["levels"]
                        levels_list = list(levels.keys())
                        bounce_msg = f"{symbol}: Bounce confirmed (short) from {levels_list}"
                        if self.gui_callback:
                            self.gui_callback(bounce_msg, "red")
                        self.log_symbol(symbol, f"ALERT: {bounce_msg}")
                        self.log_bounce_to_file(
                            symbol=symbol,
                            direction="short",
                            levels=levels,
                            bounce_candle=bounce_candle,
                            current_candle=current_candle,
                            threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0)
                        )
                        self.alerted_symbols.add(symbol)
                        self.bounce_candidates.pop(symbol)
                        return  # Exit after confirming a bounce
                    else:
                        # Remove stale candidates after a certain time period (e.g., 4 hours)
                        detection_time = bounce_data["detection_time"]
                        if (datetime.now() - detection_time).total_seconds() > 14400:  # 4 hours in seconds
                            logging.debug(f"{symbol}: Removing stale bounce candidate detected at {detection_time}")
                            self.bounce_candidates.pop(symbol)
                else:
                    logging.debug(f"{symbol}: Current candle is the same as bounce candle, waiting for next candle")
            except Exception as e:
                logging.error(f"{symbol}: Error during bounce confirmation: {e}")
                self.bounce_candidates.pop(symbol, None)  # Remove problematic candidate

        # STEP 2: Check if current candle is a new bounce candidate - FIXED INDENTATION
        if candidate_info and symbol not in self.bounce_candidates:
            self.bounce_candidates[symbol] = {
                "levels": candidate_info["levels"],
                "bounce_candle": candidate_info["candle"],
                "detection_time": datetime.now()
            }

                        # In the evaluate_bounce_candidate function, where price approaching is logged:
            if LOG_PRICE_APPROACHING:
                # Filter out 10-candle levels for approaching alerts
                approaching_levels = {k: v for k, v in candidate_info["levels"].items() 
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

                    
                    self.log_symbol(symbol, f"Price approaching levels - {level_details}", "approaching")


    def check_removal_conditions(self):
        for file_name, direction in [(LONGS_FILENAME, "long"), (SHORTS_FILENAME, "short")]:
            tickers = read_tickers(file_name)
            for symbol in tickers:
                if symbol not in self.symbol_metrics:
                    continue
                    
                metrics = self.symbol_metrics[symbol]
                current_price = metrics.get("price")
                eod_vwap = metrics.get("eod_vwap")
                dynamic_vwap = metrics.get("dynamic_vwap")
                prev_high = metrics.get("prev_high")
                prev_low = metrics.get("prev_low")
                
                if current_price is None:
                    continue
                    
                if (eod_vwap is None or dynamic_vwap is None or 
                    (direction == "long" and prev_high is None) or 
                    (direction == "short" and prev_low is None)):
                    continue
                
                if direction == "long":
                    if (current_price < eod_vwap and 
                        current_price < dynamic_vwap and 
                        current_price < prev_high):
                        self.remove_from_watchlist(symbol, direction)
                        if self.gui_callback:
                            removal_msg = f"{symbol} removed from {direction}s watchlist - price below all key levels"
                            self.gui_callback(removal_msg, "blue")
                else:
                    if (current_price > eod_vwap and 
                        current_price > dynamic_vwap and 
                        current_price > prev_low):
                        self.remove_from_watchlist(symbol, direction)
                        if self.gui_callback:
                            removal_msg = f"{symbol} removed from {direction}s watchlist - price above all key levels"
                            self.gui_callback(removal_msg, "blue")

    def remove_from_watchlist(self, symbol, direction):
        filename = LONGS_FILENAME if direction == "long" else SHORTS_FILENAME
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
            logging.info(f"{symbol} removed from {filename} due to removal condition.")
        except Exception as e:
            logging.error(f"Error removing {symbol} from {filename}: {e}")

    def run_strategy(self):
        last_warning_reset = datetime.now().date()
        
        while True:
            try:
                # Reset warning cache daily
                current_date = datetime.now().date()
                if current_date != last_warning_reset:
                    self.warned_symbols.clear()
                    last_warning_reset = current_date
                    logging.info("Daily warning cache reset completed")
                
                self.longs = read_tickers(LONGS_FILENAME)
                self.shorts = read_tickers(SHORTS_FILENAME)
                self.alerted_symbols.clear()
                self.symbol_metrics = {}
                self.build_atr_cache()
                all_symbols = set(self.longs + self.shorts)
                for sym in all_symbols:
                    if self.atr_cache.get(sym) is None:
                        continue
                    self.request_and_detect_bounce(sym)
                self.check_removal_conditions()
                wait_for_candle_close()
                if self.gui_callback:
                    self.gui_callback("Candle has closed", "candle_line")
            except Exception as e:
                logging.exception(f"Error in strategy loop: {e}")
                time.sleep(5)


    def check_dynamic_vwap_touches(self):
        results = []
        all_symbols = set(self.longs + self.shorts)
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
        all_symbols = set(self.longs + self.shorts)
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
        all_symbols = set(self.longs + self.shorts)
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
    
    def log_bounce_to_file(self, symbol, direction, levels, bounce_candle, current_candle, threshold):
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            bounce_types_list = list(levels.keys())
            bounce_types_str = ", ".join(bounce_types_list)

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
                f.write(f"{timestamp} | {symbol} | {bounce_types_str} | {direction}\n")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            file_exists = INTRADAY_BOUNCES_CSV.exists()
            with INTRADAY_BOUNCES_CSV.open("a", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=["time_local", "trade_date", "symbol", "direction", "bounce_types"],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "time_local": timestamp,
                        "trade_date": trade_date_str,
                        "symbol": symbol,
                        "direction": direction,
                        "bounce_types": ", ".join(bounce_types_list),
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
def run_bot_with_gui(gui_callback):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler("trading_bot.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])
    logger = logging.getLogger()
    logger.addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi").addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi.client").addFilter(HistoricalDataFilter())

    bot = BounceBot(gui_callback=gui_callback)
    bot.connect("127.0.0.1", 7496, clientId=125)
    api_thread = threading.Thread(target=bot.run, daemon=True)
    api_thread.start()
    while not bot.connection_status:
        time.sleep(1)
    logging.info("BounceBot is connected. Starting strategy loop...")
    strategy_thread = threading.Thread(target=bot.run_strategy, daemon=True)
    strategy_thread.start()
    return bot

##########################################
# GUI Code using Tkinter
##########################################
# Find and replace the light_grey variable definition with dark theme colors
# Around line 677 in the start_gui() function

def start_gui():
    bounce_queue = queue.Queue()
    approaching_queue = queue.Queue()
    # Replace light_grey with dark_grey
    dark_grey = "#2E2E2E"  # Dark grey color code
    text_color = "#E0E0E0"  # Light text color for dark background

    def gui_callback(message, tag):
        if tag == "approaching" or tag.startswith("approaching_"):
            approaching_queue.put((message, tag))
        elif tag == "candle_line":
            bounce_queue.put((message, tag))
            approaching_queue.put((message, tag))
        elif tag == "blue" and "removed from" in message:
            pass
        else:
            bounce_queue.put((message, tag))

    bot_instance = run_bot_with_gui(gui_callback)

    # Main bounce alerts window
    root = tk.Tk()
    root.title("BounceBot Alerts")
    root.geometry("800x600")
    root.configure(background=dark_grey)

    frame = tk.Frame(root, padx=10, pady=10, bg=dark_grey)
    frame.pack(fill=tk.BOTH, expand=True)

    text_area = scrolledtext.ScrolledText(
        frame, 
        wrap=tk.WORD, 
        width=80, 
        height=30,
        font=('Courier', 12),
        state='disabled',
        bg=dark_grey,
        fg=text_color  # Add text color
    )
    text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Configure tags with new color scheme
    text_area.tag_config("green", foreground="#50FA7B", font=('Courier', 12))  # Green for long message text
    text_area.tag_config("red", foreground="#FF5555", font=('Courier', 12))    # Red for short message text
    text_area.tag_config("pink_symbol", foreground="#FF79C6", font=('Courier', 12, 'bold'))  # Pink for long symbols
    text_area.tag_config("orange_symbol", foreground="#FFB86C", font=('Courier', 12, 'bold'))  # Orange for short symbols
    text_area.tag_config("blue", foreground="#8BE9FD", font=('Courier', 12))           # Light blue
    text_area.tag_config("candle_line", foreground="#BD93F9", overstrike=1)            # Purple

    # Create approaching prices window
    approaching_window = tk.Toplevel(root)
    approaching_window.title("Price Approaching Levels")
    approaching_window.geometry("800x600")
    approaching_window.configure(background=dark_grey)
    
    approaching_frame = tk.Frame(approaching_window, padx=10, pady=10, bg=dark_grey)
    approaching_frame.pack(fill=tk.BOTH, expand=True)
    
    approaching_text = scrolledtext.ScrolledText(
        approaching_frame,
        wrap=tk.WORD,
        width=80,
        height=30,
        font=('Courier', 12),
        state='disabled',
        bg=dark_grey,
        fg=text_color  # Add text color
    )
    approaching_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Configure tags for approaching window
    approaching_text.tag_config("approaching", foreground="#FF79C6", font=('Courier', 12))  # Pink
    approaching_text.tag_config("green", foreground="#50FA7B", font=('Courier', 12))        # Green for long message text
    approaching_text.tag_config("red", foreground="#FF5555", font=('Courier', 12))          # Red for short message text
    approaching_text.tag_config("pink_symbol", foreground="#FF79C6", font=('Courier', 12, 'bold'))  # Pink for long symbols
    approaching_text.tag_config("orange_symbol", foreground="#FFB86C", font=('Courier', 12, 'bold'))  # Orange for short symbols
    approaching_text.tag_config("blue", foreground="#8BE9FD", font=('Courier', 12))         # Light blue
    approaching_text.tag_config("approaching_green", foreground="#50FA7B", font=('Courier', 12))
    approaching_text.tag_config("approaching_red", foreground="#FF5555", font=('Courier', 12))
    approaching_text.tag_config("candle_line", foreground="#BD93F9", overstrike=1)          # Purple


    button_frame = tk.Frame(frame, bg=dark_grey)  # Add background color to button frame
    button_frame.pack(fill=tk.X, pady=10)


    def check_dvwap_touches():
        results = bot_instance.check_dynamic_vwap_touches()
        text_area.config(state='normal')
        text_area.insert(tk.END, "\n=== DVWAP Touch Check Results ===\n", "blue")
        for result in results:
            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {result}\n", "green")
        text_area.config(state='disabled')
        text_area.see(tk.END)
        root.update()

    def check_dvwap2_touches():
        results = bot_instance.check_dynamic_vwap2_touches()
        text_area.config(state='normal')
        text_area.insert(tk.END, "\n=== DVWAP2 Touch Check Results ===\n", "blue")
        for result in results:
            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {result}\n", "green")
        text_area.config(state='disabled')
        text_area.see(tk.END)
        root.update()

    dvwap_button = tk.Button(
        button_frame, 
        text="Check DVWAP Touches",
        command=check_dvwap_touches,
        relief=tk.RAISED,
        padx=10
    )
    dvwap_button.pack(side=tk.LEFT, padx=5)

    dvwap2_button = tk.Button(
        button_frame,
        text="Check DVWAP2 Touches",
        command=check_dvwap2_touches,
        relief=tk.RAISED,
        padx=10
    )
    dvwap2_button.pack(side=tk.LEFT, padx=5)

    def process_bounce_queue():
        while True:
            try:
                msg, tag = bounce_queue.get_nowait()
                text_area.config(state='normal')
                
                # Special handling for bounce confirmations to color the symbol differently
                if "Bounce confirmed" in msg:
                    parts = msg.split(":", 1)  # Split at first colon to separate symbol from rest
                    if len(parts) == 2:
                        symbol = parts[0].strip()
                        rest = ":" + parts[1]
                        
                        # Determine symbol color based on direction
                        if "(long)" in rest:
                            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            text_area.insert(tk.END, symbol, "pink_symbol")  # Pink symbol for longs
                            text_area.insert(tk.END, rest + "\n", "green")   # Green text for rest of long message
                        elif "(short)" in rest:
                            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            text_area.insert(tk.END, symbol, "orange_symbol")  # Orange symbol for shorts
                            text_area.insert(tk.END, rest + "\n", "red")       # Red text for rest of short message
                        else:
                            # Fallback if direction can't be determined
                            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    else:
                        text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                else:
                    # Standard handling for other messages
                    text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    
                text_area.config(state='disabled')
                text_area.see(tk.END)
                root.update()
            except queue.Empty:
                break
        root.after(100, process_bounce_queue)

    def process_approaching_queue():
        while True:
            try:
                msg, tag = approaching_queue.get_nowait()
                approaching_text.config(state='normal')
                
                # Special handling for approaching messages to color the symbol differently
                if "Price approaching levels" in msg:
                    parts = msg.split(":", 1)  # Split at first colon to separate symbol from rest
                    if len(parts) == 2:
                        symbol = parts[0].strip()
                        rest = ":" + parts[1]
                        
                        # Determine symbol color based on direction
                        if "(long)" in rest:
                            approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            approaching_text.insert(tk.END, symbol, "pink_symbol")  # Pink symbol for longs
                            approaching_text.insert(tk.END, rest + "\n", "approaching_green")  # Green text for rest
                        elif "(short)" in rest:
                            approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            approaching_text.insert(tk.END, symbol, "orange_symbol")  # Orange symbol for shorts
                            approaching_text.insert(tk.END, rest + "\n", "approaching_red")  # Red text for rest
                        else:
                            # Fallback if direction can't be determined
                            approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    else:
                        approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                else:
                    # Standard handling for other messages
                    approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    
                approaching_text.config(state='disabled')
                approaching_text.see(tk.END)
                approaching_window.update()
            except queue.Empty:
                break
        root.after(100, process_approaching_queue)


    def on_closing():
        bot_instance.disconnect()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    approaching_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing just the approaching window
    
    process_bounce_queue()
    process_approaching_queue()
    root.mainloop()


##########################################
# Main
##########################################
if __name__ == "__main__":
    print("Starting script...")
    reset_log_files()  # Use the new function instead of reset_log_file()
    print("Log files reset complete.")
    
    # Rest of the code remains the same...

    
    try:
        import argparse
        import sys
        print("Imports successful.")

        parser = argparse.ArgumentParser()
        parser.add_argument("--use_gui", action="store_true", help="Use the Tkinter GUI")
        print("Parser created.")
        
        args = parser.parse_args()
        print(f"Arguments parsed: {args}")
        
        # Determine whether to use GUI based on args or global setting
        use_gui = args.use_gui if args.use_gui else USE_GUI
        print(f"Using GUI: {use_gui}")
        
        if use_gui:
            print("Initializing GUI mode...")
            start_gui()
        else:
            print("Initializing console mode...")
            # Set up logging for console mode
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)

            file_handler = logging.FileHandler("trading_bot.log", mode="a")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)

            logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])
            print("Logging configured.")
            
            logger = logging.getLogger()
            logger.addFilter(HistoricalDataFilter())
            logging.getLogger("ibapi").addFilter(HistoricalDataFilter())
            logging.getLogger("ibapi.client").addFilter(HistoricalDataFilter())
            print("Filters applied.")
            
            # Initialize and run the bot without GUI
            print("Creating BounceBot instance...")
            bot = BounceBot()
            print("Connecting to IB API...")
            bot.connect("127.0.0.1", 7496, clientId=125)
            
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
