from __future__ import annotations

from datetime import datetime


class FakeBot:
    def __init__(self, callback, scanning=False):
        self.callback = callback
        self.connection_status = True
        self.rrs_threshold = 2.0
        self.rrs_timeframe_key = "5m"
        self.market_environment_user_override = False
        self.latest_bars = {"NVDA": [{"dt": datetime(2026, 9, 15, 10, 0), "close": 100.0}]}
        self.d1_zone_arms = {"NVDA": {"side": "LONG"}}
        self._scanning = bool(scanning)
        callback("RRS ready", "rrs_status")
        callback({"leaders": ["NVDA"]}, "rrs_snapshot")
        callback("NVDA LONG test alert", "red")

    def stop(self, timeout=5.0):
        self.connection_status = False

    def disconnect(self):
        self.connection_status = False

    def set_scanning_enabled(self, enabled):
        self._scanning = bool(enabled)

    def is_scanning_enabled(self):
        return self._scanning

    def set_rrs_threshold(self, value):
        self.rrs_threshold = float(value)

    def set_rrs_timeframe(self, key):
        self.rrs_timeframe_key = str(key)

    def set_market_environment(self, key):
        self.market_environment_user_override = True
        self._environment = key

    def clear_market_environment_override(self):
        self.market_environment_user_override = False

    def get_market_environment(self):
        return getattr(self, "_environment", "mixed")

    def set_bounce_type_enabled(self, key, enabled):
        setattr(self, f"bounce_{key}", bool(enabled))

    def is_bounce_type_enabled(self, key):
        return bool(getattr(self, f"bounce_{key}", True))

    def get_auto_regime_reading(self):
        return {"label": "mixed"}

    def entry_assist_state(self):
        return {"window_active": False}

    def entry_assist_board_snapshot(self):
        return {"rows": [{"symbol": "NVDA"}]}

    def m5_chart_bars(self, symbol, max_sessions=2):
        return list(self.latest_bars.get(str(symbol).upper(), ()))


def run_bot_with_gui(callback, start_scanning_enabled=False):
    return FakeBot(callback, scanning=start_scanning_enabled)
