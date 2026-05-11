from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "EVENT_TICKERS_FILE",
        "PRIORITY_SETUPS_FILE",
        "THETA_PUTS_FILE",
        "STDEV_RANGE_FILE",
        "TRADINGVIEW_REPORT_FILE",
        "MARKET_PREP_FILE",
        "MARKET_PREP_REPORT_FILE",
        "SETUP_TRACKER_FILE",
        "SCORING_CONFIG_FILE",
        "SCORING_RECOMMENDATIONS_FILE",
        "SCORING_TUNER_REPORT_FILE",
        "ATR_LENGTH",
        "ATR_MULT",
        "MIN_AVG_VOLUME_20D",
        "MIN_PRICE",
        "MIN_MARKET_CAP",
        "SETUP_TRACKER_SCHEMA_VERSION",
        "PRIORITY_SCORING_CONFIG_SCHEMA_VERSION",
        "D1_FEATURE_HISTORY_SCHEMA_VERSION",
        "SETUP_CANDIDATE_SCHEMA_VERSION",
        "USER_FAVORITES_SCHEMA_VERSION",
        "D1_WATCHLIST_SCHEMA_VERSION",
        "default_priority_scoring_config",
        "load_priority_scoring_config",
        "save_priority_scoring_config",
        "get_priority_signal_weights",
        "get_priority_attribute_adjustments",
        "get_scoring_config_metadata",
        "run_priority_scoring_tuner",
    ),
)
