from __future__ import annotations

from .._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "DAILY_BAR_COLUMNS",
        "DAILY_BAR_SOURCE_CACHE",
        "DAILY_BAR_SOURCE_IBKR",
        "DAILY_BAR_SOURCE_YAHOO",
        "_DAILY_BAR_FRAME_CACHE",
        "_DAILY_BAR_CACHE_TOUCHED_AT",
        "_DAILY_BAR_LIVE_FAILURE_AT",
        "_daily_bar_cache_file",
        "_daily_bar_cache_file_mtime",
        "_get_daily_bar_source",
        "_set_daily_bar_source",
        "_empty_daily_bar_frame",
        "_normalize_daily_bar_frame",
        "_load_cached_daily_bar_frame",
        "_write_cached_daily_bar_frame",
        "_merge_daily_bar_frames",
        "_daily_bar_cache_covers_history",
        "_daily_bar_cache_data_is_recent",
        "_daily_bar_cache_is_recent",
        "_daily_bar_live_failure_in_cooldown",
        "_mark_daily_bar_live_fetch_result",
        "_flatten_yahoo_daily_bar_columns",
        "fetch_daily_bars_from_yahoo",
        "_fetch_live_daily_bars",
        "fetch_daily_bars",
        "connect_daily_data_client",
        "disconnect_daily_data_client",
        "is_daily_data_client_connected",
    ),
)
