from __future__ import annotations

from .._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "build_combined_avwap_output_text",
        "write_priority_setup_report",
        "write_stdev_range_report",
        "write_tradingview_report",
        "write_favorite_zone_watchlist_outputs",
        "append_master_avwap_user_favorites",
    ),
)
