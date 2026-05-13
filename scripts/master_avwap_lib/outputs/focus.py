from __future__ import annotations

from .._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "build_master_avwap_focus_entries",
        "build_master_avwap_focus_side_groups",
        "build_master_avwap_focus_setup_type_text",
        "write_master_avwap_focus_feed",
        "update_master_avwap_d1_watchlist",
    ),
)
