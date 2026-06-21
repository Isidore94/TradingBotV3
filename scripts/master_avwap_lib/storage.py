from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "load_tickers",
        "load_tickers_from_paths",
        "resolve_scan_watchlist_paths",
        "resolve_master_scan_watchlist_paths",
        "load_json",
        "save_json",
        "_write_text_atomic",
        "_stable_payload_hash",
        "load_history",
        "save_history",
        "trim_history",
        "append_d1_feature_history",
        "load_rrs_environment_focus_payload",
        "get_rrs_environment_focus_hits_for_symbol",
        "build_bouncebot_focus_context",
    ),
)
