from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "compute_atr_from_ohlc",
        "compute_trend_label_20d",
        "compute_five_day_breakout_flags",
        "count_recent_band_extension_days",
        "count_recent_band_test_days",
        "compute_recent_second_band_penalty",
        "assess_first_dev_break_quality",
        "analyze_avwap_retest_behavior",
        "analyze_extreme_move_retest_setup",
        "compute_indicator_frame",
        "calc_anchored_vwap_band_history",
        "summarize_anchor_compression",
        "evaluate_anchor_compression",
        "calc_anchored_vwap_bands",
        "normalize_side",
        "classify_position_by_band",
        "get_band_context",
        "select_primary_cross_signal",
        "get_atr20",
        "bounce_up_at_level",
        "bounce_down_at_level",
        "cross_up_through_level",
        "cross_down_through_level",
        "closes_between_bands",
        "compute_major_sma_levels",
        "find_directional_trendline_candidate",
    ),
)
