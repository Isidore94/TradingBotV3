from __future__ import annotations

from .._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "build_tracker_feature_snapshot",
        "build_tracker_entry_attributes",
        "build_priority_scoring_attribute_context",
        "apply_priority_attribute_adjustments",
        "compute_multi_day_patterns",
        "sort_events_for_output",
        "event_label_sort_key",
        "format_signal_label",
        "build_priority_setup_summary",
        "apply_pre_earnings_priority_blocks",
        "build_setup_candidate_payload",
        "attach_setup_candidate_payloads",
        "assess_priority_directional_obstacles",
        "refine_priority_rows_with_directional_filters",
        "apply_clean_first_zone_score_bonus",
        "apply_tracker_scoring_guardrails",
        "apply_market_regime_score_adjustments",
        "apply_priority_rejection_score_caps",
        "apply_final_priority_buckets",
        "update_setup_tracker_from_scan",
        "backfill_setup_tracker_from_recent_sessions",
    ),
)
