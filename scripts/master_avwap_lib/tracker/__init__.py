from __future__ import annotations

from .._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "load_setup_tracker_payload",
        "save_setup_tracker_payload",
        "build_tracker_setup_record",
        "recompute_tracker_setup_record",
        "build_recent_tracker_setup_family_rows",
        "apply_recent_tracker_setup_family_adjustments",
        "rank_tracker_setup_type_rows",
        "apply_tracker_setup_type_adjustments",
        "build_tracker_stats_rows",
        "build_tracker_playbook_rows",
        "build_tracker_playbook_recommendation_rows",
        "build_tracker_setup_type_rows",
        "build_tracker_factor_view_rows",
        "export_setup_tracker_views",
        "_build_tracker_scenarios_from_setup",
        "_find_tracker_stop_candidates",
        "_evaluate_tracker_scenario_bar",
        "_flatten_tracker_scenarios",
        "_flatten_tracker_daily_marks",
        "_summarize_tracker_setup_outcome",
    ),
)
