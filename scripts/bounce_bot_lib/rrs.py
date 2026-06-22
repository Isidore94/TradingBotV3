from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "real_relative_strength",
        "load_sector_etf_map",
        "load_industry_etf_map",
        "load_and_update_industry_etf_map",
        "load_symbol_classification_cache",
        "resolve_sector_etf",
        "resolve_industry_ref_etf",
        "build_environment_focus_copy_text",
    ),
)
