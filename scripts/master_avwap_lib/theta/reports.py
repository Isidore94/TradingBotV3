from __future__ import annotations

from .._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "write_theta_put_report",
        "extract_theta_symbols_from_report",
        "extract_theta_rows_from_report",
        "extract_theta_reason_risk_rows",
    ),
)
