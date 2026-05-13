from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "append_alert_message",
        "configure_alert_tags",
        "create_rrs_confirmed_panel",
    ),
)
