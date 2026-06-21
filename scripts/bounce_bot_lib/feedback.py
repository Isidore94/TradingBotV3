from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "record_bounce_feedback",
        "_normalize_bounce_feedback_context",
        "_normalize_alert_message_payload",
    ),
)
