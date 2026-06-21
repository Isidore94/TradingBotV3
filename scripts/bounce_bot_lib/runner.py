from __future__ import annotations

from ._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "reset_log_files",
        "configure_app_logging",
        "run_bot_with_gui",
        "main",
    ),
)
