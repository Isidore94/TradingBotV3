from __future__ import annotations

from .._legacy_bridge import expose_legacy_names

expose_legacy_names(
    globals(),
    (
        "POST_EARNINGS_STOP_FAILURE_CLOSES",
        "POST_EARNINGS_STOP_LABEL",
        "POST_EARNINGS_BOUNCE_MAX_AGE_SESSIONS",
        "POST_EARNINGS_BOUNCE_MIN_SESSIONS_SINCE_GAP",
        "POST_EARNINGS_BREAK_SIGNAL",
        "POST_EARNINGS_BOUNCE_SIGNAL",
        "analyze_post_earnings_setups",
        "analyze_mid_earnings_ema_retest_setup",
        "_build_latest_earnings_release_context",
    ),
)
