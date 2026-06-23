from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SetupRow:
    symbol: str
    side: str = ""
    score: float | None = None
    bucket: str = ""
    setup_tags: list[str] = field(default_factory=list)
    key_level: str = ""
    supports: int | None = None
    hv_summary: str = ""
    theta: str = ""
    expected_r: float | None = None
    expected_r_rank: float | None = None
    days_to_earnings: int | None = None
    last_trade_date: str = ""
    source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def bucket_label(self) -> str:
        normalized = self.bucket.strip().lower()
        return {
            "favorite_setup": "Favorite",
            "near_favorite_zone": "Near",
            "high_conviction": "High Conviction",
            "post_earnings_play": "Post Earnings",
            "sma_breakout_tracking": "SMA Track",
            "stdev_retest_tracking": "Stdev Track",
            "study_hv_level": "Study",
            "study_htf_trend": "Study",
            "study_relative_avwap": "Study",
            "study_phase6": "Study",
            "study": "Study",
        }.get(normalized, normalized.replace("_", " ").title() if normalized else "Unbucketed")

    @property
    def tags_text(self) -> str:
        return ", ".join(str(tag) for tag in self.setup_tags if str(tag).strip())

    @property
    def supports_text(self) -> str:
        parts: list[str] = []
        if self.supports is not None:
            parts.append(str(self.supports))
        if self.hv_summary:
            parts.append(self.hv_summary)
        return " / ".join(parts)

    @property
    def expected_r_text(self) -> str:
        if self.expected_r is None:
            return ""
        return f"{self.expected_r:.2f}R"
