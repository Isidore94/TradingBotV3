from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarketPrepPaths:
    longs_file: str = "longs.txt"
    shorts_file: str = "shorts.txt"
    output_dir: str = "output"
    cache_dir: str = "data/cache"
    log_dir: str = "logs"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "MarketPrepPaths":
        payload = payload if isinstance(payload, dict) else {}
        return cls(
            longs_file=str(payload.get("longs_file") or cls.longs_file),
            shorts_file=str(payload.get("shorts_file") or cls.shorts_file),
            output_dir=str(payload.get("output_dir") or cls.output_dir),
            cache_dir=str(payload.get("cache_dir") or cls.cache_dir),
            log_dir=str(payload.get("log_dir") or cls.log_dir),
        )

    def resolve(self, repo_root: Path) -> dict[str, Path]:
        root = Path(repo_root)
        return {
            "longs_file": _resolve_under_root(root, self.longs_file),
            "shorts_file": _resolve_under_root(root, self.shorts_file),
            "output_dir": _resolve_under_root(root, self.output_dir),
            "cache_dir": _resolve_under_root(root, self.cache_dir),
            "log_dir": _resolve_under_root(root, self.log_dir),
        }


@dataclass(frozen=True)
class MarketPrepConfig:
    timezone: str = "America/Vancouver"
    market_timezone: str = "America/New_York"
    api_keys: dict[str, str] = field(default_factory=dict)
    features: dict[str, bool] = field(default_factory=dict)
    earnings: dict[str, Any] = field(default_factory=dict)
    forexfactory: dict[str, Any] = field(default_factory=dict)
    yfinance: dict[str, Any] = field(default_factory=dict)
    fed_calendar: dict[str, Any] = field(default_factory=dict)
    treasury_calendar: dict[str, Any] = field(default_factory=dict)
    sec_filings: dict[str, Any] = field(default_factory=dict)
    google_news_rss: dict[str, Any] = field(default_factory=dict)
    llm_summary: dict[str, Any] = field(default_factory=dict)
    ticker_lookup: dict[str, Any] = field(default_factory=dict)
    market_prep_ai: dict[str, Any] = field(default_factory=dict)
    paths: MarketPrepPaths = field(default_factory=MarketPrepPaths)
    config_path: Path | None = None
    repo_root: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        config_path: Path | None = None,
        repo_root: Path | None = None,
    ) -> "MarketPrepConfig":
        api_keys = payload.get("api_keys")
        features = payload.get("features")
        earnings = payload.get("earnings")
        forexfactory = payload.get("forexfactory")
        yfinance = payload.get("yfinance")
        fed_calendar = payload.get("fed_calendar")
        treasury_calendar = payload.get("treasury_calendar")
        sec_filings = payload.get("sec_filings")
        google_news_rss = payload.get("google_news_rss")
        llm_summary = payload.get("llm_summary")
        ticker_lookup = payload.get("ticker_lookup")
        market_prep_ai = payload.get("market_prep_ai")
        return cls(
            timezone=str(payload.get("timezone") or cls.timezone),
            market_timezone=str(payload.get("market_timezone") or cls.market_timezone),
            api_keys={str(key): str(value or "") for key, value in (api_keys or {}).items()}
            if isinstance(api_keys, dict)
            else {},
            features={str(key): bool(value) for key, value in (features or {}).items()}
            if isinstance(features, dict)
            else {},
            earnings=dict(earnings) if isinstance(earnings, dict) else {},
            forexfactory=dict(forexfactory) if isinstance(forexfactory, dict) else {},
            yfinance=dict(yfinance) if isinstance(yfinance, dict) else {},
            fed_calendar=dict(fed_calendar) if isinstance(fed_calendar, dict) else {},
            treasury_calendar=dict(treasury_calendar) if isinstance(treasury_calendar, dict) else {},
            sec_filings=dict(sec_filings) if isinstance(sec_filings, dict) else {},
            google_news_rss=dict(google_news_rss) if isinstance(google_news_rss, dict) else {},
            llm_summary=dict(llm_summary) if isinstance(llm_summary, dict) else {},
            ticker_lookup=dict(ticker_lookup) if isinstance(ticker_lookup, dict) else {},
            market_prep_ai=dict(market_prep_ai) if isinstance(market_prep_ai, dict) else {},
            paths=MarketPrepPaths.from_mapping(payload.get("paths")),
            config_path=config_path,
            repo_root=repo_root,
        )

    def resolved_paths(self) -> dict[str, Path]:
        if self.repo_root is None:
            return {}
        return self.paths.resolve(self.repo_root)


def _resolve_under_root(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path
