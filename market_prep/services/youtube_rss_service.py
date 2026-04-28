from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from market_prep.config_loader import CONFIG_DIR, load_market_prep_config

try:
    import feedparser
except ImportError:  # pragma: no cover - depends on optional local environment
    feedparser = None


YOUTUBE_CREATORS_FILE = CONFIG_DIR / "youtube_creators.json"
FEEDPARSER_MISSING_MESSAGE = "feedparser not installed. Run: pip install feedparser"
DEFAULT_YOUTUBE_CREATORS_CONFIG = {
    "creators": [
        {
            "name": "Creator Name",
            "channel_id": "",
            "keywords": ["SPY", "QQQ", "CPI", "Fed", "NVDA", "market"],
        }
    ]
}


def fetch_youtube_links(
    *,
    limit: int = 25,
    config_path: Path = YOUTUBE_CREATORS_FILE,
) -> dict[str, Any]:
    ensure_youtube_creators_config(config_path)
    generated_at = datetime.now().isoformat(timespec="seconds")
    if feedparser is None:
        payload = _payload(generated_at, [], FEEDPARSER_MISSING_MESSAGE, warnings=[FEEDPARSER_MISSING_MESSAGE])
        output_path = save_youtube_links(payload)
        payload["output_path"] = str(output_path)
        return payload

    creators = load_youtube_creators(config_path)
    videos: list[dict[str, Any]] = []
    warnings: list[str] = []
    for creator in creators:
        name = str(creator.get("name") or "YouTube Creator").strip()
        channel_id = str(creator.get("channel_id") or "").strip()
        keywords = _normalize_keywords(creator.get("keywords"))
        include_all = bool(creator.get("include_all", False))
        if not channel_id:
            warnings.append(f"Skipped {name}: blank channel_id")
            continue
        feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:
            warnings.append(f"{name}: {exc}")
            continue
        if getattr(parsed, "bozo", False):
            warnings.append(f"{name}: feed parse warning")
        for entry in getattr(parsed, "entries", []) or []:
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            matched_keywords = _matched_keywords(title, keywords)
            if keywords and not matched_keywords and not include_all:
                continue
            videos.append(
                {
                    "creator": name,
                    "title": title,
                    "published": str(entry.get("published") or entry.get("updated") or "").strip(),
                    "url": str(entry.get("link") or "").strip(),
                    "matched_keywords": matched_keywords,
                }
            )

    limited = videos[: max(0, int(limit))]
    message = "" if limited else "No configured YouTube links found."
    payload = _payload(generated_at, limited, message, warnings=warnings)
    output_path = save_youtube_links(payload)
    payload["output_path"] = str(output_path)
    return payload


def load_youtube_creators(path: Path = YOUTUBE_CREATORS_FILE) -> list[dict[str, Any]]:
    ensure_youtube_creators_config(path)
    payload = _read_json(path, DEFAULT_YOUTUBE_CREATORS_CONFIG)
    creators = payload.get("creators") if isinstance(payload, dict) else []
    return [creator for creator in creators if isinstance(creator, dict)] if isinstance(creators, list) else []


def ensure_youtube_creators_config(path: Path = YOUTUBE_CREATORS_FILE) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return False
    _write_json_atomic(target, DEFAULT_YOUTUBE_CREATORS_CONFIG)
    return True


def save_youtube_links(payload: dict[str, Any]) -> Path:
    config = load_market_prep_config()
    output_dir = config.resolved_paths().get("output_dir") or Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_date = datetime.now().date().isoformat()
    output_path = output_dir / f"youtube_links_{report_date}.txt"
    lines = [
        f"YouTube Links - {report_date}",
        "=" * 80,
        f"Generated at: {payload.get('generated_at') or 'n/a'}",
        "",
    ]
    videos = payload.get("videos") if isinstance(payload, dict) else []
    if not videos:
        lines.append(str(payload.get("message") or "No configured YouTube links found."))
    else:
        for video in videos:
            if not isinstance(video, dict):
                continue
            keywords = ", ".join(video.get("matched_keywords") or [])
            lines.append(f"{video.get('creator') or 'Creator'} - {video.get('title') or ''}")
            if video.get("published"):
                lines.append(f"Published: {video.get('published')}")
            if keywords:
                lines.append(f"Matched: {keywords}")
            if video.get("url"):
                lines.append(str(video.get("url")))
            lines.append("")
    _write_text_atomic(output_path, "\n".join(lines).rstrip() + "\n")
    return output_path


def _payload(
    generated_at: str,
    videos: list[dict[str, Any]],
    message: str,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "source": "youtube_rss",
        "videos": videos,
        "message": message,
        "warnings": warnings or [],
    }


def _normalize_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords = []
    seen = set()
    for raw_keyword in value:
        keyword = str(raw_keyword or "").strip()
        lowered = keyword.lower()
        if keyword and lowered not in seen:
            seen.add(lowered)
            keywords.append(keyword)
    return keywords


def _matched_keywords(title: str, keywords: list[str]) -> list[str]:
    lowered_title = str(title or "").lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered_title]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
