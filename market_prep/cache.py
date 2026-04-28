from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config_loader import load_market_prep_config


def get_default_cache_dir() -> Path:
    config = load_market_prep_config()
    cache_dir = config.resolved_paths().get("cache_dir")
    if cache_dir is None:
        cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def read_json_cache(path: Path, default: Any = None) -> Any:
    cache_path = Path(path)
    if not cache_path.exists():
        return default
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json_cache(path: Path, payload: Any) -> Path:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, cache_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return cache_path
