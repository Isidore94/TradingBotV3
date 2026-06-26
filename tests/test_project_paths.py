import json
import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_PATHS_FILE = ROOT_DIR / "scripts" / "project_paths.py"


def _load_project_paths(monkeypatch, tmp_path, *, google_drive_root: Path | None = None):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.delenv("TRADINGBOTV3_DATA_DIR", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE", raising=False)
    if google_drive_root is not None:
        google_drive_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("GOOGLE_DRIVE", str(google_drive_root))

    module_name = f"project_paths_under_test_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_PATHS_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_persistent_dir_prefers_google_drive(monkeypatch, tmp_path):
    drive_root = tmp_path / "My Drive"
    module = _load_project_paths(monkeypatch, tmp_path, google_drive_root=drive_root)

    expected = drive_root / "Trading" / "TradingBot"
    assert module.PERSISTENT_DATA_DIR == expected
    assert module.FOCUS_LONGS_FILE == expected / "focus_longs.txt"
    assert module.FOCUS_SHORTS_FILE == expected / "focus_shorts.txt"
    assert module.PERSISTENT_DATA_DIR_SOURCE == "google_drive_default"


def test_saved_storage_dir_still_overrides_google_drive(monkeypatch, tmp_path):
    localappdata = tmp_path / "localappdata"
    settings_dir = localappdata / "TradingBotV3"
    settings_dir.mkdir(parents=True)
    chosen = tmp_path / "custom_shared"
    (settings_dir / "local_settings.json").write_text(
        json.dumps({"shared_data_dir": str(chosen)}),
        encoding="utf-8",
    )

    monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
    drive_root = tmp_path / "My Drive"
    module = _load_project_paths(monkeypatch, tmp_path, google_drive_root=drive_root)

    assert module.PERSISTENT_DATA_DIR == chosen
    assert module.PERSISTENT_DATA_DIR_SOURCE == "local_config"
