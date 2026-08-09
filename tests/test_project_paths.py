import json
import importlib.util
import os
import sys
import time
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


def _legacy_migration_fixture(tmp_path):
    """A Windows-shaped ~/AppData store as an early macOS run left it."""
    legacy = tmp_path / "home" / "AppData" / "Local" / "TradingBotV3"
    (legacy / "machine_cache").mkdir(parents=True)
    (legacy / "logs").mkdir()
    (legacy / "local_settings.json").write_text(
        json.dumps({"shared_data_dir": "/Users/trader/My Drive/Trading/TradingBot"}),
        encoding="utf-8",
    )
    (legacy / "longs.txt").write_text("NVDA\nAMD\n", encoding="utf-8")  # user-authored
    (legacy / "machine_cache" / "earnings_cache.json").write_text("{}", encoding="utf-8")
    (legacy / "logs" / "trading_bot.log").write_text("old log line\n", encoding="utf-8")
    preferred = tmp_path / "home" / "Library" / "Application Support" / "TradingBotV3"
    return legacy, preferred


def test_legacy_appdata_dir_migrates_to_native_settings_dir(monkeypatch, tmp_path):
    module = _load_project_paths(monkeypatch, tmp_path, google_drive_root=tmp_path / "My Drive")
    legacy, preferred = _legacy_migration_fixture(tmp_path)

    chosen = module._adopt_legacy_windows_shaped_dir(legacy, preferred)

    assert chosen == preferred
    # User-authored data survives byte-for-byte in the native location.
    assert (preferred / "longs.txt").read_text(encoding="utf-8") == "NVDA\nAMD\n"
    assert "shared_data_dir" in (preferred / "local_settings.json").read_text(encoding="utf-8")
    assert (preferred / "machine_cache" / "earnings_cache.json").exists()
    assert (preferred / "logs" / "trading_bot.log").exists()
    # The legacy dir is left as an empty husk, not deleted.
    assert legacy.exists()
    assert not any(legacy.rglob("*"))

    # Idempotent: a second call with the husk is a no-op that still picks native.
    assert module._adopt_legacy_windows_shaped_dir(legacy, preferred) == preferred


def test_legacy_migration_preserves_both_sides_of_a_conflict(monkeypatch, tmp_path):
    module = _load_project_paths(monkeypatch, tmp_path, google_drive_root=tmp_path / "My Drive")
    legacy, preferred = _legacy_migration_fixture(tmp_path)
    preferred.mkdir(parents=True)
    (preferred / "longs.txt").write_text("TSLA\n", encoding="utf-8")  # both sides evolved

    chosen = module._adopt_legacy_windows_shaped_dir(legacy, preferred)

    assert chosen == preferred
    # Neither version of the user-authored watchlist is lost or overwritten.
    assert (preferred / "longs.txt").read_text(encoding="utf-8") == "TSLA\n"
    assert (preferred / "longs.txt.from-appdata").read_text(encoding="utf-8") == "NVDA\nAMD\n"
    assert not any(legacy.rglob("*"))


def test_default_settings_dir_uses_native_after_migration(monkeypatch, tmp_path):
    module = _load_project_paths(monkeypatch, tmp_path, google_drive_root=tmp_path / "My Drive")
    legacy, preferred = _legacy_migration_fixture(tmp_path)

    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    chosen = module._default_local_settings_dir()

    if sys.platform == "darwin":
        assert chosen == preferred
    else:
        # Non-darwin POSIX/Windows-without-LOCALAPPDATA migrates into
        # ~/.local/share with the same machinery.
        assert chosen == tmp_path / "home" / ".local" / "share" / "TradingBotV3"
    assert not any(legacy.rglob("*"))
    assert (chosen / "longs.txt").read_text(encoding="utf-8") == "NVDA\nAMD\n"


def test_localappdata_env_still_wins_over_everything(monkeypatch, tmp_path):
    module = _load_project_paths(monkeypatch, tmp_path, google_drive_root=tmp_path / "My Drive")
    _legacy_migration_fixture(tmp_path)

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "winappdata"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    chosen = module._default_local_settings_dir()

    assert chosen == tmp_path / "winappdata" / "TradingBotV3"
    # Windows behavior is untouched: no migration ran, legacy files intact.
    assert (tmp_path / "home" / "AppData" / "Local" / "TradingBotV3" / "longs.txt").exists()


def test_default_persistent_dir_finds_macos_cloudstorage_mount(monkeypatch, tmp_path):
    home = tmp_path / "home"
    mount = home / "Library" / "CloudStorage" / "GoogleDrive-trader@example.com" / "My Drive"
    mount.mkdir(parents=True)
    # Path.home() reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    module = _load_project_paths(monkeypatch, tmp_path)

    expected = mount / "Trading" / "TradingBot"
    assert module.PERSISTENT_DATA_DIR == expected
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


def test_wait_for_shared_drive_fails_clearly_when_drive_missing(monkeypatch, tmp_path):
    import pytest

    module = _load_project_paths(monkeypatch, tmp_path, google_drive_root=tmp_path / "My Drive")

    # Mounted/local anchors: no wait, no error.
    module._wait_for_shared_drive(tmp_path / "anything", "test")

    # Unmounted shared store + fail-fast: a clear actionable error, not a
    # mkdir traceback (and never a silent local fallback). Windows simulates
    # a missing drive letter; POSIX simulates a macOS CloudStorage mount that
    # is absent because the Drive client is not running.
    if sys.platform == "win32":
        target = next(
            (Path(f"{letter}:/") for letter in "QWXYZ" if not Path(f"{letter}:/").exists()),
            None,
        )
        if target is None:
            return  # every letter mounted on this machine; nothing to simulate
        target = target / "My Drive" / "Trading"
    else:
        home = tmp_path / "cloudhome"
        (home / "Library" / "CloudStorage").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        target = home / "Library" / "CloudStorage" / "GoogleDrive-trader@example.com" / "My Drive" / "Trading"
    monkeypatch.setenv("TRADINGBOTV3_DRIVE_WAIT_SECONDS", "0")
    with pytest.raises(RuntimeError) as excinfo:
        module._wait_for_shared_drive(target, "test_config")
    message = str(excinfo.value)
    assert "not mounted" in message
    assert "Google Drive" in message
    assert "local fallback is refused" in message


# --- orphaned atomic-write staging files ----------------------------------
# Regression cover for the ~2.3GB of `intraday_bounce_candidates<8>.csv` and
# 19MB of `.earnings_calendar_history.json.<8>.tmp` that accumulated when the
# writers' cleanup unlink failed against a cloud-sync lock.


def _aged(path: Path, *, seconds_old: float) -> Path:
    """Backdate a file's mtime so the sweep considers it stale."""
    stamp = time.time() - seconds_old
    os.utime(path, (stamp, stamp))
    return path


def test_sweep_removes_stale_staging_files_but_spares_real_ones(monkeypatch, tmp_path):
    module = _load_project_paths(monkeypatch, tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    canonical = runtime / "intraday_bounce_candidates.csv"

    old = 24 * 3600
    for name in (
        "intraday_bounce_candidates2s7hbyub.csv",
        ".earnings_calendar_history.json.toj_fmkw.tmp",
    ):
        path = runtime / name
        path.write_text("x", encoding="utf-8")
        _aged(path, seconds_old=old)

    # Must all survive: the canonical target, a sibling with a different stem,
    # a token of the wrong length, and a staging file young enough to belong to
    # a write that is still running.
    survivors = [
        canonical,
        runtime / "intraday_bounce_outcomes.csv",
        runtime / "intraday_bounce_candidatesTOOLONG12.csv",
    ]
    for path in survivors:
        path.write_text("keep", encoding="utf-8")
        _aged(path, seconds_old=old)
    fresh = runtime / "intraday_bounce_candidatesab12cd34.csv"
    fresh.write_text("in flight", encoding="utf-8")
    survivors.append(fresh)

    removed = module.sweep_stale_atomic_write_temps(
        directories=(runtime,), staged_for=(canonical,)
    )

    assert sorted(path.name for path in removed) == [
        ".earnings_calendar_history.json.toj_fmkw.tmp",
        "intraday_bounce_candidates2s7hbyub.csv",
    ]
    for path in survivors:
        assert path.exists(), f"sweep deleted {path.name}, which it must never touch"


def test_sweep_does_not_recurse_into_bar_stores(monkeypatch, tmp_path):
    """The bar directories hold thousands of parquet files; scanning them on
    every startup would be pure cost, and nothing stages temps there."""
    module = _load_project_paths(monkeypatch, tmp_path)
    data = tmp_path / "data"
    nested = data / "daily_bars"
    nested.mkdir(parents=True)
    buried = nested / ".something.json.abcd1234.tmp"
    buried.write_text("x", encoding="utf-8")
    _aged(buried, seconds_old=24 * 3600)

    assert module.sweep_stale_atomic_write_temps(directories=(data,)) == []
    assert buried.exists()


def test_sweep_survives_missing_dirs_and_locked_files(monkeypatch, tmp_path):
    module = _load_project_paths(monkeypatch, tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    locked = runtime / ".locked.json.abcd1234.tmp"
    locked.write_text("x", encoding="utf-8")
    _aged(locked, seconds_old=24 * 3600)

    def _refuse(self):
        raise PermissionError(32, "in use by another process")

    monkeypatch.setattr(Path, "unlink", _refuse)

    # A missing directory and an unlinkable file are both non-fatal: the sweep
    # reports nothing removed and startup continues.
    removed = module.sweep_stale_atomic_write_temps(
        directories=(runtime, tmp_path / "does_not_exist")
    )
    assert removed == []
