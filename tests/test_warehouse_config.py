"""Research-lake location config (plan Phase 0, docs/ULTIMATE_SETUP_DATABASE_PLAN.md).

Contract under test (sec 19.3 config surface):
- unset ``research_store_dir`` => the warehouse is a total no-op;
- the ``TRADINGBOTV3_RESEARCH_DIR`` environment override wins;
- a path inside the Drive-synced shared home folder is refused, never accepted.
"""

from pathlib import Path

import pytest

from scripts.research_warehouse import config


@pytest.fixture()
def isolated_settings(monkeypatch):
    """No env override, no persisted setting, writes captured not persisted."""
    saved: dict[str, object] = {}
    paths = config._paths()
    monkeypatch.delenv(config.RESEARCH_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "get_local_setting", lambda key, default=None: saved.get(key, default))
    monkeypatch.setattr(paths, "save_local_setting", lambda key, value: saved.__setitem__(key, value))
    return saved


def test_unset_means_disabled_no_op(isolated_settings):
    assert config.get_research_store_dir() is None
    assert config.warehouse_enabled() is False
    with pytest.raises(ValueError):
        config.ensure_lake_layout()


def test_env_override_wins_over_saved_setting(isolated_settings, monkeypatch, tmp_path):
    isolated_settings[config.RESEARCH_DIR_SETTING] = str(tmp_path / "from_setting")
    env_dir = tmp_path / "from_env"
    monkeypatch.setenv(config.RESEARCH_DIR_ENV, str(env_dir))
    assert config.get_research_store_dir() == env_dir
    assert config.warehouse_enabled() is True


def test_saved_setting_used_when_env_unset(isolated_settings, tmp_path):
    lake = tmp_path / "research_lake"
    isolated_settings[config.RESEARCH_DIR_SETTING] = str(lake)
    assert config.get_research_store_dir() == lake


def test_paths_inside_shared_home_are_refused(isolated_settings, monkeypatch):
    inside = Path(config._paths().SHARED_HOME_DIR) / "research_lake"
    monkeypatch.setenv(config.RESEARCH_DIR_ENV, str(inside))
    with pytest.raises(ValueError, match="shared home folder"):
        config.get_research_store_dir()
    # Misconfiguration surfaces as disabled, never as a silent Drive write.
    assert config.warehouse_enabled() is False
    monkeypatch.delenv(config.RESEARCH_DIR_ENV)
    with pytest.raises(ValueError, match="shared home folder"):
        config.save_research_store_dir(str(inside))


def test_save_and_clear_roundtrip(isolated_settings, tmp_path):
    lake = tmp_path / "das" / "research"
    assert config.save_research_store_dir(str(lake)) == lake
    assert config.get_research_store_dir() == lake
    config.clear_research_store_dir()
    assert config.get_research_store_dir() is None
    assert config.warehouse_enabled() is False


def test_ensure_lake_layout_is_idempotent_and_complete(isolated_settings, monkeypatch, tmp_path):
    lake = tmp_path / "lake"
    monkeypatch.setenv(config.RESEARCH_DIR_ENV, str(lake))
    for _ in range(2):  # second pass must be a no-op, never destructive
        assert config.ensure_lake_layout() == lake
    for name in config.LAKE_SUBDIRS:
        assert (lake / name).is_dir()
    for ledger in config.LAKE_LEDGERS:
        assert (lake / ledger).is_file()
    # An existing ledger is never truncated.
    manifest = lake / "manifest_log.jsonl"
    manifest.write_text('{"manifest_seq": 1}\n', encoding="utf-8")
    config.ensure_lake_layout()
    assert manifest.read_text(encoding="utf-8") == '{"manifest_seq": 1}\n'


def test_spool_dir_is_machine_local(isolated_settings):
    spool = config.research_spool_dir()
    assert spool == Path(config._paths().LOCAL_SETTINGS_DIR) / config.SPOOL_DIR_NAME


def test_details_payload_reports_state(isolated_settings, monkeypatch, tmp_path):
    details = config.get_research_store_details()
    assert details["enabled"] == "no" and details["source"] == "unset"
    monkeypatch.setenv(config.RESEARCH_DIR_ENV, str(tmp_path / "lake"))
    details = config.get_research_store_details()
    assert details["enabled"] == "yes" and details["source"] == "environment"
