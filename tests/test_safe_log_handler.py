import logging
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _record(msg):
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)


def test_safe_rotating_handler_survives_locked_rollover(tmp_path, monkeypatch):
    from project_paths import SafeRotatingFileHandler

    logfile = tmp_path / "trading_bot.log"
    handler = SafeRotatingFileHandler(str(logfile), maxBytes=50, backupCount=1)
    try:
        # Simulate Google Drive holding the file: the rename in rollover fails.
        def locked_rotate(source, dest):
            raise PermissionError(32, "The process cannot access the file")

        monkeypatch.setattr(handler, "rotate", locked_rotate)

        # A record larger than maxBytes forces a rollover attempt; it must not raise.
        handler.emit(_record("x" * 100))
        handler.emit(_record("y" * 100))

        # backed off instead of spamming, and logging kept working
        assert handler._rollover_blocked_until > time.monotonic()
        assert handler.shouldRollover(_record("z" * 100)) is False
        assert handler.stream is not None
    finally:
        handler.close()

    assert logfile.exists()
    assert logfile.read_text(encoding="utf-8").strip()  # records were still written


def test_safe_rotating_handler_rotates_normally_when_unlocked(tmp_path):
    from project_paths import SafeRotatingFileHandler

    logfile = tmp_path / "trading_bot.log"
    handler = SafeRotatingFileHandler(str(logfile), maxBytes=50, backupCount=1)
    try:
        handler.emit(_record("a" * 100))
        handler.emit(_record("b" * 100))  # should trigger a real rollover
    finally:
        handler.close()

    assert logfile.exists()
    assert (tmp_path / "trading_bot.log.1").exists()  # rolled over cleanly
