r"""A reader holding the destination open must not kill a Master AVWAP scan.

The desk lost three scheduled swing scans this way - 2026-08-17 07:30 and
10:00, 2026-08-18 12:00 - each dying after the ``output/signals`` phase with

    run manifest: "error": "PermissionError(13, 'Access is denied')"
    trading_bot.log: PermissionError: [WinError 5] Access is denied:
        '...\.master_avwap_market_prep.txt.<rand>.tmp'
        -> '...\master_avwap_market_prep.txt'
    autopilot.log: Swing scan for slot 12:00 FAILED: Master AVWAP scan
        process exited with code 1.

It is a self-inflicted race. ``write_market_prep_files`` writes the JSON
first; the desk's Market Prep panel watches that JSON with a
QFileSystemWatcher and, on the change, re-reads the *report* text file; and
Windows' ``open()`` does not grant FILE_SHARE_DELETE, so the ``os.replace``
landing milliseconds later is denied. Losing the whole scan - tracker,
reports, feature history, scan factors - because one report file was being
read for a millisecond is out of all proportion to the fault.
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402


class ReplaceRetryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_transient_permission_error_no_longer_loses_the_write(self):
        target = self.dir / "master_avwap_market_prep.txt"
        target.write_text("stale\n", encoding="utf-8")
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError(13, "Access is denied")
            return real_replace(src, dst, *args, **kwargs)

        with patch("os.replace", side_effect=flaky):
            legacy._write_text_atomic(target, "fresh\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "fresh\n")
        self.assertEqual(calls["n"], 3)

    def test_a_permanent_lock_still_raises_rather_than_lying(self):
        target = self.dir / "locked.txt"
        with patch("os.replace", side_effect=PermissionError(13, "Access is denied")):
            with self.assertRaises(PermissionError):
                legacy._write_text_atomic(target, "fresh\n")

    def test_no_temp_file_is_left_behind_after_a_permanent_failure(self):
        target = self.dir / "locked.txt"
        with patch("os.replace", side_effect=PermissionError(13, "Access is denied")):
            with self.assertRaises(PermissionError):
                legacy._write_text_atomic(target, "fresh\n")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), [])

    @unittest.skipUnless(sys.platform == "win32", "POSIX renames over an open file")
    def test_the_real_desk_race_reader_holding_the_report_open(self):
        """The literal desk failure: a plain reader open on the destination.

        Verified to fail against the un-fixed writer with the same
        ``[WinError 5] Access is denied`` the scan child logged.
        """
        target = self.dir / "master_avwap_market_prep.txt"
        target.write_text("stale\n", encoding="utf-8")
        handle = open(target, "r", encoding="utf-8")
        released = threading.Event()

        def release():
            time.sleep(0.3)
            handle.close()
            released.set()

        thread = threading.Thread(target=release, name="report-reader", daemon=True)
        thread.start()
        try:
            legacy._write_text_atomic(target, "fresh\n")
        finally:
            thread.join(timeout=5)
            if not released.is_set():
                handle.close()

        self.assertEqual(target.read_text(encoding="utf-8"), "fresh\n")

    def test_the_dataframe_writer_retries_on_the_same_terms(self):
        import pandas as pd

        target = self.dir / "rows.csv"
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(13, "Access is denied")
            return real_replace(src, dst, *args, **kwargs)

        with patch("os.replace", side_effect=flaky):
            legacy._write_dataframe_csv_atomic(pd.DataFrame({"a": [1, 2]}), target, index=False)

        self.assertIn("a", target.read_text(encoding="utf-8"))
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
