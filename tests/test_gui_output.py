import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gui  # noqa: E402


class _FakeVar:
    def __init__(self, value: str):
        self._value = value

    def get(self) -> str:
        return self._value


class _FakeText:
    def __init__(self, value: str):
        self._value = value

    def get(self, *_args) -> str:
        return self._value


class _FakeAvwapGui:
    def __init__(
        self,
        *,
        avwap_output: str,
        anchor_output: str,
        favorites: str,
        near_favorites: str,
        long_focus: str,
        short_focus: str,
        setup_types: str,
        theta: str = "",
        status: str = "Ready",
    ):
        self.status_var = _FakeVar(status)
        self.avwap_text = _FakeText(avwap_output)
        self.anchor_scan_text = _FakeText(anchor_output)
        self.favorite_symbols_text = _FakeText(favorites)
        self.near_favorite_symbols_text = _FakeText(near_favorites)
        self.long_focus_symbols_text = _FakeText(long_focus)
        self.short_focus_symbols_text = _FakeText(short_focus)
        self.setup_type_symbols_text = _FakeText(setup_types)
        self.theta_symbols_text = _FakeText(theta)


class GuiOutputTests(unittest.TestCase):
    def test_build_consolidated_gui_output_includes_widget_copy_lists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            longs_path = temp_root / "longs.txt"
            shorts_path = temp_root / "shorts.txt"
            snapshot_path = temp_root / "consolidated_gui_output.txt"
            longs_path.write_text("AAPL\nMSFT\n", encoding="utf-8")
            shorts_path.write_text("TSLA\n", encoding="utf-8")

            avwap_gui = _FakeAvwapGui(
                avwap_output="Priority output here",
                anchor_output="Anchor output here",
                favorites="AAPL, NVDA",
                near_favorites="TSLA",
                long_focus="AAPL, MSFT",
                short_focus="TSLA",
                setup_types="avwap_breakout\nLONG: AAPL, MSFT",
                theta="NVDA",
            )

            storage_details = {
                "data_dir": str(temp_root),
                "local_cache_dir": str(temp_root / "cache"),
                "output_dir": str(temp_root / "output"),
                "logs_dir": str(temp_root / "logs"),
            }

            with (
                patch.object(gui, "LONGS_FILE", longs_path),
                patch.object(gui, "SHORTS_FILE", shorts_path),
                patch.object(gui, "MAIN_GUI_OUTPUT_FILE", snapshot_path),
                patch.object(gui, "get_tracker_storage_details", return_value=storage_details),
            ):
                output = gui.build_consolidated_gui_output("full", None, avwap_gui)

            self.assertIn("AVWAP Copy/Paste Lists", output)
            self.assertIn("Favorite Setups\nAAPL, NVDA", output)
            self.assertIn("Near Favorite Zones\nTSLA", output)
            self.assertIn("Theta Plays\nNVDA", output)
            self.assertIn("Directional Longs\nAAPL, MSFT", output)
            self.assertIn("Directional Shorts\nTSLA", output)
            self.assertIn("Setup Type Copy Lists\navwap_breakout\nLONG: AAPL, MSFT", output)

    def test_build_consolidated_gui_output_falls_back_to_focus_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            longs_path = temp_root / "longs.txt"
            shorts_path = temp_root / "shorts.txt"
            snapshot_path = temp_root / "consolidated_gui_output.txt"
            focus_path = temp_root / "master_avwap_focus.json"
            tradingview_path = temp_root / "master_avwap_tradingview.txt"
            theta_path = temp_root / "master_avwap_theta_puts.txt"
            longs_path.write_text("AAPL\n", encoding="utf-8")
            shorts_path.write_text("TSLA\n", encoding="utf-8")
            theta_path.write_text("1. NVDA | close=100.00 | score=10\n", encoding="utf-8")
            focus_path.write_text(
                json.dumps(
                    {
                        "symbols": {
                            "AAPL": {"symbol": "AAPL", "side": "LONG", "setup_family": "avwap_breakout"},
                            "MSFT": {"symbol": "MSFT", "side": "LONG", "setup_family": "avwap_breakout"},
                            "TSLA": {"symbol": "TSLA", "side": "SHORT", "setup_family": "avwap_breakdown"},
                        },
                        "favorites": [{"symbol": "AAPL", "side": "LONG", "setup_family": "avwap_breakout"}],
                        "near_favorite_zones": [
                            {"symbol": "TSLA", "side": "SHORT", "setup_family": "avwap_breakdown"},
                            {"symbol": "MSFT", "side": "LONG", "setup_family": "avwap_breakout"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            tradingview_path.write_text(
                "\n".join(
                    [
                        "Best current favorite setups",
                        "LONG: AAPL",
                        "",
                        "Near favorite zones",
                        "LONG: MSFT",
                        "SHORT: TSLA",
                    ]
                ),
                encoding="utf-8",
            )

            avwap_gui = _FakeAvwapGui(
                avwap_output="Priority output here",
                anchor_output="Anchor output here",
                favorites="",
                near_favorites="",
                long_focus="",
                short_focus="",
                setup_types="",
            )

            storage_details = {
                "data_dir": str(temp_root),
                "local_cache_dir": str(temp_root / "cache"),
                "output_dir": str(temp_root / "output"),
                "logs_dir": str(temp_root / "logs"),
            }

            with (
                patch.object(gui, "LONGS_FILE", longs_path),
                patch.object(gui, "SHORTS_FILE", shorts_path),
                patch.object(gui, "MAIN_GUI_OUTPUT_FILE", snapshot_path),
                patch.object(gui, "MASTER_AVWAP_FOCUS_FILE", focus_path),
                patch.object(gui, "MASTER_AVWAP_TRADINGVIEW_REPORT_FILE", tradingview_path),
                patch.object(gui, "THETA_PUTS_FILE", theta_path),
                patch.object(gui, "get_tracker_storage_details", return_value=storage_details),
            ):
                output = gui.build_consolidated_gui_output("full", None, avwap_gui)

            self.assertIn("Favorite Setups\nAAPL", output)
            self.assertIn("Near Favorite Zones\nMSFT, TSLA", output)
            self.assertIn("Theta Plays\nNVDA", output)
            self.assertIn("Directional Longs\nAAPL, MSFT", output)
            self.assertIn("Directional Shorts\nTSLA", output)
            self.assertIn("Setup Type Copy Lists\navwap_breakout\nLONG: AAPL, MSFT", output)
            self.assertIn("avwap_breakdown\nSHORT: TSLA", output)


if __name__ == "__main__":
    unittest.main()
