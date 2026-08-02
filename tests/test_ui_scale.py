"""UI scale: the shell's one knob for fitting a 4K desk and a 1680px laptop.

The regression these guard against is not cosmetic. Sizes live in two places -
the stylesheet and explicit widget minimums - and when only the stylesheet
scaled, the laptop desk kept desktop-sized floors and squeezed its control rows
into unreadable stubs instead of wrapping.
"""

import os
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_auto_scale_reads_the_screen_not_the_platform():
    from ui import theme

    # The 4K desk after Windows display scaling, and a 2560x1440 panel.
    assert theme.auto_scale_for(2560, 1440) == 1.00
    # The MacBook's usable workspace (1680x1050 minus menu bar).
    assert theme.auto_scale_for(1680, 954) == 0.85
    # The MacBook's "more space" mode.
    assert theme.auto_scale_for(2048, 1280) == 0.95
    # Anything smaller still gets a floor rather than an unbounded shrink.
    assert theme.auto_scale_for(1366, 768) == 0.80


def test_resolve_scale_handles_auto_percentages_and_junk():
    from ui import theme

    assert theme.resolve_scale("auto", (1680, 954)) == 0.85
    assert theme.resolve_scale("0.90") == 0.90
    # A percentage typed in place of a multiplier still means the same thing.
    assert theme.resolve_scale("90") == 0.90
    assert theme.resolve_scale("nonsense") == 1.0
    # Out-of-range values clamp instead of producing an unusable shell.
    assert theme.resolve_scale("9.0") == theme.MAX_SCALE
    assert theme.resolve_scale("0.1") == theme.MIN_SCALE
    # Auto with no screen to read must not guess small.
    assert theme.resolve_scale("auto") == 1.0


def test_stylesheet_scales_every_size_and_leaves_no_token_unreplaced():
    from ui import theme

    full = theme.build_stylesheet("dark", compact=False, scale=1.0)
    small = theme.build_stylesheet("dark", compact=False, scale=0.85)

    for sheet in (full, small):
        assert "@" not in sheet, "unreplaced @token@ left in the stylesheet"

    def sizes(sheet: str) -> list[int]:
        # 1px hairline borders are deliberately not scaled.
        return sorted({int(value) for value in re.findall(r"\b(\d+)px\b", sheet) if value != "1"})

    assert sizes(small) != sizes(full)
    assert max(sizes(small)) < max(sizes(full))
    # Body type must stay legible, not collapse toward the hairline.
    assert f"font-size: {round(theme.BASE_METRICS['font_body'] * 0.85)}px" in small


def test_px_and_metrics_follow_the_applied_scale():
    from ui import theme

    theme.build_stylesheet("dark", compact=False, scale=1.0)
    assert theme.px(360) == 360
    full_row = theme.metrics()["row_height"]

    theme.build_stylesheet("dark", compact=False, scale=0.85)
    assert theme.px(360) == 306
    assert theme.metrics()["row_height"] < full_row
    # Compact density still composes with the scale.
    assert theme.metrics(compact=True)["row_height"] < theme.metrics()["row_height"]

    theme.build_stylesheet("dark", compact=False, scale=1.0)


def test_arm_bar_control_rows_wrap_instead_of_squeezing():
    """The 1680px desk truncated these to "ew HC" / "d hig" - all present, none
    identifiable. Wrapping trades vertical space for readable labels."""
    _qapp()
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    wide = bar.layout().totalHeightForWidth(1400)
    narrow = bar.layout().totalHeightForWidth(380)
    assert narrow > wide, "narrow arm bar should flow onto more rows, not compress"

    # Every control keeps its own width budget - nothing is sized to zero.
    bar.resize(380, narrow)
    bar.show()
    for button in list(bar.watch_buttons.values()) + list(bar.d1_event_buttons.values()):
        assert button.width() >= button.sizeHint().width(), button.text()


def test_desk_column_floors_shrink_with_the_scale():
    """A laptop desk cannot afford desktop-sized minimum widths: their sum is
    what leaves QSplitter no freedom and jams the layout."""
    _qapp()
    from ui import theme
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services.focus_service import FocusService

    theme.build_stylesheet("dark", compact=False, scale=1.0)
    panel = AlertCenterPanel(FocusService())
    panel.apply_scaled_metrics()
    full_floor = panel.tabs.minimumWidth()

    theme.build_stylesheet("dark", compact=False, scale=0.85)
    panel.apply_scaled_metrics()
    assert panel.tabs.minimumWidth() < full_floor

    theme.build_stylesheet("dark", compact=False, scale=1.0)
