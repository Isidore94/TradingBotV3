import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_setup_explanation_translates_execution_and_evidence():
    from research_explanations import build_research_explanation

    result = build_research_explanation(
        "setup_short_term",
        {
            "side": "LONG",
            "setup_family": "avwap_band_bounce",
            "samples_2d": 18,
            "avg_r_2d": 0.62,
            "median_r_2d": 0.41,
            "win_rate_2d": 0.67,
        },
    )

    assert result["eyebrow"] == "How this setup is executed"
    assert any("bounce close" in step.lower() for step in result["steps"])
    assert any("+0.62R" in item and "18 samples" in item for item in result["evidence"])
    assert "guarantee" in result["caution"]


def test_daytrade_and_forensics_rows_cannot_counterfeit_entry_signals():
    from research_explanations import build_research_explanation

    day = build_research_explanation(
        "daytrade_performance",
        {"direction": "long", "dimension": "bounce_type", "segment": "vwap", "sample_count": 20},
    )
    assert "not a price level" in day["summary"]
    assert any("completed M5" in step for step in day["steps"])

    research = build_research_explanation(
        "move_forensics",
        {"side": "SHORT", "pattern": "weak industry + gap down", "movers_with": 8, "lift": 2.1},
    )
    assert any("Do not enter" in step for step in research["steps"])
    assert "research context" in research["caution"].lower()


def test_plain_english_summary_respects_quality_and_sample_floors():
    from research_explanations import build_plain_english_whats_working

    empty = build_plain_english_whats_working(
        current_rows=[{"symbol": "LOW", "tier": "B"}],
        short_term_rows=[{"setup_family": "tiny", "samples_2d": 2, "avg_r_2d": 4.0}],
    )
    assert "honest result" in empty["bullets"][0]
    assert "minimum sample floor" in empty["bullets"][1]

    summary = build_plain_english_whats_working(
        current_rows=[{"symbol": "AAA", "tier": "A"}],
        short_term_rows=[
            {
                "side": "LONG",
                "setup_family": "avwap_band_bounce",
                "samples_2d": 14,
                "avg_r_2d": 0.55,
                "win_rate_2d": 0.64,
            }
        ],
        recent_rows=[
            {
                "side": "SHORT",
                "setup_family": "ema15_retest",
                "closed_setups": 7,
                "avg_closed_r": 0.8,
                "target_hit_rate": 0.57,
            }
        ],
    )
    text = " ".join(summary["bullets"])
    assert "AAA" in text and "+0.55R" in text and "+0.80R" in text


def test_shared_qt_explanation_view_renders_plain_language():
    try:
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.widgets.research_explanation_view import ResearchExplanationView
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            return
        raise

    view = ResearchExplanationView()
    view.show_row(
        "move_forensics",
        {"side": "LONG", "pattern": "strong group", "lift": 1.8, "movers_with": 12},
    )
    text = view.toPlainText()
    assert "Step by step" in text
    assert "Do not enter" in text
    assert view.isVisible()
