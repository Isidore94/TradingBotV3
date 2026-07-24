import csv
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_weighted_d1_excess_matches_house_one_and_five_day_blend():
    from ui.services.setup_group_context import weighted_d1_excess

    assert weighted_d1_excess(3.0, 7.0, 1.0, 2.0) == 3.95
    assert weighted_d1_excess(-2.0, -4.0, 1.0, 2.0) == -4.95
    assert weighted_d1_excess(None, 1.0, 0.0, 0.0) is None


def test_setup_rows_get_sector_industry_and_group_relative_strength():
    from ui.models.setup import SetupRow
    from ui.services.setup_group_context import enrich_setup_group_context

    row = SetupRow(symbol="AAA", side="LONG")
    enrich_setup_group_context(
        [row],
        context_map={
            "AAA": {
                "sector": "Technology",
                "industry": "Application Software",
                "industry_primary_source": "classification_definition",
                "sector_return_1d_pct": 1.0,
                "sector_return_5d_pct": 2.0,
                "industry_return_1d_pct": 2.0,
                "industry_return_5d_pct": 4.0,
            }
        },
        daily_returns={"AAA": (3.0, 7.0)},
    )

    assert row.sector == "Technology"
    assert row.industry == "Application Software"
    assert row.d1_vs_sector == 3.95
    assert row.d1_vs_industry == 2.3
    assert row.relative_strength_text(row.d1_vs_sector) == "RS +3.95"
    assert row.relative_strength_text(-1.25) == "RW -1.25"


def test_unmapped_classification_report_is_sorted_and_ai_ready(tmp_path):
    from ui.models.setup import SetupRow
    from ui.services.setup_group_context import write_unmapped_setup_classification_report

    rows = [
        SetupRow(
            symbol="ZZZ",
            sector="Technology",
            industry="Odd Software",
            industry_classification_source="raw_classification",
        ),
        SetupRow(
            symbol="AAA",
            industry_classification_source="unmapped",
        ),
        SetupRow(
            symbol="MAPPED",
            sector="Industrials",
            industry="Machinery",
            industry_classification_source="classification_definition",
        ),
        SetupRow(
            symbol="AAA",
            industry_classification_source="unmapped",
        ),
    ]
    path = tmp_path / "unmapped.csv"

    unresolved = write_unmapped_setup_classification_report(rows, path)

    assert [row["symbol"] for row in unresolved] == ["AAA", "ZZZ"]
    with path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written == unresolved
    assert written[0]["reason"] == "missing industry classification"
    assert written[1]["reason"] == "industry needs curated board mapping"


def test_master_setup_columns_carry_group_strength_and_expected_r():
    """Group-strength columns stay; Expected R is back, appended last.

    Expected R was dropped when the D1 group-strength columns landed, purely
    for width - the table could not show both. The compact column profile
    removes that constraint (it hides low-value columns and pins the rest to
    the viewport), and Expected R is the scan's own ranking spine, so it earns
    a slot again. It is APPENDED, never inserted: indices 0/1/2 are the star /
    dislike / symbol click targets that other tests and the panel's column
    handlers pin. Reading order is set with header.moveSection instead.
    """
    from ui.models.setup_table_model import SetupTableModel

    keys = [key for key, _label in SetupTableModel.COLUMNS]
    assert keys[:3] == ["favorite", "dislike", "symbol"]
    assert keys[-6:] == [
        "sector",
        "d1_vs_sector",
        "industry",
        "d1_vs_industry",
        "last_trade_date",
        "expected_r",
    ]
    assert not {"theta", "days_to_earnings"} & set(keys)
