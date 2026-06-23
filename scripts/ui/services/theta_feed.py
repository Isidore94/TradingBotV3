from __future__ import annotations

from pathlib import Path
from typing import Iterable

from project_paths import MASTER_AVWAP_PRIORITY_SETUPS_FILE
from ui.models.theta import ThetaRow


THETA_REPORT_FILE = MASTER_AVWAP_PRIORITY_SETUPS_FILE.with_name("master_avwap_theta_puts.txt")


def load_theta_report_text(path: Path = THETA_REPORT_FILE) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def load_theta_rows(path: Path = THETA_REPORT_FILE) -> list[ThetaRow]:
    text = load_theta_report_text(path)
    if not text:
        return []
    try:
        from master_avwap_lib.theta.reports import extract_theta_rows_from_report
    except Exception:
        return []
    return [ThetaRow.from_mapping(row) for row in extract_theta_rows_from_report(text)]


def load_theta_symbols(path: Path = THETA_REPORT_FILE) -> list[str]:
    text = load_theta_report_text(path)
    if not text:
        return []
    try:
        from master_avwap_lib.theta.reports import extract_theta_symbols_from_report
    except Exception:
        return []
    return extract_theta_symbols_from_report(text)


def format_theta_symbols(rows: Iterable[ThetaRow]) -> str:
    symbols = sorted({row.symbol for row in rows if row.symbol})
    return ", ".join(symbols)
