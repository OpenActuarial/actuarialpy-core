"""Basic output/reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_excel_report(views: dict[str, pd.DataFrame], path: str | Path, *, index: bool = False) -> Path:
    """Write a dictionary of DataFrames to an Excel workbook, one sheet per view."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output) as writer:
        for name, df in views.items():
            sheet = str(name)[:31] or "Sheet1"
            df.to_excel(writer, sheet_name=sheet, index=index)
    return output
