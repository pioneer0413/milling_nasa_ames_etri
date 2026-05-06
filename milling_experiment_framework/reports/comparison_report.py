from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_comparison_report(path: Path, comparison: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = comparison.to_string(index=False) if not comparison.empty else "No comparable metrics found."
    path.write_text("# Comparison Report\n\n```text\n" + table + "\n```\n", encoding="utf-8")
