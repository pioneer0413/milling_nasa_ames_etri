#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.analysis.comparison_analyzer import collect_comparison
from milling_experiment_framework.reports.comparison_report import generate_comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment_execution metrics.")
    parser.add_argument("--experiment-ids", nargs="+", required=True)
    parser.add_argument("--metric", default="accuracy")
    args = parser.parse_args()
    root = Path.cwd()
    comparison_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_comparison")
    out = root / "experiments" / "comparisons" / comparison_id
    out.mkdir(parents=True, exist_ok=True)
    frame = collect_comparison(root, args.experiment_ids, args.metric)
    frame.to_csv(out / "comparison.csv", index=False)
    generate_comparison_report(out / "comparison_report.md", frame)
    print({"comparison_id": comparison_id, "path": str(out / "comparison.csv"), "rows": len(frame)})


if __name__ == "__main__":
    main()
