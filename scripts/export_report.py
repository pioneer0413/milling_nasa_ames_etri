#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an experiment report.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--format", choices=["md", "html"], default="md")
    args = parser.parse_args()
    report_dir = Path.cwd() / "experiments" / "executions" / args.experiment_id / "reports"
    md = report_dir / "report.md"
    if not md.exists():
        raise SystemExit(f"Report not found: {md}")
    if args.format == "md":
        print({"report": str(md)})
        return
    html = report_dir / "report.html"
    body = md.read_text(encoding="utf-8").replace("\n", "<br>\n")
    html.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    print({"report": str(html)})


if __name__ == "__main__":
    main()
