#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one experiment_execution.")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    execution = Path.cwd() / "experiments" / "executions" / args.experiment_id
    metrics = execution / "metrics" / "metrics.json"
    config = execution / "configs" / "resolved_config.yaml"
    if not execution.exists():
        raise SystemExit(f"Experiment not found: {execution}")
    print(json.dumps({"execution_dir": str(execution), "metrics": str(metrics), "resolved_config": str(config)}, indent=2))
    if metrics.exists():
        print(metrics.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
