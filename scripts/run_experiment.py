#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.experiment import ExperimentExecution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment_execution.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML/JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate, load dataset, split, and preprocess only.")
    parser.add_argument("--smoke", action="store_true", help="Use a small subset for a fast smoke execution.")
    args = parser.parse_args()
    result = ExperimentExecution(args.config, root=Path.cwd(), smoke=args.smoke, dry_run=args.dry_run).run()
    print(result)


if __name__ == "__main__":
    main()
