#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.s2_sensor_combination_execution import S2SensorCombinationExecution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S2 sensor-combination segment-aware VB prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-mode", default="initial", help="initial, extra, all, or comma-separated seeds.")
    args = parser.parse_args()
    result = S2SensorCombinationExecution(args.config, root=Path.cwd(), dry_run=args.dry_run, seed_mode=args.seed_mode).run()
    print(result)


if __name__ == "__main__":
    main()
