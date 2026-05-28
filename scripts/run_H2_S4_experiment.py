#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.h2_s4_process_information_execution import H2S4ProcessInformationExecution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H2.S4 process-information segment-aware VB prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-mode", default="initial", help="initial, extra, all, or comma-separated seeds.")
    parser.add_argument("--yes", action="store_true", help="Skip the pre-run runtime estimate confirmation prompt.")
    args = parser.parse_args()
    result = H2S4ProcessInformationExecution(args.config, root=Path.cwd(), dry_run=args.dry_run, seed_mode=args.seed_mode, assume_yes=args.yes).run()
    print(result)


if __name__ == "__main__":
    main()
