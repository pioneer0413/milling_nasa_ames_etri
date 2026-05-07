#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.domain_shift_execution import DomainShiftExecution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed six-scenario case-pair domain-shift protocol.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(DomainShiftExecution(args.config, root=Path.cwd()).run())


if __name__ == "__main__":
    main()
