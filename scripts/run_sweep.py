#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple configs.")
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()
    print(run_sweep(args.configs, root=Path.cwd()))


if __name__ == "__main__":
    main()
