#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.h1_s1_association_execution import H1S1AssociationExecution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1.S1 segment feature to VB association EDA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = H1S1AssociationExecution(args.config, root=Path.cwd(), dry_run=args.dry_run).run()
    print(result)


if __name__ == "__main__":
    main()
