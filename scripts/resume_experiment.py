#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.training.resume import resolve_resume_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a checkpoint path for resume. Full resume training is Phase 2.")
    parser.add_argument("--experiment-id")
    parser.add_argument("--checkpoint", default="last")
    args = parser.parse_args()
    checkpoint = resolve_resume_checkpoint(args.experiment_id, args.checkpoint)
    if checkpoint and not Path(checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    print({"resume_checkpoint": checkpoint, "status": "resolved"})


if __name__ == "__main__":
    main()
