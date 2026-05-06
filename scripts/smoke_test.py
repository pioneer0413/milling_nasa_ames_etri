#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.experiment import ExperimentExecution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a separated MVP smoke test.")
    parser.add_argument("--config", default="configs/experiments/exp001_cnn1d_smoke.yaml")
    args = parser.parse_args()
    root = Path.cwd()
    result = ExperimentExecution(args.config, root=root, smoke=True).run()
    execution_dir = Path(result["execution_dir"])
    out = root / "tests" / "outputs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        execution_dir / "logs" / "run.log": out / "smoke_run.log",
        execution_dir / "metrics" / "metrics.json": out / "smoke_metrics.json",
        execution_dir / "splits" / "split_summary.json": out / "smoke_split_summary.json",
        execution_dir / "preprocessing" / "shape_trace.json": out / "smoke_shape_trace.json",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copyfile(src, dst)
    error_log = execution_dir / "logs" / "error.log"
    if error_log.exists():
        shutil.copyfile(error_log, out / "smoke_error.log")
    print({"smoke_output": str(out), **result})


if __name__ == "__main__":
    main()
