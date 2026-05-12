from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from milling_experiment_framework.utils.io import write_csv, write_json, write_yaml


PREFIX = "EXPERIMENT_ESTIMATOR"


def write_estimation_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_config": out / f"{PREFIX}_input_config.yaml",
        "resolved_grid": out / f"{PREFIX}_resolved_grid.json",
        "atomic_count_summary": out / f"{PREFIX}_atomic_count_summary.csv",
        "runtime_estimate": out / f"{PREFIX}_runtime_estimate.csv",
        "resource_estimate": out / f"{PREFIX}_resource_estimate.json",
        "warnings": out / f"{PREFIX}_warnings.json",
        "reduced_grid_suggestion": out / f"{PREFIX}_reduced_grid_suggestion.yaml",
        "report": out / f"{PREFIX}_report.md",
    }
    write_yaml(paths["input_config"], result["input_config"])
    write_json(paths["resolved_grid"], result["resolved_grid"])
    write_csv(paths["atomic_count_summary"], pd.DataFrame(result["atomic_count_summary"]))
    write_csv(paths["runtime_estimate"], pd.DataFrame(result["runtime_estimate"]))
    write_json(paths["resource_estimate"], result["resource_estimate"])
    write_json(paths["warnings"], result["warnings"])
    write_yaml(paths["reduced_grid_suggestion"], result["reduced_grid_suggestion"])
    paths["report"].write_text(render_report(result), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def render_report(result: dict[str, Any]) -> str:
    res = result["resource_estimate"]
    warnings = result["warnings"]
    reduced = result["reduced_grid_suggestion"]
    grid = result["resolved_grid"]
    lines = [
        "# Experiment Estimator Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- experiment_name: `{res['experiment_name']}`",
        f"- total atomic executions: `{res['total_atomic_executions']}`",
        f"- total sub-runs: `{res['total_sub_runs']}`",
        f"- estimated wall-clock hours: `{res['estimated_wall_clock_hours']:.3f}`",
        f"- warning level: `{res['warning_level']}`",
        f"- recommended execution plan: `{reduced.get('recommended_phase', 'reduced')}` first",
        "",
        "## 2. Grid Summary",
        "",
    ]
    for axis, values in grid.get("axes", {}).items():
        lines.append(f"- {axis}: {len(values)}")
    lines.extend(["", "## 3. Phase-wise Estimate", ""])
    for row in result["runtime_estimate"]:
        lines.append(
            f"- {row['phase']} / {row['condition_group']} / {row['model']}: "
            f"{row['atomic_executions']} atomic, {row['total_sub_runs']} sub-runs, "
            f"{row['estimated_total_hours']:.3f} h"
        )
    lines.extend(["", "## 4. Model-specific Estimate", ""])
    for row in result["runtime_estimate"]:
        lines.append(f"- {row['model']}: {row['runtime_estimation_method']} ({row['assumption']})")
    lines.extend(["", "## 5. Risk Warnings", ""])
    if warnings:
        for item in warnings:
            lines.append(f"- {item['warning_level']} `{item['warning_code']}`: {item['message']} Suggested: {item['suggested_action']}")
    else:
        lines.append("- No warnings.")
    lines.extend(["", "## 6. Reduced Grid Recommendation", ""])
    lines.append(f"- reason: {reduced.get('reason')}")
    lines.append(f"- estimated_atomic_executions: {reduced.get('estimated_atomic_executions')}")
    lines.append(f"- estimated_runtime_hours: {reduced.get('estimated_runtime_hours')}")
    lines.append(f"- recommended_grid: `{reduced.get('recommended_grid')}`")
    lines.extend(["", "## 7. Assumptions", ""])
    lines.append("- Runtime defaults are heuristic unless config/user overrides are provided.")
    lines.append("- GPU/CPU hours are approximated from model type and requested device assumptions.")
    lines.append("- Disk usage uses checkpoint and row-count heuristics.")
    lines.extend(["", "## 8. CLI Command Examples", ""])
    lines.append("```bash")
    lines.append("python scripts/estimate_experiment.py --config <config.yaml>")
    lines.append("python scripts/estimate_experiment.py --config <config.yaml> --runtime-per-atomic-sec 30")
    lines.append("python scripts/estimate_experiment.py --config H2_S2.yaml --compare-config H3_S1.yaml")
    lines.append("```")
    return "\n".join(lines) + "\n"

