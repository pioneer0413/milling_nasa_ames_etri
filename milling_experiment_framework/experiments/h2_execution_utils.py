from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from milling_experiment_framework.core.config import stable_hash
from milling_experiment_framework.experiments.execution_path import current_timestamp
from milling_experiment_framework.models.h2_regressors import canonical_model_name


@dataclass(frozen=True)
class S1RunConfig:
    """Run-configuration for H2-family feature-based VB prediction experiments."""

    process_info_path: Path
    signal_data_path: Path
    heuristic_sequence_path: Path
    seeds: list[int]
    models: list[str]
    model_params: dict[str, dict[str, Any]]


NO_SEED = -1
DETERMINISTIC_H2_MODELS = {"linear_regression", "svr"}
H2_EXECUTION_ORDER = [
    "linear_regression",
    "svr",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "mlp",
    "feature_gru",
]
H2_RUNTIME_PER_ATOMIC_SEC = {
    "linear_regression": 0.02,
    "svr": 0.04,
    "random_forest": 0.08,
    "extra_trees": 0.08,
    "xgboost": 0.10,
    "lightgbm": 0.10,
    "mlp": 0.15,
    "feature_gru": 0.25,
}


def ordered_h2_models(models: Iterable[str]) -> list[str]:
    canonical = []
    for model in models:
        name = canonical_model_name(str(model))
        if name not in canonical:
            canonical.append(name)
    order = {name: idx for idx, name in enumerate(H2_EXECUTION_ORDER)}
    return sorted(canonical, key=lambda name: order.get(name, len(order)))


def model_requires_seed(model_name: str) -> bool:
    return canonical_model_name(model_name) not in DETERMINISTIC_H2_MODELS


def effective_seeds_for_model(model_name: str, seeds: Iterable[int]) -> list[int]:
    if not model_requires_seed(model_name):
        return [NO_SEED]
    return [int(seed) for seed in seeds]


def seed_label(seed: int) -> str:
    return "deterministic" if int(seed) == NO_SEED else str(int(seed))


def model_seed_value(seed: int) -> int:
    return 0 if int(seed) == NO_SEED else int(seed)


def planned_atomic_count(models: Iterable[str], seeds: Iterable[int], condition_count: int, shift_count: int) -> int:
    return sum(len(effective_seeds_for_model(model, seeds)) * int(condition_count) * int(shift_count) for model in ordered_h2_models(models))


def runtime_per_atomic_sec(model_name: str, config: dict | None = None) -> float:
    model = canonical_model_name(model_name)
    runtime_cfg = ((config or {}).get("estimation", {}) or {}).get("runtime_per_atomic_sec", {}) or {}
    if model in runtime_cfg:
        return float(runtime_cfg[model])
    if "__default__" in runtime_cfg:
        return float(runtime_cfg["__default__"])
    return float(H2_RUNTIME_PER_ATOMIC_SEC.get(model, 0.1))


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def model_runtime_estimates(models: Iterable[str], seeds: Iterable[int], condition_count: int, test_count: int, config: dict | None = None) -> list[dict]:
    rows = []
    for model in ordered_h2_models(models):
        atomics = len(effective_seeds_for_model(model, seeds)) * int(condition_count) * int(test_count)
        sec_per_atomic = runtime_per_atomic_sec(model, config)
        rows.append(
            {
                "model": model,
                "atomic_evaluations": atomics,
                "sec_per_atomic": sec_per_atomic,
                "estimated_seconds": atomics * sec_per_atomic,
            }
        )
    return rows


def print_runtime_estimate_and_confirm(
    experiment_label: str,
    models: Iterable[str],
    seeds: Iterable[int],
    condition_count: int,
    test_count: int,
    config: dict | None,
    assume_yes: bool,
) -> None:
    rows = model_runtime_estimates(models, seeds, condition_count, test_count, config)
    total_seconds = sum(row["estimated_seconds"] for row in rows)
    print("\n[H2 execution estimate]", flush=True)
    print(f"- Experiment: {experiment_label}", flush=True)
    print(f"- Cross-test folds: {test_count}", flush=True)
    print(f"- Conditions per model before seeds: {condition_count}", flush=True)
    print("- Model estimates:", flush=True)
    for row in rows:
        print(
            "  "
            f"{row['model']}: {row['atomic_evaluations']} atomic evals, "
            f"~{format_duration(row['estimated_seconds'])} "
            f"({row['sec_per_atomic']:.3f}s/atomic heuristic)",
            flush=True,
        )
    print(f"- Total estimate: ~{format_duration(total_seconds)}", flush=True)
    print("- Deterministic models run once; seeded models use configured seeds.", flush=True)
    print("- Existing matching run signatures will be skipped during execution.", flush=True)
    if assume_yes:
        print("[H2 execution estimate] --yes supplied; continuing without prompt.\n", flush=True)
        return
    try:
        answer = input("Proceed with this H2 execution? [y/N]: ").strip().lower()
    except EOFError as exc:
        raise SystemExit("Execution aborted: confirmation prompt could not read input. Re-run with --yes to skip the prompt.") from exc
    if answer not in {"y", "yes"}:
        raise SystemExit("Execution aborted by user before model execution.")
    print("", flush=True)


@dataclass
class ModelProgressReporter:
    experiment_label: str
    print_interval_sec: float = 10.0

    def __post_init__(self) -> None:
        self.model_name = ""
        self.total = 0
        self.done = 0
        self.skipped = 0
        self.executed = 0
        self.started_at = 0.0
        self.last_print_at = 0.0

    def start_model(self, model_name: str, total: int) -> None:
        self.model_name = model_name
        self.total = max(1, int(total))
        self.done = 0
        self.skipped = 0
        self.executed = 0
        self.started_at = time.monotonic()
        self.last_print_at = 0.0
        print(f"[{self.experiment_label}] model={model_name} started: {total} atomic evaluations", flush=True)

    def step(self, skipped: bool = False) -> None:
        self.done += 1
        if skipped:
            self.skipped += 1
        else:
            self.executed += 1
        now = time.monotonic()
        if self.done >= self.total or now - self.last_print_at >= self.print_interval_sec:
            self.print_progress(now)

    def print_progress(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        elapsed = max(1e-9, now - self.started_at)
        rate = self.done / elapsed
        remaining = max(0, self.total - self.done)
        eta = remaining / rate if rate > 0 else 0.0
        percent = 100.0 * self.done / self.total
        print(
            f"[{self.experiment_label}] model={self.model_name} "
            f"{self.done}/{self.total} ({percent:5.1f}%) "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
            f"executed={self.executed} skipped={self.skipped}",
            flush=True,
        )
        self.last_print_at = now


def latest_h2_experiment_id(root: Path, scenario_id: str, topic: str) -> str | None:
    scenario_dir = root / "experiments" / "executions" / "H2" / scenario_id
    if not scenario_dir.exists():
        return None
    matches = sorted(
        path
        for path in scenario_dir.iterdir()
        if path.is_dir() and topic in path.name and "dry_run" not in path.name
    )
    return matches[-1].name if matches else None


def reusable_h2_experiment_id(root: Path, scenario_id: str, topic: str, dry_run: bool, seed_mode: str) -> str:
    if not dry_run:
        existing = latest_h2_experiment_id(root, scenario_id, topic)
        if existing:
            return existing
    suffix = "dry_run" if dry_run else f"seeds_{seed_mode}"
    return f"{current_timestamp()}_H2_{scenario_id}_{topic}_{suffix}"


def read_existing_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def existing_child_keys(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "child_execution_key" not in frame.columns:
        return set()
    return set(frame["child_execution_key"].astype(str))


def atomic_signature(payload: dict) -> str:
    return stable_hash(payload)


def existing_run_signatures(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    if "run_signature" in frame.columns:
        return set(frame["run_signature"].dropna().astype(str))
    return existing_child_keys(frame)


def concat_existing_new(existing: pd.DataFrame, rows: list[dict] | list[pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not existing.empty:
        frames.append(existing)
    if rows:
        first = rows[0]
        if isinstance(first, pd.DataFrame):
            frames.extend(rows)  # type: ignore[arg-type]
        else:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
