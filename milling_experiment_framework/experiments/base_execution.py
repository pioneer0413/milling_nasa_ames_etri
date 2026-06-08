from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from milling_experiment_framework.experiments.execution_path import execution_index_fields
from milling_experiment_framework.experiments.h2_execution_utils import (
    S1RunConfig,
    ordered_h2_models,
    reusable_h2_experiment_id,
)
from milling_experiment_framework.models.h2_regressors import resolve_h2_model_defaults


class BaseH2Execution:
    """Shared scaffold for H2-family leave-one-case-out experiments.

    Subclasses must set class-level attributes:
        _scenario_id       e.g. "S1"
        _experiment_topic  e.g. "segment_setting_effect_on_VB_prediction"
        _index_name        experiment_name written to index.csv
        _index_steady_cut  steady_cut_mode written to index.csv
    and may override any protected method to customise behaviour.
    """

    _scenario_id: str = ""
    _experiment_topic: str = ""
    _index_name: str = "unknown_experiment"
    _index_steady_cut: str = "segmentation"

    def __init__(
        self,
        config_path: str | Path,
        root: str | Path = ".",
        dry_run: bool = False,
        seed_mode: str = "initial",
        assume_yes: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.root = Path(root).resolve()
        self.dry_run = dry_run
        self.seed_mode = seed_mode
        self.assume_yes = assume_yes
        self.skipped: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _read_config(self) -> dict[str, Any]:
        import yaml

        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_seeds(self, config: dict[str, Any]) -> list[int]:
        seed_cfg = config.get("seeds", {})
        seed_list = seed_cfg.get("seed_list", list(range(20)))
        if self.seed_mode == "initial":
            return [int(s) for s in seed_cfg.get("initial_seeds", [0, 1, 2])]
        if self.seed_mode == "extra":
            initial_set = set(seed_cfg.get("initial_seeds", [0, 1, 2]))
            return [int(s) for s in seed_cfg.get("extra_seeds", [s for s in seed_list if s not in initial_set])]
        if self.seed_mode == "all":
            return [int(s) for s in seed_list]
        return [int(s) for s in self.seed_mode.split(",") if s.strip()]

    def _resolve_run_config(self, config: dict[str, Any]) -> Any:
        """Return the run-config dataclass for this experiment.

        Default implementation builds an S1RunConfig; subclasses may override
        if a different dataclass or additional fields are needed.
        """
        seeds = self._resolve_seeds(config)
        model_cfg = config.get("model_defaults", {})
        return S1RunConfig(
            process_info_path=Path(config["dataset"]["process_info_path"]),
            signal_data_path=Path(config["dataset"]["signal_data_path"]),
            heuristic_sequence_path=Path(config["dataset"]["heuristic_sequence_path"]),
            seeds=seeds,
            models=ordered_h2_models(config.get("models", ["random_forest", "mlp"])),
            model_params=resolve_h2_model_defaults(model_cfg),
        )

    def _validate_inputs(self, run_config: Any) -> dict[str, Any]:
        errors = []
        for path in [
            run_config.process_info_path,
            run_config.signal_data_path,
            run_config.heuristic_sequence_path,
        ]:
            if not path.exists():
                errors.append(f"Missing required file: {path}")
        return {"ok": not errors, "errors": errors, "warnings": []}

    def _generate_experiment_id(self) -> str:
        return reusable_h2_experiment_id(
            self.root,
            scenario_id=self._scenario_id,
            topic=self._experiment_topic,
            dry_run=self.dry_run,
            seed_mode=self.seed_mode,
        )

    # ------------------------------------------------------------------
    # Experiment index
    # ------------------------------------------------------------------

    def _update_index(
        self,
        config: dict[str, Any],
        status: str,
        best_metric: float | None,
        error_message: str | None = None,
        error_log: str | None = None,
    ) -> None:
        index_path = self.root / "experiments" / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "experiment_id": config["experiment"]["experiment_id"],
            **execution_index_fields(config),
            "experiment_name": self._index_name,
            "dataset": "mill_processed_enabled",
            "model": "random_forest,mlp",
            "input_type": "feature-based",
            "split_strategy": "leave_one_case_out_no_validation",
            "steady_cut_mode": self._index_steady_cut,
            "status": status,
            "best_metric": best_metric,
            "created_at": config["experiment"].get("created_at"),
            "error_message": error_message,
            "error_log": error_log,
        }
        if index_path.exists():
            frame = pd.read_csv(index_path)
            frame = frame.loc[frame["experiment_id"] != row["experiment_id"]]
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        else:
            frame = pd.DataFrame([row])
        frame.to_csv(index_path, index=False)
