from __future__ import annotations

from datetime import datetime
from pathlib import Path

from milling_experiment_framework.utils.io import append_text


class ExperimentLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def info(self, message: str) -> None:
        append_text(self.log_path, f"{datetime.now().isoformat(timespec='seconds')} INFO {message}\n")

    def error(self, message: str) -> None:
        append_text(self.log_path, f"{datetime.now().isoformat(timespec='seconds')} ERROR {message}\n")
