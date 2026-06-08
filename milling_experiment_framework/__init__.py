"""Manufacturing ML experiment framework."""

from pathlib import Path

from milling_experiment_framework.visualization.figure_export import enable_dual_figure_export

enable_dual_figure_export()


def get_framework_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    return "0.1.0"


__version__ = get_framework_version()
