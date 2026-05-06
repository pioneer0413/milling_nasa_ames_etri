"""Manufacturing ML experiment framework."""

from pathlib import Path


def get_framework_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    return "0.1.0"


__version__ = get_framework_version()
