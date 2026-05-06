from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml_or_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, data: Any) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_json(path: str | Path, data: Any) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = ensure_parent(path)
    frame.to_csv(path, index=False)


def append_text(path: str | Path, text: str) -> None:
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
