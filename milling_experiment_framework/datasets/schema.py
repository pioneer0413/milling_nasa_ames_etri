from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED_METADATA_COLUMNS = ["sample_id", "label", "dataset_run_id"]
RECOMMENDED_METADATA_COLUMNS = [
    "domain_id",
    "condition_id",
    "machine_id",
    "tool_id",
    "material_id",
    "process_id",
    "speed",
    "feed",
    "depth_of_cut",
    "cutting_state",
    "steady_cut_mode",
    "source_domain_flag",
    "target_domain_flag",
]


def dataset_fingerprint(metadata: pd.DataFrame, dataset_info: dict) -> dict:
    payload = {
        "metadata_columns": list(metadata.columns),
        "num_rows": int(len(metadata)),
        "dataset_info": dataset_info,
        "sample_ids_head": metadata["sample_id"].head(20).astype(str).tolist() if "sample_id" in metadata else [],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return {"fingerprint": digest, **payload}


def path_mtime(path: str | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    return p.stat().st_mtime if p.exists() else None
