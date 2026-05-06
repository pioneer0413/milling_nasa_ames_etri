from __future__ import annotations

import pandas as pd


def infer_domains(metadata: pd.DataFrame, domain_column: str = "domain_id") -> list[str]:
    if domain_column not in metadata.columns:
        return []
    return sorted(metadata[domain_column].astype(str).unique())
