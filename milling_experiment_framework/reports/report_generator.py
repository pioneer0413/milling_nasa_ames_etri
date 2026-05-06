from __future__ import annotations

from pathlib import Path


def generate_report(path: Path, config: dict, metrics: dict, artifacts: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Experiment Report: {config['experiment']['experiment_id']}

## Overview

- Experiment name: {config['experiment'].get('name')}
- Dataset: {config['dataset'].get('name')} ({config['dataset'].get('dataset_version')})
- Model: {config['model'].get('name')} / {config['model'].get('model_type')} / {config['model'].get('input_type')}
- Split strategy: {config['split'].get('strategy')}
- Steady cut mode: {config.get('steady_cut_mode')}
- Config hash: {config.get('config_hash')}

## Metrics

```json
{metrics}
```

## Artifacts

- Resolved config: {artifacts.get('resolved_config')}
- Split CSV: {artifacts.get('split_csv')}
- Predictions: {artifacts.get('predictions_csv')}
- Checkpoint or model artifact: {artifacts.get('model_artifact')}
- Environment: {artifacts.get('environment_json')}
"""
    path.write_text(text, encoding="utf-8")
