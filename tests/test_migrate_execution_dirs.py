from __future__ import annotations

import csv
from pathlib import Path

from scripts.migrate_execution_dirs import plan_migration, write_log


def test_migration_dry_run_does_not_move_files_and_writes_log(tmp_path: Path):
    root = tmp_path / "experiments" / "executions"
    legacy = root / "2026-05-06_140501_964643_H2_S1_xxx"
    legacy.mkdir(parents=True)
    (legacy / "reports").mkdir()

    rows = plan_migration(root, apply=False, copy=True)
    log_path = write_log(root, rows)

    assert legacy.exists()
    assert len(rows) == 1
    assert rows[0]["status"] == "planned"
    assert rows[0]["action"] == "dry_run_copy"
    assert rows[0]["new_path"].endswith("experiments/executions/H2/S1/2026-05-06_140501_964643_xxx")
    with log_path.open(newline="", encoding="utf-8") as f:
        log_rows = list(csv.DictReader(f))
    assert log_rows[0]["original_path"] == str(legacy)
