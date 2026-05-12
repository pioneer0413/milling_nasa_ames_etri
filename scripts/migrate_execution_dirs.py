#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milling_experiment_framework.experiments.execution_path import build_execution_dir, parse_experiment_identifier


LOG_COLUMNS = [
    "original_path",
    "new_path",
    "hypothesis_id",
    "scenario_id",
    "timestamp",
    "experiment_topic",
    "action",
    "status",
    "reason_if_skipped",
]


def collision_safe_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{path.name}_migrated_{idx}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find collision-free destination for {path}")


def plan_migration(root: Path, apply: bool = False, copy: bool = True) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        raise FileNotFoundError(f"Execution root does not exist: {root}")
    action = "copy" if copy else "move"
    if not apply:
        action = f"dry_run_{action}"

    for src in sorted(root.iterdir()):
        if not src.is_dir():
            continue
        if re.fullmatch(r"H\d+", src.name, flags=re.IGNORECASE):
            rows.append(_row(src, "", "", "", "", "", action, "skipped", "hierarchical_hypothesis_directory"))
            continue
        parsed = parse_experiment_identifier(src.name)
        if not parsed.get("hypothesis_id") or not parsed.get("scenario_id") or not parsed.get("timestamp") or not parsed.get("experiment_topic"):
            rows.append(
                _row(
                    src,
                    "",
                    parsed.get("hypothesis_id") or "",
                    parsed.get("scenario_id") or "",
                    parsed.get("timestamp") or "",
                    parsed.get("experiment_topic") or "",
                    action,
                    "skipped",
                    "cannot_parse_hypothesis_scenario_timestamp_topic",
                )
            )
            continue

        dst = Path(
            build_execution_dir(
                root,
                parsed["hypothesis_id"],
                parsed["scenario_id"],
                parsed["timestamp"],
                parsed["experiment_topic"],
            )
        )
        dst = collision_safe_destination(dst)
        status = "planned"
        reason = ""
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copytree(src, dst)
            else:
                shutil.move(str(src), str(dst))
            status = "completed"
        rows.append(
            _row(
                src,
                dst,
                parsed["hypothesis_id"],
                parsed["scenario_id"],
                parsed["timestamp"],
                parsed["experiment_topic"],
                action,
                status,
                reason,
            )
        )
    return rows


def _row(
    original_path: Path,
    new_path: str | Path,
    hypothesis_id: str,
    scenario_id: str,
    timestamp: str,
    experiment_topic: str,
    action: str,
    status: str,
    reason_if_skipped: str,
) -> dict[str, str]:
    return {
        "original_path": str(original_path),
        "new_path": str(new_path),
        "hypothesis_id": hypothesis_id,
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "experiment_topic": experiment_topic,
        "action": action,
        "status": status,
        "reason_if_skipped": reason_if_skipped,
    }


def write_log(root: Path, rows: list[dict[str, str]]) -> Path:
    log_path = root / "migration_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy flat experiment execution directories to H#/S# hierarchy.")
    parser.add_argument("--root", default="experiments/executions")
    parser.add_argument("--dry-run", action="store_true", help="Plan migration without copying or moving files.")
    parser.add_argument("--apply", action="store_true", help="Apply the planned migration.")
    parser.add_argument("--copy", action="store_true", help="Copy legacy directories to the new hierarchy.")
    parser.add_argument("--move", action="store_true", help="Move legacy directories to the new hierarchy.")
    args = parser.parse_args()

    if args.copy and args.move:
        raise ValueError("Use only one of --copy or --move.")
    apply = bool(args.apply and not args.dry_run)
    copy = not args.move
    rows = plan_migration(Path(args.root), apply=apply, copy=copy)
    log_path = write_log(Path(args.root), rows)
    planned = sum(1 for row in rows if row["status"] in {"planned", "completed"})
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    print({"log_path": str(log_path), "planned_or_completed": planned, "skipped": skipped, "applied": apply, "action": "copy" if copy else "move"})


if __name__ == "__main__":
    main()
