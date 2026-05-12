from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


PATH_SCHEMA_VERSION = "hierarchical_v1"
LEGACY_PATH_SCHEMA_VERSION = "legacy_flat"

_TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?P<timestamp>(?:\d{4}-\d{2}-\d{2}_\d{6}(?:_\d{1,6})?)|(?:\d{8}_\d{6}(?:_\d{1,6})?))[_-](?P<rest>.+)$"
)
_HS_RE = re.compile(r"(?P<hypothesis_id>H\d+)[_.-](?P<scenario_id>S\d+)", re.IGNORECASE)


def slugify_topic(value: str | None) -> str:
    """Return a filesystem-friendly experiment topic while preserving useful case."""
    if value is None:
        return ""
    topic = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    topic = re.sub(r"_+", "_", topic).strip("_")
    return topic


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def parse_experiment_identifier(name: str) -> dict[str, Any]:
    """Parse H/S identifiers, timestamp, and topic from legacy or new names.

    The parser is intentionally non-throwing so it can be used during recursive
    search and migration planning. Strict validation happens in create_execution_dir.
    """
    raw = Path(str(name)).name.strip().rstrip("/")
    timestamp: str | None = None
    rest = raw
    ts_match = _TIMESTAMP_PREFIX_RE.match(raw)
    if ts_match:
        timestamp = ts_match.group("timestamp")
        rest = ts_match.group("rest")

    hs_match = _HS_RE.search(rest)
    if not hs_match:
        return {
            "input": name,
            "name": raw,
            "hypothesis_id": None,
            "scenario_id": None,
            "timestamp": timestamp,
            "experiment_topic": slugify_topic(rest) if rest else None,
            "parse_ok": False,
            "path_schema_version": LEGACY_PATH_SCHEMA_VERSION,
        }

    hypothesis_id = hs_match.group("hypothesis_id").upper()
    scenario_id = hs_match.group("scenario_id").upper()
    topic = rest[hs_match.end() :]
    topic = re.sub(r"^[_.-]+", "", topic)
    topic = slugify_topic(topic)

    return {
        "input": name,
        "name": raw,
        "hypothesis_id": hypothesis_id,
        "scenario_id": scenario_id,
        "timestamp": timestamp,
        "experiment_topic": topic,
        "parse_ok": bool(topic),
        "path_schema_version": PATH_SCHEMA_VERSION,
    }


def build_execution_dir(
    root: str | Path,
    hypothesis_id: str,
    scenario_id: str,
    timestamp: str,
    experiment_topic: str,
) -> str:
    topic = slugify_topic(experiment_topic)
    return str(Path(root) / hypothesis_id.upper() / scenario_id.upper() / f"{timestamp}_{topic}")


def _collision_safe_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{path.name}_v{idx}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a collision-free execution directory for {path}")


def resolve_experiment_path_fields(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.setdefault("experiment", {})
    explicit_name = (
        experiment.get("experiment_id")
        or experiment.get("experiment_name")
        or experiment.get("name")
        or config.get("experiment_id")
        or config.get("experiment_name")
        or config.get("name")
    )
    parsed = parse_experiment_identifier(str(explicit_name)) if explicit_name else {}

    hypothesis_id = experiment.get("hypothesis_id") or parsed.get("hypothesis_id")
    scenario_id = experiment.get("scenario_id") or parsed.get("scenario_id")
    timestamp = experiment.get("timestamp") or parsed.get("timestamp")
    if not timestamp or str(timestamp).lower() == "auto":
        timestamp = current_timestamp()
    experiment_topic = (
        experiment.get("experiment_topic")
        or parsed.get("experiment_topic")
        or experiment.get("topic")
    )
    experiment_topic = slugify_topic(experiment_topic)

    if not hypothesis_id:
        raise ValueError(
            "Cannot infer hypothesis_id from experiment_name. "
            "Please provide experiment.hypothesis_id and experiment.scenario_id."
        )
    if not scenario_id:
        raise ValueError(
            "Cannot infer scenario_id from experiment_name. "
            "Please provide experiment.hypothesis_id and experiment.scenario_id."
        )
    if not experiment_topic:
        raise ValueError("Cannot infer experiment_topic from experiment_name. Please provide experiment.experiment_topic.")

    return {
        "hypothesis_id": str(hypothesis_id).upper(),
        "scenario_id": str(scenario_id).upper(),
        "timestamp": str(timestamp),
        "experiment_topic": experiment_topic,
        "path_schema_version": PATH_SCHEMA_VERSION,
    }


def create_execution_dir(config: dict[str, Any], root: str | Path = "experiments/executions") -> str:
    fields = resolve_experiment_path_fields(config)
    execution_dir = Path(
        build_execution_dir(
            root,
            fields["hypothesis_id"],
            fields["scenario_id"],
            fields["timestamp"],
            fields["experiment_topic"],
        )
    )
    execution_dir = _collision_safe_path(execution_dir)
    execution_dir.mkdir(parents=True, exist_ok=False)

    experiment = config.setdefault("experiment", {})
    experiment.update(fields)
    experiment["execution_dir"] = str(execution_dir)
    experiment["path_schema_version"] = PATH_SCHEMA_VERSION
    if not experiment.get("experiment_id"):
        experiment["experiment_id"] = f"{fields['timestamp']}_{fields['hypothesis_id']}_{fields['scenario_id']}_{fields['experiment_topic']}"
    return str(execution_dir)


def find_experiment_dirs(
    root: str | Path,
    hypothesis_id: str | None = None,
    scenario_id: str | None = None,
    keyword: str | None = None,
    include_legacy: bool = True,
) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    keyword_hs = parse_experiment_identifier(keyword) if keyword else {}
    if keyword and not hypothesis_id and keyword_hs.get("hypothesis_id"):
        hypothesis_id = keyword_hs["hypothesis_id"]
    if keyword and not scenario_id and keyword_hs.get("scenario_id"):
        scenario_id = keyword_hs["scenario_id"]

    hypothesis_id = hypothesis_id.upper() if hypothesis_id else None
    scenario_id = scenario_id.upper() if scenario_id else None
    keyword_text = keyword.lower() if keyword else None
    matches: list[Path] = []

    for candidate in root_path.glob("H*/S*/*"):
        if not candidate.is_dir():
            continue
        h = candidate.parent.parent.name.upper()
        s = candidate.parent.name.upper()
        if hypothesis_id and h != hypothesis_id:
            continue
        if scenario_id and s != scenario_id:
            continue
        haystack = f"{h}_{s}_{candidate.name} {candidate}".lower()
        if keyword_text and keyword_text not in haystack and keyword_text.replace(".", "_").replace("-", "_") not in haystack:
            continue
        matches.append(candidate)

    if include_legacy:
        for candidate in root_path.iterdir():
            if not candidate.is_dir() or re.fullmatch(r"H\d+", candidate.name, flags=re.IGNORECASE):
                continue
            parsed = parse_experiment_identifier(candidate.name)
            if not parsed.get("hypothesis_id") or not parsed.get("scenario_id"):
                continue
            if hypothesis_id and parsed["hypothesis_id"] != hypothesis_id:
                continue
            if scenario_id and parsed["scenario_id"] != scenario_id:
                continue
            haystack = f"{parsed['hypothesis_id']}_{parsed['scenario_id']}_{candidate.name} {candidate}".lower()
            if keyword_text and keyword_text not in haystack and keyword_text.replace(".", "_").replace("-", "_") not in haystack:
                continue
            matches.append(candidate)

    return [str(path) for path in sorted(set(matches))]


def metadata_from_execution_dir(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if re.fullmatch(r"S\d+", path.parent.name, flags=re.IGNORECASE) and re.fullmatch(r"H\d+", path.parent.parent.name, flags=re.IGNORECASE):
        parsed = parse_experiment_identifier(path.name)
        timestamp = parsed.get("timestamp")
        topic = parsed.get("experiment_topic")
        if not timestamp:
            ts_match = _TIMESTAMP_PREFIX_RE.match(path.name)
            if ts_match:
                timestamp = ts_match.group("timestamp")
                topic = slugify_topic(ts_match.group("rest"))
        return {
            "hypothesis_id": path.parent.parent.name.upper(),
            "scenario_id": path.parent.name.upper(),
            "timestamp": timestamp,
            "experiment_topic": topic,
            "execution_dir": str(path),
            "path_schema_version": PATH_SCHEMA_VERSION,
        }
    parsed = parse_experiment_identifier(path.name)
    parsed["execution_dir"] = str(path)
    parsed["path_schema_version"] = LEGACY_PATH_SCHEMA_VERSION
    return parsed


def execution_index_fields(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    return {
        "hypothesis_id": experiment.get("hypothesis_id"),
        "scenario_id": experiment.get("scenario_id"),
        "timestamp": experiment.get("timestamp"),
        "experiment_topic": experiment.get("experiment_topic"),
        "execution_dir": experiment.get("execution_dir"),
        "legacy_execution_dir": experiment.get("legacy_execution_dir"),
        "path_schema_version": experiment.get("path_schema_version", LEGACY_PATH_SCHEMA_VERSION),
    }
