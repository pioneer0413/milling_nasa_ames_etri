from __future__ import annotations

from dataclasses import dataclass

from milling_experiment_framework.domain.shift_types import ShiftType


@dataclass(frozen=True)
class DomainScenario:
    shift_type: ShiftType
    source_domain: str | None = None
    target_domain: str | None = None
