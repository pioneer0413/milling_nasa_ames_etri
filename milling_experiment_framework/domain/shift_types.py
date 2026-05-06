from __future__ import annotations

from enum import Enum


class ShiftType(str, Enum):
    CONDITION = "condition"
    TOOL = "tool"
    MACHINE = "machine"
    MATERIAL = "material"
    TIME = "time"
