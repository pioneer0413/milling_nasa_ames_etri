from __future__ import annotations

# NASA Ames milling dataset case scope (case 6 excluded — incomplete sensor data)
CASE_SCOPE: list[int] = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
EXCLUDED_CASES: list[int] = [6]

CASE_DOMAINS: dict[str, list[int]] = {f"case_{c}": [c] for c in CASE_SCOPE}
TRAIN_CASE_GROUPS: dict[str, list[int]] = {
    f"train_without_case_{c}": [other for other in CASE_SCOPE if other != c]
    for c in CASE_SCOPE
}
DOMAIN_CASES: dict[str, list[int]] = {**CASE_DOMAINS, **TRAIN_CASE_GROUPS}

# Leave-one-case-out domain-shift scenarios
SHIFT_SCENARIOS: list[tuple[str, str]] = [
    (f"train_without_case_{c}", f"case_{c}") for c in CASE_SCOPE
]

SIGNAL_COLUMNS: list[str] = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

SEGMENT_SETTINGS: list[str] = [
    "full_length",
    "steady",
    "entry",
    "exit",
    "entry_steady",
    "entry_exit",
    "steady_exit",
    "entry_steady_exit",
]

FEATURE_NAMES: list[str] = [
    "mean",
    "std",
    "max",
    "min",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "band_energy",
    "spectral_centroid",
]
