#!/usr/bin/env python3
"""NASA Ames Milling: mill.mat → raw_signal.csv 변환.

mill.mat은 struct array `mill` (1×167)으로, 각 레코드가 공정 메타
(case/run/VB/time/DOC/feed/material)와 6개 센서 시계열(각 9000×1)을 가진다.

H-시리즈 실험 러너가 기대하는 raw_signal.csv 포맷:
  - 컬럼: case, run, + 6 센서(SIGNAL_COLUMNS 순서)
  - 각 센서 셀: "[v1,v2,...]" 형태의 브래킷 문자열
    (러너의 parse_signal: np.fromstring(str(v).strip()[1:-1], sep=","))

센서 값은 float64 round-trip 가능한 최단 repr로 직렬화하여 무손실 유지한다.

사용:
  python3 scripts/convert_mill_mat_to_raw_signal.py \
      [--mat datasets/nasa/mill.mat] [--out datasets/nasa/raw_signal.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]

# milling_experiment_framework.datasets.mill_constants.SIGNAL_COLUMNS 와 동일.
# (프레임워크 패키지는 import 시 matplotlib에 의존하므로 standalone 변환기는 인라인)
SIGNAL_COLUMNS = [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
]

DEFAULT_MAT = ROOT / "datasets/nasa/mill.mat"
DEFAULT_OUT = ROOT / "datasets/nasa/raw_signal.csv"


def to_cell(arr: np.ndarray) -> str:
    """1-D float array → '[v1,v2,...]' (round-trippable shortest repr)."""
    return "[" + ",".join(map(repr, np.asarray(arr).ravel().tolist())) + "]"


def convert(mat_path: Path, out_path: Path) -> int:
    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    mill = mat["mill"].ravel()  # (167,) struct records

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["case", "run", *SIGNAL_COLUMNS]

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rec in mill:
            case = int(np.asarray(rec.case).ravel()[0])
            run = int(np.asarray(rec.run).ravel()[0])
            cells = [to_cell(getattr(rec, s)) for s in SIGNAL_COLUMNS]
            writer.writerow([case, run, *cells])

    return len(mill)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.mat.exists():
        ap.error(f"mat file not found: {args.mat}")

    n = convert(args.mat, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {n} runs → {args.out} ({size_mb:.1f} MB)")
    print(f"columns: case, run, {', '.join(SIGNAL_COLUMNS)}")


if __name__ == "__main__":
    main()
