# H16 S1: Cutting Segment Source Sensor Visualization Report (2026-06-15)

## 목적

기존 v2 cutting segment는 `smcDC` 기반 reverse-kurtosis exit boundary로 `idx_end`를 보정했다. 이번 분석은 같은 절차를 다른 센서에도 적용해, source sensor 변경이 exit boundary와 segment 정의에 어떤 차이를 만드는지 시각화 중심으로 확인했다.

## 산출물

- 실행 디렉터리: `experiments/analysis/segment_sensor_source/2026-06-15_104332_sensor_source_visualization/`
- 센서별 downstream-ready segment CSV: `datasets/cutting_segment_v2/sensor_source/`
- 생성 스크립트: `scripts/run_H16_S1_segment_sensor_source_visualization.py`

주요 figure:

- `figures/01_exit_fraction_heatmap.png`: run별 exit-start fraction heatmap
- `figures/02_shift_vs_smcDC_boxplot.png`: legacy smcDC 대비 boundary shift
- `figures/03_exit_rate_by_sensor.png`: 센서별 exit 검출 비율
- `figures/05_case_mean_shift_heatmap.png`: case별 평균 shift heatmap
- `figures/04_overlay_case*_run*.png`: Case 1/8/12/13/14 대표 run overlay

## 방법

- 대상 센서: `smcDC`, `smcAC`, `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle`
- 대상 run: `CASE_SCOPE` 15개 case의 166 runs
- base segment: `datasets/cutting_segment/seg_peng2026_steady5.csv`
- detector: source sensor별 rolling kurtosis 변화 + RMS 변화 score를 후방 구간에서 탐색
- calibration: legacy `smcDC` 검출 규모(80 runs)에 맞추기 위해 `threshold_quantile=0.97`, `threshold_mad_multiplier=5.0` 사용
- merged output: `has_exit=True`이면 `idx_end = idx_exit_start`, 아니면 `idx_end = signal_length`

## 핵심 결과

| sensor | has_exit_runs | has_exit_rate | mean_exit_frac | median_exit_frac | mean shift vs smcDC |
|---|---:|---:|---:|---:|---:|
| smcDC | 85/166 | 0.512 | 0.893 | 0.934 | -86 |
| smcAC | 9/166 | 0.054 | 0.990 | 1.000 | +798 |
| vib_table | 77/166 | 0.464 | 0.934 | 1.000 | +291 |
| vib_spindle | 122/166 | 0.735 | 0.881 | 0.910 | -191 |
| AE_table | 141/166 | 0.849 | 0.863 | 0.892 | -362 |
| AE_spindle | 145/166 | 0.873 | 0.864 | 0.876 | -337 |

## 해석

1. **`vib_table`이 가장 보수적인 대체 후보**다. 검출 수가 77/166으로 calibrated `smcDC` 85/166과 가장 비슷하다. 다만 mean shift가 +291 samples로, 같은 run에서도 exit boundary를 더 늦게 잡는 경향이 있다.

2. **`smcAC` 단독 기준은 너무 보수적**이다. 9/166 runs만 exit로 판단되어 대부분 full-length로 남는다. smcDC 대체 기준이라기보다 consensus 보조 신호로 쓰는 편이 낫다.

3. **AE 계열과 `vib_spindle`은 exit를 훨씬 자주, 더 앞쪽으로 잡는다.** AE_spindle 145/166, AE_table 141/166으로 exit rate가 높고 mean shift도 음수다. 이는 실제 exit를 민감하게 잡는 것일 수도 있지만, 후반부 AE/noise burst를 exit로 과검출할 가능성도 크다.

4. **단일 센서 교체보다 consensus 전략이 더 안전하다.** 시각화상 Case 1 run 1처럼 여러 센서가 비슷한 boundary를 잡는 경우도 있지만, `vib_spindle`처럼 no-exit로 빠지는 센서도 있다. 센서별 disagreement를 downstream feature로 쓰거나, `smcDC + vib_table` median/majority vote부터 검증하는 것이 합리적이다.

## 권장 다음 단계

1. H13/H14/H15 script의 `SEG_CSV`를 CLI 인자로 받아, `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle` segment CSV를 동일 조건으로 downstream RMSE 비교한다.
2. 첫 downstream 후보는 `vib_table` 단독과 `smcDC+vib_table` consensus다.
3. AE 계열은 바로 채택하지 말고 overlay figure에서 과검출 case를 먼저 제거/보정한 뒤 실험한다.
