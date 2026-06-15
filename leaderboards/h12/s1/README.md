# H12 S1: Input Ratio Sweep — H11 Top-1 Configs

H11에서 발견한 최적 센서 부분집합 구성에 대해 input_length_pct를 10~100%로 스윕.

## Experiment

- **T1 (Feature-GRU)**: AC+vT+vS (mask=13), Delta+Meta 15-dim, H11_S3 best config
- **T2 (XGBoost)**: AC+vS (mask=9), Delta+Meta 11-dim, H11_S1 best config
- Protocol: LOCV 15 cases, seeds=[0,1,2], observed_vb eval
- Artifact: `experiments/executions/H12/S1/2026-06-15_085640_input_ratio_sweep/`

## Results

| pct | GRU (AC+vT+vS) | XGBoost (AC+vS) |
|----:|---------------:|----------------:|
| 10% | 0.241876 | 0.202658 |
| 20% | 0.148498 | 0.164932 |
| 30% | 0.151524 | 0.143755 |
| 40% | 0.118226 | 0.135538 |
| 50% | 0.103806 | 0.123268 |
| 60% | 0.100902 | 0.118101 |
| 70% | 0.098172 | 0.112350 |
| **80% ★** | **0.081977** | 0.105246 |
| 90% | 0.093271 | **0.103660 ★** |
| 100% | 0.095010 | 0.109247 |

## Key Findings

- **GRU**: 80% is the numerical optimum (H11_S3_T1 baseline confirmed), but this is late for realistic deployment because most of the cutting signal has already been observed. RMSE increases again on both sides of 80%: earlier prefixes lack enough wear progression signal, while 90-100% adds late-stage noise/instability rather than improving the sequence model.
- **XGBoost**: 90% best (-5.1% vs H11_S1_T1 100% baseline). 100% input adds noise.
- Both models show non-monotonic response — sweet spot at 60-90%.

## Files

- `input_ratio_results.csv`: all 20 results (10 pcts × 2 models)
