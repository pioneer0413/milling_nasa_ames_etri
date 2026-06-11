# H4 S11: Segment-wise + Full-length Feature Vector 실험 보고서 (2026-06-11)

## 요약

Prefix(input ratio) 전체에서 한 번 뽑던 feature에 더해, prefix 안의 **각 cutting segment별 feature**를 따로 뽑아 모두 concat한 99차원 vector를 검증했다.
**결론**: 새로운 최고 성능은 없었으나(GRU full_only 0.0908@80% 유지), **segment feature는 낮은 input ratio(조기 예측) 구간에서 XGBoost를 일관되게 개선**(30%에서 0.1385→0.1213, −12.4% 상대). 높은 ratio에선 중복·노이즈로 악화.

## Segment 정의 재정의 (cutting_segment_v2)

- **base**: `seg_peng2026_steady5.csv` (idx_noload_end, idx_start, idx_end)
- **override**: base의 `idx_end`(=exit_start)를 `cutting_segment_v2/seg_reverse_kurtosis_smcDC.csv`의 `idx_exit_start`로 덮어씀
- **저장**: `datasets/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv` (164 runs)
- 병합 검증: 164개 모두 매칭, 무효 경계 0건, exit 평균 ~348 샘플 이동, **84/164 runs는 exit 없음**(steady_to_end; H6 S3의 non-exit 케이스와 일치)

## Feature 구성

각 블록 = `extract_features`(mean/rms/std/peak) × 6센서 = 24 delta. Delta = run − 첫 run(같은 블록).
- `full_only` (baseline): Full-prefix 블록 24 + meta 3 = **27**
- `full+seg`: **Full + Entry + Steady + Exit** 4블록 × 24 + meta 3 = **99**
  (No-load 제외 — 공중절삭 노이즈. prefix 내 없는 segment는 0-fill)

## 결과 (3-seed LOCV, prefix sweep 10–100%)

### Best per (model, featset)
| model | featset | best RMSE | @prefix |
|-------|---------|-----------|---------|
| gru | full_only | **0.0908** | 80% |
| gru | full+seg | 0.1056 | 50% |
| xgb | full_only | 0.1136 | 100% |
| xgb | full+seg | 0.1185 | 50% |

### Prefix-matched ΔRMSE (full+seg − full_only, 음수=개선)
| prefix | xgb Δ | gru Δ |
|--------|-------|-------|
| 10% | −0.001 | −0.013 |
| 20% | **−0.014** | +0.048 |
| 30% | **−0.017** ★ | +0.002 |
| 40% | −0.004 | +0.004 |
| 50% | −0.004 | **−0.006** |
| 60% | −0.000 | +0.005 |
| 70% | +0.003 | +0.007 |
| 80% | +0.002 | **+0.090** (불안정) |
| 90% | +0.006 | +0.029 |
| 100% | +0.008 | +0.025 |

## 해석

1. **XGBoost: 깨끗한 교차(crossover) 패턴.** segment feature는 **저 input ratio(20–50%)에서 일관 개선**, 30%에서 최대 −0.017(−12.4% 상대). 고 ratio(70–100%)에선 소폭 악화.
   - **물리적 이유**: prefix가 짧을 때 Full 블록만으로는 "지금 entry인지 초기 steady인지" 구분이 안 됨. Segment 분해가 **절삭 진행 위치를 명시적으로 disambiguate**해 정보 이득. prefix가 길면 Full 블록이 이미 전부 담고 있어, 추가 72차원(상당수 중복/0)은 분산만 키움.

2. **GRU: 대부분 악화, 80%에서 파국적 불안정(+0.090).** 27차원에 튜닝된 고정 하이퍼파라미터(hidden=256)가 0-블록 많은 99차원 입력을 감당 못함(학습 불안정). 50%에서만 미세 개선.

3. **새 챔피언 없음.** 전체 최고는 여전히 GRU full_only 0.0908@80%.

## 시사점 / 다음 단계

- **조기 예측(Goal 2: ≤80% input)에 한정해 segment feature가 XGBoost에 유효.** 특히 30–50% 구간. → prefix-adaptive feature set(저 ratio는 full+seg, 고 ratio는 full_only)이 합리적.
- GRU로 segment feature를 살리려면 **99차원 입력 전용 HPO 재튜닝** 또는 feature selection/차원축소 필요.
- 고 ratio에서 중복을 줄이려면 **Full 블록을 빼고 segment 블록만** 쓰는 변형, 또는 존재하는 segment만 동적으로 사용하는 방식 검토.

산출물: 스크립트 `scripts/run_H4_S11_segment_feature_vector.py` · 실행 `experiments/executions/H4/S11/2026-06-11_135310_segment_feature_vector/` (figures: `01_prefix_curves.png`, `02_delta_by_prefix.png`) · segment `datasets/cutting_segment_v2/seg_peng2026_steady5_exitfix_reverse_kurtosis.csv`
