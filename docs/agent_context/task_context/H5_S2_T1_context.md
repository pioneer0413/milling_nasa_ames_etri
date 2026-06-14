# Scenario Context Source
- ../scenario_context/H5_S2_context.md

# Task Setup
- H5_S1_T1 EDA에서 vib_table/vib_spindle이 심화 마모 구간(VB>0.5)에서 VB 상관관계가 붕괴함을 확인
  - delta_vib_table__rms: Case 7(easy) r=0.983 → Case 13(hard) r=0.497
  - delta_vib_spindle__std: Case 8(easy) r=-0.088 → Case 14(hard) r=+0.029
- 해당 센서를 피처에서 완전히 제거하고 smcAC, smcDC, AE_table, AE_spindle 4개 센서만 사용
- Feature: Delta(16)+Meta(3)=19 (GRU), Raw(16)/Raw+Meta(19)/Raw+Delta+Meta(35) (XGBoost)
- 그 외 설정은 H4_S1_T4와 동일

# Experiment Scope
- `leaderboards/h4/per_case/top5.csv`에서 Case 13, 14의 Top-5 조합만 실험 (Case 13/14만 holdout)
- 센서 제거 외 모든 조건 동일하게 유지하여 vib 제거 효과만 분리

## Case 13 (holdout): feature_gru, Delta(16)+Meta(3), prefix
| Rank | Input% | H4 RMSE (vib 포함) |
|------|--------|-------------------|
| 1    | 70%    | 0.0969            |
| 2    | 60%    | 0.1190            |
| 3    | 50%    | 0.1208            |
| 4    | 40%    | 0.1256            |
| 5    | 30%    | 0.1326            |

## Case 14 (holdout): mixed models, prefix
| Rank | Model       | Features       | Input% | H4 RMSE (vib 포함) |
|------|-------------|----------------|--------|-------------------|
| 1    | feature_gru | Delta+Meta     | 80%    | 0.0932            |
| 2    | feature_gru | Delta+Meta     | 100%   | 0.0998            |
| 3    | XGBoost     | Raw+Meta       | 30%    | 0.0998            |
| 4    | XGBoost     | Raw+Delta+Meta | 30%    | 0.0998            |
| 5    | XGBoost     | Raw            | 30%    | 0.1006            |

# Comparison Baseline
- H4_S1_T4 (vib 포함): Case 13 best 0.0969, Case 14 best 0.0932
- T1 목표: vib 제거 후 Case 13/14 RMSE 개선 여부 확인

# Expected Output
- 각 조합의 RMSE, H4 대비 delta
- vib 제거가 hard case에 미치는 효과 정량화
