# Scenario Context Source
- ../scenario_context/H5_S2_context.md

# Task Setup
- T1(vib 제거)과 동일한 실험 범위 및 모델/prefix 조건 유지
- 핵심 차이: 절대 VB 대신 마모 진행률(VB_ratio = VB / VB_max)로 정규화하여 학습
- 센서: 전체 6개 (smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle) — H4_S1_T4 동일
- Feature: Delta(24)+Meta(3)=27 (GRU), Raw(24)/Raw+Meta(27)/Raw+Delta+Meta(51) (XGBoost)

# VB 정규화 전략

## 훈련 시
- VB_ratio = VB / case_VB_max  (각 훈련 case의 실제 VB_max 사용)
- 모델 타겟: VB_ratio (0~1 범위의 마모 진행률)

## 테스트 시
- VB_max_predictor: Ridge regression, Meta(3) → VB_max
  - 각 fold마다 train cases의 (DOC, feed, material, VB_max) 쌍으로 학습
- 최종 예측: VB_pred = model_output(VB_ratio) × predicted_VB_max

# Experiment Scope
- `leaderboards/h4/per_case/top5.csv`에서 Case 13, 14의 Top-5 조합만 실험 (T1과 동일 범위)
- VB 정규화 외 모든 조건 동일

## Case 13 (holdout): feature_gru, Delta(24)+Meta(3)=27, prefix
| Rank | Input% | H4 RMSE | T1 RMSE (vib 제거) |
|------|--------|---------|--------------------|
| 1    | 70%    | 0.0969  | 0.1131             |
| 2    | 60%    | 0.1190  | 0.1160             |
| 3    | 50%    | 0.1208  | 0.1227             |
| 4    | 40%    | 0.1256  | 0.1271             |
| 5    | 30%    | 0.1326  | 0.1412             |

## Case 14 (holdout): mixed models, prefix
| Rank | Model       | Features       | Input% | H4 RMSE | T1 RMSE |
|------|-------------|----------------|--------|---------|---------|
| 1    | feature_gru | Delta+Meta     | 80%    | 0.0932  | 0.0831  |
| 2    | feature_gru | Delta+Meta     | 100%   | 0.0998  | 0.0903  |
| 3    | XGBoost     | Raw+Meta       | 30%    | 0.0998  | 0.1719  |
| 4    | XGBoost     | Raw+Delta+Meta | 30%    | 0.0998  | 0.2013  |
| 5    | XGBoost     | Raw            | 30%    | 0.1006  | 0.2649  |

# Comparison Baseline
- H4_S1_T4 (절대 VB, 전체 센서): Case 13 best=0.0969, Case 14 best=0.0932
- T1 (vib 제거): Case 13 best=0.1131, Case 14 best=0.0831

# Expected Output
- 각 조합의 RMSE, H4 및 T1 대비 delta
- VB_max 예측 정확도 (per-fold VB_max prediction RMSE)
- VB 정규화가 Case 13 extrapolation 문제를 완화하는지 확인
