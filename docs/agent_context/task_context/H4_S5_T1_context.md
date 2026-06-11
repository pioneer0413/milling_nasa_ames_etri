# Scenario Context Source
- ../scenario_context/H4_S5_context.md

# Task Setup
- H4 leader board 1위 설정(H4_S1_T4, feature_gru, Delta+Meta, prefix 80%, LOCV 15 cases)을 baseline으로 고정
- baseline GRU 하이퍼파라미터(hidden=256, layers=3, dropout=0.1, head=16, lr=1e-3, weight_decay=1e-4, epochs=200, CosineAnnealingLR, grad_clip=1.0)에서 한두 축을 변경한 여러 config를 탐색
- 각 config를 random seed 3회(0,1,2)로 LOCV 수행
- 목표: 3-seed 평균 LOCV RMSE를 baseline(0.094549) 대비 개선 (목표 RMSE < 0.8 — VB가 mm 단위라 사실상 충족, 실질 목표는 baseline 개선)
- Script: `scripts/run_H4_S5_feature_gru_hpo.py`

# Selection Rule
- config별 3-seed mean RMSE 기준 오름차순 정렬
- 최저 mean RMSE config를 최우수 설정으로 선정, std_rmse로 안정성 확인

# Expected Output
- config별 3-seed 평균 RMSE/R2/MAE 랭킹 CSV 및 마크다운 리포트
- 최우수 설정의 하이퍼파라미터와 baseline 대비 개선폭
