# Scenario Context Source
- ../scenario_context/H4_S6_context.md

# Task Setup
- 이력 run(run_1..run_{t-1})은 full-length(100%), 예측 대상 run_t만 input ratio r% 적용
- run_1(t=1)은 단일 step 시퀀스 [run_1@r%]로 진행
- ratio 적용은 대상 run에만 (이력은 항상 100%)
- per-ratio 모델: r ∈ {10,20,30,40,50,60,70,80,90} 각각 독립 학습/평가
- Sequence-to-one GRU (best H4_S5 config head32), LOCV 15 cases, seed 3회(0,1,2)
- Script: `scripts/run_H4_S6_history_aware_input_ratio.py`

# Expected Output
- ratio별 3-seed LOCV RMSE/R2/MAE 및 ratio vs 성능 곡선
- ratio×seed×case 상세, 예측값 CSV, 리포트
- leader board와는 별도 트랙 (task 정의 상이)
