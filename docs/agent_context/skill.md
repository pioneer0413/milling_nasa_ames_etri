# Experiment Execution Index

실험 실행 결과 디렉터리 목록. 기준 경로: `experiments/executions/H4/`

각 항목은 다음을 포함한다.
- **Task 설명**: 실험의 핵심 설정
- **Script**: 실행 스크립트 경로
- **Context**: Task context 명세 경로
- **Canonical run**: 최신(유효) 실행 디렉터리 경로
- **All runs**: 해당 task의 전체 실행 이력

---

## H4 / S1 — Input Signal Prefix Length Effect on Feature-Based VB Prediction

Scenario context: `docs/agent_context/scenario_context/H4_S1_context.md`

공통 설정:
- LOCV (Leave-One-Case-Out), 15 Cases (Case 6 제외, Case2-Run1·Case12-Run1 excluded)
- Models: Ridge, RandomForest, XGBoost, SVR, MLP, feature_gru (DeltaMetaGRU)
- Signal prefix: 10%~100% (10% step)
- THRESH=1e6, dropna, RF n_est=200, XGB n_est=200/lr=0.05/depth=4, SVR C=10/eps=0.05
- GRU: 2-layer / hidden=32 / dropout=0.1 / epochs=200 / weight_decay=1e-4 / CosineAnnealingLR / grad_clip=1.0
  - ※ T7·T8 GRU는 사용자 수정으로 hidden=256 / 3-layer 적용 중

---

### H4_S1_T0

- **Task**: 초기 탐색 실행 (구 H6 framework 기반, 정식 context 명세 이전)
- **Script**: 해당 없음 (구 framework 사용)
- **Context**: 없음
- **All runs** (14개):
  - `S1/T0/2026-06-01_181233_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_102232_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_105611_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_111638_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_111751_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_112118_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_112431_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_120450_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_131318_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_133021_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_133607_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_134834_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_142301_H4_S1_T0_signal_prefix_length_model_prediction_effect`
  - `S1/T0/2026-06-02_143629_H4_S1_T0_signal_prefix_length_model_prediction_effect`

---

### H4_S1_T1

- **Task**: Feature=Raw(24)+Meta(3), 모든 15 Case, tabular + DeltaMetaGRU
- **Script**: `scripts/run_H4_S1_T1_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T1_context.md`
- **Canonical run**: `S1/T1/2026-06-08_161302_H4_S1_T1_input_signal_length_effect`
- **All runs**:
  - `S1/T1/2026-06-08_151758_H4_S1_T1_input_signal_length_effect` (초기 실행)
  - `S1/T1/2026-06-08_161302_H4_S1_T1_input_signal_length_effect` ← **canonical**

---

### H4_S1_T2

- **Task**: Feature=Raw(24)+Delta(24)+Meta(3)=51, 모든 15 Case, 전 모델
- **Script**: `scripts/run_H4_S1_T2_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T2_context.md`
- **Canonical run**: `S1/T2/2026-06-08_191851_H4_S1_T2_raw_delta_meta_all_models`
- **All runs**:
  - `S1/T2/2026-06-08_154132_H4_S1_T2_raw_delta_meta_all_models`
  - `S1/T2/2026-06-08_161756_H4_S1_T2_raw_delta_meta_all_models`
  - `S1/T2/2026-06-08_191851_H4_S1_T2_raw_delta_meta_all_models` ← **canonical**

---

### H4_S1_T3

- **Task**: Feature=Raw+Meta (T1 동일), LOCV 타겟 Case 제한 = {1,2,5,8,9,12,14,16}
- **Script**: `scripts/run_H4_S1_T3_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T3_context.md`
- **Canonical run**: `S1/T3/2026-06-08_165151_H4_S1_T3_locv_target_restricted`
- **All runs**:
  - `S1/T3/2026-06-08_165151_H4_S1_T3_locv_target_restricted` ← **canonical**

---

### H4_S1_T4

- **Task**: Feature=Delta(24)+Meta(3)=27, Raw 제외, 모든 15 Case
- **Script**: `scripts/run_H4_S1_T4_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T4_context.md`
- **Canonical run**: `S1/T4/2026-06-08_170151_H4_S1_T4_delta_meta_only`
- **All runs**:
  - `S1/T4/2026-06-08_170151_H4_S1_T4_delta_meta_only` ← **canonical**

---

### H4_S1_T5

- **Task**: Feature=Raw+Delta+Meta, Case 6 포함 16 Case (VB=0.0 단일 run)
- **Script**: `scripts/run_H4_S1_T5_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T5_context.md`
- **Canonical run**: `S1/T5/2026-06-08_171530_H4_S1_T5_with_case6`
- **All runs**:
  - `S1/T5/2026-06-08_171530_H4_S1_T5_with_case6` ← **canonical**

---

### H4_S1_T6

- **Task**: Feature=Raw(24)만, Meta·Delta 없음, 모든 15 Case
- **Script**: `scripts/run_H4_S1_T6_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T6_context.md`
- **Canonical run**: `S1/T6/2026-06-08_173840_H4_S1_T6_raw_only`
- **All runs**:
  - `S1/T6/2026-06-08_173840_H4_S1_T6_raw_only` ← **canonical**

---

### H4_S1_T7

- **Task**: Feature=Delta(24)만, Raw·Meta 없음, 모든 15 Case
- **Script**: `scripts/run_H4_S1_T7_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T7_context.md`
- **Note**: GRU 아키텍처가 hidden=256 / 3-layer로 수정됨 (원본: hidden=32 / 2-layer)
- **Canonical run**: `S1/T7/2026-06-08_174208_H4_S1_T7_delta_only`
- **All runs**:
  - `S1/T7/2026-06-08_174208_H4_S1_T7_delta_only` ← **canonical**

---

### H4_S1_T8

- **Task**: Feature=Raw(24)+Delta(24)=48, Meta 없음, 모든 15 Case
- **Script**: `scripts/run_H4_S1_T8_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S1_T8_context.md`
- **Note**: GRU 아키텍처가 hidden=256 / 3-layer로 수정됨 (원본: hidden=32 / 2-layer)
- **Canonical run**: `S1/T8/2026-06-08_174655_H4_S1_T8_raw_delta_no_meta`
- **All runs**:
  - `S1/T8/2026-06-08_174655_H4_S1_T8_raw_delta_no_meta` ← **canonical**

---

## H4 / S2 — Signal Prefix Segment Coverage Visualization

Scenario context: 없음 (시각화 전용)

---

### H4_S2_T1

- **Task**: 신호 prefix 구간별 커버리지 시각화
- **Script**: 해당 없음 (구 framework 사용)
- **Context**: 없음
- **All runs**:
  - `S2/T1/2026-06-02_101900_H4_S2_T1_signal_prefix_segment_coverage_visualization`
  - `S2/T1/2026-06-02_102000_H4_S2_T1_signal_prefix_segment_coverage_visualization`

---

## H4 / S3 — Signal Segment Type Effect on Feature-Based VB Prediction

Scenario context: `docs/agent_context/scenario_context/H4_S3_context.md`

공통 설정:
- LOCV, 15 Cases, 메타데이터 보유 run만 사용 (144/164 runs)
- Heuristic segment: `datasets/metadata/heuristic_sequence_peng2026.csv`
  - 세그먼트 구조: No-load[0:idx_noload_end] / Entry[idx_noload_end:idx_start] / Steady[idx_start:idx_end] / Exit[idx_end:signal_length]
- Feature=Raw(24)+Delta(24)+Meta(3)=51 (tabular), Delta+Meta(27) GRU
- 동일 hyperparameter 설정 (H4_S1과 동일)
- prefix percentage 없음 — 지정 구간 전체 사용

---

### H4_S3_T1 (구 실행)

- **Task**: 구 framework 기반 segment 실험 (S3 정식 context 이전)
- **Script**: `scripts/run_H4_S3_feature_gru_segment_input_prediction_effect.py`
- **All runs**:
  - `S3/T1/2026-06-02_123348_H4_S3_T1_segment_input_model_prediction_effect`
  - `S3/T1/2026-06-02_123545_H4_S3_T1_segment_input_model_prediction_effect`

---

### H4_S3_T1

- **Task**: Segment=Entry+Steady+Exit (No-load 제외), Feature=Raw+Delta+Meta
- **Script**: `scripts/run_H4_S3_T1_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S3_T1_context.md`
- **Canonical run**: `S3/T1/2026-06-08_180301_H4_S3_T1_entry_steady_exit`
- **All runs**:
  - `S3/T1/2026-06-08_180234_H4_S3_T1_entry_steady_exit` (중단 재실행)
  - `S3/T1/2026-06-08_180301_H4_S3_T1_entry_steady_exit` ← **canonical**

---

### H4_S3_T2

- **Task**: Segment=Entry+Steady (Exit 제외), Feature=Raw+Delta+Meta
- **Script**: `scripts/run_H4_S3_T2_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S3_T2_context.md`
- **Canonical run**: `S3/T2/2026-06-08_180447_H4_S3_T2_entry_steady`
- **All runs**:
  - `S3/T2/2026-06-08_180447_H4_S3_T2_entry_steady` ← **canonical**

---

### H4_S3_T3

- **Task**: Segment=Steady only, Feature=Raw+Delta+Meta
- **Script**: `scripts/run_H4_S3_T3_experiment.py`
- **Context**: `docs/agent_context/task_context/H4_S3_T3_context.md`
- **Canonical run**: `S3/T3/2026-06-08_180629_H4_S3_T3_steady`
- **All runs**:
  - `S3/T3/2026-06-08_180629_H4_S3_T3_steady` ← **canonical**

---

## Feature Ablation Summary (H4_S1, @100% prefix, LOCV 15 cases)

| Task | Feature | n_feat | Best model (RMSE) |
|------|---------|--------|-------------------|
| T6   | Raw                | 24 | Ridge 0.157  |
| T7   | Delta              | 24 | XGB   0.112  |
| T8   | Raw+Delta          | 48 | XGB   0.117  |
| T1   | Raw+Meta           | 27 | Ridge 0.126  |
| T4   | Delta+Meta         | 27 | SVR   0.110  |
| T2   | Raw+Delta+Meta     | 51 | GRU   0.115  |

## Segment Ablation Summary (H4_S3, LOCV 15 cases)

| Task | Segment              | Best model (RMSE) |
|------|----------------------|-------------------|
| T1   | Entry+Steady+Exit    | SVR 0.122         |
| T2   | Entry+Steady         | XGB 0.129         |
| T3   | Steady only          | XGB 0.126         |
