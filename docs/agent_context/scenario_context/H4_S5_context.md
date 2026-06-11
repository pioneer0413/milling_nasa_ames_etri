# Experiment Scenario: feature_gru Hyperparameter Optimization on the Rank-1 Setting

## 1. Scenario Setup

### 1.1 Purpose

본 시나리오의 목적은 **H4 leader board 1위 설정(`feature_gru`, S1_T4)을 기반으로 하이퍼파라미터 탐색 및 모델 튜닝을 수행하여 VB 예측 성능을 추가로 개선**하는 것이다.

기존 1위 결과는 단일 random seed(seed=0) 결과이므로, 본 시나리오에서는 **여러 하이퍼파라미터 부분 집합에 대해 random seed 3회 실행의 평균**으로 설정 간 성능을 비교하여 seed 변동에 강인한 최우수 설정을 가린다.

---

## 2. Research Questions

1. Rank-1 `feature_gru` 설정의 하이퍼파라미터(은닉 크기, 레이어 수, dropout, weight decay, learning rate, epochs 등)를 조정하면 LOCV RMSE를 추가로 낮출 수 있는가?
2. 단일 seed 성능이 아닌 3-seed 평균 기준으로도 baseline 대비 일관되게 개선되는 설정이 존재하는가?

---

## 3. Dataset

### 3.1 Dataset

* NASA Ames Milling Dataset

---

## 4. Base Setting

본 시나리오의 baseline은 H4 leader board 1위 설정이다.

* Source task: **H4_S1_T4** (`scripts/run_H4_S1_T4_experiment.py`)
* Model: `feature_gru` (DeltaMetaGRU)
* Feature: Delta(24) + Meta(3) = 27
* Input strategy: prefix, **80% signal length**
* LOCV, 15 Cases
* Leader board rank-1 (seed=0) RMSE = **0.094549**

---

## 5. Test Scenario

### 5.1 Validation Strategy

* Leave-One-Case-Out Cross-Validation, LOCV
* 하나의 Case를 test case로 남기고 나머지 Case를 training case로 사용한다.

---

## 6. Sensor Setting

모든 sensor channel을 사용한다.

* Current: `smcAC`, `smcDC`
* Acoustic Emission: `AE_spindle`, `AE_table`
* Vibration: `vib_spindle`, `vib_table`

---

## 7. Feature Construction

baseline(S1_T4)과 동일하다.

* Statistics: Mean, RMS, Standard Deviation, Peak
* Delta Feature: Run1 statistics feature와 Run# statistics feature의 차이
* Meta Feature: `DOC`, `Feed`, `Material`
* 입력 feature: Delta(24) + Meta(3) = 27

---

## 8. Model Setting

### 8.1 Target Model

* `feature_gru` (DeltaMetaGRU) 단일 모델만 튜닝한다.

### 8.2 Hyperparameter Search Space

각 하이퍼파라미터를 baseline에서 한두 개씩 변경한 **부분 집합(config) 단위**로 탐색한다.

* `gru_hidden_size`: {64, 128, 192, 256, 384}
* `gru_num_layers`: {1, 2, 3, 4}
* `dropout`: {0.0, 0.1, 0.2, 0.3}
* `weight_decay`: {0, 1e-4, 1e-3, 3e-3}
* `learning_rate`: {5e-4, 1e-3, 2e-3, 3e-3}
* `epochs`: {120, 200, 300, 400}
* `regression_head_hidden_dim`: {16, 32}
* 그리고 위 축을 조합한 다수의 combo config

baseline config: hidden=256, layers=3, dropout=0.1, head=16, lr=1e-3, weight_decay=1e-4, epochs=200, grad_clip=1.0, CosineAnnealingLR.

---

## 9. Preprocessing Setting

baseline(H4_S1)과 동일하다.

### 9.1 Excluded Runs

* Case2-Run1
* Case12-Run1

### 9.2 VB Missing Value Handling

* Run1의 VB가 NaN인 경우 `0.0`으로 채운다.
* Run1을 제외한 VB가 NaN인 경우 선형보간(`method="index"`, `limit_area="inside"`)으로 증강한다.

### 9.3 Sequence / Scaling

* `case` 단위로 (T, D) sequence를 구성한다 (Delta + Meta).
* Target VB는 training case 통계로 표준화 후 예측값을 역변환한다.
* DL model이므로 GPU를 사용한다.

---

## 10. Seed Setting

* **Random seed 3회**(예: 0, 1, 2)를 사용한다.
* 각 config는 3개 seed로 각각 LOCV를 수행한다.

---

## 11. Evaluation Metrics

* RMSE
* R2
* MAE

---

## 12. Aggregation Rule

1. 각 seed에 대해 LOCV mean(RMSE/R2/MAE)을 계산한다.
2. config별로 3-seed LOCV mean을 평균하여 최종 성능으로 간주한다.
3. **3-seed mean RMSE가 가장 낮은 config를 최우수 설정으로 선정**한다.
4. std_rmse(seed 간 표준편차)를 함께 보고하여 안정성을 확인한다.

---

## 13. Expected Output

* config별 3-seed 평균 RMSE/R2/MAE 랭킹 (`analysis/H4_S5_config_ranking.csv`)
* 최우수 설정 및 전체 랭킹 리포트 (`analysis/H4_S5_report.md`)
* config×seed LOCV 집계 (`metrics/H4_S5_seed_metrics.csv`)
* config×seed×case 상세 (`metrics/H4_S5_per_case_metrics.csv`)

---
