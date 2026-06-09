# Scenario Context Source
- ../scenario_context/H4_S1_context.md

## 1. Task Setup

### 1.1 Experiment Type

* 기본 실험

### 1.2 Changes

* 변경 없음

---

## 2. Expected Output

### 2.1 LOCV Aggregated Results by Input Length

입력 신호 길이에 따른 전체 LOCV 집계 결과를 산출한다.

결과에는 최소한 다음 항목이 포함되어야 한다.

* Input length ratio 또는 input length condition
* Mean RMSE across LOCV folds
* Mean R2 across LOCV folds
* Mean MAE across LOCV folds

---

### 2.2 Per-Case Prediction Performance at 100% Input Length

100% 입력 길이를 사용했을 때 각 Case별 예측 성능을 산출한다.

결과에는 최소한 다음 항목이 포함되어야 한다.

* Test Case
* RMSE
* R2
* MAE

---

## 3. Final Deliverables

에이전트는 다음 결과물을 생성해야 한다.

1. 입력 길이에 따른 전체 LOCV 집계 결과
2. 100% 입력 길이에 대한 각 Case별 예측 성능
3. 실험 설정, preprocessing, feature construction, model setting, evaluation rule이 재현 가능하도록 정리된 로그 또는 결과 파일
