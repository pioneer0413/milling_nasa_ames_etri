# Experiment Scenario: Hard Case Prediction Improvement

## 1. Scenario Setup

### 1.1 Purpose

- H5_S1_T1 EDA에서 밝혀진 원인을 바탕으로 Case 13, 14 예측 성능을 향상시키는 알고리즘 개선을 검증한다
- H5_S1_T1의 핵심 결론은 Case 13, 14가 단순 노이즈가 아니라 다른 Case와 다른 마모 범위, 동일 meta 조건의 상이한 wear trajectory, 진동 센서 응답 붕괴, 고마모 feature-space 이탈을 동시에 갖는 hard case라는 점이다
- 개선 방향은 네 가지다:
  1. 심화 마모 구간에서 신뢰도가 낮은 진동 센서를 제거한 피처 선택
  2. 절대 VB 대신 정규화된 마모 진행률(VB_ratio)을 타겟으로 하는 예측
  3. Case 13, 14처럼 distribution shift가 큰 sample을 감지하는 hard-case detector
  4. Case 특성에 따라 Ridge/SVR/XGBoost/GRU 중 적합한 모델을 선택하거나 가중하는 model selector

### 1.2 Baseline

- H4_S1_T4: Delta(24)+Meta(3)=27 features, 전체 센서, 절대 VB 타겟
- LOCV RMSE @80% prefix: 0.0945 (전체 평균)
- Case 13 best RMSE: 0.0969, Case 14 best RMSE: 0.0931

### 1.3 H5_S1_T1 Findings

#### VB Range Shift

Case 13, 14는 easy cases보다 높은 VB range를 가진다.

| Case | VB_max | Easy cases max VB | Excess |
| --- | ---: | ---: | ---: |
| 13 | 1.53 mm | 0.81 mm (Case 9) | +88% |
| 14 | 1.14 mm | 0.81 mm | +41% |

단, LOCV에서 Case 14 예측 시에는 train set에 Case 13이 포함되므로 순수한 VB range extrapolation 문제가 아니다. VB range shift는 Case 13의 일부 run을 설명하지만 Case 14의 성능 저하는 설명하지 못한다.

#### Meta Feature Aliasing

Case 13/14는 easy case와 동일한 process meta를 공유하지만 wear trajectory가 크게 다르다.

| Easy Case | Hard Case | Shared Meta | Wear Gap |
| --- | --- | --- | --- |
| Case 7: VB_max=0.46 | Case 13: VB_max=1.53 | DOC=0.75, feed=0.25, material=2 | 3.3x |
| Case 8: VB_max=0.62 | Case 14: VB_max=1.14 | DOC=0.75, feed=0.50, material=2 | 1.8x |

따라서 DOC/feed/material만으로는 hard case와 easy case를 구분할 수 없다. Meta feature는 같은 조건이면 유사한 wear limit을 가질 것이라는 잘못된 prior를 모델에 줄 수 있다.

#### Vibration Sensor Breakdown

심화 마모 구간에서 vibration feature의 VB correlation이 붕괴한다.

| Feature | Easy Case | Hard Case |
| --- | ---: | ---: |
| delta_vib_table__rms, Case 7 vs 13 | r=0.983 | r=0.497 |
| delta_vib_table__peak, Case 7 vs 13 | r=0.651 | r=0.077 |
| delta_vib_table__mean, Case 7 vs 13 | r=0.983 | r=0.501 |
| delta_vib_spindle__std, Case 8 vs 14 | r=-0.088 | r=+0.029 |

해석상 vibration sensor는 정상 마모 구간에서는 VB를 잘 추적하지만, 고마모 구간에서는 포화, 에너지 전달 경로 변화, 칩 형성 변화 등으로 인해 signal-wear relationship이 역전되거나 약화될 수 있다.

#### Reliable Current Sensors

smcAC/smcDC delta feature는 전 구간에서 비교적 일관된 단조 관계를 유지한다.

| Sensor feature | Mean \|r\| pattern |
| --- | ---: |
| smcAC rms | 0.974 |
| smcAC std | 0.977 |
| smcDC rms | 0.912 |

Current feature는 Case 13/14를 포함한 고마모 영역에서도 wear progression을 안정적으로 담는 신뢰 센서 후보로 본다.

#### Feature-space Departure

PCA상 hard case의 초기 run은 easy case와 겹치지만, VB가 높아질수록 easy-case distribution 외부로 이동한다. 즉 hard case는 고마모 구간에서 모델이 학습하지 못한 feature-space로 진입한다.

### 1.4 Algorithmic Implication

- Case 13, 14는 단순히 데이터가 지저분한 것이 아니라 다른 Case와 분포가 다른 hard domain이다
- 고정 feature set, 고정 모델, 고정 input length로 전체 Case를 설명하는 접근은 한계가 있다
- H5_S2에서는 sensor reliability, hard-case detection, case-adaptive model selection을 알고리즘 설계의 중심에 둔다

---

## 2. Research Questions

1. 진동 센서(vib_table, vib_spindle)를 제거하면 Case 13, 14의 예측 오차가 감소하는가?
2. VB를 Case별 VB_max로 정규화하면 Cross-case 일반화 성능이 향상되는가?
3. Current 중심 feature가 vibration 포함 feature보다 hard case에서 더 안정적인가?
4. Early-prefix feature로 hard case 또는 out-of-distribution case를 감지할 수 있는가?
5. Case별 best model이 다르다는 H4 leader board 결과를 model selector로 활용하면 100% 이전 RMSE를 낮출 수 있는가?

---

## 3. Dataset

### 3.1 Dataset

* NASA Ames Milling Dataset

---

## 4. Test Scenario

### 4.1 Validation Strategy

* Leave-One-Case-Out Cross-Validation, LOCV

각 실험에서는 하나의 Case를 test case로 남기고, 나머지 Case를 training case로 사용한다.

---

## 5. Sensor Setting

### T2: Vibration 제외 센서

* Current: `smcAC`, `smcDC`
* Acoustic Emission: `AE_spindle`, `AE_table`
* 제외: `vib_spindle`, `vib_table`

### T3: 전체 센서 (T4 동일)

* Current: `smcAC`, `smcDC`
* Acoustic Emission: `AE_spindle`, `AE_table`
* Vibration: `vib_spindle`, `vib_table`

---

## 6. Feature Construction

### 6.1 Statistics Features

Mean, RMS, Std, Peak (H4_S3_context.md 동일)

### 6.2 Delta Feature

Run1의 statistics feature와 Run#의 statistics feature의 차이

### 6.3 Meta Feature

`DOC`, `Feed`, `Material`

---

## 7. Model Setting

### 7.1 Machine Learning Models

Ridge, Random Forest, XGBoost, SVR, MLP

### 7.2 Deep Learning Models

Feature-sequence GRU (DeltaMetaGRU, hidden=256, 3-layer, epochs=200, lr=1e-03) — H4_S1_T4 최신 설정

---

## 8. Preprocessing Setting

H4_S3_context.md §8 동일 (Excluded Runs, VB Missing Value Handling, Scaling)

---

## 9. Evaluation

### 9.1 Primary Metric

전체 LOCV RMSE (15-fold 평균) — H4_S1_T4 대비 비교

### 9.2 Secondary Metric

Case 13, Case 14 개별 RMSE — Hard case 개선 효과 정량화

### 9.3 Diagnostic Metric

- Case 13/14 제외 평균 RMSE
- Case 13/14 전용 평균 RMSE
- 100% 이전 prefix에서의 best RMSE
- Hard-case detector의 precision/recall
- Model selector가 선택한 model/input length 분포

---

## 10. Seed Setting

Single random seed (seed=0)

---

## 11. Input Strategy

Signal prefix 10%~100% (10% step) — H4_S1_T4 동일

---

## 12. Candidate Algorithm Directions

### 12.1 Sensor Reliability Weighted Feature Selection

- Vibration feature를 제거하거나 낮은 가중치로 처리한다
- Current feature(smcAC/smcDC)와 AE feature를 중심으로 Delta+Meta feature를 구성한다
- Sensor group별 모델을 별도 학습한 뒤 validation 성능 또는 uncertainty에 따라 가중한다

### 12.2 Hard-case Detector

- Early prefix에서 PCA distance, nearest-neighbor distance, feature z-score, model disagreement를 계산한다
- Easy distribution에서 멀어지는 Case를 hard case로 표시한다
- Hard case로 판단되면 longer prefix 또는 conservative ensemble로 전환한다

### 12.3 Case-adaptive Model Selector

- H4 leader board에서 Case별 best model/input length가 다르다는 결과를 알고리즘화한다
- Selector는 case id를 직접 사용하지 않고 process meta, early signal feature, sensor reliability score, model disagreement를 입력으로 사용한다
- 후보 모델은 Ridge, SVR, XGBoost, feature_gru로 제한해 과도한 탐색을 피한다
- 평가 시 test case를 제외한 train cases 내부에서 selector를 학습하는 nested LOCV를 사용한다

### 12.4 Target Transformation

- 절대 VB와 함께 VB_ratio 또는 case-relative wear progress를 예측한다
- 단, Case 13/14의 VB_max 자체가 unknown인 early prediction 상황에서는 oracle VB_max를 사용하지 않는 설정과 upper-bound 설정을 분리한다

### 12.5 Expected Outcome

- 전체 평균 RMSE뿐 아니라 Case 13/14의 RMSE가 낮아지는지 확인한다
- 목표는 100% 이전 prefix에서 평균 RMSE 0.08 이하에 근접하거나, hard case를 탐지해 별도 처리하는 알고리즘적 근거를 확보하는 것이다
