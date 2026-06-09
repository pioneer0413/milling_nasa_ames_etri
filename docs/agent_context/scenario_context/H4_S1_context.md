# Experiment Scenario: Input Signal Length Effect on Feature-Based VB Prediction

## 1. Scenario Setup

### 1.1 Purpose

본 실험의 목적은 **마모 예측을 위해 입력 신호를 전부 사용하는 것이 Feature 기반 모델에서 반드시 좋은 선택이 아님을 검증**하는 것이다.

즉, 한 run 내 sensor signal의 입력 길이를 증가시켰을 때 VB 예측 성능이 어떻게 변화하는지 확인하고, 특정 입력 길이 이후 성능 개선이 포화되는지 분석한다.

---

## 2. Research Questions

1. 입력 신호 길이 증가에 따른 VB 예측 성능 변화 추이는 어떠한가?
2. 특정 입력 길이 이후에는 성능 개선이 포화되는가?

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

### 5.1 Sensor Channels

모든 sensor channel을 사용한다.

#### Current

* `smcAC`
* `smcDC`

#### Acoustic Emission

* `AE_spindle`
* `AE_table`

#### Vibration

* `vib_spindle`
* `vib_table`

---

## 6. Feature Construction

### 6.1 Statistics Features

각 sensor signal에 대해 다음 statistics feature를 계산한다.

#### Mean

```python
s.mean()
```

#### Root Mean Square, RMS

```python
np.sqrt((s ** 2).mean())
```

#### Standard Deviation

```python
s.std()
```

#### Peak

```python
np.abs(s).max()
```

---

### 6.2 Delta Feature

Delta feature는 **Run1의 statistics feature와 Run#의 statistics feature 차이**로 정의한다.

즉, 각 Case 내에서 Run1을 기준으로 이후 run의 feature 변화량을 계산한다.

---

### 6.3 Meta Feature

다음 meta feature를 함께 사용한다.

* `DOC`
* `Feed`
* `Material`

---

## 7. Model Setting

### 7.1 Machine Learning Models

다음 machine learning model을 사용한다.

* Ridge
* Random Forest
* XGBoost
* SVR

---

### 7.2 Deep Learning Models

다음 deep learning model을 사용한다.

* MLP
* Feature-sequence model

---

## 8. Preprocessing Setting

### 8.1 Excluded Runs

다음 run은 실험에서 제외한다.

* Case2-Run1
* Case12-Run1

---

### 8.2 VB Missing Value Handling

#### Run1의 VB가 NaN인 경우

* `0.0`으로 채운다.

#### Run1을 제외한 VB가 NaN인 경우

* 선형보간으로 증강한다.
* 보간 설정은 다음과 같다.

```python
method = "index"
limit_area = "inside"
```

---

### 8.3 Scaling

* Standard Scaling을 적용한다.
* LOCV fold마다 training case 기준으로 scaler를 fit하고, 동일한 scaler를 test case에 적용한다.

---

### 8.4 Deep Learning Model 처리 조건

Deep learning model을 사용하는 경우 다음 조건을 만족해야 한다.

* GPU 사용
* 병렬 처리가 가능한 데이터 구조 사용

---

## 9. Sequence Data Preparation

### 9.1 Time Series Structuring

RNN, GRU 계열 모델에서는 데이터를 `case` 단위로 구성한다.

각 `case`는 여러 `run`에 걸친 feature sequence로 표현한다.

즉, 하나의 case는 다음과 같은 sequence 형태를 가진다.

```text
case = [run1_feature, run2_feature, ..., runT_feature]
```

---

### 9.2 Delta + Meta Sequencing

`GRU-DeltaMeta` 모델에서는 각 run의 feature를 Run1 대비 delta feature로 계산하고, 여기에 meta feature를 append한다.

사용하는 meta feature는 다음과 같다.

* `DOC`
* `Feed`
* `Material`

따라서 각 case는 다음 형태의 sequence로 구성된다.

```text
(T, D)
```

여기서:

* `T`: 해당 case의 run 개수
* `D`: delta feature와 meta feature를 포함한 feature dimension

---

### 9.3 Padding

Batch training 시 case별 sequence 길이가 다를 수 있으므로 다음 함수를 사용한다.

```python
torch.nn.utils.rnn.pad_sequence
```

Padding 값은 `NaN`으로 설정한다.

Loss 계산 시에는 `NaN` padding 영역이 학습에 영향을 주지 않도록 masking한다.

---

### 9.4 Per-Case Scaling

Sequence model에서는 각 LOCO-CV fold마다 다음 방식으로 scaling을 수행한다.

1. Training case들의 feature를 기준으로 `StandardScaler`를 fit한다.
2. Fit된 scaler를 training case와 test case 모두에 적용한다.
3. Test case 정보가 scaler fitting 과정에 포함되지 않도록 한다.

---

## 10. Seed Setting

* Single random seed를 사용한다.

---

## 11. Evaluation Metrics

다음 지표를 사용하여 예측 성능을 평가한다.

* RMSE
* R2
* MAE

---

## 12. Aggregation Rule

Leave-One-Case-Out 결과의 평균을 최종 예측 성능으로 간주한다.

즉, 각 case를 test case로 사용했을 때의 성능을 계산한 뒤, 전체 case에 대해 평균 성능을 산출한다.

---