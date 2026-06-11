# Scenario Setup

## Purpose

- Cumulative Sum 기반 Feature의 효과 확인

## Research Question

1. Raw Signal을 Cumulative Sum으로 변환한 다음 Feature 추출하여 입력하면 예측 성능 상 이점이 있는가?

## Dataset and Case Setting

1. Dataset
    1. NASA Ames Milling Dataset
2. Case
    1. 전처리 단계에서 필터링 되지 않은 모든 Case

## Test Scenario

- Leave-one-case-out (LOCV)

## Sensor Setting

- All Sensors
    - Current: [`smcAC`, `smcDC`]
    - Acoustic Emission: [`AE_spindle`, `AE_table`]
    - Vibration: [`vib_spindle`, `vib_table`]

## Feature Construction

- Statistics
    - Mean: `s.mean()`
    - Root Mean Square: `np.sqrt(s**2).mean())`
    - Standard Deviation: `s.std()`
    - Peak: `np.abs(s).max()`
- Delta Feature: Run1과 Run#의 Statistics Feature의 차
- Meta Feature: `DOC`, `Feed`, `Material`

## Model Setting

- Machine Learning Models
    - Ridge
    - Random Forest
    - XGBoost
    - SVR
- Deep Learning Models
    - MLP
    - Feature-sequence model

## Preprocessing Setting

- Case2-Run1, Case12-Run1 제외
- Run1이 VB가 NaN인 경우 0.0으로 채움
- Run1 제외 VB가 NaN인 경우 선형보간으로 증강함
    - `method='index', limit_area='inside'`
- Standard Scaling
- Deep Learning Model인 경우 GPU 사용 및 병렬 처리가 가능한 데이터 구조 사용

### Sequence Data Preparation

- **Time Series Structuring**: For Recurrent Neural Networks (RNNs) and Gated Recurrent Units (GRUs), data is structured as sequences of features per `case` across different `run`s.
- **Delta + Meta Sequencing**: For `GRU-DeltaMeta` models, features are calculated as `delta` relative to the first run, and `meta` (DOC, feed, material) are appended to each time step, forming a `(T, D)` sequence per case, where `T` is the number of runs and `D` is the feature dimension.
- **Padding**: When training in batches, `torch.nn.utils.rnn.pad_sequence` is used to handle variable-length sequences by padding shorter sequences with `NaN`s, which are then masked during loss calculation.
- **Per-Case Scaling**: For sequence models, `StandardScaler` is fitted on the training cases and then used to transform features for both training and testing cases within each LOCO-CV fold.

## Seed Setting

- Single random seed

## Evaluation Metrics

- RMSE
- R2
- MAE

## Aggregation Rule

- Leave-one-case-out 결과의 평균을 최종 예측 성능으로 간주

## Expected Output

- LOCV 집계 결과
- 각 Case 별 성능