# Milling Experiment Framework - 에이전트 가이드

## 개요

이 문서는 제조 데이터(밀링/가공) 기반 머신러닝 실험을 위한 프레임워크의 구성을 다른 에이전트가 이해할 수 있도록 작성된 가이드입니다.

## 프로젝트 목적

이 프레임워크는 다음과 같은 목표를 가집니다:

1. **다양한 모델 비교**: feature-based, timeseries-based, hybrid 모델을 동일한 실험 구조에서 비교
2. **신호 처리 조건 관리**: full signal, sliding window, steady cut-only, air cut removal 등 절삭 구간 조건을 실험 변수로 관리
3. **재현 가능한 실험**: config, split, preprocessing, model, metric, prediction, checkpoint, log, report가 experiment_execution 단위로 체계적으로 저장

## 핵심 용어 정의

### dataset_run vs experiment_execution

- **dataset_run**: 데이터셋 내부에서 하나의 가공 실험, 측정 반복, tool pass 등을 나타내는 단위
  - metadata column으로 존재 (dataset_run_id, trial_id, cycle_id 등)
  - split leakage 방지를 위한 group key로 사용됨

- **experiment_execution**: 프레임워크가 실제로 실행한 실험 단위
  - 고유한 experiment_id를 가짐 (예: `2026-05-06_143012_exp001_cnn1d_steady_cut`)
  - 학습, 평가, 로그, checkpoint, report가 생성되는 단위

### input_type과 model_type

- **input_type**: 모델이 입력으로 사용하는 데이터 표현 방식
  - `feature-based`: 통계적 특징량 입력
  - `timeseries-based`: 원본 시계열 신호 입력
  - `hybrid`: 특징량 + 시계열 혼합 입력

- **model_type**: 모델 구현 계열
  - `ML`: SVM, Random Forest, XGBoost 등
  - `DL`: CNN1D, LSTM, GRU, Transformer 등
  - `hybrid`: 혼합 모델

### steady_cut_mode

절삭/가공 시계열 데이터에서 어떤 구간을 실험 입력으로 사용할지 정의:
- `full_signal`: 전체 신호 사용
- `sliding_window`: 고정 길이 슬라이딩 윈도우
- `steady_cut_only`: 안정 절삭 구간만 사용
- `air_cut_removal`: 공기 절삭(무부하) 구간 제거
- `segmentation`: 세그먼트 기반 분할

## 프로젝트 구조

```
milling_experiment_framework/
├── configs/                     # YAML/JSON 설정 파일 저장소
│   ├── datasets/                # 데이터셋별 경로, 센서, 라벨 설정
│   ├── preprocessing/           # 필터링, 정규화, windowing 설정
│   ├── models/                  # 모델별 하이퍼파라미터 설정
│   ├── training/                # epoch, batch size, optimizer 설정
│   ├── evaluation/              # metric, 평가 단위 설정
│   └── experiments/             # 최상위 실험 config 파일들
│
├── datasets/                    # 데이터 로딩 및 표준 schema 관리
│   ├── loaders/                 # 데이터셋별 로딩 로직
│   └── schema.py                # 내부 표준 데이터 포맷 정의
│
├── preprocessing/               # 데이터 전처리 패키지
│   ├── filters/                 # 신호 필터링 모듈
│   ├── normalization/           # 정규화 및 leakage 방지 로직
│   ├── windows/                 # sliding window 생성 모듈
│   ├── steady_cut/              # steady cut 처리 모듈
│   └── features/                # feature 추출 모듈
│
├── splits/                      # train/validation/test 분할 전략
│   ├── random_split.py          # 무작위 분할
│   ├── dataset_run_wise_split.py # dataset_run 단위 분할
│   ├── condition_wise_split.py  # 공정 조건별 분할
│   ├── tool_wise_split.py       # 공구 단위 분할
│   ├── machine_wise_split.py    # 장비별 분할
│   └── split_manager.py         # 분할 전략 관리자
│
├── domain/                      # domain 정의와 domain shift 시나리오
│   ├── domain_definition.py     # source/target domain 정의
│   ├── domain_scenario.py       # 실험 시나리오 정의
│   └── shift_types.py           # domain shift 유형 enum
│
├── models/                      # 모델 구현
│   ├── ml/                      # 머신러닝 모델
│   ├── dl/                      # 딥러닝 모델
│   ├── hybrid/                  # hybrid 모델
│   ├── base.py                  # 공통 인터페이스
│   └── factory.py               # 모델 factory
│
├── training/                    # 학습 루프 관리
│   ├── trainer.py               # 공통 학습 루프
│   ├── losses.py                # loss 정의
│   ├── optimizers.py            # optimizer 생성
│   ├── callbacks.py             # callback 로직
│   └── checkpoint.py            # checkpoint 저장/로드
│
├── evaluation/                  # 성능 metric 계산
│   ├── metrics.py               # 공통 metric 함수
│   ├── evaluator.py             # task별 evaluator
│   ├── classification.py        # 분류 metric
│   ├── regression.py            # 회귀 metric
│   └── anomaly_detection.py     # 이상 탐지 metric
│
├── analysis/                    # 실험 결과 분석
│   ├── execution_analyzer.py    # 단일 실험 분석
│   ├── comparison_analyzer.py   # 여러 실험 비교
│   ├── domain_analyzer.py       # domain별 성능 분석
│   └── error_analyzer.py        # 오차 분석
│
├── visualization/               # 결과 시각화
│   ├── learning_curve.py        # 학습 곡선
│   ├── confusion_matrix.py      # 혼동 행렬
│   ├── regression_plots.py      # 회귀 플롯
│   └── comparison_plots.py      # 비교 플롯
│
├── experiments/                 # 실행된 실험 관리
│   ├── executions/              # experiment_id별 결과 저장
│   ├── comparisons/             # 실험 비교 결과
│   └── experiment.py            # 실험 실행 흐름 조립
│
├── experiment_logging/          # 실험 로깅 (loggingAvoiding stdlib conflict)
│   ├── experiment_logger.py     # 실험 메타정보 기록
│   ├── artifact_logger.py       # 산출물 저장
│   └── environment.py           # 실행 환경 기록
│
├── registry/                    # 이름 기반 호출 registry
│   ├── dataset_registry.py      # 데이터셋 registry
│   ├── model_registry.py        # 모델 registry
│   ├── preprocessor_registry.py # 전처리 registry
│   └── split_registry.py        # 분할 전략 registry
│
├── core/                        # 공통 실행 엔진
│   ├── config.py                # config 로딩/검증
│   ├── pipeline.py              # 전체 pipeline 조립
│   ├── seed.py                  # random seed 관리
│   └── types.py                 # 공통 타입 정의
│
├── reports/                     # 보고서 생성
│   ├── report_generator.py      # 단일 실험 보고서
│   └── comparison_report.py     # 비교 보고서
│
└── utils/                       # 공통 유틸리티
    ├── io.py                    # 파일 입출력
    ├── paths.py                 # 경로 관리
    └── dataframe.py             # DataFrame 처리
```

## 실험 실행 흐름

1. **Config 로딩**: `scripts/run_experiment.py --config configs/experiments/*.yaml`
2. **Config 검증**: 필수 키, input/output 호환성, metadata column 확인
3. **데이터 로딩**: registry를 통해 dataset loader 호출
4. **Split 생성**: strategy에 따라 train/validation/test 분할
5. **전처리**: normalization, windowing, feature extraction 등
6. **모델 학습**: trainer가 epoch별 학습 수행
7. **평가**: evaluator가 metric 계산
8. **결과 저장**: experiment_id별 디렉토리에 모든 산출물 저장

## 표준 실험 출력 구조

```
experiments/executions/{experiment_id}/
├── configs/
│   ├── input_config.yaml        # 원본 config
│   ├── resolved_config.yaml     # 병합된 config
│   └── config_validation.json   # 검증 결과
├── data/
│   ├── dataset_summary.json     # 데이터셋 요약
│   ├── label_distribution.csv   # 라벨 분포
│   └── shape_summary.json       # shape 정보
├── splits/
│   ├── split.csv                # 분할 결과
│   ├── split_summary.json       # 분할 요약
│   └── leakage_check.json       # leakage 검사 결과
├── preprocessing/
│   ├── preprocessing_summary.json
│   ├── normalization_params.json
│   └── shape_trace.json
├── model/
│   ├── model_summary.json
│   ├── model_architecture.txt
│   └── model_config.yaml
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── run.log
│   ├── environment.json
│   └── reproducibility.json
├── metrics/
│   ├── metrics.json             # 최종 metric
│   ├── metrics.csv
│   └── training_history.csv
├── predictions/
│   └── predictions.csv          # 샘플별 예측 결과
├── analysis/
│   ├── analysis_summary.json
│   └── error_summary.csv
├── figures/
│   ├── learning_curve.png
│   ├── confusion_matrix.png
│   └── prediction_vs_target.png
└── reports/
    └── report.md                # 사람용 보고서
```

## Config 시스템

### 기본 Config 구조

```yaml
experiment:
  experiment_id: null           # null이면 자동 생성
  name: experiment_name
  seed: 42

dataset:
  name: dataset_name
  dataset_version: v1

task:
  type: classification          # classification, regression, anomaly_detection
  target_column: label
  num_classes: 3

preprocessing:
  output_type: timeseries       # features, timeseries, hybrid
  cache:
    enabled: false
  steps:
    - name: normalization
      method: zscore
      fit_on: train
    - name: windowing
      window_size: 2048
      stride: 512

split:
  strategy: random              # random, dataset_run_wise, condition_wise, etc.
  validation_ratio: 0.2
  test_ratio: 0.2
  leakage_check: true

model:
  name: cnn1d
  model_type: DL                # ML, DL, hybrid
  input_type: timeseries-based  # feature-based, timeseries-based, hybrid
  params: {}

training:
  epochs: 100
  batch_size: 32
  optimizer: adam
  learning_rate: 0.001

evaluation:
  metrics: [accuracy, f1_macro]

checkpoint:
  enabled: true
  monitor: val_loss
  save_best: true
  save_last: true

logging:
  save_environment: true
  save_git_state: true

report:
  enabled: true
  formats: [md]
```

### Config 호환성 규칙

- `preprocessing.output_type`과 `model.input_type`이 호환되어야 함:
  - `features` → `feature-based`
  - `timeseries` → `timeseries-based`
  - `hybrid` → `hybrid`

## Registry 시스템

프레임워크는 확장성을 위해 registry 패턴을 사용합니다:

```python
# 예시: 새로운 모델 등록
from milling_experiment_framework.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("my_custom_model")
class MyCustomModel:
    def __init__(self, **params):
        ...
```

등록 가능한 registry:
- `DATASET_REGISTRY`: 데이터셋 로더
- `PREPROCESSOR_REGISTRY`: 전처리 단계
- `SPLIT_REGISTRY`: 분할 전략
- `MODEL_REGISTRY`: 모델
- `FEATURE_REGISTRY`: feature extractor

## 주요 기능

### 1. Domain Shift 실험

source/target domain을 정의하고 domain 일반화 성능을 평가:

```yaml
split:
  strategy: condition_wise
  group_key: condition_id
  target_groups: [condition_04]  # target domain
```

### 2. Steady Cut 처리

절삭 구간별로 성능 비교:

```yaml
preprocessing:
  steps:
    - name: steady_cut
      mode: steady_cut_only
      method: threshold
      reference_signal: spindle_current
      threshold: 0.2
```

### 3. Checkpoint 관리

학습 재개를 위한 checkpoint 저장:

```yaml
checkpoint:
  enabled: true
  save_best: true
  save_last: true
  max_keep: 3
  resume_from: null
```

### 4. Leakage 방지

split 생성 후 leakage 검사 자동 수행:

```yaml
split:
  leakage_check: true
```

## CLI 사용법

```bash
# 단일 실험 실행
python scripts/run_experiment.py --config configs/experiments/exp.yaml

# Dry-run (검증만)
python scripts/run_experiment.py --config configs/experiments/exp.yaml --dry-run

# Smoke test (빠른 검증)
python scripts/run_experiment.py --config configs/experiments/exp.yaml --smoke

# 실험 재개
python scripts/resume_experiment.py --experiment-id EXP_ID

# 실험 비교
python scripts/compare_experiments.py --experiment-ids EXP1 EXP2 EXP3

# 실험 검사
python scripts/inspect_experiment.py --experiment-id EXP_ID

# 보고서 내보내기
python scripts/export_report.py --experiment-id EXP_ID
```

## 데이터 Schema

### 필수 metadata columns

- `sample_id`: 샘플 식별자
- `label`: 타겟 라벨
- `dataset_run_id`: 데이터셋 내 실행 ID
- `sequence_index` 또는 `timestamp`: 시계열 인덱스

### 권장 metadata columns

- `domain_id`: domain 식별자
- `condition_id`: 공정 조건
- `machine_id`: 장비 ID
- `tool_id`: 공구 ID
- `material_id`: 재료 ID

### 내부 데이터 포맷

**Timeseries:**
- X: [num_samples, num_channels, sequence_length]
- y: [num_samples]
- metadata: DataFrame (num_samples rows)

**Features:**
- X: [num_samples, num_features]
- y: [num_samples]
- metadata: DataFrame (num_samples rows)

**Hybrid:**
- X_timeseries: [num_samples, num_channels, sequence_length]
- X_features: [num_samples, num_features]
- y: [num_samples]
- metadata: DataFrame (num_samples rows)

## 버전 관리

프레임워크는 Git 기반 버전 관리를 사용합니다:

- 모든 실험은 git commit hash와 dirty state를 기록
- config schema 변경 시 version number 갱신
- patch note를 통해 변경 이력 관리 (`docs/patch_notes/`)

## 테스트

```
tests/
├── smoke/                     # 빠른 동작 검증
├── unit/                      # 단위 테스트
├── integration/               # 통합 테스트
└── outputs/                   # 테스트 결과
```

## 확장 가이드

### 새로운 모델 추가

1. `milling_experiment_framework/models/`에 모델 구현
2. `registry/model_registry.py`에 등록
3. smoke test 추가

### 새로운 전처리 단계 추가

1. `milling_experiment_framework/preprocessing/`에 구현
2. `registry/preprocessor_registry.py`에 등록
3. config schema 업데이트 (필요시)

### 새로운 분할 전략 추가

1. `milling_experiment_framework/splits/`에 구현
2. `registry/split_registry.py`에 등록
3. leakage check 로직 포함

## 참고 문서

- `design_specification.md`: 상세 설계 명세서
- `configs/`: 설정 파일 예시
- `scripts/`: 실행 스크립트
- `docs/`: 추가 문서