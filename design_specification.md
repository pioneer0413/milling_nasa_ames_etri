Design Specification

# Your role

- 너는 제조 데이터 기반 머신러닝/딥러닝 실험용 SW 프레임워크를 설계하는 시니어 ML 엔지니어다.

# Objective

- 여러 제조 데이터 세트에 대해 feature-based, timeseries-based, hybrid 모델을 동일한 실험 구조에서 비교할 수 있어야 한다.
- 절삭/가공 시계열 데이터의 경우 full signal, sliding window, steady cut-only, air cut removal 조건을 실험 변수로 관리할 수 있어야 한다.
- 실험 결과는 재현 가능해야 하며, config, split, preprocessing, model, metric, prediction, checkpoint, log, report가 experiment_execution 단위로 저장되어야 한다.
- dataset 내부의 run 개념과 실험 실행 단위의 run 개념을 혼동하지 않도록 용어를 명확히 분리해야 한다.

# Terminology

dataset_run:
- 데이터셋 내부에서 하나의 가공 실험, 측정 반복, tool pass, machining cycle 등을 나타내는 단위
- metadata column으로 존재할 수 있음
- 예: dataset_run_id, trial_id, cycle_id, pass_id
- split leakage 방지를 위한 group key로 사용될 수 있음

experiment_execution:
- 하나의 config를 기반으로 프레임워크가 실제로 실행한 실험 실행 단위
- 학습, 평가, 로그, checkpoint, report가 생성되는 단위
- 하나의 experiment_execution은 고유한 experiment_id를 가짐
- 예: experiment_id = 2026-05-06_143012_exp001_cnn1d_steady_cut

experiment_id:
- 하나의 experiment_execution을 식별하는 고유 ID
- 실험 결과 저장 디렉터리명, index.csv, report, log, comparison에서 공통으로 사용
- dataset 내부의 run_id와 혼동되지 않도록 run_id라는 명칭은 사용하지 않음

input_type:
- 모델이 입력으로 사용하는 데이터 표현 방식
- feature-based, timeseries-based, hybrid 중 하나

model_type:
- 모델 구현 계열
- ML, DL, hybrid 중 하나

steady_cut_mode:
- 절삭/가공 시계열 데이터에서 어떤 구간을 실험 입력으로 사용할지 정의하는 실험 변수
- full_signal, sliding_window, steady_cut_only, air_cut_removal, segmentation 중 하나

# Entire structure

milling_experiment_framework/
├── configs/                     # 실험, 데이터셋, 모델, 전처리, 학습 조건을 YAML/JSON으로 정의하는 설정 파일 저장소
│   ├── datasets/                # 데이터셋별 경로, 센서, 라벨, 메타데이터 설정
│   ├── preprocessing/           # 필터링, 정규화, windowing, steady cut 등 전처리 설정
│   ├── models/                  # ML/DL/hybrid 모델별 하이퍼파라미터 설정
│   ├── training/                # epoch, batch size, optimizer, learning rate 등 학습 설정
│   ├── evaluation/              # task별 metric, 평가 단위, 리포트 출력 설정
│   └── experiments/             # 하나의 experiment_execution을 정의하는 최상위 config 파일들
│
├── datasets/                    # raw/processed 데이터와 데이터 로더, 표준 schema를 관리하는 패키지
│   ├── raw/                     # 원본 제조 데이터가 저장되는 위치
│   ├── processed/               # 전처리 또는 변환이 끝난 데이터가 저장되는 위치
│   ├── metadata/                # dataset_run, tool, condition, machine, label 등 실험 메타데이터 저장소
│   ├── loaders/                 # 데이터셋별 로딩 로직 구현 위치
│   └── schema.py                # 내부 표준 데이터 포맷과 metadata schema 정의
│
├── preprocessing/               # 모델 입력으로 사용하기 전 데이터 변환 과정을 담당하는 패키지
│   ├── filters/                 # bandpass, lowpass 등 신호 필터링 모듈
│   ├── normalization/           # z-score, min-max 등 정규화 및 leakage 방지용 fit/transform 로직
│   ├── windows/                 # 고정 길이 sliding window 생성 모듈
│   ├── steady_cut/              # full signal, steady cut-only, air cut removal, segmentation 처리 모듈
│   ├── features/                # time/frequency/time-frequency domain feature 추출 모듈
│   └── pipeline.py              # 여러 전처리 단계를 순서대로 실행하는 preprocessing pipeline
│
├── splits/                      # train/validation/test 분할 전략과 domain shift split을 관리하는 패키지
│   ├── random_split.py          # 동일 분포 가정의 무작위 분할 구현
│   ├── dataset_run_wise_split.py # dataset_run 단위 leakage를 막기 위한 분할 구현
│   ├── condition_wise_split.py  # 공정 조건 변화에 따른 condition-wise 분할 구현
│   ├── tool_wise_split.py       # 공구 단위 일반화 성능 평가를 위한 tool-wise 분할 구현
│   ├── machine_wise_split.py    # 장비 변화 일반화를 평가하기 위한 machine-wise 분할 구현
│   ├── time_based_split.py      # 시간 순서 또는 drift를 반영하는 time-based 분할 구현
│   └── split_manager.py         # config에 따라 적절한 split 전략을 선택하고 split.csv를 생성하는 관리자
│
├── domain/                      # source/target domain 정의와 domain shift 시나리오를 관리하는 패키지
│   ├── domain_definition.py     # domain_id, source domain, target domain 정의 로직
│   ├── domain_scenario.py       # condition/tool/machine/material shift 등 실험 시나리오 정의
│   └── shift_types.py           # domain shift 유형을 enum 또는 상수로 관리
│
├── models/                      # feature-based, timeseries-based, hybrid 모델 구현을 관리하는 패키지
│   ├── ml/                      # SVM, Random Forest, XGBoost 등 머신러닝 모델 구현
│   ├── dl/                      # CNN1D, LSTM, TCN, Transformer 등 딥러닝 모델 구현
│   ├── hybrid/                  # feature와 timeseries representation을 함께 사용하는 hybrid 모델 구현
│   ├── base.py                  # 모든 모델이 따라야 할 공통 인터페이스 정의
│   └── factory.py               # config의 model.name에 따라 모델 객체를 생성하는 factory
│
├── training/                    # 모델 학습 루프, loss, optimizer, checkpoint를 관리하는 패키지
│   ├── trainer.py               # 공통 학습 루프와 train/validation 절차 구현
│   ├── losses.py                # classification, regression, contrastive 등 loss 정의
│   ├── optimizers.py            # optimizer와 scheduler 생성 로직
│   ├── callbacks.py             # early stopping, metric monitoring 등 callback 로직
│   ├── checkpoint.py            # best model 및 epoch별 checkpoint 저장/로드 로직
│   ├── resume.py                # checkpoint 기반 학습 재개 로직
│   └── history.py               # epoch별 학습 기록 관리
│
├── evaluation/                  # 모델 성능 metric 계산을 담당하는 패키지
│   ├── metrics.py               # 공통 metric 계산 함수 모음
│   ├── evaluator.py             # task type에 따라 평가 절차를 실행하는 evaluator
│   ├── classification.py        # accuracy, F1, AUROC 등 분류 metric 계산
│   ├── regression.py            # MAE, RMSE, R² 등 회귀 metric 계산
│   ├── anomaly_detection.py     # AUROC, AUPRC, false alarm rate 등 이상 탐지 metric 계산
│   └── domain_eval.py           # source/target/domain별 metric 계산
│
├── analysis/                    # 실험 결과 분석, 비교, 진단을 담당하는 패키지
│   ├── execution_analyzer.py    # 단일 experiment_execution의 metric, prediction, split, config 분석
│   ├── comparison_analyzer.py   # 여러 experiment_execution 비교 분석
│   ├── domain_analyzer.py       # domain/source/target별 성능 차이 분석
│   ├── steady_cut_analyzer.py   # steady_cut_mode별 성능 차이 분석
│   ├── error_analyzer.py        # 오분류, residual, high-error sample 분석
│   ├── feature_analyzer.py      # feature importance 및 feature distribution 분석
│   └── summary_tables.py        # 보고서용 요약 테이블 생성
│
├── visualization/               # 실험 결과 시각화를 담당하는 패키지
│   ├── learning_curve.py        # train/validation loss 및 metric curve 시각화
│   ├── confusion_matrix.py      # confusion matrix 시각화
│   ├── regression_plots.py      # prediction-vs-target, residual plot 생성
│   ├── domain_plots.py          # domain별 성능 시각화
│   ├── steady_cut_plots.py      # steady cut 조건별 성능 시각화
│   ├── feature_plots.py         # feature importance 및 feature distribution 시각화
│   ├── comparison_plots.py      # 여러 experiment_execution 비교 시각화
│   └── plot_manager.py          # config에 따라 필요한 figure를 생성하는 관리자
│
├── experiments/                 # 실제 실행된 experiment_execution과 비교 결과를 관리하는 공간
│   ├── executions/              # experiment_id별 config, metric, prediction, checkpoint, report 저장 위치
│   ├── comparisons/             # 여러 experiment_execution 비교 결과 저장 위치
│   ├── index.csv                # 전체 experiment_execution 인덱스
│   ├── sweep.py                 # 여러 config 조합을 반복 실행하는 sweep 로직
│   └── experiment.py            # 하나의 experiment_execution 전체 실행 흐름을 조립하는 객체
│
├── logging/                     # 실험 중 생성되는 설정, metric, artifact, 로그 저장 기능을 구현하는 패키지
│   ├── experiment_logger.py     # config, seed, git hash, 실행 환경 등 실험 메타정보 기록
│   ├── artifact_logger.py       # model, split.csv, predictions.csv, figure 등 산출물 저장
│   ├── environment.py           # Python, OS, package, CUDA, GPU 등 실행 환경 기록
│   ├── shape_tracer.py          # pipeline 및 model shape trace 기록
│   └── result_writer.py         # metrics.json, training_history.csv 등 결과 파일 작성 로직
│
├── reports/                     # 분석/시각화 결과를 사람이 읽을 수 있는 보고서로 조립하는 패키지
│   ├── report_generator.py      # 단일 experiment_execution 보고서 생성
│   ├── comparison_report.py     # 여러 experiment_execution 비교 보고서 생성
│   └── templates/               # 보고서 템플릿 파일 저장 위치
│
├── registry/                    # dataset/model/preprocessor/split/feature를 이름 기반으로 호출하기 위한 registry 패키지
│   ├── dataset_registry.py      # dataset.name으로 데이터셋 로더를 찾는 registry
│   ├── model_registry.py        # model.name으로 모델 클래스를 찾는 registry
│   ├── preprocessor_registry.py # preprocessing step 이름으로 전처리 클래스를 찾는 registry
│   ├── split_registry.py        # split.strategy 이름으로 split 클래스를 찾는 registry
│   └── feature_registry.py      # feature extractor 이름으로 feature 추출기를 찾는 registry
│
├── core/                        # 프로젝트 전반에서 공유되는 실행 엔진, config, 타입, seed 관리 패키지
│   ├── config.py                # YAML/JSON config 로딩, 병합, 검증 로직
│   ├── pipeline.py              # dataset→split→preprocess→train→evaluate 전체 pipeline 조립
│   ├── seed.py                  # random seed 고정 및 재현성 설정
│   └── types.py                 # Sample, DatasetOutput, SplitOutput 등 공통 타입 정의
│
├── scripts/                     # CLI로 실행되는 진입점 스크립트 모음
│   ├── run_experiment.py        # 단일 experiment config를 실행하는 CLI
│   ├── run_sweep.py             # 여러 실험 조합을 batch 실행하는 CLI
│   ├── resume_experiment.py     # checkpoint에서 학습을 재개하는 CLI
│   ├── compare_experiments.py   # 여러 experiment_execution 결과를 비교하는 CLI
│   ├── inspect_experiment.py    # 특정 experiment_id의 config/metric/log 요약 조회 CLI
│   ├── export_report.py         # 특정 experiment_id의 report export CLI
│   ├── prepare_dataset.py       # raw 데이터를 내부 표준 포맷으로 변환하는 CLI
│   └── smoke_test.py            # 전체 pipeline이 최소 동작하는지 확인하는 smoke test 실행 CLI
│
├── tests/                       # unit/integration/smoke test 코드와 테스트 산출물을 관리하는 디렉터리
│   ├── smoke/                   # 데이터 로딩, 전처리, 모델 forward, 1-batch 학습 등 smoke test 코드
│   ├── unit/                    # 개별 함수와 클래스 단위 테스트 코드
│   ├── integration/             # 여러 모듈이 연결된 end-to-end 테스트 코드
│   └── outputs/                 # smoke/unit/integration test 결과 파일 저장 위치
│
├── cache/                       # 전처리, windowing, feature extraction 등 재사용 가능한 중간 산출물 캐시
│   ├── preprocessing/           # preprocessing pipeline 결과 캐시
│   ├── windows/                 # windowing 결과 캐시
│   └── features/                # feature extraction 결과 캐시
│
├── docs/                        # 프레임워크 문서, patch note, schema 변경 이력을 관리하는 문서 디렉터리
│   ├── patch_notes/             # 버전별 상세 patch note markdown 저장 위치
│   ├── schemas/                 # config/output/dataset schema 문서 저장 위치
│   ├── usage/                   # CLI 사용법, 실험 실행 예제 문서 저장 위치
│   └── architecture.md          # 전체 구조와 설계 원칙 문서
│
├── CHANGELOG.md                 # 최신 변경 사항을 요약하는 changelog
├── VERSION                      # 현재 framework version을 기록하는 파일
│
└── utils/                       # 여러 패키지에서 공통으로 사용하는 보조 함수 모음
    ├── io.py                    # 파일 읽기/쓰기, JSON/YAML/CSV 저장 유틸
    ├── paths.py                 # 프로젝트 경로 생성 및 관리 유틸
    ├── dataframe.py             # metadata 및 prediction DataFrame 처리 유틸
    └── plot_utils.py            # 공통 plot 스타일, figure 저장 유틸

# Responsibility boundary

evaluation/
- metric 계산을 담당한다.
- accuracy, F1, MAE, RMSE, AUROC 등 정량 평가값을 산출한다.

analysis/
- metric, prediction, split, config를 해석하고 비교한다.
- 어떤 domain, condition, steady_cut_mode에서 성능이 달라지는지 분석한다.
- error sample, residual, misclassification 원인을 요약한다.

visualization/
- 분석 결과와 metric을 figure로 변환한다.
- learning curve, confusion matrix, residual plot, domain performance plot 등을 생성한다.

reports/
- metric, analysis, visualization 결과를 사람이 읽을 수 있는 report.md/html/pdf로 조립한다.

logging/
- 실행 중 발생한 로그, 환경 정보, shape trace, 재현성 정보를 저장한다.

experiments/executions/
- 실제 experiment_execution별 산출물이 저장되는 위치다.
- logging, reports, evaluation, visualization 패키지는 저장 기능을 구현하지만, 최종 산출물은 experiment_id 단위로 이 위치에 모은다.

# Standard experiment output directory

experiments/executions/{experiment_id}/
├── configs/
│   ├── input_config.yaml
│   ├── resolved_config.yaml
│   └── config_validation.json
├── data/
│   ├── dataset_summary.json
│   ├── dataset_fingerprint.json
│   ├── label_distribution.csv
│   ├── metadata_summary.json
│   └── shape_summary.json
├── splits/
│   ├── split.csv
│   ├── split_summary.json
│   └── leakage_check.json
├── preprocessing/
│   ├── preprocessing_summary.json
│   ├── normalization_params.json
│   ├── feature_summary.json
│   ├── steady_cut_summary.json
│   └── shape_trace.json
├── model/
│   ├── model_summary.json
│   ├── model_architecture.txt
│   ├── model_config.yaml
│   └── feature_importance.csv
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── epoch_050.pt
├── artifacts/
│   ├── model.joblib
│   └── additional_artifacts
├── logs/
│   ├── run.log
│   ├── error.log
│   ├── environment.json
│   └── reproducibility.json
├── metrics/
│   ├── metrics.json
│   ├── metrics.csv
│   ├── domain_metrics.csv
│   ├── training_history.csv
│   └── threshold_summary.json
├── predictions/
│   ├── predictions.csv
│   └── prediction_errors.csv
├── analysis/
│   ├── analysis_summary.json
│   ├── error_summary.csv
│   ├── domain_analysis.csv
│   ├── steady_cut_analysis.csv
│   └── summary_tables.csv
├── figures/
│   ├── learning_curve.png
│   ├── confusion_matrix.png
│   ├── prediction_vs_target.png
│   ├── residual_plot.png
│   ├── domain_performance.png
│   ├── steady_cut_performance.png
│   └── feature_importance.png
└── reports/
    ├── report.md
    ├── report.html
    └── report.pdf

# Dataset schema

Required metadata columns:
- sample_id
- label
- dataset_run_id
- sequence_index 또는 timestamp

Recommended metadata columns:
- domain_id
- condition_id
- machine_id
- tool_id
- material_id
- process_id
- speed
- feed
- depth_of_cut
- cutting_state
- steady_cut_mode
- source_domain_flag
- target_domain_flag

Internal data format for timeseries:
- X: [num_samples, num_channels, sequence_length]
- y: [num_samples]
- metadata: DataFrame with num_samples rows

Internal data format for features:
- X: [num_samples, num_features]
- y: [num_samples]
- metadata: DataFrame with num_samples rows

Internal data format for hybrid:
- X_timeseries: [num_samples, num_channels, sequence_length]
- X_features: [num_samples, num_features]
- y: [num_samples]
- metadata: DataFrame with num_samples rows

# Task schema

task.type:
- classification
- regression
- anomaly_detection
- rul_prediction
- representation_learning

Example:
task:
  type: classification
  target_column: label
  num_classes: 3
  positive_label: null

# Input/output compatibility

preprocessing.output_type = features
- compatible model.input_type = feature-based
- example models: SVM, RandomForest, XGBoost, LogisticRegression, MLP with feature input

preprocessing.output_type = timeseries
- compatible model.input_type = timeseries-based
- example models: CNN1D, LSTM, GRU, TCN, Transformer

preprocessing.output_type = hybrid
- compatible model.input_type = hybrid
- example models: CNN + statistical features, LSTM + process parameters, late fusion model

# Steady cut schema

steady_cut.mode:
- full_signal
- sliding_window
- steady_cut_only
- air_cut_removal
- segmentation

steady_cut.method:
- none
- threshold
- statistics
- heuristic
- ml
- dl

steady_cut fields:
- mode
- method
- reference_signal
- threshold
- window_size
- stride
- overlap
- segment_label_column
- remove_transient
- save_segments

steady_cut must be stored in:
- resolved_config.yaml
- preprocessing_summary.json
- steady_cut_summary.json
- metrics.csv
- comparison.csv
- experiments/index.csv

# Functional requirements

## FR-CKPT: Checkpoint 관리

FR-CKPT-01: 매 epoch 또는 지정 interval마다 checkpoint를 저장할 수 있어야 한다.
FR-CKPT-02: validation metric 기준 best checkpoint를 저장할 수 있어야 한다.
FR-CKPT-03: 마지막 epoch의 last checkpoint를 저장할 수 있어야 한다.
FR-CKPT-04: checkpoint에서 학습을 재개할 수 있어야 한다.
FR-CKPT-05: checkpoint에는 model_state, optimizer_state, scheduler_state, epoch, best_metric, config_hash가 포함되어야 한다.
FR-CKPT-06: checkpoint 저장 개수를 제한할 수 있어야 한다.
FR-CKPT-07: ML 모델의 경우 pickle/joblib artifact로 저장할 수 있어야 한다.

## FR-LOG: 실행 Logging

FR-LOG-01: 실험 시작/종료 시각을 기록해야 한다.
FR-LOG-02: 실행 config와 resolved config를 저장해야 한다.
FR-LOG-03: random seed를 기록해야 한다.
FR-LOG-04: dataset 버전, sample 수, label 분포를 기록해야 한다.
FR-LOG-05: split 전략과 split 결과 요약을 기록해야 한다.
FR-LOG-06: preprocessing 단계와 파라미터를 기록해야 한다.
FR-LOG-07: epoch별 train/val loss와 metric을 기록해야 한다.
FR-LOG-08: 에러 발생 시 traceback을 error.log에 저장해야 한다.
FR-LOG-09: Python, PyTorch, CUDA, OS 등 실행 환경을 environment.json에 저장해야 한다.
FR-LOG-10: 가능하면 git commit hash와 dirty state를 기록해야 한다.

## FR-RESULT: 실험 결과 저장

FR-RESULT-01: 전체 metric은 metrics.json으로 저장해야 한다.
FR-RESULT-02: 여러 실험 비교를 위해 metrics.csv도 저장할 수 있어야 한다.
FR-RESULT-03: sample별 예측 결과는 predictions.csv로 저장해야 한다.
FR-RESULT-04: classification은 confusion matrix를 저장해야 한다.
FR-RESULT-05: regression은 residual/error table을 저장해야 한다.
FR-RESULT-06: domain별 성능은 domain_metrics.csv로 저장해야 한다.
FR-RESULT-07: steady cut 조건별 성능 비교가 가능하도록 preprocessing 조건을 함께 저장해야 한다.
FR-RESULT-08: split.csv를 반드시 저장해야 한다.

## FR-CONFIG: Config 검증 및 관리

FR-CONFIG-01: 필수 config key 누락 시 실행 전에 에러를 발생시켜야 한다.
FR-CONFIG-02: model.input_type과 preprocessing.output_type이 호환되는지 검증해야 한다.
FR-CONFIG-03: split.strategy에 필요한 metadata column이 존재하는지 검증해야 한다.
FR-CONFIG-04: steady_cut.mode에 필요한 reference_signal이 존재하는지 검증해야 한다.
FR-CONFIG-05: resolved_config.yaml을 저장해야 한다.
FR-CONFIG-06: task.type에 따라 loss, metric, model output dimension이 호환되는지 검증해야 한다.
FR-CONFIG-07: experiment_id가 명시되지 않은 경우 자동 생성해야 한다.
FR-CONFIG-08: config hash를 생성해 resolved_config.yaml과 checkpoint에 저장해야 한다.

## FR-RESUME: 학습 재개

FR-RESUME-01: last checkpoint에서 학습을 재개할 수 있어야 한다.
FR-RESUME-02: resume 시 기존 experiment_id를 유지할지 새 experiment_id를 만들지 선택할 수 있어야 한다.
FR-RESUME-03: resume 로그에는 재개 epoch와 checkpoint path를 기록해야 한다.

## FR-DRYRUN: Dry-run 실행

FR-DRYRUN-01: config 로딩과 validation만 수행할 수 있어야 한다.
FR-DRYRUN-02: 데이터셋 로딩 가능 여부만 확인할 수 있어야 한다.
FR-DRYRUN-03: preprocessing pipeline shape 변화만 출력할 수 있어야 한다.
FR-DRYRUN-04: split 결과 요약만 생성할 수 있어야 한다.

## FR-SMOKE: Smoke Test

FR-SMOKE-01: 샘플 데이터 일부만 사용해 전체 pipeline을 빠르게 실행할 수 있어야 한다.
FR-SMOKE-02: DL 모델은 1 batch forward/backward가 되는지 확인해야 한다.
FR-SMOKE-03: 결과는 tests/outputs/smoke/에 저장해야 한다.
FR-SMOKE-04: smoke test는 실제 experiment_execution 결과와 구분되어야 한다.

## FR-SHAPE: Shape Trace

FR-SHAPE-01: 각 pipeline 단계의 X/y/metadata shape를 기록해야 한다.
FR-SHAPE-02: DL model forward 전후 tensor shape를 기록해야 한다.
FR-SHAPE-03: shape trace는 dry-run과 smoke test에서 확인 가능해야 한다.

## FR-INDEX: Experiment Index

FR-INDEX-01: 실험 시작 시 experiments/index.csv에 running 상태로 등록해야 한다.
FR-INDEX-02: 실험 종료 시 finished/failed 상태와 주요 metric을 업데이트해야 한다.
FR-INDEX-03: experiment_id, dataset, model, split, steady_cut 조건으로 검색 가능해야 한다.
FR-INDEX-04: 실패한 experiment_execution은 error message와 error.log 경로를 index에 기록해야 한다.

## FR-COMPARE: 실험 비교

FR-COMPARE-01: 여러 experiment_id의 metrics를 모아 comparison.csv를 생성할 수 있어야 한다.
FR-COMPARE-02: model별, split별, steady_cut 조건별 성능 비교를 지원해야 한다.
FR-COMPARE-03: best experiment_execution을 metric 기준으로 자동 정렬할 수 있어야 한다.

## FR-ANALYSIS: 실험 결과 분석

FR-ANALYSIS-01: 단일 experiment_execution의 metrics, predictions, split, config를 로드해 결과를 분석할 수 있어야 한다.
FR-ANALYSIS-02: classification task는 오분류 sample을 label, domain, condition 기준으로 분석할 수 있어야 한다.
FR-ANALYSIS-03: regression task는 residual과 absolute error를 sample, domain, condition 기준으로 분석할 수 있어야 한다.
FR-ANALYSIS-04: domain_id가 존재하는 경우 source/target/domain별 성능 차이를 분석할 수 있어야 한다.
FR-ANALYSIS-05: steady_cut_mode가 존재하는 경우 full signal, sliding window, steady cut-only, air cut removal 조건별 성능 차이를 분석할 수 있어야 한다.
FR-ANALYSIS-06: 여러 experiment_execution을 model, input_type, split_strategy, steady_cut_mode 기준으로 비교할 수 있어야 한다.
FR-ANALYSIS-07: 분석 결과는 summary table 형태로 저장할 수 있어야 한다.

## FR-VIS: 실험 결과 시각화

FR-VIS-01: DL 학습 experiment_execution의 train/validation loss 및 metric curve를 시각화할 수 있어야 한다.
FR-VIS-02: classification task의 confusion matrix를 시각화할 수 있어야 한다.
FR-VIS-03: regression task의 prediction-vs-target plot과 residual plot을 생성할 수 있어야 한다.
FR-VIS-04: domain별 성능 차이를 bar plot 또는 table figure로 시각화할 수 있어야 한다.
FR-VIS-05: steady cut 조건별 성능 차이를 시각화할 수 있어야 한다.
FR-VIS-06: 여러 experiment_execution의 metric을 비교하는 plot을 생성할 수 있어야 한다.
FR-VIS-07: 생성된 figure는 experiments/executions/{experiment_id}/figures/ 또는 experiments/comparisons/{comparison_id}/figures/에 저장해야 한다.

## FR-LEAKAGE: Leakage 방지

FR-LEAKAGE-01: split 생성 후 동일 sample_id가 train/validation/test에 중복 존재하지 않는지 검사해야 한다.
FR-LEAKAGE-02: group-wise split에서는 동일 dataset_run_id, tool_id, machine_id, condition_id가 의도와 다르게 train/test에 동시에 존재하는지 검사해야 한다.
FR-LEAKAGE-03: normalization, feature selection, PCA 등 fit이 필요한 preprocessing은 train split에서만 fit되어야 한다.
FR-LEAKAGE-04: leakage check 결과는 leakage_check.json으로 저장해야 한다.

## FR-CACHE: Cache 관리

FR-CACHE-01: 동일 dataset/config/preprocessing hash에 대해 전처리 결과를 재사용할 수 있어야 한다.
FR-CACHE-02: cache 사용 여부는 config에서 제어할 수 있어야 한다.
FR-CACHE-03: cache artifact는 config hash와 dataset fingerprint를 함께 저장해야 한다.
FR-CACHE-04: cache hit/miss 여부를 run.log에 기록해야 한다.

## FR-CLI: CLI 사용성

FR-CLI-01: 단일 실험은 run_experiment.py --config로 실행할 수 있어야 한다.
FR-CLI-02: dry-run은 --dry-run 옵션으로 실행할 수 있어야 한다.
FR-CLI-03: smoke test는 smoke_test.py 또는 --smoke 옵션으로 실행할 수 있어야 한다.
FR-CLI-04: 학습 재개는 resume_experiment.py --experiment-id 또는 --checkpoint로 실행할 수 있어야 한다.
FR-CLI-05: 여러 실험 비교는 compare_experiments.py --experiment-ids로 실행할 수 있어야 한다.
FR-CLI-06: 특정 실험 요약은 inspect_experiment.py --experiment-id로 확인할 수 있어야 한다.
FR-CLI-07: 특정 실험 보고서는 export_report.py --experiment-id로 생성할 수 있어야 한다.

## FR-EXTENSIBILITY: 확장성

FR-EXTENSIBILITY-01: 프레임워크는 새로운 dataset loader를 기존 코드 수정 최소화로 추가할 수 있어야 한다.

FR-EXTENSIBILITY-02: 프레임워크는 새로운 preprocessing step을 registry에 등록하는 방식으로 추가할 수 있어야 한다.

FR-EXTENSIBILITY-03: 프레임워크는 새로운 split strategy를 split_registry에 등록하는 방식으로 추가할 수 있어야 한다.

FR-EXTENSIBILITY-04: 프레임워크는 새로운 feature extractor를 feature_registry에 등록하는 방식으로 추가할 수 있어야 한다.

FR-EXTENSIBILITY-05: 프레임워크는 새로운 ML/DL/hybrid model을 model_registry에 등록하는 방식으로 추가할 수 있어야 한다.

FR-EXTENSIBILITY-06: 프레임워크는 새로운 evaluation metric을 기존 evaluator 구조를 깨지 않고 추가할 수 있어야 한다.

FR-EXTENSIBILITY-07: 프레임워크는 새로운 analysis module과 visualization module을 독립적으로 추가할 수 있어야 한다.

FR-EXTENSIBILITY-08: 새로운 기능 추가 시 기존 config schema와 backward compatibility를 가능한 한 유지해야 한다.

FR-EXTENSIBILITY-09: config schema 변경이 필요한 경우 변경 사유, 영향 범위, migration 방법을 patch note에 기록해야 한다.

FR-EXTENSIBILITY-10: 확장 기능은 최소한의 smoke test 또는 unit test를 함께 제공해야 한다.

## FR-VERSION: Git 기반 버전 관리

FR-VERSION-01: 프레임워크의 모든 코드, config template, schema, 문서, patch note는 git으로 버전 관리되어야 한다.

FR-VERSION-02: 주요 변경 사항은 git commit 단위로 추적 가능해야 한다.

FR-VERSION-03: 각 experiment_execution은 실행 시점의 git commit hash를 environment.json 또는 reproducibility.json에 기록해야 한다.

FR-VERSION-04: 각 experiment_execution은 실행 시점의 git dirty state를 기록해야 한다.

FR-VERSION-05: release 또는 milestone 단위로 git tag를 사용할 수 있어야 한다.

FR-VERSION-06: config schema, output schema, dataset schema가 변경될 경우 semantic version 또는 내부 version number를 갱신해야 한다.

FR-VERSION-07: experiment_execution의 resolved_config.yaml에는 framework_version, config_schema_version, dataset_schema_version을 포함해야 한다.

FR-VERSION-08: 이전 버전에서 생성된 experiment_execution 결과를 읽을 수 있도록 가능한 한 backward compatibility를 유지해야 한다.

FR-VERSION-09: backward compatibility를 깨는 변경은 patch note에 breaking change로 명시해야 한다.

## FR-PATCHNOTE: Patch Note 관리

FR-PATCHNOTE-01: 프레임워크에 변경이 발생할 때마다 patch note를 작성해야 한다.

FR-PATCHNOTE-02: patch note는 변경 날짜, 버전, 작성자, 관련 git commit hash를 포함해야 한다.

FR-PATCHNOTE-03: patch note에는 변경 요약, 변경 이유, 영향 범위, migration 필요 여부를 기록해야 한다.

FR-PATCHNOTE-04: patch note에는 추가된 기능, 수정된 기능, 제거된 기능, 버그 수정, breaking change를 구분하여 기록해야 한다.

FR-PATCHNOTE-05: config schema 변경 사항은 patch note에 별도 항목으로 기록해야 한다.

FR-PATCHNOTE-06: output directory structure 변경 사항은 patch note에 별도 항목으로 기록해야 한다.

FR-PATCHNOTE-07: preprocessing, split, model, evaluation, analysis, visualization 로직 변경은 patch note에 영향 범위와 함께 기록해야 한다.

FR-PATCHNOTE-08: 실험 결과 재현성에 영향을 줄 수 있는 변경은 patch note에 명확히 표시해야 한다.

FR-PATCHNOTE-09: patch note는 `docs/patch_notes/` 아래에 버전별 markdown 파일로 저장해야 한다.

FR-PATCHNOTE-10: 최신 변경 요약은 `CHANGELOG.md`에도 반영해야 한다.

FR-PATCHNOTE-11: patch note 작성 없이 기능 변경, schema 변경, output 구조 변경을 merge하지 않아야 한다.

FR-PATCHNOTE-12: 각 patch note는 관련 테스트 결과 또는 smoke test 결과 경로를 포함해야 한다.

# Patch Note Template

파일 위치:
docs/patch_notes/v{version}.md

예:
docs/patch_notes/v0.2.0.md

내용:

# Patch Note: v0.2.0

## Metadata

- Date:
- Author:
- Git commit:
- Related issue:
- Framework version:
- Config schema version:
- Dataset schema version:
- Output schema version:

## Summary

- 이번 변경의 핵심 요약

## Added

- 새로 추가된 기능

## Changed

- 변경된 기능 또는 동작

## Fixed

- 수정된 버그

## Removed

- 제거된 기능

## Breaking Changes

- 기존 config, output, API, CLI와 호환되지 않는 변경 사항
- 필요한 migration 방법

## Schema Changes

### Config schema

- 변경 전:
- 변경 후:
- 영향 범위:

### Dataset schema

- 변경 전:
- 변경 후:
- 영향 범위:

### Output schema

- 변경 전:
- 변경 후:
- 영향 범위:

## Reproducibility Impact

- 기존 experiment_execution 재현에 영향이 있는지
- 영향이 있다면 어떤 조건에서 결과가 달라질 수 있는지

## Migration Guide

- 기존 config 또는 결과물을 새 버전에 맞게 바꾸는 방법

## Test Evidence

- Unit test:
- Integration test:
- Smoke test:
- Test output path:

## Related Experiment Executions

- 영향을 받은 experiment_id:

# Output requirements

## OR-CONFIG: Config 산출물

OR-CONFIG-01: 각 experiment_execution은 실행에 사용된 원본 config를 저장해야 한다.
OR-CONFIG-02: 각 experiment_execution은 include/default/override가 모두 반영된 resolved_config를 저장해야 한다.
OR-CONFIG-03: resolved_config에는 dataset, task, preprocessing, split, model, training, evaluation, logging 설정이 모두 포함되어야 한다.
OR-CONFIG-04: resolved_config에는 experiment_id, experiment_name, created_at, seed가 포함되어야 한다.
OR-CONFIG-05: config validation 결과를 저장해야 하며, 실패 시 어떤 key 또는 조건이 문제였는지 기록해야 한다.

저장 위치:
experiments/executions/{experiment_id}/configs/
- input_config.yaml
- resolved_config.yaml
- config_validation.json

## OR-DATA: 데이터 요약 산출물

OR-DATA-01: 각 experiment_execution은 사용한 데이터셋 이름, 경로, 버전 또는 생성 시점을 저장해야 한다.
OR-DATA-02: 전체 sample 수, train/val/test sample 수를 저장해야 한다.
OR-DATA-03: label 분포를 전체 및 split별로 저장해야 한다.
OR-DATA-04: metadata column 목록과 주요 domain column을 저장해야 한다.
OR-DATA-05: 센서 channel 목록과 입력 shape 정보를 저장해야 한다.
OR-DATA-06: 전처리 전후 sample 수 변화와 shape 변화를 저장해야 한다.
OR-DATA-07: dataset fingerprint를 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/data/
- dataset_summary.json
- dataset_fingerprint.json
- label_distribution.csv
- metadata_summary.json
- shape_summary.json

## OR-SPLIT: Split 산출물

OR-SPLIT-01: 각 experiment_execution은 sample별 train/val/test 할당 결과를 split.csv로 저장해야 한다.
OR-SPLIT-02: split.csv에는 sample_id, dataset_run_id, domain_id, label, split 컬럼이 포함되어야 한다.
OR-SPLIT-03: condition-wise, tool-wise, machine-wise split을 사용할 경우 해당 group key 컬럼을 split.csv에 포함해야 한다.
OR-SPLIT-04: split_summary에는 split 전략, group_key, source domain, target domain 정보를 저장해야 한다.
OR-SPLIT-05: split별 sample 수와 label 분포를 저장해야 한다.
OR-SPLIT-06: leakage 방지를 위해 동일 group이 train/test에 동시에 존재하는지 검사한 결과를 저장해야 한다.
OR-SPLIT-07: split 생성에 사용된 random seed를 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/splits/
- split.csv
- split_summary.json
- leakage_check.json

## OR-PREPROCESSING: 전처리 산출물

OR-PREPROCESSING-01: 각 experiment_execution은 적용된 preprocessing pipeline 순서를 저장해야 한다.
OR-PREPROCESSING-02: 각 preprocessing step의 parameter를 저장해야 한다.
OR-PREPROCESSING-03: normalization을 수행한 경우 train set에서 fit된 mean/std 또는 min/max를 저장해야 한다.
OR-PREPROCESSING-04: windowing을 수행한 경우 window_size, stride, overlap, window 수를 저장해야 한다.
OR-PREPROCESSING-05: feature extraction을 수행한 경우 feature 목록과 feature dimension을 저장해야 한다.
OR-PREPROCESSING-06: steady cut 또는 air cut 처리를 수행한 경우 mode, method, reference_signal, threshold 또는 segmentation 기준을 저장해야 한다.
OR-PREPROCESSING-07: 전처리 단계별 입력/출력 shape trace를 저장해야 한다.
OR-PREPROCESSING-08: 전처리 후 생성된 cache 또는 intermediate artifact의 경로를 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/preprocessing/
- preprocessing_summary.json
- normalization_params.json
- feature_summary.json
- steady_cut_summary.json
- shape_trace.json

## OR-MODEL: 모델 산출물

OR-MODEL-01: 각 experiment_execution은 사용한 model.name, model_type, input_type을 저장해야 한다.
OR-MODEL-02: model_type은 ML, DL, hybrid 중 하나로 저장해야 한다.
OR-MODEL-03: input_type은 feature-based, timeseries-based, hybrid 중 하나로 저장해야 한다.
OR-MODEL-04: 모델 hyperparameter 전체를 저장해야 한다.
OR-MODEL-05: DL 모델의 경우 model architecture summary를 저장해야 한다.
OR-MODEL-06: ML 모델의 경우 학습된 estimator artifact를 저장해야 한다.
OR-MODEL-07: feature importance를 제공하는 모델의 경우 feature_importance.csv를 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/model/
- model_summary.json
- model_architecture.txt
- model_config.yaml
- feature_importance.csv

ML artifact 저장 위치:
experiments/executions/{experiment_id}/artifacts/
- model.joblib

## OR-CHECKPOINT: Checkpoint 산출물

OR-CHECKPOINT-01: DL 학습 experiment_execution은 last checkpoint를 저장해야 한다.
OR-CHECKPOINT-02: validation metric을 사용하는 경우 best checkpoint를 저장해야 한다.
OR-CHECKPOINT-03: checkpoint에는 model_state, optimizer_state, scheduler_state, epoch, best_metric, config_hash가 포함되어야 한다.
OR-CHECKPOINT-04: save_interval이 설정된 경우 epoch별 checkpoint를 저장해야 한다.
OR-CHECKPOINT-05: max_keep이 설정된 경우 보관 개수를 초과한 checkpoint를 자동 정리해야 한다.
OR-CHECKPOINT-06: resume 학습 시 사용한 checkpoint 경로와 재개 epoch를 저장해야 한다.
OR-CHECKPOINT-07: ML 모델은 checkpoint 대신 최종 학습 artifact를 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/checkpoints/
- best.pt
- last.pt
- epoch_050.pt

## OR-LOG: 실행 로그 산출물

OR-LOG-01: 각 experiment_execution은 실행 로그를 run.log로 저장해야 한다.
OR-LOG-02: 에러 발생 시 error.log와 traceback을 저장해야 한다.
OR-LOG-03: 실행 시작 시각, 종료 시각, 총 실행 시간을 저장해야 한다.
OR-LOG-04: Python, OS, package version, CUDA, GPU 정보를 environment.json으로 저장해야 한다.
OR-LOG-05: git commit hash와 dirty state를 저장해야 한다.
OR-LOG-06: 사용한 random seed와 deterministic 설정 여부를 저장해야 한다.
OR-LOG-07: epoch별 학습 진행 상황은 training_history.csv로 저장해야 한다.
OR-LOG-08: dry-run 또는 smoke test 실행 여부를 로그에 명확히 표시해야 한다.

저장 위치:
experiments/executions/{experiment_id}/logs/
- run.log
- error.log
- environment.json
- reproducibility.json

학습 이력 저장 위치:
experiments/executions/{experiment_id}/metrics/
- training_history.csv

## OR-METRIC: 성능 지표 산출물

OR-METRIC-01: 최종 평가 metric은 metrics.json으로 저장해야 한다.
OR-METRIC-02: 여러 실험 비교를 위해 metrics.csv도 저장할 수 있어야 한다.
OR-METRIC-03: classification task는 accuracy, precision, recall, F1-score 중 설정된 metric을 저장해야 한다.
OR-METRIC-04: regression task는 MAE, RMSE, R² 중 설정된 metric을 저장해야 한다.
OR-METRIC-05: anomaly detection task는 AUROC, AUPRC, F1, false alarm rate 중 설정된 metric을 저장해야 한다.
OR-METRIC-06: 전체 평균 성능뿐 아니라 split별 성능을 저장해야 한다.
OR-METRIC-07: domain_id 또는 condition_id가 존재하는 경우 domain별 성능을 저장해야 한다.
OR-METRIC-08: best metric과 best epoch를 저장해야 한다.
OR-METRIC-09: metric 계산에 사용된 threshold 또는 post-processing 설정을 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/metrics/
- metrics.json
- metrics.csv
- domain_metrics.csv
- threshold_summary.json

## OR-PREDICTION: 예측 결과 산출물

OR-PREDICTION-01: sample별 예측 결과를 predictions.csv로 저장해야 한다.
OR-PREDICTION-02: predictions.csv에는 sample_id, y_true, y_pred, split 컬럼이 포함되어야 한다.
OR-PREDICTION-03: classification task는 class probability 또는 decision score를 저장할 수 있어야 한다.
OR-PREDICTION-04: regression task는 residual 또는 absolute_error를 저장해야 한다.
OR-PREDICTION-05: anomaly detection task는 anomaly_score와 threshold 적용 결과를 저장해야 한다.
OR-PREDICTION-06: domain_id, condition_id, tool_id, machine_id 등 주요 metadata를 함께 저장해야 한다.
OR-PREDICTION-07: error analysis를 위해 오분류 또는 큰 오차 sample 목록을 별도로 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/predictions/
- predictions.csv
- prediction_errors.csv

## OR-ANALYSIS: 분석 산출물

OR-ANALYSIS-01: 단일 experiment_execution 분석 결과는 analysis_summary.json으로 저장해야 한다.
OR-ANALYSIS-02: classification task의 오분류 분석 결과는 error_summary.csv로 저장해야 한다.
OR-ANALYSIS-03: regression task의 residual 분석 결과는 error_summary.csv 또는 residual_summary.csv로 저장해야 한다.
OR-ANALYSIS-04: domain별 성능 차이 분석 결과는 domain_analysis.csv로 저장해야 한다.
OR-ANALYSIS-05: steady_cut_mode별 성능 차이 분석 결과는 steady_cut_analysis.csv로 저장해야 한다.
OR-ANALYSIS-06: 보고서용 요약 테이블은 summary_tables.csv로 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/analysis/
- analysis_summary.json
- error_summary.csv
- domain_analysis.csv
- steady_cut_analysis.csv
- summary_tables.csv

## OR-FIGURE: 시각화 산출물

OR-FIGURE-01: DL 학습 experiment_execution은 learning curve figure를 저장해야 한다.
OR-FIGURE-02: classification task는 confusion matrix figure를 저장해야 한다.
OR-FIGURE-03: regression task는 prediction-vs-target 또는 residual plot을 저장해야 한다.
OR-FIGURE-04: domain 정보가 있는 경우 domain별 성능 figure를 저장해야 한다.
OR-FIGURE-05: steady cut 조건 비교 실험에서는 steady_cut_mode별 성능 figure를 저장할 수 있어야 한다.
OR-FIGURE-06: feature importance가 있는 경우 feature importance figure를 저장할 수 있어야 한다.
OR-FIGURE-07: 모든 figure는 파일 경로가 report에 연결될 수 있어야 한다.

저장 위치:
experiments/executions/{experiment_id}/figures/
- learning_curve.png
- confusion_matrix.png
- prediction_vs_target.png
- residual_plot.png
- domain_performance.png
- steady_cut_performance.png
- feature_importance.png

## OR-REPORT: 보고서 산출물

OR-REPORT-01: 각 experiment_execution은 사람이 읽을 수 있는 report.md를 생성할 수 있어야 한다.
OR-REPORT-02: report에는 실험 목적, dataset, preprocessing, split, model, training, evaluation 설정이 포함되어야 한다.
OR-REPORT-03: report에는 주요 metric, domain별 metric, analysis summary, figure 링크가 포함되어야 한다.
OR-REPORT-04: report에는 best checkpoint 또는 model artifact 경로가 포함되어야 한다.
OR-REPORT-05: report에는 실행 환경과 재현성 정보가 포함되어야 한다.
OR-REPORT-06: 필요 시 report.html 또는 report.pdf로 export할 수 있어야 한다.

저장 위치:
experiments/executions/{experiment_id}/reports/
- report.md
- report.html
- report.pdf

## OR-INDEX: 실험 인덱스 산출물

OR-INDEX-01: 실험 시작 시 experiments/index.csv에 experiment_execution 정보를 running 상태로 등록해야 한다.
OR-INDEX-02: 실험 종료 시 status를 finished 또는 failed로 업데이트해야 한다.
OR-INDEX-03: index.csv에는 experiment_id, experiment_name, dataset, model, input_type, split_strategy, steady_cut_mode, status, best_metric, created_at이 포함되어야 한다.
OR-INDEX-04: 실패한 experiment_execution은 error message와 error.log 경로를 index에 기록해야 한다.
OR-INDEX-05: index.csv는 여러 실험을 검색, 필터링, 비교하는 기준 파일로 사용되어야 한다.

저장 위치:
experiments/
- index.csv

## OR-COMPARISON: 실험 비교 산출물

OR-COMPARISON-01: 여러 experiment_execution을 비교한 결과는 experiments/comparisons/ 아래에 저장해야 한다.
OR-COMPARISON-02: comparison.csv에는 experiment_id, dataset, model, input_type, split_strategy, steady_cut_mode, 주요 metric이 포함되어야 한다.
OR-COMPARISON-03: model별, split별, steady_cut_mode별 비교 결과를 생성할 수 있어야 한다.
OR-COMPARISON-04: 비교 기준 metric에 따라 best experiment_execution을 정렬할 수 있어야 한다.
OR-COMPARISON-05: comparison report를 markdown 또는 html로 생성할 수 있어야 한다.

저장 위치:
experiments/comparisons/{comparison_id}/
- comparison.csv
- comparison_report.md
- figures/

## OR-SMOKE: Smoke test 산출물

OR-SMOKE-01: smoke test 결과는 실제 experiment_execution과 분리하여 tests/outputs/smoke/에 저장해야 한다.
OR-SMOKE-02: smoke test는 dataset loading, preprocessing, split, model forward, 1-batch training 결과를 저장해야 한다.
OR-SMOKE-03: smoke test 결과에는 shape_trace, smoke_metrics, smoke_log가 포함되어야 한다.
OR-SMOKE-04: smoke test 실패 시 실패 단계와 error message를 저장해야 한다.

저장 위치:
tests/outputs/smoke/
- smoke_run.log
- smoke_metrics.json
- smoke_split_summary.json
- smoke_shape_trace.json
- smoke_error.log

# Recommended minimum output set

experiments/executions/{experiment_id}/
- configs/resolved_config.yaml
- splits/split.csv
- splits/leakage_check.json
- logs/run.log
- logs/environment.json
- metrics/metrics.json
- metrics/training_history.csv
- predictions/predictions.csv
- reports/report.md

DL experiment_execution인 경우 추가:
experiments/executions/{experiment_id}/checkpoints/
- best.pt
- last.pt

feature-based ML experiment_execution인 경우 추가:
experiments/executions/{experiment_id}/artifacts/
- model.joblib

# Example config

experiment:
  experiment_id: null
  name: cnn1d_steady_cut_condition_split
  seed: 42

dataset:
  name: example_milling
  raw_path: datasets/raw/example_milling
  metadata_path: datasets/metadata/example_milling.csv
  dataset_version: v1

task:
  type: classification
  target_column: label
  num_classes: 3
  positive_label: null

preprocessing:
  output_type: timeseries
  cache:
    enabled: true
  steps:
    - name: normalization
      method: zscore
      fit_on: train
    - name: steady_cut
      mode: steady_cut_only
      method: threshold
      reference_signal: spindle_current
      threshold: 0.2
    - name: windowing
      window_size: 2048
      stride: 512

split:
  strategy: condition_wise
  group_key: condition_id
  target_groups: [condition_04]
  validation_ratio: 0.2
  leakage_check: true

model:
  name: cnn1d
  model_type: DL
  input_type: timeseries-based
  params:
    num_channels: 3
    num_classes: 3
    hidden_dim: 128

training:
  epochs: 100
  batch_size: 64
  optimizer: adam
  learning_rate: 0.001
  seed: 42

checkpoint:
  enabled: true
  monitor: val_loss
  mode: min
  save_best: true
  save_last: true
  save_interval: 5
  max_keep: 3
  resume_from: null

evaluation:
  metrics: [accuracy, f1_macro]
  group_metrics: [domain_id, condition_id]

logging:
  save_environment: true
  save_git_state: true
  shape_trace: true

report:
  enabled: true
  formats: [md]

## OR-VERSION: 버전 관리 산출물

OR-VERSION-01: 각 experiment_execution은 framework_version을 저장해야 한다.
OR-VERSION-02: 각 experiment_execution은 config_schema_version을 저장해야 한다.
OR-VERSION-03: 각 experiment_execution은 dataset_schema_version을 저장해야 한다.
OR-VERSION-04: 각 experiment_execution은 output_schema_version을 저장해야 한다.
OR-VERSION-05: 각 experiment_execution은 git commit hash와 dirty state를 저장해야 한다.

저장 위치:
experiments/executions/{experiment_id}/logs/
- environment.json
- reproducibility.json

## OR-PATCHNOTE: Patch Note 산출물

OR-PATCHNOTE-01: 각 버전 변경은 docs/patch_notes/v{version}.md에 기록되어야 한다.
OR-PATCHNOTE-02: 최신 변경 요약은 CHANGELOG.md에 기록되어야 한다.
OR-PATCHNOTE-03: 현재 framework version은 VERSION 파일에 기록되어야 한다.
OR-PATCHNOTE-04: patch note에는 관련 git commit hash와 테스트 결과 경로가 포함되어야 한다.

저장 위치:
docs/patch_notes/
- v0.1.0.md
- v0.2.0.md

프로젝트 root:
- CHANGELOG.md
- VERSION

# CLI examples

단일 실험 실행:
python scripts/run_experiment.py --config configs/experiments/exp001.yaml

Dry-run:
python scripts/run_experiment.py --config configs/experiments/exp001.yaml --dry-run

Smoke test:
python scripts/smoke_test.py --config configs/experiments/exp001.yaml

학습 재개:
python scripts/resume_experiment.py --experiment-id 2026-05-06_143012_exp001_cnn1d_steady_cut --checkpoint last

실험 비교:
python scripts/compare_experiments.py --experiment-ids exp001 exp002 exp003 --metric f1_macro

실험 요약 확인:
python scripts/inspect_experiment.py --experiment-id exp001

보고서 생성:
python scripts/export_report.py --experiment-id exp001 --format html

# MVP scope

## MVP 필수

- config loading/validation
- dataset loader 1개
- metadata schema
- task schema
- preprocessing pipeline
- normalization with train-only fit
- windowing
- steady_cut mode 중 full_signal, sliding_window 우선 지원
- random_split, dataset_run-wise split
- leakage check
- model 2개: RandomForest, CNN1D
- trainer
- evaluator
- split.csv 저장
- metrics.json 저장
- predictions.csv 저장
- run.log 저장
- environment.json 저장
- smoke test
- experiment index 저장

## Phase 2

- condition/tool/machine-wise split
- domain shift scenario manager
- steady_cut_only, air_cut_removal
- feature extraction 확장
- hybrid model
- comparison analyzer
- visualization
- report generator
- cache

## Phase 3

- sweep
- resume 고도화
- hyperparameter search
- domain adaptation
- test-time adaptation
- dashboard 또는 UI

# 최종 요구 사항
- 본 프레임워크는 확장 가능한 구조로 설계되어야 하며, dataset loader, preprocessing step, split strategy, feature extractor, model, metric, analysis, visualization module은 registry 또는 독립 모듈 방식으로 추가 가능해야 한다.
- 모든 코드, config, schema, 문서, patch note는 git으로 버전 관리되어야 한다.
- 프레임워크 변경이 발생할 때마다 patch note를 작성해야 하며, 특히 config schema, output schema, dataset schema, preprocessing/split/model/evaluation 로직 변경은 변경 이유와 영향 범위, 재현성 영향, migration 방법을 상세히 기록해야 한다.