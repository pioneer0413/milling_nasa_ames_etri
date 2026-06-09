# Framework Common Context for Agents

작성일: 2026-06-08
대상: `milling_experiment_framework` 개발 또는 실험 자동화를 맡는 모든 에이전트

이 문서는 개별 H/S/T 태스크 컨텍스트를 읽기 전에 공유해야 할 공통 맥락이다. 새 기능을 만들거나 기존 실험을 수정하기 전에, 이 파일을 먼저 읽고 현재 코드 기준의 실행 구조와 실험 원칙을 맞춘다.

## 1. 프로젝트의 현재 목적

이 저장소는 NASA Ames milling 데이터와 관련 가공 데이터를 대상으로, 공구 마모 지표인 `VB`를 예측하거나 feature/VB 관계를 분석하는 제조 데이터 실험 프레임워크다.

핵심 목표는 다음과 같다.

- feature-based, timeseries-based, hybrid 모델을 같은 실험 산출물 구조에서 비교한다.
- 절삭 신호를 full signal, segment, steady/entry/exit, prefix length, sliding/window 형태로 바꿔가며 성능 변화를 추적한다.
- train/test 누수 없이 source-domain 또는 leave-one-case-out 기준으로 전처리와 모델 학습을 수행한다.
- 모든 실행은 config, resolved config, split, metrics, predictions, logs, reports, figures를 `experiment_execution` 단위로 남긴다.

현재 프레임워크 버전은 `VERSION = 0.1.0`이며, `CHANGELOG.md` 기준으로 MVP 파이프라인 위에 H1-H4 실험군이 추가되어 있다.

## 2. 가장 중요한 용어

- `dataset_run`: 데이터셋 내부의 가공 run/pass/cycle 단위다. 예: `case`, `run`, `dataset_run_id`.
- `experiment_execution`: 하나의 config 또는 스크립트 실행으로 생성되는 프레임워크 실행 단위다. 결과 디렉터리와 `experiments/index.csv`의 기본 단위다.
- `experiment_id`: experiment_execution 식별자다. dataset 내부 run id와 섞지 않는다.
- `case`: NASA milling 데이터의 주요 domain 단위다. 현재 기본 case scope는 `[1,2,3,4,5,7,8,9,10,11,12,13,14,15,16]`이고 case 6은 제외된다.
- `VB`: tool wear regression target. 일반적으로 `label`에도 복사되어 metadata validation 또는 공통 evaluator와 연결된다.
- `segment_setting`: H2/H3 계열에서 쓰는 절삭 구간 조합이다. 현재 기본값은 `full_length`, `steady`, `entry`, `exit`, `entry_steady`, `entry_exit`, `steady_exit`, `entry_steady_exit`.
- `source_domain`, `target_domain`: domain shift 또는 LOCV에서 train/eval domain을 명시하는 값이다. target domain은 평가 외 용도로 사용하지 않는다.

## 3. 코드와 문서의 신뢰 순서

현재 저장소에는 설계 명세, 사용 문서, 실험 스크립트가 함께 있다. 일부 문서는 오래된 스크립트명이나 목표 상태를 포함한다.

새 작업을 할 때 신뢰 순서는 다음을 따른다.

1. 실제 실행 코드와 테스트
2. 현재 config 파일
3. 최신 usage 문서와 agent context
4. `design_specification.md` 같은 목표 설계 문서

특히 `docs/usage/cli.md`에는 예전 `run_s1_experiment.py` 같은 이름이 남아 있을 수 있다. 현재 H2 실행 진입점은 `scripts/run_H2_S1_experiment.py`, `scripts/run_H2_S2_experiment.py`, `scripts/run_H2_S3_experiment.py`, `scripts/run_H2_S4_experiment.py` 형태를 우선 확인한다.

## 4. 저장소 지도

주요 디렉터리는 다음 역할을 가진다.

- `milling_experiment_framework/core`: config resolve/validation, seed, top-level pipeline wrapper.
- `milling_experiment_framework/experiments`: 일반 실행, H1/H2/domain-shift/estimator 실행 클래스.
- `milling_experiment_framework/preprocessing`: MVP pipeline 단계와 H3/H4 보조 preprocessing utilities.
- `milling_experiment_framework/models`: registry 기반 모델과 H2 전용 sklearn-style regressor catalog.
- `milling_experiment_framework/splits`: random, dataset-run-wise, condition/tool/machine/time split.
- `milling_experiment_framework/evaluation`: classification/regression/anomaly metrics와 grouped metrics.
- `milling_experiment_framework/analysis`, `visualization`, `reports`: 실행 결과 해석, figure, report 생성.
- `milling_experiment_framework/experiment_logging`: stdlib `logging`과 충돌하지 않도록 분리된 logging 모듈.
- `configs/experiments`: H1/H2 실험 config. H4 일부는 config보다 standalone script 인자 중심이다.
- `configs/models`: cumulative S-DNN, hybrid LSTM process 등 모델별 상세 config.
- `scripts`: CLI 진입점과 H1/H3/H4 standalone 분석/실험 스크립트.
- `docs/agent_context`: 에이전트용 공통/시나리오/태스크 컨텍스트.
- `experiments/executions`: 실제 실행 결과. 개발 중 임의 삭제하지 않는다.
- `cache`: window/feature/preprocessing 중간 결과.

`AGENTS.md`와 `CLAUDE.md`는 현재 빈 파일이다. 에이전트 온보딩 문맥은 이 파일과 `docs/agent_context` 하위 문서를 우선한다.

## 5. 두 개의 실행 계층

현재 프레임워크에는 크게 두 종류의 실행 흐름이 공존한다.

### 5.1 일반 MVP 파이프라인

진입점:

```bash
python scripts/run_experiment.py --config <config.yaml>
```

구현:

- `milling_experiment_framework/core/pipeline.py`
- `milling_experiment_framework/experiments/experiment.py`
- 실행 클래스: `ExperimentExecution`

흐름:

1. `load_and_resolve_config`
2. `ExperimentPaths` 생성 및 표준 디렉터리 준비
3. dataset registry로 데이터 로드
4. metadata validation
5. split 생성
6. preprocessing pipeline 적용
7. model registry로 모델 생성
8. ML 또는 DL 학습
9. evaluator로 metrics 계산
10. predictions, analysis, figures, report, index 저장

이 계층은 `dataset.name`이 registry에 등록되어 있어야 한다. 현재 등록 dataset은 `example_milling`이다.

### 5.2 H2 계열 전용 실행 계층

진입점:

```bash
python scripts/run_H2_S1_experiment.py --config configs/experiments/H2_S1_segment_setting_effect_on_VB_prediction.yaml --dry-run
python scripts/run_H2_S2_experiment.py --config configs/experiments/H2_S2_sensor_combination_effect_on_segment_aware_VB_prediction.yaml --dry-run
python scripts/run_H2_S3_experiment.py --config configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors.yaml --dry-run
python scripts/run_H2_S4_experiment.py --config configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml --dry-run
```

구현:

- `BaseH2Execution`
- `S1SegmentExecution`
- `S2SensorCombinationExecution`
- `H2S3FeatureCombinationExecution`
- `H2S4ProcessInformationExecution`
- 공통 helper: `milling_experiment_framework/experiments/h2_execution_utils.py`

이 계층은 일반 dataset registry를 쓰지 않고, `process_info_path`, `signal_data_path`, `heuristic_sequence_path` CSV를 직접 읽는다. `S1SegmentExecution`의 feature table builder가 H2.S2-S4에서도 helper로 재사용된다.

H2 실행은 full grid가 클 수 있으므로 기본적으로 runtime estimate와 confirmation prompt가 있다. 자동 실행 시에는 `--yes`를 사용한다.

## 6. Experiment path 규칙

경로 관리는 `milling_experiment_framework/experiments/execution_path.py`와 `milling_experiment_framework/utils/paths.py`가 담당한다.

새 계층형 path schema:

```text
experiments/executions/H{n}/S{n}/{timestamp}_{experiment_topic}/
```

예:

```text
experiments/executions/H2/S1/2026-05-06_140501_segment_setting_effect/
```

`experiment_id`에 `H2_S1` 같은 H/S 식별자가 있으면 계층형 경로로 파싱된다. 그렇지 않으면 legacy flat 경로 `experiments/executions/{experiment_id}`가 사용된다.

표준 하위 디렉터리:

```text
configs/
data/
splits/
preprocessing/
model/
checkpoints/
artifacts/
logs/
metrics/
predictions/
analysis/
figures/
reports/
```

모든 신규 실행/기능은 가능하면 이 구조를 유지한다.

## 7. Config 시스템

일반 파이프라인 config는 `milling_experiment_framework/core/config.py`가 resolve한다.

필수 top-level key:

- `experiment`
- `dataset`
- `task`
- `preprocessing`
- `split`
- `model`
- `training`
- `evaluation`

중요 호환 규칙:

- `preprocessing.output_type: features` -> `model.input_type: feature-based`
- `preprocessing.output_type: timeseries` -> `model.input_type: timeseries-based`
- `preprocessing.output_type: hybrid` -> `model.input_type: hybrid`

`load_and_resolve_config`는 다음을 자동 보강한다.

- `experiment.created_at`
- `experiment.experiment_id` 자동 생성
- `framework_version`
- `config_schema_version`
- `dataset_schema_version`
- `output_schema_version`
- `steady_cut_mode`
- `config_hash`

H2 계열 config는 일반 `model` key 대신 `models`, `model_defaults`, `seeds`를 주로 사용한다. H2 executor가 자체 `_resolved_config`를 만든다.

## 8. Registry 기준

현재 등록 항목은 코드 import side effect로 등록된다.

Dataset:

- `example_milling`

Preprocessor:

- `normalization`
- `steady_cut`
- `windowing`
- `statistical_features`

Split:

- `random`
- `dataset_run_wise`
- `condition_wise`
- `tool_wise`
- `machine_wise`
- `time_based`

General model registry:

- `random_forest`
- `cnn1d`
- `cumulative_s_dnn`
- `feature_gru`
- `hybrid_lstm_process`

H2 model catalog:

- `linear_regression`
- `svr`
- `random_forest`
- `extra_trees`
- `xgboost`
- `lightgbm`
- `mlp`
- `feature_gru`

일반 model registry와 H2 model catalog는 같은 것이 아니다. H2 실험에서 모델을 추가할 때는 `models/h2_regressors.py`와 H2 helper의 seed/runtime 정책을 같이 확인한다.

## 9. 데이터와 누수 방지 원칙

Metadata 최소 요구:

- `sample_id`
- `label`
- `dataset_run_id`
- `sequence_index` 또는 `timestamp`

NASA milling/H2-H4에서 자주 쓰는 columns:

- `case`
- `run`
- `VB`
- `DOC`
- `feed`
- `material` 또는 `material_name`
- sensors: `smcAC`, `smcDC`, `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle`

공통 누수 방지 원칙:

- scaler/imputer/feature preprocessor는 train 또는 source train에만 fit한다.
- target/test domain은 evaluation 외 목적으로 쓰지 않는다.
- `case`, `run`, `VB`, `domain_id`, `dataset_run_id`, `sample_id`, `split`, `source_domain`, `target_domain` 같은 identifier/target columns는 feature vector에 섞지 않는다.
- LOCV에서는 target case가 train/validation에 들어가면 안 된다.
- H2 deterministic 모델인 `linear_regression`, `svr`은 seed 한 번만 실행하며, seeded 모델은 configured seeds를 사용한다.

VB preprocessing은 실험군별로 다를 수 있다.

- `preprocessing/vb_common.py`: case별 run 순서로 missing VB를 선형 보간하고 각 case의 first run을 제외하는 공통 helper.
- H4.S1.T1/T2 스크립트: Run1 VB NaN을 `0.0`으로 채우고 다른 NaN은 보간하며, Case2-Run1과 Case12-Run1을 제외한다.
- H2 실행: process/signal/heuristic CSV merge 후 `VB` NaN row를 제거하는 흐름이 기본이다.

새 작업은 어느 VB rule을 적용하는지 결과 config/report에 명시해야 한다.

## 10. Preprocessing 지도

MVP pipeline 단계:

- `normalization`: zscore/minmax, train split 기준 fit.
- `steady_cut`: `full_signal`은 no-op, `sliding_window`는 windowing 실행, 그 외 mode는 현재 record-only 성격.
- `windowing`: `[N, C, T]` timeseries를 window sample로 확장하고 `parent_sample_id`, `window_start`, `window_end`를 metadata에 추가.
- `statistical_features`: `[N, C, T]`에서 mean/std/min/max/rms 기반 feature matrix 생성.

추가 helper:

- `preprocessing/windowing/overlapped_windowing.py`: fixed overlapped temporal windows, mask/padding/truncation 기록.
- `preprocessing/features/window_feature_extractor.py`: sensor별 handcrafted feature extraction.
- `preprocessing/sequences/feature_sequence_builder.py`: case/run 순서의 feature sequence 생성, source train only imputer/scaler.
- `preprocessing/segmentation/cutting_state_segmenter.py`: Peng2026-inspired no_load/entry/steady/exit segmentation.

H2의 segment feature extraction은 `S1SegmentExecution._build_feature_table` 내부에 구현되어 있으며, MVP `statistical_features` step과 동일하지 않다.

## 11. H1-H4 실험 지도

### H1

목적: feature와 segment, VB 관계를 분석하는 EDA/해석 계열.

대표 코드:

- `experiments/h1_s1_association_execution.py`
- `scripts/run_h1_s1_eda.py`
- `scripts/run_H1_*`

주의: 일부 H1 코드는 H2 helper 변경 후 stale signature를 가질 수 있다. 실행 전 `--dry-run` 또는 최소 smoke 확인이 필요하다.

### H2

목적: segment-aware feature-based VB prediction 실험군.

- H2.S1: segment setting effect.
- H2.S2: sensor group/combination effect.
- H2.S3: feature group/combination effect.
- H2.S4: process information combination effect.

기본 case scope는 case 6 제외 15개 case이며, split은 leave-one-case-out style이다.

H2.S4 process factors:

- A: `DOC`
- B: `feed`
- C: `material_name`
- D: `time`

H2.S4는 numeric/categorical process preprocessing을 `ColumnTransformer`로 처리한다.

### H3

목적: steady cut 길이/위치와 feature/model prediction effect 분석. 주로 `scripts/run_H3_*` standalone script에 구현되어 있다.

새 H3 작업은 같은 output directory convention과 leakage policy를 유지하되, 각 script의 argparse와 local helper를 먼저 읽는다.

### H4

목적: input signal prefix length, segment coverage, feature GRU segment input, hyperparameter tuning 등 고차 ablation.

현재 `docs/agent_context/scenario_context/H4_S1_context.md`와 `task_context/H4_S1_T*.md`가 존재한다.

H4.S1 공통:

- 목적: 입력 신호 길이가 VB 예측 성능에 미치는 영향 검증.
- 검증: Leave-One-Case-Out CV.
- sensors: `smcAC`, `smcDC`, `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle`.
- prefix percents: 보통 10%-100%.
- metrics: RMSE, R2, MAE.
- T1: tabular 모델은 Raw+Meta, GRU는 Delta+Meta sequence.
- T2: 모든 모델이 Raw+Delta+Meta 사용.
- T3: LOCV target cases를 `(1,2,5,8,9,12,14,16)`로 제한.

H4 standalone scripts는 package 내부 `write_json`과 다른 local writer를 쓸 수 있으므로, 출력 schema를 수정할 때 script-local helper도 같이 확인한다.

## 12. Estimator와 grid 규모

실험 규모 추정:

```bash
python scripts/estimate_experiment.py --config <config.yaml>
python scripts/run_experiment.py --config <config.yaml> --estimate-only
```

구현:

- `experiments/estimator/experiment_estimator.py`
- `experiments/estimator/schema.py`
- `experiments/estimator/cost_model.py`
- `experiments/estimator/report_writer.py`

Estimator는 `models`, `seeds`, `shifts`, `sensor_combinations`, `segment_settings`, `feature_combinations`, `process_combinations`, `input_representations`, `modes`, `base_learners`, `sequence_modes`, `hyperparameter_grid`를 atomic axes로 해석한다.

H2/H4 전체 grid는 매우 커질 수 있다. full run 전에 estimator 또는 dry-run으로 atomic count와 output plan을 확인한다.

## 13. 결과 산출물 원칙

일반 실행에서 기대하는 핵심 파일:

- `configs/input_config.yaml`
- `configs/resolved_config.yaml`
- `configs/config_validation.json`
- `data/dataset_summary.json`
- `splits/split.csv`
- `splits/leakage_check.json`
- `preprocessing/preprocessing_summary.json`
- `metrics/metrics.json`
- `metrics/metrics.csv`
- `metrics/training_history.csv`
- `predictions/predictions.csv`
- `analysis/analysis_summary.json`
- `logs/run.log`
- `logs/environment.json`
- `logs/reproducibility.json`
- `reports/report.md`

H2/H4 scripts may use prefixed filenames such as `H2_S4_metrics.json`, `H4_S1_shift_metrics.csv`. 새 파일을 추가할 때는 기존 prefix convention을 유지한다.

`experiments/index.csv`에는 최소한 `experiment_id`, H/S metadata, execution_dir, dataset, model, input_type, split_strategy, steady_cut_mode, status, best_metric, error fields가 들어간다.

## 14. 개발 전 체크리스트

새 에이전트는 작업 시작 전에 다음을 확인한다.

1. 이번 작업이 일반 MVP pipeline인지, H2 executor인지, H3/H4 standalone script인지 구분한다.
2. 관련 config와 script argparse를 실제 코드에서 읽는다.
3. 사용 데이터 파일이 `datasets/processed` 또는 `datasets/metadata`에 존재하는지 확인한다.
4. target/test domain이 preprocessing fit, validation, hyperparameter tuning에 섞이지 않는지 확인한다.
5. 결과 파일명이 기존 실험군 prefix와 맞는지 확인한다.
6. full run 전에 가능한 경우 `--dry-run`, estimator, smoke/unit test 중 하나를 실행한다.
7. schema/output 구조 변경이면 `docs/patch_notes/`, `CHANGELOG.md`, 관련 tests를 고려한다.
8. 기존 `experiments/executions`와 user-created untracked files를 임의로 삭제하거나 되돌리지 않는다.

## 15. 빠른 명령 모음

가상환경은 문서상 다음 경로를 사용한다.

```bash
/home/hwkang/manufacturing_data_exercise/02_milling_v2/venv312
```

대표 명령:

```bash
venv312/bin/python scripts/run_experiment.py --config <config.yaml> --dry-run
venv312/bin/python scripts/run_H2_S1_experiment.py --config configs/experiments/H2_S1_segment_setting_effect_on_VB_prediction.yaml --dry-run
venv312/bin/python scripts/run_H2_S2_experiment.py --config configs/experiments/H2_S2_sensor_combination_effect_on_segment_aware_VB_prediction.yaml --dry-run
venv312/bin/python scripts/run_H2_S3_experiment.py --config configs/experiments/H2_S3_feature_combination_effect_on_segment_aware_VB_prediction_all_sensors.yaml --dry-run
venv312/bin/python scripts/run_H2_S4_experiment.py --config configs/experiments/H2_S4_process_information_combination_effect_on_segment_aware_VB_prediction.yaml --dry-run
venv312/bin/python scripts/estimate_experiment.py --config <config.yaml>
venv312/bin/python -m pytest tests/unit tests/smoke
```

Full H2 run에서 confirmation prompt를 생략하려면 `--yes`를 붙인다.

## 16. 알려진 날카로운 부분

- `docs/usage/cli.md` 일부 명령은 현재 파일명과 다를 수 있다. `rg --files scripts` 또는 실제 script를 확인한다.
- H1 일부 실행 코드는 H2 helper dataclass 변경 전 흔적이 있을 수 있다. 수정 전 실행 가능성을 확인한다.
- `steady_cut` preprocessing은 `steady_cut_only`, `air_cut_removal`, `segmentation`을 validation/config에 기록하지만 MVP pipeline에서는 full processing이 아니라 record-only일 수 있다.
- `lightgbm`은 requirements에는 있으나 환경마다 설치/동작 상태를 확인한다.
- H4 scripts는 package 공통 writer와 별도 local writer를 가진다. JSON/YAML encoding, output directory, resume signature 파일을 script 내부에서 확인한다.
- notebook `prj01_nasa_milling.ipynb`와 untracked generated files는 사용자가 열어둔 작업일 수 있으므로, 요청 없이 정리하지 않는다.

## 17. 개별 컨텍스트 작성 규칙

새 scenario/task context를 만들 때는 다음 순서를 권장한다.

1. 상위 scenario 목적과 research question.
2. 데이터 범위, 제외 case/run, missing VB 처리.
3. feature construction과 segment/prefix/window 정의.
4. split/validation policy와 leakage guard.
5. 모델 목록과 seed policy.
6. metrics와 aggregation rule.
7. expected output files.
8. 기존 공통 규칙에서 달라지는 점.

파일 위치:

- 시나리오 공통: `docs/agent_context/scenario_context/`
- 개별 태스크: `docs/agent_context/task_context/`
- 프레임워크 공통: `docs/agent_context/framework_common_context.md`
