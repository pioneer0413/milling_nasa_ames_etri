# Case 13 / 14 Prediction Optimization — Working Context

> 이 문서는 작업 요청자와 에이전트(Claude) 간 단일 공유 문서다.
> sandbox/prj01_case13_14_optimization/ 아래의 모든 스크립트·결과를 이 문서 하나로 추적한다.

---

## 0. 원본 작업 요청 (verbatim)

> 여기다가 Case13과 Case14에 대한 예측 성능을 RMSE 기준 0.8 미만으로 달성하는 모델과 그 하이퍼파라미터를 찾아내. 두 Case에 동시에 RMSE<0.8을 만족할 필요는 없다. @docs/agent_context/scenario_context/H5_S2_context.md 이 파일을 참고해라. 활용 가능한 모든 방법을 동원해라. 달성할 때까지 탐색 및 실험을 반복해라. sandbox/prj01_case13_14_optimization에 네 마음대로 스크립트, 실험 결과, 문서 등을 추가/수정/출력해도 된다. @leader_board/H4_case_model_top5.csv 이 파일에 Case 13, 14에 대해 우수한 성능을 보인 모델과 설정에 대해 우선 탐색해라. 추가 요구 및 너의 프로그램과 맥락을 공유하기 위한 단일 문서로 prj01_case13_14_optimization/ 아래에 context.md 단 하나의 문서만을 둔다. 거기에 작업 요청자인 내가 확인하기 쉽도록 네 탐색 시도를 문서화 해서 저장해라. 지금 프롬프트 또한 그 문서의 상단에 저장해라.

### 0.1 목표 임계값 정정 (요청자 확인 완료)

원문은 "RMSE < 0.8"이나, 기존 best가 이미 Case13=0.0969 / Case14=0.0931로 0.8을 크게 만족하는 모순이 있어 확인 결과 **0.08의 오타**임을 요청자가 확정. 본 작업의 목표는:

> **Case 13, Case 14 각각 LOCV RMSE < 0.08** (두 Case 동시 만족 불필요)

---

## 1. 결론 (TL;DR)

**두 Case 모두 목표 달성.** 핵심 레버는 H5_S1_T1 EDA 결론 그대로 — 고마모 구간에서 붕괴하는 **진동 센서를 버리고 신뢰성 높은 전류 센서(smcAC/smcDC)의 delta feature**만 사용하는 것이다.

| Case | 달성 RMSE | 목표 | H4 baseline | 개선 | 모델 | Feature | Prefix |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| **13** | **0.0602** | <0.08 | 0.0969 | **−38%** | SVR (rbf) | current-delta(8)+meta(3) | 50% |
| **14** | **0.0460** | <0.08 | 0.0932 | **−51%** | XGBoost | current-delta(8) | 20% |

두 결과 모두 `verify_best.py`로 재현 확인됨 (deterministic, seed 고정).

### 최종 하이퍼파라미터

**Case 13 — SVR**
- features: `smcAC/smcDC`의 delta(mean,rms,std,peak) 8개 + meta(DOC,feed,material) 3개 = **11-dim**
- prefix: **50%**, StandardScaler 적용
- `SVR(kernel="rbf", C=200, gamma=0.005, epsilon=0.005)`
- 결과: **RMSE=0.0602, R²=0.9829, MAE=0.0492**

**Case 14 — XGBoost**
- features: `smcAC/smcDC`의 delta 8개 = **8-dim** (meta 미사용, 스케일링 불필요)
- prefix: **20%**
- `XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=1.0, random_state=42, n_jobs=1)`
- 결과: **RMSE=0.0460, R²=0.9827, MAE=0.0380**

> 견고성: 단발 운(luck)이 아니다. Case 13은 sub-0.08 조합이 **49개**(SVR+전류 delta 패밀리 전반), Case 14는 **28개** 존재. 아래 §4 참조.

---

## 2. 접근 전략

LOCV(Leave-One-Case-Out) 프로토콜은 기존 H4/H5 스크립트와 동일하게 유지(target case 제외 14개 case로 학습, target의 모든 run을 단일 prefix에서 예측). 그 위에서 **활용 가능한 모든 방법**을 다음 축으로 전수 탐색했다.

- **Feature set (11종):** 전체6센서 vs 무진동4센서(smcAC/smcDC/AE) vs 전류2센서, 각각 Raw / Delta / Raw+Delta / +Meta 조합
- **Model (12종 × HP 그리드 = 127 설정):** Ridge, Lasso, ElasticNet, BayesianRidge, Huber, SVR, KernelRidge, KNN, XGBoost, RandomForest, ExtraTrees, GradientBoosting
- **Prefix:** 10%~100% (10% step)
- **Target:** Case 13, 14

= 총 **27,940 조합** 1차 ML 스윕. EDA 가설(전류 신뢰 / 진동 붕괴 / meta aliasing)을 검증 가능한 형태로 직접 시험.

신호 파싱이 비싸므로 feature matrix(Raw+Delta, 6센서, 10 prefix, VB/VB_ratio/meta 포함)를 1회 빌드 후 `cache/feat_matrix.pkl`로 캐싱하고 모든 스윕이 이를 재사용.

---

## 3. 탐색 시도 로그 (시간순)

### 시도 1 — 기존 leader board 우선 검토
`leader_board/H4_case_model_top5.csv` 확인. Case 13/14의 기존 상위는 전부 **feature_gru, Delta+Meta(27d), 전체 센서**. Case13 best 0.0969(@70%), Case14 best 0.0932(@80%). 모두 0.08 위. → GRU·전체센서 단일 접근의 한계 확인, 다른 모델·feature 축 탐색 필요 판단.

### 시도 2 — feature matrix 캐시 구축 (`build_features.py`)
6센서 × {mean,rms,std,peak}의 Raw/Delta + meta + VB(+VB_ratio)를 prefix 10~100%에서 추출. 1630 rows × 57 cols. (parquet 엔진 부재로 pickle 저장.)

### 시도 3 — 1차 ML 대규모 스윕 (`optimize_ml.py`, 27,940 조합)
- 1차 실행은 트리/XGBoost가 작은 데이터(≈150 rows)에 매 fit마다 전체 스레드를 생성하는 오버헤드로 27분 이상 미완료 → 중단.
- **수정:** 모델 단일 스레드 강제(`n_jobs=1`, `OMP_NUM_THREADS=1`) + 외부 루프 `joblib.Parallel(n_jobs=-1)` 병렬화 → **1.5분 완료**.
- 결과: **Case 13 sub-0.08 1개(best 0.0677), Case 14 sub-0.08 28개(best 0.0460).** 두 Case 모두 전류센서 feature set이 압도적 1위(아래 표).

### 시도 4 — Case 13 정밀 재탐색 (`refine_case13.py`)
1차에서 Case 13의 sub-0.08가 1개뿐이라 견고성 확인이 필요. 전류/무진동 feature 패밀리에 SVR(C,gamma,epsilon)·KernelRidge(alpha,gamma)를 4,950 조합 정밀 스윕. → **sub-0.08 49개, best 0.0602**. 단발성이 아닌 안정적 패밀리임을 확인.

### 시도 5 — 최종 재현·검증 (`verify_best.py`)
두 우승 설정을 독립 재실행해 RMSE/R²/MAE와 per-run 예측을 출력. (1차 출력 시 subsample 표시 절단으로 0.8을 오용했다가 CSV의 정확값 subsample=1.0로 정정 → Case14 0.0460 정확 재현.) 두 설정 모두 목표 달성 재확인.

### (선택) GRU 스윕 (`optimize_gru.py`)
GRU 하이퍼파라미터(hidden/layers/dropout/lr/epochs) 스윕 스크립트도 준비됨. ML 스윕만으로 두 Case 목표를 이미 충분히(견고하게) 달성하여 필수 실행은 아님. 추가 비교가 필요하면 `python optimize_gru.py` 또는 `--quick`로 실행 가능.

---

## 4. 핵심 분석 — 왜 전류센서 delta인가

### Feature set 별 best RMSE (1차 스윕)

| feature_set | Case 13 | Case 14 |
| --- | ---: | ---: |
| **delta_meta_cur** (전류 delta+meta) | **0.0677** | 0.0640 |
| **delta_only_cur** (전류 delta) | 0.2570 | **0.0460** |
| delta_meta_novib / delta_cur_ae (무진동) | 0.0861 | 0.0991 |
| rawdelta_meta_cur | 0.1002 | 0.0575 |
| delta_meta_all (전체센서, H4와 동형) | 0.1021 | 0.1046 |
| rawdelta_meta_all (전체 51d) | 0.1197 | 0.1095 |

→ **전체 센서를 쓸수록 hard case 성능이 나빠진다.** 진동 센서를 제거하고 전류 중심으로 갈수록 단조롭게 개선. H5_S1_T1의 "vibration breakdown / current reliable" 결론을 정량 재확인.

### Model 별 best RMSE

| model | Case 13 | Case 14 |
| --- | ---: | ---: |
| **SVR (rbf)** | **0.0677** | 0.0564 |
| **XGBoost** | 0.2322 | **0.0460** |
| KernelRidge (rbf) | 0.1303 | 0.0486 |
| GradBoost | 0.1968 | 0.0575 |

- **Case 13**: SVR(rbf)가 독보적. 비선형 커널이 고마모 구간의 feature-space 이탈을 부드럽게 외삽. 선형/트리는 0.13 이상으로 부진.
- **Case 14**: XGBoost·KernelRidge·SVR·GradBoost가 모두 sub-0.08로 경쟁적 → 견고.

### Case별 특성 차이
- **Case 13**(VB_max 1.53, 가장 높은 마모): meta를 *포함*한 전류 delta가 유리(+meta로 process 조건 보정). 중간 prefix(50%)에서 best.
- **Case 14**(VB_max 1.14): meta *없이* 전류 delta만으로 충분, 매우 짧은 prefix(20%)에서도 best. meta aliasing(§H5 findings)을 피하는 편이 오히려 유리.

---

## 5. 재현 방법

```bash
cd /home/hwkang/manufacturing_data_exercise/02_milling_v2
PY=venv312/bin/python

# 1) feature matrix 캐시 (최초 1회, ~1.5s)
$PY sandbox/prj01_case13_14_optimization/build_features.py

# 2) 두 우승 설정 재현·검증 (핵심) — Case13 0.0602 / Case14 0.0460
$PY sandbox/prj01_case13_14_optimization/verify_best.py

# (선택) 전체 ML 스윕 재현 (~1.5분, 27,940 조합)
$PY sandbox/prj01_case13_14_optimization/optimize_ml.py

# (선택) Case 13 정밀 재탐색 (sub-0.08 49개)
$PY sandbox/prj01_case13_14_optimization/refine_case13.py
```

---

## 6. 디렉터리 구성

| 파일 | 내용 |
| --- | --- |
| `context.md` | (본 문서) 단일 공유 문서 |
| `build_features.py` | 신호→feature matrix 추출·캐싱 |
| `optimize_ml.py` | 27,940 조합 ML 대규모 병렬 스윕 |
| `refine_case13.py` | Case 13 SVR/KernelRidge 정밀 스윕 |
| `optimize_gru.py` | (선택) GRU 하이퍼파라미터 스윕 |
| `verify_best.py` | 두 우승 설정 재현·검증 |
| `cache/feat_matrix.pkl` | 캐싱된 feature matrix (1630×57) |
| `cache/ml_results.csv` | 1차 스윕 전체 결과 |
| `cache/case13_refine.csv` | Case 13 정밀 스윕 결과 |
| `cache/ml_sweep.log` | 스윕 실행 로그 |

---

## 7. 한계 및 참고

- 본 탐색은 목표(RMSE<0.08 달성 설정 발견)에 맞춰 test-case RMSE 기준으로 최적 설정을 선별했다. 다만 선택된 방향(전류센서 delta)은 H5_S1_T1 EDA에서 사전 동기화된 가설이며, 임의 과적합이 아니라 **feature set 전반에 걸친 일관된 경향**(§4)으로 뒷받침된다. sub-0.08 조합이 Case13=49 / Case14=28개로 다수 존재한다는 점도 견고성의 근거다.
- 완전한 일반화 주장(미지 case 대상)에는 nested-LOCV로 selector를 학습하는 추가 검증이 필요하며, 이는 H5_S2 §12.3의 model selector 방향과 연결된다. 현재 작업의 범위(두 hard case 각각 목표 달성)는 충족.
- Case 14 best가 prefix 20%로 매우 짧다 — 짧은 입력으로도 전류 delta가 마모를 잘 담는다는 신호. 더 보수적 운영을 원하면 KernelRidge delta_only_cur @60%(0.0486)도 sub-0.08 대안.
