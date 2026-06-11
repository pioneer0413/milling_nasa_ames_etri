# prj02 current-only — Input-Ratio Experiment

> 작업 요청자와 에이전트(Claude) 간 단일 공유 문서.
> sandbox/prj02_current-only/ 아래 스크립트·결과를 이 문서 하나로 추적한다.

---

## 0. 작업 요청 (verbatim)

> sandbox/prj02_current-only에서 SVR과 XGboost만 모델로 사용하고 Sensor는 Current, Feature는 Delta-base만 사용해서 전체 Case에 대한 Input Ratio 실험을 수행해. context.md로 단일 공유 문서 만들어. 실험 완료 후 결과 보고해.

---

## 1. 실험 설정

| 항목 | 값 |
| --- | --- |
| **모델** | SVR, XGBoost (이 둘만) |
| **센서** | Current only → `smcAC`, `smcDC` |
| **피처** | Delta-base only → 각 센서의 delta(mean,rms,std,peak) = **8개** (Raw·Meta 없음) |
| **독립변수** | Input Ratio = signal prefix 10%~100% (10% step) |
| **대상** | 전체 15개 Case `[1,2,3,4,5,7,8,9,10,11,12,13,14,15,16]` |
| **검증** | Leave-One-Case-Out CV (target 제외 14개로 학습, target run 예측) |
| **타겟** | 절대 VB |
| **평가** | (case, ratio, model)별 RMSE → ratio·model별 **15-fold 평균 RMSE** |
| **전처리** | VB 보간(index, inside) + first-run 0 채움, EXCLUDED_RUNS {(2,1),(12,1)} 제외, smcAC/smcDC만 사용 |
| **Delta 기준** | run의 raw stat − 해당 case 첫 유효 run의 raw stat |

### 고정 하이퍼파라미터
입력 비율(input ratio)만을 실험 변수로 두기 위해 모델별 HP는 합리적 기본값으로 **고정**한다.

- **SVR**: `kernel='rbf', C=100, gamma='scale', epsilon=0.05` (+ StandardScaler)
- **XGBoost**: `n_estimators=200, max_depth=3, learning_rate=0.1, subsample=1.0, random_state=42`

> 참고: prj01에서 case별로 HP를 튜닝하고 meta를 포함하면 hard case가 크게 개선됨(예 Case13 0.0602). 본 실험은 요청 제약(delta-only, 튜닝 없는 ratio 효과 측정)에 맞춘 별개 설계다.

---

## 2. 결과 요약 (TL;DR)

- **XGBoost가 거의 전 구간에서 SVR을 앞선다** (이 고정 HP·delta-only 조건).
- **모델별 최적 input ratio**:
  - XGBoost: **50%** — 15-fold 평균 RMSE **0.1514** (mean R²=0.404)
  - SVR: **60%** — 15-fold 평균 RMSE **0.1996** (mean R²=−0.368)
- 중간 비율(40~60%)이 전반적으로 유리. 10% 같은 매우 짧은 입력은 평균적으로 가장 나쁨(특히 SVR 0.2943).
- 평균 RMSE는 소수 hard case(13, 12, 1)가 끌어올린다. case별 best-ratio 기준으로 보면 XGBoost는 15개 중 **6개가 RMSE<0.08**, 7개가 <0.10.

### 15-fold 평균 LOCV RMSE — input ratio별

| ratio | SVR | XGBoost |
| ---: | ---: | ---: |
| 10% | 0.2943 | 0.1877 |
| 20% | 0.2393 | 0.1824 |
| 30% | 0.2337 | 0.1881 |
| 40% | 0.2039 | 0.1538 |
| **50%** | 0.2423 | **0.1514** |
| **60%** | **0.1996** | 0.1662 |
| 70% | 0.2254 | 0.1700 |
| 80% | 0.2485 | 0.1613 |
| 90% | 0.2340 | 0.1577 |
| 100% | 0.2112 | 0.1567 |

### 모델 종합 비교 (case별 best-ratio RMSE 기준)

| 모델 | 평균(15-case best) | RMSE<0.08 | RMSE<0.10 |
| --- | ---: | ---: | ---: |
| XGBoost | **0.1108** | **6/15** | 7/15 |
| SVR | 0.1348 | 3/15 | 6/15 |

---

## 3. Case별 best-ratio RMSE

각 case에서 ratio를 바꿔가며 얻은 최소 RMSE와 그 ratio.

| Case | SVR (best@ratio) | XGBoost (best@ratio) |
| ---: | --- | --- |
| 1 | 0.2688 @40% | 0.1167 @50% |
| 2 | 0.1328 @40% | 0.0958 @40% |
| 3 | 0.1274 @30% | 0.0586 @40% |
| 4 | 0.0515 @60% | **0.0491 @100%** |
| 5 | 0.0992 @40% | 0.0669 @50% |
| 7 | 0.0579 @60% | **0.0409 @100%** |
| 8 | 0.1349 @40% | 0.1360 @60% |
| 9 | 0.1186 @100% | 0.1317 @10% |
| 10 | 0.1766 @30% | 0.0597 @70% |
| 11 | 0.0993 @20% | 0.1007 @50% |
| 12 | 0.2112 @10% | 0.1720 @10% |
| 13 | 0.2893 @100% | 0.3537 @100% |
| 14 | 0.1073 @20% | **0.0460 @20%** |
| 15 | **0.0629 @50%** | 0.1076 @40% |
| 16 | **0.0844 @80%** | 0.1267 @10% |

---

## 4. 해석

- **XGBoost > SVR (delta-only, no meta).** 비선형 트리가 8차원 전류 delta에서 비단조·포화 구간을 더 잘 흡수. SVR(rbf, 기본 gamma='scale')은 hard case에서 음의 R²로 크게 무너짐.
- **중간 input ratio(40~60%)가 최적.** 너무 짧으면(10~20%) 마모 신호가 덜 누적되어 평균 성능 저하, 100%까지 가도 추가 이득은 작고 일부 case는 오히려 악화. → 전류 delta는 절반 정도의 입력에서 이미 마모 진행을 충분히 담는다.
- **delta-only 제약의 영향.** Case 13은 두 모델 모두 부진(SVR 0.289 / XGB 0.354). prj01에서 Case 13은 meta 포함이 결정적(0.0602)이었고 delta-only-current는 0.257로 나쁜 것과 일치 — 본 실험은 의도적으로 meta를 뺀 조건이므로 예상된 결과.
- **이미 잘 풀리는 case.** Case 4·7·14는 전류 delta만으로 XGBoost RMSE<0.05로 매우 우수.

---

## 5. 재현 방법

```bash
cd /home/hwkang/manufacturing_data_exercise/02_milling_v2
venv312/bin/python sandbox/prj02_current-only/run_experiment.py
```

최초 실행 시 `cache/feat_current_delta.pkl`(전류 delta feature matrix)를 빌드해 캐싱하고, 이후 재사용한다.

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `context.md` | (본 문서) 단일 공유 문서 |
| `run_experiment.py` | feature 빌드 + LOCV input-ratio 실험 (SVR/XGBoost) |
| `cache/feat_current_delta.pkl` | 전류 delta feature matrix (1630×12) |
| `cache/results.csv` | (model, ratio, case)별 RMSE/MAE/R² — 300행 |
| `cache/summary.csv` | (model, ratio)별 15-fold 평균 RMSE |
