# prj05 stacking — base = SVR/RF/XGBoost, meta-learner = MLP — LOCV @100%, 5 seeds

> 작업 요청자와 에이전트(Claude) 간 단일 공유 문서.
> sandbox/prj05/ 아래 스크립트·결과를 이 문서 하나로 추적한다.

---

## 0. 작업 요청 (verbatim)

> prj05에서 base learner를 SVR, RandomForest, XGBoost만 쓰고 Meta-learner를 MLP를 사용해. 검증은 동일하게 수행해.

(검증 = prj04와 동일: 입력 100%, 누수 없는 nested LOCV, 5-seed, Raw+Delta+Meta(51) feature.)

---

## 1. 구성

| 항목 | 값 |
| --- | --- |
| **Base learners** | SVR(rbf,C=100,γ=scale,ε=0.05), RandomForest(300트리), XGBoost(200,d3,lr0.1) |
| **Meta-learner** | **MLP** `hidden=(16,8)`, ReLU, max_iter=3000, (입력 StandardScaler) |
| Feature | Raw+Delta+Meta = 51 |
| 입력 길이 | 100% only |
| 검증 | 누수 없는 nested LOCV (inner LOCV로 OOF 생성 → MLP meta 학습 → outer 예측) |
| 반복 | seed 5회 `[0,1,2,3,4]`, mean±std |

nested 절차는 prj04와 동일. base learner만 3종으로 줄이고 meta-learner를 Ridge→MLP로 교체.

---

## 2. 결과 요약 (TL;DR)

> **Stacking(MLP meta) 15-fold LOCV RMSE @100% = 0.1511 ± 0.0254** (5 seed mean±std)

### 전체 비교 — 15-fold 평균 RMSE (mean ± std over 5 seeds)
| 구성 | RMSE | 비고 |
| --- | ---: | --- |
| **simple average** | **0.1112 ± 0.0003** | base 3종 단순 평균 (이 실험 최고) |
| SVR (단일) | 0.1155 ± 0.0000 | 최고 단일 base |
| XGBoost (단일) | 0.1204 ± 0.0000 | |
| RandomForest (단일) | 0.1242 ± 0.0011 | |
| **Stacking (MLP meta)** | **0.1511 ± 0.0254** | ← 요청 모델 (가장 나쁨·불안정) |

### Stacking seed별 15-fold 평균 RMSE
| seed | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.1306 | 0.1595 | 0.1342 | 0.1392 | 0.1919 |

### 핵심 관찰 (정직 보고)
- **MLP meta-learner는 stacking 성능을 오히려 악화시켰다.** 0.1511로 단순 평균(0.1112)·단일 SVR(0.1155)·prj04의 Ridge meta(0.1207)보다 모두 나쁘다.
- **불안정성이 크다**: seed 간 std ±0.0254 (0.131~0.192). prj04 Ridge meta(±0.0030)의 8배 이상. MLP의 가중치 초기화·비선형성이 OOF 소표본에서 큰 분산을 유발.
- 원인: meta 학습 데이터는 OOF 예측 **~150행 × 3입력**에 불과하다. 이 소표본·저차원에서 비선형 MLP는 과적합·불안정하며, 선형 meta(Ridge)나 단순 평균보다 일반화가 나쁘다. base가 3종으로 적어 비선형 결합으로 얻을 이득도 거의 없다.

---

## 3. prj04 대비

| 실험 | base | meta | RMSE @100% |
| --- | --- | --- | ---: |
| prj04 | Ridge·SVR·RF·XGB·MLP (5종) | Ridge | 0.1207 ± 0.0030 |
| **prj05** | SVR·RF·XGB (3종) | **MLP** | **0.1511 ± 0.0254** |
| (참고) prj05 simple avg | SVR·RF·XGB (3종) | 평균 | **0.1112 ± 0.0003** |

- **흥미로운 부수 결과:** 약한 base(Ridge·MLP)를 base에서 빼고 **SVR·RF·XGB만 단순 평균**하면 0.1112로, prj04/prj05를 통틀어 가장 좋다. 즉 이 데이터에서는 정교한 meta-learner보다 강한 base 소수의 단순 평균이 우월하다.
- 반대로 meta-learner로 MLP를 쓰면(prj05) 가장 나쁘다 → meta-learner 선택이 stacking 성패를 좌우하며, 소표본에서는 단순·선형 결합이 유리.

---

## 4. Case별 Stacking(MLP) RMSE (seed 5회 mean ± std)

| Case | RMSE | R² | | Case | RMSE | R² |
| ---: | ---: | ---: |---| ---: | ---: | ---: |
| 5 | 0.0782 ± 0.021 | 0.890 | | 16 | 0.1618 ± 0.079 | 0.211 |
| 3 | 0.0908 ± 0.039 | 0.470 | | 10 | 0.1673 ± 0.036 | 0.407 |
| 7 | 0.0989 ± 0.022 | 0.511 | | 8 | 0.1782 ± 0.084 | 0.021 |
| 2 | 0.0999 ± 0.039 | 0.416 | | 14 | 0.1878 ± 0.056 | 0.692 |
| 15 | 0.1221 ± 0.036 | 0.681 | | 12 | 0.1954 ± 0.058 | −0.175 |
| 11 | 0.1350 ± 0.062 | 0.545 | | **13** | **0.3031 ± 0.093** | 0.535 |
| 4 | 0.1382 ± 0.082 | −0.283 | | | | |
| 1 | 0.1509 ± 0.073 | −0.099 | | 9 | 0.1585 ± 0.045 | 0.577 |

- case별 std가 prj04보다 전반적으로 크다(예 Case 8 ±0.084, Case 13 ±0.093) — MLP meta의 불안정성이 case 단위에서도 확인됨.

---

## 5. 재현 방법

```bash
cd /home/hwkang/manufacturing_data_exercise/02_milling_v2
venv312/bin/python sandbox/prj05/run_experiment.py
```
최초 실행 시 `cache/feat_rawdeltameta_100.pkl` 빌드/재사용. nested-LOCV 75 fold joblib 병렬.

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `context.md` | (본 문서) 단일 공유 문서 |
| `run_experiment.py` | base 3종 + MLP meta nested-LOCV stacking |
| `cache/per_case.csv` | (seed, case)별 stack/avg/각 base RMSE (75행) |
| `cache/summary.csv` | case별 stack RMSE mean±std |
| `cache/feat_rawdeltameta_100.pkl` | 100% Raw+Delta+Meta feature matrix |
| `cache/run.log` | 실행 로그 |

---

## 7. 결론

요청대로 base를 SVR·RandomForest·XGBoost로, meta-learner를 MLP로 구성해 prj04와 동일한 nested-LOCV(@100%, 5 seed)로 검증했다. 결과 **0.1511 ± 0.0254**는 prj04(Ridge meta, 0.1207)와 단순 평균(0.1112)보다 모두 나쁘고 seed 분산도 크다. OOF meta 학습 데이터가 ~150행·3입력으로 작아 비선형 MLP meta-learner가 과적합·불안정해지기 때문이다. 이 문제에서는 **강한 base 소수의 단순 평균 또는 선형 meta-learner가 MLP meta보다 우월**하다는 점이 핵심 교훈이다.
