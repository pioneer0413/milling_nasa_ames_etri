# prj06 stacking — base = SVR/RF/XGBoost, meta-learner = Ridge — LOCV @100%, 5 seeds

> 작업 요청자와 에이전트(Claude) 간 단일 공유 문서.
> sandbox/prj06/ 아래 스크립트·결과를 이 문서 하나로 추적한다.

---

## 0. 작업 요청 (verbatim)

> 그럼 prj06에서는 meta-learner를 ridge로 두고 base-learner는 동일하게 해서 다시 실험하고 결과를 보고해.

(base = prj05와 동일: SVR·RandomForest·XGBoost. meta-learner만 MLP→**Ridge**. 검증은 prj04/prj05와 동일.)

---

## 1. 구성

| 항목 | 값 |
| --- | --- |
| **Base learners** | SVR(rbf,C=100,γ=scale,ε=0.05), RandomForest(300), XGBoost(200,d3,lr0.1) |
| **Meta-learner** | **Ridge(alpha=1.0)** — 3개 base 예측 → 최종 VB |
| Feature | Raw+Delta+Meta = 51 |
| 입력 길이 | 100% only |
| 검증 | 누수 없는 nested LOCV (inner LOCV로 OOF 생성 → Ridge meta 학습 → outer 예측) |
| 반복 | seed 5회 `[0,1,2,3,4]`, mean±std |

prj05와 유일한 차이는 meta-learner(MLP→Ridge).

---

## 2. 결과 요약 (TL;DR)

> **Stacking(Ridge meta) 15-fold LOCV RMSE @100% = 0.1164 ± 0.0003** (5 seed mean±std)

### 전체 비교 — 15-fold 평균 RMSE (mean ± std over 5 seeds)
| 구성 | RMSE | 비고 |
| --- | ---: | --- |
| **simple average** | **0.1112 ± 0.0003** | base 3종 단순 평균 (최고) |
| SVR (단일) | 0.1155 ± 0.0000 | 최고 단일 base |
| **Stacking (Ridge meta)** | **0.1164 ± 0.0003** | ← 요청 모델 |
| XGBoost (단일) | 0.1204 ± 0.0000 | |
| RandomForest (단일) | 0.1242 ± 0.0011 | |

### Stacking seed별 15-fold 평균 RMSE
| seed | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.1159 | 0.1163 | 0.1168 | 0.1167 | 0.1162 |

### 핵심 관찰
- **meta-learner를 MLP→Ridge로 바꾸자 정확도·안정성이 모두 급개선됐다.** prj05 MLP meta 0.1511 ± 0.0254 → prj06 Ridge meta **0.1164 ± 0.0003** (RMSE −23%, seed std는 ~85배 작아짐).
- 다만 Ridge stack(0.1164)도 **단순 평균(0.1112)을 넘지는 못했고** 최고 단일 SVR(0.1155)과 사실상 동률. prj01~05에서 반복 확인된 패턴 — 이 소표본 LOCV에서는 강한 base의 단순 결합이 매우 강력.
- **seed 분산이 극히 작다(±0.0003)** → Ridge meta stacking은 deterministic에 가깝고 재현성·안정성이 우수.

---

## 3. Meta-learner(Ridge) 계수 (seed·fold 평균)

| base | SVR | RandomForest | XGBoost | intercept |
| --- | ---: | ---: | ---: | ---: |
| coef | **+0.485** | +0.222 | +0.313 | −0.015 |

- **SVR에 최대 가중(0.485)**, XGBoost(0.313), RandomForest(0.222) 순. 계수 합 ≈ 1.0.
- 성능 순서(SVR>XGB>RF)와 가중 순서가 일치 → meta-learner가 합리적으로 학습. prj04(5 base)에서도 SVR이 최대였던 것과 일관.

---

## 4. stacking 실험 종합 비교 (prj04 / prj05 / prj06)

| 실험 | base | meta | RMSE @100% | seed std |
| --- | --- | --- | ---: | ---: |
| prj04 | Ridge·SVR·RF·XGB·MLP (5) | Ridge | 0.1207 | ±0.0030 |
| prj05 | SVR·RF·XGB (3) | MLP | 0.1511 | ±0.0254 |
| **prj06** | **SVR·RF·XGB (3)** | **Ridge** | **0.1164** | **±0.0003** |
| (참고) | SVR·RF·XGB (3) | simple average | **0.1112** | ±0.0003 |

**결론적 흐름:**
1. base에서 약한 모델(Ridge·MLP) 제거 → 개선 (prj04 0.1207 → prj06 0.1164).
2. meta로 MLP는 치명적(0.1511), Ridge는 우수(0.1164) → **소표본 stacking에는 선형 meta가 정답.**
3. 그래도 가장 좋은 건 여전히 **3 strong base의 단순 평균(0.1112)** — 정교한 meta-learner가 단순 평균을 못 이긴다.

---

## 5. Case별 Stacking(Ridge) RMSE (seed 5회 mean ± std)

| Case | RMSE | R² | | Case | RMSE | R² |
| ---: | ---: | ---: |---| ---: | ---: | ---: |
| 2 | 0.0417 ± 0.000 | 0.909 | | 16 | 0.1196 ± 0.002 | 0.638 |
| 7 | 0.0585 ± 0.000 | 0.835 | | 10 | 0.1230 ± 0.001 | 0.691 |
| 5 | 0.0595 ± 0.002 | 0.940 | | 1 | 0.1283 ± 0.001 | 0.332 |
| 3 | 0.0698 ± 0.001 | 0.728 | | 12 | 0.1402 ± 0.001 | 0.435 |
| 15 | 0.0892 ± 0.001 | 0.841 | | 8 | 0.1482 ± 0.001 | 0.425 |
| 11 | 0.0943 ± 0.000 | 0.811 | | 14 | 0.1713 ± 0.001 | 0.761 |
| 9 | 0.1008 ± 0.002 | 0.839 | | **13** | **0.2945 ± 0.002** | 0.591 |
| 4 | 0.1066 ± 0.001 | 0.402 | | | | |

- prj05(MLP) 대비 case별 std가 한 자릿수 수준으로 작아짐 — Ridge meta의 안정성 확인.
- hard case는 여전히 Case 13(0.295)·14(0.171)·8·12가 평균을 끌어올림.

---

## 6. 재현 방법

```bash
cd /home/hwkang/manufacturing_data_exercise/02_milling_v2
venv312/bin/python sandbox/prj06/run_experiment.py
```
최초 실행 시 `cache/feat_rawdeltameta_100.pkl` 빌드/재사용. nested-LOCV 75 fold joblib 병렬.

## 7. 산출물

| 파일 | 내용 |
| --- | --- |
| `context.md` | (본 문서) 단일 공유 문서 |
| `run_experiment.py` | base 3종 + Ridge meta nested-LOCV stacking |
| `cache/per_case.csv` | (seed, case)별 stack/avg/각 base RMSE (75행) |
| `cache/summary.csv` | case별 stack RMSE mean±std |
| `cache/meta_coef.csv` | (seed, fold)별 Ridge meta 계수 |
| `cache/feat_rawdeltameta_100.pkl` | 100% Raw+Delta+Meta feature matrix |
| `cache/run.log` | 실행 로그 |

---

## 8. 결론

base를 SVR·RandomForest·XGBoost로 고정하고 meta-learner만 Ridge로 두어 동일 검증(@100%, 5 seed nested LOCV)을 수행했다. **결과 0.1164 ± 0.0003**으로, prj05의 MLP meta(0.1511 ± 0.0254)를 정확도·안정성 모두에서 크게 앞선다. Ridge meta 계수는 SVR↑(0.485)로 합리적이며 seed 분산이 거의 0이다. 다만 stacking 자체는 3 strong base의 **단순 평균(0.1112)**을 넘지 못했다 — 소표본 LOCV에서는 선형 meta조차 단순 평균 대비 추가 이득이 없다는 점이 일관된 결론이다.
