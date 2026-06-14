# prj04 stacking-ensemble + meta-learner — LOCV @100%, 5 seeds

> 작업 요청자와 에이전트(Claude) 간 단일 공유 문서.
> sandbox/prj04_stacking_ensemble_meta-learning/ 아래 스크립트·결과를 이 문서 하나로 추적한다.

---

## 0. 작업 요청 (verbatim)

> prj04_stacking_ensemble_meta-learning에서는 @leaderboards/h4/per_case/top5.csv 에 있는 ML 모델만을 사용해서 stacking-ensemble을 구성해. 그리고 마지막에 meta-learner를 사용해서 최종 VB를 예측하는 모델을 만들어. 그것도 입력 100%, 5-seed LOCV로 실험 후 결과를 보고해.

---

## 1. 구성

### Base learners — leaderboard의 ML 모델만
`leaderboards/h4/per_case/top5.csv`의 `model_type==ML`에 등장하는 5개 모델만 사용 (DL인 `feature_gru`는 제외):

| Base learner | 설정 |
| --- | --- |
| Ridge | alpha=1.0 |
| SVR | rbf, C=100, gamma='scale', epsilon=0.05 |
| RandomForest | n_estimators=300, random_state=seed |
| XGBoost | n_estimators=200, max_depth=3, lr=0.1, subsample=1.0, random_state=seed |
| MLP | hidden=(64,32), max_iter=1500, random_state=seed |

- **Meta-learner:** `Ridge(alpha=1.0)` — 5개 base 예측을 입력받아 최종 VB 예측.
- Feature: Raw+Delta+Meta(51), 입력 길이 **100% only**. Ridge/SVR/MLP는 train-fit StandardScaler 적용, 트리는 원본.

### 누수 없는 nested LOCV stacking
```
Outer fold (held-out case T 예측):
  1) Inner LOCV (train cases c≠T): 각 c에 대해 (c 제외 train)으로 5 base 학습→c 예측
     ⇒ 모든 train run의 out-of-fold(OOF) base 예측 생성
  2) meta-learner를 [OOF base 예측 → VB]로 학습
  3) 전체 train case(≠T)로 5 base 재학습 → T의 base 예측 생성
  4) meta-learner가 T의 base 예측 → 최종 VB
```
전 과정을 seed 5회(`[0,1,2,3,4]`) 반복, mean±std 보고. (folds 75개 = 15 case × 5 seed, joblib 병렬.)

---

## 2. 결과 요약 (TL;DR)

> **Stacking(meta-learner) 15-fold LOCV RMSE @100% = 0.1207 ± 0.0030** (5 seed mean±std)

### 전체 비교 — 15-fold 평균 RMSE (mean ± std over 5 seeds)
| 구성 | RMSE | 비고 |
| --- | ---: | --- |
| **SVR (단일)** | **0.1155 ± 0.0000** | 최고 단일 base |
| simple average | 0.1195 ± 0.0090 | 5 base 단순 평균 |
| XGBoost (단일) | 0.1204 ± 0.0000 | |
| **Stacking (meta-learner)** | **0.1207 ± 0.0030** | ← 요청 모델 |
| RandomForest (단일) | 0.1242 ± 0.0011 | |
| Ridge (단일) | 0.1471 ± 0.0000 | |
| MLP (단일) | 0.2407 ± 0.0323 | 가장 약함 |

### Stacking seed별 15-fold 평균 RMSE
| seed | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.1177 | 0.1245 | 0.1175 | 0.1214 | 0.1224 |

### 핵심 관찰 (정직 보고)
- **Stacking은 최고 단일 base(SVR 0.1155)를 능가하지 못했다.** simple average·XGBoost와 사실상 동률(0.120 부근).
- 이유: base 다양성이 충분치 않고 SVR이 단독으로 우세한 데다, MLP가 약해(0.24) 앙상블에 잡음을 더한다. meta-learner는 이를 일부 보정하지만 단일 SVR을 넘지는 못함.
- **장점은 안정성**이다. Stacking의 seed 간 std(±0.0030)는 simple average(±0.0090)보다 작아, 어떤 단일 모델을 고를지 모르는 상황에서 robust한 선택지다.

---

## 3. Meta-learner 계수 (seed·fold 평균)

| base | Ridge | SVR | RandomForest | XGBoost | MLP | intercept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coef | +0.169 | **+0.400** | +0.168 | +0.239 | +0.040 | −0.014 |

- meta-learner가 **SVR에 가장 큰 가중(0.400)**, 약한 **MLP는 거의 0(0.040)**으로 합리적으로 가중. 계수 합 ≈ 1.0.
- 즉 stacking은 "성능 좋은 base를 더 신뢰"하는 방향으로 올바르게 학습했으나, 그 결과가 단일 SVR을 넘어서진 못했다.

---

## 4. Case별 Stacking RMSE (seed 5회 mean ± std)

| Case | RMSE | R² | | Case | RMSE | R² |
| ---: | ---: | ---: |---| ---: | ---: | ---: |
| 5 | 0.0554 ± 0.011 | 0.946 | | 10 | 0.1196 ± 0.008 | 0.707 |
| 2 | 0.0598 ± 0.024 | 0.788 | | 12 | 0.1391 ± 0.014 | 0.440 |
| 3 | 0.0631 ± 0.003 | 0.777 | | 4 | 0.1670 ± 0.008 | −0.468 |
| 7 | 0.0758 ± 0.006 | 0.722 | | 8 | 0.1685 ± 0.010 | 0.255 |
| 11 | 0.0913 ± 0.004 | 0.822 | | 14 | 0.1756 ± 0.002 | 0.748 |
| 15 | 0.0929 ± 0.011 | 0.826 | | **13** | **0.2898 ± 0.008** | 0.604 |
| 16 | 0.0986 ± 0.021 | 0.745 | | | | |
| 9 | 0.1000 ± 0.005 | 0.841 | | | | |

- 우수: Case 5(0.055), 2(0.060), 3(0.063).
- hard case: Case 13(0.290), 14(0.176), 8(0.169), 4(0.167). prj03(weighted GRU)와 마찬가지로 100% 입력·전체 case 조건에서 hard case가 평균을 끌어올림. (Case 13은 prj01에서 보았듯 meta·전류 튜닝 없이는 어려움.)

---

## 5. 재현 방법

```bash
cd /home/hwkang/manufacturing_data_exercise/02_milling_v2
venv312/bin/python "sandbox/prj04_stacking_ensemble_meta-learning/run_experiment.py"
```
최초 실행 시 `cache/feat_rawdeltameta_100.pkl`을 빌드(또는 prj03 캐시 재사용). nested-LOCV 75 fold는 joblib 병렬로 수 분 내 완료.

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `context.md` | (본 문서) 단일 공유 문서 |
| `run_experiment.py` | base 5종 + Ridge meta-learner nested-LOCV stacking |
| `cache/per_case.csv` | (seed, case)별 stack/avg/각 base RMSE (75행) |
| `cache/summary.csv` | case별 stack RMSE mean±std |
| `cache/meta_coef.csv` | (seed, fold)별 meta-learner 계수 |
| `cache/feat_rawdeltameta_100.pkl` | 100% Raw+Delta+Meta feature matrix |
| `cache/run.log` | 실행 로그 |

---

## 7. 결론

요청대로 leaderboard의 ML 5종(Ridge·SVR·RandomForest·XGBoost·MLP)만으로 누수 없는 nested-LOCV stacking을 구성하고 Ridge meta-learner로 최종 VB를 예측했다. **결과 0.1207 ± 0.0030 @100%**는 안정적이나 최고 단일 모델 SVR(0.1155)을 능가하지 못했다 — 이는 base 다양성 부족·SVR 단독 우세·약한 MLP 잡음에 기인하는 전형적 stacking 한계다. meta-learner 계수는 SVR↑·MLP↓로 합리적이었다. stacking의 실익은 절대 성능보다 **모델 선택 위험을 줄이는 안정성**(낮은 seed 분산)에 있다.
