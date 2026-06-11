# prj03 weighted-feature GRU — LOCV @100%, 5 seeds

> 작업 요청자와 에이전트(Claude) 간 단일 공유 문서.
> sandbox/prj03_weighted_feature/ 아래 스크립트·결과를 이 문서 하나로 추적한다.

---

## 0. 작업 요청 (verbatim)

> prj03_weighted_feature에서는 Feature-based GRU에 Raw+Delta+Meta Feature vector에 학습 가능한 가중치를 부여해서 인코딩한 다음 GRU에 넣어 sequence를 모델링하고 최종적으로 VB를 예측하는 모델을 만들어서 100% 길이 입력에 대해 LOCV 성능 평가 실험을 수행해라. LOCV를 서로 다른 seed로 5번 반복해라. 완료 후 결과를 보고해라.

---

## 1. 모델 구조

요청대로 **학습 가능한 per-feature 가중치 → 인코딩 → GRU → VB 예측** 흐름.

```
x  (B, T, 51)                         T = case의 run 수, 51 = Raw24+Delta24+Meta3
  → x * w          w: 학습가능 per-feature weight (nn.Parameter, init=1)  [feature gating]
  → Linear(51→64) + ReLU                                                  [encoder]
  → GRU(64 → hidden=256, 3-layer, dropout=0.1)                           [sequence model]
  → Linear(256→16) + ReLU → Linear(16→1)                                 [head] → run별 VB
```

- 입력 feature vector: **Raw+Delta+Meta = 51차원** (6센서 × {mean,rms,std,peak}의 Raw 24 + Delta 24 + Meta {DOC,feed,material} 3)
- `feat_weight`는 51차원 학습 파라미터로, 입력 feature에 element-wise 곱해진 뒤 인코더로 전달 → 학습 종료 후 |w|를 feature 중요도로 해석·저장.

### 학습/평가 설정

| 항목 | 값 |
| --- | --- |
| 입력 길이 | **100% only** |
| 검증 | Leave-One-Case-Out (전체 15 case: target 제외 14개 학습 → target run-sequence 예측) |
| 반복 | **서로 다른 seed 5회** `[0,1,2,3,4]` (모델·numpy·cuda seed 모두 설정) |
| 옵티마이저 | Adam, lr=1e-3, weight_decay=1e-4, CosineAnnealingLR, grad-clip 1.0 |
| epochs | 200 |
| feature 스케일 | per-fold StandardScaler (train case로 fit) |
| target 스케일 | train batch z-score (예측 시 역변환) |
| 전처리 | VB 보간 + first-run 0, EXCLUDED_RUNS {(2,1),(12,1)} 제외 |
| 지표 | (seed, case)별 RMSE → seed별 15-fold 평균 → seed 간 mean±std |

---

## 2. 결과 요약 (TL;DR)

> **전체 15-fold LOCV RMSE @100% = 0.1306 ± 0.0087** (5 seed mean±std)

### Seed별 15-fold 평균 RMSE
| seed | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.1276 | 0.1206 | 0.1391 | 0.1253 | 0.1403 |

seed 간 변동(±0.0087)은 작아 결과는 안정적.

### Case별 RMSE (seed 5회 mean ± std)
| Case | RMSE | R² | | Case | RMSE | R² |
| ---: | ---: | ---: |---| ---: | ---: | ---: |
| 7 | 0.0373 ± 0.004 | 0.932 | | 10 | 0.1061 ± 0.034 | 0.752 |
| 2 | 0.0679 ± 0.011 | 0.754 | | 13 | 0.1115 ± 0.011 | 0.941 |
| 3 | 0.0832 ± 0.026 | 0.583 | | 14 | 0.1381 ± 0.021 | 0.841 |
| 11 | 0.0913 ± 0.019 | 0.816 | | 8 | 0.1492 ± 0.020 | 0.409 |
| 16 | 0.0930 ± 0.025 | 0.769 | | 15 | 0.1548 ± 0.053 | 0.476 |
| 9 | 0.0930 ± 0.013 | 0.861 | | 5 | 0.1824 ± 0.019 | 0.431 |
| 4 | 0.1061 ± 0.010 | 0.405 | | 12 | 0.1995 ± 0.057 | −0.217 |
| | | | | **1** | **0.3450 ± 0.106** | **−4.192** |

- 잘 풀리는 case: 7(0.037), 2(0.068), 3(0.083).
- 평균을 끌어올리는 hard case: **Case 1**(0.345, R²<0)이 압도적, 이어 12, 5, 15. (Case 1은 100% 입력·LOCV에서 외삽이 크게 빗나감.)
- Case 13/14는 R²는 높지만(0.94/0.84) RMSE는 0.11/0.14 — VB 범위가 커서 절대 오차가 큼.

---

## 3. 학습된 feature weight 분석

| 통계 | 값 |
| --- | --- |
| min / max | 0.955 / 1.058 |
| mean ± std | 1.002 ± 0.012 |

**상위(가중↑):** AE_table__delta_std, smcAC__delta_mean, AE_table__delta_peak, smcAC__delta_std, smcDC__delta_mean, material, AE_spindle__rms …
**하위(가중↓):** AE_spindle__delta_mean, AE_spindle__peak, feed, vib_table__rms, vib_table__delta_mean, vib_table__delta_std …

**관찰 (중요):** 학습된 가중치는 1.0에서 거의 벗어나지 않았다(±0.012). 즉 명시적 per-feature 게이트의 실효는 작다. 원인은 구조적이다 — 바로 뒤의 `Linear(51→64)` 인코더가 이미 feature별 선형 스케일을 학습할 수 있어 별도 곱셈 가중치가 **중복(redundant)**이고, weight_decay가 1.0 주변으로 끌어당기기 때문. 그럼에도 미세하게 **전류(smcAC/smcDC) delta·AE delta 계열이 상대적으로 가중↑, 진동(vib_table) 계열이 가중↓**으로 나타나, prj01/prj02의 "전류·AE 신뢰 / 진동 비신뢰" 경향과 방향이 일치한다.

---

## 4. 해석 및 참고

- 이 실험은 요청한 구조(학습가능 가중치 인코딩 + GRU)를 100% 입력·전체 case·5 seed로 평가한 것이며, 결과는 0.1306 ± 0.0087로 안정적이다.
- 100%-only·전체 case 평균이라 best-prefix를 case별로 고르는 H4 leader board류 수치(예 0.0945@80%)와 직접 비교는 부적절하다. 100% 입력은 case별 최적 prefix가 아닌 경우가 많아 평균이 더 높게 나온다.
- per-feature 학습 가중치의 효과가 작았던 점은 구조적 중복 때문이며(§3), 게이트 효과를 키우려면 ① 인코더 앞 가중치에 sparsity/L1 규제, ② softmax/sigmoid attention 형태, ③ 인코더 제거 후 가중치만으로 직접 GRU 투입 등의 변형이 필요하다. (현재 범위 밖, 필요 시 확장 가능.)

---

## 5. 재현 방법

```bash
cd /home/hwkang/manufacturing_data_exercise/02_milling_v2
venv312/bin/python sandbox/prj03_weighted_feature/run_experiment.py
```
최초 실행 시 `cache/feat_rawdeltameta_100.pkl`(100% Raw+Delta+Meta matrix, 163×54)를 빌드·캐싱 후 재사용. GPU(cuda) 사용.

## 6. 산출물

| 파일 | 내용 |
| --- | --- |
| `context.md` | (본 문서) 단일 공유 문서 |
| `run_experiment.py` | 모델 정의 + feature 빌드 + 5-seed LOCV |
| `cache/feat_rawdeltameta_100.pkl` | 100% Raw+Delta+Meta feature matrix |
| `cache/per_case.csv` | (seed, case)별 RMSE/MAE/R² (75행) |
| `cache/summary.csv` | case별 mean±std RMSE |
| `cache/feature_weights.csv` | (seed, fold)별 학습된 51개 feature weight |
| `cache/run.log` | 실행 로그 |
