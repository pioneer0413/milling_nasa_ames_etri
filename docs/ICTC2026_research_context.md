# ICTC 2026 — Research Context

ICTC 2026 논문 작업의 Notion 마스터 페이지와 하위 페이지 맥락을 repo에 고정한 문서.
어느 디바이스에서 clone해도 연구 배경/가설/도메인 지식을 Notion 접근 없이 파악할 수 있게 한다.
(스냅샷 기준일: 2026-06-16. Notion이 최신 원본이며, 본 문서는 요약·맥락 보존용.)

## 0. 페이지 맵

| 페이지 | 역할 | URL |
|---|---|---|
| 🧑‍🎓 ICTC 2026 (master) | 최상위 허브, 마감 2026-06-26 | https://app.notion.com/p/3603fe1c1eaf80959ecee525cda74fd3 |
| 📝 Outline | 논문 골격 (Intro→Objective) | https://app.notion.com/p/3523fe1c1eaf802995d1c08ebc6c9a49 |
| 🔬 Experiment | 실험 공통 설정 + 실험 DB | https://app.notion.com/p/3523fe1c1eaf80cea68ef8c6e1810b64 |
| 💡 Idea | 기술 메모·인사이트·세그멘테이션 알고리즘 | https://app.notion.com/p/3523fe1c1eaf803f8d17c1a95f466335 |
| 🧠 Knowledge Base | 용어집 + 데이터셋 도메인 지식 | https://app.notion.com/p/3523fe1c1eaf80c2ad1df61e5d6a0dbe |

상세 DB/관계형 구조는 `~/.claude/.../memory/reference_notion_ictc2026.md` 참고.

---

## 1. 핵심 연구 가설 (Outline → Problem Definition)

> **run 간 순차적 의존성을 명시적으로 모델링하는 run-sequence 모델(GRU/LSTM/RNN)** 은
> **run을 서로 독립적으로 처리하는 run-independent 모델(RF/XGBoost 등)** 대비
> 더 낮은 VB 예측 오차(LOCV RMSE)를 달성한다.

- **근거**: 공구 마모는 누적 열화(cumulative degradation) — 이전 run까지의 마모 이력이 현재 VB 추정에 유의미한 정보를 제공. run-independent 모델은 run 간 순서(마모 trajectory) 정보를 잃는다.
- **기각 조건**: run-sequence RMSE ≥ run-independent RMSE (LOCV 평균). 기각 시 해석 = run-level feature만으로 충분, 시퀀스 모델링 실익 없음.
- **검증 설계**: Feature-GRU(run 시퀀스) vs XGBoost/RF(run 독립), LOCV, observed-VB-only RMSE.

### Introduction 논리 (KNOWN / UNKNOWN / WHY-HOW)
- KNOWN: 마모는 비가역·단조 증가, run은 누적 과정, 연속 run은 비독립.
- UNKNOWN: 기존 연구 대부분 run을 독립 처리 → run-to-run 의존성·이력 미반영. 시퀀스 시간 맥락의 기여도·효과적 구조 미규명.
- HOW: 공구 사용을 run 시퀀스로 정의, feature engineering으로 run-level feature vector 추출 후 GRU regressor로 run 간 시간 의존성 학습.

> ⚠️ **연구 방향 전환 이력**: 초기에는 "segment-aware tool wear prediction"(절삭 구간 entry/steady/exit 분석) 접근이었으나, 현재는 "run-sequence modeling"으로 전환. 구 접근(H1~H9 가설, M1/M2 방법론)은 Outline의 **Archive** 토글로 이동.

---

## 2. 데이터셋 (Outline → Dataset and Domain, Knowledge Base)

### D1: NASA Ames Milling
- 167 run, 센서 시계열 길이 9000. **유효 run = 144개** (VB 없음 23개 + 비정상/길이≠9000 2개 제외).
- 라벨 VB = flank wear \[mm\]. 16 case 중 case 6(run 1개)·이상 run 제외 → 유효 **15 case**.
- 통제 변수: DOC{0.75, 1.5} × Feed{0.25, 0.5} × Material{cast iron, stainless steel}.
- 센서 6종: smcAC/smcDC(전류), vib_table/vib_spindle(진동), AE_table/AE_spindle(음향). **전류 외 신호는 raw가 아니라 HP/LP/RMS 가공됨.**
- `EXCLUDED_RUNS = {(2,1), (12,1)}`, `NON_OBSERVED_RUNS` = VB 보간/대치 19 run(RMSE 집계 제외).

### D2: PHM2010
- 2010 PHM Data Challenge. 6mm 볼노즈 카바이드 커터 RUL 추정 과제. 6 레코드(c1~c6) 중 wear 라벨 공개분 **c1/c4/c6** = case 1/4/6, 각 **315 run**.
- 채널 7개: force_x/y/z(N), vib_x/y/z(g), ae_rms(V). 50 kHz 샘플링. ae_rms는 하드웨어 RMS 처리값.
- 라벨: flute 3개 개별 측정 VB_flute_1/2/3 \[µm\]. EOL 기준 200µm.
- **라벨 정의(중요)**: `VB_max = max(VB_flute_1, VB_flute_2, VB_flute_3)` — 공식 Task가 "어느 한 flute의 최대 마모량이 wear limit 초과 금지"이므로 flute 평균이 아닌 max가 평가 대상.
- 대회 scoring은 비대칭(과대추정 패널티 ≈ 과소추정의 5배)이나, 본 연구는 RUL이 아닌 **VB_max 직접 회귀**로 접근.

### Valid Prediction Error (인접 run 구분 기준)
모델 RMSE가 이 값보다 크면 연속 두 run의 마모 상태를 구분 불가.
| 데이터셋 | mean \|ΔVB\| | median |
|---|---|---|
| NASA Ames | 0.082 mm | 0.070 mm |
| PHM2010 (VB_max) | 0.593 µm | 0.433 µm |

---

## 3. 실험 공통 설정 (Experiment → Common Setup)

- **Model Pool**: ML(Ridge/RF/XGBoost/SVR) + DL(Feature MLP, Feature RNN/GRU/LSTM, Signal CNN/GRU/LSTM).
- **Seed Policy**: 5 random seed.
- **Evaluation Protocol**: LOCV(Leave-One-Case-Out) + Observed-VB-only(NON_OBSERVED_RUNS 제외) + mean±std 보고.
- **Metric**: RMSE⭐, MAE, R².
- NASA: 15 cases / LOCV 15 folds, 종속변수 VB. PHM2010: 3 cases / LOCV 3 folds, 종속변수 VB_max.

### 실험 DB (Run-sequence Modeling 워크스페이스)
- **Run-sequence Modeling** DB: 활성 실험 (B/I/E class). 예) [B1] PHM2010 Feature baseline, [E5] 하이퍼파라미터 튜닝.
- **Analysis** DB: 분석 노트, Run-sequence Modeling과 관계형 연결. 예) "PHM2010에서 run-sequence가 run-independent에 뒤지는 이유".

### Performance Objective (현재 best)
| 데이터셋 | 목표 (LOCV RMSE) | 달성 | Best |
|---|---|---|---|
| NASA Ames | ≤ 0.080 mm | No | FeatLSTM 0.092217 |
| PHM2010 | ≤ 10 µm | No | FeatLSTM 22.04 (raw feature 기준) |

> 단, E5_S1(Delta feature)에서 PHM2010 sequence 모델이 RF를 역전 (FeatLSTM delta 20.23 / FeatGRU raw+delta 20.39µm) — 본 repo 실험 라인 참조.

---

## 4. 핵심 인사이트 (Idea)

- **선형 보간 VB**는 모델 공통적으로 성능 향상에 도움 ⭐.
- `run=1`의 VB=0은 유효하지 않음 → 전처리에서 제거 후 학습.
- **목표 RMSE는 0.1 mm 이내**여야 실질적으로 유효한 예측 ⭐ (RMSE 스케일 해석 표: ≤0.03 매우 우수 … ≥0.10 주장 조심).
- **Critical Mistake 1** ⭐: H1_S1/S2/S3 집계 시 **sensor에 대해 평균**을 내버려 센서별 feature/segment 특성 해석이 왜곡됨(`feature_segment_rank_average`류 표가 smcAC/smcDC의 강점을 희석). sensor-level 원본 산출물에는 정보 보존.
- **어려운 Case**: Case 13/14(신호 패턴 상이 → 난이도↑), Case 1/12 근본 원인은 H7_S1에서 규명:
  - **Case 1**: 유일한 non-monotone VB 궤적(Run15 peak→Run17 감소) → GRU 단조증가만 학습 → 파국적 오차. GRU/XGB 3.1배. (피처 문제 아님, OOD 아님)
  - **Case 12**: Delta baseline 분포 편이(`(12,1)` 제외로 기준=Run2 VB=0.05) → 두 모델 동등 실패 = 피처 파이프라인 결함.
- **실시간성(향후)**: 전체 cut 신호 입력 → cut 완료 후 예측. 실제 공정은 다음 cut 전 교체 결정 필요 → cut 간격 내 추론 성능도 만족해야 함.
- Heuristic Segmentation Algorithm **V1**(window L=256, no-load/steady end·start 검출) / **V2**(target smcDC, w=300/stride=100, exit-cut 식별) 기록 존재.
- 관련 연구 비교표(Ghosh 2007, García-Nieto 2016, Cai 2020, Karabacak 2024, Zhang 2021, Jeon&Rhee 2024, Li 2025 TCN-BiGRU-SA, Peng 2025 PINN 등) 보유.

---

## 5. 도메인 용어 (Knowledge Base 발췌)

- **Flank wear (VB)**: 공구 옆면(flank face) 마모. 절삭날에서 마모 영역 끝까지의 길이.
- **Entry / Regular(Steady) / Exit cut**: 공구가 소재에 들어가는 순간 / 안정 절삭 / 빠져나가는 순간.
- **마모 종류**: rounding(날 둥글어짐), crater wear(rake face 패임), flank wear(flank face 마찰 마모).
- **센서 물리**: AE=고주파(shear zone 변형/파괴), Vibration=저주파(절삭력 변화), Cutting force=전류로 간접 측정(스핀들 모터 전류 ∝ 토크 ∝ 절삭력).
- **공정 조건 → domain shift**: DOC·Feed 증가 시 절삭력/열/신호 진폭 증가, Material(cast iron 취성 vs stainless steel 연성·난삭)이 가장 강한 domain shift 요인.

---

## 6. 본 repo 실험과의 연결

Notion 실험 DB의 PHM2010 라인은 본 repo에서 재현 가능:
- `scripts/run_B1_S1_phm2010_feature_baseline.py` — [B1] 9-model LOCV-3 baseline.
- `scripts/run_B1_S2_phm2010_flute_augment.py` — flute-label 증강(궤적 2→6).
- `scripts/run_E5_S1_phm2010_delta_feature.py` — [E5] Delta feature 비대칭 해소(sequence가 RF 역전).

NASA 라인(H 시리즈)은 `datasets/nasa/raw_signal.csv`(gitignored)가 필요. 자세한 데이터 가용성은 `README.md` 참조.
